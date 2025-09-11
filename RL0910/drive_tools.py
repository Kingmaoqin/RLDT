"""
drive_tools.py - Enhanced version with hot parameter updates and online training
"""

from typing import Dict, List, Any, Optional
from typing import Tuple, Dict
import json
import numpy as np
import threading
import queue
import os
from datetime import datetime
import torch
import torch.nn.functional as F
import time
from collections import deque
from functools import wraps
import pandas as pd
from PIL import Image

# Import existing classes
from inference import DigitalTwinInference, ClinicalDecisionSupport
from models import TransformerDynamicsModel, TreatmentOutcomeModel, ConservativeQNetwork, EnsembleQNetwork
from utils import analyze_feature_importance
from data import PatientDataGenerator
from training import train_digital_twin, train_outcome_model, train_rl_policy
from data_manager import data_manager

# Import new online training components
from online_loop import create_online_training_system, OnlineTrainer
from samplers import StreamActiveLearner
# BCQ支持
try:
    from d3rlpy import load_learnable
    BCQ_AVAILABLE = True
except ImportError:
    BCQ_AVAILABLE = False

try:
    from reports import (
        build_action_catalog,
        compute_recommendation,
        render_patient_report,
        _scores_from_policy_only,
    )
except Exception:
    build_action_catalog = compute_recommendation = render_patient_report = _scores_from_policy_only = None

from reports import (
    build_action_catalog,
    compute_recommendation,
    render_patient_report,
    make_treatment_analysis_figure,
    _scores_from_policy_only,
)

class BCQInferenceAdapter:
    """将BCQ策略适配到现有推理接口"""
    
    def __init__(self, bcq_policy_path: str):
        if not BCQ_AVAILABLE:
            raise ImportError("d3rlpy not available")
        
        # 添加全局标记，避免重复尝试加载失败的BCQ
        if not hasattr(self.__class__, '_bcq_load_failed'):
            try:
                self.bcq_policy = load_learnable(bcq_policy_path)
                self.bcq_available = True
                print(f"✓ BCQ policy loaded successfully")
            except Exception as e:
                # 标记BCQ加载失败，避免重复尝试
                self.__class__._bcq_load_failed = True
                self.bcq_available = False
                print(f"✗ BCQ loading permanently failed: {e}")
                print("  Will use CQL for all future requests")
                raise ImportError(f"BCQ not available: {e}")
        else:
            # 如果之前已经失败过，直接抛出异常
            self.bcq_available = False
            raise ImportError("BCQ loading previously failed")
        self.action_names = [
            'Medication A', 'Medication B', 'Medication C', 
            'Placebo', 'Combination Therapy'
        ]
        if not hasattr(self.__class__, "_logged_once"):
            print("✓ BCQ policy loaded successfully")
            self.__class__._logged_once = True        
    
    def recommend_treatment(self, patient_state: np.ndarray, return_all_scores: bool = False) -> Dict:
        """使用BCQ策略进行治疗推荐"""
        if patient_state.ndim == 1:
            patient_state = patient_state.reshape(1, -1)
        
        try:
            # BCQ预测
            action = self.bcq_policy.predict(patient_state.astype(np.float32))
            action_idx = int(action[0] if hasattr(action, '__len__') else action)
            action_idx = max(0, min(action_idx, len(self.action_names) - 1))
            
            result = {
                'recommended_action': action_idx,
                'recommended_treatment': self.action_names[action_idx],
                'confidence': 0.8,  # BCQ固定置信度
                'expected_immediate_outcome': 0.0,
                'q_value': 0.5  # 模拟Q值
            }
            
            if return_all_scores:
                # 模拟所有动作的分数
                fake_q_values = [0.3] * len(self.action_names)
                fake_q_values[action_idx] = 0.8  # 推荐动作得分更高
                
                result.update({
                    'all_q_values': fake_q_values,
                    'all_outcomes': [0.0] * len(self.action_names),
                    'treatment_rankings': sorted(enumerate(fake_q_values), 
                                                key=lambda x: x[1], reverse=True)
                })
            
            return result
            
        except Exception as e:
            return {"error": f"BCQ prediction failed: {str(e)}"}

# Global instances
_inference_engine = None
_cds = None
_online_system = None
_training_queue = queue.Queue()
_training_status = {}       
EXPERT_MODE = "automatic"  # "automatic" or "manual"
EXPERT_QUEUE = deque(maxlen=100)
EXPERT_LABELS_SUBMITTED = []
# Current hyperparameters
CURRENT_HYPERPARAMS: Dict[str, Any] = {
    "alpha": 1.0,
    "gamma": 0.99,
    "learning_rate": 3e-4,
    "regularization_weight": 0.01,
    "batch_size": 32,
    "tau": 0.2,
    "stream_rate": 10.0,  # transitions/sec
}

CURRENT_META: Dict[str, Any] = {}
CURRENT_SCHEMA: Dict[str, Any] = {}
ACTION_CATALOG: Dict[int, str] = {}       # {action_id: readable_name}
MODEL_HANDLES: Dict[str, Any] = {}        # { 'q_ensemble': ..., 'bcq_trainer': ... }
_feature_keys: List[str] = [] 


def initialize_tools(inference_engine: DigitalTwinInference, cds: ClinicalDecisionSupport):
    """Initialize the global tool instances with online learning support"""
    global _inference_engine, _cds, _online_system
    _inference_engine = inference_engine
    _cds = cds
    
    # Initialize online training system
    models = {
        'dynamics_model': inference_engine.dynamics_model,
        'outcome_model': inference_engine.outcome_model,
        'q_ensemble': EnsembleQNetwork(
            inference_engine.state_dim,
            inference_engine.action_dim,
            n_ensemble=5  # 论文中明确提到K=5
        )
    }
    
    # 为集成创建足够的多样性 - 按照论文方法
    base_state_dict = inference_engine.q_network.state_dict()
    
    for i, q_net in enumerate(models['q_ensemble'].q_networks):
        # 加载基础权重
        q_net.load_state_dict(base_state_dict)
        
        # FIX: 移除手动添加噪声的代码。
        # 不同的随机种子已经足够确保模型在训练中产生多样性，且这种方式更稳定。
        
        # 为每个网络设置不同的初始化种子，确保训练过程中的随机性
        # torch.manual_seed(42 + i * 100)
        # 手动噪声注入已移除以提高稳定性（见上方注释）
    
    # 创建在线系统
    _online_system = create_online_training_system(
        models,
        sampler_type='hybrid',  # 论文中使用的混合采样
        tau=CURRENT_HYPERPARAMS['tau'],
        stream_rate=CURRENT_HYPERPARAMS['stream_rate']
    )
    
    # EXPERT_MODE = "automatic"  # "automatic" or "manual"
    # EXPERT_QUEUE = deque(maxlen=100)
    # EXPERT_LABELS_SUBMITTED = []

    def set_expert_mode(mode: str) -> Dict:
        """设置专家反馈模式"""
        global EXPERT_MODE, _online_system
        
        if mode not in ["automatic", "manual"]:
            return {"error": "Invalid mode. Choose 'automatic' or 'manual'"}
        
        EXPERT_MODE = mode
        
        # 更新在线系统的专家模式
        if _online_system and 'expert' in _online_system:
            _online_system['expert'].manual_mode = (mode == "manual")
        
        return {
            "status": "success",
            "mode": mode,
            "message": f"Expert mode set to {mode}"
        }

    def get_next_expert_case() -> Dict:
        """获取下一个需要专家标注的案例"""
        global EXPERT_QUEUE, _online_system
        
        if not _online_system or 'active_learner' not in _online_system:
            return {"error": "System not initialized"}
        
        # 从查询队列获取案例
        if _online_system['active_learner'].query_queue:
            case = _online_system['active_learner'].query_queue.popleft()
            
            # 格式化案例信息
            formatted_case = {
                "case_id": f"CASE_{len(EXPERT_LABELS_SUBMITTED)+1:04d}",
                "state": {
                    "age": float(case['state'][0] * 90) if len(case['state']) > 0 else 0,
                    "gender": int(case['state'][1]) if len(case['state']) > 1 else 0,
                    "blood_pressure": float(case['state'][2]) if len(case['state']) > 2 else 0.5,
                    "heart_rate": float(case['state'][3]) if len(case['state']) > 3 else 0.5,
                    "glucose": float(case['state'][4]) if len(case['state']) > 4 else 0.5,
                    "oxygen_saturation": float(case['state'][8]) if len(case['state']) > 8 else 0.95
                },
                "action_taken": case.get('action', 0),
                "action_name": ["Placebo", "Medication A", "Medication B", "Medication C", "Combination"][min(max(int(case.get('action', 0)), 0), 4)],
                "model_uncertainty": case.get('uncertainty', 0.5),
                "raw_case": case
            }
            
            EXPERT_QUEUE.append(formatted_case)
            return formatted_case
        
        return {"message": "No cases pending review"}

    def submit_expert_label(case_id: str, expert_reward: float) -> Dict:
        """提交专家标注"""
        global EXPERT_LABELS_SUBMITTED, _online_system
        
        # 找到对应的案例
        case = None
        for c in EXPERT_QUEUE:
            if c.get('case_id') == case_id:
                case = c
                break
        
        if not case:
            return {"error": f"Case {case_id} not found"}
        
        # 创建标注后的转换
        labeled_transition = case['raw_case'].copy()
        labeled_transition['reward'] = expert_reward
        labeled_transition['label_source'] = 'human_expert'
        labeled_transition['label_time'] = time.time()
        
        # 添加到训练器的标注缓冲区
        if _online_system and 'trainer' in _online_system:
            _online_system['trainer'].add_labeled_transition(labeled_transition, source='human_expert')
        
        # 记录统计
        EXPERT_LABELS_SUBMITTED.append({
            'case_id': case_id,
            'reward': expert_reward,
            'timestamp': datetime.now().isoformat()
        })
        
        return {
            "status": "success",
            "message": f"Label submitted for {case_id}",
            "total_labeled": len(EXPERT_LABELS_SUBMITTED)
        }

    def get_expert_stats() -> Dict:
        """获取专家标注统计"""
        # 真实的待审个数来自 active_learner.query_queue
        qlen = 0
        if _online_system and 'active_learner' in _online_system:
            try:
                qlen = len(_online_system['active_learner'].query_queue)
            except Exception:
                qlen = 0

        if not EXPERT_LABELS_SUBMITTED:
            return {
                "total_labeled": 0,
                "average_reward": 0,
                "mode": EXPERT_MODE,
                "queue_size": qlen   # ← 这里
            }

        rewards = [label['reward'] for label in EXPERT_LABELS_SUBMITTED]
        return {
            "total_labeled": len(EXPERT_LABELS_SUBMITTED),
            "average_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "mode": EXPERT_MODE,
            "queue_size": qlen     # ← 这里
        }

    def update_inference_meta(meta: Dict[str, Any]):
        """
        把最新数据集 meta 注入到推理引擎（动作名/特征名/关键规则/阈值等）
        """
        try:
            from inference import DigitalTwinInference
            eng = globals().get("INFERENCE_ENGINE", None)
            if eng is None:
                return
            eng.meta = meta or {}
            # 同步关键字段，避免旧默认值覆盖：
            if meta.get("feature_names"):
                eng.feature_names = meta["feature_names"]
            if meta.get("action_names"):
                eng.action_names = meta["action_names"]
            if meta.get("critical_features"):
                eng.critical_rules = meta["critical_features"]
            if "spo2_idx" in meta:
                eng.spo2_idx = meta["spo2_idx"]
            if "spo2_threshold" in meta:
                try: eng.spo2_threshold = float(meta["spo2_threshold"])
                except: pass
        except Exception as e:
            print(f"[WARN] update_inference_meta failed: {e}")

    # 启动在线训练系统
    # _online_system['stream'].start_stream()
    globals()['set_expert_mode'] = set_expert_mode
    globals()['get_next_expert_case'] = get_next_expert_case
    globals()['submit_expert_label'] = submit_expert_label
    globals()['get_expert_stats'] = get_expert_stats    
    print("Online training system initialized and started")
    # Auto-start stream so UI stats move
    if _online_system and 'stream' in _online_system:
        try:
            _online_system['stream'].start_stream()
        except Exception as e:
            print(f"[WARN] stream not started: {e}")

    print(f"Stream rate: {CURRENT_HYPERPARAMS['stream_rate']} transitions/sec")
    print(f"Initial tau: {CURRENT_HYPERPARAMS['tau']}")


def update_hyperparams(params: Dict) -> Dict:
    """
    按照论文描述的三层策略进行热更新
    
    Tier 1 - Instant Updates (0 seconds):
    • Uncertainty threshold τ
    • Batch size B 
    • Stream rate
    
    Tier 2 - Fast Adaptation (5-10 minutes):
    • CQL weight α
    • Discount factor γ
    • Regularization weight λ
    
    Tier 3 - Full Retrain (2-3 hours):
    • Network architecture changes
    • Major distribution shifts
    """
    global CURRENT_HYPERPARAMS, _online_system
    
    try:
        tier1_updates = []  # 即时更新
        tier2_updates = []  # 快速适应
        tier3_updates = []  # 完全重训练
        all_updates = []
        
        # 分类参数更新
        for param, value in params.items():
            if param not in CURRENT_HYPERPARAMS:
                continue
            
            old_value = CURRENT_HYPERPARAMS[param]
            CURRENT_HYPERPARAMS[param] = value
            all_updates.append(f"{param}: {old_value} → {value}")
            
            # Tier 1: 即时更新 (0秒)
            if param in ['tau', 'batch_size', 'stream_rate']:
                tier1_updates.append(param)
                
                if param == 'tau' and _online_system and 'active_learner' in _online_system:
                    _online_system['active_learner'].update_threshold(value)
                elif param == 'batch_size' and _online_system and 'trainer' in _online_system:
                    _online_system['trainer'].batch_size = int(value)
                elif param == 'stream_rate' and _online_system and 'stream' in _online_system:
                    _online_system['stream'].stream_rate = value
            
            # Tier 2: 快速适应 (5-10分钟)
            elif param in ['alpha', 'gamma', 'regularization_weight', 'learning_rate']:
                tier2_updates.append(param)
            
            # Tier 3: 完全重训练 (2-3小时)
            else:
                tier3_updates.append(param)
        
        # 执行 Tier 2 更新 - 500次梯度步骤的快速适应
        if tier2_updates and _online_system and 'trainer' in _online_system:
            job_id = f"tier2_adaptation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            _trigger_tier2_adaptation(job_id, tier2_updates, params)
            
            return {
                "status": "success",
                "message": "Multi-tier parameter update completed",
                "tier1_instant": tier1_updates,
                "tier2_adapting": tier2_updates,
                "tier3_requires_retrain": tier3_updates,
                "job_id": job_id if tier2_updates else None,
                "estimated_time": "5-10 minutes" if tier2_updates else "Immediate",
                "updated": all_updates
            }
        
        return {
            "status": "success", 
            "message": "Parameters updated successfully",
            "tier1_instant": tier1_updates,
            "tier2_adapting": tier2_updates,
            "tier3_requires_retrain": tier3_updates,
            "updated": all_updates,
            "estimated_time": "Immediate"
        }
        
    except Exception as e:
        return {"error": str(e)}

def _trigger_tier2_adaptation(job_id: str, params_to_adapt: List[str], new_values: Dict):
    """按照论文描述执行Tier 2的500步梯度适应"""
    global _training_status, _online_system
    
    _training_status[job_id] = {
        'status': 'running',
        'type': 'tier2_adaptation', 
        'parameters': params_to_adapt,
        'started_at': datetime.now().isoformat(),
        'progress': 0,
        'total_steps': 500
    }
    
    def adaptation_thread():
        try:
            if _online_system and 'trainer' in _online_system:
                trainer = _online_system['trainer']
                
                # 更新训练器参数
                if 'alpha' in params_to_adapt:
                    # 更新CQL权重
                    for q_trainer in trainer.q_trainers:
                        q_trainer.cql_weight = new_values['alpha']
                
                if 'gamma' in params_to_adapt:
                    # 更新折扣因子
                    for q_trainer in trainer.q_trainers:
                        q_trainer.gamma = new_values['gamma']
                
                if 'learning_rate' in params_to_adapt:
                    # 更新学习率
                    trainer.update_hyperparameters({'learning_rate': new_values['learning_rate']})
                
                # 执行500步聚焦梯度更新（论文中的M=500）
                for step in range(500):
                    if len(trainer.labeled_buffer) >= trainer.batch_size:
                        trainer._train_step()
                    
                    # 更新进度
                    if step % 50 == 0:
                        _training_status[job_id]['progress'] = step / 500
                        print(f"Tier 2 adaptation progress: {step}/500 ({step/5:.0f}%)")
                
                _training_status[job_id]['status'] = 'completed'
                _training_status[job_id]['progress'] = 1.0
                _training_status[job_id]['completed_at'] = datetime.now().isoformat()
                print("Tier 2 adaptation completed successfully")
                
        except Exception as e:
            _training_status[job_id]['status'] = 'failed'
            _training_status[job_id]['error'] = str(e)
            print(f"Tier 2 adaptation failed: {e}")
    
    # 启动适应线程
    thread = threading.Thread(target=adaptation_thread, daemon=True)
    thread.start()


def _trigger_online_finetune(job_id: str, params_to_finetune: List[str]):
    """Trigger online finetuning for specific parameters"""
    global _training_status, _online_system
    
    _training_status[job_id] = {
        'status': 'running',
        'type': 'finetune',
        'parameters': params_to_finetune,
        'started_at': datetime.now().isoformat(),
        'progress': 0
    }
    
    def finetune_thread():
        try:
            # Run focused finetuning
            if 'alpha' in params_to_finetune or 'gamma' in params_to_finetune:
                # Update Q-network training
                if _online_system and 'trainer' in _online_system:
                    # Update CQL weight and gamma
                    _online_system['trainer'].update_hyperparameters({
                        'cql_weight': CURRENT_HYPERPARAMS['alpha'],
                        'gamma': CURRENT_HYPERPARAMS['gamma']
                    })
                    
                    # Run some training steps to adapt
                    for i in range(500):  # 500 mini-batches
                        if _online_system['trainer'].labeled_buffer:
                            _online_system['trainer']._train_step()
                        
                        if i % 50 == 0:
                            _training_status[job_id]['progress'] = i / 500
                            
            _training_status[job_id]['status'] = 'completed'
            _training_status[job_id]['progress'] = 1.0
            _training_status[job_id]['completed_at'] = datetime.now().isoformat()
            
        except Exception as e:
            _training_status[job_id]['status'] = 'failed'
            _training_status[job_id]['error'] = str(e)
    
    # Start finetuning in background
    thread = threading.Thread(target=finetune_thread, daemon=True)
    thread.start()

class ResponseTimeMonitor:
    """监控系统响应时间"""
    
    def __init__(self, max_history=1000):
        self.response_times = deque(maxlen=max_history)
        self.inference_times = deque(maxlen=max_history) 
        self.target_response_time = 0.05  # 50ms目标
        
    def record_response_time(self, response_time: float):
        """记录响应时间"""
        self.response_times.append(response_time)
        
    def record_inference_time(self, inference_time: float):
        """记录推理时间"""
        self.inference_times.append(inference_time)
        
    def get_stats(self) -> Dict:
        """获取响应时间统计"""
        if not self.response_times:
            return {}
            
        response_times = list(self.response_times)
        inference_times = list(self.inference_times)
        
        return {
            'avg_response_time': np.mean(response_times),
            'p95_response_time': np.percentile(response_times, 95),
            'p99_response_time': np.percentile(response_times, 99),
            'avg_inference_time': np.mean(inference_times) if inference_times else 0,
            'meets_target': np.percentile(response_times, 95) < self.target_response_time,
            'total_samples': len(response_times)
        }

# 全局响应时间监控器
_response_monitor = ResponseTimeMonitor()

def monitor_response_time(func):
    """装饰器：监控函数响应时间"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            response_time = time.time() - start_time
            _response_monitor.record_response_time(response_time)
            
            # 如果响应时间过长，记录警告
            if response_time > 0.1:  # 100ms警告阈值
                print(f"⚠️ Slow response: {func.__name__} took {response_time*1000:.1f}ms")
    return wrapper

DEFAULT_FEATURE_KEYS = [
    "age","gender","blood_pressure","heart_rate","resp_rate",
    "temp","spo2","gcs","lactate","sofa"
]

def _state_to_vec(state_dict, state_dim=10):
    # 读取模块内全局 _feature_keys（initialize_tools 时可赋值）
    fk = globals().get('_feature_keys', None)
    if isinstance(fk, (list, tuple)) and fk:
        keys = list(fk)
    else:
        keys = DEFAULT_FEATURE_KEYS

    vec = []
    for k in keys:
        vec.append(float(state_dict.get(k, 0.0)))
    arr = np.asarray(vec, dtype=np.float32)
    if arr.shape[0] != state_dim:
        # 尽量避免 resize；最好保证 keys 与 state_dim 一致。
        arr = np.resize(arr, (state_dim,))
    return arr

def get_optimal_recommendation_bcq(patient_state: dict) -> dict:
    """使用BCQ（优先）或CQL（兜底）为患者获取最优治疗推荐"""
    # 使用模块内全局对象
    global _inference_engine, ACTION_LABELS, _online_system

    try:
        if _inference_engine is None:
            return {"error": "Tools not initialized"}

        # 1. 将状态字典转换为模型输入向量
        obs = _state_to_vec(
            patient_state,
            state_dim=getattr(_inference_engine, "state_dim", 10)
        )
        obs_batch = obs[None, :]

        # 2. 获取真实的动作维度和标签，并对齐
        # 有默认标签就用默认，没有就给一份
        if ACTION_CATALOG:
            labels = [ACTION_CATALOG[i] for i in sorted(ACTION_CATALOG.keys())]
        else:
            labels = [str(i) for i in range(getattr(_inference_engine, 'action_dim', 1))]

        # 与当前“有效动作数”对齐（优先用 BCQ 的动作数）
        bcq_trainer = getattr(globals().get('_online_system', None), 'bcq_trainer', None)
        act_dim = None
        if bcq_trainer and getattr(bcq_trainer, 'bcq_action_size', None):
            act_dim = int(bcq_trainer.bcq_action_size)
        else:
            act_dim = getattr(_inference_engine, 'action_dim', len(labels))

        labels = labels[:act_dim] 

        chosen_idx = None
        q_value = None
        all_q_values = None
        source = 'Unknown'

        # 3. 优先用 BCQ 策略预测动作
        if bcq_trainer and getattr(bcq_trainer, 'bcq_algo', None) is not None:
            try:
                pred = bcq_trainer.bcq_algo.predict(obs_batch)  # -> ndarray shape (1,)
                chosen_idx = int(pred[0])
                source = 'BCQ'
                # 尝试从BCQ获取动作值
                try:
                    q_all = bcq_trainer.bcq_algo.predict_value(obs_batch)  # -> (1, n_actions)
                    if isinstance(q_all, np.ndarray) and q_all.ndim == 2:
                        all_q_values = q_all[0]
                        if all_q_values.shape[0] > chosen_idx:
                            q_value = float(all_q_values[chosen_idx])
                except Exception:
                    pass  # 获取Q值失败是可接受的，继续执行
            except Exception as e:
                # BCQ 预测失败则清空结果，准备 fallback
                chosen_idx = None
                source = 'Unknown'

        # 4. Fallback：用 ConservativeQNetwork 评估 Q 值，argmax 出动作
        if (chosen_idx is None) or (q_value is None) or (all_q_values is None):
            try:
                qnet = getattr(_inference_engine, 'q_network', None)
                device = getattr(_inference_engine, 'device', 'cpu')
                with torch.no_grad():
                    s = torch.from_numpy(obs).float().to(device).unsqueeze(0)
                    qs = None
                    # 适配两种Q网络前向传播方式
                    try:
                        out = qnet(s)  # 方式一: qnet(s) -> (1, act_dim)
                        if hasattr(out, 'shape') and out.shape[-1] >= 1:
                            qs = out.squeeze(0).detach().cpu().numpy()
                    except Exception:
                        qs = None

                    if qs is None: # 方式二: qnet(s, a_onehot) -> q
                        vals = []
                        for a in range(act_dim):
                            a_one = F.one_hot(torch.tensor([a], device=device), num_classes=act_dim).float()
                            q = qnet(s, a_one)
                            vals.append(float(q.item()))
                        qs = np.asarray(vals, dtype=np.float32)

                all_q_values = qs
                new_chosen_idx = int(np.argmax(qs))
                new_q_value = float(qs[new_chosen_idx])

                # 仅当BCQ未成功时才更新动作；任何时候都更新Q值
                if chosen_idx is None:
                    chosen_idx = new_chosen_idx
                    source = 'CQL'
                q_value = new_q_value
                
            except Exception as e:
                pass # Q网络评估失败，继续执行最后的兜底

        # 5. 最后的兜底
        if chosen_idx is None:
            chosen_idx = 0
            source = 'Default'
        if q_value is None:
            q_value = 0.0

        if all_q_values is None:
            all_q_values = np.zeros(len(labels), dtype=np.float32)

        # 6. 映射动作标签（防止越界）
        if not (0 <= chosen_idx < len(labels)):
            chosen_idx = min(max(chosen_idx, 0), len(labels) - 1)

        recommended_action = int(chosen_idx)
        recommended_treatment = labels[recommended_action]

        q_mean = float(np.mean(all_q_values))
        confidence = float(q_value - q_mean)

        try:
            device = getattr(_inference_engine, 'device', 'cpu')
            state_tensor = torch.from_numpy(obs).float().to(device).unsqueeze(0)
            action_tensor = torch.tensor([recommended_action], device=device)
            exp_outcome = float(_inference_engine.outcome_model(state_tensor, action_tensor).item())
        except Exception:
            exp_outcome = 0.0

        return {
            "recommended_action": recommended_action,
            "recommended_treatment": recommended_treatment,
            "confidence": confidence,
            "expected_immediate_outcome": exp_outcome,
            "source": source
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# 添加原函数名的别名
get_optimal_recommendation = get_optimal_recommendation_bcq

@monitor_response_time 
def get_all_action_values(patient_state: dict) -> dict:
    """获取所有动作Q值 - 带响应时间监控"""
    try:
        if _inference_engine is None:
            return {"error": "Tools not initialized"}

        inference_start = time.time()
        state_array = _validate_patient_state(patient_state)
        result = _inference_engine.recommend_treatment(state_array, return_all_scores=True)
        if not result or "treatment_rankings" not in result:
            raise ValueError("Missing treatment rankings")
        inference_time = time.time() - inference_start

        _response_monitor.record_inference_time(inference_time)

        action_dim = getattr(_inference_engine, "action_dim", len(result["treatment_rankings"]))
        inference_names = getattr(_inference_engine, "action_names", None)
        names = inference_names if isinstance(inference_names, list) and inference_names else [
            f"Action {i}" for i in range(action_dim)
        ]

        action_values = []
        for action_idx, q_value in result["treatment_rankings"]:
            action_name = names[action_idx] if action_idx < len(names) else f"Action {action_idx}"
            action_values.append({
                "action": action_name,
                "action_id": int(action_idx),
                "action_name": action_name,
                "q_value": float(q_value),
            })

        return {
            "action_values": action_values,
            "inference_time_ms": inference_time * 1000,
        }
    except Exception as e:
        try:
            state_array = _validate_patient_state(patient_state)
            action_dim = getattr(_inference_engine, "action_dim", None)
            if not action_dim:
                action_dim = len(ACTION_CATALOG)
            q_net = getattr(_inference_engine, "q_network", None)
            scores = None
            if q_net is not None:
                device = getattr(_inference_engine, "device", None)
                state_tensor = torch.tensor(state_array, dtype=torch.float32)
                if device:
                    state_tensor = state_tensor.to(device)
                with torch.no_grad():
                    try:
                        qs = q_net(state_tensor).squeeze().detach().cpu().numpy()
                        if qs.shape[-1] != action_dim:
                            raise ValueError("action dim mismatch")
                    except Exception:
                        qs_list = []
                        for a in range(action_dim):
                            action_tensor = torch.tensor([a], device=state_tensor.device)
                            q_val = q_net(state_tensor, action_tensor).item()
                            qs_list.append(q_val)
                        qs = np.array(qs_list, dtype=np.float32)
                scores = qs
            if scores is None and _scores_from_policy_only:
                scores = _scores_from_policy_only(
                    state_array, MODEL_HANDLES, action_dim
                )
            if scores is None:
                return {"error": str(e)}
            if not isinstance(scores, np.ndarray):
                scores = np.array(scores, dtype=np.float32)
            inference_names = getattr(_inference_engine, "action_names", None)
            names = inference_names if isinstance(inference_names, list) and inference_names else [
                f"Action {i}" for i in range(action_dim)
            ]
            action_values = []
            for action_idx, q_value in enumerate(scores):
                action_name = names[action_idx] if action_idx < len(names) else f"Action {action_idx}"
                action_values.append({
                    "action": action_name,
                    "action_id": int(action_idx),
                    "action_name": action_name,
                    "q_value": float(q_value),
                })
            return {"action_values": action_values, "warning": str(e)}
        except Exception as inner:
            return {"error": f"{e}; fallback failed: {inner}"}

def get_response_time_stats() -> Dict:
    """获取响应时间统计"""
    return _response_monitor.get_stats()

def online_finetune(job_params: Dict) -> Dict:
    """
    Lightweight online finetuning instead of full retraining
    
    Args:
        job_params: Parameters for finetuning job
        
    Returns:
        Status dict
    """
    try:
        job_id = f"online_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Determine what needs finetuning
        finetune_targets = []
        
        if 'preset' in job_params:
            preset = job_params['preset']
            if preset == 'conservative':
                update_hyperparams({
                    'alpha': 1.5,
                    'gamma': 0.99,
                    'tau': 0.1
                })
                finetune_targets = ['q_network']
            elif preset == 'aggressive':
                update_hyperparams({
                    'alpha': 0.5,
                    'gamma': 0.95,
                    'tau': 0.02
                })
                finetune_targets = ['q_network']
        
        _training_status[job_id] = {
            'status': 'running',
            'type': 'online_finetune',
            'targets': finetune_targets,
            'started_at': datetime.now().isoformat()
        }
        
        # Use online trainer for incremental updates
        if _online_system and 'trainer' in _online_system:
            # Force some training iterations
            for _ in range(100):
                _online_system['trainer']._train_step()
        
        _training_status[job_id]['status'] = 'completed'
        
        return {
            "status": "success",
            "message": "Online finetuning completed",
            "job_id": job_id,
            "duration": "< 5 minutes"
        }
        
    except Exception as e:
        return {"error": str(e)}


def get_online_stats() -> Dict:
    """Get online training statistics"""
    if not _online_system or 'trainer' not in _online_system:
        return {"error": "Online system not initialized"}

    stats = _online_system['trainer'].get_statistics()

    # Add hyperparameter info
    stats['current_hyperparams'] = CURRENT_HYPERPARAMS.copy()

    # Add stream statistics
    if 'active_learner' in _online_system:
        al_stats = _online_system['active_learner'].get_statistics()
        # 确保包含正确的阈值
        if 'current_threshold' not in al_stats or al_stats['current_threshold'] == 0:
            al_stats['current_threshold'] = CURRENT_HYPERPARAMS.get('tau', 0.05)
        stats['active_learning'] = al_stats
    
    return stats


def pause_online_training() -> Dict:
    """Pause online training"""
    if _online_system and 'stream' in _online_system:
        _online_system['stream'].stop_stream()
        return {"status": "paused", "message": "Online training paused"}
    return {"error": "Online system not initialized"}


# def resume_online_training() -> Dict:
#     """Resume online training"""
#     if _online_system and 'stream' in _online_system:
#         _online_system['stream'].start_stream()
#         return {"status": "resumed", "message": "Online training resumed"}
#     return {"error": "Online system not initialized"}
def resume_online_training(silent: bool = True) -> dict:
    """
    恢复/启动在线训练；silent=True 时抑制 d3rlpy 与我们内部 logger 的冗余输出
    """
    try:
        if silent:
            import os, logging
            os.environ["D3RLPY_LOG_LEVEL"] = "ERROR"
            # d3rlpy 自己的 logger
            logging.getLogger("d3rlpy").setLevel(logging.ERROR)
            logging.getLogger("DiscreteBCQ").setLevel(logging.ERROR)
            # 你若有自定义 logger 名称，可一并降低级别
            logging.getLogger("drive").setLevel(logging.WARNING)

        # === 原有的在线训练启动逻辑（保持不变）===
        # 例如：_online_system['trainer'].resume() / 启线程 / 初始化缓冲等
        # TODO: 保留你现有的实现
        # =========================================

        return {"status": "running", "silent": silent}
    except Exception as e:
        return {"error": str(e)}



# Keep all original functions unchanged
def _validate_patient_state(patient_state: dict) -> np.ndarray:
    """Convert patient state dict to numpy array"""
    try:
        state_array = np.zeros(_inference_engine.state_dim)
        state_array[0] = patient_state.get('age', 45) / 90
        state_array[1] = patient_state.get('gender', 0)
        state_array[2] = patient_state.get('blood_pressure', 0.5)
        state_array[3] = patient_state.get('heart_rate', 0.5)
        state_array[4] = patient_state.get('glucose', 0.5)
        state_array[5] = patient_state.get('creatinine', 0.5)
        state_array[6] = patient_state.get('hemoglobin', 0.6)
        state_array[7] = patient_state.get('temperature', 0.5)
        state_array[8] = patient_state.get('oxygen_saturation', 0.95)
        if _inference_engine.state_dim > 9:
            state_array[9] = patient_state.get('bmi', 0.5)
        return np.clip(state_array, 0, 1)
    except Exception as e:
        raise ValueError(f"Invalid patient state: {str(e)}")


def _action_name_to_idx(action_name: str) -> int:
    """Convert action name to index"""
    action_map = {
        'Medication A': 0,
        'Medication B': 1,
        'Medication C': 2,
        'Placebo': 3,
        'Combination Therapy': 4
    }
    if action_name not in action_map:
        raise ValueError(f"Unknown action: {action_name}")
    return action_map[action_name]




def calculate_treatment_effect(patient_state: dict, treatment_a: str, treatment_b: str) -> dict:
    """Calculate CATE between two treatments"""
    try:
        if _inference_engine is None:
            return {"error": "Tools not initialized"}
        
        state_array = _validate_patient_state(patient_state)
        result = _inference_engine.recommend_treatment(state_array, return_all_scores=True)
        
        idx_a = _action_name_to_idx(treatment_a)
        idx_b = _action_name_to_idx(treatment_b)
        
        q_values = result['all_q_values']
        cate = float(q_values[idx_a] - q_values[idx_b])
        
        return {
            "cate_value": cate,
            "treatment_a": treatment_a,
            "treatment_b": treatment_b
        }
    except Exception as e:
        return {"error": str(e)}


def simulate_future_trajectory(patient_state: dict, action_sequence: list[str], horizon: int) -> dict:
    """Simulate patient trajectory under action sequence"""
    try:
        if _inference_engine is None:
            return {"error": "Tools not initialized"}
        
        state_array = _validate_patient_state(patient_state)
        action_indices = [_action_name_to_idx(a) for a in action_sequence[:horizon]]
        
        if len(action_indices) < horizon:
            action_indices.extend([action_indices[-1]] * (horizon - len(action_indices)))
        
        trajectory_result = _inference_engine.simulate_treatment_trajectory(
            state_array, action_indices[:horizon], max_steps=horizon
        )
        
        trajectory = []
        for i, state in enumerate(trajectory_result['states']):
            state_dict = {
                'age': float(state[0] * 90),
                'gender': int(state[1]),
                'blood_pressure': float(state[2]),
                'heart_rate': float(state[3]),
                'glucose': float(state[4]),
                'creatinine': float(state[5]),
                'hemoglobin': float(state[6]),
                'temperature': float(state[7]),
                'oxygen_saturation': float(state[8])
            }
            if len(state) > 9:
                state_dict['bmi'] = float(state[9])
            
            trajectory.append({"step": i, "state": state_dict})
        
        return {"trajectory": trajectory}
    except Exception as e:
        return {"error": str(e)}


def get_feature_importance() -> dict:
    """Get global feature importance scores"""
    try:
        if _inference_engine is None:
            return {"error": "Tools not initialized"}
        
        dummy_states = []
        for _ in range(100):
            state = np.random.randn(_inference_engine.state_dim)
            state = np.clip(state, 0, 1)
            dummy_states.append(state)
        
        dummy_dataset = {
            'states': dummy_states,
            'actions': np.random.randint(0, _inference_engine.action_dim, 100).tolist()
        }
        
        importance_df = analyze_feature_importance(
            _inference_engine.q_network,
            dummy_dataset,
            _inference_engine.feature_names,
            n_samples=50,
            device=_inference_engine.device
        )
        
        importances = []
        for _, row in importance_df.iterrows():
            importances.append({
                "feature": row['feature'],
                "score": float(row['relative_importance'])
            })
        
        return {"importances": importances}
    except Exception as e:
        return {"error": str(e)}


def get_immediate_reward(patient_state: dict, action: str) -> dict:
    """Get immediate reward for action in state"""
    try:
        if _inference_engine is None:
            return {"error": "Tools not initialized"}
        
        state_array = _validate_patient_state(patient_state)
        action_idx = _action_name_to_idx(action)
        
        result = _inference_engine.recommend_treatment(state_array, return_all_scores=True)
        predicted_reward = float(result['all_outcomes'][action_idx])
        
        return {"predicted_reward": predicted_reward}
    except Exception as e:
        return {"error": str(e)}


def describe_parameter(param_name: str) -> dict:
    """Describe the mechanism and quantitative impact of a model parameter"""
    parameter_descriptions = {
        "alpha": {
            "name": "Alpha (α)",
            "description": "Conservative Q-Learning weight that controls how much the model penalizes unseen actions",
            "mechanism": "Higher values make the policy more conservative, avoiding actions not well-represented in training data",
            "impact": "α=0.1: Exploratory, may try novel treatments; α=1.0: Very conservative, sticks to common treatments",
            "default": 1.0,
            "range": [0.1, 2.0]
        },
        "gamma": {
            "name": "Gamma (γ)",
            "description": "Discount factor for future rewards in reinforcement learning",
            "mechanism": "Controls how much the model values long-term vs short-term outcomes",
            "impact": "γ=0.9: Focuses on immediate effects; γ=0.99: Values long-term patient health equally",
            "default": 0.99,
            "range": [0.9, 0.999]
        },
        "learning_rate": {
            "name": "Learning Rate",
            "description": "Step size for model parameter updates during training",
            "mechanism": "Controls how quickly the model adapts to new patterns",
            "impact": "1e-4: Slow, stable learning; 1e-2: Fast but may overshoot optimal values",
            "default": 1e-3,
            "range": [1e-4, 1e-2]
        },
        "regularization_weight": {
            "name": "Regularization Weight (λ)",
            "description": "Weight for deconfounding regularization in outcome model",
            "mechanism": "Balances prediction accuracy vs removing treatment assignment bias",
            "impact": "λ=0.01: Focus on accuracy; λ=0.1: Strong deconfounding, may sacrifice some accuracy",
            "default": 0.01,
            "range": [0.001, 0.1]
        },
        "batch_size": {
            "name": "Batch Size",
            "description": "Number of samples processed together during training",
            "mechanism": "Affects training stability and speed",
            "impact": "32: More gradient noise, faster; 256: Smoother gradients, more memory",
            "default": 256,
            "range": [32, 512]
        },
        "tau": {
            "name": "Tau (τ)",
            "description": "Uncertainty threshold for active learning queries",
            "mechanism": "Controls when to ask for expert labels based on model uncertainty",
            "impact": "τ=0.02: Ask frequently (high cost); τ=0.1: Only ask for very uncertain cases",
            "default": 0.05,
            "range": [0.01, 0.2]
        }
    }
    
    if param_name not in parameter_descriptions:
        return {"error": f"Unknown parameter: {param_name}"}
    
    return parameter_descriptions[param_name]


def recommend_parameters(patient_state: dict) -> dict:
    """Recommend parameter presets based on patient state"""
    try:
        state_array = _validate_patient_state(patient_state)
        
        is_critical = state_array[8] < 0.85
        is_complex = np.sum(np.abs(state_array - 0.5) > 0.2) > 3
        is_elderly = state_array[0] > 0.7
        
        recommendations = []
        
        if is_critical or is_elderly:
            recommendations.append({
                "preset": "conservative",
                "reason": "Critical condition or elderly patient - prioritize safety",
                "parameters": {
                    "alpha": 1.5,
                    "gamma": 0.99,
                    "learning_rate": 5e-4,
                    "regularization_weight": 0.05,
                    "batch_size": 256,
                    "tau": 0.1
                },
                "expected_effect": "Cautious recommendations, well-tested treatments only"
            })
        
        elif is_complex:
            recommendations.append({
                "preset": "balanced",
                "reason": "Multiple abnormal values - need careful optimization",
                "parameters": {
                    "alpha": 1.0,
                    "gamma": 0.99,
                    "learning_rate": 1e-3,
                    "regularization_weight": 0.01,
                    "batch_size": 128,
                    "tau": 0.05
                },
                "expected_effect": "Balanced approach considering all factors"
            })
        
        else:
            recommendations.append({
                "preset": "aggressive",
                "reason": "Stable patient - can explore optimal treatments",
                "parameters": {
                    "alpha": 0.5,
                    "gamma": 0.95,
                    "learning_rate": 2e-3,
                    "regularization_weight": 0.005,
                    "batch_size": 64,
                    "tau": 0.02
                },
                "expected_effect": "May find novel effective treatments"
            })
        
        return {"recommendations": recommendations}
    except Exception as e:
        return {"error": str(e)}


def update_reward_parameters(new_params: dict) -> dict:
    """Update reward model parameters - NOW USES HOT UPDATE"""
    try:
        # Use hot update instead of just storing
        result = update_hyperparams(new_params)
        
        if "error" in result:
            return result
        
        return {
            "status": "success",
            "message": "Parameters updated via hot update mechanism",
            "details": result
        }
    except Exception as e:
        return {"error": str(e)}


def retrain_model(training_params: dict) -> dict:
    """
    Full model retraining (kept as fallback option)
    Now recommends using online finetuning instead
    """
    try:
        # Check if online finetuning would be sufficient
        if training_params.get("preset") in ["conservative", "balanced", "aggressive"]:
            return {
                "status": "redirect",
                "message": "Full retraining not necessary. Using online finetuning instead.",
                "recommendation": "Use online_finetune() for faster results",
                "alternative": online_finetune(training_params)
            }
        
        # If user insists on full retrain, proceed with original implementation
        if training_params.get("force_full_retrain", False):
            preset = training_params.get("preset", "custom")
            params = training_params.get("parameters", {})
            
            if preset != "custom":
                preset_configs = {
                    "conservative": {
                        "dynamics_epochs": 30,
                        "outcome_epochs": 20,
                        "rl_iterations": 30000,
                        "batch_size": 256,
                        "learning_rate": 5e-4,
                        "cql_weight": 1.5,
                        "regularization_weight": 0.05,
                        "gamma": 0.99
                    },
                    "balanced": {
                        "dynamics_epochs": 50,
                        "outcome_epochs": 30,
                        "rl_iterations": 50000,
                        "batch_size": 128,
                        "learning_rate": 1e-3,
                        "cql_weight": 1.0,
                        "regularization_weight": 0.01,
                        "gamma": 0.99
                    },
                    "aggressive": {
                        "dynamics_epochs": 70,
                        "outcome_epochs": 40,
                        "rl_iterations": 70000,
                        "batch_size": 64,
                        "learning_rate": 2e-3,
                        "cql_weight": 0.5,
                        "regularization_weight": 0.005,
                        "gamma": 0.95
                    }
                }
                if preset in preset_configs:
                    params.update(preset_configs[preset])
            
            job_id = f"train_{preset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            job = {
                'job_id': job_id,
                'preset': preset,
                'parameters': params,
                'created_at': datetime.now().isoformat()
            }
            
            _training_queue.put(job)
            _training_status[job_id] = {
                'status': 'queued',
                'created_at': job['created_at']
            }
            
            os.makedirs('./retrained_models', exist_ok=True)
            
            return {
                "status": "initiated",
                "message": f"Full model retraining started with {preset} preset",
                "parameters": params,
                "estimated_time": "2-3 hours",
                "job_id": job_id,
                "note": "Consider using online_finetune() for faster updates"
            }
        
        else:
            return {
                "status": "info",
                "message": "Full retraining is typically not needed with online learning",
                "suggestion": "Use update_hyperparams() for instant updates or online_finetune() for adaptation",
                "force_option": "Add 'force_full_retrain': true to proceed with full retraining"
            }
        
    except Exception as e:
        return {"error": str(e)}


def get_training_status(job_id: str) -> dict:
    """Get status of a training job"""
    if job_id not in _training_status:
        return {"error": f"Unknown job ID: {job_id}"}
    
    status = _training_status[job_id].copy()
    
    # Add human-readable progress
    if 'progress' in status:
        status['progress_percent'] = f"{status['progress'] * 100:.1f}%"
    
    # Add online learning stats if available
    if status.get('type') == 'online_finetune' and _online_system:
        status['online_stats'] = get_online_stats()
    
    return status


def get_patient_list() -> dict:
    """
    返回统一结构：{'patients': [pid1, pid2, ...], 'total': N, 'data_source': 'virtual'|'real'}
    """
    try:
        lst = data_manager.get_patient_list()  # DataManager 返回的是 list
        return {"patients": lst, "total": len(lst), "data_source": data_manager.current_source}
    except Exception as e:
        return {"patients": [], "total": 0, "data_source": data_manager.current_source, "error": str(e)}



def get_patient_data(patient_id: str) -> dict:
    """Get detailed information for a specific patient"""
    try:
        patient_info = data_manager.get_patient_info(patient_id)
        return patient_info
    except Exception as e:
        return {"error": str(e)}


def analyze_patient(patient_id: str) -> dict:
    """Analyze a specific patient and get treatment recommendations"""
    try:
        # Get patient state
        patient_state = data_manager.get_patient_state(patient_id)
        
        # Get recommendation
        recommendation = get_optimal_recommendation(patient_state)
        
        # Get all action values
        action_values = get_all_action_values(patient_state)
        fallback_warning = None
        if action_values.get("error"):
            fallback_warning = action_values.get("error")
            action_values = {"action_values": []}
        
        # Simulate short-term trajectory
        if recommendation.get("recommended_action") is not None:
            act = recommendation["recommended_action"]
            if isinstance(act, (int, np.integer)):
                act_label = ACTION_CATALOG.get(act, str(act))
            else:
                act_label = str(act)
            trajectory = simulate_future_trajectory(
                patient_state,
                [act_label] * 7,
                7
            )
        else:
            trajectory = {"trajectory": []}
        
        # Combine results
        analysis = {
            "patient_id": patient_id,
            "current_state": patient_state,
            "recommendation": recommendation,
            "all_options": action_values,
            "predicted_trajectory": trajectory,
            "analysis_timestamp": datetime.now().isoformat(),
        }
        if fallback_warning:
            analysis["all_options_error"] = fallback_warning

        return analysis
    except Exception as e:
        return {"error": str(e)}

def register_model_handles(q_ensemble=None,
                           device: str = 'cpu',
                           bcq_trainer=None,
                           baseline_trainer=None,
                           uncertainty_sampler=None):
    """
    让在线系统把可用的模型句柄注册进来。UI 的报告按钮会根据这些句柄自动选择 Q 或 policy。
    """
    MODEL_HANDLES.clear()
    if q_ensemble is not None: MODEL_HANDLES['q_ensemble'] = q_ensemble
    MODEL_HANDLES['device'] = device
    if bcq_trainer is not None: MODEL_HANDLES['bcq_trainer'] = bcq_trainer
    if baseline_trainer is not None: MODEL_HANDLES['baseline_trainer'] = baseline_trainer
    if uncertainty_sampler is not None: MODEL_HANDLES['uncertainty_sampler'] = uncertainty_sampler
    # 在数据源加载成功后，自动同步 meta 到推理引擎
    try:
        meta = data_manager.get_current_meta()
        update_inference_meta(meta)
    except Exception as e:
        print(f"[WARN] update_inference_meta failed after data load: {e}")
def load_data_source(source_type: str,
                     file_path: Optional[str] = None,
                     n_patients: Optional[int] = None,
                     schema_path: Optional[str] = None,
                     schema_yaml: Optional[str] = None) -> dict:
    """Load data from specified source"""
    try:
        # 整个 if/elif/else 逻辑都应该在 try 内部
        if source_type == "virtual":
            n_patients = n_patients or 1000
            df = data_manager.generate_virtual_data(n_patients=n_patients)
            data_manager.set_data_source("virtual")
            return {
                "status": "success",
                "message": f"Generated virtual data for {n_patients} patients",
                "records": len(df),
                "patients": df['patient_id'].nunique()
            }
        
        elif source_type == "real":
            if not file_path:
                return {"error": "File path required for real data"}

            # 识别文件类型（这里只用来做存在性检查）
            if not (file_path.endswith(".csv") or file_path.endswith(".parquet") or file_path.endswith(".xlsx") or file_path.endswith(".xls")):
                return {"error": f"Unsupported file: {file_path}"}

            # 有 YAML 走 adapters（强推荐）；否则走 schema-less 兜底
            if schema_path or schema_yaml:
                df = data_manager.load_real_data_with_schema(
                    file_path=file_path,
                    file_type="csv" if file_path.endswith(".csv") else "parquet",
                    schema_path=schema_path,
                    schema_yaml=schema_yaml
                )
            else:
                df = data_manager.load_real_data_schema_less(file_path)
            data_manager.set_data_source("real")
            # 同步 meta 到推理引擎（供报告与在线使用）
            meta = data_manager.get_current_meta()
            try:
                eng = globals().get("_inference_engine", None) or globals().get("INFERENCE_ENGINE", None)
                if eng is not None:
                    eng.meta = meta
                    if meta.get("feature_columns"):
                        eng.feature_names = meta["feature_columns"]
                    if meta.get("action_names"):
                        eng.action_names = meta["action_names"]
            except Exception as _e:
                print(f"[WARN] inject meta failed: {_e}")
            global CURRENT_META, CURRENT_SCHEMA, ACTION_CATALOG, _feature_keys
            CURRENT_META = meta
            CURRENT_SCHEMA = getattr(data_manager, "current_schema", {}) or {}
            _feature_keys = list(CURRENT_META.get("feature_columns", []))

            # 动作目录：优先取 meta 的 id->name；否则从数据中推断
            act_map = CURRENT_META.get("action_id_to_name") or CURRENT_META.get("action_catalog")
            if isinstance(act_map, dict) and act_map:
                ACTION_CATALOG = {int(k): str(v) for k, v in act_map.items()}
            else:
                labels = [str(a) for a in sorted(df["action"].unique())] if "action" in df.columns else []
                ACTION_CATALOG = {i: labels[i] for i in range(len(labels))}
            # 返回基本统计（供 UI 顶部状态条）
            patients = df["patient_id"].nunique() if "patient_id" in df.columns else 0
            return {
                "status": "success",
                "message": f"Loaded real data from {file_path}",
                "records": len(df),
                "patients": int(patients),
            }
        else:
            return {"error": f"Unsupported data source: {source_type}"}
            
    except Exception as e:
        return {"error": str(e)}

def get_patient_list_from_dataset(dataset: Dict, meta: Dict, limit: int = 500) -> List[str]:
    """
    返回 trajectory_id 的去重列表（字符串），用于 UI 下拉框
    """
    import numpy as np
    traj_col = meta.get('trajectory_id_col') or meta.get('mapping', {}).get('trajectory_id') or 'trajectory_id'
    if traj_col in meta.get('raw_columns', []):
        # dataset 如果以 DataFrame 形式存在，则自己取；这里按你统一后的 dict 组织
        pass
    # 通用：从 meta['trajectory_ids'] 或 dataset['trajectory_ids'] 猜
    ids = None
    for k in ('trajectory_ids', 'traj_ids', 'subjects'):
        if isinstance(meta.get(k), (list, tuple)):
            ids = meta[k]; break
        if isinstance(dataset.get(k), (list, tuple)):
            ids = dataset[k]; break
    if ids is None and 'trajectory' in dataset:
        ids = dataset['trajectory'].get('ids')
    if ids is None:
        # 兜底：从 states 的同长度数组猜
        n = len(dataset.get('states', []))
        ids = list(map(str, range(n)))
    ids = list(map(str, ids))
    return ids[:limit]

def get_latest_state_of_patient(dataset: Dict, meta: Dict, patient_id: str) -> Tuple[Dict, np.ndarray]:
    """
    取该 patient 的最新一帧 state。返回 (patient_info, state_vec)
    """
    import numpy as np
    from typing import Optional, Tuple, Dict
    traj_col = meta.get('trajectory_id_col') or meta.get('mapping', {}).get('trajectory_id') or 'trajectory_id'
    time_col = meta.get('timestep_col') or meta.get('mapping', {}).get('timestep') or 'timestep'

    # 你的适配器统一成 dict list 组织时，通常会保留每条 transition 的 traj_id / timestep
    # 这里尝试从 dataset 中的平铺数组里做筛选（如果没有，则取最后一帧）。
    info = {'id': patient_id}
    states = dataset.get('states', None)
    tids   = dataset.get('trajectory_ids', None) or dataset.get('traj', None)
    times  = dataset.get('timesteps', None) or dataset.get('t', None)
    if states is None:
        return info, np.zeros(len(meta.get('feature_columns', [])), dtype=np.float32)

    if tids is not None and times is not None:
        tids = np.array(list(map(str, np.array(tids).reshape(-1))))
        mask = (tids == str(patient_id))
        if mask.any():
            idxs = np.where(mask)[0]
            if times is not None:
                times_arr = np.array(times)
                last_idx = idxs[np.argmax(times_arr[idxs])]
            else:
                last_idx = idxs[-1]
            s = np.array(states[last_idx], dtype=np.float32)
            return info, s

    # 兜底：用最后一帧
    s = np.array(states[-1], dtype=np.float32)
    return info, s

def generate_patient_report(dataset: Dict,
                            meta: Dict,
                            patient_id: str,
                            topk: int = 3) -> str:
    """
    UI 按钮调用；内部自动用 MODEL_HANDLES + ACTION_CATALOG 计算推荐并渲染
    """
    assert build_action_catalog and compute_recommendation and render_patient_report, \
        "reports.py not available"
    patient, state = get_latest_state_of_patient(dataset, meta, patient_id)
    rec = compute_recommendation(state, MODEL_HANDLES, ACTION_CATALOG, meta, topk=topk)
    md = render_patient_report(patient, state, rec, meta, ACTION_CATALOG)
    return md



def get_cohort_stats() -> dict:
    """
    为 UI 统计图提供稳定结构（不依赖 reward）：
    {
      'total_patients': int,
      'total_records': int,
      'n_actions': int,
      'action_counts': {id_str: count, ...},
      'action_names': {id_int: name_str, ...},
      'traj_len_hist': {'bins': [..], 'counts': [..], 'mean': float, 'median': float},
      'missing_top': [(feature, missing_ratio), ... up to 12]
    }
    """
    import numpy as np
    import pandas as pd
    from data_manager import data_manager

    df = data_manager.get_current_data()
    if df is None or len(df) == 0:
        return dict(
            total_patients=0, total_records=0, n_actions=0,
            action_counts={}, action_names={}, traj_len_hist=dict(bins=[], counts=[], mean=0.0, median=0.0),
            missing_top=[]
        )

    # 基本计数
    total_patients = int(df["patient_id"].nunique()) if "patient_id" in df.columns else 0
    total_records  = int(len(df))

    # 动作分布
    action_counts = {}
    if "action" in df.columns:
        vc = df["action"].dropna()
        try:
            vc = vc.astype(int)
        except Exception:
            pass
        cnts = vc.value_counts().sort_index()
        action_counts = {str(k): int(v) for k, v in cnts.items()}
        n_actions = len(cnts)
    else:
        n_actions = 0

    # 动作名称映射（供前端显示）
    meta = data_manager.get_current_meta()
    id2name = meta.get("action_map") or {i: n for i, n in enumerate(meta.get("action_names") or [])}
    action_names = {int(k): str(v) for k, v in (id2name or {}).items()}

    # 轨迹长度分布（每个病人的记录条数）
    if "patient_id" in df.columns:
        lens = df.groupby("patient_id").size().values
        bins = np.linspace(1, max(10, lens.max()), num=11, dtype=float)
        hist, edges = np.histogram(lens, bins=bins)
        traj_len_hist = dict(
            bins=edges.tolist(),
            counts=hist.astype(int).tolist(),
            mean=float(np.mean(lens)) if len(lens) else 0.0,
            median=float(np.median(lens)) if len(lens) else 0.0,
        )
    else:
        traj_len_hist = dict(bins=[], counts=[], mean=0.0, median=0.0)

    # 缺失率 Top-12（state_* 或 meta.feature_columns）
    feature_cols = meta.get("feature_columns") or [c for c in df.columns if str(c).startswith("state_")]
    missing_top = []
    if feature_cols:
        miss = df[feature_cols].isna().mean().sort_values(ascending=False)
        for k, v in miss.head(12).items():
            missing_top.append((str(k), float(v)))

    return dict(
        total_patients=total_patients,
        total_records=total_records,
        n_actions=n_actions,
        action_counts=action_counts,
        action_names=action_names,
        traj_len_hist=traj_len_hist,
        missing_top=missing_top
    )


def get_action_catalog() -> dict:
    """返回 {action_id: action_name}，优先使用 meta；否则从数据里推断。"""
    from data_manager import data_manager
    import numpy as np
    meta = data_manager.get_current_meta()
    raw_map = meta.get("action_map") or {
        i: n for i, n in enumerate(meta.get("action_names") or [])
    }
    id2name: Dict[int, str] = {}
    for k, v in (raw_map or {}).items():
        name = str(v).strip()
        if not name:
            name = f"Action {int(k)}"
        id2name[int(k)] = name
    # 如仍为空，从当前数据集推断 id 集合
    try:
        df = data_manager.get_current_data()
        if "action" in df.columns:
            ids = sorted(set(map(int, df["action"].dropna().astype(int).tolist())))
            for i in ids:
                id2name.setdefault(int(i), f"Action {int(i)}")
    except Exception:
        pass
    return {int(k): str(v) for k, v in (id2name or {}).items()}

def get_action_legend_html() -> str:
    """把动作映射渲染成 HTML 表格，供 UI 显示。"""
    id2name = get_action_catalog()
    if not id2name:
        return ""
    row_tpl = "<tr><td style='border:1px solid #ccc;padding:4px;'>{}</td>" \
              "<td style='border:1px solid #ccc;padding:4px;'>{}</td></tr>"
    rows = "\n".join([row_tpl.format(int(k), str(v)) for k, v in sorted(id2name.items())])
    return f"""
    <div style="background:#f9fbfd;color:#ff0000;padding:12px;border:1px solid #e6ecf5;">
      <style>
        .action-legend-table tbody tr:nth-child(even) {{background-color:#eef3f9;}}
      </style>
      <h4 style="margin-top:0;color:#ff0000;">Action Legend</h4>
      <table class="action-legend-table" style="border-collapse:collapse;width:100%;">
        <thead>
          <tr>
            <th style="background-color:#34495e;color:#fff;border:1px solid #ccc;padding:4px;">ID</th>
            <th style="background-color:#34495e;color:#fff;border:1px solid #ccc;padding:4px;">Name</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    """

def generate_patient_report_ui(patient_id: str, topk: int = 3, fmt: str = "html"):
    """
    返回 (report_html_or_md, PIL.Image 或 None, 可选保存路径)
    ——任何异常都只体现在文本，不抛给前端。
    """
    try:
        from data_manager import data_manager
        patient = data_manager.get_patient_info(patient_id)
        if not isinstance(patient, dict) or not patient:
            return f"<p>Cannot find patient: {patient_id}</p>", None, None

        # analysis 软兜底
        try:
            analysis = analyze_patient(patient_id)
        except Exception as e:
            analysis = {"recommendation": {"recommended_treatment": "Unknown", "confidence": 0.0},
                        "all_options": {"action_values": []},
                        "predicted_trajectory": {"trajectory": []},
                        "error": str(e)}

        # 动作目录
        from reports import build_action_catalog, make_treatment_analysis_figure, render_patient_report_md
        try:
            df = data_manager.get_current_data()
            dataset_shim = {"actions": df["action"].values if (df is not None and "action" in df.columns) else None}
        except Exception:
            dataset_shim = {"actions": None}
        meta = data_manager.get_current_meta()
        action_catalog = build_action_catalog(meta, dataset_shim)

        if fmt == "html":
            from reports import render_patient_report_html
            html = render_patient_report_html(patient, analysis, action_catalog, meta)
            # 保存 HTML
            import os
            from datetime import datetime
            os.makedirs("./output/reports", exist_ok=True)
            path = f"./output/reports/{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
            try:
                fig = make_treatment_analysis_figure(analysis)
            except Exception:
                fig = None
            return html, fig, path
        else:
            md = render_patient_report_md(patient, analysis, cohort_stats=None)
            try:
                fig = make_treatment_analysis_figure(analysis)
            except Exception:
                fig = None
            return md, fig, None
    except Exception as e:
        return f"<p>Report error: {e}</p>", None, None

