"""
inference.py - Inference and deployment utilities

This module provides tools for using the trained digital twin system
to make treatment recommendations for new patients.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
# --- BCQ support (optional) ---
def _bcq_predict_action(algo, state_np):
    import numpy as np
    x = np.asarray(state_np, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    try:
        out = algo.predict(x)
    except Exception:
        # some versions use .predict_value / .predict_best_action
        out = getattr(algo, 'predict_best_action', algo.predict)(x)
    return int(out[0] if hasattr(out, '__len__') else out)

import time
from collections import defaultdict
from typing import Dict, List, Union
import warnings
warnings.filterwarnings("ignore")
import os
import sys
from collections import defaultdict

from models import (
    TransformerDynamicsModel,
    TreatmentOutcomeModel,
    ConservativeQNetwork,
    EnsembleQNetwork
)


class DigitalTwinInference:
    """Main inference class for the digital twin system"""
    
    def __init__(self,
                 dynamics_model: TransformerDynamicsModel,
                 outcome_model: TreatmentOutcomeModel,
                 q_network: ConservativeQNetwork,
                 state_dim: int,
                 action_dim: int,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.dynamics_model = dynamics_model.to(device)
        self.outcome_model = outcome_model.to(device)
        self.q_network = q_network.to(device)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Set models to eval mode
        self.dynamics_model.eval()
        self.outcome_model.eval()
        self.q_network.eval()
        
        # Optional BCQ policy (d3rlpy)
        self.bcq_policy = None

        # Action names for interpretability
        self.action_names = [
            'Medication A',
            'Medication B',
            'Medication C',
            'Placebo',
            'Combination Therapy'
        ]

        # Feature names
        self.feature_names = [
            'Age', 'Gender', 'Blood Pressure', 'Heart Rate',
            'Glucose Level', 'Creatinine', 'Hemoglobin',
            'Temperature', 'Oxygen Saturation', 'BMI'
        ][:state_dim]

        # Optional metadata defaults
        self.critical_rules = []
        self.spo2_idx = None
        self.spo2_threshold = 0.80
        self.norm = {"method": "none"}
        self.meta = {}

        # Set model_dir to a default value or allow it to be passed as an argument
        self.model_dir = getattr(self, 'model_dir', '/home/xqin5/RL_DT_MTE_Finalversion/output/models')
        meta_path = os.path.join(self.model_dir, "dataset_meta.json")
        if os.path.exists(meta_path):
            import json
            with open(meta_path, "r") as f:
                self.meta = json.load(f)
        self.critical_rules = self.meta.get("critical_features", self.critical_rules)
        self.feature_names = self.meta.get("feature_names", self.feature_names)
        self.action_names  = self.meta.get("action_names", self.action_names)
        self.spo2_idx      = self.meta.get("spo2_idx", self.spo2_idx)
        self.spo2_threshold= float(self.meta.get("spo2_threshold", self.spo2_threshold))

        # 反归一化/归一化（如需）
        self.norm = self.meta.get("norm", self.norm)
            
    def recommend_treatment(self, 
                          patient_state: np.ndarray,
                          return_all_scores: bool = False) -> Dict:
        """
        Recommend optimal treatment for a patient
        
        Args:
            patient_state: Current patient state vector
            return_all_scores: Whether to return Q-values for all actions
            
        Returns:
            Dictionary with recommended action and additional info
        """
        with torch.no_grad():
            # Convert to tensor
            state_tensor = torch.FloatTensor(patient_state).unsqueeze(0).to(self.device)

            # If BCQ policy is provided, use it to choose action
            if getattr(self, 'bcq_policy', None) is not None:
                import numpy as np
                try:
                    optimal_action = _bcq_predict_action(self.bcq_policy, patient_state)
                except Exception as e:
                    optimal_action = None
                outcomes = []
                for action in range(self.action_dim):
                    action_tensor = torch.tensor([action], device=self.device)
                    outcome = self.outcome_model(state_tensor, action_tensor).item()
                    outcomes.append(outcome)
                if optimal_action is None:
                    optimal_action = int(np.argmax(outcomes))
                result = {
                    'recommended_action': optimal_action,
                    'recommended_treatment': self.action_names[optimal_action],
                    'confidence': float(outcomes[optimal_action] - float(np.mean(outcomes))),
                    'expected_immediate_outcome': outcomes[optimal_action],
                    'policy': 'BCQ'
                }
                if return_all_scores:
                    result['all_q_values'] = outcomes  # for plotting
                    result['all_outcomes'] = outcomes
                    result['treatment_rankings'] = sorted(
                        enumerate(outcomes), key=lambda x: x[1], reverse=True
                    )
                return result
            
            # Get Q-values for all actions
            q_values = self.q_network(state_tensor).squeeze().cpu().numpy()
            
            # Get optimal action
            optimal_action = int(np.argmax(q_values))
            
            # Compute expected outcomes for all actions
            outcomes = []
            for action in range(self.action_dim):
                action_tensor = torch.tensor([action], device=self.device)
                outcome = self.outcome_model(state_tensor, action_tensor).item()
                outcomes.append(outcome)
            
            result = {
                'recommended_action': optimal_action,
                'recommended_treatment': self.action_names[optimal_action],
                'confidence': float(q_values[optimal_action] - np.mean(q_values)),
                'expected_immediate_outcome': outcomes[optimal_action]
            }
            
            if return_all_scores:
                result['all_q_values'] = q_values.tolist()
                result['all_outcomes'] = outcomes
                result['treatment_rankings'] = sorted(
                    enumerate(q_values), key=lambda x: x[1], reverse=True
                )
            
            return result
    
    def simulate_treatment_trajectory(self,
                                      initial_state: np.ndarray,
                                      treatment_plan: List[int],
                                      max_steps: Optional[int] = None) -> Dict:
        """
        Simulate patient trajectory under a specific treatment plan.
        
        Args:
            initial_state: Initial patient state.
            treatment_plan: List of actions to take.
            max_steps: Maximum simulation steps.
            
        Returns:
            Dictionary with trajectory information.
        """
        if max_steps is None:
            max_steps = len(treatment_plan)
        
        with torch.no_grad():
            # Correctly initialize master history lists
            state_history_tensors = [torch.FloatTensor(initial_state).to(self.device)]
            action_history_list = []
            
            trajectory = {
                'states': [initial_state],
                'actions': [],
                'rewards': [],
                'cumulative_reward': 0.0
            }
            
            for t in range(min(max_steps, len(treatment_plan))):
                current_state = state_history_tensors[-1]
                action = treatment_plan[t]
                action_tensor = torch.tensor([action], device=self.device)
                
                # Predict immediate outcome using the current state
                reward = self.outcome_model(current_state.unsqueeze(0), action_tensor).item()
                
                # Append the current action to the action history
                action_history_list.append(action)
                
                # The history for prediction includes all states and actions up to the current time t
                predict_states_hist = state_history_tensors
                predict_actions_hist = action_history_list

                # Keep history to a manageable size for the model input
                if len(predict_states_hist) > 10:
                    predict_states_hist = predict_states_hist[-10:]
                    predict_actions_hist = predict_actions_hist[-10:]

                # Prepare sequences for the model. Their lengths will now always match.
                states_seq = torch.stack(predict_states_hist).unsqueeze(0)
                actions_seq = torch.tensor(predict_actions_hist, device=self.device).unsqueeze(0)
                
                # Predict next state using the full, correctly aligned history
                next_state = self.dynamics_model.predict_next_state(
                    states_seq, actions_seq
                ).squeeze(0)
                
                # Update master state history for the next iteration
                state_history_tensors.append(next_state)
                
                # Store results in the trajectory log
                trajectory['states'].append(next_state.cpu().numpy())
                trajectory['actions'].append(action)
                trajectory['rewards'].append(reward)
                trajectory['cumulative_reward'] += reward * (0.99 ** t)
                
                # Check termination conditions using the newly predicted state
                # === NEW: 规则式终止条件 ===
                def _violates(rule, value):
                    op = rule["op"]
                    thr = rule["threshold"]
                    if op == "<":  return value <  thr
                    if op == "<=": return value <= thr
                    if op == ">":  return value >  thr
                    if op == ">=": return value >= thr
                    raise ValueError(f"未知比较符: {op}")
                
                stop_loop = False
                for rule in self.critical_rules:
                    if not rule.get("as_terminal", False): 
                        continue
                    idx = rule["index"]
                    if idx < len(next_state) and _violates(rule, float(next_state[idx])):
                        print(f"Early stop at step {t+1} due to rule: {rule['display_name']} {rule['op']} {rule['threshold']}")
                        stop_loop = True
                        break
                if stop_loop:
                    break
            
            return trajectory
    
    def compare_treatment_plans(self,
                              patient_state: np.ndarray,
                              treatment_plans: Dict[str, List[int]],
                              horizon: int = 20) -> pd.DataFrame:
        """
        Compare multiple treatment plans for a patient.
        
        Args:
            patient_state: Initial patient state.
            treatment_plans: Dictionary mapping plan names to action sequences.
            horizon: Simulation horizon.
            
        Returns:
            DataFrame with comparison results.
        """
        results = []
        
        for plan_name, plan in treatment_plans.items():
            # Extend plan to horizon if needed
            if len(plan) < horizon:
                plan = plan + [plan[-1]] * (horizon - len(plan))
            
            # Simulate trajectory
            trajectory = self.simulate_treatment_trajectory(
                patient_state, plan, horizon
            )
            
            # Compute metrics
            final_state = trajectory['states'][-1]
            row = {
                'Treatment Plan': plan_name,
                'Cumulative Reward': trajectory['cumulative_reward'],
                'Avg Immediate Reward': float(np.mean(trajectory['rewards'])),
                'Final Health Score': self._compute_health_score(final_state),
            }
            for rule in self.critical_rules:
                i = rule["index"]
                if i < len(final_state):
                    row[f"Final {rule['display_name']}"] = float(final_state[i])
            results.append(row)
        
        
        return pd.DataFrame(results).sort_values('Cumulative Reward', ascending=False)

    def _compute_health_score(self, state: np.ndarray) -> float:
        # 基准项：让状态尽量居中（归一化后0~1），避免极值；也可按需改
        center = 0.5
        base = -float(np.mean(np.abs(state - center)))
        # 规则奖励：满足“健康方向”的规则加权加分，否则减分
        # 这里把规则本身视作“健康方向”，即满足 rule.op threshold 视为正向
        bonus = 0.0
        for rule in self.critical_rules:
            idx = rule["index"]
            if idx >= len(state): 
                continue
            val = float(state[idx])
            op  = rule["op"]; thr = rule["threshold"]; w = rule["weight"]
            ok = (op == "<" and val < thr) or (op == "<=" and val <= thr) \
                or (op == ">" and val > thr) or (op == ">=" and val >= thr)
            bonus += (w if ok else -w)
        return base + bonus

    
    def explain_recommendation(self,
                             patient_state: np.ndarray,
                             n_simulations: int = 100) -> Dict:
        """
        Provide detailed explanation for treatment recommendation
        
        Args:
            patient_state: Current patient state
            n_simulations: Number of simulations to run
            
        Returns:
            Dictionary with detailed explanation
        """
        recommendation = self.recommend_treatment(patient_state, return_all_scores=True)
        
        # Identify key factors influencing the decision
        state_dict = {self.feature_names[i]: patient_state[i] 
                     for i in range(len(self.feature_names))}
        
        # Identify abnormal values
        abnormal_features = []
        for i, feature in enumerate(self.feature_names):
            if i == 1:  # Skip gender
                continue
            if abs(patient_state[i] - 0.5) > 0.2:  # Significant deviation
                abnormal_features.append({
                    'feature': feature,
                    'value': patient_state[i],
                    'status': 'High' if patient_state[i] > 0.5 else 'Low'
                })
        
        # Simulate counterfactual outcomes
        counterfactuals = {}
        for action in range(self.action_dim):
            outcomes = []
            for _ in range(n_simulations):
                # Add small noise to initial state
                noisy_state = patient_state + np.random.normal(0, 0.01, size=patient_state.shape)
                noisy_state = np.clip(noisy_state, 0, 1)
                
                trajectory = self.simulate_treatment_trajectory(
                    noisy_state, [action] * 10, max_steps=10
                )
                outcomes.append(trajectory['cumulative_reward'])
            
            counterfactuals[self.action_names[action]] = {
                'mean_outcome': np.mean(outcomes),
                'std_outcome': np.std(outcomes),
                'confidence_interval': (np.percentile(outcomes, 5), 
                                      np.percentile(outcomes, 95))
            }
        
        explanation = {
            'recommendation': recommendation,
            'patient_profile': state_dict,
            'abnormal_features': abnormal_features,
            'counterfactual_outcomes': counterfactuals,
            'explanation_text': self._generate_explanation_text(
                recommendation, abnormal_features, counterfactuals
            )
        }
        
        return explanation
    
    def _generate_explanation_text(self, 
                                 recommendation: Dict,
                                 abnormal_features: List[Dict],
                                 counterfactuals: Dict) -> str:
        """Generate human-readable explanation"""
        
        text = f"Treatment Recommendation: {recommendation['recommended_treatment']}\n\n"
        
        text += "Patient Assessment:\n"
        if abnormal_features:
            for feature in abnormal_features:
                text += f"- {feature['feature']}: {feature['status']} ({feature['value']:.3f})\n"
        else:
            text += "- All vitals within normal range\n"
        
        text += f"\nExpected Outcomes:\n"
        rec_treatment = recommendation['recommended_treatment']
        rec_outcome = counterfactuals[rec_treatment]
        
        text += f"- Recommended treatment ({rec_treatment}): "
        text += f"Expected outcome {rec_outcome['mean_outcome']:.3f} "
        text += f"(95% CI: {rec_outcome['confidence_interval'][0]:.3f} - "
        text += f"{rec_outcome['confidence_interval'][1]:.3f})\n"
        
        # Compare with alternatives
        text += "\nAlternative treatments:\n"
        sorted_treatments = sorted(
            counterfactuals.items(), 
            key=lambda x: x[1]['mean_outcome'], 
            reverse=True
        )
        
        for treatment, outcome in sorted_treatments[1:3]:  # Top 2 alternatives
            diff = rec_outcome['mean_outcome'] - outcome['mean_outcome']
            text += f"- {treatment}: {outcome['mean_outcome']:.3f} "
            text += f"(difference: {diff:+.3f})\n"
        
        return text
    
    def visualize_treatment_trajectory(self,
                                     trajectory: Dict,
                                     save_path: Optional[str] = None):
        """Visualize a treatment trajectory"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot key vitals over time
        states_array = np.array(trajectory['states'])
        timesteps = range(len(trajectory['states']))
        
        # Blood pressure
        axes[0, 0].plot(timesteps, states_array[:, 2], 'b-', linewidth=2)
        axes[0, 0].axhline(y=0.5, color='g', linestyle='--', alpha=0.5)
        axes[0, 0].set_title('Blood Pressure')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Normalized BP')
        
        # Glucose level
        axes[0, 1].plot(timesteps, states_array[:, 4], 'r-', linewidth=2)
        axes[0, 1].axhline(y=0.5, color='g', linestyle='--', alpha=0.5)
        axes[0, 1].set_title('Glucose Level')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Normalized Glucose')
        
        # Oxygen saturation
        axes[1, 0].plot(timesteps, states_array[:, 8], 'g-', linewidth=2)
        axes[1, 0].axhline(y=0.95, color='g', linestyle='--', alpha=0.5)
        axes[1, 0].axhline(y=0.8, color='r', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('Oxygen Saturation')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('O2 Saturation')
        
        # Treatment actions
        if trajectory['actions']:
            action_names_short = ['A', 'B', 'C', 'P', 'Combo']
            action_labels = [action_names_short[a] for a in trajectory['actions']]
            axes[1, 1].bar(range(len(trajectory['actions'])), 
                          trajectory['actions'], 
                          color='purple', alpha=0.7)
            axes[1, 1].set_title('Treatment Sequence')
            axes[1, 1].set_xlabel('Time Step')
            axes[1, 1].set_ylabel('Treatment')
            axes[1, 1].set_yticks(range(5))
            axes[1, 1].set_yticklabels(action_names_short)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()

class SafeDigitalTwinInference(DigitalTwinInference):
    """
    An enhanced inference engine that wraps the original engine with a safety-checking layer.
    """
    def __init__(self, *args, **kwargs):
        # First, initialize the parent class (DigitalTwinInference) completely
        super().__init__(*args, **kwargs)
        # Then, add the new safety checker component
        self.safety_checker = ClinicalSafetyChecker()
            
    def recommend_treatment(self, 
                            patient_state: np.ndarray,
                            return_all_scores: bool = False,
                            safety_check: bool = True) -> Dict:
        """
        Provides a treatment recommendation that includes a mandatory safety check.
        """
        # 1. Get the standard recommendation from the parent class
        result = super().recommend_treatment(patient_state, return_all_scores)
        
        # 2. If safety checks are enabled, validate the recommendation
        if safety_check:
            safety_result = self.safety_checker.validate_treatment_recommendation(
                patient_state, result['recommended_treatment']
            )
            result['safety_check'] = safety_result
                        
            # 3. If the recommended treatment is not approved, override it
            if not safety_result['approved']:
                print(f"WARNING: Safety override! Original recommendation '{result['recommended_treatment']}' rejected. "
                      f"Reason: {safety_result['reason']}. Using alternative '{safety_result['alternative']}'.")
                
                # Store original recommendation for logging and switch to the safe alternative
                result['original_recommendation'] = result['recommended_treatment']
                result['recommended_treatment'] = safety_result['alternative']
                result['safety_override'] = True
                
                # Optional: Re-calculate confidence and outcome for the new recommendation
                alt_action_idx = self.action_names.index(safety_result['alternative'])
                if 'all_q_values' in result:
                    result['confidence'] = float(result['all_q_values'][alt_action_idx] - np.mean(result['all_q_values']))
                if 'all_outcomes' in result:
                    result['expected_immediate_outcome'] = result['all_outcomes'][alt_action_idx]

        return result

class OptimizedDigitalTwinInference(DigitalTwinInference):
    """优化的推理引擎，专注于低延迟"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 预热模型
        self._warmup_models()
        
        # 推理缓存
        self.inference_cache = {}
        self.cache_size = 1000
        
        # 模型量化（如果支持）
        self._try_quantization()
    
    def _warmup_models(self):
        """预热模型以减少首次推理延迟"""
        print("🔥 Warming up models for optimal performance...")
        
        dummy_state = torch.randn(1, self.state_dim).to(self.device)
        dummy_action = torch.randint(0, self.action_dim, (1,)).to(self.device)
        
        # 预热Q网络
        with torch.no_grad():
            for _ in range(10):
                _ = self.q_network(dummy_state)
        
        # 预热outcome模型
        with torch.no_grad():
            for _ in range(10):
                _ = self.outcome_model(dummy_state, dummy_action)
        
        print("✅ Model warmup complete")
    
    def _try_quantization(self):
        """尝试模型量化以提升推理速度"""
        try:
            if self.device == 'cpu':
                # CPU量化
                self.q_network = torch.quantization.quantize_dynamic(
                    self.q_network, {torch.nn.Linear}, dtype=torch.qint8
                )
                print("✅ Applied dynamic quantization for CPU inference")
        except Exception as e:
            print(f"⚠️ Quantization not applied: {e}")
    
    def _get_cache_key(self, state: np.ndarray) -> str:
        """生成缓存键"""
        # 使用状态的哈希作为缓存键（四舍五入以增加命中率）
        rounded_state = np.round(state, decimals=2)
        return hash(rounded_state.tobytes())
    
    def recommend_treatment(self, 
                          patient_state: np.ndarray,
                          return_all_scores: bool = False,
                          use_cache: bool = True) -> Dict:
        """优化的治疗推荐"""
        
        # 检查缓存
        if use_cache:
            cache_key = self._get_cache_key(patient_state)
            if cache_key in self.inference_cache:
                cached_result = self.inference_cache[cache_key]
                cached_result['from_cache'] = True
                return cached_result
        
        # 执行推理
        start_time = time.time()
        result = super().recommend_treatment(patient_state, return_all_scores)
        inference_time = time.time() - start_time
        
        result['inference_time_ms'] = inference_time * 1000
        result['from_cache'] = False
        
        # 更新缓存
        if use_cache and len(self.inference_cache) < self.cache_size:
            cache_key = self._get_cache_key(patient_state)
            self.inference_cache[cache_key] = result.copy()
        
        return result
    
    def batch_recommend(self, patient_states: List[np.ndarray]) -> List[Dict]:
        """批量推理以提升吞吐量"""
        if not patient_states:
            return []
        
        # 批处理推理
        with torch.no_grad():
            state_tensors = torch.stack([
                torch.FloatTensor(state) for state in patient_states
            ]).to(self.device)
            
            # 批量Q值计算
            q_values_batch = self.q_network(state_tensors)
            
            results = []
            for i, q_values in enumerate(q_values_batch):
                optimal_action = int(q_values.argmax())
                confidence = float(q_values[optimal_action] - q_values.mean())
                
                results.append({
                    'recommended_action': optimal_action,
                    'recommended_treatment': self.action_names[optimal_action],
                    'confidence': confidence
                })
            
        return results

class ClinicalSafetyChecker:
    """临床安全性检查器"""
    
    def __init__(self):
        # 定义临床约束
        self.vital_sign_ranges = {
            'blood_pressure': (0.3, 0.8),    # 正常范围
            'heart_rate': (0.4, 0.7),
            'glucose': (0.3, 0.7),
            'oxygen_saturation': (0.85, 1.0),  # 最关键
            'temperature': (0.45, 0.55)
        }
        
        self.contraindications = {
            # 药物禁忌症
            'Medication A': lambda state: state[4] > 0.9,  # 高血糖禁用A
            'Medication B': lambda state: state[2] > 0.8,  # 高血压禁用B
            'Combination Therapy': lambda state: state[0] > 0.8 or state[8] < 0.8  # 老年或低氧禁用组合
        }
        
    def check_patient_safety(self, state: np.ndarray) -> Dict:
        """检查患者状态安全性"""
        warnings = []
        
        # 检查生命体征
        if len(state) >= 9:  # 确保有足够的特征
            if state[8] < 0.8:  # 氧饱和度
                warnings.append("CRITICAL: Low oxygen saturation (<80%)")
            if state[2] > 0.85:  # 血压
                warnings.append("WARNING: High blood pressure")
            if state[4] > 0.85:  # 血糖
                warnings.append("WARNING: High glucose level")
        
        return {
            'safe': len(warnings) == 0,
            'warnings': warnings,
            'critical': any('CRITICAL' in w for w in warnings)
        }
    
    def validate_treatment_recommendation(self, state: np.ndarray, 
                                        recommended_action: str) -> Dict:
        """验证治疗建议的安全性"""
        safety_check = self.check_patient_safety(state)
        
        # 检查禁忌症
        contraindication_check = False
        if recommended_action in self.contraindications:
            contraindication_check = self.contraindications[recommended_action](state)
        
        # 如果是危重状态，只推荐保守治疗
        if safety_check['critical'] and recommended_action != 'Placebo':
            return {
                'approved': False,
                'reason': 'Critical patient condition requires conservative treatment',
                'alternative': 'Placebo',
                'safety_warnings': safety_check['warnings']
            }
        
        # 检查禁忌症
        if contraindication_check:
            return {
                'approved': False,
                'reason': f'Contraindication detected for {recommended_action}',
                'alternative': 'Placebo',
                'safety_warnings': safety_check['warnings']
            }
        
        return {
            'approved': True,
            'safety_warnings': safety_check['warnings']
        }

class ClinicalDecisionSupport:
    """High-level clinical decision support interface"""
    
    def __init__(self, inference_engine: DigitalTwinInference):
        self.engine = inference_engine
        self.treatment_history = []
    
    def create_patient_report(self, 
                            patient_data: Dict,
                            output_path: str = 'patient_report.html') -> str:
        """
        Create comprehensive patient report
        
        Args:
            patient_data: Dictionary with patient information
            output_path: Path to save HTML report
            
        Returns:
            HTML content of the report
        """
        # Extract patient state
        state = self._extract_patient_state(patient_data)
        
        # Get recommendation and explanation
        explanation = self.engine.explain_recommendation(state)
        
        # Compare treatment options
        treatment_plans = {
            'Recommended': [explanation['recommendation']['recommended_action']] * 20,
            'Conservative': [3] * 20,  # Placebo
            'Aggressive': [4] * 20,  # Combination
            'Alternating': [0, 1] * 10  # Alternate between A and B
        }
        
        comparison = self.engine.compare_treatment_plans(state, treatment_plans)
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Patient Treatment Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2 {{ color: #2c3e50; }}
                .recommendation {{ 
                    background-color: #e8f4f8; 
                    padding: 20px; 
                    border-radius: 10px;
                    margin: 20px 0;
                }}
                table {{ 
                    border-collapse: collapse; 
                    width: 100%; 
                    margin: 20px 0;
                }}
                th, td {{ 
                    border: 1px solid #ddd; 
                    padding: 12px; 
                    text-align: left; 
                }}
                th {{ background-color: #34495e; color: white; }}
                .warning {{ color: #e74c3c; font-weight: bold; }}
                .success {{ color: #27ae60; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>Digital Twin Treatment Recommendation Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Patient Information</h2>
            <p><strong>ID:</strong> {patient_data.get('patient_id', 'Unknown')}</p>
            <p><strong>Age:</strong> {int(state[0] * 72 + 18)} years</p>
            <p><strong>Gender:</strong> {'Male' if state[1] == 1 else 'Female'}</p>
            
            <h2>Current Vitals</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Status</th>
                </tr>
        """
        
        # Add vital signs to table
        vital_indices = [2, 3, 4, 5, 6, 8]
        vital_names = ['Blood Pressure', 'Heart Rate', 'Glucose', 
                      'Creatinine', 'Hemoglobin', 'O2 Saturation']
        
        for idx, name in zip(vital_indices, vital_names):
            value = state[idx]
            if abs(value - 0.5) > 0.2 and idx != 8:
                status = '<span class="warning">Abnormal</span>'
            elif idx == 8 and value < 0.9:
                status = '<span class="warning">Low</span>'
            else:
                status = '<span class="success">Normal</span>'
            
            html_content += f"""
                <tr>
                    <td>{name}</td>
                    <td>{value:.3f}</td>
                    <td>{status}</td>
                </tr>
            """
        
        html_content += f"""
            </table>
            
            <div class="recommendation">
                <h2>Treatment Recommendation</h2>
                <p><strong>Recommended Treatment:</strong> {explanation['recommendation']['recommended_treatment']}</p>
                <p><strong>Confidence Level:</strong> {explanation['recommendation']['confidence']:.3f}</p>
                <p><strong>Expected Immediate Outcome:</strong> {explanation['recommendation']['expected_immediate_outcome']:.3f}</p>
            </div>
            
            <h2>Treatment Comparison</h2>
            {comparison.to_html(index=False, escape=False)}
            
            <h2>Detailed Explanation</h2>
            <pre>{explanation['explanation_text']}</pre>
            
            <h2>Disclaimer</h2>
            <p><em>This report is generated by an AI system and should be used for decision support only. 
            Always consult with qualified healthcare professionals before making treatment decisions.</em></p>
        </body>
        </html>
        """
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return html_content
    def initialize_tools(inference_engine: DigitalTwinInference, cds: 'ClinicalDecisionSupport'):
        """使用优化的推理引擎"""
        global _inference_engine, _cds, _online_system
        
        # 创建优化的推理引擎
        _inference_engine = OptimizedDigitalTwinInference(
            inference_engine.dynamics_model,
            inference_engine.outcome_model, 
            inference_engine.q_network,
            inference_engine.state_dim,
            inference_engine.action_dim,
            inference_engine.device
        )
        
        _cds = cds
        _online_system = ClinicalDecisionSupport(_inference_engine)    

    def _extract_patient_state(self, patient_data: Dict) -> np.ndarray:
        """Extract state vector from patient data dictionary"""
        # This would be customized based on your data format
        state = np.zeros(self.engine.state_dim)
        
        # Example mapping (adjust based on your needs)
        state[0] = patient_data.get('age', 45) / 90  # Normalize age
        state[1] = patient_data.get('gender', 0)
        state[2] = patient_data.get('blood_pressure', 0.5)
        state[3] = patient_data.get('heart_rate', 0.5)
        state[4] = patient_data.get('glucose', 0.5)
        state[5] = patient_data.get('creatinine', 0.5)
        state[6] = patient_data.get('hemoglobin', 0.6)
        state[7] = patient_data.get('temperature', 0.5)
        state[8] = patient_data.get('oxygen_saturation', 0.95)
        state[9] = patient_data.get('bmi', 0.5) if self.engine.state_dim > 9 else 0
        
        return np.clip(state, 0, 1)


# inference.py

def load_trained_models(state_dim: int, 
                        action_dim: int,
                        device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> DigitalTwinInference:
    """Load all trained models and create inference engine"""
    
    # Initialize models
    dynamics_model = TransformerDynamicsModel(state_dim, action_dim)
    outcome_model  = TreatmentOutcomeModel(state_dim, action_dim)
    q_network      = ConservativeQNetwork(state_dim, action_dim)
    
    base_dir = '/home/xqin5/RL_DT_MTE_Finalversion/output/models'
    
    dynamics_model.load_state_dict(
        torch.load(f'{base_dir}/best_dynamics_model.pth', map_location=device)
    )
    outcome_model.load_state_dict(
        torch.load(f'{base_dir}/best_outcome_model.pth', map_location=device)
    )
    q_network.load_state_dict(
        torch.load(f'{base_dir}/best_q_network.pth', map_location=device)
    )
    
    # Create inference engine
    inference_engine = DigitalTwinInference(
        dynamics_model, outcome_model, q_network,
        state_dim, action_dim, device
    )
    
    return inference_engine



if __name__ == "__main__":
    # Example usage
    print("Loading trained models...")
    
    # Assume models are already trained
    state_dim = 10
    action_dim = 5
    
    # For demonstration, create dummy models
    dynamics_model = TransformerDynamicsModel(state_dim, action_dim)
    outcome_model = TreatmentOutcomeModel(state_dim, action_dim)
    q_network = ConservativeQNetwork(state_dim, action_dim)
    
    # Create inference engine
    inference_engine = DigitalTwinInference(
        dynamics_model, outcome_model, q_network,
        state_dim, action_dim
    )
    
    # Example patient
    patient_state = np.array([0.6, 1, 0.7, 0.65, 0.75, 0.6, 0.45, 0.5, 0.88, 0.55])
    
    print("\nGetting treatment recommendation...")
    recommendation = inference_engine.recommend_treatment(patient_state, return_all_scores=True)
    
    print(f"\nRecommended Treatment: {recommendation['recommended_treatment']}")
    print(f"Confidence: {recommendation['confidence']:.3f}")
    print(f"Expected Outcome: {recommendation['expected_immediate_outcome']:.3f}")
    
    print("\nTreatment Rankings:")
    for rank, (action, score) in enumerate(recommendation['treatment_rankings']):
        print(f"{rank+1}. {inference_engine.action_names[action]}: {score:.3f}")
    
    # Get detailed explanation
    print("\nGenerating detailed explanation...")
    explanation = inference_engine.explain_recommendation(patient_state, n_simulations=20)
    print(explanation['explanation_text'])
    
    # Create clinical decision support
    cds = ClinicalDecisionSupport(inference_engine)
    
    # Generate patient report
    patient_data = {
        'patient_id': 'P12345',
        'age': 55,
        'gender': 1,
        'blood_pressure': 0.7,
        'heart_rate': 0.65,
        'glucose': 0.75,
        'creatinine': 0.6,
        'hemoglobin': 0.45,
        'temperature': 0.5,
        'oxygen_saturation': 0.88,
        'bmi': 0.55
    }
    
    print("\nGenerating patient report...")
    report_path = 'patient_report_demo.html'
    cds.create_patient_report(patient_data, report_path)
    print(f"Report saved to: {report_path}")