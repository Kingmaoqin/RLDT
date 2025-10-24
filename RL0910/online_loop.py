"""
online_loop.py - Online incremental training loop with active learning
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple
from collections import deque
import time
import os
from datetime import datetime
import threading
import queue
import json
from threading import Lock
from typing import Callable
from datetime import datetime
import random
from models import TransformerDynamicsModel, TreatmentOutcomeModel, ConservativeQNetwork, EnsembleQNetwork
from training import ConservativeQLearning, DigitalTwinTrainer, OutcomeModelTrainer
from samplers import StreamActiveLearner
try:
    from d3rlpy import load_learnable
    from d3rlpy.dataset import MDPDataset
    import d3rlpy
    BCQ_AVAILABLE = True
except ImportError:
    BCQ_AVAILABLE = False
    print("Warning: d3rlpy not available, falling back to CQL")


# class BCQOnlineTrainer:
#     """BCQ的真正在线训练实现"""
#     def __init__(self, bcq_policy_path: str, device: str = 'cuda'):
#         if not BCQ_AVAILABLE:
#             raise ImportError("d3rlpy not available for BCQ")

#         # 为了避免在两个加载路径中重复初始化代码，定义一个内部辅助函数
#         def _post_load_initialization(self):
#             # 记录离线 BCQ 模型的动作空间大小
#             self._bcq_built = False
#             self.bcq_action_size = None
#             try:
#                 cfg = getattr(self.bcq_algo, "config", None)
#                 self.bcq_action_size = getattr(cfg, "action_size", None)
#                 if self.bcq_action_size is None and getattr(self.bcq_algo, "impl", None) is not None:
#                     self.bcq_action_size = getattr(self.bcq_algo.impl, "action_size", None)
#             except Exception:
#                 pass
#             if self.bcq_action_size is not None:
#                 print(f"[BCQ] Loaded policy expects action_size={self.bcq_action_size}")

#             # 保证训练线程可用的缓冲区属性
#             if not hasattr(self, "labeled_buffer"):
#                 self.labeled_buffer = []
#             self.online_buffer = self.labeled_buffer # 做成引用
            
#             # 互斥锁和计数器
#             if not hasattr(self, "_buffer_lock"):
#                 self._buffer_lock = threading.Lock()
#             self._transition_count = 0
#             self._manual_transition_count = False

#             # 指导①：补齐默认字段
#             # --- defaults for online loop bookkeeping ---
#             try:
#                 from drive_tools import CURRENT_HYPERPARAMS as _HP
#                 _default_bs = int(_HP.get("batch_size", 32))
#             except Exception:
#                 _default_bs = 32
#             self._batch_size = int(os.getenv("BCQ_BATCH_SIZE", str(_default_bs)))
            
#             # 每多少条新样本触发一次更新；可用环境变量覆盖
#             self._update_frequency = max(1, int(os.getenv("BCQ_UPDATE_FREQUENCY", "20")))

#             # 统计辅助
#             self._update_count = 0
#             self._last_update_count = 0
#             self.updates_done = 0
#             self.update_calls = 0
#             self._last_update_walltime = time.time()
#             self.training_losses = deque(maxlen=100)
            
#         # 加载预训练BCQ模型
#         try:
#             self.bcq_algo = load_learnable(bcq_policy_path)
#             print(f"✓ BCQ policy loaded successfully from {bcq_policy_path}")
#             _post_load_initialization(self)
#         except Exception as e:
#             print(f"BCQ loading failed with error: {e}")
#             print("Attempting alternative loading method...")
#             try:
#                 import torch
#                 checkpoint = torch.load(bcq_policy_path, map_location=device)
#                 if hasattr(checkpoint, 'predict'):
#                     self.bcq_algo = checkpoint
#                     print("✓ BCQ policy loaded via torch.load")
#                     _post_load_initialization(self)
#                 else:
#                     raise Exception("Checkpoint doesn't contain BCQ algorithm")
#             except Exception as e2:
#                 print(f"Alternative loading also failed: {e2}")
#                 raise ImportError(f"Cannot load BCQ policy: {e}")

#         self.device = device
#         print(f"✓ BCQ online trainer ready: update_freq={self.update_frequency}, batch_size={self.batch_size}")

#     # 指导②：增加属性（getter/setter）
#     @property
#     def batch_size(self) -> int:
#         return getattr(self, "_batch_size", 32)

#     @batch_size.setter
#     def batch_size(self, value):
#         try:
#             self._batch_size = max(1, int(value))
#         except Exception:
#             self._batch_size = 32

#     @property
#     def update_frequency(self) -> int:
#         return getattr(self, "_update_frequency", 20)

#     @update_frequency.setter
#     def update_frequency(self, value):
#         try:
#             self._update_frequency = max(1, int(value))
#         except Exception:
#             self._update_frequency = 20

#     @property
#     def update_count(self) -> int:
#         return getattr(self, "_update_count", 0)

#     @update_count.setter
#     def update_count(self, value):
#         try:
#             self._update_count = int(value)
#         except Exception:
#             pass # 忽略非法赋值

#     @property
#     def transition_count(self):
#         if getattr(self, "_manual_transition_count", False):
#             return self._transition_count
#         try:
#             return len(self.online_buffer)
#         except Exception:
#             return self._transition_count

#     @transition_count.setter
#     def transition_count(self, value):
#         try:
#             v = int(value)
#         except Exception:
#             v = 0
#         try:
#             with self._buffer_lock:
#                 self._transition_count = v
#                 self._manual_transition_count = True
#         except Exception:
#             self._transition_count = v
#             self._manual_transition_count = True

#     def add_transition(self, transition: Dict):
#         """添加transition到在线buffer"""
#         formatted_transition = {
#             'observation': np.asarray(transition['state'], dtype=np.float32),
#             'action': int(transition['action']),
#             'reward': float(transition['reward']),
#             'next_observation': np.asarray(transition['next_state'], dtype=np.float32),
#             'terminal': bool(transition.get('done', False))
#         }
#         self.online_buffer.append(formatted_transition)
#         self._transition_count += 1
        
#         if (self._transition_count % self.update_frequency == 0 and 
#             len(self.online_buffer) >= self.batch_size):
#             return self.perform_update()
        
#         return {'updated': False, 'buffer_size': len(self.online_buffer)}

#     def on_update(self, steps: int = 0):
#         """每次完成一次 bcq.fit 后调用，用于记录更新次数等。"""
#         self.update_calls += 1
#         self._last_update_walltime = time.time()

# def perform_update(self) -> Dict:
#     try:
#         with self._buffer_lock:
#             buffer_len = len(self.online_buffer)
#             if buffer_len < self.batch_size:
#                 return {'updated': False, 'reason': 'not enough samples'}

#             idx = np.random.choice(buffer_len, self.batch_size, replace=False)
#             batch_data = [self.online_buffer[i] for i in idx]

#         # --------- 组 batch 张量 ---------
#         observations = np.asarray([t['observation'] for t in batch_data], dtype=np.float32)
#         next_observations = np.asarray([t['next_observation'] for t in batch_data], dtype=np.float32)
#         actions_raw = np.asarray([int(t['action']) for t in batch_data], dtype=np.int64)
#         rewards = np.asarray([float(t['reward']) for t in batch_data], dtype=np.float32)
#         terminals = np.asarray([bool(t.get('terminal', t.get('done', False))) for t in batch_data], dtype=np.bool_)
#         timeouts = np.zeros_like(terminals, dtype=np.bool_)
#         if not terminals.any():
#             timeouts[-1] = True

#         # --------- 动作维度校准 ---------
#         if getattr(self, "bcq_action_size", None):
#             active_action_dim = int(self.bcq_action_size)
#         else:
#             active_action_dim = int(actions_raw.max() + 1) if actions_raw.size > 0 else 1

#         invalid = actions_raw >= active_action_dim
#         if invalid.any():
#             keep = ~invalid
#             if keep.sum() == 0:
#                 return {'updated': False, 'reason': 'all actions out of range'}
#             observations = observations[keep]
#             next_observations = next_observations[keep]
#             rewards = rewards[keep]
#             terminals = terminals[keep]
#             timeouts = timeouts[keep]
#             actions_raw = actions_raw[keep]

#         if actions_raw.size > 0:
#             actions_raw = np.clip(actions_raw, 0, active_action_dim - 1)
#         actions = actions_raw.reshape(-1, 1)

#         # --------- 构建 d3rlpy MDPDataset（向后兼容）---------
#         try:
#             dataset = MDPDataset(
#                 observations=observations,
#                 actions=actions,
#                 rewards=rewards,
#                 terminals=terminals,
#                 timeouts=timeouts,
#                 next_observations=next_observations,
#                 discrete_action=True,
#                 action_size=active_action_dim,
#             )
#         except TypeError:
#             try:
#                 dataset = MDPDataset(
#                     observations=observations,
#                     actions=actions,
#                     rewards=rewards,
#                     terminals=terminals,
#                     timeouts=timeouts,
#                     next_observations=next_observations,
#                 )
#             except TypeError:
#                 dataset = MDPDataset(
#                     observations=observations,
#                     actions=actions,
#                     rewards=rewards,
#                     terminals=(terminals | timeouts),
#                 )

#         # --------- 首次/不一致时强制 build ---------
#         need_build = not getattr(self, "_bcq_built", False)
#         try:
#             impl = getattr(self.bcq_algo, "impl", None) or getattr(self.bcq_algo, "_impl", None)
#             impl_action_size = getattr(impl, "action_size", None) if impl is not None else None
#             if impl_action_size is not None and int(impl_action_size) != active_action_dim:
#                 need_build = True
#         except Exception:
#             pass

#         if need_build:
#             try:
#                 if hasattr(self.bcq_algo, "_impl"):
#                     self.bcq_algo._impl = None
#                 if hasattr(self.bcq_algo, "impl"):
#                     self.bcq_algo.impl = None
#                 self.bcq_algo.build_with_dataset(dataset)
#                 self._bcq_built = True
#                 self.bcq_action_size = active_action_dim
#                 print(f"[BCQ] Rebuilt with action_size={active_action_dim}")
#             except Exception as e_build:
#                 print(f"[BCQ] Rebuild failed (fallback to current impl): {e_build}")
#                 self._bcq_built = True  # 避免反复重建

#         # --------- 训练（多版本兼容）---------
#         update_steps = 20
#         training_results = None
#         try:
#             training_results = self.bcq_algo.fit(
#                 dataset,
#                 n_steps=update_steps,
#                 n_steps_per_epoch=update_steps,
#                 show_progress=False,
#                 save_interval=None,
#             )
#         except TypeError:
#             try:
#                 training_results = self.bcq_algo.fit(
#                     dataset,
#                     n_steps=update_steps,
#                     n_steps_per_epoch=update_steps,
#                     show_progress=False,
#                 )
#             except TypeError:
#                 try:
#                     training_results = self.bcq_algo.fit(dataset, n_epochs=1, show_progress=False)
#                 except TypeError:
#                     training_results = self.bcq_algo.fit(dataset, show_progress=False)

#         # --------- 记账 ---------
#         self._update_count = getattr(self, "_update_count", 0) + 1
#         self.updates_done = self._update_count
#         self._last_update_count = self._update_count
#         self.on_update(update_steps)

#         # 记录损失
#         latest = None
#         if training_results is not None:
#             hist = getattr(training_results, 'history', None)
#             if isinstance(hist, dict):
#                 for k in ('loss', 'td_loss', 'imitator_loss', 'objective_loss'):
#                     v = hist.get(k)
#                     if v:
#                         latest = float(v[-1])
#                         break
#         if latest is not None:
#             self.training_losses.append(latest)

#         return {
#             'updated': True,
#             'update_count': self.update_count,
#             'buffer_size': buffer_len,
#             'avg_loss': float(np.mean(list(self.training_losses))) if self.training_losses else 0.0,
#         }

#     except Exception as e:
#         print(f"BCQ online update failed: {e}")
#         return {'updated': False, 'error': str(e)}
    
#     # This 'def' must be at the same indentation level as 'def perform_update'
#     # There should not be an open 'try' statement immediately before this line
#     def predict(self, state: np.ndarray) -> int:
#         """BCQ策略预测"""
#         try:
#             if state.ndim == 1:
#                 state = state.reshape(1, -1)
            
#             action = self.bcq_algo.predict(state.astype(np.float32))
#             return int(action[0] if hasattr(action, '__len__') else action)
#         except Exception as e:
#             print(f"BCQ prediction error: {e}")
#             return 0
    
#     def get_statistics(self) -> Dict:
#         """获取训练统计信息"""
#         return {
#             'bcq_updates': self.update_count,
#             'bcq_buffer_size': len(self.online_buffer),
#             'bcq_avg_loss': np.mean(list(self.training_losses)) if self.training_losses else 0.0,
#             'transitions_processed': self.transition_count
#         }

# === Baseline 算法工厂（DQN / DoubleDQN / NFQ / CQL / CalQL[可选]）===
try:
    from d3rlpy.algos import DQNConfig, DoubleDQNConfig, NFQConfig
    try:
        from d3rlpy.algos import DiscreteCQLConfig as _CQLConfig
    except Exception:
        from d3rlpy.algos import CQLConfig as _CQLConfig
except Exception as _e:
    DQNConfig = DoubleDQNConfig = NFQConfig = _CQLConfig = None
    print("[WARN] d3rlpy.algos not fully available:", _e)

# 可选 Cal-QL（没有就用 CQL 兜底）
_HAS_CALQL = False
try:
    from calql import CalQL as _CalQLAlgo  # 没有此包会 ImportError
    _HAS_CALQL = True
except Exception:
    _HAS_CALQL = False

def _build_algo_by_name(name: str, device: str = "cuda"):
    n = (name or "dqn").lower()
    if n == "dqn":
        return DQNConfig().create(device=device)
    if n == "double_dqn":
        return DoubleDQNConfig().create(device=device)
    if n == "nfq":
        return NFQConfig().create(device=device)
    if n == "calql":
        if _HAS_CALQL:
            return _CalQLAlgo(device=device)
        print("[GenericTrainer] calql package not found — fallback to CQL in d3rlpy.")
        return _CQLConfig().create(device=device)
    if n == "cql":
        return _CQLConfig().create(device=device)
    # 默认安全选 DQN
    return DQNConfig().create(device=device)


class GenericD3RLPyOnlineTrainer:
    """
    与 BCQOnlineTrainer 同接口：
      - add_transition(transition) -> dict
      - perform_update() -> dict
      - predict(state: np.ndarray) -> int
      - get_statistics() -> dict

    通过环境变量 RL_BASELINE 选择 {bcq(不在此类)、dqn,double_dqn,nfq,cql,calql}
    """
    def __init__(self, algo_name: str = "dqn", device: str = "cuda"):
        import threading, time
        from collections import deque
        self.device = device
        self._algo_name = algo_name.strip().lower()
        self._batch_size = int(os.getenv("BATCH_SIZE", os.getenv("BCQ_BATCH_SIZE", 32)))
        self._update_frequency = max(1, int(os.getenv("UPDATE_FREQUENCY", os.getenv("BCQ_UPDATE_FREQUENCY", "20"))))
        self._update_steps = int(os.getenv("UPDATE_STEPS", os.getenv("BCQ_UPDATE_STEPS", "20")))

        self.labeled_buffer = []
        self.online_buffer = self.labeled_buffer
        self._buffer_lock = threading.Lock()
        self._built = False
        self._update_count = 0
        self.training_losses = deque(maxlen=200)
        self._last_update_walltime = time.time()

        # 初始化算法实例（未 build；用首批覆盖所有动作的样本来 build）
        self.algo = _build_algo_by_name(self._algo_name, device=self.device)
        print(f"[GenericTrainer] algo={self._algo_name} device={self.device} "
              f"(batch={self._batch_size}, freq={self._update_frequency}, steps={self._update_steps})")

    # 与在线循环兼容的属性
    @property
    def batch_size(self): return int(self._batch_size)
    @batch_size.setter
    def batch_size(self, v):
        try: self._batch_size = max(1, int(v))
        except: pass

    @property
    def update_frequency(self): return int(self._update_frequency)
    @update_frequency.setter
    def update_frequency(self, v):
        try: self._update_frequency = max(1, int(v))
        except: pass

    @property
    def update_count(self): return int(self._update_count)

    # ---------- 内部工具 ----------
    def _make_dataset(self, samples):
        import numpy as np
        from d3rlpy.dataset import MDPDataset
        obs = np.asarray([s["observation"] for s in samples], dtype=np.float32)
        next_obs = np.asarray([s["next_observation"] for s in samples], dtype=np.float32)
        acts = np.asarray([int(s["action"]) for s in samples], dtype=np.int64).reshape(-1, 1)
        rews = np.asarray([float(s["reward"]) for s in samples], dtype=np.float32)
        terms = np.asarray([bool(s["terminal"]) for s in samples], dtype=np.bool_)
        timeouts = np.zeros_like(terms, dtype=np.bool_)
        if not terms.any():
            timeouts[-1] = True
        try:
            return MDPDataset(observations=obs, actions=acts, rewards=rews,
                              terminals=terms, timeouts=timeouts, next_observations=next_obs)
        except TypeError:
            try:
                return MDPDataset(observations=obs, actions=acts, rewards=rews,
                                  terminals=terms, timeouts=timeouts)
            except TypeError:
                return MDPDataset(observations=obs, actions=acts, rewards=rews,
                                  terminals=(terms | timeouts))

    def _first_build_indices(self):
        # 选一批样本覆盖“已出现的所有动作类别”，避免 action_size 过小
        import random
        first_idx = {}
        for i, s in enumerate(self.online_buffer):
            a = int(s["action"])
            if a not in first_idx:
                first_idx[a] = i
        idxs = list(first_idx.values())
        if len(idxs) < min(self._batch_size, len(self.online_buffer)):
            rest = [i for i in range(len(self.online_buffer)) if i not in idxs]
            random.shuffle(rest)
            need = min(self._batch_size, len(self.online_buffer)) - len(idxs)
            idxs += rest[:need]
        return idxs

    def _ensure_built(self):
        if self._built or len(self.online_buffer) == 0:
            return
        idxs = self._first_build_indices()
        ds = self._make_dataset([self.online_buffer[i] for i in idxs])
        try:
            self.algo.build_with_dataset(ds)
        except Exception:
            pass
        self._built = True

    # ---------- 对外接口 ----------
    def add_transition(self, transition: dict):
        import numpy as np
        try:
            t = {
                "observation": np.asarray(transition["state"], dtype=np.float32),
                "action": int(transition["action"]),
                "reward": float(transition["reward"]),
                "next_observation": np.asarray(transition["next_state"], dtype=np.float32),
                "terminal": bool(transition.get("done", transition.get("terminal", False))),
            }
        except Exception as e:
            return {"updated": False, "error": f"bad transition: {e}"}

        with self._buffer_lock:
            self.online_buffer.append(t)
            buf_len = len(self.online_buffer)
            need_update = (buf_len % self._update_frequency == 0) and (buf_len >= self._batch_size)

        if need_update:
            return self.perform_update()
        return {"updated": False, "buffer_size": buf_len}

    def perform_update(self):
        import numpy as np
        with self._buffer_lock:
            buf_len = len(self.online_buffer)
            if buf_len == 0:
                return {"updated": False, "reason": "empty buffer"}
            # 先保证按“全动作覆盖”做一次 build
            self._ensure_built()
            size = min(self._batch_size, buf_len)
            batch_idxs = np.random.choice(buf_len, size=size, replace=False)
            batch = [self.online_buffer[i] for i in batch_idxs]

        ds = self._make_dataset(batch)
        tr = None
        try:
            tr = self.algo.fit(ds, n_steps=self._update_steps,
                               n_steps_per_epoch=self._update_steps,
                               show_progress=False, save_interval=None)
        except TypeError:
            try:
                tr = self.algo.fit(ds, n_steps=self._update_steps,
                                   n_steps_per_epoch=self._update_steps,
                                   show_progress=False)
            except TypeError:
                try:
                    tr = self.algo.fit(ds, n_epochs=1, show_progress=False)
                except TypeError:
                    tr = self.algo.fit(ds, show_progress=False)
        except Exception as e:
            msg = str(e)
            # 动作维度不够：重建 + 覆盖
            if "Class values must be smaller than num_classes" in msg:
                with self._buffer_lock:
                    cov_idxs = self._first_build_indices()
                    cov = [self.online_buffer[i] for i in cov_idxs]
                ds_cov = self._make_dataset(cov)
                # 重新创建算法并 build
                self.algo = _build_algo_by_name(self._algo_name, device=self.device)
                try:
                    self.algo.build_with_dataset(ds_cov)
                except Exception:
                    pass
                # 再试一次
                tr = self.algo.fit(ds, n_steps=self._update_steps,
                                   n_steps_per_epoch=self._update_steps,
                                   show_progress=False)
            else:
                print(f"[GenericTrainer] update failed: {e}")
                return {"updated": False, "error": msg, "buffer_size": buf_len}

        self._update_count += 1
        # 记录损失
        try:
            if tr and hasattr(tr, "history"):
                hist = getattr(tr, "history", {})
                last = None
                if isinstance(hist, dict):
                    for k in ("loss", "td_loss", "imitator_loss"):
                        if k in hist and hist[k]:
                            last = float(hist[k][-1])
                            break
                if last is not None:
                    self.training_losses.append(last)
        except Exception:
            pass

        avg_loss = float(np.mean(list(self.training_losses))) if self.training_losses else 0.0
        return {"updated": True, "update_count": self._update_count,
                "buffer_size": buf_len, "avg_loss": avg_loss}

    def predict(self, state: np.ndarray) -> int:
        import numpy as np
        try:
            x = np.asarray(state, dtype=np.float32)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            a = self.algo.predict(x)
            return int(a[0] if hasattr(a, "__len__") else a)
        except Exception:
            return 0

    def get_statistics(self):
        import numpy as np
        return {
            "algo": self._algo_name,
            "updates_done": int(self._update_count),
            "buffer_size": int(len(self.online_buffer)),
            "avg_loss": float(np.mean(list(self.training_losses))) if self.training_losses else 0.0,
        }



class BCQOnlineTrainer:
    """BCQ 的真正在线训练实现（去重 + 兼容 d3rlpy 多版本 + 动作越界防护 + 统计）"""
    def __init__(self, bcq_policy_path: str, device: str = 'cuda'):
        if not BCQ_AVAILABLE:
            raise ImportError("d3rlpy not available for BCQ")

        # 封装后置初始化，便于两种加载路径公用
        def _post_load_initialization(self):
            # 记录离线 BCQ 模型的动作空间大小
            self._bcq_built = False
            self.bcq_action_size = None
            try:
                cfg = getattr(self.bcq_algo, "config", None)
                self.bcq_action_size = getattr(cfg, "action_size", None)
                if self.bcq_action_size is None and getattr(self.bcq_algo, "impl", None) is not None:
                    self.bcq_action_size = getattr(self.bcq_algo.impl, "action_size", None)
            except Exception:
                pass
            if self.bcq_action_size is not None:
                print(f"[BCQ] Loaded policy expects action_size={self.bcq_action_size}")

            # 经验缓冲区 + 互斥锁
            if not hasattr(self, "online_buffer"):
                from collections import deque
                self.online_buffer = deque(maxlen=10000)
            if not hasattr(self, "_buffer_lock"):
                import threading
                self._buffer_lock = threading.Lock()

            # 计数 / 超参（允环境变量覆盖）
            import os, time
            self._transition_count = 0
            self._manual_transition_count = False
            try:
                from drive_tools import CURRENT_HYPERPARAMS as _HP
                _default_bs = int(_HP.get("batch_size", 32))
            except Exception:
                _default_bs = 32
            self._batch_size = int(os.getenv("BCQ_BATCH_SIZE", str(_default_bs)))
            self._update_frequency = max(1, int(os.getenv("BCQ_UPDATE_FREQUENCY", "20")))
            self._update_count = 0
            self._last_update_count = 0
            self.updates_done = 0
            self.update_calls = 0
            self._last_update_walltime = time.time()
            from collections import deque as _dq
            self.training_losses = _dq(maxlen=100)

        # 先走 d3rlpy.load_learnable，失败再尝试 torch.load
        try:
            self.bcq_algo = load_learnable(bcq_policy_path)
            print(f"✓ BCQ policy loaded successfully from {bcq_policy_path}")
            _post_load_initialization(self)
        except Exception as e:
            print(f"BCQ loading failed with error: {e}")
            print("Attempting alternative loading method...")
            try:
                import torch
                checkpoint = torch.load(bcq_policy_path, map_location=device)
                if hasattr(checkpoint, 'predict'):
                    self.bcq_algo = checkpoint
                    print("✓ BCQ policy loaded via torch.load")
                    _post_load_initialization(self)
                else:
                    raise Exception("Checkpoint doesn't contain BCQ algorithm")
            except Exception as e2:
                print(f"Alternative loading also failed: {e2}")
                raise ImportError(f"Cannot load BCQ policy: {e}")

        self.device = device
        print(f"✓ BCQ online trainer ready: update_freq={self.update_frequency}, batch_size={self.batch_size}")

    # --- 属性封装（外部访问这些字段时不再报 AttributeError） ---
    @property
    def batch_size(self) -> int:
        return getattr(self, "_batch_size", 32)
    @batch_size.setter
    def batch_size(self, value):
        try:
            self._batch_size = max(1, int(value))
        except Exception:
            self._batch_size = 32

    @property
    def update_frequency(self) -> int:
        return getattr(self, "_update_frequency", 20)
    @update_frequency.setter
    def update_frequency(self, value):
        try:
            self._update_frequency = max(1, int(value))
        except Exception:
            self._update_frequency = 20

    @property
    def update_count(self) -> int:
        return getattr(self, "_update_count", 0)
    @update_count.setter
    def update_count(self, value):
        try:
            self._update_count = int(value)
        except Exception:
            pass

    @property
    def transition_count(self):
        if getattr(self, "_manual_transition_count", False):
            return self._transition_count
        try:
            return len(self.online_buffer)
        except Exception:
            return getattr(self, "_transition_count", 0)
    @transition_count.setter
    def transition_count(self, value):
        try:
            v = int(value)
        except Exception:
            v = 0
        try:
            with self._buffer_lock:
                self._transition_count = v
                self._manual_transition_count = True
        except Exception:
            self._transition_count = v
            self._manual_transition_count = True

    # --- 在线数据接入 ---
    def add_transition(self, transition: Dict):
        """添加 transition 到在线 buffer；按频率触发训练"""
        import numpy as _np
        formatted = {
            'observation': _np.asarray(transition['state'], dtype=_np.float32),
            'action': int(transition['action']),
            'reward': float(transition['reward']),
            'next_observation': _np.asarray(transition['next_state'], dtype=_np.float32),
            'terminal': bool(transition.get('done', False))
        }
        with self._buffer_lock:
            self.online_buffer.append(formatted)
            self._transition_count += 1

        if (self._transition_count % self.update_frequency == 0 and
            len(self.online_buffer) >= self.batch_size):
            return self.perform_update()

        return {'updated': False, 'buffer_size': len(self.online_buffer)}

    def on_update(self, steps: int = 0):
        """每次 fit() 后做记录"""
        import time as _t
        self.update_calls += 1
        self._last_update_walltime = _t.time()

    # --- 关键：执行一次小步增量训练 ---
    def perform_update(self) -> Dict:
        import numpy as _np
        try:
            # 采样批次
            with self._buffer_lock:
                buf_len = len(self.online_buffer)
                if buf_len < self.batch_size:
                    return {'updated': False, 'reason': 'not enough samples'}
                idx = _np.random.choice(buf_len, self.batch_size, replace=False)
                batch = [self.online_buffer[i] for i in idx]

            obs  = _np.stack([t['observation']      for t in batch]).astype(_np.float32)
            nobs = _np.stack([t['next_observation'] for t in batch]).astype(_np.float32)
            act_raw = _np.asarray([int(t['action']) for t in batch], dtype=_np.int64)
            rew  = _np.asarray([float(t['reward'])  for t in batch], dtype=_np.float32)
            terminal = _np.asarray([bool(t['terminal']) for t in batch], dtype=_np.bool_)
            timeout  = _np.zeros_like(terminal, dtype=_np.bool_)
            if not terminal.any():
                timeout[-1] = True

            # 动作越界防护（避免 “Class values must be smaller than num_classes.”）
            if getattr(self, "bcq_action_size", None) is not None and act_raw.size > 0:
                # 过滤掉 >= action_size 的非法样本；若全非法则跳过本次更新
                keep = act_raw < int(self.bcq_action_size)
                if not keep.any():
                    return {'updated': False, 'reason': 'all actions out of range'}
                obs, nobs, rew, terminal, timeout, act_raw = \
                    obs[keep], nobs[keep], rew[keep], terminal[keep], timeout[keep], act_raw[keep]

            # d3rlpy 对离散动作期望 (N,1) int64
            actions = act_raw.reshape(-1, 1).astype(_np.int64)

            # 构建 MDPDataset（兼容不同版本参数）
            try:
                dataset = MDPDataset(
                    observations=obs, actions=actions, rewards=rew,
                    terminals=terminal, timeouts=timeout, next_observations=nobs
                )
            except TypeError:
                try:
                    dataset = MDPDataset(
                        observations=obs, actions=actions, rewards=rew,
                        terminals=terminal, timeouts=timeout
                    )
                except TypeError:
                    dataset = MDPDataset(
                        observations=obs, actions=actions, rewards=rew,
                        terminals=(terminal | timeout)
                    )

            # 首次用数据集 build（锁定 action_size 等内部实现），避免后续 batch 不一致
            if not getattr(self, "_bcq_built", False):
                try:
                    self.bcq_algo.build_with_dataset(dataset)
                except Exception:
                    pass
                self._bcq_built = True

            # 兼容多版本 fit() 签名
            steps = 20
            result = None
            try:
                result = self.bcq_algo.fit(
                    dataset,
                    n_steps=steps, n_steps_per_epoch=steps,
                    show_progress=False, save_interval=None
                )
            except TypeError:
                try:
                    result = self.bcq_algo.fit(
                        dataset,
                        n_steps=steps, n_steps_per_epoch=steps,
                        show_progress=False
                    )
                except TypeError:
                    try:
                        result = self.bcq_algo.fit(dataset, n_epochs=1, show_progress=False)
                    except TypeError:
                        result = self.bcq_algo.fit(dataset, show_progress=False)

            # 统计
            self._update_count = getattr(self, "_update_count", 0) + 1
            self.updates_done = self._update_count
            self._last_update_count = self._update_count
            self.on_update(steps)

            # 记录损失（不同版本键不同：loss / td_loss / imitator_loss）
            if result is not None and hasattr(result, 'history'):
                hist = getattr(result, 'history', {})
                last = None
                if isinstance(hist, dict):
                    for k in ('loss', 'td_loss', 'imitator_loss', 'td_error'):
                        if k in hist and hist[k]:
                            try:
                                last = float(hist[k][-1])
                                break
                            except Exception:
                                pass
                if last is not None:
                    self.training_losses.append(last)

            return {
                'updated': True,
                'update_count': self.update_count,
                'buffer_size': buf_len,
                'avg_loss': float(_np.mean(list(self.training_losses))) if self.training_losses else 0.0
            }

        except Exception as e:
            print(f"BCQ online update failed: {e}")
            return {'updated': False, 'error': str(e)}

    def predict(self, state: np.ndarray) -> int:
        """单步动作预测"""
        try:
            if state.ndim == 1:
                state = state.reshape(1, -1)
            action = self.bcq_algo.predict(state.astype(np.float32))
            return int(action[0] if hasattr(action, '__len__') else action)
        except Exception as e:
            print(f"BCQ prediction error: {e}")
            return 0

    def get_statistics(self) -> Dict:
        """供外部监控/打印"""
        import numpy as _np
        try:
            buf = len(self.online_buffer)
        except Exception:
            buf = 0
        return {
            'bcq_updates': self.update_count,
            'bcq_buffer_size': buf,
            'bcq_avg_loss': float(_np.mean(list(self.training_losses))) if self.training_losses else 0.0,
            'transitions_processed': self.transition_count
        }



class OnlineTrainer:
    """Manages online incremental training with active learning"""
    
    def __init__(self,
                 dynamics_model: TransformerDynamicsModel,
                 outcome_model: TreatmentOutcomeModel,
                 q_ensemble: EnsembleQNetwork,
                 active_learner: StreamActiveLearner,
                 batch_size: int = 32,
                 update_freq: int = 10,
                 save_freq: int = 600,  # Save every 10 minutes
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize online trainer
        
        Args:
            dynamics_model: Transformer dynamics model
            outcome_model: Treatment outcome model
            q_ensemble: Ensemble Q-network
            active_learner: Active learning sampler
            batch_size: Batch size for training
            update_freq: Update models every N transitions
            save_freq: Save checkpoint every N seconds
            device: Computing device
        """
        self.dynamics_model = dynamics_model.to(device)
        self.outcome_model = outcome_model.to(device)
        self.q_ensemble = q_ensemble.to(device)
        self.active_learner = active_learner
        self.device = device
        
        # Training parameters
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.save_freq = save_freq
        
        # Initialize incremental trainers
        self.dynamics_trainer = self._create_incremental_dynamics_trainer()
        self.outcome_trainer = self._create_incremental_outcome_trainer()
        
        # BCQ/CQL训练器初始化
        self.use_bcq = False
        self.bcq_trainer = None
        self.q_trainers = []
        
        # 检查BCQ模型是否存在
        bcq_path = os.path.join('./output/models', 'best_bcq_policy.d3')
        algo_choice = os.getenv("RL_BASELINE", "bcq").strip().lower()

        # 1) 仍想用 BCQ 且模型在：走原 BCQ 路线
        if algo_choice == "bcq" and os.path.exists(bcq_path) and BCQ_AVAILABLE:
            try:
                self.bcq_trainer = BCQOnlineTrainer(bcq_path, device)
                self.use_bcq = True
                print("✓ Using BCQ for online learning")
            except Exception as e:
                print(f"BCQ loading failed, fallback to baseline {algo_choice}: {e}")
                # 回落到通用 baseline（默认 dqn）
                algo_choice = os.getenv("RL_BASELINE_FALLBACK", "dqn")
                self.bcq_trainer = GenericD3RLPyOnlineTrainer(algo_name=algo_choice, device=device)
                self.use_bcq = True
                print(f"✓ Using baseline algo={algo_choice} for online learning")
        # 2) 明确要求 baseline：启用通用 trainer
        else:
            self.bcq_trainer = GenericD3RLPyOnlineTrainer(algo_name=algo_choice, device=device)
            self.use_bcq = True
            print(f"✓ Using baseline algo={algo_choice} for online learning")

        self.base_tau = 0.15 if self.use_bcq else 0.1
        # Keep track of expected dimensions for robust sanitization of streaming data
        self.state_dim = getattr(self.dynamics_model, 'state_dim', None)
        if self.state_dim is None and getattr(self.q_ensemble, 'q_networks', None):
            try:
                self.state_dim = getattr(self.q_ensemble.q_networks[0], 'state_dim', None)
            except Exception:
                self.state_dim = None

        self.action_dim = getattr(self.dynamics_model, 'action_dim', None)
        if self.action_dim is None and getattr(self.q_ensemble, 'q_networks', None):
            try:
                self.action_dim = getattr(self.q_ensemble.q_networks[0], 'action_dim', None)
            except Exception:
                self.action_dim = None

        self._last_sanitization_log = 0.0
        self._sanitization_events = 0

        # Buffers
        self.labeled_buffer = deque(maxlen=10000)
        self.weak_buffer = deque(maxlen=50000)
        self.query_buffer = deque(maxlen=1000)
        
        # Statistics
        self.stats = {
            'total_transitions': 0,
            'total_updates': 0,
            'total_queries': 0,
            'last_save_time': time.time(),
            'training_times': deque(maxlen=100)
        }
        self.evaluation_history = deque(maxlen=1000)
        # EMA for parameter stability
        self.ema_alpha = 0.99
        self._init_ema_params()
        
        # Training thread
        self.training_queue = queue.Queue()
        self.is_running = True
        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.training_thread.start()
        self.training_lock = threading.Lock()
        self.last_train_time = 0
        self.shift_detector = DistributionShiftDetector()
        print("Online trainer initialized with incremental dynamics and outcome models")
        self.learning_scheduler = ProgressiveLearningScheduler()
        self.base_learning_rate = 3e-4
        # self.base_tau = 0.1
    def _create_incremental_dynamics_trainer(self) -> DigitalTwinTrainer:
        """Create trainer for incremental dynamics updates - 按照论文实现"""
        trainer = DigitalTwinTrainer(
            self.dynamics_model,
            learning_rate=1e-4, 
            device=self.device
        )
        
        # Freeze all but last 2 layers
        self._freeze_transformer_layers(self.dynamics_model, num_trainable=2)
        
        return trainer
    def _freeze_transformer_layers(self, model: nn.Module, num_trainable: int = 2):
        """按照论文冻结Transformer的早期层"""
        # 获取Transformer的编码器层
        if hasattr(model, 'transformer'):
            encoder_layers = list(model.transformer.layers)
            
            # 冻结前面的层
            for layer in encoder_layers[:-num_trainable]:
                for param in layer.parameters():
                    param.requires_grad = False
            
            # 确保最后几层可训练
            for layer in encoder_layers[-num_trainable:]:
                for param in layer.parameters():
                    param.requires_grad = True
        
        # 始终保持输出层可训练
        if hasattr(model, 'output_projection'):
            for param in model.output_projection.parameters():
                param.requires_grad = True    

    def _create_incremental_outcome_trainer(self) -> OutcomeModelTrainer:
        """Create trainer for incremental outcome updates - 按照论文实现"""
        trainer = OutcomeModelTrainer(
            self.outcome_model,
            learning_rate=1e-4,
            regularization_weight=0.01,
            device=self.device
        )
        
        # 按照论文：冻结编码器，只训练输出头
        for name, param in self.outcome_model.named_parameters():
            if 'outcome_head' in name or 'treatment_discriminator' in name:
                param.requires_grad = True  # 保持输出相关层可训练
            else:
                param.requires_grad = False  # 冻结编码器
        
        return trainer
    
    def _create_incremental_q_trainers(self) -> List[ConservativeQLearning]:
            """Create trainers for each Q-network in ensemble"""
            trainers = []
            
            for q_net in self.q_ensemble.q_networks:
                trainer = ConservativeQLearning(
                    q_net,
                    [self.dynamics_model],  # 将单个模型包装为列表
                    self.outcome_model,
                    learning_rate=3e-4,
                    cql_weight=1.0,
                    device=self.device
                )
                trainers.append(trainer)
            
            return trainers
    
    def _freeze_early_layers(self, model: nn.Module, num_trainable: int = 2):
        """Freeze all but last N layers"""
        all_layers = list(model.children())
        
        # Freeze early layers
        for layer in all_layers[:-num_trainable]:
            for param in layer.parameters():
                param.requires_grad = False
        
        # Ensure last layers are trainable
        for layer in all_layers[-num_trainable:]:
            for param in layer.parameters():
                param.requires_grad = True
    
    def _init_ema_params(self):
        """Initialize EMA parameters"""
        self.ema_params = {}
        
        for name, model in [('dynamics', self.dynamics_model),
                           ('outcome', self.outcome_model)]:
            self.ema_params[name] = {}
            for param_name, param in model.named_parameters():
                if param.requires_grad:
                    self.ema_params[name][param_name] = param.data.clone()
    
    def _update_ema(self, model_name: str, model: nn.Module):
        """按照论文实现EMA: θ̄_{t+1} = α θ̄_t + (1-α) θ_{t+1}, α = 0.99"""
        alpha = 0.99  # 论文中明确提到的α值

        for param_name, param in model.named_parameters():
            if param.requires_grad and param_name in self.ema_params[model_name]:
                # EMA更新公式
                self.ema_params[model_name][param_name] = (
                    alpha * self.ema_params[model_name][param_name] +
                    (1 - alpha) * param.data
                )

    def _log_sanitization(self, message: str):
        """Rate-limited logging for data sanitization events."""
        self._sanitization_events += 1
        now = time.time()
        if now - self._last_sanitization_log > 5.0:
            print(f"[SANITIZER] {message} (total events: {self._sanitization_events})")
            self._last_sanitization_log = now

    def _sanitize_vector(self, vector, target_dim: Optional[int], field_name: str) -> np.ndarray:
        """Coerce incoming vectors to the expected dimensionality with graceful fallbacks."""
        changed = False
        original_len = None

        if isinstance(vector, torch.Tensor):
            vector = vector.detach().cpu().numpy()

        if vector is None:
            original_len = 0
            if target_dim and target_dim > 0:
                arr = np.zeros(target_dim, dtype=np.float32)
            else:
                arr = np.zeros(1, dtype=np.float32)
            changed = True
        else:
            try:
                arr = np.asarray(vector, dtype=np.float32).reshape(-1)
                original_len = arr.size
            except Exception:
                target = target_dim or 1
                arr = np.zeros(target, dtype=np.float32)
                original_len = 0
                changed = True

        if not np.all(np.isfinite(arr)):
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            changed = True

        effective_target = target_dim if target_dim and target_dim > 0 else arr.size

        if effective_target and arr.size != effective_target:
            if arr.size < effective_target:
                arr = np.pad(arr, (0, effective_target - arr.size), mode='constant')
            else:
                arr = arr[:effective_target]
            changed = True

        if changed:
            self._log_sanitization(
                f"{field_name}: coerced length {original_len} -> {arr.size}"
            )

        return arr.astype(np.float32, copy=False)

    def _sanitize_action(self, action) -> int:
        """Ensure actions are integers within the supported range."""
        original_action = action
        try:
            if isinstance(action, (list, tuple, np.ndarray)):
                action = np.asarray(action).reshape(-1)[0]
            action = int(action)
        except Exception:
            action = 0

        if self.action_dim is not None:
            if action < 0 or action >= self.action_dim:
                clipped = int(np.clip(action, 0, self.action_dim - 1))
                self._log_sanitization(
                    f"action: clipped out-of-range value {original_action} -> {clipped}"
                )
                action = clipped

        return action

    def _sanitize_reward(self, reward) -> float:
        """Coerce rewards to finite float values."""
        try:
            if isinstance(reward, (list, tuple, np.ndarray)):
                reward = np.asarray(reward).reshape(-1)[0]
            reward = float(reward)
        except Exception:
            reward = 0.0

        if not np.isfinite(reward):
            self._log_sanitization("reward: replaced non-finite value with 0.0")
            reward = 0.0

        return reward

    def _sanitize_transition(self, transition: Optional[Dict], context: str = 'stream') -> Dict:
        """Return a safe-to-use transition dict with correct dimensions."""
        base = transition if isinstance(transition, dict) else {}
        if base is transition:
            sanitized = dict(base)
        else:
            try:
                sanitized = dict(base)
            except Exception:
                sanitized = {}

        state_dim = self.state_dim
        sanitized_state = self._sanitize_vector(
            sanitized.get('state'), state_dim, f'{context}.state'
        )
        sanitized_next_state = self._sanitize_vector(
            sanitized.get('next_state', sanitized_state), state_dim, f'{context}.next_state'
        )

        sanitized_action = self._sanitize_action(sanitized.get('action', 0))
        sanitized_reward = self._sanitize_reward(sanitized.get('reward', 0.0))

        sanitized['state'] = sanitized_state
        sanitized['next_state'] = sanitized_next_state
        sanitized['action'] = sanitized_action
        sanitized['reward'] = sanitized_reward
        sanitized['done'] = bool(sanitized.get('done', False))

        return sanitized
    
    def process_transition(self, transition: Dict) -> str:
        """
        Process an incoming transition, incorporating active learning and distribution shift detection.
        """
        # 1. Normalize data: Convert any tensors to numpy/python types
        if isinstance(transition['state'], torch.Tensor):
            transition['state'] = transition['state'].cpu().numpy()
        if isinstance(transition['next_state'], torch.Tensor):
            transition['next_state'] = transition['next_state'].cpu().numpy()
        if isinstance(transition['action'], torch.Tensor):
            transition['action'] = transition['action'].cpu().numpy().item()
        if isinstance(transition['reward'], torch.Tensor):
            transition['reward'] = transition['reward'].cpu().numpy().item()

        # 1.5 Sanitize transition to guard against schema mismatches or invalid values
        transition = self._sanitize_transition(transition, context='stream')

        # 2. Update counters and monitor distribution shift (FIX APPLIED)
        self.stats['total_transitions'] += 1

        # Add the current state to the distribution shift detector
        self.shift_detector.add_state(transition['state'])
        
        # Every 1000 transitions, check for a significant distribution shift
        if self.stats['total_transitions'] > 0 and self.stats['total_transitions'] % 1000 == 0:
            shift_result = self.shift_detector.detect_shift()
            severity = shift_result.get('severity', 0)
            severity_str = f"{severity:.3f}" if isinstance(severity, (int, float)) else str(severity)
            print(f"Periodic distribution shift check at step {self.stats['total_transitions']}: "
                f"Shift detected: {shift_result.get('shift_detected')}, "
                f"Severity: {severity_str}")
            
            if shift_result.get('shift_detected'):
                # If a shift is detected, trigger the adaptation strategy
                self._handle_distribution_shift(shift_result)
                
        # Print status periodically
        if self.stats['total_transitions'] % 100 == 0:
            print(f"Processed {self.stats['total_transitions']} transitions")

        # 3. Active learning decision logic
        decision, batch = self.active_learner.process_transition(transition)
        
        if decision == 'query':
            self.query_buffer.append(transition)
            self.stats['total_queries'] += 1
            return 'queried'
            
        elif decision == 'query_batch':
            sanitized_batch = [
                self._sanitize_transition(item, context='query_batch') for item in (batch or [])
            ]
            self.query_buffer.extend(sanitized_batch)
            self.stats['total_queries'] += len(sanitized_batch)
            return f'queried_batch_{len(sanitized_batch)}'
            
        elif decision == 'weak':
            self.weak_buffer.append(transition)
            
            # Use a lock to prevent race conditions when triggering training
            with self.training_lock:
                current_time = time.time()
                # Check if there's enough data and enough time has passed
                if (len(self.labeled_buffer) >= self.batch_size and
                    current_time - self.last_train_time > 1.0):
                    
                    self.training_queue.put('train')
                    self.last_train_time = current_time
                    return 'training'
            
            return 'buffered'
        
        return 'buffered'
    def _handle_distribution_shift(self, shift_result: Dict):
        """处理检测到的分布偏移"""
        print("🔄 Adapting to distribution shift...")
        
        # 1. 降低不确定性阈值，增加查询频率
        current_tau = self.active_learner.get_statistics().get('current_threshold', 0.1)
        new_tau = max(current_tau * 0.7, 0.02)  # 降低30%但不低于0.02
        self.active_learner.update_threshold(new_tau)
        
        # 2. 增加学习率以快速适应
        self.update_hyperparameters({'learning_rate': 5e-4})  # 临时提高学习率
        
        # 3. 清理旧的经验重放缓冲区
        if hasattr(self, 'q_trainers'):
            for q_trainer in self.q_trainers:
                # 保留最近的经验，清理旧经验
                if len(q_trainer.replay_buffer) > 5000:
                    recent_experiences = list(q_trainer.replay_buffer)[-5000:]
                    q_trainer.replay_buffer.clear()
                    q_trainer.replay_buffer.extend(recent_experiences)
        
        print(f"✅ Adaptation complete: tau={new_tau:.3f}, lr=5e-4")  

    def add_labeled_transition(self, transition: Dict, source: str = 'expert'):
        """Add expert-labeled transition"""
        sanitized = self._sanitize_transition(transition, context='labeled')
        sanitized['label_source'] = source
        sanitized['label_time'] = time.time()
        self.labeled_buffer.append(sanitized)
    
    def _training_loop(self):
        """Background training loop with mixed blocking/non-blocking approach."""
        while self.is_running:
            try:
                # 非阻塞检查队列命令
                try:
                    command = self.training_queue.get_nowait()
                    if command is None or command == 'stop':
                        break
                    elif command == 'save':
                        self._save_checkpoint()
                    elif command == 'train':
                        pass  # 继续执行下面的训练逻辑
                except queue.Empty:
                    pass  # 没有命令，继续正常训练
                
                # 主动检查是否有足够数据进行训练
                if len(self.labeled_buffer) >= min(self.batch_size, 8):
                    self._train_step()
                
                # 检查是否需要保存
                if time.time() - self.stats['last_save_time'] > self.save_freq:
                    self._save_checkpoint()
                    self.stats['last_save_time'] = time.time()
                    
                # 适当的睡眠时间
                time.sleep(0.1)
                    
            except Exception as e:
                print(f"Training loop error: {e}")
                time.sleep(1.0)
    
    def _train_step(self):
        """Executes a training step, incorporating progressive learning adjustments."""
        start_time = time.time()

        # ----------------- Batch Preparation -----------------
        # FIX: The check should be against the minimum required batch size, not the full batch size.
        # The trigger condition is min(self.batch_size, 8), so we match that here.
        if len(self.labeled_buffer) < min(self.batch_size, 8):
            return
        actual_batch_size = min(self.batch_size, len(self.labeled_buffer))
        print(f"Using actual batch size: {actual_batch_size}")  

        # Sample batch
        indices = np.random.choice(len(self.labeled_buffer), actual_batch_size, replace=True)
        batch = []
        for idx in indices:
            sanitized = self._sanitize_transition(self.labeled_buffer[idx], context='train')
            try:
                self.labeled_buffer[idx] = sanitized
            except Exception:
                pass
            batch.append(sanitized)

        # Convert to tensors with additional safeguards
        try:
            states = torch.stack([torch.FloatTensor(t['state']) for t in batch]).to(self.device)
            actions = torch.LongTensor([t['action'] for t in batch]).to(self.device)
            rewards = torch.FloatTensor([t['reward'] for t in batch]).to(self.device)
            next_states = torch.stack([torch.FloatTensor(t['next_state']) for t in batch]).to(self.device)
        except RuntimeError as tensor_error:
            self._log_sanitization(f"batch tensorization failed ({tensor_error}); skipping training step")
            return

        # ----------------- Model Training -----------------
        # Update dynamics model
        if hasattr(self.dynamics_trainer, 'train_online'):
            sequences = self._create_sequences(batch)
            if sequences:
                dynamics_loss = self.dynamics_trainer.train_online(sequences)
                self._update_ema('dynamics', self.dynamics_model)

        # Update outcome model
        outcome_batch = {
            'state': states,
            'action': actions,
            'reward': rewards
        }
        outcome_loss = self.outcome_trainer.train_step_online(outcome_batch)
        self._update_ema('outcome', self.outcome_model)

        # Update Q-networks
        # RL模型更新 (BCQ vs CQL)   
        rl_metrics = {}
        
        if self.use_bcq and self.bcq_trainer:
            # BCQ在线训练
            for t in batch:
                bcq_result = self.bcq_trainer.add_transition(t)
                
                # --- 使用新的、统一的打印逻辑替换旧版 ---
                if bcq_result.get('updated'):
                    algo_name = getattr(self.bcq_trainer, "_algo_name", "bcq")
                    print(
                        f"[{algo_name.upper()}] updated: avg_loss={bcq_result.get('avg_loss', 0):.4f}, "
                        f"updates={bcq_result.get('update_count', -1)}, "
                        f"buffer={bcq_result.get('buffer_size', -1)}"
                    )
                else:
                    # 保留对“跳过更新”原因的打印
                    reason = bcq_result.get('reason') or bcq_result.get('error')
                    if reason:
                        # 您可以考虑将这里的 "BCQ" 也替换为动态名称
                        algo_name = getattr(self.bcq_trainer, "_algo_name", "bcq")
                        print(f"[{algo_name.upper()}] update skipped: {reason}")
            
            rl_metrics = self.bcq_trainer.get_statistics()
        else:
            # CQL训练 (原有逻辑)
            for q_trainer in self.q_trainers:
                for t in batch:
                    transition = {
                        'state': torch.FloatTensor(t['state']),
                        'action': torch.tensor(t['action'], dtype=torch.long),
                        'reward': torch.tensor(t['reward'], dtype=torch.float32),
                        'next_state': torch.FloatTensor(t['next_state'])
                    }
                    q_trainer.add_to_replay_buffer(transition)
                
                # Multiple gradient steps for data efficiency
                for _ in range(min(20, max(1, len(q_trainer.replay_buffer) // 100))):
                    q_trainer.train_step()
        
        # ----------------- EMA Application -----------------
        # Every 100 steps, apply the stable EMA weights back to the online models
        if self.stats['total_updates'] > 0 and self.stats['total_updates'] % 100 == 0:
            self._apply_ema_weights()
            print(f"Applied EMA weights at step {self.stats['total_updates']}.")

        # ----------------- Statistics and Adaptive Logic -----------------
        # (FIX APPLIED) Check for and apply scheduled learning phase changes
        update_info = self.learning_scheduler.should_update_hyperparams(
            self.stats['total_transitions']
        )
        if update_info:
            print(f"\n📚 Entering '{update_info['phase']}' learning phase at transition {self.stats['total_transitions']}")
            
            # Adjust learning rate based on the schedule
            new_lr = self.base_learning_rate * update_info['learning_rate_multiplier']
            self.update_hyperparameters({'learning_rate': new_lr})
            
            # Adjust uncertainty threshold (tau) based on the schedule
            new_tau = self.base_tau * update_info['tau_multiplier']
            self.active_learner.update_threshold(new_tau)
            
            print(f"  - Learning rate set to: {new_lr:.2e}")
            print(f"  - Uncertainty threshold (tau) set to: {new_tau:.3f}\n")

        # Update step statistics
        self.stats['total_updates'] += 1
        self.stats['training_times'].append(time.time() - start_time)

        # Log evaluation history periodically
        if self.stats['total_updates'] % 10 == 0:
            self.evaluation_history.append({
                'timestamp': time.time(),
                'active_learning': self.active_learner.get_statistics(),
                'performance': np.mean(list(self.stats['training_times'])) if self.stats['training_times'] else 0
            })

        # Adaptively fine-tune the active learning threshold based on recent query rate
        if self.stats['total_updates'] % 50 == 0:
            al_stats = self.active_learner.get_statistics()
            current_query_rate = al_stats.get('query_rate', 0)
            current_tau = al_stats.get('current_threshold', 0.05)
            
            print(f"[ADAPTIVE] Update {self.stats['total_updates']}: Query rate: {current_query_rate:.1%}, Tau: {current_tau:.3f}")
            
            if current_query_rate > 0.2:
                new_tau = min(current_tau * 1.2, 0.8)
                self.active_learner.update_threshold(new_tau)
                print(f"[ADAPTIVE] Query rate high, increasing tau to {new_tau:.3f}")
            
            # If query rate is too low, decrease threshold to query more
            elif current_query_rate < 0.05:
                new_tau = max(current_tau * 0.8, 0.01)  # Decrease by 20%, with a floor of 0.01
                self.active_learner.update_threshold(new_tau)
                print(f"Query rate low ({current_query_rate:.1%}). Fine-tuning tau to {new_tau:.3f}")


    def _apply_ema_weights(self):
        """将EMA权重应用到模型（用于稳定性）"""
        for model_name, model in [('dynamics', self.dynamics_model), 
                                ('outcome', self.outcome_model)]:
            if model_name in self.ema_params:
                for param_name, param in model.named_parameters():
                    if param.requires_grad and param_name in self.ema_params[model_name]:
                        # 临时保存当前权重并应用EMA权重
                        param.data.copy_(self.ema_params[model_name][param_name])
                        print(f"Applied EMA weight for {model_name}.{param_name}")
                        
    def _create_sequences(self, batch: List[Dict]) -> List[Dict]:
        """Create sequences from transitions for dynamics training"""
        # Group by trajectory if available
        sequences = []
        
        # For now, treat each transition independently
        # In production, you'd group by patient trajectory
        for trans in batch[:10]:  # Limit sequences
            seq = {
                'states': torch.stack([
                    torch.FloatTensor(trans['state']),
                    torch.FloatTensor(trans['next_state'])
                ]),
                'actions': torch.tensor([trans['action']]),
                'length': 2
            }
            sequences.append(seq)
        
        return sequences
    
    def _save_checkpoint(self):
        """Save model checkpoint"""
        checkpoint_dir = './checkpoints/online'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save models
        torch.save(self.dynamics_model.state_dict(), 
                  f'{checkpoint_dir}/dynamics_{timestamp}.pth')
        torch.save(self.outcome_model.state_dict(), 
                  f'{checkpoint_dir}/outcome_{timestamp}.pth')
        torch.save(self.q_ensemble.state_dict(), 
                  f'{checkpoint_dir}/q_ensemble_{timestamp}.pth')
        
        # Save EMA parameters
        torch.save(self.ema_params, 
                  f'{checkpoint_dir}/ema_{timestamp}.pth')
        
        # Save statistics
        stats_to_save = self.get_statistics()
        # FIX: Convert deque to list for JSON serialization
        if 'training_times' in stats_to_save and isinstance(stats_to_save['training_times'], deque):
            stats_to_save['training_times'] = list(stats_to_save['training_times'])

        with open(f'{checkpoint_dir}/stats_{timestamp}.json', 'w') as f:
            json.dump(stats_to_save, f, indent=2)
        
        print(f"Checkpoint saved at {timestamp}")
    
    def get_statistics(self) -> Dict:
        """Get training statistics"""
        stats = self.stats.copy()
        
        # Add buffer sizes
        stats['labeled_buffer_size'] = len(self.labeled_buffer)
        stats['weak_buffer_size'] = len(self.weak_buffer)
        stats['query_buffer_size'] = len(self.query_buffer)
        
        # Add timing stats
        if self.stats['training_times']:
            stats['avg_training_time'] = np.mean(list(self.stats['training_times']))
            stats['max_training_time'] = np.max(list(self.stats['training_times']))
        
        # Add active learning stats
        stats.update(self.active_learner.get_statistics())
        
        return stats
    
    def update_hyperparameters(self, params: Dict):
        """Update training hyperparameters on the fly"""
        if 'learning_rate' in params:
            # Update all optimizers
            for trainer in [self.dynamics_trainer] + self.q_trainers:
                for param_group in trainer.optimizer.param_groups:
                    param_group['lr'] = params['learning_rate']
            
            if hasattr(self.outcome_trainer, 'optimizer'):
                for param_group in self.outcome_trainer.optimizer.param_groups:
                    param_group['lr'] = params['learning_rate']
        
        if 'batch_size' in params:
            self.batch_size = params['batch_size']
        
        if 'tau' in params:
            self.active_learner.update_threshold(params['tau'])
        
        if 'cql_weight' in params:
            for trainer in self.q_trainers:
                trainer.cql_weight = params['cql_weight']
        
        print(f"Updated hyperparameters: {params}")
    
    def stop(self):
        """Stops the training loop by sending a sentinel value to the queue."""
        print("Attempting to stop the online training thread...")
        if self.is_running:
            self.is_running = False
            # Put a sentinel value on the queue to unblock the thread's get() call.
            self.training_queue.put(None) 
            if self.training_thread:
                # Wait for the thread to finish its work.
                self.training_thread.join(timeout=5) 
                if self.training_thread.is_alive():
                    print("Warning: Training thread did not stop gracefully.")
        print("Online training stopped")



class ProgressiveLearningScheduler:
    """渐进式学习调度器"""
    
    def __init__(self):
        self.learning_phases = {
            'bootstrap': {'duration': 1000, 'lr_multiplier': 2.0, 'tau_multiplier': 0.5},
            'exploration': {'duration': 3000, 'lr_multiplier': 1.5, 'tau_multiplier': 0.8},
            'exploitation': {'duration': float('inf'), 'lr_multiplier': 1.0, 'tau_multiplier': 1.0}
        }
        
        self.current_phase = 'bootstrap'
        self.phase_start_time = 0
        self.transition_count = 0
        
    def update_phase(self, total_transitions: int) -> bool:
        """更新学习阶段"""
        phase_changed = False
        
        if self.current_phase == 'bootstrap' and total_transitions >= 1000:
            self.current_phase = 'exploration'
            self.transition_count = total_transitions
            phase_changed = True
            print("🔄 Transitioning to EXPLORATION phase")
            
        elif self.current_phase == 'exploration' and total_transitions >= 4000:
            self.current_phase = 'exploitation'
            self.transition_count = total_transitions
            phase_changed = True
            print("🔄 Transitioning to EXPLOITATION phase")
        
        return phase_changed
    
    def get_current_multipliers(self) -> Dict:
        """获取当前阶段的参数乘数"""
        return self.learning_phases[self.current_phase]
    
    def should_update_hyperparams(self, total_transitions: int) -> Optional[Dict]:
        """检查是否需要更新超参数"""
        if self.update_phase(total_transitions):
            multipliers = self.get_current_multipliers()
            return {
                'learning_rate_multiplier': multipliers['lr_multiplier'],
                'tau_multiplier': multipliers['tau_multiplier'],
                'phase': self.current_phase
            }
        return None

class DistributionShiftDetector:
    """检测患者群体分布偏移"""
    
    def __init__(self, window_size=1000, shift_threshold=0.1):
        self.window_size = window_size
        self.shift_threshold = shift_threshold
        self.reference_states = deque(maxlen=window_size)
        self.current_states = deque(maxlen=window_size//2)
        self.last_check_time = time.time()
        
    def add_state(self, state: np.ndarray):
        """添加新状态到监控"""
        if len(self.reference_states) < self.window_size:
            self.reference_states.append(state)
        else:
            self.current_states.append(state)
            
    def detect_shift(self) -> Dict:
            """使用Kolmogorov-Smirnov测试检测分布偏移"""
            if len(self.current_states) < 100:  # 需要足够样本
                return {'shift_detected': False, 'p_value': 1.0, 'severity': 0.0, 'affected_dimensions': []}
                
            from scipy import stats
            
            ref_data = np.array(list(self.reference_states))
            curr_data = np.array(list(self.current_states))
            
            # 对每个特征维度进行KS测试
            p_values = []
            for dim in range(ref_data.shape[1]):
                _, p_val = stats.ks_2samp(ref_data[:, dim], curr_data[:, dim])
                p_values.append(p_val)
                
            min_p_value = min(p_values)
            shift_detected = min_p_value < self.shift_threshold
            
            if shift_detected:
                print(f"🚨 Distribution shift detected! p-value: {min_p_value:.4f}")
                
            # 计算严重程度
            severity = 1.0 - min_p_value if shift_detected else 0.0
            return {
                'shift_detected': shift_detected,
                'p_value': min_p_value,
                'severity': severity,
                'affected_dimensions': [i for i, p in enumerate(p_values) if p < self.shift_threshold]
            }

class OnlineDataStream:
    """Simulates streaming patient data"""
    
    def __init__(self, 
                 data_source: Callable,
                 stream_rate: float = 1.0,
                 add_noise: bool = True):
        """
        Initialize data stream
        
        Args:
            data_source: Function that returns transitions
            stream_rate: Transitions per second
            add_noise: Add realistic noise to transitions
        """
        self.data_source = data_source
        self.stream_rate = stream_rate
        self.add_noise = add_noise
        self.is_streaming = False
        self.stream_thread = None
        self.callbacks = []
        self.training_lock = threading.Lock()
        self.last_train_time = 0

    def add_callback(self, callback: Callable):
        """Add callback for new transitions"""
        self.callbacks.append(callback)
    
    def start_stream(self):
        """Start data streaming"""
        self.is_streaming = True
        self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.stream_thread.start()
        print(f"Data stream started at {self.stream_rate} transitions/sec")
    
    def stop_stream(self):
        """Stop data streaming"""
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join()
        print("Data stream stopped")
    
    def _stream_loop(self):
        """Main streaming loop with proper rate control"""
        while self.is_streaming:
            loop_start = time.perf_counter()
            
            # 获取和处理数据
            transition = self.data_source()
            if transition is None:
                time.sleep(0.1)
                continue
            
            if self.add_noise:
                transition = self._add_noise(transition)
            
            # 调用回调函数
            for callback in self.callbacks:
                try:
                    callback(transition)
                except Exception as e:
                    print(f"Callback error: {e}")
            
            # 精确的速率控制
            process_time = time.perf_counter() - loop_start
            target_interval = 1.0 / self.stream_rate
            sleep_time = max(0, target_interval - process_time)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _add_noise(self, transition: Dict) -> Dict:
        """Add realistic noise to transition"""
        noisy_transition = transition.copy()
        
        # Add small Gaussian noise to states
        if 'state' in transition:
            noise = np.random.normal(0, 0.01, size=transition['state'].shape)
            noisy_transition['state'] = transition['state'] + noise
            noisy_transition['state'] = np.clip(noisy_transition['state'], 0, 1)
        
        if 'next_state' in transition:
            noise = np.random.normal(0, 0.01, size=transition['next_state'].shape)
            noisy_transition['next_state'] = transition['next_state'] + noise
            noisy_transition['next_state'] = np.clip(noisy_transition['next_state'], 0, 1)
        
        return noisy_transition


class ExpertSimulator:
    """Simulates expert labeling with delay"""
    
    def __init__(self,
                 label_delay: float = 2.0,
                 accuracy: float = 0.95):
        """
        Initialize expert simulator
        
        Args:
            label_delay: Seconds to wait before returning label
            accuracy: Probability of correct label
        """
        self.label_delay = label_delay
        self.accuracy = accuracy
        self.label_queue = queue.Queue()
        self.is_running = True
        self.label_thread = threading.Thread(target=self._label_loop, daemon=True)
        self.label_thread.start()
    
    def request_label(self, transition: Dict, callback: Callable):
        """Request expert label for transition"""
        self.label_queue.put({
            'transition': transition,
            'callback': callback,
            'request_time': time.time()
        })
    
    def _label_loop(self):
        """Process labeling requests"""
        while self.is_running:
            try:
                request = self.label_queue.get(timeout=1.0)
                
                # 添加调试
                print(f"Expert labeling request received")
                
                # Simulate delay
                time.sleep(self.label_delay)
                
                # Generate label (possibly with error)
                labeled_transition = self._generate_label(request['transition'])
                
                # Call callback
                request['callback'](labeled_transition)
                
                # 添加调试
                print(f"Expert label provided")
                
            except queue.Empty:
                continue
    
    def _generate_label(self, transition: Dict) -> Dict:
        """Generate expert label"""
        labeled = transition.copy()
        
        # Simulate expert providing true reward
        if np.random.rand() < self.accuracy:
            # Correct label - compute true reward
            labeled['reward'] = self._compute_true_reward(transition)
            labeled['label_quality'] = 'expert_verified'
        else:
            # Noisy label
            labeled['reward'] = transition.get('reward', 0) + np.random.normal(0, 0.1)
            labeled['label_quality'] = 'expert_noisy'
        
        return labeled
    
    def _compute_true_reward(self, transition: Dict) -> float:
        """Compute true reward for transition"""
        # Simplified reward based on state improvement
        state = transition['state']
        next_state = transition['next_state']
        
        # Health improvement
        health_improvement = 0.0
        
        # Oxygen saturation (most critical)
        o2_improvement = (next_state[8] - state[8]) * 5.0
        health_improvement += o2_improvement
        
        # Other vitals
        for i in [2, 3, 4, 5]:  # BP, HR, glucose, creatinine
            # Reward moving toward normal (0.5)
            old_deviation = abs(state[i] - 0.5)
            new_deviation = abs(next_state[i] - 0.5)
            improvement = old_deviation - new_deviation
            health_improvement += improvement * 2.0
        
        return health_improvement
    
    def stop(self):
        """Stop expert simulator"""
        self.is_running = False
        self.label_thread.join()

from data import PatientDataGenerator
def create_online_training_system(models: Dict,
                                  sampler_type: str = 'hybrid',
                                  tau: float = 0.05,
                                  stream_rate: float = 1.0) -> Dict:
    """
    Create complete online training system with distribution shift simulation.
    
    Args:
        models: Dict with dynamics_model, outcome_model, q_ensemble
        sampler_type: Type of active learning sampler
        tau: Uncertainty threshold
        stream_rate: Data streaming rate
        
    Returns:
        Dict with trainer, stream, expert, and controller
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create active learner
    # 检查BCQ是否可用，调整active learner配置
    bcq_path = os.path.join('./output/models', 'best_bcq_policy.d3')
    use_bcq = os.path.exists(bcq_path) and BCQ_AVAILABLE
    
    if use_bcq:
        print("✓ Found BCQ policy, adapting active learner")
        # BCQ情况下，适当调整tau阈值
        tau = max(tau, 0.1)  # BCQ需要更高的阈值
    
    # Create active learner
    active_learner = StreamActiveLearner(
        models['q_ensemble'],
        sampler_type=sampler_type,
        tau=tau,
        device=device
    )
    
    # Create online trainer
    trainer = OnlineTrainer(
        models['dynamics_model'],
        models['outcome_model'],
        models['q_ensemble'],
        active_learner,
        device=device
    )
    
    # Create expert simulator
    expert = ExpertSimulator(label_delay=0.1)

    # --- START: MODIFIED DATA GENERATION LOGIC ---

    # 创建两个不同的数据生成器
    offline_generator = PatientDataGenerator(n_patients=1000, seed=42)
    online_generator = PatientDataGenerator(n_patients=5000, seed=123) # 不同的seed和患者
    
    # 预先生成离线数据集
    full_dataset = offline_generator.generate_dataset()

    # 模拟分布偏移
    distribution_shift_time = 5000
    transition_count = [0]

    def data_generator():
        # 跟踪总共生成了多少转换
        current_count = transition_count[0]
        
        # 前5000个样本使用原始分布
        if current_count < distribution_shift_time:
            idx = current_count % len(full_dataset['states'])
            transition = {
                'state': full_dataset['states'][idx],
                'action': full_dataset['actions'][idx],
                'reward': full_dataset['rewards'][idx],
                'next_state': full_dataset['next_states'][idx]
            }
        else:
            # 5000个样本后，模拟分布偏移（例如更多老年患者）
            state = online_generator._generate_initial_state()
            state[0] = np.clip(state[0] + 0.3, 0, 1)  # 增加年龄
            action = online_generator._behavior_policy(state)
            next_state = online_generator._transition_dynamics(state, action)
            reward = online_generator._compute_reward(state, action, next_state)
            
            transition = {
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state
            }
        
        # 每次调用都增加计数器
        transition_count[0] += 1
        return transition

    # --- END: MODIFIED DATA GENERATION LOGIC ---
    
    stream = OnlineDataStream(data_generator, stream_rate=stream_rate)
    
    # Connect components
    def process_transition(transition):
        status = trainer.process_transition(transition)
        
        if 'queried' in status:
            # 修改回调以触发训练检查
            def labeled_callback(labeled_trans):
                trainer.add_labeled_transition(labeled_trans)
                # 添加：检查是否可以训练
                if len(trainer.labeled_buffer) >= min(trainer.batch_size, 8):
                    trainer.training_queue.put('train')
                    print(f"Labeled buffer size: {len(trainer.labeled_buffer)}, triggering training")
            
            expert.request_label(transition, labeled_callback)
    
    stream.add_callback(process_transition)
    
    return {
        'trainer': trainer,
        'stream': stream,
        'expert': expert,
        'active_learner': active_learner
    }


# Extension for DigitalTwinTrainer to support online updates
def train_online(self, sequences: List[Dict]) -> float:
    """Online training step for dynamics model"""
    self.model.train()
    total_loss = 0.0
    
    for seq in sequences:
        states = seq['states'].unsqueeze(0).to(self.device)
        if states.shape[1] < 2:
            continue
            
        # Single action for transition
        actions = seq['actions'].unsqueeze(0).to(self.device)
        
        # Forward pass - predict next state
        predicted_next = self.model(states[:, :-1], actions)
        target_next = states[:, 1:]
        
        # Loss
        loss = nn.functional.mse_loss(predicted_next, target_next)
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(sequences) if sequences else 0.0


# Monkey patch the method
DigitalTwinTrainer.train_online = train_online


# Extension for OutcomeModelTrainer to support online updates
def train_step_online(self, batch: Dict) -> float:
    """Single online training step"""
    self.model.train()
    
    states = batch['state']
    actions = batch['action']
    rewards = batch['reward']
    
    # Forward
    predicted_rewards = self.model(states, actions).squeeze()
    
    # Loss
    loss = nn.functional.mse_loss(predicted_rewards, rewards)
    
    # Add regularization if needed
    reg_loss = self.model.compute_regularization_loss(states, actions)
    total_loss = loss + self.regularization_weight * reg_loss
    
    # Backward
    self.optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
    self.optimizer.step()
    
    return total_loss.item()


# Monkey patch the method
OutcomeModelTrainer.train_step_online = train_step_online


if __name__ == "__main__":
    print("Testing Online Training System...")
    
    # Create mock models
    from models import TransformerDynamicsModel, TreatmentOutcomeModel, EnsembleQNetwork
    
    state_dim = 10
    action_dim = 5
    
    models = {
        'dynamics_model': TransformerDynamicsModel(state_dim, action_dim),
        'outcome_model': TreatmentOutcomeModel(state_dim, action_dim),
        'q_ensemble': EnsembleQNetwork(state_dim, action_dim)
    }
    
    # Create online system
    system = create_online_training_system(models, stream_rate=2.0)
    
    # Start streaming
    system['stream'].start_stream()
    
    print("System running. Press Ctrl+C to stop...")
    
    try:
        # Run for a bit
        time.sleep(10)
        
        # Print statistics
        stats = system['trainer'].get_statistics()
        print("\nTraining Statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        
    except KeyboardInterrupt:
        print("\nStopping...")
    
    # Cleanup
    system['stream'].stop_stream()
    system['trainer'].stop()
    system['expert'].stop()
    
    print("Test complete!")