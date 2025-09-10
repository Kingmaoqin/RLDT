# -*- coding: utf-8 -*-
"""
training_bcq_patched.py
Date: 2025-08-14

Adds a BCQ-based offline RL training entry point: `train_rl_policy_bcq(...)` using d3rlpy.
- Converts our dataset dict to d3rlpy.MDPDataset (discrete actions assumed).
- Trains DiscreteBCQ with runtime-compatible fit() signature.
- Saves to <model_dir>/best_bcq_policy.d3 (compatible with d3rlpy.load_learnable).
"""

from typing import Dict
import os
import numpy as np

def _to_mdp_dataset(ds: Dict[str, np.ndarray]):
    """Dict -> d3rlpy.dataset.MDPDataset (robust to action shape differences)."""
    from d3rlpy.dataset import MDPDataset
    obs = ds['states'].astype(np.float32)
    rew = ds['rewards'].astype(np.float32)
    ter = ds['terminals'].astype(np.float32)
    actions = ds['actions']
    try:
        act = actions.astype(np.int64).reshape(-1)
        return MDPDataset(observations=obs, actions=act, rewards=rew, terminals=ter)
    except Exception:
        act = actions.astype(np.int64).reshape(-1, 1).astype(np.float32)
        return MDPDataset(observations=obs, actions=act, rewards=rew, terminals=ter)

def _build_bcq_discrete(state_dim: int, action_size: int, seed: int = 42, use_gpu: bool = False):
    """Version-agnostic BCQ builder (do NOT pass shapes to constructors)."""
    algo = None; how = None
    # Try config API first (without shapes)
    try:
        from d3rlpy.algos.qlearning.bcq import DiscreteBCQConfig
        cfg = DiscreteBCQConfig()
        try:
            algo = cfg.create()
            how = "d3rlpy.algos.qlearning.bcq.DiscreteBCQConfig.create()"
        except TypeError:
            # some builds require device kw
            device = 'cuda' if use_gpu else 'cpu'
            algo = cfg.create(device=device)
            how = "d3rlpy.algos.qlearning.bcq.DiscreteBCQConfig.create(device=...)"
    except Exception:
        pass
    # Direct class (no args)
    if algo is None:
        try:
            from d3rlpy.algos.qlearning.bcq import DiscreteBCQ
            algo = DiscreteBCQ()
            how = "d3rlpy.algos.qlearning.bcq.DiscreteBCQ()"
        except Exception:
            try:
                from d3rlpy.algos import DiscreteBCQ as DiscreteBCQTop
                algo = DiscreteBCQTop()
                how = "d3rlpy.algos.DiscreteBCQ() (top-level)"
            except Exception as e:
                raise RuntimeError(f"Failed to construct DiscreteBCQ: {e}")
    if hasattr(algo, 'set_seed'):
        try:
            algo.set_seed(seed)
        except Exception:
            pass
    return algo, how

def _fit_algo_compat(algo, dataset, n_steps: int = 20000):
    """Handle fit() signature differences across d3rlpy versions."""
    try:
        return algo.fit(dataset, n_steps=n_steps, n_steps_per_epoch=1000)
    except TypeError:
        try:
            return algo.fit(dataset, n_steps=n_steps)
        except TypeError:
            epochs = max(1, n_steps // 1000)
            return algo.fit(dataset, epochs=epochs)

def train_rl_policy_bcq(dataset: Dict[str, np.ndarray], model_dir: str, seed: int = 42, n_steps: int = 20000) -> str:
    """
    Train DiscreteBCQ from d3rlpy and save model.
    Returns path to <model_dir>/best_bcq_policy.d3
    """
    os.makedirs(model_dir, exist_ok=True)
    state_dim = int(dataset['states'].shape[1])
    action_size = int(dataset['actions'].max()) + 1
    mdp = _to_mdp_dataset(dataset)
    algo, how = _build_bcq_discrete(state_dim, action_size, seed=seed)
    print(f"[BCQ] Built via: {how}")
    _fit_algo_compat(algo, mdp, n_steps=n_steps)
    out_path = os.path.join(model_dir, 'best_bcq_policy.d3')
    # 必须保存“learnable 对象”才能被 d3rlpy.base.load_learnable 读取
    algo.save(out_path)  # 这会用 pickle 存整个算法对象
    print(f"[BCQ] Saved (learnable) to: {out_path}")
    return out_path
