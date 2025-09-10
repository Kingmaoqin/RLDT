# -*- coding: utf-8 -*-
"""
evaluation_bcq_minipatch.py
Date: 2025-08-14

Runtime patch so that your existing evaluation.py uses BCQ decisions for policy evaluation
WITHOUT editing the big evaluation.py file.

How it works:
- Loads <model_dir>/best_bcq_policy.d3 via d3rlpy.load_learnable (with fallbacks).
- Wraps it as a simple "policy.act(state)->int".
- Monkey-patches ComprehensiveEvaluator.evaluate_policy(...) to temporarily replace
  `self.q_network` with an adapter that forces argmax to the BCQ-chosen action — only during
  policy simulation — and then restores the original q_network.
"""

import os
import numpy as np

def _load_bcq_policy(model_dir: str):
    path = os.path.join(model_dir, 'best_bcq_policy.d3')
    if not os.path.exists(path):
        print(f"[BCQ-PATCH] No BCQ file at {path}. Will keep original q_network actions.")
        return None
    # Lazy import d3rlpy only if necessary
    try:
        from d3rlpy import load_learnable
        algo = load_learnable(path)
    except Exception:
        try:
            from d3rlpy.algos import load_learnable as ll
            algo = ll(path)
        except Exception as e:
            print(f"[BCQ-PATCH] Failed to load BCQ policy: {e}")
            return None
    return algo

class _D3Policy:
    def __init__(self, algo): self.algo = algo
    def act(self, state: np.ndarray) -> int:
        x = np.asarray(state, dtype=np.float32).reshape(1, -1)
        out = self.algo.predict(x)
        a0 = out[0] if hasattr(out, '__len__') else out
        try: return int(a0)
        except Exception: return int(np.asarray(a0).item())

class _QFromPolicyAdapter:
    """A fake 'Q-network' whose argmax equals external policy's chosen action."""
    def __init__(self, policy, action_dim: int):
        self.policy = policy
        self.action_dim = int(action_dim)

    def __call__(self, state_batch_tensor):
        import torch
        if state_batch_tensor.dim() == 1:
            state_batch = state_batch_tensor.unsqueeze(0)
        else:
            state_batch = state_batch_tensor
        B = state_batch.shape[0]
        q = torch.zeros(B, self.action_dim, dtype=torch.float32, device=state_batch_tensor.device)
        for i in range(B):
            a = self.policy.act(state_batch[i].detach().cpu().numpy())
            a = max(0, min(self.action_dim - 1, int(a)))
            q[i, a] = 1.0  # put mass on the chosen action
        return q

def apply(evaluation_module, model_dir: str):
    """
    Patch evaluation.ComprehensiveEvaluator.evaluate_policy so that
    BCQ policy is used during policy simulation.
    """
    bcq_algo = _load_bcq_policy(model_dir)
    if bcq_algo is None:
        print("[BCQ-PATCH] Skipping patch: BCQ not found.")
        return False

    bcq_policy = _D3Policy(bcq_algo)

    CE = evaluation_module.ComprehensiveEvaluator
    original_eval_policy = CE.evaluate_policy

    def eval_policy_with_bcq(self, n_episodes: int = 100, max_horizon: int = 50):
        # Temporarily replace q_network with adapter
        original_q = self.q_network
        try:
            adapter = _QFromPolicyAdapter(bcq_policy, action_dim=getattr(self, 'action_dim', 5))
            self.q_network = adapter
            print("[BCQ-PATCH] Using BCQ policy for policy evaluation episodes.")
            return original_eval_policy(self, n_episodes=n_episodes, max_horizon=max_horizon)
        finally:
            self.q_network = original_q

    CE.evaluate_policy = eval_policy_with_bcq
    print("[BCQ-PATCH] Patched ComprehensiveEvaluator.evaluate_policy to use BCQ decisions.")
    return True
