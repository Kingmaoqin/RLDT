# -*- coding: utf-8 -*-
"""
train_offline_bcq_from_dt.py

Purpose
-------
Train a *d3rlpy-compatible* offline BCQ policy by generating an offline
dataset using your trained Dynamics (TransformerDynamicsModel) and
Outcome (TreatmentOutcomeModel). This follows the "digital twin + reward"
setup from your proposal and produces a proper `.d3` artifact for BCQ.

Output
------
- <model-dir>/best_bcq_policy.d3           (d3rlpy loadable)
- <model-dir>/reward_stats.json            (mean/std for eval)
- <model-dir>/offline_dataset_stats.json   (dataset meta)

Usage (example)
---------------
python train_offline_bcq_from_dt.py \
  --dynamics "/home/xqin5/RL_DT_MTE_OnlinewithLLM/outputs/main_seed42/models/best_dynamics_model.pth" \
  --outcome  "/home/xqin5/RL_DT_MTE_OnlinewithLLM/outputs/main_seed42/models/best_outcome_model.pth"  \
  --state-dim 10 --action-dim 5 \
  --episodes 2000 --horizon 30 \
  --model-dir ./output/models \
  --bcq-steps 50000 \
  --behavior epsilon_softmax --epsilon 0.2 --temperature 2.0

After training, verify it loads:
python - <<'PY'
from d3rlpy.base import load_learnable
load_learnable('./output/models/best_bcq_policy.d3')
print('BCQ .d3 load OK')
PY
"""
import argparse
import json
import os
from typing import Tuple

import numpy as np
import torch

from models import TransformerDynamicsModel, TreatmentOutcomeModel
from training_bcq_patched import train_rl_policy_bcq


def _device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@torch.no_grad()
def _load_model_flex(model: torch.nn.Module, path: str, name: str, device: str) -> None:
    print(f"[Load] {name} from: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found at {path}")
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        ckpt = ckpt["state_dict"]

    model_keys = set(model.state_dict().keys())
    matched, skipped = 0, 0
    new_sd = {}
    for k, v in ckpt.items():
        if k in model_keys and model.state_dict()[k].shape == v.shape:
            new_sd[k] = v
            matched += 1
        else:
            skipped += 1
    print(f"  -> matched: {matched}, skipped/mismatch: {skipped}")
    model.load_state_dict(new_sd, strict=False)
    model.to(device).eval()


@torch.no_grad()
def _predict_next_state(dyn: TransformerDynamicsModel,
                        s: np.ndarray, a: int, device: str) -> np.ndarray:
    # 组装 (B=1, T=1, state_dim) 和 (B=1, T=1) 的时间批
    s_t = torch.from_numpy(s.astype('float32')).view(1, 1, -1).to(device)
    a_t = torch.tensor([[a]], dtype=torch.long, device=device)

    # 你的工程里 Dynamics 用的是 predict_next_state(states, actions)
    if hasattr(dyn, 'predict_next_state'):
        ns = dyn.predict_next_state(s_t, a_t)
    else:
        # 少数实现可能直接 forward(states, actions)
        ns = dyn(s_t, a_t)

    # 兼容返回类型：可能是张量 / 元组 / 字典
    if isinstance(ns, (tuple, list)):
        ns = ns[0]
    if isinstance(ns, dict):
        ns = ns.get('next_state', ns.get('pred', ns))

    # 兼容形状：(1,1,state_dim) 或 (1,state_dim)
    if isinstance(ns, torch.Tensor) and ns.dim() == 3:
        ns = ns[:, -1, :]     # 取最后一个时间步
    return ns.squeeze(0).detach().cpu().numpy()



@torch.no_grad()
def _predict_reward(out: TreatmentOutcomeModel,
                    s: np.ndarray, a: int, device: str) -> float:
    st = torch.from_numpy(s.astype('float32')).view(1, -1).to(device)
    at = torch.tensor([a], dtype=torch.long, device=device)
    r  = out(st, at)                   # (1, 1)
    if isinstance(r, (tuple, list)):
        r = r[0]
    return float(r.view(-1)[0].detach().cpu().item())


def _sample_action_uniform(action_dim: int) -> int:
    return int(np.random.randint(0, action_dim))


@torch.no_grad()
def _sample_action_epsilon_softmax(out: TreatmentOutcomeModel, s: np.ndarray,
                                   action_dim: int, device: str,
                                   epsilon: float = 0.2, temperature: float = 2.0) -> int:
    if np.random.rand() < epsilon:
        return _sample_action_uniform(action_dim)

    st = torch.from_numpy(s.astype('float32')).view(1, -1).to(device)
    rewards = []
    for a in range(action_dim):
        at = torch.tensor([a], dtype=torch.long, device=device)
        r  = out(st, at)
        if isinstance(r, (tuple, list)):
            r = r[0]
        rewards.append(float(r.view(-1)[0].detach().cpu().item()))
    rewards = np.array(rewards, dtype=np.float32)

    # temperature softmax
    logits = rewards / max(1e-6, float(temperature))
    probs  = np.exp(logits - logits.max())
    probs /= probs.sum() + 1e-8
    return int(np.random.choice(action_dim, p=probs))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Train offline BCQ from Dynamics+Outcome (simulated dataset)")
    p.add_argument("--dynamics", required=True, help="Path to trained dynamics model .pth")
    p.add_argument("--outcome",  required=True, help="Path to trained outcome  model .pth")
    p.add_argument("--state-dim", type=int, required=True)
    p.add_argument("--action-dim", type=int, required=True)
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--horizon",  type=int, default=30)
    p.add_argument("--model-dir", default="./output/models")
    p.add_argument("--bcq-steps", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)

    # behavior policy
    p.add_argument("--behavior", choices=["epsilon_softmax", "uniform"], default="epsilon_softmax")
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--temperature", type=float, default=2.0)

    # initial state
    p.add_argument("--init-mean", type=float, default=0.5)
    p.add_argument("--init-std",  type=float, default=0.15)
    return p


def main():
    args = build_parser().parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = _device()
    print(f"Using device: {device}")

    # 1) Load models
    dyn = TransformerDynamicsModel(args.state_dim, args.action_dim)
    out = TreatmentOutcomeModel(args.state_dim, args.action_dim)
    _load_model_flex(dyn, args.dynamics, "Dynamics Model", device)
    _load_model_flex(out, args.outcome,  "Outcome  Model", device)

    # 2) Roll trajectories to build offline dataset
    N  = args.episodes * args.horizon
    S  = np.zeros((N, args.state_dim), dtype=np.float32)
    A  = np.zeros((N,), dtype=np.int64)
    R  = np.zeros((N,), dtype=np.float32)
    D  = np.zeros((N,), dtype=bool)

    def init_state():
        s0 = np.random.normal(args.init_mean, args.init_std, size=(args.state_dim,)).astype(np.float32)
        return np.clip(s0, 0.0, 1.0)

    idx = 0
    for ep in range(args.episodes):
        s = init_state()
        for t in range(args.horizon):
            # behavior action
            if args.behavior == "uniform":
                a = _sample_action_uniform(args.action_dim)
            else:
                a = _sample_action_epsilon_softmax(out, s, args.action_dim, device, args.epsilon, args.temperature)

            # reward & next state via learned models
            r  = _predict_reward(out, s, a, device)
            ns = _predict_next_state(dyn, s, a, device)

            # write transition
            S[idx] = s
            A[idx] = a
            R[idx] = r
            D[idx] = (t == args.horizon - 1)  # terminal at episode end
            idx += 1

            s = ns

    S = S[:idx]; A = A[:idx]; R = R[:idx]; D = D[:idx]
    print(f"[Dataset] {len(S)} transitions, states {S.shape}, actions {A.shape}")

    # 3) Save stats
    os.makedirs(args.model_dir, exist_ok=True)
    reward_stats = {
        "mean": float(R.mean()),
        "std":  float(R.std() + 1e-6),
        "state_dim": int(args.state_dim),
        "action_dim": int(args.action_dim)
    }
    with open(os.path.join(args.model_dir, "reward_stats.json"), "w") as f:
        json.dump(reward_stats, f, indent=2)
    with open(os.path.join(args.model_dir, "offline_dataset_stats.json"), "w") as f:
        json.dump({"N": int(len(S)), "episodes": int(args.episodes), "horizon": int(args.horizon)}, f, indent=2)
    print("[Write] reward_stats.json, offline_dataset_stats.json")

    # 4) Train BCQ and save
    dataset = dict(states=S, actions=A, rewards=R, terminals=D)
    out_path = train_rl_policy_bcq(dataset, model_dir=args.model_dir, n_steps=args.bcq_steps, seed=args.seed)
    print(f"[Done] BCQ saved to: {out_path}")


if __name__ == "__main__":
    main()
