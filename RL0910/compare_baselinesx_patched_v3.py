
# -*- coding: utf-8 -*-
"""
compare_baselinesx_patched_v3.py
v3 changes (multi-seed evaluation & aggregation):
- New arg: --seeds "0,1,2,3,4" (default five seeds)
- set_global_seed(seed) for reproducibility
- run_once(seed, args): executes a full evaluation pipeline for a given seed
- main(): loops over seeds, saves per-seed results and aggregated results with 95% CI
Other logic (dataset, simulator, d3rlpy builders) kept from v2.
"""

import os, sys, json, argparse, inspect
from typing import Dict, List
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)

from data import PatientDataGenerator
from models import TransformerDynamicsModel, TreatmentOutcomeModel, ConservativeQNetwork

def _flex_sd(ckpt):
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt: sd = ckpt['state_dict']
        elif 'model_state_dict' in ckpt: sd = ckpt['model_state_dict']
        else: sd = ckpt
    else:
        sd = ckpt
    return { (k[7:] if k.startswith('module.') else k): v for k, v in sd.items() }

def load_dyn(path, sdim, adim, device):
    m = TransformerDynamicsModel(sdim, adim).to(device)
    m.load_state_dict(_flex_sd(torch.load(path, map_location=device)), strict=False)
    m.eval(); return m

def load_out(path, sdim, adim, device):
    m = TreatmentOutcomeModel(sdim, adim).to(device)
    m.load_state_dict(_flex_sd(torch.load(path, map_location=device)), strict=False)
    m.eval(); return m

def load_q(path, sdim, adim, device):
    q = ConservativeQNetwork(sdim, adim).to(device)
    q.load_state_dict(_flex_sd(torch.load(path, map_location=device)), strict=False)
    q.eval(); return q

class QPolicy:
    def __init__(self, q, device): self.q, self.device = q, device
    def act(self, s: np.ndarray)->int:
        with torch.no_grad():
            return int(self.q(torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)).argmax(1).item())

class D3Policy:
    def __init__(self, algo): self.algo = algo
    def act(self, s: np.ndarray) -> int:
        x = np.asarray(s, dtype=np.float32).reshape(1, -1)
        out = self.algo.predict(x)
        a0 = out[0] if hasattr(out, '__len__') else out
        try: return int(a0)
        except Exception: return int(np.asarray(a0).item())

def build_dataset(n_patients=100, seed=42):
    gen = PatientDataGenerator(n_patients=n_patients, seed=seed)
    d = gen.generate_dataset()
    states = np.asarray(d['states'], np.float32)
    actions = np.asarray(d['actions'], np.int64)
    rewards = np.asarray(d['rewards'], np.float32)
    next_states = np.asarray(d['next_states'], np.float32)
    traj = np.asarray(d['trajectory_ids'], np.int64)
    terminals = np.zeros_like(rewards, np.float32)
    terminals[:-1] = (traj[1:] != traj[:-1]).astype(np.float32)
    terminals[-1] = 1.0
    return dict(states=states, actions=actions, rewards=rewards, next_states=next_states,
                terminals=terminals, state_dim=states.shape[1],
                action_dim=int(actions.max()+1), transitions=len(states))

def to_mdp(ds):
    from d3rlpy.dataset import MDPDataset
    actions = ds['actions'].astype(np.int64).reshape(-1)
    try:
        return MDPDataset(observations=ds['states'].astype(np.float32),
                          actions=actions,
                          rewards=ds['rewards'].astype(np.float32),
                          terminals=ds['terminals'].astype(np.float32))
    except Exception:
        return MDPDataset(observations=ds['states'].astype(np.float32),
                          actions=actions.reshape(-1,1),
                          rewards=ds['rewards'].astype(np.float32),
                          terminals=ds['terminals'].astype(np.float32))

class BehaviorClf(nn.Module):
    def __init__(self, sdim, adim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sdim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, adim),
        )
    def forward(self, x): return self.net(x)
    @torch.no_grad()
    def predict_proba(self, x: np.ndarray, device: str) -> np.ndarray:
        t = torch.tensor(x, dtype=torch.float32, device=device)
        p = torch.softmax(self.net(t), dim=-1)
        return p.detach().cpu().numpy()

def train_behavior_clf(ds: dict, device='cpu', epochs=5, lr=1e-3):
    m = BehaviorClf(ds['state_dim'], ds['action_dim']).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    S = torch.tensor(ds['states'], dtype=torch.float32, device=device)
    A = torch.tensor(ds['actions'], dtype=torch.long, device=device)
    n = S.shape[0]; bs = 256
    m.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            loss = ce(m(S[idx]), A[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); return m

def simulate(policy,
             dynamics_models,
             outcome_model,
             test_states,
             reward_mean,
             reward_std,
             use_normalized_reward=True,
             episodes=200,
             horizon=50,
             uncertainty_penalty_weight=0.5,
             gamma=0.99,
             max_history=10,
             device='cpu',
             behavior_clf=None):
    rng = np.random.default_rng(0)
    rets = []; low_spo2 = 0
    action_counts = np.zeros(dynamics_models[0].action_embedding.num_embeddings, dtype=np.int64)
    uncert_log = []; beta_prob_log = []
    for _ in range(episodes):
        idx = rng.integers(0, len(test_states))
        state = torch.tensor(test_states[idx], dtype=torch.float32, device=device)
        s_hist=[state]; a_hist=[]; ep=0.0
        for t in range(horizon):
            a = policy.act(state.detach().cpu().numpy()); action_counts[a]+=1
            a_t = torch.tensor([a], device=device, dtype=torch.long)
            with torch.no_grad():
                raw = outcome_model(state.unsqueeze(0), a_t).item()
            r = (raw - reward_mean)/max(reward_std,1e-6) if use_normalized_reward else raw

            a_hist.append(a)
            sseq = torch.stack(s_hist).unsqueeze(0)
            aseq = torch.tensor(a_hist, device=device, dtype=torch.long).unsqueeze(0)
            preds=[]
            with torch.no_grad():
                for m in dynamics_models:
                    p = m.predict_next_state(sseq, aseq)
                    if p is not None: preds.append(p)
            if preds:
                stk = torch.stack([p.squeeze(0) if p.dim()==2 else p for p in preds],0)
                mean_next = stk.mean(0).squeeze(0); std_next = stk.std(0).squeeze(0)
                nxt = torch.clamp(mean_next, 0.0, 1.0); uncert=float(std_next.mean().item())
            else:
                nxt = torch.clamp(state + torch.randn_like(state)*0.01, 0.0, 1.0); uncert=0.0
            uncert_log.append(uncert)

            if behavior_clf is not None:
                prob = behavior_clf.predict_proba(state.detach().cpu().numpy().reshape(1,-1), device=device)[0]
                beta_prob_log.append(float(prob[a]))

            ep += (r - uncertainty_penalty_weight*uncert) * (gamma**t)
            state = nxt; s_hist.append(state)
            if len(s_hist)>max_history: s_hist.pop(0); a_hist.pop(0)
            if nxt[8] < 0.80: low_spo2 += 1; break
        rets.append(ep)

    rets = np.array(rets, np.float32)
    mean=float(rets.mean()); std=float(rets.std())
    boot = np.random.default_rng(1).choice(rets, size=(1000, len(rets)), replace=True).mean(1)
    ci_low=float(np.percentile(boot,2.5)); ci_high=float(np.percentile(boot,97.5))
    q25,q50,q75 = np.percentile(rets,[25,50,75])
    safety_rate=float(1.0 - low_spo2/max(episodes,1))
    probs = action_counts / max(action_counts.sum(),1)
    entropy = float(-(probs*np.log(probs+1e-12)).sum())

    res = dict(mean_return=mean, ci_low=ci_low, ci_high=ci_high, std_return=std,
               safety_rate=safety_rate, action_entropy=entropy,
               return_median=float(q50), return_p25=float(q25), return_p75=float(q75),
               return_min=float(rets.min()), return_max=float(rets.max()),
               sharpe_like=float(mean/(std+1e-6)))
    if len(uncert_log)>0:
        ua = np.asarray(uncert_log, np.float32)
        res.update(uncertainty_mean=float(ua.mean()), uncertainty_p95=float(np.percentile(ua,95)))
    if len(beta_prob_log)>0:
        ba = np.asarray(beta_prob_log, np.float32)
        res.update(nll_behavior=float(-np.log(ba+1e-12).mean()),
                   **{f'coverage@{thr:.2f}': float((ba>=thr).mean()) for thr in (0.05,0.10,0.20)})
    return res

def build_discrete_algo(name: str, use_gpu: bool, seed: int):
    import importlib, inspect
    NAME = name.upper()
    DEVICE = 'cuda' if use_gpu else 'cpu'

    CANDIDATES = {
        'BC': [
            'd3rlpy.algos.classic.DiscreteBC',
            'd3rlpy.algos.bc.DiscreteBC',
            'd3rlpy.algos.classic.DiscreteBCConfig',
            'd3rlpy.algos.bc.DiscreteBCConfig',
            'd3rlpy.algos.classic.BCConfig',
            'd3rlpy.algos.bc.BCConfig',
            'd3rlpy.algos.classic.BC',
            'd3rlpy.algos.bc.BC',
            'd3rlpy.algos.BC',
        ],
        'NFQ': [
            'd3rlpy.algos.nfq.NFQ',
            'd3rlpy.algos.nfq.NFQConfig',
            'd3rlpy.algos.NFQ',
            'd3rlpy.algos.NFQConfig',
        ],
        'DQN': [
            'd3rlpy.algos.qlearning.dqn.DQN',
            'd3rlpy.algos.qlearning.dqn.DQNConfig',
            'd3rlpy.algos.dqn.DQN',
            'd3rlpy.algos.dqn.DQNConfig',
            'd3rlpy.algos.DQN',
            'd3rlpy.algos.DQNConfig',
        ],
        'DOUBLEDQN': [
            'd3rlpy.algos.qlearning.doubledqn.DoubleDQN',
            'd3rlpy.algos.qlearning.doubledqn.DoubleDQNConfig',
            'd3rlpy.algos.doubledqn.DoubleDQN',
            'd3rlpy.algos.doubledqn.DoubleDQNConfig',
            'd3rlpy.algos.DoubleDQN',
            'd3rlpy.algos.DoubleDQNConfig',
        ],
        'IQN': [
            'd3rlpy.algos.qlearning.iqn.IQN',
            'd3rlpy.algos.qlearning.iqn.IQNConfig',
            'd3rlpy.algos.iqn.IQN',
            'd3rlpy.algos.iqn.IQNConfig',
            'd3rlpy.algos.IQN',
            'd3rlpy.algos.IQNConfig',
        ],
        'CATEGORICALDQN': [
            'd3rlpy.algos.qlearning.categorical_dqn.CategoricalDQN',
            'd3rlpy.algos.qlearning.categorical_dqn.CategoricalDQNConfig',
            'd3rlpy.algos.categorical_dqn.CategoricalDQN',
            'd3rlpy.algos.categorical_dqn.CategoricalDQNConfig',
            'd3rlpy.algos.CategoricalDQN',
            'd3rlpy.algos.CategoricalDQNConfig',
        ],
        'RAINBOW': [
            'd3rlpy.algos.qlearning.rainbow.RainbowDQN',
            'd3rlpy.algos.qlearning.rainbow.RainbowDQNConfig',
            'd3rlpy.algos.rainbow.RainbowDQN',
            'd3rlpy.algos.rainbow.RainbowDQNConfig',
            'd3rlpy.algos.RainbowDQN',
            'd3rlpy.algos.RainbowDQNConfig',
        ],
        'BCQ': [
            'd3rlpy.algos.qlearning.bcq.DiscreteBCQ',
            'd3rlpy.algos.bcq.DiscreteBCQ',
            'd3rlpy.algos.qlearning.bcq.DiscreteBCQConfig',
            'd3rlpy.algos.bcq.DiscreteBCQConfig',
            'd3rlpy.algos.qlearning.bcq.BCQConfig',
            'd3rlpy.algos.bcq.BCQConfig',
            'd3rlpy.algos.qlearning.bcq.BCQ',
            'd3rlpy.algos.bcq.BCQ',
            'd3rlpy.algos.BCQ',
            'd3rlpy.algos.BCQConfig',
        ],
        'CQL': [
            'd3rlpy.algos.qlearning.cql.DiscreteCQL',
            'd3rlpy.algos.cql.DiscreteCQL',
            'd3rlpy.algos.qlearning.cql.DiscreteCQLConfig',
            'd3rlpy.algos.cql.DiscreteCQLConfig',
            'd3rlpy.algos.qlearning.cql.CQLConfig',
            'd3rlpy.algos.cql.CQLConfig',
            'd3rlpy.algos.qlearning.cql.CQL',
            'd3rlpy.algos.cql.CQL',
            'd3rlpy.algos.CQL',
            'd3rlpy.algos.CQLConfig',
        ],
    }

    def _instantiate_class(cls):
        sig = inspect.signature(cls); kw={}
        if 'use_gpu' in sig.parameters: kw['use_gpu']=use_gpu
        if 'seed' in sig.parameters: kw['seed']=seed
        if 'action_space' in sig.parameters: kw['action_space']='discrete'
        return cls(**kw)

    def _create_from_config(cfg_cls):
        sig = inspect.signature(cfg_cls); ckw={}
        if 'action_space' in sig.parameters: ckw['action_space']='discrete'
        cfg = cfg_cls(**ckw) if ckw else cfg_cls()
        create_sig = inspect.signature(cfg.create); akw={}
        if 'device' in create_sig.parameters: akw['device']=DEVICE
        return cfg.create(**akw)

    algo, how = None, ''
    for target in CANDIDATES.get(NAME, []):
        try:
            mp, cn = target.rsplit('.',1)
            mod = __import__(mp, fromlist=[cn])
            cls = getattr(mod, cn)
            cand = _create_from_config(cls) if 'Config' in cn else _instantiate_class(cls)
            algo, how = cand, target
            break
        except Exception:
            continue
    return algo, how

def fit_algo(algo, dataset, n_steps=20000):
    sig = inspect.signature(algo.fit); kw={}
    if 'n_steps' in sig.parameters: kw['n_steps']=n_steps
    if 'n_steps_per_epoch' in sig.parameters: kw['n_steps_per_epoch']=1000
    if 'save_interval' in sig.parameters: kw['save_interval']=n_steps+1
    if 'scorers' in sig.parameters: kw['scorers']={}
    algo.fit(dataset, **kw)

# --------------------- NEW: seeding + per-seed run + aggregation ---------------------

def set_global_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # for better reproducibility (may slow down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_once(seed: int, args) -> List[dict]:
    set_global_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # dataset & stats
    ds = build_dataset(args.n_patients, seed)
    stats_path = os.path.join(args.model_dir, 'reward_stats.json')
    if os.path.exists(stats_path):
        st = json.load(open(stats_path))
        reward_mean, reward_std = float(st.get('mean', 0.0)), float(st.get('std', 1.0))
        use_norm = True
    else:
        reward_mean = float(ds['rewards'].mean())
        reward_std  = float(ds['rewards'].std() + 1e-6)
        use_norm = True

    # models
    dynamics_models = []
    for i in range(5):
        p = os.path.join(args.model_dir, f'dynamics_model_{i}.pth')
        if os.path.exists(p):
            dynamics_models.append(load_dyn(p, ds['state_dim'], ds['action_dim'], device))
    if not dynamics_models:
        p = os.path.join(args.model_dir, 'best_dynamics_model.pth')
        if os.path.exists(p):
            m = load_dyn(p, ds['state_dim'], ds['action_dim'], device)
            dynamics_models = [m]*5
        else:
            dynamics_models = [TransformerDynamicsModel(ds['state_dim'], ds['action_dim']).to(device).eval() for _ in range(5)]
    outcome_model = load_out(os.path.join(args.model_dir, 'best_outcome_model.pth'), ds['state_dim'], ds['action_dim'], device)

    test_states = ds['states']
    behavior_clf = train_behavior_clf(ds, device=device)

    rows: List[dict] = []

    # your policy
    q_path = os.path.join(args.model_dir, 'best_q_network.pth')
    if os.path.exists(q_path):
        pol = QPolicy(load_q(q_path, ds['state_dim'], ds['action_dim'], device), device)
        r = simulate(pol, dynamics_models, outcome_model, test_states, reward_mean, reward_std, use_norm,
                     args.episodes, args.horizon, uncertainty_penalty_weight=0.5, gamma=0.99,
                     max_history=10, device=device, behavior_clf=behavior_clf)
        r['algo'] = 'Your CQL (custom)'
        r['seed'] = seed
        rows.append(r)

    # baselines (train-from-scratch per seed)
    if args.train_baselines:
        try:
            import d3rlpy  # noqa
            mdp = to_mdp(ds)
            names = [x.strip() for x in args.baselines.split(',') if x.strip()]
            for n in names:
                try:
                    algo, how = build_discrete_algo(n, torch.cuda.is_available(), seed)
                    if algo is None:
                        print(f"[SKIP][seed={seed}] Could not construct a {n.upper()} for your d3rlpy version.")
                        continue
                    if hasattr(algo, 'set_seed'):
                        try: algo.set_seed(seed)
                        except Exception: pass
                    print(f"[INFO][seed={seed}] Built baseline {n.upper()} via {how}")
                    fit_algo(algo, mdp, n_steps=args.n_steps)
                    pol = D3Policy(algo)
                    r = simulate(pol, dynamics_models, outcome_model, test_states, reward_mean, reward_std, use_norm,
                                 args.episodes, args.horizon, uncertainty_penalty_weight=0.5, gamma=0.99,
                                 max_history=10, device=device, behavior_clf=behavior_clf)
                    r['algo'] = f'd3rlpy-{n.upper()}'
                    r['seed'] = seed
                    rows.append(r)
                except Exception as e:
                    print(f"[SKIP][seed={seed}] {n.upper()} evaluation failed: {e}")
                    continue
        except Exception as e:
            print(f"[SKIP][seed={seed}] d3rlpy baselines failed entirely: {e}")

    return rows

# ------------------------------------ main ------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', type=str, default='./output/models')
    ap.add_argument('--output_dir', type=str, default='./comparison_results')
    ap.add_argument('--episodes', type=int, default=200)
    ap.add_argument('--horizon', type=int, default=50)
    ap.add_argument('--n_patients', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)  # kept for backward-compat (single run)
    ap.add_argument('--baselines', type=str, default='bc,nfq,cql,bcq,dqn,doubledqn,iqn,categoricaldqn,rainbow')
    ap.add_argument('--train_baselines', action='store_true')
    ap.add_argument('--n_steps', type=int, default=20000)
    ap.add_argument('--seeds', type=str, default='0,1,2,3,4',
                    help='comma-separated random seeds for repeated runs (e.g., "0,1,2,3,4")')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # parse seeds and loop
    seed_list = [int(s) for s in args.seeds.split(',') if s.strip()]
    print(f"[INFO] Running seeds: {seed_list}")

    per_seed_rows: List[dict] = []
    for s in seed_list:
        rows = run_once(s, args)
        per_seed_rows.extend(rows)

    if not per_seed_rows:
        print("[ERROR] No results to report."); return

    # save per-seed
    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_csv = os.path.join(args.output_dir, 'baseline_comparison_by_seed.csv')
    per_seed_df.to_csv(per_seed_csv, index=False)
    print(f"Saved per-seed table to: {per_seed_csv}")

    # aggregate across seeds
    def _ci95(x: np.ndarray):
        if len(x) == 0: return (np.nan, np.nan)
        rng = np.random.default_rng(0)
        boots = rng.choice(x, size=(10000, len(x)), replace=True).mean(axis=1)
        return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    agg_cols_mean = [
        'mean_return','std_return',
        'return_median','return_p25','return_p75','return_min','return_max','sharpe_like',
        'safety_rate','action_entropy',
        'nll_behavior','coverage@0.05','coverage@0.10','coverage@0.20',
        'uncertainty_mean','uncertainty_p95',
    ]

    grouped = per_seed_df.groupby('algo', dropna=False)
    summary = grouped[agg_cols_mean].mean(numeric_only=True).reset_index()

    cis = grouped['mean_return'].apply(lambda s: pd.Series(_ci95(s.to_numpy()), index=['ci_low','ci_high']))
    summary = summary.merge(cis, on='algo', how='left')

    agg_csv = os.path.join(args.output_dir, 'baseline_comparison_agg.csv')
    summary.to_csv(agg_csv, index=False)
    print(f"Saved aggregated table to: {agg_csv}")

    # plot aggregated
    plt.figure(figsize=(11,6))
    xs = np.arange(len(summary)); means = summary['mean_return'].values
    yerr = np.vstack([means-summary['ci_low'].values, summary['ci_high'].values-means])
    plt.bar(xs, means)
    plt.errorbar(xs, means, yerr=yerr, fmt='none', capsize=5)
    plt.xticks(xs, summary['algo'].values, rotation=20, ha='right')
    plt.ylabel('Mean Return across seeds (95% CI)')
    plt.title('Policy Comparison — aggregated over seeds')
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, 'baseline_comparison_agg.png')
    plt.savefig(plot_path, dpi=180)
    print(f"Saved aggregated plot to: {plot_path}")

if __name__ == '__main__':
    main()
