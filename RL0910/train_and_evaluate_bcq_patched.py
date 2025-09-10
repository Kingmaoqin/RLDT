# -*- coding: utf-8 -*-
"""
train_and_evaluate_bcq_patched.py
Date: 2025-08-14

Switches RL stage to BCQ training (using training_bcq_patched.train_rl_policy_bcq).
- Keeps your Phase-1 dynamics/outcome pipeline unchanged (assumes models are already trained),
  but still writes reward_stats.json for evaluation normalization.
"""

import os, json
import numpy as np

from data import PatientDataGenerator
from training_bcq_patched import train_rl_policy_bcq

def _ensure_dir(d): os.makedirs(d, exist_ok=True)

from schema import SchemaSpec
from data_manager import load_user_dataset

def main(output_dir: str = './output', data_path: str = None, schema_yaml: str = None):
    model_dir = os.path.join(output_dir, 'models')
    _ensure_dir(model_dir)

    # === NEW: 从 YAML 读取 SchemaSpec（如未提供，用默认 tabular 自动推断） ===
    if schema_yaml and os.path.exists(schema_yaml):
        import yaml
        with open(schema_yaml, "r") as f:
            spec_dict = yaml.safe_load(f)
        spec = SchemaSpec(**spec_dict)
    else:
        spec = SchemaSpec(data_type="tabular")  # 自动推断列映射

    if data_path is None:
        raise ValueError("请提供 --data_path 指向用户数据文件")

    dataset, meta = load_user_dataset(
        data_path=data_path,
        schema=spec,
        save_meta_to=os.path.join(model_dir, 'dataset_meta.json'),
        fit_normalization=True
    )

    states  = np.asarray(dataset['states'],  np.float32)
    actions = np.asarray(dataset['actions'], np.int64)
    rewards = np.asarray(dataset['rewards'], np.float32)
    terms   = np.asarray(dataset['terminals'], np.float32)

    # 保存 reward 统计与维度信息（评估/推理会用到）
    stats = dict(
        mean=float(rewards.mean()), std=float(rewards.std() + 1e-6),
        state_dim=int(states.shape[1]), action_dim=int(actions.max() + 1)
    )
    with open(os.path.join(model_dir, 'reward_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    # === Phase 3: offline RL (BCQ) ===
    mdp_dict = dict(states=states, actions=actions, rewards=rewards, terminals=terms)
    bcq_path = train_rl_policy_bcq(mdp_dict, model_dir=model_dir, seed=42, n_steps=20000)
    print(f"BCQ trained and saved to {bcq_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train BCQ with user dataset")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Where to write models and artifacts")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to user dataset file (csv/parquet/json/h5/npz/zip)")
    parser.add_argument("--schema_yaml", type=str, default=None,
                        help="YAML file describing schema/adapters (optional)")

    args = parser.parse_args()

    main(
        output_dir=args.output_dir,
        data_path=args.data_path,
        schema_yaml=args.schema_yaml
    )
