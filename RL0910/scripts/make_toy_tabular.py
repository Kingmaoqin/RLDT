# scripts/make_toy_tabular.py
import numpy as np, pandas as pd

rng = np.random.default_rng(0)
rows = []
for traj in range(5):            # 5条轨迹
    T = rng.integers(15, 25)     # 每条轨迹15~24步
    risk = rng.uniform(0.2, 0.8, size=T)
    sbp  = rng.normal(120, 15, size=T)
    labA = rng.normal(0.0, 1.0, size=T)
    labB = rng.normal(5.0, 2.0, size=T)

    # 动作：0/1/2（三分类）
    action = rng.integers(0, 3, size=T)

    # 奖励：表达式 1.0 - 2*risk - 0.01*sbp（越低risk越好、越低sbp越好）
    reward = 1.0 - 2.0*risk - 0.01*sbp

    for t in range(T):
        rows.append(dict(
            subject_id=traj, visit=t, treatment=action[t],
            risk=risk[t], sbp=sbp[t], labA=labA[t], labB=labB[t],
            outcome=reward[t], is_last=(t==T-1)
        ))
df = pd.DataFrame(rows)
df.to_csv("./data/toy_tabular.csv", index=False)
print("[OK] wrote ./data/toy_tabular.csv with", len(df), "rows")
