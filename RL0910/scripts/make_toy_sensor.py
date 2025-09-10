# scripts/make_toy_sensor.py
import numpy as np, pandas as pd

rng = np.random.default_rng(1)
rows = []
for subj in range(3):
    T = 400
    acc_x = rng.normal(0, 0.8, size=T)
    acc_y = rng.normal(0, 0.8, size=T)
    acc_z = rng.normal(0, 1.2, size=T)
    # 伪标签：0=rest,1=walk,2=run
    activity = np.zeros(T, dtype=object)
    activity[100:250] = "walk"
    activity[250:350] = "run"
    # 奖励表达式将惩罚加速度幅度（越平稳越好）
    for t in range(T):
        rows.append(dict(
            subject=f"S{subj}", frame=t,
            acc_x=acc_x[t], acc_y=acc_y[t], acc_z=acc_z[t],
            activity=activity[t]
        ))
df = pd.DataFrame(rows)
df.to_csv("./data/toy_sensor.csv", index=False)
print("[OK] wrote ./data/toy_sensor.csv with", len(df), "rows")
