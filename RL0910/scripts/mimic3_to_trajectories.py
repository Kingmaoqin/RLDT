# scripts/mimic3_to_trajectories.py
# 将中间表 vitals.csv / labs.csv / inputs_fluid_vaso.csv 汇总为离线RL的轨迹CSV
# - 轨迹单位：ICUSTAY_ID
# - 时间步：按 CHARTTIME 排序后从 0 递增
# - 动作：0 无；1 仅补液；2 仅升压；3 同时有
# - 即时奖励：r_t = sofa_t - sofa_{t+1}；末步叠加终止奖励（生存+15 / 死亡-15，可调）
# - 终止：每条 ICU stay 的最后一步

import os
import argparse
import numpy as np
import pandas as pd
from typing import List

pd.options.mode.chained_assignment = None


def read_csv_auto(path: str, parse_dates: List[str] = None) -> pd.DataFrame:
    """支持 .csv 或 .csv.gz，若文件不存在则报错"""
    if not os.path.exists(path):
        # 兼容可能的 .gz
        gz = path + ".gz" if not path.endswith(".gz") else path[:-3]
        cand = gz if os.path.exists(gz) else None
        if cand is None:
            raise FileNotFoundError(f"文件不存在: {path}")
        path = cand
    return pd.read_csv(path, parse_dates=parse_dates, low_memory=False)


def compute_sofa_approx(row: pd.Series) -> float:
    """
    极简SOFA近似（仅作baseline）：
    - Cardiovascular: MAP 低则加分（表示严重）
    - Resp: SpO2 低则加分（先统一到0~1）
    - Renal: Creatinine 高则加分
    - Liver: Bilirubin 高则加分
    - Coag: Platelet 低则加分
    """
    score = 0.0
    # MAP
    mapv = row.get("MAP")
    if pd.notnull(mapv):
        if mapv < 70: score += 1
        if mapv < 60: score += 2
        if mapv < 50: score += 3
    # SpO2 scale to [0,1] if in percent
    spo2 = row.get("SPO2")
    if pd.notnull(spo2):
        s = float(spo2)
        if s > 1.5:  # 视作百分比
            s = s / 100.0
        if s < 0.92: score += 1
        if s < 0.88: score += 2
        if s < 0.85: score += 3
    # Creatinine
    cr = row.get("CREATININE")
    if pd.notnull(cr):
        if cr >= 1.2: score += 1
        if cr >= 2.0: score += 2
        if cr >= 3.5: score += 3
    # Bilirubin
    bili = row.get("BILIRUBIN")
    if pd.notnull(bili):
        if bili >= 1.2: score += 1
        if bili >= 2.0: score += 2
        if bili >= 6.0: score += 3
    # Platelet（越低越糟）
    plt = row.get("PLATELET")
    if pd.notnull(plt):
        if plt < 150: score += 1
        if plt < 100: score += 2
        if plt < 50:  score += 3
    return float(score)


def main():
    ap = argparse.ArgumentParser(description="Build trajectories CSV from MIMIC-III intermediate tables")
    ap.add_argument("--input_dir", required=True,
                    help="MIMIC-III v1.4 目录（包含 ICUSTAYS.csv(.gz), ADMISSIONS.csv(.gz)）")
    ap.add_argument("--output_dir", required=True, help="输出目录")
    ap.add_argument("--vitals_csv", type=str, default=None,
                    help="路径：vitals.csv（含 ICUSTAY_ID, CHARTTIME, HR, MAP, ...）")
    ap.add_argument("--labs_csv", type=str, default=None,
                    help="路径：labs.csv（含 HADM_ID, CHARTTIME, CREATININE, ...）")
    ap.add_argument("--inputs_csv", type=str, default=None,
                    help="路径：inputs_fluid_vaso.csv（含 ICUSTAY_ID, CHARTTIME, FLUID_ML, VASOPRESSOR_FLAG）")
    ap.add_argument("--max_icu", type=int, default=None,
                    help="仅输出前 N 个 ICU stays，用于冒烟")
    ap.add_argument("--terminal_bonus_survive", type=float, default=15.0,
                    help="终止奖励（生存）")
    ap.add_argument("--terminal_penalty_death", type=float, default=-15.0,
                    help="终止惩罚（死亡）")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- 路径推断 ---
    vitals_path = args.vitals_csv or os.path.join(args.input_dir, "vitals.csv")
    labs_path   = args.labs_csv   or os.path.join(args.input_dir, "labs.csv")
    inputs_path = args.inputs_csv or os.path.join(args.input_dir, "inputs_fluid_vaso.csv")

    print("[info] reading ICUSTAYS / ADMISSIONS ...")
    icu = read_csv_auto(os.path.join(args.input_dir, "ICUSTAYS.csv"), parse_dates=None)
    adm = read_csv_auto(os.path.join(args.input_dir, "ADMISSIONS.csv"), parse_dates=None)

    # 只保留用得到的列，统一类型
    icu = icu[["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"]].dropna()
    icu["ICUSTAY_ID"] = icu["ICUSTAY_ID"].astype(int)
    icu["HADM_ID"] = icu["HADM_ID"].astype(int)

    adm = adm[["HADM_ID", "HOSPITAL_EXPIRE_FLAG"]].dropna()
    adm["HADM_ID"] = adm["HADM_ID"].astype(int)
    adm["HOSPITAL_EXPIRE_FLAG"] = adm["HOSPITAL_EXPIRE_FLAG"].astype(int)

    # 固定子集（冒烟）
    if args.max_icu:
        keep_ids = icu["ICUSTAY_ID"].drop_duplicates().astype(int).head(args.max_icu).tolist()
        icu = icu[icu["ICUSTAY_ID"].isin(keep_ids)].copy()
        print(f"[info] restricting to first {len(keep_ids)} ICU stays")

    # --- 读取中间表 ---
    print("[info] reading vitals / labs / inputs ...")
    vit = read_csv_auto(vitals_path, parse_dates=["CHARTTIME"])
    lab = read_csv_auto(labs_path,   parse_dates=["CHARTTIME"])
    inp = read_csv_auto(inputs_path, parse_dates=["CHARTTIME"])

    # 统一列名大写（避免大小写差异）
    vit.columns = [c.upper() for c in vit.columns]
    lab.columns = [c.upper() for c in lab.columns]
    inp.columns = [c.upper() for c in inp.columns]

    # 仅保留当前子集 ICU（若有限制）
    if args.max_icu:
        vit = vit[vit["ICUSTAY_ID"].isin(icu["ICUSTAY_ID"])]
        inp = inp[inp["ICUSTAY_ID"].isin(icu["ICUSTAY_ID"])]

    # 将 HADM_ID 映射到 base（后面合并 labs 需要）
    hadm_map = icu.set_index("ICUSTAY_ID")["HADM_ID"].to_dict()

    # --- 构造 base 时间线（ICU 粒度）---
    # 将 vitals 和 inputs 的时间戳并集作为时间线
    base = pd.merge(
        vit[["ICUSTAY_ID", "CHARTTIME"]].drop_duplicates(),
        inp[["ICUSTAY_ID", "CHARTTIME"]].drop_duplicates(),
        on=["ICUSTAY_ID", "CHARTTIME"], how="outer"
    ).sort_values(["ICUSTAY_ID", "CHARTTIME"]).reset_index(drop=True)

    if args.max_icu:
        base = base[base["ICUSTAY_ID"].isin(icu["ICUSTAY_ID"])]

    # 将 vitals 特征并到 base（按 exact timestamp 连接；若存在时间不对齐，依然可通过 ffill 修复）
    vital_cols = [c for c in vit.columns if c not in ("ICUSTAY_ID", "CHARTTIME")]
    base = base.merge(vit, on=["ICUSTAY_ID", "CHARTTIME"], how="left")

    # 将 inputs（流体量/升压药）并到 base
    # 兼容历史列名：HAS_VASO vs VASOPRESSOR_FLAG
    if "VASOPRESSOR_FLAG" not in inp.columns and "HAS_VASO" in inp.columns:
        inp = inp.rename(columns={"HAS_VASO": "VASOPRESSOR_FLAG"})
    for col in ("FLUID_ML", "VASOPRESSOR_FLAG"):
        if col not in inp.columns:
            if col == "FLUID_ML":
                inp[col] = 0.0
            else:
                inp[col] = 0
    inp_part = inp[["ICUSTAY_ID", "CHARTTIME", "FLUID_ML", "VASOPRESSOR_FLAG"]]
    base = base.merge(inp_part, on=["ICUSTAY_ID", "CHARTTIME"], how="left")

    # 若仍有缺失：补 0
    base["FLUID_ML"] = base["FLUID_ML"].fillna(0.0)
    base["VASOPRESSOR_FLAG"] = base["VASOPRESSOR_FLAG"].fillna(0).astype(int)

    # 添加 HADM_ID（用于和 labs merge_asof）
    base["HADM_ID"] = base["ICUSTAY_ID"].map(hadm_map)

    # --- 将 labs 以 asof 方式合并到 base（同一 HADM 内按最近过去时间对齐，48h 容忍）---
    lab = lab[["HADM_ID", "CHARTTIME"] + [c for c in lab.columns if c not in ("HADM_ID", "CHARTTIME")]].copy()
    lab = lab.sort_values(["HADM_ID", "CHARTTIME"]).reset_index(drop=True)

    base = base.sort_values(["HADM_ID", "CHARTTIME"]).reset_index(drop=True)
    # merge_asof 需要每个组单调递增；我们对每个 HADM 分组处理
    merged_list = []
    tol = pd.Timedelta("48h")  # 容忍窗口
    lab_cols = [c for c in lab.columns if c not in ("HADM_ID", "CHARTTIME")]
    for hadm_id, g_base in base.groupby("HADM_ID", sort=False):
        gb = g_base.sort_values("CHARTTIME")
        gl = lab[lab["HADM_ID"] == hadm_id].sort_values("CHARTTIME")
        if gl.empty:
            merged_list.append(gb)
            continue
        mg = pd.merge_asof(gb, gl, on="CHARTTIME", direction="backward", tolerance=tol)
        merged_list.append(mg)
    base = pd.concat(merged_list, ignore_index=True)

    # --- 前向填充 & 全局补齐 ---
    # 需要的特征集合（尽量全）：
    desired_features = ["HR", "MAP", "SBP", "DBP", "RR", "TEMP", "SPO2",
                        "CREATININE", "BILIRUBIN", "PLATELET", "LACTATE", "FLUID_ML"]
    present = [c for c in desired_features if c in base.columns]
    missing = [c for c in desired_features if c not in base.columns]
    if missing:
        for c in missing:
            base[c] = np.nan  # 占位
        print(f"[warn] missing features (not present in any row): {missing}")

    # 组内时间排序 & 前向填充
    base = base.sort_values(["ICUSTAY_ID", "CHARTTIME"]).reset_index(drop=True)
    base[present] = base.groupby("ICUSTAY_ID")[present].ffill()

    # 为“整列缺失”的特征准备安全默认值（可按需调整，当前是合理的临床常见中值/基线）
    SAFE_DEFAULTS = {
        "HR": 80.0, "MAP": 75.0, "SBP": 120.0, "DBP": 70.0, "RR": 16.0, "TEMP": 37.0,
        "SPO2": 0.97, "CREATININE": 1.0, "BILIRUBIN": 0.8, "PLATELET": 220.0, "LACTATE": 1.5,
        "FLUID_ML": 0.0
    }

    # 分列补缺：整列NaN -> 默认值；否则 -> 全局中位数
    fully_missing_cols = []
    for c in desired_features:
        col = base[c]
        if col.isna().all():
            default_val = SAFE_DEFAULTS.get(c, 0.0)
            base[c] = default_val
            fully_missing_cols.append(c)
        elif col.isna().any():
            base[c] = col.fillna(col.median())

    if fully_missing_cols:
        print(f"[warn] fully-missing features filled with SAFE_DEFAULTS: {fully_missing_cols}")

    # --- 生成 timestep、动作、奖励、终止 ---
    # timestep：每个 ICU stay 内 0..T-1
    base["visit"] = base.groupby("ICUSTAY_ID").cumcount()

    # 动作（四类）
    base["HAS_FLUID"] = (base["FLUID_ML"].fillna(0.0) > 0.0).astype(int)
    base["HAS_VASO"] = base["VASOPRESSOR_FLAG"].fillna(0).astype(int)
    base["action_id"] = (base["HAS_FLUID"] * 1) + (base["HAS_VASO"] * 2)

    # 计算 SOFA 近似与即时奖励
    feat_for_sofa = ["MAP", "SPO2", "CREATININE", "BILIRUBIN", "PLATELET"]
    for f in feat_for_sofa:
        if f not in base.columns:
            base[f] = np.nan
    base["sofa_approx"] = base[feat_for_sofa].apply(
        lambda r: compute_sofa_approx(r), axis=1
    )

    # r_t = sofa_t - sofa_{t+1}
    base["reward"] = 0.0
    for icu_id, g in base.groupby("ICUSTAY_ID"):
        idx = g.index
        s = g["sofa_approx"].to_numpy(dtype=float)
        r = np.zeros_like(s, dtype=float)
        if len(s) > 1:
            r[:-1] = s[:-1] - s[1:]
        base.loc[idx, "reward"] = r

    # 终止奖励（按 HOSPITAL_EXPIRE_FLAG）
    icu_adm = icu.merge(adm, on="HADM_ID", how="left")
    hadm_to_expire = icu_adm.set_index("ICUSTAY_ID")["HOSPITAL_EXPIRE_FLAG"].fillna(0).astype(int).to_dict()
    base["terminal"] = 0
    last_idx_by_icu = base.groupby("ICUSTAY_ID")["visit"].transform("max")
    is_last = base["visit"] == last_idx_by_icu
    base.loc[is_last, "terminal"] = 1

    # 给末步叠加终止奖励
    for icu_id, g in base.groupby("ICUSTAY_ID"):
        last_index = g.index.max()
        expire = hadm_to_expire.get(int(icu_id), 0)
        if expire == 1:
            base.loc[last_index, "reward"] += args.terminal_penalty_death
        else:
            base.loc[last_index, "reward"] += args.terminal_bonus_survive

    # --- 输出 ---
    out = base.rename(columns={"ICUSTAY_ID": "subject_id"})
    # 输出列（若某些特征不存在，会被上面补出来）
    feature_cols = ["HR", "MAP", "SBP", "DBP", "RR", "TEMP", "SPO2",
                    "CREATININE", "BILIRUBIN", "PLATELET", "LACTATE", "FLUID_ML", "sofa_approx"]
    keep_cols = ["subject_id", "visit", "action_id", "reward", "terminal"] + feature_cols
    # 确保全部存在
    for c in keep_cols:
        if c not in out.columns:
            out[c] = np.nan if c not in ("subject_id", "visit", "action_id", "reward", "terminal") else 0
    out = out[keep_cols].sort_values(["subject_id", "visit"]).reset_index(drop=True)

    # 类型规范
    out["subject_id"] = out["subject_id"].astype(int)
    out["visit"] = out["visit"].astype(int)
    out["action_id"] = out["action_id"].astype(int)
    out["terminal"] = out["terminal"].astype(int)
    out["reward"] = out["reward"].astype(float)

    out_path = os.path.join(args.output_dir, "mimic3_trajectories.csv")
    out.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path} rows={len(out):,} icu_stays={out['subject_id'].nunique():,}")
    print("[hint] 对应的 YAML: configs/mimic3_tabular.yaml 里的 feature_cols 要与上面输出列一致")


if __name__ == "__main__":
    main()
