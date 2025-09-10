# scripts/mimic3_prepare_lightweight.py
import os, re, argparse, pandas as pd
from pandas import DataFrame
pd.options.mode.chained_assignment = None

def _gz(path): return path if path.endswith(".gz") or path.endswith(".csv") else path + ".gz"

def _load_items_chartevents(d_items_path: str) -> DataFrame:
    di = pd.read_csv(_gz(d_items_path), low_memory=False)
    di.columns = [c.upper() for c in di.columns]
    di = di[di["LINKSTO"].str.lower() == "chartevents"]
    keep = [c for c in ["ITEMID", "LABEL"] if c in di.columns]
    return di[keep]


def _load_labitems(d_labitems_path: str) -> DataFrame:
    dl = pd.read_csv(_gz(d_labitems_path), low_memory=False)
    dl.columns = [c.upper() for c in dl.columns]
    # 只保留确实存在的列；后续只需要 ITEMID/LABEL 即可
    keep = [c for c in ["ITEMID", "LABEL"] if c in dl.columns]
    return dl[keep]

def _pick_itemids_by_label(df_items: DataFrame, patterns) -> dict:
    # patterns: dict[name] = regex
    out = {}
    labels = df_items["LABEL"].astype(str)
    for name, pat in patterns.items():
        regex = re.compile(pat, flags=re.IGNORECASE)
        m = labels.str.contains(regex, na=False)
        out[name] = set(df_items.loc[m, "ITEMID"].astype(int).tolist())
    return out

def _chartevents_to_vitals(mimic_dir: str, out_csv: str, itemids_map: dict, chunksize=2_000_000):
    ce = os.path.join(mimic_dir, "CHARTEVENTS.csv.gz")
    usecols = ["ICUSTAY_ID","CHARTTIME","ITEMID","VALUENUM","VALUEUOM"]
    temp_csv = out_csv + ".tmp"
    if os.path.exists(temp_csv):
        os.remove(temp_csv)

    all_itemids = set().union(*itemids_map.values())
    id2name = {iid: nm for nm, s in itemids_map.items() for iid in s}

    # 关键：统一固定列集合，避免每块宽表列数不同
    expected_vars = sorted(list(itemids_map.keys()))
    fixed_cols = ["ICUSTAY_ID", "CHARTTIME"] + expected_vars

    for i, chunk in enumerate(pd.read_csv(ce, usecols=usecols, parse_dates=["CHARTTIME"],
                                          chunksize=chunksize, low_memory=False)):
        # 过滤到我们关心的 ITEMID
        chunk = chunk[chunk["ITEMID"].isin(all_itemids)]
        if chunk.empty:
            continue
        chunk = chunk[pd.notnull(chunk["VALUENUM"])]
        chunk["VAR"] = chunk["ITEMID"].map(id2name)

        # 透视为宽表（同一时刻多条取均值）
        pivot = chunk.pivot_table(index=["ICUSTAY_ID","CHARTTIME"],
                                  columns="VAR", values="VALUENUM", aggfunc="mean").reset_index()

        # 对齐固定列集合，不存在的变量列补 NA，并按固定顺序重排
        for v in expected_vars:
            if v not in pivot.columns:
                pivot[v] = pd.NA
        pivot = pivot[fixed_cols]

        header = not os.path.exists(temp_csv)
        pivot.to_csv(temp_csv, index=False, mode="a", header=header)

        if i % 5 == 0:
            print(f"[CE] written chunk {i}, rows={len(pivot):,}", flush=True)

    if not os.path.exists(temp_csv):
        raise FileNotFoundError("chartevents 里没筛出任何 vitals 候选，请检查 pattern/ITEMID。")

    # 合并重复 (ICUSTAY_ID, CHARTTIME)
    df = pd.read_csv(temp_csv, parse_dates=["CHARTTIME"])
    df = df.groupby(["ICUSTAY_ID","CHARTTIME"], as_index=False).mean(numeric_only=True)
    df.to_csv(out_csv, index=False)
    os.remove(temp_csv)


def _labevents_to_labs(mimic_dir: str, out_csv: str, lab_itemids: dict, chunksize=1_000_000):
    le = os.path.join(mimic_dir, "LABEVENTS.csv.gz")
    usecols = ["HADM_ID","CHARTTIME","ITEMID","VALUENUM","VALUEUOM"]
    temp_csv = out_csv + ".tmp"
    if os.path.exists(temp_csv):
        os.remove(temp_csv)

    all_itemids = set().union(*lab_itemids.values())
    id2name = {iid: nm for nm, s in lab_itemids.items() for iid in s}

    expected_vars = sorted(list(lab_itemids.keys()))
    fixed_cols = ["HADM_ID","CHARTTIME"] + expected_vars

    for i, chunk in enumerate(pd.read_csv(le, usecols=usecols, parse_dates=["CHARTTIME"],
                                          chunksize=chunksize, low_memory=False)):
        chunk = chunk[chunk["ITEMID"].isin(all_itemids)]
        if chunk.empty:
            continue
        chunk = chunk[pd.notnull(chunk["VALUENUM"])]
        chunk["VAR"] = chunk["ITEMID"].map(id2name)

        pivot = chunk.pivot_table(index=["HADM_ID","CHARTTIME"],
                                  columns="VAR", values="VALUENUM", aggfunc="last").reset_index()

        for v in expected_vars:
            if v not in pivot.columns:
                pivot[v] = pd.NA
        pivot = pivot[fixed_cols]

        header = not os.path.exists(temp_csv)
        pivot.to_csv(temp_csv, index=False, mode="a", header=header)

        if i % 5 == 0:
            print(f"[LE] written chunk {i}, rows={len(pivot):,}", flush=True)

    if not os.path.exists(temp_csv):
        raise FileNotFoundError("labevents 里没筛出任何 labs 候选。")

    df = pd.read_csv(temp_csv, parse_dates=["CHARTTIME"])
    df = df.groupby(["HADM_ID","CHARTTIME"], as_index=False).last()
    df.to_csv(out_csv, index=False)
    os.remove(temp_csv)


def _inputs_to_fluid_vaso(mimic_dir: str, out_csv: str, vaso_regex: str, fluid_regex: str,
                          chunksize=1_000_000):
    # Metavision
    mv = os.path.join(mimic_dir, "INPUTEVENTS_MV.csv.gz")
    cv = os.path.join(mimic_dir, "INPUTEVENTS_CV.csv.gz")
    d_items = pd.read_csv(os.path.join(mimic_dir, "D_ITEMS.csv.gz"), low_memory=False)
    d_items.columns = [c.upper() for c in d_items.columns]
    dmap = d_items.set_index("ITEMID")["LABEL"].astype(str).to_dict()

    def _process_one(path, is_mv: bool):
        if not os.path.exists(path): return None
        if is_mv:
            usecols = ["ICUSTAY_ID","STARTTIME","ITEMID","AMOUNT","AMOUNTUOM","RATE","RATEUOM"]
            timecol = "STARTTIME"
        else:
            usecols = ["ICUSTAY_ID","CHARTTIME","ITEMID","AMOUNT","AMOUNTUOM","RATE","RATEUOM"]
            timecol = "CHARTTIME"

        dfs = []
        for chunk in pd.read_csv(path, usecols=usecols, parse_dates=[timecol],
                                 chunksize=chunksize, low_memory=False):
            if "ICUSTAY_ID" not in chunk:  # 某些CV缺列
                continue
            chunk["LABEL"] = chunk["ITEMID"].map(dmap)
            lab = chunk["LABEL"].fillna("").str.lower()

            is_vaso  = lab.str.contains(vaso_regex, regex=True)
            is_fluid = lab.str.contains(fluid_regex, regex=True)

            # 体积估算：优先 AMOUNT；没有就用 RATE 换算（近似，保守取0）
            vol = chunk["AMOUNT"].fillna(0.0)
            vol = vol.where(vol.notna(), 0.0)

            df = pd.DataFrame({
                "ICUSTAY_ID": chunk["ICUSTAY_ID"],
                "CHARTTIME": chunk[timecol],
                "FLUID_ML": vol.where(is_fluid, 0.0),
                "VASOPRESSOR_FLAG": is_vaso.astype(int)
            })
            dfs.append(df)
        if dfs:
            out = pd.concat(dfs, ignore_index=True)
            out = out[pd.notnull(out["ICUSTAY_ID"])]
            return out
        return None

    mvdf = _process_one(mv, True)
    cvdf = _process_one(cv, False)
    df = pd.concat([x for x in [mvdf, cvdf] if x is not None], ignore_index=True)
    if df is None or df.empty:
        raise FileNotFoundError("未从 INPUTEVENTS_*.csv.gz 中抽取到数据。")
    df.sort_values(["ICUSTAY_ID","CHARTTIME"], inplace=True)
    df.to_csv(out_csv, index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic_dir", required=True,
                    help="MIMIC-III v1.4 目录，包含 *.csv.gz")
    ap.add_argument("--out_dir", required=True,
                    help="输出 vitals.csv / labs.csv / inputs_fluid_vaso.csv 的目录")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 读取字典表
    d_items = _load_items_chartevents(os.path.join(args.mimic_dir, "D_ITEMS.csv.gz"))
    d_labitems = _load_labitems(os.path.join(args.mimic_dir, "D_LABITEMS.csv.gz"))

    # 变量匹配规则（用 label 正则自动匹配，不写死 ITEMID）
    vital_patterns = {
        "HR":   r"\b(heart\s*rate|^hr$)\b",
        "MAP":  r"\b(mean\s*arterial|^map$)\b",
        "SBP":  r"\b(systolic.*blood.*pressure|^sbp$)\b",
        "DBP":  r"\b(diastolic.*blood.*pressure|^dbp$)\b",
        "RR":   r"\b(respiratory\s*rate|^rr$)\b",
        "TEMP": r"\b(temp(erature)?|cel(si(u)?s)|fahrenheit)\b",
        "SPO2": r"\b(spo2|oxygen\s*saturation|o2\s*sat|oximetry)\b",
    }
    lab_patterns = {
        "CREATININE": r"\bcreatinine\b",
        "BILIRUBIN":  r"\bbilirubin\b",
        "PLATELET":   r"\b(platelet|plt)\b",
        "LACTATE":    r"\blactate\b",
    }

    vit_itemids = _pick_itemids_by_label(d_items, vital_patterns)
    lab_itemids = _pick_itemids_by_label(d_labitems, lab_patterns)

    # 抽取三份中间表
    _chartevents_to_vitals(args.mimic_dir, os.path.join(args.out_dir, "vitals.csv"), vit_itemids)
    _labevents_to_labs(args.mimic_dir, os.path.join(args.out_dir, "labs.csv"), lab_itemids)

    # 常见升压药/补液关键词（不写死 ITEMID，按 label 模糊匹配）
    vaso_regex  = r"(norepinephrine|noradrenalin|epinephrine|adrenaline|vasopressin|phenylephrine|dopamine|dobutamine)"
    fluid_regex = r"(normal\s*saline|sodium\s*chloride|ringer|lactated|plasmalyte|dextrose|albumin|hartmann|fluid)"

    _inputs_to_fluid_vaso(args.mimic_dir, os.path.join(args.out_dir, "inputs_fluid_vaso.csv"),
                          vaso_regex=vaso_regex, fluid_regex=fluid_regex)

    print("[OK] wrote vitals.csv, labs.csv, inputs_fluid_vaso.csv to", args.out_dir)

if __name__ == "__main__":
    main()
