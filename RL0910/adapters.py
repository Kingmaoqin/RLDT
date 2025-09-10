# adapters.py
import os, json, math, zipfile
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from schema import SchemaSpec, ColumnMapping, ALIASES, FEATURE_ALIASES
def _derive_reward_series(df: pd.DataFrame, spec: SchemaSpec, mp: ColumnMapping) -> pd.Series:
    rs = spec.reward_spec
    if rs is None:
        raise ValueError("未提供 reward 列，且缺少 reward_spec，无法派生 reward")
    # 1) 直接列：
    if rs.column and rs.column in df.columns:
        return df[rs.column].astype(float)
    # 2) 标签映射：
    if rs.label_col and rs.label_col in df.columns and rs.label_to_reward:
        return df[rs.label_col].map(lambda x: rs.label_to_reward.get(x, rs.label_to_reward.get(str(x), 0.0))).astype(float)
    # 3) 表达式：
    if rs.expression:
        # 安全起见，用 pandas.eval（不支持任意 Python 调用）
        # 表达式使用列名，如 "1.0 - abs(hr-70)/30"
        return pd.eval(rs.expression, engine="python", local_dict={c: df[c] for c in df.columns}).astype(float)
    raise ValueError("reward_spec 未能派生 reward：请提供 column / label_to_reward / expression 之一")

def _read_any(path: str) -> pd.DataFrame:
    p = path.lower()
    if p.endswith(".csv"):   return pd.read_csv(path)
    if p.endswith(".parquet"): return pd.read_parquet(path)
    if p.endswith(".json"):  return pd.read_json(path, lines=p.endswith(".jsonl"))
    if p.endswith(".h5") or p.endswith(".hdf5"):
        # 约定第一个 key 为表
        with pd.HDFStore(path, mode="r") as st:
            keys = st.keys()
            return st[keys[0]]
    if p.endswith(".npz"):
        arrs = np.load(path, allow_pickle=True)
        # 尝试取 'data' 或第一个数组为 2D
        for k in ("data",):
            if k in arrs:
                return pd.DataFrame(arrs[k])
        for k in arrs.files:
            a = arrs[k]
            if isinstance(a, np.ndarray) and a.ndim==2:
                return pd.DataFrame(a)
        raise ValueError("NPZ不含2D表结构")
    if p.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            # 取第一个类似 csv/parquet/json 文件
            for name in zf.namelist():
                l = name.lower()
                if l.endswith((".csv",".parquet",".json",".jsonl",".h5",".hdf5",".npz")):
                    with zf.open(name) as f:
                        tmp_path = f"/tmp/_tmp_{os.path.basename(l)}"
                        with open(tmp_path, "wb") as out:
                            out.write(f.read())
                        return _read_any(tmp_path)
        raise ValueError("ZIP未发现支持的文件")
    raise ValueError(f"不支持的文件类型: {path}")

def _auto_map_columns(df: pd.DataFrame, mapping: ColumnMapping, allow_missing_action: bool = False) -> ColumnMapping:
    cols = set(c.lower() for c in df.columns)
    def pick(alias_set):
        for c in df.columns:
            if c.lower() in alias_set:
                return c
        return None
    mp = ColumnMapping(
        trajectory_id = mapping.trajectory_id or pick(ALIASES["trajectory_id"]),
        timestep      = mapping.timestep or pick(ALIASES["timestep"]),
        action        = mapping.action or pick(ALIASES["action"]),
        reward        = mapping.reward or pick(ALIASES["reward"]),
        terminal      = mapping.terminal or pick(ALIASES["terminal"]),
        feature_cols  = mapping.feature_cols[:] if mapping.feature_cols else []
    )
    # 特征列：若未指定，则排除以上字段余下即为特征
    if not mp.feature_cols:
        exclude = {x for x in [mp.trajectory_id, mp.timestep, mp.action, mp.reward, mp.terminal] if x}
        mp.feature_cols = [c for c in df.columns if c not in exclude]
    # 校验
    need = [mp.trajectory_id, mp.timestep]
    if any(v is None for v in need):
        raise ValueError(f"列映射缺失（trajectory_id/timestep 至少应提供）：{mp}")

    if not allow_missing_action and mp.action is None:
        raise ValueError(f"列映射缺失（action 列缺失，且未声明允许缺失）：{mp}")
    return mp

def _build_meta(df: pd.DataFrame, spec: SchemaSpec, mp: ColumnMapping) -> Dict[str, Any]:
    # 动作名
    a_col = mp.action
    if spec.action_names:
        action_names = spec.action_names
    else:
        if a_col and a_col in df.columns:
            u = df[a_col].unique()
            if np.issubdtype(df[a_col].dtype, np.number):
                action_ids = np.unique(df[a_col].astype(int))
                action_names = [f"action_{i}" for i in sorted(action_ids)]
            else:
                action_names = [str(x) for x in sorted(map(str, u))]
        else:
            # 某些 sensor 流程可能没有显式动作列，此处留空由上游处理
            action_names = []

    # 特征名
    feature_names = spec.feature_names or mp.feature_cols

    # 基本 meta
    meta_dict = dict(
        feature_names=feature_names,
        action_names=action_names,
        trajectory_id_col=mp.trajectory_id,
        timestep_col=mp.timestep,
        action_col=a_col,
        reward_col=mp.reward,
        terminal_col=mp.terminal,
    )

    # 解析 critical_features 规则 -> 固定索引
    cf_rules = []
    fname_lower = [str(n).lower() for n in feature_names]
    for rule in getattr(spec, "critical_features", []) or []:
        idx = None
        disp = getattr(rule, "display_name", None)

        if getattr(rule, "index", None) is not None:
            idx = int(rule.index)
            if disp is None and 0 <= idx < len(feature_names):
                disp = feature_names[idx]
        else:
            aliases = getattr(rule, "name_or_aliases", None)
            if isinstance(aliases, str):
                aliases = [aliases]
            aliases = [a.lower() for a in (aliases or [])]
            for i, nm in enumerate(fname_lower):
                if nm in aliases:
                    idx = i
                    if disp is None:
                        disp = feature_names[i]
                    break

        if idx is None:
            raise ValueError(f"critical feature 无法定位列：{rule}")

        cf_rules.append(dict(
            index=idx,
            op=rule.op,
            threshold=float(rule.threshold),
            weight=float(rule.weight),
            as_terminal=bool(rule.as_terminal),
            display_name=disp or f"f{idx}",
        ))

    meta_dict["critical_features"] = cf_rules

    # （可选，向后兼容）若配置里仍有 spo2 别名，尝试自动定位
    if getattr(spec, "critical_feature_alias", None):
        alias_set = {s.lower() for s in FEATURE_ALIASES.get(spec.critical_feature_alias, set())}
        spo2_idx = None
        for i, name in enumerate(feature_names):
            if name.lower() in alias_set:
                spo2_idx = i
                break
        meta_dict["spo2_idx"] = spo2_idx
        meta_dict["spo2_threshold"] = float(getattr(spec, "spo2_threshold", 0.80))

    return meta_dict

def _derive_reward_series(df: pd.DataFrame, spec: SchemaSpec, mp: ColumnMapping) -> pd.Series:
    rs = spec.reward_spec
    if rs is None:
        raise ValueError("未提供 reward 列，且缺少 reward_spec，无法派生 reward")

    # 1) 直接列
    if rs.column and rs.column in df.columns:
        return df[rs.column].astype(float)

    # 2) 标签映射 -> reward
    if rs.label_col and rs.label_col in df.columns and rs.label_to_reward:
        return df[rs.label_col].map(
            lambda x: rs.label_to_reward.get(x, rs.label_to_reward.get(str(x), 0.0))
        ).astype(float)

    # 3) 表达式（基于列名）
    if rs.expression:
        return pd.eval(rs.expression, engine="python",
                       local_dict={c: df[c] for c in df.columns}).astype(float)

    raise ValueError("reward_spec 未能派生 reward：请提供 column / label_to_reward / expression 之一")


def _encode_actions(a: pd.Series, action_names: List[str]) -> np.ndarray:
    # 若 a 已经是整数且在 0..K-1，则直接返回；否则做映射
    if np.issubdtype(a.dtype, np.number) and a.min() >= 0 and a.max() < len(action_names):
        return a.astype(np.int64).to_numpy()
    name_to_id = {str(n): i for i, n in enumerate(action_names)}
    return a.astype(str).map(name_to_id).astype(np.int64).to_numpy()

def _make_transitions_tabular(df: pd.DataFrame, mp: ColumnMapping, meta: Dict[str,Any], spec: Optional[SchemaSpec]=None) -> Dict[str, np.ndarray]:
    # 按 (traj, timestep) 排序并检查连续性
    df = df.sort_values([mp.trajectory_id, mp.timestep]).reset_index(drop=True)
    # 动作编码
    actions = _encode_actions(df[mp.action], meta["action_names"])
    # 终止：如无则按轨迹边界构造
    if mp.terminal and mp.terminal in df:
        terminals = df[mp.terminal].astype(bool).to_numpy()
    else:
        tid = df[mp.trajectory_id].to_numpy()
        ts  = df[mp.timestep].to_numpy()
        next_tid = np.r_[tid[1:], -1]
        next_ts  = np.r_[ts[1:],  -1]
        terminals = ((next_tid != tid) | (next_ts != ts + 1))
    # 状态与下一状态
    X = df[mp.feature_cols].to_numpy(dtype=np.float32)
    # next_state：向后对齐一位（跨轨迹处用自身或掩码）
    X_next = np.vstack([X[1:], X[-1:]]).copy()
    X_next[np.where(terminals)] = X[np.where(terminals)]
    # 轨迹/时间/奖励
    traj = pd.factorize(df[mp.trajectory_id])[0].astype(np.int64)
    t    = df[mp.timestep].astype(np.int64).to_numpy()
    if (mp.reward and mp.reward in df.columns):
        rewards = df[mp.reward].astype(np.float32).to_numpy()
    else:
        if spec is None:
            raise ValueError("Tabular 数据缺少 reward 列且未提供 SchemaSpec 以派生 reward")
        rewards = _derive_reward_series(df, spec, mp).astype(np.float32).to_numpy()

        
    return dict(
        states=X, actions=actions, rewards=rewards,
        next_states=X_next, trajectory_ids=traj, timesteps=t,
        terminals=terminals.astype(np.float32)
    )

def _window_view(arr: np.ndarray, length: int, stride: int) -> np.ndarray:
    # 生成 (N_window, length, D) 的滑窗视图，然后可以 flatten 到 (N_window, length*D)
    N, D = arr.shape
    if N < length:
        return np.empty((0, length, D), dtype=arr.dtype)
    n_win = 1 + (N - length) // stride
    out = np.empty((n_win, length, D), dtype=arr.dtype)
    idx = 0
    for s in range(0, N - length + 1, stride):
        out[idx] = arr[s:s+length]
        idx += 1
    return out

def _make_transitions_sensor(df: pd.DataFrame, spec: SchemaSpec, mp: ColumnMapping, meta: Dict[str,Any]) -> Dict[str, np.ndarray]:
    # 传感器数据假设：每行一个时间点，mp.feature_cols 为多通道信号；需要滑窗 -> 状态
    assert spec.window.enabled, "Sensor 模式必须提供 windowing 配置"
    df = df.sort_values([mp.trajectory_id, mp.timestep]).reset_index(drop=True)
    traj_ids = pd.factorize(df[mp.trajectory_id])[0]
    timesteps= df[mp.timestep].astype(np.int64).to_numpy()

    X = df[mp.feature_cols].to_numpy(dtype=np.float32)
    length, stride = spec.window.length, spec.window.stride

    # 根据每条轨迹分别滑窗
    all_states, all_actions, all_rewards, all_next_states, all_traj, all_ts, all_term = [], [], [], [], [], [], []
    # 动作来源：1) 明确 action 列 2) label_col + label_to_action 3) derive_action_fn
    use_action_col = mp.action in df.columns
    for tid in np.unique(traj_ids):
        seg = df[traj_ids==tid]
        Xseg = seg[mp.feature_cols].to_numpy(dtype=np.float32)
        # 状态：flatten window
        W = _window_view(Xseg, length, stride)      # (M, L, D)
        if W.shape[0] == 0: 
            continue
        states = W.reshape(W.shape[0], -1)         # (M, L*D)

        # 下一状态：右移一个窗口（同轨迹内）
        Wn = W.copy()
        Wn[:-1] = W[1:]
        next_states = Wn.reshape(Wn.shape[0], -1)

        # 时间戳取窗口末端
        ts = seg[mp.timestep].to_numpy()
        ts_win = ts[length-1::stride][:states.shape[0]]  # 对齐窗口数

        # 动作
        if use_action_col:
            a = seg[mp.action].to_numpy()
            a_win = a[length-1::stride][:states.shape[0]]
            actions = _encode_actions(pd.Series(a_win), meta["action_names"])
        elif spec.window.label_col and spec.window.label_col in seg.columns and spec.window.label_to_action:
            labels = seg[spec.window.label_col].to_numpy()
            lab_win = labels[length-1::stride][:states.shape[0]]
            a_map = spec.window.label_to_action
            actions = np.array([a_map.get(x, a_map.get(str(x), 0)) for x in lab_win], dtype=np.int64)
        elif spec.window.derive_action_fn:
            # 用户提供函数：每个窗口 -> 动作id
            actions = np.array([spec.window.derive_action_fn(pd.DataFrame(W[i], columns=mp.feature_cols)) for i in range(W.shape[0])], dtype=np.int64)
        else:
            raise ValueError("Sensor 数据未提供动作来源：action列 / label映射 / derive_action_fn 三者其一")

        # 奖励：必须给定 reward 列，否则报错（不做简化）
        if mp.reward and mp.reward in seg:
            r_series = seg[mp.reward].astype(float)
        else:
            if spec.reward_spec is None:
                raise ValueError("Sensor 数据缺少 reward 列，且未提供 reward_spec 用于派生")
            r_series = _derive_reward_series(seg, spec, mp)

        r = r_series.to_numpy(dtype=np.float32)
        # 将逐点 reward 聚合成每个窗口一个 reward
        agg = getattr(spec.reward_spec, "window_agg", "last") if spec.reward_spec else "last"
        if agg not in ("last", "mean", "sum"):
            raise ValueError(f"不支持的 window_agg: {agg}")

        # 拿到窗口索引范围（每个窗口对应原始序列的 [s, s+length)）
        M = W.shape[0]
        r_win = np.empty(M, dtype=np.float32)
        start_idxs = list(range(0, len(r) - length + 1, stride))
        for i, s in enumerate(start_idxs[:M]):
            e = s + length
            if agg == "last": r_win[i] = r[e-1]
            elif agg == "mean": r_win[i] = float(r[s:e].mean())
            else: r_win[i] = float(r[s:e].sum())

        # 终止：窗口末端终止
        if mp.terminal and mp.terminal in seg:
            term = seg[mp.terminal].astype(bool).to_numpy()
            term_win = term[length-1::stride][:states.shape[0]]
        else:
            term_raw = np.zeros_like(ts, dtype=bool)
            term_raw[-1] = True
            term_win = term_raw[length-1::stride][:states.shape[0]]

        # 收集
        M = states.shape[0]
        all_states.append(states)
        all_next_states.append(next_states)
        all_actions.append(actions)
        all_rewards.append(r_win)
        all_traj.append(np.full(M, int(tid), dtype=np.int64))
        all_ts.append(ts_win)
        all_term.append(term_win.astype(np.float32))

    return dict(
        states=np.vstack(all_states),
        actions=np.concatenate(all_actions),
        rewards=np.concatenate(all_rewards),
        next_states=np.vstack(all_next_states),
        trajectory_ids=np.concatenate(all_traj),
        timesteps=np.concatenate(all_ts),
        terminals=np.concatenate(all_term),
    )

class TabularAdapter:
    @staticmethod
    def load(path: str, spec: SchemaSpec) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        df = _read_any(path)
        mp = _auto_map_columns(df, spec.mapping)
        meta = _build_meta(df, spec, mp)
        ds  = _make_transitions_tabular(df, mp, meta, spec)
        return ds, meta

class SensorAdapter:
    @staticmethod
    def load(path: str, spec: SchemaSpec) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        df = _read_any(path)
        allow_missing_action = False
        if spec.window:
            has_label2act = bool(spec.window.label_col and spec.window.label_to_action)
            has_derive_fn = bool(spec.window.derive_action_fn)
            allow_missing_action = has_label2act or has_derive_fn

        mp = _auto_map_columns(df, spec.mapping, allow_missing_action=allow_missing_action)
        meta = _build_meta(df, spec, mp)
        ds  = _make_transitions_sensor(df, spec, mp, meta)
        return ds, meta
