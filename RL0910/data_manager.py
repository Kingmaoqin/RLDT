"""
data_manager.py - 管理虚拟数据和真实数据的读取
"""

import os

# pandas 依赖在部分环境中会尝试加载与 NumPy 不兼容的 pyarrow
# 可用变量禁用 Arrow backend，避免 _ARRAY_API not found 报错
os.environ.setdefault("PANDAS_USE_PYARROW_EXTENSION_ARRAY", "0")
os.environ.setdefault("PANDAS_USE_PYARROW_BACKEND", "0")

import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
from data import PatientDataGenerator
import json
from adapters import TabularAdapter, SensorAdapter
from schema import SchemaSpec
from pandas_compat import get_pandas

try:
    import yaml
except Exception:
    yaml = None

pd = get_pandas()


class DataManager:
    """统一的数据管理接口"""
    
    def __init__(self):
        self.virtual_data = None
        self.real_data = None
        self.current_source = None  # "virtual" / "real" once a dataset is prepared
        self.real_data_path = None
        self.current_meta = {}
        self.current_schema = {}
        self.training_history = []

    # ------------------------------------------------------------------
    # Convenience accessors & helpers
    # ------------------------------------------------------------------

    def get_current_meta(self) -> dict:
        """Return metadata for the current dataset."""
        return getattr(self, "current_meta", {}) or {}

    def get_current_schema(self) -> dict:
        """Return schema information for the current dataset."""
        return getattr(self, "current_schema", {}) or {}


    def _resolve_action_name(self, action_value: Union[int, str]) -> str:
        """Resolve an action identifier to a display name using cached metadata."""

        meta = self.get_current_meta()
        action_map = meta.get("action_map") or {}
        candidates = [action_value, str(action_value)]
        for candidate in candidates:
            if candidate in action_map:
                return str(action_map[candidate])
            try:
                icandidate = int(candidate)
            except Exception:
                continue
            if icandidate in action_map:
                return str(action_map[icandidate])

        action_names = meta.get("action_names") or []
        if isinstance(action_value, (int, np.integer)) and int(action_value) < len(action_names):
            return str(action_names[int(action_value)])
        return str(action_value)

    def _update_feature_metadata(self, df: "pd.DataFrame", feature_cols: List[str]):
        """Compute descriptive statistics for feature columns to improve UI scaling."""

        if df is None or not len(feature_cols):
            return

        stats = {}
        name_map = {}
        for col in feature_cols:
            series = pd.to_numeric(df[col], errors="coerce")
            clean = col[6:] if str(col).startswith("state_") else str(col)
            stats[clean] = {
                "min": float(series.min(skipna=True)) if len(series) else 0.0,
                "max": float(series.max(skipna=True)) if len(series) else 0.0,
                "mean": float(series.mean(skipna=True)) if len(series) else 0.0,
                "std": float(series.std(skipna=True)) if len(series) else 0.0,
                "column": str(col),
            }
            name_map[clean] = str(col)

        meta = self.current_meta or {}
        meta["feature_stats"] = stats
        meta["feature_ranges"] = {
            k: {"min": v.get("min", 0.0), "max": v.get("max", 0.0)}
            for k, v in stats.items()
        }
        meta["feature_column_map"] = name_map
        self.current_meta = meta

    def register_training_run(self, summary: Dict[str, Any]):
        """Append a training summary for diagnostics and UI messaging."""

        summary = dict(summary)
        summary.setdefault("recorded_at", datetime.now().isoformat())

        history = list(getattr(self, "training_history", []) or [])
        history.append(summary)
        self.training_history = history

        meta = self.get_current_meta()
        runs = list(meta.get("training_runs", []))
        runs.append(summary)
        meta["training_runs"] = runs
        meta["last_training"] = summary
        self.current_meta = meta

    def generate_virtual_data(self, n_patients: int = 1000, seed: int = 42) -> pd.DataFrame:
        """生成虚拟数据"""
        print(f"Generating virtual data for {n_patients} patients...")
        generator = PatientDataGenerator(n_patients=n_patients, seed=seed)
        data_dict = generator.generate_dataset()
        
        # 转换为DataFrame
        self.virtual_data = generator.create_dataframe(data_dict)
        
        # 添加患者ID
        patient_ids = []
        for pid in self.virtual_data['trajectory_id'].unique():
            patient_ids.extend(
                [f"P{pid:04d}"]
                * len(self.virtual_data[self.virtual_data['trajectory_id'] == pid])
            )
        self.virtual_data['patient_id'] = patient_ids
        feature_cols = [
            c
            for c in self.virtual_data.columns
            if c.startswith("state_") and not c.startswith("next_state_")
        ]
        unique_actions = (
            sorted(self.virtual_data['action'].unique())
            if 'action' in self.virtual_data.columns
            else []
        )
        action_names = [f"Action {a}" for a in unique_actions]
        action_map = None
        if unique_actions:
            action_map = {}
            for raw, name in zip(unique_actions, action_names):
                try:
                    key = int(raw)
                except Exception:
                    key = raw
                action_map[key] = name

        self.current_meta = {
            "feature_columns": feature_cols,
            "action_names": action_names if action_names else None,
            "action_map": action_map,
        }
        self._update_feature_metadata(self.virtual_data, feature_cols)
        self.current_source = "virtual"
        print(f"Generated {len(self.virtual_data)} records for {n_patients} patients")
        return self.virtual_data
    
    def load_real_data(self, file_path: str, file_type: str = "csv") -> pd.DataFrame:
        """加载真实数据"""
        try:
            if file_type == "csv":
                self.real_data = pd.read_csv(file_path)
            elif file_type == "parquet":
                self.real_data = pd.read_parquet(file_path)
            elif file_type == "excel":
                self.real_data = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            self.real_data_path = file_path
            print(f"Loaded real data from {file_path}: {len(self.real_data)} records")

            # 确保有patient_id列
            if 'patient_id' not in self.real_data.columns:
                self.real_data['patient_id'] = [f"R{i:04d}" for i in range(len(self.real_data))]
            feature_cols = [
                c
                for c in self.real_data.columns
                if c.startswith("state_") and not c.startswith("next_state_")
            ]
            if not feature_cols:
                drop_cols = {"patient_id", "timestep", "action", "reward", "terminal"}
                numeric = self.real_data.select_dtypes(include=["int64", "float64", "float32", "int32"]).columns.tolist()
                feature_cols = [c for c in numeric if c not in drop_cols]
            unique_actions = (
                sorted(self.real_data["action"].unique())
                if "action" in self.real_data.columns
                else []
            )
            action_names = [str(a) for a in unique_actions]
            action_map = {}
            for raw in unique_actions:
                key = raw
                try:
                    key = int(raw)
                except Exception:
                    pass
                action_map[key] = str(raw)
            self.current_meta = {
                "feature_columns": feature_cols,
                "action_names": action_names if action_names else None,
                "action_map": action_map if action_map else None,
            }
            self._update_feature_metadata(self.real_data, feature_cols)

            self.current_source = "real"
            return self.real_data
        except Exception as e:
            print(f"Error loading real data: {e}")
            raise

    def _load_schema_spec(self,
                          schema_path: Optional[str] = None,
                          schema_yaml: Optional[str] = None) -> "SchemaSpec":
        """Parse SchemaSpec with graceful fallback for older deployments."""

        from schema import SchemaSpec

        if schema_yaml:
            loader = getattr(SchemaSpec, "from_yaml_text", None)
            if callable(loader):
                return loader(schema_yaml)
            if yaml is None:
                raise ImportError("PyYAML is required to parse inline schema text")
            data = yaml.safe_load(schema_yaml)
            return SchemaSpec.from_dict(data)

        if schema_path:
            loader = getattr(SchemaSpec, "from_yaml_file", None)
            if callable(loader):
                return loader(schema_path)
            if yaml is None:
                raise ImportError("PyYAML is required to parse schema files")
            with open(schema_path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh.read())
            return SchemaSpec.from_dict(data)

        raise ValueError("真实数据需要提供 schema_path 或 schema_yaml 才能映射到统一结构")

    def load_real_data_with_schema(self,
                                file_path: str,
                                file_type: str,
                                schema_path: Optional[str] = None,
                                schema_yaml: Optional[str] = None) -> pd.DataFrame:
        """
        使用 YAML Schema 通过 adapters 将任意真实数据映射到统一 RL 结构，
        并落地到 self.real_data / self.current_meta。
        """
        from adapters import TabularAdapter, SensorAdapter

        # 1) 读取 Schema
        if schema_yaml:
            spec = SchemaSpec.from_yaml_text(schema_yaml)
        elif schema_path:
            spec = SchemaSpec.from_yaml_file(schema_path)
        else:
            raise ValueError("Real data need schema_path or schema_yaml")

        # 2) 选择适配器
        kind = getattr(spec, "kind", "tabular").lower()
        if kind.startswith("tab"):
            ds, meta = TabularAdapter.load(file_path, spec)
        else:
            ds, meta = SensorAdapter.load(file_path, spec)

        # 3) 拼装 UI 统一表：patient_id / timestep / action / reward / state_<name>
        n = ds["states"].shape[0]
        traj = ds["trajectory_ids"].astype(int)
        t    = ds["timesteps"].astype(int)
        act  = ds["actions"].astype(int)
        rew  = ds["rewards"].astype(float)

        df = pd.DataFrame({
            "patient_id": [f"R{int(x):04d}" for x in traj],
            "timestep": t,
            "action": act,
            "reward": rew,
        })

        feat_names = list(meta.get("feature_names") or meta.get("feature_columns") or [])
        X = ds["states"]
        feature_columns: List[str] = []
        for i, name in enumerate(feat_names):
            safe = str(name).strip().replace(" ", "_").lower()
            col_name = f"state_{safe}"
            df[col_name] = X[:, i].astype(float)
            feature_columns.append(col_name)

        if feature_columns:
            df[feature_columns] = (
                df[feature_columns]
                .apply(pd.to_numeric, errors="coerce")
                .astype(np.float32)
            )

        # 4) 落地到管理器
        self.real_data = df
        self.current_source = "real"
        self.real_data_path = file_path
        meta = dict(meta)
        meta.setdefault("feature_columns", feature_columns)
        meta.setdefault("feature_names", feat_names)
        meta.setdefault("schema_source", schema_path or "inline")
        meta.setdefault("data_type", getattr(spec, "data_type", "tabular"))
        meta.setdefault("reward_spec", getattr(spec, "reward_spec", None))
        meta.setdefault("normalization", getattr(spec, "normalization", None))

        # 记录特征 dtype 以便诊断
        if feature_columns:
            meta["feature_dtypes"] = {
                name: str(df[name].dtype) for name in feature_columns
            }

        self.current_meta = meta  # <— 供 UI/推理使用
        self._update_feature_metadata(df, feature_columns)
        self.current_schema = spec

        print(f"[DataManager] Real data (schema) loaded: {len(df)} rows, "
            f"{df['patient_id'].nunique()} patients, {len(feat_names)} features.")
        return df

    def load_real_data_schema_less(self, file_path: str) -> "pd.DataFrame":

        df = pd.read_csv(file_path)

        # 1) patient_id
        pid_col = None
        for c in ["patient_id", "subject_id", "hadm_id", "icustay_id", "trajectory_id", "subject"]:
            if c in df.columns:
                pid_col = c; break
        if pid_col is None:
            # 尝试 subject_id + visit 组合
            if "subject_id" in df.columns and "visit" in df.columns:
                df["patient_id"] = df["subject_id"].astype(str) + "_" + df["visit"].astype(str)
            else:
                df["patient_id"] = [f"R{int(i):04d}" for i in range(len(df))]
        else:
            df.rename(columns={pid_col: "patient_id"}, inplace=True)

        # 2) timestep
        t_col = None
        for c in ["timestep", "time", "frame", "step"]:
            if c in df.columns:
                t_col = c; break
        if t_col is None:
            df["timestep"] = df.groupby("patient_id").cumcount()
        else:
            df.rename(columns={t_col: "timestep"}, inplace=True)
        df["timestep"] = (
            pd.to_numeric(df["timestep"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

        # 3) action / reward / terminal
        if "action" not in df.columns:
            for c in ["action_id", "treatment_id", "act"]:
                if c in df.columns:
                    df.rename(columns={c: "action"}, inplace=True)
                    break
        if "action" not in df.columns:
            df["action"] = -1
        df["action"] = (
            pd.to_numeric(df["action"], errors="coerce")
            .fillna(-1)
            .astype(int)
        )

        if "reward" not in df.columns:
            for c in ["r", "return", "sofa_delta"]:
                if c in df.columns:
                    df.rename(columns={c: "reward"}, inplace=True)
                    break
        if "reward" not in df.columns:
            df["reward"] = 0.0
        df["reward"] = pd.to_numeric(df["reward"], errors="coerce").fillna(0.0)

        if "terminal" not in df.columns:
            for c in ["done", "is_terminal", "terminal_flag"]:
                if c in df.columns:
                    df.rename(columns={c: "terminal"}, inplace=True)
                    break
        if "terminal" not in df.columns:
            # 每个病人最后一条视为终止
            df["terminal"] = (
                df["patient_id"] != df["patient_id"].shift(-1)
            ).astype(int)
        df["terminal"] = (
            pd.to_numeric(df["terminal"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

        # 4) 特征列：优先 state_* 前缀；否则选数值列中排除关键列
        feature_cols = [c for c in df.columns if c.startswith("state_")]
        if not feature_cols:
            drop_cols = {"patient_id", "timestep", "action", "reward", "terminal"}
            numeric = df.select_dtypes(include=["int64", "float64", "float32", "int32"]).columns.tolist()
            feature_cols = [c for c in numeric if c not in drop_cols]

            # 统一成 state_* 前缀，避免 UI/绘图分支找不到
            new_names = {}
            for i, c in enumerate(feature_cols):
                new_names[c] = f"state_{str(c).strip().replace(' ','_').lower()}"
            df.rename(columns=new_names, inplace=True)
            feature_cols = [new_names[c] for c in feature_cols]

        if feature_cols:
            df[feature_cols] = (
                df[feature_cols]
                .apply(pd.to_numeric, errors="coerce")
                .astype(np.float32)
            )

        # 5) 更新管理器缓存与 meta
        unique_actions = (
            sorted(pd.unique(df["action"].dropna()))
            if "action" in df.columns
            else []
        )
        self.real_data = df
        self.real_data_path = file_path
        self.current_source = "real"
        action_map = {}
        for raw in unique_actions:
            key = raw
            try:
                key = int(raw)
            except Exception:
                pass
            action_map[key] = str(raw)

        self.current_meta = {
            "feature_columns": feature_cols,
            "action_names": [str(a) for a in unique_actions] if len(unique_actions) else None,
            "action_map": action_map if action_map else None,
        }
        self._update_feature_metadata(df, feature_cols)
        print(f"[DataManager] Real schema-less loaded: {len(df)} rows, "
            f"{df['patient_id'].nunique()} patients, {len(feature_cols)} features.")
        return df

    def set_data_source(self, source: str, ensure_available: bool = False):
        """设置当前使用的数据源"""
        if source not in ["virtual", "real"]:
            raise ValueError("Source must be 'virtual' or 'real'")
        self.current_source = source
        print(f"Data source set to: {source}")
        if ensure_available:
            if source == "virtual" and self.virtual_data is None:
                raise ValueError("Virtual dataset not generated yet")
            if source == "real" and self.real_data is None:
                raise ValueError("Real dataset not loaded yet")
    def get_current_data(self) -> pd.DataFrame:
        """获取当前激活的数据"""
        if self.current_source == "virtual":
            if self.virtual_data is None:
                raise ValueError("Virtual dataset not generated. Use the Data Management tab to create it.")
            return self.virtual_data
        if self.current_source == "real":
            if self.real_data is None:
                raise ValueError("No real data loaded. Please load data first.")
            return self.real_data
        raise ValueError("No dataset active. Choose a data source in the UI.")
    
    def get_patient_list(self) -> List[str]:
        """获取患者列表"""
        data = self.get_current_data()
        # 获取每个患者的最新记录
        latest_records = data.groupby('patient_id').last().reset_index()
        return latest_records['patient_id'].tolist()
    
    def get_patient_info(self, patient_id: str) -> Dict:
        """获取特定患者的信息"""
        data = self.get_current_data()
        patient_data = data[data['patient_id'] == patient_id]
        
        if patient_data.empty:
            raise ValueError(f"Patient {patient_id} not found")
        
        # 获取最新状态
        latest_record = patient_data.iloc[-1]
        
        # 构建患者信息
        meta = self.get_current_meta()
        current_state = self._extract_state_from_record(latest_record)

        def _labelize(val):
            return self._resolve_action_name(val)

        patient_info = {
            'patient_id': patient_id,
            'total_records': len(patient_data),
            'current_state': current_state,
            'trajectory': self._get_patient_trajectory(patient_data),
            'treatment_history': patient_data['action'].tolist(),
            'treatment_labels': [_labelize(v) for v in patient_data['action'].tolist()],
            'outcome_history': patient_data['reward'].tolist(),
            'feature_stats': meta.get('feature_stats', {}),
        }

        return patient_info
    
    def get_patient_state(self, patient_id: str, timestep: Optional[int] = None) -> Dict:
        """获取患者在特定时间点的状态"""
        data = self.get_current_data()
        patient_data = data[data['patient_id'] == patient_id]
        
        if patient_data.empty:
            raise ValueError(f"Patient {patient_id} not found")
        
        if timestep is None:
            # 获取最新状态
            record = patient_data.iloc[-1]
        else:
            # 获取特定时间点
            timestep_data = patient_data[patient_data['timestep'] == timestep]
            if timestep_data.empty:
                raise ValueError(f"No data for patient {patient_id} at timestep {timestep}")
            record = timestep_data.iloc[0]
        
        return self._extract_state_from_record(record)
    
    def _extract_state_from_record(self, record: pd.Series) -> Dict:
        """从记录中提取状态信息"""
        state = {}
        
        # 提取状态特征
        feature_columns = [col for col in record.index if col.startswith('state_') and not col.startswith('next_state_')]

        for col in feature_columns:
            feature_name = col.replace('state_', '')
            value = record[col]
            try:
                state[feature_name] = float(value)
            except Exception:
                state[feature_name] = value

        # 添加其他信息
        state['timestep'] = int(record.get('timestep', 0))
        last_action_val = record.get('action', -1)
        try:
            state['last_action'] = int(last_action_val)
        except Exception:
            state['last_action'] = last_action_val
        state['last_action_label'] = self._resolve_action_name(last_action_val)
        state['last_reward'] = float(record.get('reward', 0))

        return state
    
    def _get_patient_trajectory(self, patient_data: pd.DataFrame) -> List[Dict]:
        """获取患者的完整轨迹"""
        trajectory = []
        
        for _, record in patient_data.iterrows():
            state = self._extract_state_from_record(record)
            act = record['action']
            try:
                act_id = int(act)
            except Exception:
                act_id = act
            trajectory.append({
                'timestep': state['timestep'],
                'state': state,
                'action': act_id,
                'action_label': self._resolve_action_name(act),
                'reward': float(record['reward'])
            })
        
        return trajectory
    
    def get_cohort_statistics(self, filter_criteria: Optional[Dict] = None) -> Dict:
        """获取队列统计信息，容错处理缺失或异常列"""

        data = self.get_current_data().copy()

        # 应用过滤条件
        if filter_criteria:
            for key, value in filter_criteria.items():
                if key in data.columns:
                    data = data[data[key] == value]

        total_records = int(len(data))
        total_patients = int(data['patient_id'].nunique()) if 'patient_id' in data.columns else 0

        avg_traj_len = 0.0
        if 'patient_id' in data.columns and not data.empty:
            mean_val = data.groupby('patient_id').size().mean()
            avg_traj_len = float(mean_val) if pd.notna(mean_val) else 0.0

        action_distribution: Dict[str, int] = {}
        if 'action' in data.columns:
            counts = data['action'].value_counts(dropna=False)
            action_distribution = {str(k): int(v) for k, v in counts.to_dict().items()}

        avg_reward = 0.0
        if 'reward' in data.columns:
            reward_series = pd.to_numeric(data['reward'], errors='coerce')
            if reward_series.notna().any():
                avg_reward = float(reward_series.mean())

        stats = {
            'total_patients': total_patients,
            'total_records': total_records,
            'avg_trajectory_length': avg_traj_len,
            'action_distribution': action_distribution,
            'avg_reward': avg_reward,
            'feature_stats': {},
        }

        # 计算特征统计
        feature_columns = [
            col for col in data.columns
            if col.startswith('state_') and not col.startswith('next_state_')
        ]
        for col in feature_columns:
            feature_name = col.replace('state_', '')
            series = pd.to_numeric(data[col], errors='coerce')
            if series.notna().any():
                stats['feature_stats'][feature_name] = {
                    'mean': float(series.mean()),
                    'std': float(series.std()),
                    'min': float(series.min()),
                    'max': float(series.max()),
                }

        return stats
    
    def export_patient_data(self, patient_id: str, output_path: str):
        """导出特定患者的数据"""
        data = self.get_current_data()
        patient_data = data[data['patient_id'] == patient_id]
        
        if patient_data.empty:
            raise ValueError(f"Patient {patient_id} not found")
        
        # 根据文件扩展名选择格式
        if output_path.endswith('.csv'):
            patient_data.to_csv(output_path, index=False)
        elif output_path.endswith('.json'):
            patient_info = self.get_patient_info(patient_id)
            with open(output_path, 'w') as f:
                json.dump(patient_info, f, indent=2)
        else:
            raise ValueError("Unsupported export format. Use .csv or .json")
        
        print(f"Exported patient {patient_id} data to {output_path}")

def load_user_dataset(
    data_path: str,
    schema: SchemaSpec,
    save_meta_to: str = "./output/models/dataset_meta.json",
    fit_normalization: bool = True
) -> tuple[dict, dict]:
    """
    统一入口：读取任意用户数据 -> 内部标准数据字典 + 元数据
    - data_path: 用户上传文件路径
    - schema:    SchemaSpec（可从 YAML 读入）
    - save_meta_to: 保存 meta 的 json 路径
    """
    if schema.data_type == "tabular":
        ds, meta = TabularAdapter.load(data_path, schema)
    elif schema.data_type == "sensor":
        ds, meta = SensorAdapter.load(data_path, schema)
    else:
        raise ValueError(f"未知 data_type: {schema.data_type}")

    if fit_normalization and schema.normalization.method != "none":
        X = ds["states"]; Xn = ds["next_states"]
        if schema.normalization.method == "standard":
            mu = X.mean(axis=0); std = X.std(axis=0) + 1e-6
            ds["states"] = (X - mu) / std
            ds["next_states"] = (Xn - mu) / std
            meta["norm"] = {"method":"standard", "mean": mu.tolist(), "std": std.tolist()}
        elif schema.normalization.method == "minmax":
            lo = X.min(axis=0); hi = X.max(axis=0); span = (hi - lo); span[span==0] = 1.0
            ds["states"] = (X - lo) / span
            ds["next_states"] = (Xn - lo) / span
            meta["norm"] = {"method":"minmax", "min": lo.tolist(), "max": hi.tolist()}
        else:
            meta["norm"] = {"method":"none"}

    os.makedirs(os.path.dirname(save_meta_to), exist_ok=True)
    with open(save_meta_to, "w") as f:
        json.dump(meta, f, indent=2)

    if 'feature_columns' not in meta and 'features' in meta:
        meta['feature_columns'] = meta['features']

    # 安装动作名字（如果 schema 显式给了）
    schema = meta.get('schema', None)
    if schema and isinstance(schema, dict):
        # 支持两种写法：
        #   actions:
        #     names: [xxx, yyy]
        #   或
        #   action_map: {0: 'xxx', 1:'yyy'}
        act = schema.get('actions') or {}
        if isinstance(act.get('names'), (list, tuple)):
            meta['action_names'] = list(map(str, act['names']))
        if isinstance(act.get('map'), dict):
            meta['action_map'] = {int(k): str(v) for k, v in act['map'].items()}

    # 关键规则（critical_features）若放在 schema 里，也抄一份
    rules = schema.get('critical_features') if schema else None
    if rules and 'critical_features' not in meta:
        meta['critical_features'] = rules
    return ds, meta

# 全局数据管理器实例
data_manager = DataManager()
