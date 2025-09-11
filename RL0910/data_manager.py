"""
data_manager.py - 管理虚拟数据和真实数据的读取
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Union
from data import PatientDataGenerator
import json
from adapters import TabularAdapter, SensorAdapter
from schema import SchemaSpec
from typing import Tuple


class DataManager:
    """统一的数据管理接口"""
    
    def __init__(self):
        self.virtual_data = None
        self.real_data = None
        self.current_source = "virtual"  # "virtual" or "real"
        self.real_data_path = None
        self.current_meta = {}
        self.current_schema = {}

    def get_current_meta(self) -> dict:
        """Return metadata for the current dataset."""
        return getattr(self, "current_meta", {}) or {}

    def get_current_schema(self) -> dict:
        """Return schema information for the current dataset."""
        return getattr(self, "current_schema", {}) or {}


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
        self.current_meta = {
            "feature_columns": feature_cols,
            "action_names": action_names if action_names else None,
            "action_map": {
                a: name for a, name in zip(unique_actions, action_names)
            } if unique_actions else None,
        }        
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
            
            return self.real_data
        except Exception as e:
            print(f"Error loading real data: {e}")
            raise

    def load_real_data_with_schema(self,
                                file_path: str,
                                file_type: str,
                                schema_path: Optional[str] = None,
                                schema_yaml: Optional[str] = None) -> pd.DataFrame:
        """
        使用 YAML Schema 通过 adapters 将任意真实数据映射到统一 RL 结构，
        并落地到 self.real_data / self.current_meta。
        """
        from schema import SchemaSpec
        from adapters import TabularAdapter, SensorAdapter

        # 1) 读取 Schema
        if schema_yaml:
            spec = SchemaSpec.from_yaml_text(schema_yaml)
        elif schema_path:
            with open(schema_path, "r", encoding="utf-8") as f:
                spec = SchemaSpec.from_yaml_text(f.read())
        else:
            raise ValueError("真实数据需要提供 schema_path 或 schema_yaml 才能映射到统一结构")

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
        for i, name in enumerate(feat_names):
            safe = str(name).strip().replace(" ", "_").lower()
            df[f"state_{safe}"] = X[:, i].astype(float)

        # 4) 落地到管理器
        self.real_data = df
        self.current_source = "real"
        self.real_data_path = file_path
        self.current_meta = meta  # <— 供 UI/推理使用

        print(f"[DataManager] Real data (schema) loaded: {len(df)} rows, "
            f"{df['patient_id'].nunique()} patients, {len(feat_names)} features.")
        return df

    def load_real_data_schema_less(self, file_path: str) -> "pd.DataFrame":

        import pandas as pd
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

        # 3) action / reward / terminal
        if "action" not in df.columns:
            for c in ["action_id", "treatment_id", "act"]:
                if c in df.columns:
                    df.rename(columns={c: "action"}, inplace=True); break
        if "reward" not in df.columns:
            for c in ["r", "return", "sofa_delta"]:
                if c in df.columns:
                    df.rename(columns={c: "reward"}, inplace=True); break
            if "reward" not in df.columns:
                df["reward"] = 0.0

        if "terminal" not in df.columns:
            for c in ["done", "is_terminal", "terminal_flag"]:
                if c in df.columns:
                    df.rename(columns={c: "terminal"}, inplace=True); break
            if "terminal" not in df.columns:
                # 每个病人最后一条视为终止
                df["terminal"] = (df["patient_id"] != df["patient_id"].shift(-1)).astype(int)

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

        # 5) 更新管理器缓存与 meta
        self.real_data = df
        self.real_data_path = file_path
        self.current_source = "real"
        self.current_meta = {
            "feature_columns": feature_cols,
            "action_names": None,   # 可在 UI 里按需要推断/显示 id
            "action_map": None,
        }
        print(f"[DataManager] Real schema-less loaded: {len(df)} rows, "
            f"{df['patient_id'].nunique()} patients, {len(feature_cols)} features.")
        return df

    def set_data_source(self, source: str):
        """设置当前使用的数据源"""
        if source not in ["virtual", "real"]:
            raise ValueError("Source must be 'virtual' or 'real'")
        self.current_source = source
        print(f"Data source set to: {source}")
        if source == "virtual":
            if self.virtual_data is None:
                self.generate_virtual_data()
            if self.current_meta is None:
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
                self.current_meta = {
                    "feature_columns": feature_cols,
                    "action_names": action_names if action_names else None,
                    "action_map": {
                        a: name for a, name in zip(unique_actions, action_names)
                    } if unique_actions else None,
                }    
    def get_current_data(self) -> pd.DataFrame:
        """获取当前激活的数据"""
        if self.current_source == "virtual":
            if self.virtual_data is None:
                self.generate_virtual_data()
            return self.virtual_data
        else:
            if self.real_data is None:
                raise ValueError("No real data loaded. Please load data first.")
            return self.real_data
    
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
        patient_info = {
            'patient_id': patient_id,
            'total_records': len(patient_data),
            'current_state': self._extract_state_from_record(latest_record),
            'trajectory': self._get_patient_trajectory(patient_data),
            'treatment_history': patient_data['action'].tolist(),
            'outcome_history': patient_data['reward'].tolist()
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
            
            # 转换特征名称和值
            if feature_name == 'age':
                state['age'] = int(value * 72 + 18)  # 转换回实际年龄
            elif feature_name == 'gender':
                state['gender'] = int(value)
            else:
                state[feature_name] = float(value)
        
        # 添加其他信息
        state['timestep'] = int(record.get('timestep', 0))
        state['last_action'] = int(record.get('action', -1))
        state['last_reward'] = float(record.get('reward', 0))
        
        return state
    
    def _get_patient_trajectory(self, patient_data: pd.DataFrame) -> List[Dict]:
        """获取患者的完整轨迹"""
        trajectory = []
        
        for _, record in patient_data.iterrows():
            state = self._extract_state_from_record(record)
            trajectory.append({
                'timestep': state['timestep'],
                'state': state,
                'action': int(record['action']),
                'reward': float(record['reward'])
            })
        
        return trajectory
    
    def get_cohort_statistics(self, filter_criteria: Optional[Dict] = None) -> Dict:
        """获取队列统计信息"""
        data = self.get_current_data()
        
        # 应用过滤条件
        if filter_criteria:
            for key, value in filter_criteria.items():
                if key in data.columns:
                    data = data[data[key] == value]
        
        # 计算统计信息
        stats = {
            'total_patients': data['patient_id'].nunique(),
            'total_records': len(data),
            'avg_trajectory_length': data.groupby('patient_id').size().mean(),
                'action_distribution': {
        str(k): int(v) 
        for k, v in data['action'].value_counts().to_dict().items()
    },
            'avg_reward': data['reward'].mean(),
            'feature_stats': {}
        }
        
        # 计算特征统计
        feature_columns = [col for col in data.columns if col.startswith('state_') and not col.startswith('next_state_')]
        for col in feature_columns:
            feature_name = col.replace('state_', '')
            stats['feature_stats'][feature_name] = {
                'mean': float(data[col].mean()),
                'std': float(data[col].std()),
                'min': float(data[col].min()),
                'max': float(data[col].max())
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
