from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable, Union
from typing import Tuple

# 常用列名别名池（自动推断用）
ALIASES = {
    "trajectory_id": {"trajectory_id", "traj_id", "traj", "episode", "episode_id", "patient_id", "subject", "subject_id", "id"},
    "timestep":      {"timestep", "time", "step", "index", "frame", "ts"},
    "action":        {"action", "treatment", "arm", "action_id", "a", "label"},
    "reward":        {"reward", "outcome", "utility", "score", "r"},
    "terminal":      {"done", "terminal", "is_terminal", "is_done", "absorbing"},
}

# 常用生理信号别名（可选）
FEATURE_ALIASES = {
    "spo2": {"spo2", "o2", "oxygen", "pulseox", "oxygen_saturation", "spO2", "O2_Sat"},
}

@dataclass
class ColumnMapping:
    trajectory_id: Optional[str] = None
    timestep: Optional[str] = None
    action: Optional[str] = None
    reward: Optional[str] = None
    terminal: Optional[str] = None
    feature_cols: List[str] = field(default_factory=list)

@dataclass
class WindowingSpec:
    enabled: bool = False
    length: int = 10
    stride: int = 1
    label_col: Optional[str] = None
    label_to_action: Optional[Dict[Any, int]] = None
    derive_action_fn: Optional[Callable[[Any], int]] = None

@dataclass
class NormalizationSpec:
    method: str = "standard"  # "standard" | "minmax" | "none"
    clip_min: Optional[float] = 0.0
    clip_max: Optional[float] = 1.0

# === 放在 SchemaSpec 之前：关键特征规则 & 奖励派生规范 ===
@dataclass
class CriticalFeatureRule:
    name_or_aliases: Union[str, List[str], None] = None
    index: Optional[int] = None
    op: str = ">"
    threshold: float = 0.0
    weight: float = 1.0
    as_terminal: bool = False
    display_name: Optional[str] = None

@dataclass
class RewardSpec:
    column: Optional[str] = None
    label_col: Optional[str] = None
    label_to_reward: Optional[Dict[Any, float]] = None
    expression: Optional[str] = None
    window_agg: str = "last"  # "last" | "mean" | "sum"

@dataclass
class SchemaSpec:
    data_type: str = "tabular"   # "tabular" | "sensor"
    mapping: ColumnMapping = field(default_factory=ColumnMapping)
    window: WindowingSpec = field(default_factory=WindowingSpec)
    normalization: NormalizationSpec = field(default_factory=NormalizationSpec)
    action_names: Optional[List[str]] = None
    feature_names: Optional[List[str]] = None

    # 兼容旧字段（如果你还在用的话，不用也没关系）
    critical_feature_alias: Optional[str] = "spo2"
    spo2_threshold: float = 0.80

    # 新字段：完全通用的关键特征与奖励派生
    critical_features: List[CriticalFeatureRule] = field(default_factory=list)
    reward_spec: Optional[RewardSpec] = None

    def __post_init__(self):
        # 允许从 YAML dict 自动转换为 dataclass
        if isinstance(self.mapping, dict):
            self.mapping = ColumnMapping(**self.mapping)
        if isinstance(self.window, dict):
            self.window = WindowingSpec(**self.window)
        if isinstance(self.normalization, dict):
            self.normalization = NormalizationSpec(**self.normalization)
        if self.critical_features and isinstance(self.critical_features[0], dict):
            self.critical_features = [CriticalFeatureRule(**d) for d in self.critical_features]
        if isinstance(self.reward_spec, dict):
            self.reward_spec = RewardSpec(**self.reward_spec)
