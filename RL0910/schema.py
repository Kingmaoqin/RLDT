from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable, Union
from typing import Tuple

import textwrap

try:
    import yaml
except Exception:  # pragma: no cover - yaml is an optional dependency at runtime
    yaml = None

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

    # ------------------------------------------------------------------
    # YAML helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_yaml_available():
        if yaml is None:
            raise ImportError(
                "PyYAML is required to load schema definitions. "
                "Install it with `pip install pyyaml` or provide an already-parsed dict."
            )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchemaSpec":
        """Construct a SchemaSpec from a Python dictionary."""

        if not isinstance(data, dict):
            raise TypeError(
                f"SchemaSpec.from_dict expected a mapping, got {type(data).__name__}"
            )

        # 兼容带有顶层 schema 键的结构
        if "schema" in data and isinstance(data["schema"], dict):
            data = data["schema"]

        return cls(**data)

    @classmethod
    def from_yaml_text(cls, text: str) -> "SchemaSpec":
        """Create a SchemaSpec instance from a YAML string."""

        if text is None:
            raise ValueError("SchemaSpec.from_yaml_text requires a non-empty string")

        cleaned = textwrap.dedent(text).strip()
        if not cleaned:
            raise ValueError("Schema YAML is empty")

        cls._ensure_yaml_available()

        parsed = yaml.safe_load(cleaned)
        if parsed is None:
            raise ValueError("Schema YAML did not contain any data")

        if isinstance(parsed, list):
            if not parsed:
                raise ValueError("Schema YAML contained an empty list")
            parsed = parsed[-1]

        return cls.from_dict(parsed)

    @classmethod
    def from_yaml_file(cls, path: str, encoding: str = "utf-8") -> "SchemaSpec":
        """Load SchemaSpec directly from a YAML file path."""

        with open(path, "r", encoding=encoding) as handle:
            return cls.from_yaml_text(handle.read())
