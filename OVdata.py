#!/usr/bin/env python3
"""Generate an offline-ready dataset from the CPTAC ovarian cohort.

This utility downloads (if necessary) the CPTAC ovarian multi-modal dataset
and converts the proteomics, transcriptomics, and clinical tables into a
normalized feature table compatible with the RLDT offline pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# Guard against older pyarrow wheels compiled for NumPy 1.x which can cause
# `_ARRAY_API` errors when pandas enables the Arrow backend automatically.
os.environ.setdefault("PANDAS_USE_PYARROW_BACKEND", "0")
os.environ.setdefault("PANDAS_USE_PYARROW_EXTENSION_ARRAY", "0")

import numpy as np
from RL0910.pandas_compat import get_pandas
pd = get_pandas()


def _import_cptac():
    """Import the cptac package with helpful diagnostics."""
    try:
        import cptac  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        if exc.__class__.__name__ == "NoInternetError":
            print(
                "[ERROR] CPTAC index missing. Connect to the internet and rerun,\n"
                "or execute `python -m cptac download` once to bootstrap the index.",
                file=sys.stderr,
            )
            sys.exit(1)
        raise

    from cptac.exceptions import (  # type: ignore
        NoInternetError,
        DataSourceNotFoundError,
        MissingFileError,
    )

    return cptac, NoInternetError, DataSourceNotFoundError, MissingFileError


def _ensure_dataset(
    cptac_module,
    no_internet_error,
    datasource_error,
    missing_file_error,
):
    """Load the CPTAC ovarian dataset, downloading it if necessary."""
    download_attempted = False
    try:
        dataset = cptac_module.Ov()
        return dataset
    except no_internet_error as err:  # pragma: no cover - depends on connectivity
        print(f"[ERROR] {err}", file=sys.stderr)
        print(
            "Please connect to the CPTAC data portal and rerun, or download the dataset manually.",
            file=sys.stderr,
        )
        sys.exit(1)
    except (datasource_error, missing_file_error, FileNotFoundError):
        download_attempted = True
    except Exception as err:  # pragma: no cover - defensive fallback
        print(f"[WARN] Initial CPTAC load failed: {err}", file=sys.stderr)
        download_attempted = True

    if download_attempted:
        print("Attempting to download CPTAC ovarian assets (requires internet)...")
        last_error: Exception | None = None
        for cancer_key in ("ov", "ovarian", "Ovarian"):
            try:
                cptac_module.download_cancer(cancer_key)
                last_error = None
                break
            except Exception as inner:  # pragma: no cover - depends on remote service
                last_error = inner

        if last_error is not None:
            print(f"[ERROR] Unable to download CPTAC ovarian dataset: {last_error}", file=sys.stderr)
            sys.exit(1)

        try:
            dataset = cptac_module.Ov()
            return dataset
        except Exception as err:  # pragma: no cover - defensive
            print(f"[ERROR] CPTAC dataset still unavailable after download: {err}", file=sys.stderr)
            sys.exit(1)

    raise RuntimeError("Unexpected CPTAC import state")


def _standardize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a flat string index keyed by patient identifier."""
    out = df.copy()
    if isinstance(out.index, pd.MultiIndex):
        out.index = out.index.get_level_values(0)
    out.index = out.index.astype(str)
    out = out[~out.index.duplicated(keep="first")]
    return out


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns to single strings."""
    if isinstance(df.columns, pd.MultiIndex):
        flat = []
        for col in df.columns:
            pieces = [str(level) for level in col if level not in (None, "", "nan")]
            label = "_".join(pieces).strip("_") or "feature"
            flat.append(label)
        df = df.copy()
        df.columns = flat
    else:
        df = df.copy()
        df.columns = [str(c) for c in df.columns]
    return df


def _safe_label(label: str) -> str:
    """Create a filesystem and YAML friendly slug."""
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in label)
    cleaned = cleaned.strip("_").lower()
    return cleaned or "feature"


def _unique_name(base: str, registry: set[str]) -> str:
    name = base
    counter = 2
    while name in registry:
        name = f"{base}_{counter}"
        counter += 1
    registry.add(name)
    return name


def _select_top_variance(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if top_n <= 0 or df.empty:
        return df
    if df.shape[1] <= top_n:
        return df
    variances = df.var(axis=0, skipna=True)
    keep = variances.sort_values(ascending=False).head(top_n).index
    return df.loc[:, keep]


def _prepare_modality(
    raw_df: pd.DataFrame,
    modality_name: str,
    prefix: str,
    top_n: int,
    min_coverage: float,
    name_registry: set[str],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float | str]]]:
    """Convert a CPTAC table into numeric features and collect statistics."""
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(), {}

    df = _flatten_columns(_standardize_index(raw_df))
    if df.empty:
        return df, {}

    numeric = {}
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            numeric[col] = pd.to_numeric(series, errors="coerce")
        else:
            numeric[col] = series.astype("category").cat.codes.replace(-1, np.nan)
    numeric_df = pd.DataFrame(numeric, index=df.index).astype(float)
    if numeric_df.empty:
        return numeric_df, {}

    coverage = numeric_df.notna().mean()
    keep_cols = coverage[coverage >= min_coverage].index.tolist()
    numeric_df = numeric_df.loc[:, keep_cols]
    if numeric_df.empty:
        return numeric_df, {}

    medians = numeric_df.median()
    filled = numeric_df.fillna(medians)
    filtered = _select_top_variance(filled, top_n)
    medians = medians.loc[filtered.columns]
    coverage = coverage.loc[filtered.columns]
    variances = filtered.var(axis=0, skipna=True)

    rename_map: Dict[str, str] = {}
    stats: Dict[str, Dict[str, float | str]] = {}
    for original in filtered.columns:
        base = _safe_label(original)
        unique = _unique_name(f"{prefix}_{base}", name_registry)
        state_name = f"state_{unique}"
        rename_map[original] = state_name
        stats[state_name] = {
            "original_name": original,
            "modality": modality_name,
            "coverage": float(coverage.get(original, float("nan"))),
            "median": float(medians.get(original, float("nan"))),
            "variance": float(variances.get(original, float("nan"))),
        }

    prepared = filtered.rename(columns=rename_map)
    return prepared, stats


def _combine_modalities(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    indices = pd.Index([])
    dataframes = [df for df in frames if not df.empty]
    for df in dataframes:
        indices = indices.union(df.index)
    combined = pd.DataFrame(index=indices.sort_values())
    for df in dataframes:
        combined = combined.join(df, how="left")
    return combined


def _normalize_features(
    df: pd.DataFrame,
    apply_minmax: bool,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    stats: Dict[str, Dict[str, float]] = {}
    if df.empty:
        return df, stats

    if apply_minmax:
        mins = df.min(axis=0)
        maxs = df.max(axis=0)
        denom = (maxs - mins).replace(0, np.nan)
        scaled = (df - mins) / denom
        scaled = scaled.clip(0.0, 1.0)
        scaled = scaled.fillna(0.5)
        for col in scaled.columns:
            stats[col] = {
                "min": float(mins.get(col, float("nan"))),
                "max": float(maxs.get(col, float("nan"))),
            }
        return scaled, stats

    filled = df.copy()
    medians = filled.median(axis=0)
    filled = filled.fillna(medians)
    for col in filled.columns:
        stats[col] = {
            "mean": float(filled[col].mean()),
            "std": float(filled[col].std(ddof=0)),
        }
    return filled, stats


def _derive_actions(
    risk: pd.Series,
    desired_actions: int,
) -> Tuple[pd.Series, List[str]]:
    if risk.empty:
        return pd.Series(dtype=int), ["Undefined"]

    unique_values = risk.nunique(dropna=True)
    n_actions = max(1, min(desired_actions, unique_values))

    if n_actions <= 1:
        actions = pd.Series(0, index=risk.index, dtype=int)
        return actions, ["Single bucket"]

    quantiles = pd.qcut(
        risk.rank(method="first"),
        q=n_actions,
        labels=False,
        duplicates="drop",
    ).astype(int)
    action_names = [f"Risk quantile {i + 1}/{len(quantiles.unique())}" for i in range(len(quantiles.unique()))]
    return quantiles, action_names


def _yaml_quote(value: str) -> str:
    if value == "":
        return "''"
    if any(ch in value for ch in ":{}[]&,#|>!-?%*@\"'\t\n\r "):
        return json.dumps(value, ensure_ascii=False)
    return value


def build_dataset(args: argparse.Namespace) -> None:
    cptac_module, no_internet_error, datasource_error, missing_file_error = _import_cptac()
    dataset = _ensure_dataset(cptac_module, no_internet_error, datasource_error, missing_file_error)

    np.random.seed(args.seed)

    proteomics = dataset.get_proteomics()
    transcriptomics = dataset.get_transcriptomics()
    clinical = dataset.get_clinical()

    name_registry: set[str] = set()
    prot_df, prot_stats = _prepare_modality(
        proteomics,
        "proteomics",
        "prot",
        args.max_proteins,
        args.min_coverage,
        name_registry,
    )
    rna_df, rna_stats = _prepare_modality(
        transcriptomics,
        "transcriptomics",
        "rna",
        args.max_transcripts,
        args.min_coverage,
        name_registry,
    )
    clin_df, clin_stats = _prepare_modality(
        clinical,
        "clinical",
        "clin",
        args.max_clinical,
        max(0.35, args.min_coverage * 0.5),
        name_registry,
    )

    combined = _combine_modalities([prot_df, rna_df, clin_df])
    if combined.empty:
        print(
            "[ERROR] No features satisfied the filtering criteria. Consider lowering --min-coverage or increasing --max-* limits.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Fill residual gaps with column medians before normalization
    medians = combined.median(axis=0)
    combined = combined.fillna(medians)

    normalized, scaling_stats = _normalize_features(combined, not args.no_normalize)
    state_columns = list(normalized.columns)
    if not state_columns:
        print("[ERROR] Normalized feature matrix is empty after preprocessing.", file=sys.stderr)
        sys.exit(1)

    risk_score = normalized[state_columns].mean(axis=1)
    actions, action_names = _derive_actions(risk_score, args.actions)
    rewards = (0.5 - risk_score) * 2.0
    rewards = rewards.clip(-1.0, 1.0)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_df = pd.DataFrame(
        {
            "patient_id": normalized.index.astype(str),
            "timestep": 0,
            "action": actions.reindex(normalized.index).fillna(0).astype(int),
            "reward": rewards.reindex(normalized.index).astype(float),
            "terminal": True,
        }
    )
    dataset_df = dataset_df.join(normalized[state_columns])

    csv_path = output_dir / "ovarian_offline_dataset.csv"
    dataset_df.to_csv(csv_path, index=False)

    npz_path = output_dir / "ovarian_offline_dataset.npz"
    np.savez(
        npz_path,
        states=normalized[state_columns].to_numpy(dtype=np.float32),
        actions=dataset_df["action"].to_numpy(dtype=np.int64),
        rewards=dataset_df["reward"].to_numpy(dtype=np.float32),
        terminals=np.ones(len(dataset_df), dtype=np.bool_),
        patient_ids=dataset_df["patient_id"].to_numpy(dtype="U"),
        timesteps=dataset_df["timestep"].to_numpy(dtype=np.int64),
        state_columns=np.array(state_columns),
        action_names=np.array(action_names, dtype="U"),
        risk_scores=risk_score.reindex(normalized.index).to_numpy(dtype=np.float32),
    )

    feature_catalog = {**prot_stats, **rna_stats, **clin_stats}
    for column, stats in feature_catalog.items():
        stats.update(scaling_stats.get(column, {}))

    feature_labels = {
        column: f"{stats['modality']}::{stats['original_name']}"
        for column, stats in feature_catalog.items()
    }

    schema_lines = [
        "data_type: tabular",
        "mapping:",
        "  trajectory_id: patient_id",
        "  timestep: timestep",
        "  action: action",
        "  reward: reward",
        "  terminal: terminal",
        "  feature_cols:",
    ]
    for col in state_columns:
        schema_lines.append(f"    - {col}")

    if args.no_normalize:
        schema_lines.extend([
            "normalization:",
            "  method: none",
        ])
    else:
        schema_lines.extend([
            "normalization:",
            "  method: minmax",
            "  clip_min: 0.0",
            "  clip_max: 1.0",
        ])

    schema_lines.append("action_names:")
    for name in action_names:
        schema_lines.append(f"  - {_yaml_quote(name)}")

    schema_lines.append("feature_names:")
    for col in state_columns:
        schema_lines.append(f"  - {_yaml_quote(feature_labels.get(col, col))}")

    schema_lines.extend(
        [
            "reward_spec:",
            "  expression: normalized_risk_score",
            "  window_agg: last",
        ]
    )

    schema_path = output_dir / "ovarian_offline_schema.yaml"
    schema_path.write_text("\n".join(schema_lines) + "\n", encoding="utf-8")

    metadata = {
        "dataset": "CPTAC Ovarian (proteomics + transcriptomics + clinical)",
        "patients": int(dataset_df["patient_id"].nunique()),
        "records": int(len(dataset_df)),
        "features": len(state_columns),
        "actions": {
            "count": len(action_names),
            "names": action_names,
            "definition": "Risk quantiles derived from the normalized multi-modal feature mean",
        },
        "reward": {
            "range": [-1.0, 1.0],
            "definition": "Centered risk score (higher is better) from combined normalized features",
        },
        "normalization": "minmax" if not args.no_normalize else "none",
        "feature_catalog": feature_catalog,
        "files": {
            "csv": csv_path.name,
            "npz": npz_path.name,
            "schema": schema_path.name,
        },
    }

    metadata_path = output_dir / "ovarian_offline_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("✔ Dataset prepared")
    print(f"  • CSV table      : {csv_path}")
    print(f"  • Numpy archive  : {npz_path}")
    print(f"  • Schema YAML    : {schema_path}")
    print(f"  • Metadata JSON  : {metadata_path}")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and format the CPTAC ovarian cohort for RLDT offline workflows",
    )
    parser.add_argument(
        "--output-dir",
        default="RL0910/data/cptac_ovarian",
        help="Directory to store the generated dataset files (default: %(default)s)",
    )
    parser.add_argument(
        "--max-proteins",
        type=int,
        default=75,
        help="Maximum number of proteomic features to retain (variance-ranked).",
    )
    parser.add_argument(
        "--max-transcripts",
        type=int,
        default=75,
        help="Maximum number of transcriptomic features to retain (variance-ranked).",
    )
    parser.add_argument(
        "--max-clinical",
        type=int,
        default=40,
        help="Maximum number of engineered clinical features to retain.",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.65,
        help="Minimum fraction of patients required for a feature to be kept (0-1).",
    )
    parser.add_argument(
        "--actions",
        type=int,
        default=5,
        help="Number of action buckets derived from the risk score quantiles.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable min-max normalization (features remain on their native scale).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when deriving surrogate actions.",
    )

    args = parser.parse_args(argv)
    if not 0 < args.min_coverage <= 1:
        parser.error("--min-coverage must be within (0, 1].")
    if args.actions <= 0:
        parser.error("--actions must be a positive integer.")
    return args


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    build_dataset(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
