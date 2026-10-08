"""Data loading, validation and profiling tools (pure functions, no agent state)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ProblemType = Literal["classification", "regression"]


class DataValidationError(ValueError):
    """Input data is unusable; surfaced to the user as-is."""


def load_dataset(
    path: str | Path,
    *,
    allowed_extensions: tuple[str, ...],
    max_mb: int,
    max_rows: int,
) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise DataValidationError(f"dataset not found: {p.name}")
    ext = p.suffix.lower()
    if ext not in allowed_extensions:
        raise DataValidationError(f"unsupported file type {ext!r}; allowed: {allowed_extensions}")
    size_mb = p.stat().st_size / 1_048_576
    if size_mb > max_mb:
        raise DataValidationError(f"file is {size_mb:.1f} MB, limit is {max_mb} MB")

    if ext == ".csv":
        df = pd.read_csv(p, nrows=max_rows + 1, low_memory=False)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(p, nrows=max_rows + 1)
    else:
        df = pd.read_parquet(p)
    if len(df) > max_rows:
        raise DataValidationError(f"dataset has more than {max_rows} rows")
    df.columns = [str(c).strip() for c in df.columns]
    if df.columns.duplicated().any():
        raise DataValidationError("dataset has duplicate column names")
    return df


def validate_target(df: pd.DataFrame, target: str, min_rows: int) -> pd.DataFrame:
    """Checks the target column and drops rows where it is missing."""
    if target not in df.columns:
        raise DataValidationError(
            f"target column {target!r} not found. Columns: {', '.join(map(str, df.columns[:30]))}"
        )
    df = df.loc[df[target].notna()].reset_index(drop=True)
    if len(df) < min_rows:
        raise DataValidationError(f"only {len(df)} rows with a non-null target (need {min_rows})")
    if df[target].nunique() < 2:
        raise DataValidationError(f"target {target!r} has a single value; nothing to learn")
    return df


def infer_problem_type(y: pd.Series) -> ProblemType:
    if pd.api.types.is_bool_dtype(y) or not pd.api.types.is_numeric_dtype(y):
        return "classification"
    n_unique = y.nunique()
    if pd.api.types.is_float_dtype(y) and not np.allclose(y.dropna() % 1, 0):
        return "regression"
    # Integer-valued: few distinct values reads as class labels.
    return "classification" if n_unique <= max(20, int(0.01 * len(y))) and n_unique <= 50 else "regression"


def looks_like_id(s: pd.Series) -> bool:
    """Unique-per-row identifiers leak row identity and never generalise."""
    n = len(s)
    if n < 20 or s.nunique(dropna=True) < 0.98 * n:
        return False
    if pd.api.types.is_float_dtype(s) or looks_like_datetime(s):
        return False  # unique timestamps are features (expanded later), not identifiers
    name = str(s.name).lower()
    if name in {"id", "uuid", "index"} or name.endswith(("_id", " id", "-id")):
        return True
    if pd.api.types.is_integer_dtype(s):
        return bool(s.is_monotonic_increasing or s.is_monotonic_decreasing)
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def looks_like_datetime(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
        return False
    sample = s.dropna().astype(str).head(200)
    if sample.empty or sample.str.len().median() < 6:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() > 0.9


def profile(df: pd.DataFrame, target: str) -> dict[str, Any]:
    """A compact, JSON-safe profile. This is what agents (and LLMs) see, never raw rows."""
    features = [c for c in df.columns if c != target]
    columns = {}
    for c in features:
        s = df[c]
        info: dict[str, Any] = {
            "dtype": str(s.dtype),
            "missing_pct": round(float(s.isna().mean() * 100), 2),
            "n_unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            desc = s.describe()
            info.update(
                kind="numeric",
                mean=_f(desc.get("mean")),
                std=_f(desc.get("std")),
                min=_f(desc.get("min")),
                max=_f(desc.get("max")),
                skew=_f(s.skew()),
            )
        elif looks_like_datetime(s):
            info["kind"] = "datetime"
        else:
            info["kind"] = "categorical"
            info["top_values"] = {str(k): int(v) for k, v in s.value_counts().head(5).items()}
        info["id_like"] = looks_like_id(s)
        info["constant"] = info["n_unique"] <= 1
        columns[c] = info

    y = df[target]
    problem_type = infer_problem_type(y)
    target_info: dict[str, Any] = {"name": target, "dtype": str(y.dtype), "n_unique": int(y.nunique())}
    if problem_type == "classification":
        dist = y.value_counts(normalize=True)
        target_info["class_balance"] = {str(k): round(float(v), 4) for k, v in dist.head(20).items()}
        target_info["minority_share"] = round(float(dist.min()), 4)
    else:
        target_info.update(mean=_f(y.mean()), std=_f(y.std()), skew=_f(y.skew()))

    return {
        "n_rows": int(len(df)),
        "n_features": len(features),
        "duplicate_rows": int(df.duplicated().sum()),
        "total_missing": int(df[features].isna().sum().sum()),
        "inferred_problem_type": problem_type,
        "target": target_info,
        "columns": columns,
    }


def split(
    df: pd.DataFrame, target: str, problem_type: ProblemType, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Holdout split done once, up front, so nothing downstream can peek at test rows."""
    stratify = None
    if problem_type == "classification":
        counts = df[target].value_counts()
        if counts.min() >= 2:
            stratify = df[target]
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=stratify)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def _f(v: Any) -> float | None:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(v) or np.isinf(v) else round(v, 4)
