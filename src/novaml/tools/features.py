"""Declarative feature engineering.

Agents (LLM or policy) never write feature code that runs at serving time.
They emit a `FeaturePlan`: a small, validated, declarative spec. A deterministic
sklearn transformer executes it *inside* the model pipeline, so

* the same transformation runs at training and at inference (no train/serve skew),
* it is fitted within each CV fold (no leakage),
* and nothing an LLM produced is ever `exec`-ed in the serving path.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.base import BaseEstimator, TransformerMixin


class RatioFeature(BaseModel):
    numerator: str
    denominator: str


class FeaturePlan(BaseModel):
    drop_columns: list[str] = Field(default_factory=list, description="Columns to remove (IDs, leakage, constants).")
    log_transform: list[str] = Field(default_factory=list, description="Non-negative skewed numeric columns to log1p.")
    datetime_columns: list[str] = Field(default_factory=list, description="Columns to expand into year/month/day-of-week.")
    ratio_features: list[RatioFeature] = Field(default_factory=list, description="Numeric column ratios to add.")
    rationale: str = ""


def validate_plan(plan: FeaturePlan, profile: dict[str, Any], target: str) -> tuple[FeaturePlan, list[str]]:
    """Drop anything in the plan that references unknown/unsuitable columns."""
    cols: dict[str, dict] = profile["columns"]
    warnings: list[str] = []

    def keep(names: list[str], ok, why: str) -> list[str]:
        out = []
        for n in dict.fromkeys(names):
            if n == target:
                warnings.append(f"ignored {n!r}: target cannot be used in feature plan")
            elif n not in cols:
                warnings.append(f"ignored {n!r}: unknown column")
            elif not ok(cols[n]):
                warnings.append(f"ignored {n!r}: {why}")
            else:
                out.append(n)
        return out

    drop = keep(plan.drop_columns, lambda c: True, "")
    if len(drop) >= len(cols):
        warnings.append("plan would drop every feature; keeping all columns")
        drop = []
    numeric = lambda c: c.get("kind") == "numeric"  # noqa: E731
    log = keep(
        [c for c in plan.log_transform if c not in drop],
        lambda c: numeric(c) and (c.get("min") is None or c["min"] >= 0),
        "log needs a non-negative numeric column",
    )
    dt = keep(
        [c for c in plan.datetime_columns if c not in drop],
        lambda c: c.get("kind") in ("datetime", "categorical"),
        "not a date-like column",
    )
    ratios = []
    for r in plan.ratio_features:
        a = keep([r.numerator], numeric, "ratio needs numeric columns")
        b = keep([r.denominator], numeric, "ratio needs numeric columns")
        if a and b and a[0] != b[0] and a[0] not in drop and b[0] not in drop:
            ratios.append(RatioFeature(numerator=a[0], denominator=b[0]))
    clean = FeaturePlan(
        drop_columns=drop,
        log_transform=log,
        datetime_columns=dt,
        ratio_features=ratios[:10],
        rationale=plan.rationale,
    )
    return clean, warnings


def default_plan(profile: dict[str, Any]) -> FeaturePlan:
    """Rule-based plan used when no LLM is configured or the LLM call fails."""
    cols = profile["columns"]
    drop = [n for n, c in cols.items() if c["id_like"] or c["constant"] or c["missing_pct"] > 95]
    dt = [n for n, c in cols.items() if c.get("kind") == "datetime" and n not in drop]
    log = [
        n
        for n, c in cols.items()
        if c.get("kind") == "numeric"
        and n not in drop
        and (c.get("min") or 0) >= 0
        and abs(c.get("skew") or 0) > 2
    ]
    reasons = []
    if drop:
        reasons.append(f"drop identifier/constant/empty columns {drop}")
    if dt:
        reasons.append(f"expand dates {dt}")
    if log:
        reasons.append(f"log1p heavily skewed {log}")
    return FeaturePlan(
        drop_columns=drop,
        log_transform=log,
        datetime_columns=dt,
        rationale="; ".join(reasons) or "no transformations needed",
    )


class FeaturePlanTransformer(BaseEstimator, TransformerMixin):
    """Applies a `FeaturePlan` (given as a plain dict so the estimator pickles cleanly)."""

    def __init__(self, plan: dict | None = None):
        self.plan = plan

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        plan = FeaturePlan(**(self.plan or {}))
        X = X.copy()
        for r in plan.ratio_features:
            num = pd.to_numeric(X[r.numerator], errors="coerce")
            den = pd.to_numeric(X[r.denominator], errors="coerce")
            X[f"{r.numerator}_per_{r.denominator}"] = num / den.where(den != 0)
        for c in plan.log_transform:
            X[c] = np.log1p(pd.to_numeric(X[c], errors="coerce").clip(lower=0))
        for c in plan.datetime_columns:
            ts = pd.to_datetime(X[c], errors="coerce", format="mixed")
            X[f"{c}_year"] = ts.dt.year
            X[f"{c}_month"] = ts.dt.month
            X[f"{c}_dayofweek"] = ts.dt.dayofweek
            X = X.drop(columns=[c])
        X = X.drop(columns=[c for c in plan.drop_columns if c in X.columns])
        return X
