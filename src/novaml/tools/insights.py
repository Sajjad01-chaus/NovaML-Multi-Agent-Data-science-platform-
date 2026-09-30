"""Deterministic EDA facts. Always computed, whether or not an LLM is available.

They ground the analyst (the LLM starts from verified facts rather than guessing)
and are the offline fallback when no model is configured.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_insights(df: pd.DataFrame, target: str, problem_type: str, profile: dict[str, Any]) -> dict[str, Any]:
    y = df[target]
    X = df.drop(columns=[target])
    insights: list[str] = []
    leakage: list[str] = []
    cols = profile["columns"]

    # Association with target
    assoc: dict[str, float] = {}
    y_num = y.astype("category").cat.codes if problem_type == "classification" else pd.to_numeric(y, errors="coerce")
    for c in X.columns:
        info = cols.get(c, {})
        if info.get("id_like") or info.get("constant"):
            continue
        if info.get("kind") == "numeric":
            x = pd.to_numeric(X[c], errors="coerce")
            if problem_type == "regression":
                r = x.corr(y_num)
            else:
                r = _correlation_ratio(y.astype(str), x)
        else:
            r = _cramers_v(X[c].astype(str), y.astype(str)) if problem_type == "classification" else _correlation_ratio(X[c].astype(str), y_num)
        if r is not None and np.isfinite(r):
            assoc[c] = round(float(abs(r)), 4)
    top = sorted(assoc.items(), key=lambda kv: kv[1], reverse=True)[:5]
    if top:
        insights.append("strongest associations with target: " + ", ".join(f"{k} ({v:.2f})" for k, v in top))
    for c, v in assoc.items():
        if v > 0.99:  # strong-but-legitimate signal (e.g. iris petals, eta~0.97) stays
            leakage.append(c)
    if leakage:
        insights.append(f"possible target leakage (near-perfect association): {leakage}")

    # Missingness
    heavy = [c for c, i in cols.items() if i["missing_pct"] > 30]
    if heavy:
        insights.append(f"columns with >30% missing: {heavy}")
    predictive_missing = []
    for c in X.columns[X.isna().any()]:
        m = X[c].isna()
        # Standardised difference in target between rows with and without the value.
        gap = abs(y_num[m].mean() - y_num[~m].mean()) / (y_num.std() or 1)
        if m.sum() >= 10 and gap > 0.3:
            predictive_missing.append(c)
    if predictive_missing:
        insights.append(f"missingness looks predictive of the target in: {predictive_missing}")

    # Shape / type issues
    ids = [c for c, i in cols.items() if i["id_like"]]
    if ids:
        insights.append(f"identifier-like columns (should be dropped): {ids}")
    high_card = [c for c, i in cols.items() if i.get("kind") == "categorical" and i["n_unique"] > 50 and not i["id_like"]]
    if high_card:
        insights.append(f"high-cardinality categoricals (rare levels will be pooled): {high_card}")
    skewed = [c for c, i in cols.items() if i.get("kind") == "numeric" and abs(i.get("skew") or 0) > 2]
    if skewed:
        insights.append(f"heavily skewed numeric columns: {skewed}")
    if problem_type == "classification":
        share = profile["target"].get("minority_share")
        if share is not None and share < 0.2:
            insights.append(f"imbalanced target: minority class share {share:.1%}")
    if profile.get("duplicate_rows"):
        insights.append(f"{profile['duplicate_rows']} duplicate rows in training data")

    return {"insights": insights, "leakage_suspects": leakage, "target_association": dict(top)}


def _correlation_ratio(categories: pd.Series, values: pd.Series) -> float | None:
    """Eta: how much of a numeric variable's variance is explained by a categorical."""
    d = pd.DataFrame({"c": categories, "v": values}).dropna()
    if d.empty or d["v"].var() == 0:
        return None
    means = d.groupby("c")["v"].agg(["mean", "count"])
    between = (means["count"] * (means["mean"] - d["v"].mean()) ** 2).sum()
    total = ((d["v"] - d["v"].mean()) ** 2).sum()
    return float(np.sqrt(between / total)) if total else None


def _cramers_v(a: pd.Series, b: pd.Series) -> float | None:
    ct = pd.crosstab(a, b)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return None
    n = ct.values.sum()
    expected = np.outer(ct.sum(1), ct.sum(0)) / n
    chi2 = ((ct.values - expected) ** 2 / expected).sum()
    k = min(ct.shape) - 1
    return float(np.sqrt(chi2 / (n * k)))
