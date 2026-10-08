"""Deterministic EDA facts. Always computed, whether or not an LLM is available.

They ground the analyst (the LLM starts from verified facts rather than guessing)
and are the offline fallback when no model is configured.

Feature strength is measured by *cross-validated single-feature models*
(a shallow tree on one column) rather than correlation statistics: CV scores are
comparable across numeric and categorical columns, are not inflated by
high-cardinality columns (Cramér's V is: `ticket` on Titanic scores 0.9), and
expose post-outcome leakage that only shows up through missingness (e.g. a
`boat` column that is filled in only for survivors).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

MAX_ROWS = 5_000
MAX_FEATURES = 100
TOP_LEVELS = 30

# A column is a leakage suspect when, on its own, it predicts the target almost
# perfectly AND far better than every other column combined. The second condition
# separates leaks (Titanic `boat`: 0.97 alone vs 0.78 for everything else) from
# legitimately strong features (iris `petal width`: 0.95 alone vs 0.95 for the rest).
LEAK_SCORE = {"classification": 0.95, "regression": 0.95}
LEAK_MISSING_ONLY = {"classification": 0.90, "regression": 0.80}
LEAK_MARGIN = {"classification": 0.10, "regression": 0.20}


def _encode(s: pd.Series, kind: str) -> pd.DataFrame:
    miss = s.isna().astype(float)
    if kind == "numeric":
        x = pd.to_numeric(s, errors="coerce")
        x = x.fillna(x.median() if x.notna().any() else 0.0)
    else:
        v = s.astype("string").fillna("__missing__")
        top = v.value_counts().index[:TOP_LEVELS]
        x = pd.Series(pd.Categorical(v.where(v.isin(top), "__other__")).codes, index=s.index, dtype=float)
    return pd.DataFrame({"x": x.to_numpy(), "missing": miss.to_numpy()})


def _cv_score(X: pd.DataFrame, y: pd.Series, problem_type: str, seed: int, flexible: bool = False) -> float:
    n = len(y)
    leaf = max(5, n // 100)
    if flexible:  # multi-feature model for the "everything else" comparison
        model = (
            HistGradientBoostingClassifier(max_iter=60, random_state=seed)
            if problem_type == "classification"
            else HistGradientBoostingRegressor(max_iter=60, random_state=seed)
        )
    elif problem_type == "classification":
        model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=leaf, random_state=seed)
    else:
        model = DecisionTreeRegressor(max_depth=3, min_samples_leaf=leaf, random_state=seed)
    if problem_type == "classification":
        cv = StratifiedKFold(3, shuffle=True, random_state=seed) if y.value_counts().min() >= 3 else KFold(3, shuffle=True, random_state=seed)
        scoring = "balanced_accuracy"
    else:
        cv, scoring = KFold(3, shuffle=True, random_state=seed), "r2"
    return float(np.mean(cross_val_score(model, X, y, cv=cv, scoring=scoring)))


def feature_strength(df: pd.DataFrame, target: str, problem_type: str, profile: dict[str, Any], seed: int = 0) -> dict[str, dict[str, float]]:
    """CV score of a shallow tree on each column alone, and on its missingness alone."""
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=seed)
    y = df[target]
    if problem_type == "regression":
        y = pd.to_numeric(y, errors="coerce")
    out: dict[str, dict[str, float]] = {}
    cols = [c for c, i in profile["columns"].items() if not i.get("constant") and c in df.columns][:MAX_FEATURES]
    for c in cols:
        kind = profile["columns"][c].get("kind", "categorical")
        X = _encode(df[c], kind)
        scores = {"score": round(_cv_score(X, y, problem_type, seed), 4)}
        if X["missing"].mean() >= 0.05:
            scores["missing_only"] = round(_cv_score(X[["missing"]], y, problem_type, seed), 4)
        out[c] = scores
    return out


def rest_score(df: pd.DataFrame, target: str, problem_type: str, profile: dict[str, Any], exclude: str, seed: int = 0) -> float:
    """CV score of a flexible model on every usable column except `exclude`."""
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=seed)
    y = df[target] if problem_type == "classification" else pd.to_numeric(df[target], errors="coerce")
    usable = [
        c for c, i in profile["columns"].items()
        if c != exclude and c in df.columns and not i.get("id_like") and not i.get("constant")
    ][:MAX_FEATURES]
    if not usable:
        return 0.0 if problem_type == "regression" else 1 / max(2, y.nunique())
    parts = [_encode(df[c], profile["columns"][c].get("kind", "categorical")).add_prefix(f"{i}_") for i, c in enumerate(usable)]
    return _cv_score(pd.concat(parts, axis=1), y, problem_type, seed, flexible=True)


def compute_insights(df: pd.DataFrame, target: str, problem_type: str, profile: dict[str, Any]) -> dict[str, Any]:
    cols = profile["columns"]
    insights: list[str] = []
    strength = feature_strength(df, target, problem_type, profile)
    chance = 1 / max(2, profile["target"]["n_unique"]) if problem_type == "classification" else 0.0
    metric = "balanced_accuracy" if problem_type == "classification" else "r2"

    ranked = sorted(((c, s["score"]) for c, s in strength.items() if not cols[c].get("id_like")), key=lambda kv: kv[1], reverse=True)
    top = ranked[:5]
    if top:
        insights.append(
            f"strongest single-feature predictors (CV {metric}, chance={chance:.2f}): "
            + ", ".join(f"{c} ({v:.2f})" for c, v in top)
        )

    leakage: list[str] = []
    for c, s in strength.items():
        if cols[c].get("id_like"):
            continue
        by_value = s["score"] >= LEAK_SCORE[problem_type]
        by_missing = s.get("missing_only", -1) >= LEAK_MISSING_ONLY[problem_type]
        if not (by_value or by_missing):
            continue
        rest = rest_score(df, target, problem_type, profile, exclude=c)
        if max(s["score"], s.get("missing_only", -1)) - rest < LEAK_MARGIN[problem_type]:
            insights.append(f"{c!r} is very predictive ({metric}={s['score']:.2f}) but so are the other columns ({rest:.2f}); treated as real signal")
            continue
        leakage.append(c)
        if by_missing and not by_value:
            how = f"whether {c!r} is missing alone reaches {metric}={s['missing_only']:.2f}"
        else:
            how = f"{c!r} alone reaches {metric}={s['score']:.2f}"
        insights.append(f"possible target leakage: {how} vs {rest:.2f} for all other columns combined (likely recorded after the outcome)")

    predictive_missing = [
        c for c, s in strength.items() if c not in leakage and s.get("missing_only", -1) >= chance + 0.1
    ]
    if predictive_missing:
        insights.append(f"missingness itself is predictive in: {predictive_missing}")

    heavy = [c for c, i in cols.items() if i["missing_pct"] > 30]
    if heavy:
        insights.append(f"columns with >30% missing: {heavy}")
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

    return {
        "insights": insights,
        "leakage_suspects": sorted(leakage),
        "feature_strength": dict(top),
        "chance_score": round(chance, 4),
    }
