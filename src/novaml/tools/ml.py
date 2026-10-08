"""Model registry, leak-free pipelines, cross-validation, tuning and holdout metrics."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    get_scorer,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from novaml.tools.features import FeaturePlanTransformer

BASELINE = "baseline"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    description: str
    classifier: Callable[[int], Any]
    regressor: Callable[[int], Any]
    param_space: dict[str, Any] = field(default_factory=dict)

    def build(self, problem_type: str, seed: int, class_weight: str | None = None) -> Any:
        if problem_type != "classification":
            return self.regressor(seed)
        est = self.classifier(seed)
        if class_weight and "class_weight" in est.get_params():
            est.set_params(class_weight=class_weight)
        return est


# Factories, not instances: every run gets fresh estimators (the old module-level
# MODEL_MAP shared fitted objects across runs and users).
REGISTRY: dict[str, ModelSpec] = {
    s.name: s
    for s in [
        ModelSpec(
            "linear",
            "Regularised linear/logistic model. Fast, interpretable, strong on near-linear signal.",
            lambda s: LogisticRegression(max_iter=2000, random_state=s),
            lambda s: Ridge(random_state=s),
            {"C": loguniform(1e-3, 1e2), "alpha": loguniform(1e-3, 1e2)},
        ),
        ModelSpec(
            "random_forest",
            "Bagged decision trees. Robust default, handles interactions, little tuning.",
            lambda s: RandomForestClassifier(n_estimators=200, random_state=s, n_jobs=1),
            lambda s: RandomForestRegressor(n_estimators=200, random_state=s, n_jobs=1),
            {"max_depth": [None, 6, 12, 24], "min_samples_leaf": randint(1, 10), "max_features": ["sqrt", 0.5, 1.0]},
        ),
        ModelSpec(
            "extra_trees",
            "Extremely randomised trees. Lower variance than RF, fast on wide data.",
            lambda s: ExtraTreesClassifier(n_estimators=200, random_state=s, n_jobs=1),
            lambda s: ExtraTreesRegressor(n_estimators=200, random_state=s, n_jobs=1),
            {"max_depth": [None, 8, 16], "min_samples_leaf": randint(1, 10)},
        ),
        ModelSpec(
            "hist_gradient_boosting",
            "Histogram gradient boosting (LightGBM-style). Usually strongest on tabular data, scales to large n.",
            lambda s: HistGradientBoostingClassifier(random_state=s),
            lambda s: HistGradientBoostingRegressor(random_state=s),
            {"learning_rate": loguniform(0.01, 0.3), "max_leaf_nodes": randint(8, 64), "l2_regularization": uniform(0, 1)},
        ),
        ModelSpec(
            "gradient_boosting",
            "Classic gradient boosting. Strong on small/medium data, slower to train.",
            lambda s: GradientBoostingClassifier(random_state=s),
            lambda s: GradientBoostingRegressor(random_state=s),
            {"learning_rate": loguniform(0.01, 0.3), "n_estimators": randint(50, 300), "max_depth": randint(2, 6)},
        ),
        ModelSpec(
            "knn",
            "k-nearest neighbours. Useful on low-dimensional, locally smooth data.",
            lambda s: KNeighborsClassifier(),
            lambda s: KNeighborsRegressor(),
            {"n_neighbors": randint(3, 40), "weights": ["uniform", "distance"]},
        ),
    ]
}


def available_models() -> dict[str, str]:
    return {name: spec.description for name, spec in REGISTRY.items()}


# --- Metrics -----------------------------------------------------------------
# name -> (sklearn scorer, higher_is_better)
METRICS: dict[str, dict[str, tuple[str, bool]]] = {
    "classification": {
        "f1_macro": ("f1_macro", True),
        "accuracy": ("accuracy", True),
        "balanced_accuracy": ("balanced_accuracy", True),
        "roc_auc": ("roc_auc", True),
    },
    "regression": {
        "r2": ("r2", True),
        "rmse": ("neg_root_mean_squared_error", False),
        "mae": ("neg_mean_absolute_error", False),
    },
}
DEFAULT_METRIC = {"classification": "f1_macro", "regression": "r2"}


def resolve_metric(problem_type: str, requested: str | None, n_classes: int | None = None) -> str:
    allowed = METRICS[problem_type]
    if requested in allowed and not (requested == "roc_auc" and n_classes != 2):
        return requested
    return DEFAULT_METRIC[problem_type]


def to_display(metric: str, problem_type: str, score: float) -> float:
    """sklearn scorers negate error metrics; flip them back for humans."""
    _, higher = METRICS[problem_type][metric]
    return float(score if higher else -score)


# --- Pipelines ---------------------------------------------------------------
def _as_str(X):
    return pd.DataFrame(X).astype("string").fillna("__missing__").astype(str)


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("to_str", FunctionTransformer(_as_str, feature_names_out="one-to-one")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5, max_categories=30, sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, make_column_selector(dtype_include=np.number)),
            ("cat", categorical, make_column_selector(dtype_exclude=np.number)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(
    model_name: str, problem_type: str, feature_plan: dict | None, seed: int, class_weight: str | None = None
) -> Pipeline:
    """Feature plan -> imputation/encoding -> estimator, all fitted per CV fold."""
    if model_name == BASELINE:
        est = (
            DummyClassifier(strategy="most_frequent")
            if problem_type == "classification"
            else DummyRegressor(strategy="mean")
        )
    else:
        est = REGISTRY[model_name].build(problem_type, seed, class_weight)
    return Pipeline(
        [
            ("features", FeaturePlanTransformer(feature_plan)),
            ("preprocess", build_preprocessor()),
            ("model", est),
        ]
    )


def make_cv(problem_type: str, y: pd.Series, folds: int, seed: int):
    if problem_type == "classification":
        min_class = int(y.value_counts().min())
        k = max(2, min(folds, min_class))
        if min_class >= 2:
            return StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return KFold(n_splits=folds, shuffle=True, random_state=seed)


def cross_validate_pipeline(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    problem_type: str,
    metric: str,
    folds: int,
    seed: int,
    n_jobs: int = 1,
) -> dict[str, float]:
    scorer, _ = METRICS[problem_type][metric]
    t0 = time.perf_counter()
    res = cross_validate(
        pipe,
        X,
        y,
        cv=make_cv(problem_type, y, folds, seed),
        scoring=scorer,
        return_train_score=True,
        n_jobs=n_jobs,
        error_score="raise",
    )
    return {
        "cv_score": float(np.mean(res["test_score"])),
        "cv_std": float(np.std(res["test_score"])),
        "train_score": float(np.mean(res["train_score"])),
        "seconds": round(time.perf_counter() - t0, 3),
    }


def tune_pipeline(
    pipe: Pipeline,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    problem_type: str,
    metric: str,
    folds: int,
    seed: int,
    n_iter: int,
    n_jobs: int = 1,
) -> tuple[Pipeline, dict[str, Any], dict[str, float]]:
    """Randomised search over the registry's space for this model."""
    spec = REGISTRY[model_name]
    est_params = pipe.named_steps["model"].get_params()
    space = {f"model__{k}": v for k, v in spec.param_space.items() if k in est_params}
    if not space:
        return pipe, {}, cross_validate_pipeline(
            pipe, X, y, problem_type=problem_type, metric=metric, folds=folds, seed=seed, n_jobs=n_jobs
        )
    scorer, _ = METRICS[problem_type][metric]
    t0 = time.perf_counter()
    search = RandomizedSearchCV(
        pipe,
        space,
        n_iter=n_iter,
        scoring=scorer,
        cv=make_cv(problem_type, y, folds, seed),
        random_state=seed,
        n_jobs=n_jobs,
        return_train_score=True,
        error_score=np.nan,
    )
    search.fit(X, y)
    i = search.best_index_
    stats = {
        "cv_score": float(search.best_score_),
        "cv_std": float(search.cv_results_["std_test_score"][i]),
        "train_score": float(search.cv_results_["mean_train_score"][i]),
        "seconds": round(time.perf_counter() - t0, 3),
    }
    params = {k.removeprefix("model__"): _jsonable(v) for k, v in search.best_params_.items()}
    return search.best_estimator_, params, stats


def holdout_metrics(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, problem_type: str) -> dict[str, float]:
    pred = pipe.predict(X)
    if problem_type == "regression":
        return {
            "r2": float(r2_score(y, pred)),
            "rmse": float(root_mean_squared_error(y, pred)),
            "mae": float(mean_absolute_error(y, pred)),
        }
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
    }
    if y.nunique() == 2 and hasattr(pipe, "predict_proba"):
        with contextlib.suppress(ValueError, AttributeError):
            out["roc_auc"] = float(get_scorer("roc_auc")(pipe, X, y))
    return out


def _jsonable(v: Any) -> Any:
    if isinstance(v, np.generic):
        return v.item()
    return v


__all__ = [
    "BASELINE",
    "DEFAULT_METRIC",
    "METRICS",
    "REGISTRY",
    "available_models",
    "build_pipeline",
    "cross_validate_pipeline",
    "holdout_metrics",
    "resolve_metric",
    "to_display",
    "tune_pipeline",
]
