import numpy as np
import pandas as pd
import pytest

from novaml.tools import data as dt
from novaml.tools.features import (
    FeaturePlan,
    FeaturePlanTransformer,
    RatioFeature,
    default_plan,
    validate_plan,
)
from novaml.tools.ml import (
    REGISTRY,
    build_pipeline,
    cross_validate_pipeline,
    holdout_metrics,
    resolve_metric,
)


@pytest.fixture
def messy(messy_csv):
    df = pd.read_csv(messy_csv).dropna(subset=["churn"]).reset_index(drop=True)
    return df, dt.profile(df, "churn")


def test_default_plan_drops_ids_and_expands_dates(messy):
    _, prof = messy
    plan = default_plan(prof)
    assert "customer_id" in plan.drop_columns and "constant" in plan.drop_columns
    assert plan.datetime_columns == ["signup date"]
    assert "annual income" in plan.log_transform


def test_validate_plan_rejects_bad_references(messy):
    _, prof = messy
    raw = FeaturePlan(
        drop_columns=["churn", "ghost"],
        log_transform=["plan"],
        ratio_features=[RatioFeature(numerator="annual income", denominator="tenure"), RatioFeature(numerator="plan", denominator="tenure")],
    )
    clean, warnings = validate_plan(raw, prof, "churn")
    assert clean.drop_columns == []
    assert clean.log_transform == []
    assert len(clean.ratio_features) == 1
    assert any("target" in w for w in warnings) and any("unknown" in w for w in warnings)


def test_transformer_applies_plan(messy):
    df, prof = messy
    plan = default_plan(prof)
    plan.ratio_features = [RatioFeature(numerator="annual income", denominator="tenure")]
    out = FeaturePlanTransformer(plan.model_dump()).fit_transform(df.drop(columns=["churn"]))
    assert "customer_id" not in out and "signup date" not in out
    assert {"signup date_year", "signup date_month", "annual income_per_tenure"} <= set(out.columns)
    assert np.isfinite(out["annual income"]).all()


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_registry_model_trains_on_messy_data(messy, name):
    df, prof = messy
    plan = default_plan(prof).model_dump()
    X, y = df.drop(columns=["churn"]), df["churn"]
    pipe = build_pipeline(name, "classification", plan, 0)
    stats = cross_validate_pipeline(pipe, X, y, problem_type="classification", metric="f1_macro", folds=3, seed=0)
    assert 0 <= stats["cv_score"] <= 1
    pipe.fit(X, y)
    # Unseen categories and missing values at inference must not crash.
    X_new = X.head(3).copy()
    X_new.loc[:, "plan"] = ["platinum", None, "basic"]
    assert len(pipe.predict(X_new)) == 3
    assert "f1_macro" in holdout_metrics(pipe, X, y, "classification")


def test_resolve_metric():
    assert resolve_metric("classification", "roc_auc", 3) == "f1_macro"
    assert resolve_metric("classification", "roc_auc", 2) == "roc_auc"
    assert resolve_metric("regression", "bogus") == "r2"


def test_fresh_estimators_per_pipeline():
    a = build_pipeline("random_forest", "classification", None, 0).named_steps["model"]
    b = build_pipeline("random_forest", "classification", None, 0).named_steps["model"]
    assert a is not b
