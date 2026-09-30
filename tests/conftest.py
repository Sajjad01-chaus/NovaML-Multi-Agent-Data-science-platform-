from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes, load_iris

from novaml.config import Settings
from novaml.service import NovaML


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "var",
        llm_provider="none",
        cv_folds=3,
        tuning_iterations=3,
        auto_approve=True,
        _env_file=None,
    )


@pytest.fixture
def svc(settings: Settings):
    s = NovaML(settings, provider=None)
    yield s
    s.close()


@pytest.fixture
def iris_csv(tmp_path: Path) -> Path:
    d = load_iris(as_frame=True)
    df = d.frame.rename(columns={"target": "species"})
    df["species"] = df["species"].map(dict(enumerate(d.target_names)))
    p = tmp_path / "iris.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def diabetes_csv(tmp_path: Path) -> Path:
    df = load_diabetes(as_frame=True).frame.rename(columns={"target": "progression"})
    p = tmp_path / "diabetes.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def messy_csv(tmp_path: Path) -> Path:
    """IDs, dates, missing values, categoricals, skew, a missing target, odd column names."""
    rng = np.random.default_rng(0)
    n = 400
    income = rng.lognormal(10, 1, n)
    tenure = rng.integers(0, 60, n).astype(float)
    plan = rng.choice(["basic", "pro", "enterprise"], n, p=[0.6, 0.3, 0.1])
    logit = -1 + 0.04 * tenure * -1 + (plan == "basic") * 1.2 + rng.normal(0, 0.5, n)
    churn = np.where(logit > -0.5, "yes", "no")
    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "signup date": pd.date_range("2020-01-01", periods=n, freq="D").astype(str),
            "annual income": income,
            "tenure": tenure,
            "plan": plan,
            "constant": 1,
            "churn": churn,
        }
    )
    df.loc[rng.choice(n, 40, replace=False), "tenure"] = np.nan
    df.loc[rng.choice(n, 30, replace=False), "plan"] = np.nan
    df.loc[[3, 7], "churn"] = np.nan
    p = tmp_path / "messy.csv"
    df.to_csv(p, index=False)
    return p
