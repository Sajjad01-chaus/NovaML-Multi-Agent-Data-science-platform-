"""Benchmark cases: datasets with known-good answers.

Each case carries *expectations*, not just a dataset. Some test model quality
(a score floor), and some test the agents' decisions:

* `max_score`   a ceiling: scoring above it means leakage got through
* `must_drop`   columns that must be removed (planted leaks, identifiers)
* `must_keep`   strong but legitimate columns that must survive
* `metrics_ok`  acceptable optimisation metrics (e.g. class-balanced when imbalanced)

The `core` suite is fully offline and deterministic (bundled sklearn data plus
seeded synthetic scenarios), so it can gate CI. `extended` pulls real datasets
from OpenML and is meant for on-demand runs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EvalCase:
    name: str
    load: Callable[[], pd.DataFrame]
    target: str
    problem_type: str
    min_score: float
    """Floor on the holdout eval metric (balanced_accuracy or r2)."""
    max_score: float | None = None
    must_drop: tuple[str, ...] = ()
    must_keep: tuple[str, ...] = ()
    metrics_ok: tuple[str, ...] | None = None
    description: str = ""
    suite: str = "core"
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def eval_metric(self) -> str:
        # Fixed per problem type so runs that optimised different metrics stay comparable.
        return "balanced_accuracy" if self.problem_type == "classification" else "r2"


# --- bundled sklearn datasets -------------------------------------------------
def _sk(loader_name: str, target: str, labelled: bool = True) -> Callable[[], pd.DataFrame]:
    def load() -> pd.DataFrame:
        import sklearn.datasets as ds

        d = getattr(ds, loader_name)(as_frame=True)
        df = d.frame.rename(columns={"target": target})
        if labelled:  # integer class codes -> readable class names
            df[target] = df[target].map(dict(enumerate(map(str, d.target_names))))
        return df

    return load


# --- synthetic scenarios (seeded) ----------------------------------------------
def _titanic_like(n: int = 1000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sex = rng.choice(["male", "female"], n)
    pclass = rng.choice([1, 2, 3], n, p=[0.25, 0.25, 0.5])
    age = rng.normal(30, 13, n).clip(1, 80).round()
    fare = rng.lognormal(3 - 0.5 * (pclass - 1), 0.6, n).round(2)
    p = 0.15 + 0.5 * (sex == "female") + 0.15 * (pclass == 1) - 0.1 * (age > 60)
    survived = (rng.random(n) < p.clip(0.02, 0.98)).astype(int)
    boat = np.where(survived == 1, rng.choice(["1", "2", "A", "C", "13", "15"], n), None)
    boat = np.where((survived == 1) & (rng.random(n) < 0.05), None, boat)
    df = pd.DataFrame({"passenger_id": np.arange(1, n + 1), "sex": sex, "pclass": pclass, "age": age, "fare": fare, "boat": boat, "survived": survived})
    df.loc[rng.random(n) < 0.2, "age"] = np.nan
    return df


def _target_copy(n: int = 600, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = 3 * X[:, 0] - 2 * X[:, 1] + X[:, 2] * X[:, 3] + rng.normal(0, 1, n)
    df = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"])
    df["price_in_cents"] = (y * 100 + rng.normal(0, 1, n)).round()  # the target, re-encoded
    df["price"] = y
    return df


def _messy_churn(n: int = 900, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 72, n).astype(float)
    plan = rng.choice(["basic", "pro", "enterprise"], n, p=[0.6, 0.3, 0.1])
    monthly = np.where(plan == "basic", 20, np.where(plan == "pro", 50, 120)) + rng.normal(0, 5, n)
    tickets = rng.poisson(1.5, n)
    logit = 0.8 - 0.06 * tenure + 0.5 * tickets + 0.8 * (plan == "basic")
    churn = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "yes", "no")
    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:06d}" for i in rng.permutation(n)],
            "signup_date": pd.Timestamp("2021-01-01") + pd.to_timedelta(rng.integers(0, 1000, n), unit="D"),
            "tenure_months": tenure,
            "plan": plan,
            "monthly_charges": monthly.round(2),
            "support_tickets": tickets,
            "region": rng.choice(["north", "south", "east", "west"], n),
            "constant_flag": 1,
            "churn": churn,
        }
    )
    df["signup_date"] = df["signup_date"].dt.strftime("%Y-%m-%d")
    df.loc[rng.random(n) < 0.1, "tenure_months"] = np.nan
    df.loc[rng.random(n) < 0.08, "plan"] = np.nan
    df.loc[rng.random(n) < 0.01, "churn"] = np.nan
    return df


def _imbalanced(n: int = 2000, seed: int = 5) -> pd.DataFrame:
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=n, n_features=12, n_informative=5, n_redundant=2, weights=[0.95], flip_y=0.01, class_sep=1.0, random_state=seed)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    df["fraud"] = np.where(y == 1, "fraud", "ok")
    return df


def _high_cardinality(n: int = 1200, seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.choice(["red", "green", "blue"], n)
    y = ((a + (b == "red") * 0.8 + rng.normal(0, 0.8, n)) > 0.4).astype(int)
    return pd.DataFrame({"signal": a, "colour": b, "ticket_code": [f"TK{rng.integers(0, 900)}" for _ in range(n)], "noise": rng.normal(size=n), "label": y})


def _pure_noise(n: int = 800, seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 8)), columns=[f"n{i}" for i in range(8)])
    df["cat"] = rng.choice(list("abcde"), n)
    df["outcome"] = rng.choice(["a", "b"], n)
    return df


def _skewed_regression(n: int = 1000, seed: int = 19) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sqft = rng.lognormal(7, 0.5, n)
    rooms = rng.integers(1, 7, n)
    age = rng.integers(0, 80, n)
    price = np.exp(4 + 0.9 * np.log(sqft) + 0.05 * rooms - 0.004 * age + rng.normal(0, 0.15, n))
    return pd.DataFrame({"sqft": sqft.round(), "rooms": rooms, "age_years": age, "neighbourhood": rng.choice(list("ABCDEFG"), n), "price": price.round(-2)})


# --- OpenML (network; extended suite) -----------------------------------------
def _openml(name: str, version: int, target: str, drop: tuple[str, ...] = (), add_id: bool = False) -> Callable[[], pd.DataFrame]:
    def load() -> pd.DataFrame:
        from sklearn.datasets import fetch_openml

        df = fetch_openml(name, version=version, as_frame=True).frame.drop(columns=list(drop))
        if add_id:
            df.insert(0, "row_id", range(1, len(df) + 1))
        return df.rename(columns={df.columns[-1]: target}) if target not in df.columns else df

    return load


CASES: list[EvalCase] = [
    EvalCase("iris", _sk("load_iris", "species"), "species", "classification", 0.90, must_keep=("petal width (cm)", "petal length (cm)"), description="Strong legitimate features must not be mistaken for leakage."),
    EvalCase("wine", _sk("load_wine", "cultivar"), "cultivar", "classification", 0.90),
    EvalCase("breast_cancer", _sk("load_breast_cancer", "diagnosis"), "diagnosis", "classification", 0.92),
    EvalCase("digits", _sk("load_digits", "digit", labelled=False), "digit", "classification", 0.90, description="Wide (64 features), 10 classes."),
    EvalCase("diabetes", _sk("load_diabetes", "progression", labelled=False), "progression", "regression", 0.35),
    EvalCase("leak_missingness", _titanic_like, "survived", "classification", 0.62, max_score=0.85, must_drop=("boat", "passenger_id"), must_keep=("sex", "pclass"), description="`boat` is filled in only for survivors (post-outcome leak)."),
    EvalCase("leak_target_copy", _target_copy, "price", "regression", 0.60, max_score=0.97, must_drop=("price_in_cents",), must_keep=("f1", "f2", "f3", "f4"), description="The target re-encoded in another unit; f3*f4 is a real interaction."),
    EvalCase("messy_churn", _messy_churn, "churn", "classification", 0.62, must_drop=("customer_id", "constant_flag"), must_keep=("tenure_months", "support_tickets", "plan"), description="IDs, dates, missing values, a constant, a missing target."),
    EvalCase("imbalanced_fraud", _imbalanced, "fraud", "classification", 0.70, metrics_ok=("balanced_accuracy", "roc_auc", "f1_macro"), description="5% minority class: accuracy would be misleading."),
    EvalCase("high_cardinality_noise", _high_cardinality, "label", "classification", 0.70, max_score=0.90, must_keep=("signal",), description="A 900-level noise code must not look predictive."),
    EvalCase("pure_noise", _pure_noise, "outcome", "classification", 0.40, max_score=0.60, description="No signal: an honest pipeline does not beat the baseline."),
    EvalCase("skewed_regression", _skewed_regression, "price", "regression", 0.80, must_keep=("sqft",), description="Log-normal target and size feature."),
    EvalCase("titanic", _openml("titanic", 1, "survived", drop=("body",), add_id=True), "survived", "classification", 0.70, max_score=0.88, must_drop=("boat",), suite="extended", description="Real data with the classic `boat` leak kept in."),
    EvalCase("adult", _openml("adult", 2, "class"), "class", "classification", 0.72, suite="extended"),
    EvalCase("credit_g", _openml("credit-g", 1, "class"), "class", "classification", 0.60, suite="extended"),
    EvalCase("california_housing", _sk("fetch_california_housing", "MedHouseVal", labelled=False), "MedHouseVal", "regression", 0.70, suite="extended"),
]


def get_cases(suite: str = "core", names: list[str] | None = None) -> list[EvalCase]:
    pool = [c for c in CASES if suite == "all" or c.suite == suite]
    if names:
        by_name = {c.name: c for c in CASES}
        unknown = [n for n in names if n not in by_name]
        if unknown:
            raise KeyError(f"unknown eval cases: {unknown}")
        pool = [by_name[n] for n in names]
    return pool
