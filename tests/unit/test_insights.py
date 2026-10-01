import numpy as np
import pandas as pd

from novaml.tools import data as dt
from novaml.tools.insights import compute_insights, feature_strength


def titanic_like(n=800, seed=0):
    """Survival driven by sex/class, plus the classic leaks: `boat` is only filled
    in for survivors, and high-cardinality `ticket` carries no real signal."""
    rng = np.random.default_rng(seed)
    sex = rng.choice(["male", "female"], n)
    pclass = rng.choice([1, 2, 3], n, p=[0.25, 0.25, 0.5])
    p = 0.2 + 0.5 * (sex == "female") + 0.1 * (pclass == 1)
    survived = (rng.random(n) < p).astype(int)
    boat = np.where(survived == 1, rng.choice(["1", "2", "A", "C", "13"], n), None)
    boat = np.where((survived == 1) & (rng.random(n) < 0.05), None, boat)  # a few survivors without a boat
    df = pd.DataFrame(
        {
            "sex": sex,
            "pclass": pclass,
            "age": rng.normal(30, 12, n).round(),
            "ticket": [f"T{rng.integers(0, 600)}" for _ in range(n)],
            "boat": boat,
            "survived": survived,
        }
    )
    return df, dt.profile(df, "survived")


def test_missingness_leak_is_flagged_and_real_signal_is_not():
    df, prof = titanic_like()
    r = compute_insights(df, "survived", "classification", prof)
    assert r["leakage_suspects"] == ["boat"]
    assert any("boat" in i and "leakage" in i for i in r["insights"])
    strength = r["feature_strength"]
    assert strength["sex"] > strength.get("ticket", 0.5)


def test_high_cardinality_noise_is_not_inflated():
    df, prof = titanic_like()
    fs = feature_strength(df, "survived", "classification", prof)
    assert fs["ticket"]["score"] < 0.6  # chance is 0.5; Cramér's V gave ~0.9 here


def test_target_copy_is_flagged_for_regression():
    rng = np.random.default_rng(1)
    y = rng.normal(100, 20, 500)
    df = pd.DataFrame({"x": rng.normal(size=500), "y_in_cents": y * 100, "y": y})
    r = compute_insights(df, "y", "regression", dt.profile(df, "y"))
    assert r["leakage_suspects"] == ["y_in_cents"]


def test_legitimately_strong_feature_is_kept():
    from sklearn.datasets import load_iris

    df = load_iris(as_frame=True).frame
    r = compute_insights(df, "target", "classification", dt.profile(df, "target"))
    assert r["leakage_suspects"] == []  # petal features are strong (~0.93) but not leaks
