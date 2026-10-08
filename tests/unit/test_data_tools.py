import pandas as pd
import pytest

from novaml.tools import data as dt


def test_infer_problem_type():
    assert dt.infer_problem_type(pd.Series(["a", "b"] * 20)) == "classification"
    assert dt.infer_problem_type(pd.Series([0, 1] * 20)) == "classification"
    assert dt.infer_problem_type(pd.Series([x * 0.37 for x in range(100)])) == "regression"
    assert dt.infer_problem_type(pd.Series(range(500))) == "regression"


def test_looks_like_id():
    assert dt.looks_like_id(pd.Series([f"u{i}" for i in range(100)], name="user"))
    assert dt.looks_like_id(pd.Series(range(100), name="row_id"))
    assert not dt.looks_like_id(pd.Series([i * 0.5 for i in range(100)], name="price"))
    assert not dt.looks_like_id(pd.Series(["a", "b"] * 50, name="paid"))


def test_validate_target_errors():
    df = pd.DataFrame({"x": range(30), "y": [1] * 30})
    with pytest.raises(dt.DataValidationError, match="single value"):
        dt.validate_target(df, "y", 10)
    with pytest.raises(dt.DataValidationError, match="not found"):
        dt.validate_target(df, "nope", 10)


def test_load_dataset_guards(tmp_path):
    bad = tmp_path / "x.exe"
    bad.write_text("hi")
    with pytest.raises(dt.DataValidationError, match="unsupported"):
        dt.load_dataset(bad, allowed_extensions=(".csv",), max_mb=1, max_rows=10)
    big = tmp_path / "big.csv"
    big.write_text("a\n" + "\n".join(map(str, range(50))))
    with pytest.raises(dt.DataValidationError, match="more than"):
        dt.load_dataset(big, allowed_extensions=(".csv",), max_mb=1, max_rows=10)


def test_profile_is_json_safe_and_flags_columns(messy_csv):
    import json

    df = pd.read_csv(messy_csv).dropna(subset=["churn"])
    prof = dt.profile(df, "churn")
    json.dumps(prof)
    cols = prof["columns"]
    assert cols["customer_id"]["id_like"]
    assert cols["constant"]["constant"]
    assert cols["signup date"]["kind"] == "datetime"
    assert prof["inferred_problem_type"] == "classification"
    assert "churn" not in cols
