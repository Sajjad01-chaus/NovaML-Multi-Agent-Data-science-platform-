"""Serving app for an exported NovaML model bundle.

The old deployer string-formatted column names into generated Python source, so a
CSV header such as ``x: int\\nimport os; os.system(...)`` became code execution in
the deployed API. Here nothing is generated: one fixed app reads the bundle's
JSON schema at startup and builds the request model dynamically, with column
names used only as data (field aliases).
"""

# No `from __future__ import annotations` here: FastAPI must see the real
# (locally defined) request model class, not a string annotation.
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def load_bundle(bundle_dir: str | Path) -> tuple[Any, dict, dict]:
    d = Path(bundle_dir)
    model = joblib.load(d / "model.joblib")
    schema = json.loads((d / "input_schema.json").read_text(encoding="utf-8"))
    card = json.loads((d / "model_card.json").read_text(encoding="utf-8"))
    return model, schema, card


def create_app(bundle_dir: str | Path):
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, ConfigDict, Field, create_model

    model, schema, card = load_bundle(bundle_dir)
    fields: dict[str, Any] = {}
    for i, col in enumerate(schema["columns"]):
        py_type = float if col["type"] == "number" else str
        fields[f"f{i}"] = (py_type | None, Field(default=None, alias=col["name"]))
    Row = create_model(  # noqa: N806
        "Row", __config__=ConfigDict(populate_by_name=False, extra="forbid"), **fields
    )

    class PredictRequest(BaseModel):
        rows: list[Row] = Field(min_length=1, max_length=1000)  # type: ignore[valid-type]

    app = FastAPI(title=f"NovaML model: {card.get('model_name')}", version=card.get("run_id", "0"))
    columns = [c["name"] for c in schema["columns"]]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/metadata")
    def metadata() -> dict:
        return card

    @app.post("/predict")
    def predict(req: PredictRequest) -> dict:
        records = [r.model_dump(by_alias=True) for r in req.rows]
        df = pd.DataFrame.from_records(records, columns=columns)
        try:
            preds = model.predict(df)
        except Exception as e:  # model-side failure, not a client error
            raise HTTPException(status_code=500, detail=f"prediction failed: {type(e).__name__}") from e
        out: dict[str, Any] = {"predictions": [_py(p) for p in preds]}
        if card.get("problem_type") == "classification" and hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(df)
                classes = [_py(c) for c in model.classes_]
                out["probabilities"] = [dict(zip(map(str, classes), map(float, row), strict=True)) for row in proba]
            except Exception:
                pass
        return out

    return app


def _py(v: Any) -> Any:
    return v.item() if hasattr(v, "item") else v
