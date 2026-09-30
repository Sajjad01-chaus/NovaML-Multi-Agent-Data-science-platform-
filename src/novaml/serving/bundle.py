"""Exports a trained pipeline as a self-describing, servable bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

SERVE_PY = '''"""Serve this NovaML bundle:  uvicorn serve:app --port 8000"""
from pathlib import Path

from novaml.serving.app import create_app

app = create_app(Path(__file__).parent)
'''


def input_schema(X: pd.DataFrame) -> dict[str, Any]:
    return {
        "columns": [
            {
                "name": str(c),
                "type": "number"
                if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])
                else "string",
            }
            for c in X.columns
        ]
    }


def write_bundle(
    out_dir: Path, pipeline: Any, X_example: pd.DataFrame, model_card: dict[str, Any]
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_dir / "model.joblib")
    (out_dir / "input_schema.json").write_text(json.dumps(input_schema(X_example), indent=2), encoding="utf-8")
    (out_dir / "model_card.json").write_text(json.dumps(model_card, indent=2, default=str), encoding="utf-8")
    (out_dir / "serve.py").write_text(SERVE_PY, encoding="utf-8")
    return out_dir
