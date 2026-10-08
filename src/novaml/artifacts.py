"""Run-scoped artifact storage.

Graph state only carries *references* (artifact keys) plus small JSON summaries.
DataFrames, fitted models and plots live here. That keeps checkpoints small and
serialisable, and makes it possible to swap the local backend for S3/MinIO
without touching the agents.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

_KEY_RE = re.compile(r"^[A-Za-z0-9_\-./]+$")


class ArtifactStore:
    def __init__(self, root: Path, run_id: str):
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", run_id):
            raise ValueError(f"invalid run id: {run_id!r}")
        self.run_id = run_id
        self.root = (Path(root) / run_id).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        """Resolve a key to a path, refusing anything that escapes the run directory."""
        if not _KEY_RE.match(key) or ".." in key.split("/"):
            raise ValueError(f"invalid artifact key: {key!r}")
        p = (self.root / key).resolve()
        if self.root not in p.parents and p != self.root:
            raise ValueError(f"artifact key escapes run dir: {key!r}")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def exists(self, key: str) -> bool:
        return self.path(key).exists()

    # DataFrames -------------------------------------------------------------
    def save_frame(self, key: str, df: pd.DataFrame) -> str:
        df.to_parquet(self.path(key), index=False)
        return key

    def load_frame(self, key: str) -> pd.DataFrame:
        return pd.read_parquet(self.path(key))

    # Models -----------------------------------------------------------------
    def save_model(self, key: str, model: Any) -> str:
        joblib.dump(model, self.path(key))
        return key

    def load_model(self, key: str) -> Any:
        return joblib.load(self.path(key))

    # JSON / text ------------------------------------------------------------
    def save_json(self, key: str, obj: Any) -> str:
        self.path(key).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
        return key

    def load_json(self, key: str) -> Any:
        return json.loads(self.path(key).read_text(encoding="utf-8"))

    def save_text(self, key: str, text: str) -> str:
        self.path(key).write_text(text, encoding="utf-8")
        return key
