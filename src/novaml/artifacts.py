"""Run-scoped artifact storage.

Graph state only carries *references* (artifact keys) plus small JSON summaries.
DataFrames, fitted models, bundles and transcripts live here, so checkpoints stay
small and any worker can pick up any run.

Two backends share one byte-oriented interface:
* `LocalArtifactStore`: a directory per namespace (dev, tests, single node)
* `S3ArtifactStore`: an S3 bucket, or MinIO/R2 via `endpoint_url` (multi-worker)
"""

from __future__ import annotations

import io
import json
import re
import tempfile
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

_NS_RE = re.compile(r"^[A-Za-z0-9_\-]+$")
_KEY_RE = re.compile(r"^[A-Za-z0-9_\-./]+$")


def _check_key(key: str) -> str:
    if not _KEY_RE.match(key) or key.startswith("/") or ".." in key.split("/"):
        raise ValueError(f"invalid artifact key: {key!r}")
    return key


class ArtifactStore(ABC):
    """Namespaced key -> bytes store with typed helpers. A namespace is a run id (or 'datasets')."""

    def __init__(self, namespace: str):
        if not _NS_RE.fullmatch(namespace):
            raise ValueError(f"invalid namespace: {namespace!r}")
        self.namespace = namespace

    # -- backend primitives ----------------------------------------------------
    @abstractmethod
    def put_bytes(self, key: str, data: bytes) -> str: ...

    @abstractmethod
    def get_bytes(self, key: str) -> bytes: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def list(self, prefix: str = "") -> list[str]: ...

    @abstractmethod
    def uri(self, key: str = "") -> str: ...

    # -- typed helpers -----------------------------------------------------------
    def save_frame(self, key: str, df: pd.DataFrame) -> str:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        return self.put_bytes(key, buf.getvalue())

    def load_frame(self, key: str) -> pd.DataFrame:
        return pd.read_parquet(io.BytesIO(self.get_bytes(key)))

    def save_model(self, key: str, model: Any) -> str:
        buf = io.BytesIO()
        joblib.dump(model, buf)
        return self.put_bytes(key, buf.getvalue())

    def load_model(self, key: str) -> Any:
        return joblib.load(io.BytesIO(self.get_bytes(key)))

    def save_json(self, key: str, obj: Any) -> str:
        return self.put_bytes(key, json.dumps(obj, indent=2, default=str).encode())

    def load_json(self, key: str) -> Any:
        return json.loads(self.get_bytes(key))

    def save_text(self, key: str, text: str) -> str:
        return self.put_bytes(key, text.encode())

    def put_dir(self, prefix: str, local_dir: Path) -> str:
        """Upload a local directory tree under `prefix/`."""
        for p in sorted(Path(local_dir).rglob("*")):
            if p.is_file():
                self.put_bytes(f"{prefix}/{p.relative_to(local_dir).as_posix()}", p.read_bytes())
        return self.uri(prefix)

    def get_dir(self, prefix: str, local_dir: Path) -> Path:
        """Download everything under `prefix/` into `local_dir`."""
        for key in self.list(prefix + "/"):
            dest = Path(local_dir) / key[len(prefix) + 1 :]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(self.get_bytes(key))
        return Path(local_dir)

    def zip_dir(self, prefix: str) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for key in self.list(prefix + "/"):
                z.writestr(key[len(prefix) + 1 :], self.get_bytes(key))
        return buf.getvalue()


class LocalArtifactStore(ArtifactStore):
    def __init__(self, root: Path, namespace: str):
        super().__init__(namespace)
        self.root = (Path(root) / namespace).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        """Resolve a key to a path, refusing anything that escapes the namespace directory."""
        p = (self.root / _check_key(key)).resolve()
        if self.root not in p.parents and p != self.root:
            raise ValueError(f"artifact key escapes namespace: {key!r}")
        return p

    def put_bytes(self, key: str, data: bytes) -> str:
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)  # atomic: readers never see a half-written artifact
        return key

    def get_bytes(self, key: str) -> bytes:
        return self.path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self.path(key).exists()

    def list(self, prefix: str = "") -> list[str]:
        base = self.root
        return sorted(
            p.relative_to(base).as_posix()
            for p in base.rglob("*")
            if p.is_file() and not p.name.endswith(".tmp") and p.relative_to(base).as_posix().startswith(prefix)
        )

    def uri(self, key: str = "") -> str:
        return str(self.path(key)) if key else str(self.root)


class S3ArtifactStore(ArtifactStore):
    def __init__(self, bucket: str, namespace: str, prefix: str = "runs", client: Any = None, **client_kwargs: Any):
        super().__init__(namespace)
        if client is None:
            import boto3

            client = boto3.client("s3", **{k: v for k, v in client_kwargs.items() if v is not None})
        self.client = client
        self.bucket = bucket
        self.base = f"{prefix.strip('/')}/{namespace}" if prefix else namespace

    def _k(self, key: str) -> str:
        return f"{self.base}/{_check_key(key)}"

    def put_bytes(self, key: str, data: bytes) -> str:
        self.client.put_object(Bucket=self.bucket, Key=self._k(key), Body=data)
        return key

    def get_bytes(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=self._k(key))["Body"].read()

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=self._k(key))
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def list(self, prefix: str = "") -> list[str]:
        out: list[str] = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=f"{self.base}/{prefix}"):
            out += [o["Key"][len(self.base) + 1 :] for o in page.get("Contents", [])]
        return sorted(out)

    def uri(self, key: str = "") -> str:
        return f"s3://{self.bucket}/{self.base}" + (f"/{key}" if key else "")


def make_store(settings: Any, namespace: str) -> ArtifactStore:
    if settings.artifact_backend == "s3":
        if not settings.s3_bucket:
            raise ValueError("NOVAML_S3_BUCKET is required for the s3 artifact backend")
        return S3ArtifactStore(
            settings.s3_bucket,
            namespace,
            endpoint_url=settings.s3_endpoint_url,
            region_name=settings.s3_region,
        )
    return LocalArtifactStore(settings.runs_dir, namespace)


def materialize_bundle(store: ArtifactStore, prefix: str = "bundle") -> Path:
    """Local copy of a stored bundle (no-op for the local backend)."""
    if isinstance(store, LocalArtifactStore):
        return store.path(prefix)
    d = Path(tempfile.mkdtemp(prefix="novaml-bundle-"))
    return store.get_dir(prefix, d)


__all__ = [
    "ArtifactStore",
    "LocalArtifactStore",
    "S3ArtifactStore",
    "make_store",
    "materialize_bundle",
]
