"""Deterministic provider for tests and offline evals.

Responses are queued per output schema. Each entry is a schema instance, a dict,
an exception to raise, or a callable ``(system, user) -> instance``.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from novaml.llm.base import LLMError, T, Usage


class ScriptedProvider:
    name = "scripted"
    model = "scripted-v1"

    def __init__(self, script: dict[str, list[Any]] | None = None, tokens_per_call: int = 100):
        self.queues: dict[str, deque] = defaultdict(deque)
        for schema_name, items in (script or {}).items():
            self.queues[schema_name].extend(items)
        self.tokens_per_call = tokens_per_call
        self.calls: list[dict[str, str]] = []

    def add(self, schema_name: str, *items: Any) -> ScriptedProvider:
        self.queues[schema_name].extend(items)
        return self

    def structured(self, schema: type[T], system: str, user: str) -> tuple[T, Usage]:
        self.calls.append({"schema": schema.__name__, "system": system, "user": user})
        q = self.queues.get(schema.__name__)
        if not q:
            raise LLMError(f"scripted: no response queued for {schema.__name__}")
        item = q.popleft()
        if isinstance(item, Exception):
            raise item
        if callable(item) and not isinstance(item, schema):
            item = item(system, user)
        value = item if isinstance(item, schema) else schema.model_validate(item)
        half = self.tokens_per_call // 2
        return value, Usage(half, self.tokens_per_call - half)
