"""Provider-neutral LLM interface: every agent decision is a typed, structured call."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    model: str | None = None  # model that actually answered (may be a fallback)

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMError(RuntimeError):
    """Provider call failed, was refused, or returned unparseable output."""

    def __init__(self, message: str, usage: Usage | None = None):
        super().__init__(message)
        self.usage = usage or Usage()  # tokens burned by failed attempts still count


class LLMProvider(Protocol):
    name: str
    model: str

    def structured(self, schema: type[T], system: str, user: str) -> tuple[T, Usage]:
        """Return an instance of `schema` plus token usage, or raise `LLMError`."""
        ...
