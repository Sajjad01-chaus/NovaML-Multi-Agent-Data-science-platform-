"""Agent base class and the shared context injected into every node."""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import structlog
from langgraph.errors import GraphBubbleUp

from novaml.artifacts import ArtifactStore
from novaml.config import Settings
from novaml.llm.base import LLMProvider
from novaml.llm.gateway import LLMGateway, tokens_spent
from novaml.log import get_logger
from novaml.sandbox.executor import Sandbox
from novaml.state import RunState
from novaml.tools.data import DataValidationError

log = get_logger(__name__)


@dataclass
class AgentContext:
    settings: Settings
    provider: LLMProvider | None = None
    sandbox: Sandbox | None = None  # None -> SubprocessSandbox from settings

    def store(self, state: RunState) -> ArtifactStore:
        return ArtifactStore(self.settings.runs_dir, state["run_id"])

    def gateway(self, state: RunState) -> LLMGateway:
        return LLMGateway(self.provider, self.settings, tokens_spent(state.get("llm_calls")))


class Agent:
    name: str = "agent"

    def run(self, state: RunState, ctx: AgentContext) -> dict[str, Any]:
        raise NotImplementedError

    def say(self, text: str) -> str:
        return f"[{self.name}] {text}"


def as_node(agent: Agent, ctx: AgentContext) -> Callable[[RunState], dict[str, Any]]:
    """Wraps an agent with timing, structured logging and failure capture.

    Expected failures (bad data) and unexpected ones both mark the run `failed`
    with a readable error instead of crashing the worker; routing sends failed
    runs straight to END.
    """

    def node(state: RunState) -> dict[str, Any]:
        structlog.contextvars.bind_contextvars(run_id=state.get("run_id"), agent=agent.name)
        t0 = time.perf_counter()
        status = "ok"
        try:
            update = agent.run(state, ctx) or {}
        except GraphBubbleUp:
            raise  # human-in-the-loop interrupts must reach LangGraph untouched
        except DataValidationError as e:
            status = "failed"
            update = {"status": "failed", "errors": [f"{agent.name}: {e}"]}
        except Exception as e:
            status = "failed"
            log.error("agent_failed", error=repr(e), tb=traceback.format_exc(limit=5))
            update = {"status": "failed", "errors": [f"{agent.name}: {type(e).__name__}: {e}"]}
        ms = round((time.perf_counter() - t0) * 1000, 1)
        log.info("agent_done", status=status, ms=ms)
        structlog.contextvars.unbind_contextvars("agent")
        event = {"node": agent.name, "status": status, "ms": ms, "ts": time.time()}
        update["events"] = [*update.get("events", []), event]
        return update

    node.__name__ = agent.name
    return node
