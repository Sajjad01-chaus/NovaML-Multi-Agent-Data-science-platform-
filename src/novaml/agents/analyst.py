"""Analyst: a bounded ReAct loop that writes pandas code, runs it in the sandbox,
reads the output and iterates until it can answer the planner's questions.

Starts from deterministic facts (tools/insights.py) so the model verifies and
extends known signal instead of hallucinating it. Without an LLM, the facts alone
are the output.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from novaml.agents.base import Agent, AgentContext
from novaml.llm.gateway import as_data
from novaml.sandbox.executor import SubprocessSandbox
from novaml.state import RunState
from novaml.tools.insights import compute_insights


class AnalystStep(BaseModel):
    thought: str = Field(description="Brief reasoning about what to check next.")
    action: Literal["run_code", "finish"]
    code: str | None = Field(default=None, description="pandas code using `df`; print() what you need to see.")
    insights: list[str] = Field(default_factory=list, description="On finish: concise, evidence-backed findings.")
    leakage_suspects: list[str] = Field(default_factory=list, description="On finish: columns that likely leak the target.")


SYSTEM = (
    "You are a senior data analyst investigating a training dataset before modelling. "
    "You can run Python against a pandas DataFrame named `df` (the training split; the "
    "target column is included). `pd` and `np` are available. Only pandas/numpy/scipy/"
    "math/statistics are importable; no file, network or OS access. Print concise "
    "results (aggregates, not raw rows). Each step either runs one snippet or finishes "
    "with evidence-backed insights. Only claim what the outputs show."
)


class AnalystAgent(Agent):
    name = "analyst"

    def run(self, state: RunState, ctx: AgentContext) -> dict[str, Any]:
        s = ctx.settings
        store = ctx.store(state)
        target, pt = state["target"], state["problem_type"]
        train = store.load_frame(state["data_refs"]["train"])
        facts = compute_insights(train, target, pt, state["profile"])
        result = {"insights": list(facts["insights"]), "leakage_suspects": list(facts["leakage_suspects"]), "steps": [], "source": "policy"}
        calls: list[dict] = []
        gw = ctx.gateway(state)

        if gw.provider is not None:
            sandbox = ctx.sandbox or SubprocessSandbox(s.sandbox_timeout_s, s.sandbox_max_output_chars, s.sandbox_memory_mb)
            questions = state.get("plan", {}).get("analysis_questions", [])
            transcript: list[str] = []
            for i in range(s.analyst_max_steps + 1):
                last = i == s.analyst_max_steps
                user = (
                    f"Target: {target!r} ({pt})\nQuestions:\n- " + "\n- ".join(questions or ["What matters for predicting the target?"])
                    + f"\n\nVerified facts:\n{as_data(facts)}\n\nColumns: {as_data(list(train.columns))}\n"
                    + ("\nPrevious steps:\n" + "\n".join(transcript) if transcript else "")
                    + ("\n\nYou are out of steps: finish now." if last else "")
                )
                d = gw.decide(agent=self.name, schema=AnalystStep, system=SYSTEM, user=user, policy=lambda: AnalystStep(thought="fallback", action="finish"))
                calls.append(d.record)
                step = d.value
                if d.source == "policy":
                    break  # LLM unavailable mid-loop: keep the deterministic facts
                if step.action == "finish" or last or not step.code:
                    result["insights"] += [x for x in step.insights if x not in result["insights"]][:8]
                    cols = set(train.columns) - {target}
                    result["leakage_suspects"] = sorted(set(result["leakage_suspects"]) | {c for c in step.leakage_suspects if c in cols})
                    result["source"] = "llm"
                    break
                obs = sandbox.run(step.code, train)
                result["steps"].append({"thought": step.thought, "code": step.code, "ok": obs.ok, "output": obs.stdout, "error": obs.error, "ms": obs.duration_ms})
                transcript.append(
                    f"[step {i + 1}] thought: {step.thought}\ncode:\n{step.code}\n"
                    + (f"output:\n{obs.stdout}" if obs.ok else f"error: {obs.error}\noutput:\n{obs.stdout}")
                )

        store.save_json("analysis.json", result)
        msgs = [self.say(f"{len(result['insights'])} insights ({result['source']}, {len(result['steps'])} code steps)")]
        msgs += [self.say(f"- {x}") for x in result["insights"][:6]]
        return {"analysis": {k: result[k] for k in ("insights", "leakage_suspects", "source")} | {"code_steps": len(result["steps"])}, "llm_calls": calls, "messages": msgs}
