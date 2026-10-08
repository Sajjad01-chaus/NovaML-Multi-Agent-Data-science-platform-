"""Ingests, validates, splits and profiles the dataset."""

from __future__ import annotations

from typing import Any

from novaml.agents.base import Agent, AgentContext
from novaml.state import RunState
from novaml.tools import data as data_tools
from novaml.tools.ml import resolve_metric


class ProfilerAgent(Agent):
    name = "profiler"

    def run(self, state: RunState, ctx: AgentContext) -> dict[str, Any]:
        s = ctx.settings
        df = data_tools.load_dataset(
            state["dataset_path"],
            allowed_extensions=s.allowed_extensions,
            max_mb=s.max_upload_mb,
            max_rows=s.max_rows,
        )
        target = state["target"]
        n_before = len(df)
        df = data_tools.validate_target(df, target, s.min_rows)

        requested = state.get("requested_problem_type")
        problem_type = requested if requested in ("classification", "regression") else data_tools.infer_problem_type(df[target])
        train, test = data_tools.split(df, target, problem_type, s.test_size, s.random_state)

        store = ctx.store(state)
        refs = {
            "train": store.save_frame("data/train.parquet", train),
            "test": store.save_frame("data/test.parquet", test),
        }
        # Profile the training split only: nothing downstream should learn from test rows.
        prof = data_tools.profile(train, target)
        prof["problem_type"] = problem_type
        metric = resolve_metric(problem_type, None, prof["target"]["n_unique"])

        msgs = [self.say(f"loaded {n_before} rows x {df.shape[1]} columns")]
        if n_before != len(df):
            msgs.append(self.say(f"dropped {n_before - len(df)} rows with missing target"))
        msgs.append(
            self.say(f"{problem_type} on {target!r}; train={len(train)} test={len(test)}; metric={metric}")
        )
        return {
            "problem_type": problem_type,
            "metric": metric,
            "profile": prof,
            "data_refs": refs,
            "messages": msgs,
        }
