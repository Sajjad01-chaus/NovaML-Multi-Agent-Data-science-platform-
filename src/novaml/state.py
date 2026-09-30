"""Graph state.

Rules:
* Only JSON-serialisable values. Frames and models live in the ArtifactStore and
  are referenced by key, so checkpoints stay small and survive restarts.
* Nodes return *partial updates*; list fields with `operator.add` are append-only
  logs (the old code mutated and returned the whole state, which duplicates
  entries under add-reducers).
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class RunState(TypedDict, total=False):
    # Inputs
    run_id: str
    dataset_path: str
    target: str
    requested_problem_type: str | None

    # Data understanding
    problem_type: str
    metric: str
    profile: dict[str, Any]
    data_refs: dict[str, str]  # {"train": key, "test": key}

    # Features
    feature_plan: dict[str, Any]
    feature_warnings: list[str]

    # Model selection + human review
    candidate_models: list[str]
    selection_reasoning: str
    approved_models: list[str]

    # Training / evaluation
    leaderboard: Annotated[list[dict[str, Any]], operator.add]
    best: dict[str, Any]
    holdout: dict[str, Any]

    # Deployment
    bundle_dir: str
    model_card: dict[str, Any]

    # Bookkeeping
    status: str  # running | awaiting_approval | completed | failed
    messages: Annotated[list[str], operator.add]
    events: Annotated[list[dict[str, Any]], operator.add]
    llm_calls: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[str], operator.add]
