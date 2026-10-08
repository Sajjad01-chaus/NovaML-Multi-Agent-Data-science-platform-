# NovaML

An autonomous, multi-agent data science platform. Upload a tabular dataset, name the
target column, and a team of agents plans the run, investigates the data by writing and
executing code in a sandbox, engineers features, picks model families, waits for your
approval, trains and cross-validates, critiques its own results and iterates, then
exports a servable model with a model card.

## Quick start

**As a service** (API, 2 workers, Postgres, MinIO, UI):

```bash
docker compose up --build          # UI http://localhost:8501 · API docs http://localhost:8080/docs
docker compose up --scale worker=4 # more throughput
python scripts/smoke.py            # end-to-end check against the running stack
```

**As a library / CLI** (single process, SQLite and local files):

```bash
pip install -e ".[dev,serve,ui,groq]"
novaml run path/to/data.csv --target churn --auto-approve
novaml run path/to/data.csv --target churn            # pauses for human approval
novaml resume <run_id> --approve linear random_forest
novaml serve var/runs/<run_id>/bundle                 # POST /predict
streamlit run ui/streamlit_app.py
```

**LLM: free Groq key.** Set `GROQ_API_KEY` (free at console.groq.com) and the agents'
decisions are made by `openai/gpt-oss-120b`, falling back to `qwen/qwen3.8-27b` and
`openai/gpt-oss-20b` when a model hits its free-tier rate limit (each model has its own
quota). The chain is checked against your account's live model list, so retired models
are skipped automatically. Without a key, every decision is made by a deterministic
policy, so the platform runs fully offline. Each run is capped at 60K tokens by default.
See `.env.example`.

## Architecture

```
                      ┌──────────── supervisor (routes on state, enforces budgets) ────────────┐
                      │                                                                          │
  profiler → planner → analyst → feature_engineer → model_selector → human_review → trainer → evaluator → critic
                          │                                              (interrupt)                        │
                   sandboxed code                                                      accept → deployer ───┘
                                             ▲                    ▲             ▲              │
                                             └── revise_features ─┴─ try_other ─┴── tune ──────┘
```

| Agent | Decides (LLM or policy) | Guardrails |
|---|---|---|
| planner | metric to optimise, whether to run analysis, questions, risks | metric validated against the problem type |
| analyst | ReAct loop: writes pandas code, reads output, reports findings | sandbox, step cap, starts from verified facts |
| feature_engineer | declarative `FeaturePlan` (drops, logs, dates, ratios) | validated against real columns; executed by deterministic code |
| model_selector | 2–4 model families from the registry | unknown names rejected; tried models excluded |
| human_review | a person approves which models to train | `interrupt()`; checkpointed, resumable from any process |
| critic | accept / tune / revise features / try other models | sees CV only (never holdout); capped rounds |

Details and design decisions: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Service architecture

```
 UI / client ──HTTP──▶ FastAPI ──▶ Postgres: runs · jobs (queue) · run_events
                          │                    ▲
                          │ enqueue (same txn) │ claim: FOR UPDATE SKIP LOCKED · lease + heartbeat
                          ▼                    │
                     worker × N ── agents ──▶ Postgres checkpoints (after every agent)
                                       └───▶ S3 / MinIO: datasets, models, bundles
```

| Concern | How |
|---|---|
| Long runs vs. request latency | The API only validates, enqueues and reads; workers do the work |
| Queue | Postgres table, no extra broker: a run and its job are created in one transaction |
| Concurrency | `SKIP LOCKED` claims; tested with 20 jobs across 6 workers, none claimed twice |
| Worker crash | Leases plus heartbeats; an expired lease re-queues the job, and the run resumes from its last checkpoint. Tested with `kill -9` mid-training: earlier agents did not re-run |
| Any worker, any run | State lives in Postgres and S3; one worker can pause a run for approval and another resume it |
| Retries | Idempotency keys on run creation; retried jobs skip finished work; dead-letter after N attempts |
| Human approval | `POST /v1/runs/{id}/approve` does compare-and-set and enqueues a resume job; double submits get 409 |
| Progress | Agents emit events; `GET /v1/runs/{id}/events` streams them (SSE, resumable with `Last-Event-ID`) |
| Cancellation | Takes effect at the next agent boundary; queued runs are cancelled immediately |
| Ops | `/healthz`, `/readyz` (database and storage), `/v1/queue`; SIGTERM drains the current job |

Local dev and unit tests use SQLite and the local filesystem through the same code path;
CI runs the same API/queue/worker tests on real Postgres and MinIO, and boots the
compose stack for a smoke run.

## Evaluation

`novaml eval` runs 12 offline benchmark cases (5 real datasets plus 7 scenarios with
planted problems: leaks, IDs, imbalance, pure noise, high-cardinality noise). It scores
the outcome *and* the agents' decisions: was the leak dropped, was real signal kept, was
a sensible metric chosen. See [docs/EVALS.md](docs/EVALS.md).

| | rule policies | Groq LLM (`gpt-oss-120b`) |
|---|---|---|
| cases passed (current checks) | 12/12, 35/35 checks | 6/6 rerun cases, 24/24 checks |
| full 12-case suite (before `keep` checks) | 12/12 | 12/12 |
| mean lift over baseline (balanced acc. / R²) | 0.50 | 0.49 |
| decisions made by the LLM / invalid outputs | n/a | 99% / 0% |
| tokens, time (full suite) | 0, ~2 min | ~240K, ~20 min |

The LLM rerun after the feature-drop fix covered the six cases with real-signal checks;
the full suite wasn't rerun because of the free tier's daily token limit.

What the evals caught and fixed along the way:
- **A leak got through:** a target re-encoded in another unit scored R² 1.0. The detector
  now scores numeric columns by rank correlation and compares each column against all the
  others combined.
- **Imbalance:** the 5%-fraud case scored 0.67 balanced accuracy. Class weighting raised it to 0.84.
- **The LLM discarded real signal:** it dropped both halves of an `f3 × f4` interaction and
  "redundant" correlated features. Drops now require evidence (identifier, leak, constant
  or empty). After the fix, the affected cases match the policy scores and keep their features.

Honest reading: with these guardrails, LLM decisions match a strong rule baseline on
outcomes. The LLM's value is in reasoning the rules can't do, like naming `boat` as a
post-outcome column from domain knowledge, and in explaining its decisions. CI runs the
policy suite on every push as a regression gate.

## Development

```bash
pytest               # 124 tests: unit, end-to-end, service, headless UI; no network, no keys
NOVAML_TEST_DATABASE_URL=postgresql://... NOVAML_TEST_S3_ENDPOINT=http://... pytest tests/integration/test_server.py  # on Postgres + S3
novaml eval          # 12-case benchmark (see docs/EVALS.md)
pytest -m live       # optional: real Groq call (needs GROQ_API_KEY)
ruff check src tests
```

## Roadmap

- [x] Phase 0: runnable, tested package; durable HITL; leak-free ML; safe serving
- [x] Phase 1: agentic core (planner, sandboxed analyst, critic reflection loop, supervisor)
- [x] Phase 7: eval harness (benchmark datasets, decision-quality scoring, LLM vs policy, CI regression gate)
- [x] Phase 2–3: FastAPI service, Postgres job queue and workers, crash recovery, Postgres checkpointer, S3 artifacts, SSE, Docker Compose
- [ ] Phase 5: OpenTelemetry tracing, LLM traces, metrics dashboards
- [ ] Phase 6: auth and tenancy, container sandbox backend, PII detection, rate limits
- [ ] Phase 4: LLM gateway caching and multi-provider fallback routing
- [ ] Phase 8–9: load tests, cloud deployment (compose stack done in Phase 2–3)
