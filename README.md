# NovaML

An autonomous, multi-agent data science platform. Upload a tabular dataset, name the
target column, and a team of agents plans the run, investigates the data by writing and
executing code in a sandbox, engineers features, picks model families, waits for your
approval, trains and cross-validates, critiques its own results and iterates, then
exports a servable model with a model card.

## Quick start

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

## Development

```bash
pytest               # 76 tests: unit + end-to-end + headless UI, no network, no keys
pytest -m live       # optional: real Groq call (needs GROQ_API_KEY)
ruff check src tests
```

## Roadmap

- [x] Phase 0: runnable, tested package; durable HITL; leak-free ML; safe serving
- [x] Phase 1: agentic core (planner, sandboxed analyst, critic reflection loop, supervisor)
- [ ] Phase 7: eval harness (benchmark datasets, decision-quality scoring, CI regression gate)
- [ ] Phase 2–3: FastAPI service, job queue and workers, Postgres checkpointer, object storage
- [ ] Phase 5: OpenTelemetry tracing, LLM traces, metrics dashboards
- [ ] Phase 6: auth and tenancy, container sandbox backend, PII detection, rate limits
- [ ] Phase 4: LLM gateway caching and multi-provider fallback routing
- [ ] Phase 8–9: load tests, Docker/Compose, deployment
