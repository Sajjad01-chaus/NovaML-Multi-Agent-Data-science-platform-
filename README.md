# NovaML

An autonomous, multi-agent data science platform. Upload a tabular dataset, name the
target column, and a team of agents profiles the data, plans features, picks model
families, waits for your approval, trains and cross-validates, and exports a servable
model with a model card.

## Quick start

```bash
pip install -e ".[dev,serve,ui]"
novaml run path/to/data.csv --target churn --auto-approve
novaml serve var/runs/<run_id>/bundle          # POST /predict
streamlit run ui/streamlit_app.py
```

No API key is needed: with `NOVAML_LLM_PROVIDER=none` (the default) every agent
decision uses a deterministic policy. Set `NOVAML_LLM_PROVIDER=anthropic` (or `groq`)
and the matching API key to let an LLM make those decisions. See `.env.example`.

## Architecture

```
profiler -> feature_engineer -> model_selector -> human_review -> trainer -> evaluator -> deployer
                                                  (interrupt)
```

| Concern | How it is handled |
|---|---|
| State | JSON-only graph state; frames and models live in a run-scoped `ArtifactStore` and are referenced by key |
| Durability | LangGraph SQLite checkpointer on disk; a run paused for approval resumes from any process |
| Human-in-the-loop | `interrupt()` before training; resume input is validated against the candidates |
| LLM calls | One gateway: typed structured output, token budget per run, cost accounting, policy fallback on error / refusal / invalid output |
| Leakage | Holdout split first; imputation, encoding and feature plans are fitted inside each CV fold; selection uses CV only, holdout is report-only |
| Serving | Fixed FastAPI app + JSON input schema; column names are data, never code |
| Failures | Each agent is wrapped: timing, structured logs, errors recorded and routed to END |

## Development

```bash
pytest               # unit + end-to-end (no network, no keys)
ruff check src tests
```
