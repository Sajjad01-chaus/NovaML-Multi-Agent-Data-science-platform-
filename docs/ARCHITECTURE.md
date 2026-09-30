# Architecture and design decisions

## Run lifecycle

1. `NovaML.start()` creates a run (`run_id` = LangGraph thread id) and invokes the graph.
2. The **supervisor** inspects state and routes to the next agent. Every agent returns to it.
3. At `human_review` the graph calls `interrupt()`. The checkpoint is written to SQLite
   and `start()` returns `awaiting_approval`.
4. `NovaML.resume(run_id, approved)` validates the approval against the pending
   candidates and resumes with `Command(resume=...)`, from any process.
5. After evaluation the **critic** either accepts (→ deployer → END) or sends the run back
   for another round. Rounds and total steps are capped.

## Decisions

### D1. Deterministic routing, model-driven judgement
Routing is a pure function of state (`supervisor.next_step`). The LLM makes the
*judgement calls* (metric, analysis, features, models, accept/iterate), and those
decisions are recorded in state. The supervisor only enforces ordering invariants and
budgets.
*Why:* an LLM router adds latency and cost for choices that are mostly forced, and
can loop. This way every run replays exactly from its checkpoint, and routing is unit-tested.

### D2. Every LLM decision has a policy twin
`LLMGateway.decide(schema, policy, validate)` returns the LLM's typed answer only if the
provider succeeds, the run's token budget allows it, and the answer passes domain
validation. Otherwise it returns the deterministic policy's answer and records why.
*Why:* graceful degradation (outages, refusals, budget exhaustion), fully offline CI and
evals, and an easy A/B baseline: "LLM vs. policy" on the same benchmark.

### D2b. Free-tier friendly by design
The default provider is Groq's free tier: it is used automatically when `GROQ_API_KEY` is set.
Free-tier quotas are per model (roughly 30 req/min, 8K tokens/min and 200K tokens/day
for `openai/gpt-oss-120b`), so `GroqProvider` walks a fallback chain of models on 429s,
retired models, unsupported features, 5xx errors or unparseable output. Only an auth error
stops the chain. Tokens burned by failed attempts still count against the run budget.
Structured output uses tool calling, which every Groq chat model supports; Groq's strict
JSON-schema mode is limited to a few models and forbids optional fields.

### D3. LLM output is data, never code, in the serving path
Feature engineering is a declarative `FeaturePlan` executed by a fixed sklearn
transformer inside the pipeline. The generated FastAPI app is replaced by one fixed app
that reads a JSON schema.
*Why:* no train/serve skew, no leakage (fitted per CV fold), and nothing produced by a
model or taken from a CSV header is ever executed at inference.

### D4. Free-form code only in a sandbox, only for analysis
The analyst may write pandas code, which runs out of process after a static AST policy
check. The child gets an isolated interpreter, an empty temp dir, a scrubbed environment
(no API keys), restricted builtins with a guarded `__import__`, a wall-clock timeout
(plus memory and CPU rlimits on POSIX), and truncated output. Analysis results inform
decisions; they never become artifacts.
*Production note:* for multi-tenant use, implement the `Sandbox` protocol with a container
backend (no network, read-only rootfs, cgroup limits). The process sandbox is the
portable default, not a hard security boundary against a determined attacker.

### D5. Holdout is report-only
The test split is created before anything else. Model selection and every critic
decision use cross-validation only. The holdout is scored once per evaluation, for
reporting.
*Why:* a reflection loop that looked at test scores would overfit the test set.

### D6. State holds references, not objects
Graph state is JSON only. DataFrames, fitted pipelines and analysis transcripts live in a
run-scoped `ArtifactStore` (path-traversal-safe keys).
*Why:* small, portable checkpoints, restart safety, and a clean swap to object storage.

### D7. Prompts treat dataset content as untrusted
Column names, category values and analysis output are wrapped in `<data>` tags, and the
system prompt tells the model to treat them as data. Independently, every answer is
schema-constrained and validated against the registry and the real columns, so an
injected instruction can at worst produce a rejected answer.

## Bounded cost and time

| Budget | Setting | Default |
|---|---|---|
| LLM tokens per run | `NOVAML_RUN_TOKEN_BUDGET` | 60k (sized for Groq free tier) |
| Critic improvement rounds | `NOVAML_MAX_IMPROVEMENT_ROUNDS` | 2 |
| Supervisor steps | `NOVAML_MAX_GRAPH_STEPS` | 40 |
| Analyst code steps | `NOVAML_ANALYST_MAX_STEPS` | 4 |
| Sandbox wall-clock per snippet | `NOVAML_SANDBOX_TIMEOUT_S` | 30s |
| Tuning iterations per model | `NOVAML_TUNING_ITERATIONS` | 10 |

## Observability (today)

- Structured JSON logs (structlog) with `run_id` and `agent` bound on every line.
- `events`: per-node status and latency. `llm_calls`: per-decision provider, model,
  tokens, cost, latency, source (llm/policy) and fallback reason.
- The model card aggregates LLM usage, CV and holdout metrics, lift over baseline and warnings.

Phase 5 replaces this with OpenTelemetry spans and exported metrics.
