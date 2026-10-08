# Evaluation

`novaml eval` runs benchmark datasets through the real service and scores both
the **outcome** and the **decisions** the agents made.

```bash
novaml eval                                  # core suite, deterministic policies (offline, ~2 min)
novaml eval --mode llm                       # same suite, Groq makes the decisions (~20 min, ~240K tokens)
novaml eval --mode llm --compare-with evals/baseline_policy.json
novaml eval --suite extended                 # real OpenML datasets (network)
novaml eval --gate evals/baseline_policy.json   # exit 1 on regressions (runs in CI)
```

Reports go to `evals/results/` as JSON and Markdown.

## Cases

The **core** suite is offline and seeded, so it is reproducible and can gate CI.

| case | what it tests |
|---|---|
| iris, wine, breast_cancer, digits, diabetes | bundled sklearn data: quality floors; strong legitimate features must survive |
| leak_missingness | Titanic-style `boat`: filled in only for survivors; must be dropped, and the score must stay realistic |
| leak_target_copy | the target re-encoded in another unit must be dropped; `f3*f4` is a real interaction that must be kept |
| messy_churn | IDs, dates, missing values, a constant, missing targets |
| imbalanced_fraud | 5% minority: a class-balanced metric must be chosen and the models class-weighted |
| high_cardinality_noise | a 900-level code must not look predictive |
| pure_noise | no signal: an honest pipeline must not beat the baseline |
| skewed_regression | log-normal target and size feature |

The **extended** suite (titanic with `boat`, adult, credit-g, California housing) pulls
real data from OpenML.

## Checks

| check | fails when |
|---|---|
| `min_score` | the holdout eval metric (balanced accuracy / R²) is below the case floor |
| `max_score` | the score is *too good*: leakage got through |
| `drop:<col>` | a planted leak or identifier was kept |
| `keep:<col>` | a column with real signal was dropped |
| `metric_choice` | an inappropriate optimisation metric was chosen |

Scores use a fixed metric per problem type, so runs that optimised different metrics stay
comparable. LLM and policy runs use identical settings (3-fold CV, 1 improvement round,
5 tuning iterations), so the comparison isolates the decision-maker.

## Regression gate

`evals/baseline_policy.json` is the committed reference. In CI the policy suite runs on
every push, and the build fails if a check that passed in the baseline now fails, a run
crashes, or a score drops by more than 0.02. Disabling leakage detection, for example,
fails the gate on four checks with precise reasons.

Update the baseline deliberately, when a change is supposed to move the numbers:

```bash
novaml eval --write-baseline evals/baseline_policy.json
```
