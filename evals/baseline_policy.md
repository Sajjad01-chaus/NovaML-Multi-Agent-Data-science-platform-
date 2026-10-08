# NovaML eval: `policy`

2026-10-08T08:56:04+00:00 · 12/12 cases passed · 35/35 checks · decisions 23/23 · mean lift 0.5018 · 0 tokens · 128.5s

| case | pass | score | baseline | metric optimised | best model | failed checks | LLM decisions | tokens | s |
|---|---|---|---|---|---|---|---|---|---|
| iris | yes | 0.9667 | 0.3333 | f1_macro | linear |  | 0/4 | 0 | 6.2 |
| wine | yes | 1.0000 | 0.3333 | f1_macro | linear |  | 0/4 | 0 | 6.2 |
| breast_cancer | yes | 0.9692 | 0.5000 | f1_macro | linear |  | 0/4 | 0 | 9.0 |
| digits | yes | 0.9690 | 0.1000 | balanced_accuracy | random_forest |  | 0/4 | 0 | 38.6 |
| diabetes | yes | 0.4541 | -0.0120 | r2 | linear |  | 0/4 | 0 | 5.4 |
| leak_missingness | yes | 0.7772 | 0.5000 | f1_macro | gradient_boosting |  | 0/4 | 0 | 5.8 |
| leak_target_copy | yes | 0.8562 | -0.0325 | r2 | gradient_boosting |  | 0/4 | 0 | 3.8 |
| messy_churn | yes | 0.7113 | 0.5000 | f1_macro | gradient_boosting |  | 0/4 | 0 | 6.4 |
| imbalanced_fraud | yes | 0.8401 | 0.5000 | roc_auc | linear |  | 0/4 | 0 | 29.0 |
| high_cardinality_noise | yes | 0.8079 | 0.5000 | f1_macro | linear |  | 0/4 | 0 | 6.2 |
| pure_noise | yes | 0.5189 | 0.5000 | f1_macro | linear |  | 0/4 | 0 | 7.1 |
| skewed_regression | yes | 0.8733 | -0.0000 | r2 | linear |  | 0/4 | 0 | 4.8 |
