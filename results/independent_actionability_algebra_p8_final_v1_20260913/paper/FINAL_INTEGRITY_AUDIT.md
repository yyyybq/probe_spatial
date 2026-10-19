# Final Integrity Audit

Status: **COMPLETE_VALID**

| Check | Result |
|---|---|
| frozen request SHA256 | PASS |
| feature-label join and actual-load audit | PASS |
| DINO exact feature coverage 12,358/12,358 | PASS |
| DINO final evaluator complete | PASS |
| F3R exact feature coverage 12,358/12,358 | PASS |
| F3R final evaluator complete | PASS |
| V-JEPA2 exact feature coverage 12,358/12,358 | PASS |
| V-JEPA2 final evaluator complete | PASS |
| Wan exact feature coverage 12,358/12,358 | PASS |
| Wan final evaluator complete | PASS |
| nonempty actionability_per_sample.csv | PASS |
| nonempty dynamics_per_sample.csv | PASS |
| nonempty algebra_per_sample.csv | PASS |
| candidate-swap equivariance (all direct scorers) | PASS |
| three full-dynamics checkpoints per model | PASS |
| RUN_MATRIX has no pending or running branch | PASS |
| nonempty FIGURE_1_ACTIONABILITY.pdf | PASS |
| nonempty FIGURE_2_DYNAMICS_ALGEBRA.pdf | PASS |
| nonempty FIGURE_3_DIFFICULTY_LOOPS.pdf | PASS |

Validation was used during pilot development and is not represented as a blind test. No old C-series checkpoint or canonical-v2 feature store is part of these final result inputs.

Rebuild all aggregate tables, figures and result prose without retraining:

```bash
PAPER_OUTPUT_DIR=/tmp/independent_p8_paper_rebuild bash scripts/build_independent_p8_final_paper.sh
```
