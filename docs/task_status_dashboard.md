> Historical dashboard: this snapshot is not a live completion or final-result report. See [the 2026-10-11 results audit](RESULTS_STATUS_AUDIT_20261011.md), including unresolved metrics and incomplete exports.

# Task Status Dashboard

Refreshed **2026-08-22**. Frozen universe expanded **1096 → 1260** to include previously mid-layer-only multilayer gaps (cog C/mask, wan mask, etc.).

## Overall Status

| lifecycle status | count | percentage |
| --- | ---: | ---: |
| TRAIN_AND_EVAL_COMPLETE | 828 | 65.7% |
| TRAIN_COMPLETE_EVAL_NOT_STARTED | 341 | 27.1% |
| TRAIN_FAILED | 63 | 5.0% |
| TRAIN_PARTIAL_OR_INTERRUPTED | 16 | 1.3% |
| TRAIN_NOT_STARTED | 12 | 1.0% |

## Expansion (2026-08-22)

- Added **164** keys to close layer-grid gaps
- cogvideox: AD/CF/PI/mask (+ego_belief L20) → full cog layer grid
- wan: streaming_mask (+CF p24 holes, ego_belief L20) → full wan layer grid
- Other models (lingbot/minimax/vjepa2/videomae): already full-grid; single-layer baselines unchanged
- Backup of pre-expand ledger: `experiments/task_status_1096_before_expand_20260822.csv`

