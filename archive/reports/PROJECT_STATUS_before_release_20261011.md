# Probe Spatial — Project Status

Refreshed **2026-08-22**.

## Experiment Status (frozen universe)

Universe expanded **1096 → 1260** (multilayer gap fill).  
Source: `experiments/task_status_1096.csv` (filename kept; count is now 1260).

| lifecycle | count | pct |
| --- | ---: | ---: |
| TRAIN_AND_EVAL_COMPLETE | 828 | 65.7% |
| TRAIN_COMPLETE_EVAL_NOT_STARTED | 341 | 27.1% |
| TRAIN_FAILED | 63 | 5.0% |
| TRAIN_PARTIAL_OR_INTERRUPTED | 16 | 1.3% |
| TRAIN_NOT_STARTED | 12 | 1.0% |

Training: {'COMPLETE': 1169, 'FAILED': 63, 'PARTIAL_OR_INTERRUPTED': 16, 'JOB_DECLARED': 4, 'NOT_STARTED': 8}.

### Expansion notes
- Added 164 missing (model, probe, prefix, layer) keys so cog/wan match their full layer grids used by depth/view/ego probes.
- Audit: lingbot / minimax_h3 / vjepa2 / videomae already complete; dino/f3r/aether/qwen remain single-layer by design.
