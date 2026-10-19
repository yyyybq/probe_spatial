# probe_spatial

Spatial diagnostic probes for frozen video foundation models, extending
[VidFM3D](https://github.com/zxhuang1698/VidFM3D).

The repository contains reusable research code, historical experiments, and
ongoing experiments. **Implemented code and completed jobs do not imply final
scientific results.** The [2026-10-15 experiment inventory](docs/EXPERIMENT_INVENTORY_20261015.md)
updates the [2026-10-11 inventory and release plan](docs/RELEASE_READINESS_20261011.md):
it separates canonical V2, timestep, and other analyses, records which
checkpoints were found to hold non-finite weights, and lists the remaining
release work by kind. The [2026-10-15 audit](docs/RELEASE_AUDIT_20261015.md)
and the [2026-10-11 audit](docs/RELEASE_AUDIT_20261011.md) list code fixes,
validation, and remaining release blockers.

Training now aborts with `NonFiniteInputError` when a feature batch contains
NaN/inf, and both evaluators record non-finite predictions as invalid samples
(`prediction_finite=False`, counted in the summary) instead of scoring them.
Summaries written by the updated evaluators carry an `evaluator_version` field;
numbers from older summaries are not overwritten.

## Probe suite

| Family | Implementation | Scope |
| --- | --- | --- |
| A1 | `streaming_depth`, `streaming_movement`, `streaming_mask` | Depth, camera motion, and identity |
| A2 | `view_consistency` | Shared viewing region |
| A3 | `abnormal` | Legacy non-streaming temporal-shuffle control |
| B1 / B2 | `ego_belief`, `ego_belief_v2` | Hidden-object localization; B2 uses an appearance query |
| C1 / C2 | `action_dynamics`, `path_integration` | Action-conditioned latent prediction and rollout |
| C3 | C1 checkpoint + action-sensitivity evaluation | Evaluation only; no separate C3 training |

Temporal experiments use ScanNet++ trajectories. Infinigen views are not a
video trajectory and cannot support temporal claims; see
[TEMPORAL_DATA_VALIDITY.md](TEMPORAL_DATA_VALIDITY.md).
C probes require exact isolated target-frame caches. See
[EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) and
[the C-probe contract](protocols/C_PROBES.md).

## Installation and tests

Use Python **3.11**. Run commands from the repository root. The editable install
is intended for this source checkout; configs and experiment data are not a
standalone Python wheel.

For CPU tests, without downloading datasets or model weights:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
python -m pytest -q
```

For GPU training and feature extraction, install a compatible CUDA PyTorch
build, then `python -m pip install -r requirements.txt` and
`python -m pip install -e . --no-deps`. Backends may need separate environments,
model access, or optional dependencies such as PyTorch3D. The historical
`requirements-lock.txt` is an environment snapshot, not a lock proven to work
for every backend. `requirements-dev.txt` records the CPU test environment.

Tests requiring frozen local campaign artifacts are preserved and excluded
from default collection. Run them where those artifacts exist with
`python -m pytest --run-artifact-tests -m artifact`. Local subprocess tests use
temporary synthetic inputs and remain in the default suite; see
[tests/README.md](tests/README.md).

## Run a streaming experiment

Keep data, features, and checkpoints outside the checkout. Define paths for
your machine:

```bash
export PROJECT_ROOT="$PWD"
export DATASET_ROOT=/data/InsScene-15K
export INSCENE_DATA_ROOT="$DATASET_ROOT/data"
export INSCENE_STREAMING_FEAT_ROOT="$DATASET_ROOT/FEAT_STREAMING"
export INSCENE_TARGET_FEAT_ROOT="$DATASET_ROOT/FEAT_TARGET"
```

Preview a single-model experiment before extracting features or training:

```bash
DRY_RUN=1 VFM=wan PROBES="view_consistency" PREFIX_LENGTHS="8" LAYERS="default" \
  bash scripts/run_streaming_probe_sweep.sh
```

Run the same command with `DRY_RUN=0` after preparing the data and model.
The script extracts the requested streaming caches and runs training/evaluation.
Set `EXTRACT_STREAMING=0 EXTRACT_TARGETS=0` when reusing compatible caches.
Set `PROBES="action_dynamics path_integration"` for C1/C2; those probes also
need target-isolated features. Models, layers, prefixes, target horizons, and
cache provenance must agree with the selected protocol.

An alternative end-to-end entrypoint is
`scripts/reproduce_streaming.sh`, configured with `INSCENE_BASE`, `VFMS`,
`LAYERS`, `PROBES`, and step flags `DOWNLOAD/EXTRACT/TRAIN/EVAL`.
Use `DRY_RUN=1` to inspect its commands. These general entrypoints do not
reconstruct the private cluster campaigns or their canonical-v2 receipts.

Offline model resolution honors `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`,
`HF_HOME`, `XDG_CACHE_HOME`, then the current user's default Hugging Face cache.
With `HF_HUB_OFFLINE=1`, missing or incomplete snapshots fail explicitly.
Optional AOSS storage requires its client and `AOSS_CONF` (default
`~/.petreloss.conf`); local datasets do not require AOSS.

## Results and reproducibility

- [2026-10-19 handoff](docs/HANDOFF_20261019.md): what the project is and the latest accepted evaluations. Read this before the dated status pages.
- [PROJECT_STATUS.md](PROJECT_STATUS.md): 2026-10-16 evidence snapshot; counts there are superseded by the handoff.
- [2026-10-19 audit](docs/RELEASE_AUDIT_20261019.md): independent check of the published branch, code fixes, per-group status and remaining release work. Accepted tables are `val`-split, seed-42 evaluations of final-epoch checkpoints, not held-out test results.
- `metadata/release_inventory/release_status_20261019.csv`: one row per trained run with its accepted evaluation and caveats; rebuild with `python3 scripts/build_release_status_20261019.py`.
- [2026-10-15 experiment inventory](docs/EXPERIMENT_INVENTORY_20261015.md): per-group completion, checkpoint tensor validity, accepted result scope, and remaining release work by kind.
- [Experiment inventory and release plan](docs/RELEASE_READINESS_20261011.md): the 2026-10-11 scope definitions for canonical V2, timestep, and other analyses.
- [Per-experiment CSVs](metadata/release_inventory/README.md): exact identities, evidence paths, completion, checkpoint tensor status and result status; rebuild with `python scripts/update_release_inventory_20261015.py` then `python scripts/summarize_release_inventory.py`.
- [Earlier results audit](docs/RESULTS_STATUS_AUDIT_20261011.md): historical ledger and earlier campaign snapshots, with links to later reconciliation.
- [File map](docs/FILE_MAP.md): source, experiment, and historical evidence locations.
- `experiments/task_status_1096.csv`: historical ledger with **1260** rows; the filename is retained for compatibility.
- `results/paper_ready_results.csv`: historical table eligibility, **not final-result certification**.

Check the historical tables with:

```bash
python scripts/check_task_status_consistency.py
python scripts/check_task_status_consistency.py --require-final
```

The current snapshot fails the first check because 49 eligible experiment IDs
have no exported paper-table row. The second also fails while the table contains
`POLICY_NOT_FROZEN` rows. Validation scores, unresolved metrics, smoke tests,
and submission receipts must not be presented as final held-out results.
Use the split, checkpoint-selection rule, seeds, and protocol documented for
each result; do not merge scores across incompatible protocols.

## Repository layout

| Path | Purpose |
| --- | --- |
| `vidfm3d/`, `features/`, `configs/` | Training, evaluation, extractors, and experiment configs |
| `tests/` | Portable regression tests plus opt-in artifact tests |
| `scripts/` | General entrypoints and retained historical cluster workflows |
| `protocols/`, `manifests/` | Scientific contracts and dataset selection records |
| `analyses/`, `active_repr_probe/` | Analysis tools and experimental extensions |
| `experiments/`, `results/`, `reports/` | Status ledgers and result evidence, each with its own scope |
| `archive/` | Historical documentation retained for traceability |

`runtime/`, `model_cache/`, and the adjacent `robocasa365/` workspace are local
working material excluded from the public source tree. Dated submit/recovery
scripts remain for provenance; they are not public quick-start commands.
The [previous README](archive/reports/README_before_release_20261011.md)
preserves older commands and experiment notes.

## License and attribution

The repository's [LICENSE](LICENSE) declares **CC BY-NC 4.0**. Third-party code
and model weights retain their own terms; see [THIRD_PARTY.md](THIRD_PARTY.md)
for the local notice inventory and unresolved attribution work.

This project builds on VidFM3D, Fast3R, CroCo/DUSt3R, VGGT, and the model
backends referenced in `features/`. Preserve their copyright headers and
notices when redistributing code. The upstream VidFM3D citation is:

```bibtex
@article{huang2025vidfm3d,
  title = {How Much 3D Do Video Foundation Models Encode?},
  author = {Huang, Zixuan and Li, Xiang and Lv, Zhaoyang and Rehg, James M.},
  journal = {arXiv preprint arXiv:2512.19949},
  year = {2025}
}
```
