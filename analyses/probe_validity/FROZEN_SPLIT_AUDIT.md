# Frozen Split Audit

Status: `NOT_INDEPENDENT`

Audit date: 2026-08-30. This audit distinguishes a split file that exists from
the split assignment actually bound to a run. No score from a candidate test
split was used to reach this classification.

## 1. Split inventory

| Split record | Created by | Date / commit | Rows | Scene / trajectory / sample counts | SHA256 | Actual use |
|---|---|---|---:|---|---|---|
| `configs/splits/inscene15k_v1.json` | `scripts/make_inscene_split_manifest.py` | commit `df620e1093c3f8335d6245fadffcd15cb5f88572`, committed 2026-06-19; file mtime 2026-06-29 | 2,370 scene assignments | train 1,896; val 237; test 237. ScanNet++ only: 725 / 91 / 91 | `e01a2194143405ed8f08613ec0bc672c7c237fa9bb21d2b8da441e7ad56bf086` | Declared project split, but not bound to the C h124 manifests or the historical B2 runs audited below. |
| `manifests/c_retrain_20260722/inscene_c_h124_prefix{8,12}.jsonl` | `scripts/build_inscene_streaming_manifest.py` | 2026-07-22; metadata `git_commit=null` | 2,724 per prefix | 681 scenes; 2,724 scene-window samples; train 2,448, val 276 | P8 `772194f077c45cc85a942f17f46f50103125ae8f5b1c983c08881c1759607e0c`; P12 `0ae3f4dbe26ddc2ba6b936b7a5bde5353a8d912da304287533cff6182d93851b` | Historical C1/C2/C3 train/eval. Manifest metadata contains no split-manifest provenance; assignments match the builder's seed-42 legacy 90/10 branch, not `inscene15k_v1`. |
| `manifests/c_retrain_20260722/inscene_c_h124_prefix{16,24}.jsonl` | same builder | 2026-07-22; metadata `git_commit=null` | 2,723 / 2,721 | 681 scenes; 2,723 / 2,721 samples; P24 train 2,445, val 276 | P16 `875e6f106b85dd6ca5e54707af98425772bf6596d198e9ae478f8777313e6aa0`; P24 `ebc496eb0587cda372ccf29a7c5244ef454096151f73c8eed9511967ded1902d` | Historical C1/C2/C3 train/eval. Same legacy split lineage. |
| `artifacts/tstep_canonical_v2/manifests/inscene_c_h124_prefix{8,12,16,24}.jsonl` | `scripts/build_tstep_canonical_manifests.py` | 2026-08-30; source hashes pinned, no independent split creation | 2,721 each | 681 scenes; 2,721 shared scene-window units; train 2,445, val 276 | P8 `01c1d682...`; P12 `3ddf7322...`; P16 `16a5a007...`; P24 `864bc031...` | Aligns temporal windows across prefixes. It inherits the historical C split and does not create a held-out test. |
| Historical B2 P12 runs, for example `logs/inscene15k_streaming/runs/inscene15k_streaming_b2_vjepa2_p12_layer23/.hydra/config.yaml` | runtime branch in `vidfm3d/data/components/inscene15k_dataset.py` | run artifacts predate this audit; config has no scene manifest or split-manifest binding | no persisted full manifest; 165 valid-object eval samples in the replay artifact | 65 observed eval scenes for V-JEPA2; trajectory ID is only recoverable as scene ID | no run-bound manifest hash | `eval_split=val`; dataset code used deterministic legacy scene split when the environment did not bind a split manifest. This is development validation, not test. |

P8/P12/P16/P24 canonical-v2 full hashes are shown above in abbreviated form
where space is tight; the authoritative complete hashes are in each adjacent
`.meta.json`. The exact P12 hash is shown in full because P12 is the primary
protocol in this study.

## 2. Assignment unit

- `inscene15k_v1`: source-stratified **scene** assignment before expansion. The
  builder sorts each source, applies NumPy seed 42, and assigns test, val, then
  train.
- Historical C h124: **scene** assignment from the legacy deterministic 90/10
  branch, followed by up to four motion-uniform windows per scene. A trajectory
  is therefore represented by `(scene_id, streaming_window_id)`; `sample_id`
  identifies that window and prefix.
- Canonical-v2: inherits the historical scene split and intersects exact shared
  `(scene_id, window_id)` units across prefixes.
- Historical B2: **scene** assignment at runtime, then temporal-window and valid
  hidden-object filtering. There is no evidence of random row-level splitting.

## 3. Exact overlap audit

The P12 historical C manifest contains 612 train scenes and 69 val scenes.
Within that manifest, train/val overlap is zero for scene IDs, namespaced raw
frame IDs, `(scene, window)` trajectory IDs, sample IDs, and namespaced source
paths. The same is true for canonical-v2 P12. Raw frame names without a scene
namespace are not valid identifiers and were not used for leakage decisions.

The decisive cross-manifest overlap is:

| Current use | Master train | Master val | Master test | Outside master ScanNet++ universe |
|---|---:|---:|---:|---:|
| C h124 train scenes (612) | 488 | 61 | **63** | 0 |
| C h124 val scenes (69) | 55 | 5 | **9** | 0 |
| B2 replay val scenes with valid targets (65) | 49 | 6 | **10** | 0 |

Thus 72 of the 91 nominal ScanNet++ test scenes in `inscene15k_v1` occur in
the C h124 development universe, including 63 used for probe training. At
least 10 nominal test scenes also appear in the observed B2 validation rows.
This is direct scene-level overlap, so trajectory, sample, frame, and path
independence cannot be claimed for that candidate test split.

## 4. Freeze and selection history

Evidence that the protocol was developed on validation rather than frozen
before test evaluation:

- `EXPERIMENT_PROTOCOL.md` says layers and hyperparameters are selected on val
  and evaluated once on test, but no final test table or run-bound test
  manifest was found.
- `README.md` documents `eval_split=val` for ordinary evaluation and a separate
  intended `eval_split=test` command. Historical C and B2 Hydra configs audited
  here use `eval_split=val`.
- `docs/probing_spatial_progress.md` records prefix/layer interpretation and
  explicitly states that artifacts are mainly validation summaries and that no
  final held-out test publication artifact was found.
- `docs/selection_leakage_audit.md` says best-layer and best-prefix headline
  claims require a held-out confirmation and that hyperparameter provenance is
  incomplete.
- `scripts/run_missing_eval_shard.py` hard-codes `+eval_split=val`; the full
  missing-eval campaign therefore does not supply a test confirmation.
- Candidate-pool design, action interventions, B2 priors, and order
  perturbations in `analyses/probe_validity/run_probe_validity.py` were created
  after the validation artifacts existed. They were not predeclared against an
  untouched test set.

No evidence was found that the master test was intentionally used for
checkpoint selection under its `test` label. That does not rescue independence:
its scenes were reassigned into the actual training and validation manifests.

## 5. Classification and consequence

Classification: **`NOT_INDEPENDENT`**.

`inscene15k_v1` is a real deterministic scene split, but its ScanNet++ test
partition is not independent of the probes under study. The actual C and B2
pipelines used different legacy scene assignments, consuming nominal test
scenes during training and development. The current val split remains usable
for the selected paper-strengthening analyses, with development-set language
and clustered uncertainty. It must not be called untouched test.

Per the study gate, no new split is silently generated and the final frozen
confirmation stage is blocked. A future confirmatory split must be created at
scene level from scenes absent from every historical train/val manifest, pinned
before any score inspection, and accompanied by exact scene, trajectory,
sample, raw-frame, and source-path overlap checks.
