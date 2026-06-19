# Experiment protocol

## Frozen decisions

- Split by scene with `configs/splits/inscene15k_v1.json`; do not tune on test.
- Run at least seeds 41, 42 and 43. Select layers and hyperparameters on val,
  then evaluate the selected frozen setting once on test.
- Report aggregate and Infinigen/ScanNet++ metrics with scene-bootstrap 95% CIs.
- B1 is conditioned on past object masks. B2 is conditioned on an object query.
  Neither receives camera pose or an explicit last/current-frame role flag.
- B1/B2 GT is expressed in the final observation's camera coordinate system.
- C1/C2/C3 target features must match their action frame exactly.
- ScanNet++ caches must be extracted after the valid-frame indexing fix; older
  caches used all JPGs while the dataset indexed only image+mask pairs.

## Required baselines

- A2: majority, balanced accuracy, AUROC and PR-AUC.
- A3: paired ranking, AUROC, multiple shuffle seeds and shuffle strengths.
- B1/B2: object-conditioned random-feature and unconditional controls.
- C1: no-action, shuffled-action and last-observation-copy controls.
- Every probe: a parameter-matched shallow/linear head where applicable.

## Publishing results

Every result row must identify Git commit, dirty status, split-manifest hash,
feature-cache manifest, VFM model id/revision, feature layer/timestep, seed and
checkpoint. Do not publish a batch-local retrieval score as final R@K.

Training logs name C1's batch-local diagnostic `dyn_inbatch_R@1`; final reports
must use `global_R@1` from `eval_diag.py`.
