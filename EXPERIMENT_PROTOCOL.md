# Experiment protocol

## Frozen decisions

- Split by scene with `configs/splits/inscene15k_v1.json`; do not tune on test.
- Run at least seeds 41, 42 and 43. Select layers and hyperparameters on val,
  then evaluate the selected frozen setting once on test.
- Report aggregate and Infinigen/ScanNet++ metrics with scene-bootstrap 95% CIs.
- B1 is conditioned on past object masks plus final-view global feature. B2 is
  conditioned on an object query. Neither receives camera pose or an explicit
  last/current-frame role flag.
- B1 ignores frames in which the conditioned object is not visible; it does not
  replace missing object tokens with global scene features. The only global B1
  context is the final sampled frame.
- The active B2 definition is query + all-frame patch tokens. There is no
  B2-last implementation or experiment in this repository.
- B1/B2 GT is expressed in the final observation's camera coordinate system.
- C1/C2/C3 input features must come from causal `context_segment` forwards, and
  target features must come from exact `target_isolated` rows. Do not feed these
  probes features sliced from the normal full-clip cache.
- ScanNet++ caches must be extracted after the valid-frame indexing fix; older
  caches used all JPGs while the dataset indexed only image+mask pairs.

## Required baselines

- A2: majority, balanced accuracy, AUROC and PR-AUC.
- A3: paired ranking, AUROC, multiple shuffle seeds and shuffle strengths.
- B1/B2: object-conditioned random-feature and unconditional controls.
- C1: no-action, shuffled-action and last-observation-copy controls.
- Every probe: a parameter-matched shallow/linear head where applicable.

## B1/B2 readout capacity

B1 and B2 expose `model.probe.decoder_type` with three values:

- `linear`: one affine readout, no attention or hidden nonlinearity.
- `mlp`: one hidden layer with GELU and dropout, no attention.
- `transformer`: the existing sequence/all-patch readout.

For B1, linear/MLP flatten the ordered masked object tokens, the final-view
global token and visibility bits. For B2, linear/MLP concatenate the object
query with one global-pooled token per ordered frame. The B2 Transformer still
sees every patch, so results across decoder types measure both readout capacity
and token-access differences; they must not be described as parameter-matched
architecture ablations.

Example:

```bash
python vidfm3d/train.py experiment=inscene15k_ext/ego_belief_wan_v1 \
  model.probe.decoder_type=linear
python vidfm3d/train.py experiment=inscene15k_ext/ego_belief_v2_wan_v1 \
  model.probe.decoder_type=mlp
```

## Deferred storage optimization

Do not implement node-local scratch staging or packed feature shards yet. Both
add path switching, copy lifecycle, capacity, indexing and cache-invalidation
failure modes. Reconsider them only after measurements show NAS saturation,
for example rising dataloader wait time, low GPU utilization attributable to
I/O, or metadata latency that worsens with job concurrency. Atomic cache writes,
checksums and manifests remain enabled because they are useful independently of
storage layout.

## Publishing results

Every result row must identify Git commit, dirty status, split-manifest hash,
feature-cache manifest, VFM model id/revision, feature layer/timestep, seed and
checkpoint. Do not publish a batch-local retrieval score as final R@K.

Training logs name C1's batch-local diagnostic `dyn_inbatch_R@1`; final reports
must use `global_R@1` from `eval_diag.py`.
