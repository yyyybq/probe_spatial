# Training logic audit

This document follows one diagnostic-probe run from command line to final
metrics. It describes the code as it exists, including behavior that still
needs an experimental decision.

## 1. Entry point and configuration

1. Training starts with `python vidfm3d/train.py experiment=<name> [overrides]`.
2. Hydra composes `configs/train.yaml`, the selected experiment, model,
   callbacks, logger, trainer and path configs. The resolved output directory is
   `logs/<task>/runs/<run_folder_name>`.
3. The default seed is 42. Lightning seeds Python, NumPy, PyTorch and workers.
4. A run manifest is written before data/model construction. It records the
   resolved config, Git state, selected environment roots and split manifest.
5. `ProbeExtensionLitModule` jobs requesting more than eight processes are
   rejected unless `allow_large_probe_ddp=true`. Independent experiments should
   be distributed with a Slurm array, not one hundred-way DDP.
6. Autoresume looks only for `<output>/checkpoints/last.ckpt`. Because experiment
   names use a stable run folder, rerunning the same experiment resumes it.
7. `torch.load` is globally wrapped with `weights_only=False`, and Lightning's
   checkpoint callback is monkey-patched to force `last.ckpt` updates.

Important consequence: changing a scientific override while retaining the same
run folder can resume an incompatible old run. Give decoder, seed and feature
layer sweeps distinct `job_name`/`paths.run_folder_name` values.

## 2. Required caches

- `normal`: frozen VFM features used by non-causal/full-clip probes.
- `shuffled`: the same frame identities encoded in shuffled temporal context;
  required by A3.
- `context_segment`: causal input segments forwarded as videos without future
  target frames; required by C1/C2/C3 inputs.
- `target_isolated`: exact future target-frame features from an independent
  forward; required by C1/C2/C3 targets.
- `streaming_prefix`: each observed prefix is forwarded independently. This is
  now the shared online-history cache for streaming A1/A2/B1/B2/C1/C2/C3.
  The default sweep uses exact prefix lengths `4,8,16,32,64`; training keeps one
  fixed `prefix_len` per run to avoid variable-length batches.

Wan defaults to layer 20 at timestep 749, CogVideoX to layer 20 at timestep 749,
and V-JEPA2 to layer 23. Cache files are safetensors with atomic replacement,
checksums and sidecar manifests. Dataset channel counts and `feat_postfix` must
match the extractor exactly.

## 3. Dataset construction and split

1. `VideoProbeDataModule` evaluates the dataset constructor strings from YAML.
2. `InsScene15KDataset` scans Infinigen scenes and ScanNet++ frames having both
   image and instance-mask files.
3. `INSCENE_SPLIT_MANIFEST` filters scene keys before window expansion. Without
   a manifest, train/val use a deterministic legacy 90/10 scene split. A real
   test split requires the manifest.
4. Long scenes become length-200 windows at stride 100. Split happens first, so
   windows from one scene cannot cross train and validation.
5. Train uses a random or distributed sampler and `drop_last=true`; validation
   is sequential/distributed and keeps the last batch.
6. Train sampling is stochastic. Val/test now default to deterministic
   per-index sampling (`seed=0`) even when the YAML omits a seed.

The use of Python `eval` for dataset strings is trusted-config behavior, not a
safe general configuration API.

## 4. Constructing one sample

For the standard four-view probes:

1. Select four ordered indices within at most the first 76 frames of the scene
   window, normally with a minimum gap of five.
2. Snap indices to the configured feature-grid divisor (currently four).
3. Load RGB, instance IDs, depth, intrinsics and world-to-camera extrinsics.
4. Resize to 288x512, scale intrinsics and mark positive depth as valid.
5. Back-project depth to point maps. Rebase geometry to the first selected
   camera and normalize scene scale.
6. Load the scene cache and map each original frame index linearly to a cached
   temporal feature index. With `feat_pixalign=true`, return only selected
   features as `(S,Hf,Wf,C)` fp32 tensors.
7. Remap instance IDs within the sample and generate probe-specific targets.

For C2/C3 with explicit horizons, indices are `[t, t+h1, ...]` and a
geometry-only fast path avoids loading RGB/masks. Frame `t` is the
action-reference frame used to encode relative motion; it is not an additional
condition beyond the input feature sequence.

For streaming probes, `InsScene15KDataset(streaming_prefix=True)` expands each
scene into online prefix samples. A/B probes consume the prefix itself. C probes
load the observed input from the `streaming_prefix` cache `[I_0..I_t]`, encode
actions from the prefix tail `I_t` to future target frames, and load future
supervision from `target_isolated`. The streaming C2/C3 default horizons are
`[1,4,8,16]`, so `prefix_len=64` remains feasible for 81-frame Wan clips.

## 4.1 Current streaming code path audit

This was rechecked against the current implementation after introducing the
shared streaming interface:

1. `features/run_inscene15k.py --mode streaming_prefix --prefix-lengths
   "4,8,16,32,64"` forwards each requested prefix independently and writes
   `prefix_<tail>/feature...sft` plus `prefix_index.npy`.
2. `scripts/run_streaming_probe_sweep.sh` extracts that cache once and trains
   separate fixed-shape jobs for each `(probe, prefix_len, layer)`. Mixed prefix
   lengths are not collated in the same batch.
3. Streaming A1/A2/B1/B2 configs set `streaming_prefix=True` and
   `prefix_min_len=prefix_max_len=prefix_len`; the dataset therefore returns
   exactly the selected prefix.
4. Streaming B1/B2 use `streaming_shared_hidden_obj=True`: the object id is
   selected once per scene from the first four-frame prefix. It must be visible
   in frames `[0,1,2]`, hidden at frame `3`, and hidden at every configured
   prefix tail such as `[7,15,31,63]`. Prefix jobs then reuse that raw object id.
5. Streaming B1/B2 geometry is rebased with `ref_idx=len(prefix)-1`, and
   `compute_object_target_for_id(..., last_frame_idx=-1)` expresses the shared
   object's target in the current prefix-final camera coordinates. Only masks
   inside the current prefix are returned.
6. Streaming B1 receives per-frame object masks plus the global pooled final
   prefix frame. No camera pose is supplied.
7. Streaming B2 receives one object query plus all patch tokens from the prefix.
   No camera pose and no explicit current-frame role token are supplied.
8. Streaming C1/C2/C3 go through `_getitem_feature_action_diag`: input features
   are loaded from the prefix cache; target features are loaded from
   `target_isolated`; actions are encoded from the prefix tail to each future
   target frame.
9. Non-streaming C1 still uses `context_segment` for input and
   `target_isolated` for target. Non-streaming C2/C3 use the same fast path but
   load `context_segment` inputs ending at their action-reference frame.

## 5. Probe-specific supervision

### A2 view consistency

- GT projects valid source-frame 3D points into each destination frame, counts
  positive-depth in-bounds projections and symmetrizes with the minimum of both
  directions. It does not test destination depth or occlusion consistency.
- The head globally pools each frame, projects it, builds pair features
  `[zi,zj,zi*zj,abs(zi-zj)]`, and applies an MLP.
- Loss is hard BCE for overlap >=0.4 or <=0.05 plus 0.5 times soft BCE on all
  off-diagonal overlaps.
- Final evaluation reports MAE, hard/balanced accuracy, AUROC and PR-AUC.

### A3 abnormal temporal context

- Each item loads paired normal and shuffled-context caches. Missing shuffled
  caches invalidate the item rather than using zero as an easy cue.
- Normal and shuffled examples are concatenated with balanced labels 0/1.
- The head spatially pools frames and uses a learned-position Transformer CLS
  classifier. Loss is binary cross entropy.
- Final evaluation reports paired threshold accuracy, paired ranking accuracy,
  AUROC, PR-AUC and score separation.

### B1 hidden-object localization

- Candidate object: >=200 valid pixels in at least one past frame and <200 in
  the final frame. Selection favors the most qualifying past frames, then pixel
  count. Samples without a candidate are invalid.
- In streaming configs, the candidate is not reselected per prefix. One raw
  object id is selected from prefix `4`: visible in frames `[0,1,2]`, hidden at
  frame `3`, and hidden at every configured prefix tail. The same object id is
  reused for prefix `4,8,16,32,64`; the target reference camera still changes
  with the current prefix tail.
- Condition: GT object masks in qualifying past frames plus one global-pooled
  final-frame feature. Invisible object frames contribute no object token and no
  per-frame global substitute.
- Target: object centroid transformed to the final camera, represented as
  `(azimuth, elevation, log_distance)`. The final camera pose is a label
  convention and is not supplied to B1.
- Transformer: masked-pool one object token per visible frame, add final global
  token and order embeddings, mask invisible object tokens and regress from a
  learned query. In streaming configs this Transformer accepts variable
  sequence lengths up to `max_seq_len=64`.
- Linear/MLP: flatten the four ordered masked object tokens, the final global
  token and four visibility bits. `decoder_type={linear,mlp,transformer}`
  selects the readout. Linear/MLP remain fixed-length and are not used for
  streaming sweeps.
- Loss: weighted SmoothL1 with weights `[1,1,0.5]`; azimuth residual is wrapped
  on the circle. Metrics are azimuth/elevation degree error and log-distance
  absolute error.

### B2 object-query belief

- Candidate and target are exactly the same as B1.
- Condition: mask-pool the chosen object from the past frame where it has the
  most pixels, yielding one object-query vector. The mask and camera pose are
  not passed to the head, although the backbone vector may encode position.
- Transformer: the query attends to every pooled patch from all ordered frames
  with learned frame/row/column embeddings. In streaming configs the number of
  ordered frames is the selected `prefix_len` and may be 4, 8, 16, 32 or 64.
  There is no explicit current-frame role embedding.
- Linear/MLP: concatenate the object query with one globally pooled token per
  ordered frame. These modes do not receive all patches, remain fixed-length,
  and are not used for streaming sweeps.
- Output: a joint 16x8 azimuth/elevation classification and scalar log distance.
  Loss is joint-bin CE plus 0.3 SmoothL1. Metrics are top-1/top-3, spherical
  angular error and log-distance error.
- No `B2-last` class, config, script or dispatch branch exists. The only current
  B2 is the all-frame definition above.

### C1 action-conditioned prediction

- Input is loaded from a `context_segment` cache ending at the last input frame,
  e.g. `[I_1..I_48]` forwarded together without the future target. If the
  context segment or exact target row is missing, the sample is invalid.
- In streaming configs, input instead comes from the shared `streaming_prefix`
  cache `[I_0..I_t]`. The target is `t+h`; C1 defaults to `h=1`.
- Action is the relative camera transform from the context tail to target
  camera: first two rotation rows plus translation (nine values). Target is the
  spatial mean of the exact isolated target feature.
- The head pools input frames, combines them with action/query tokens in a
  Transformer and predicts a C-dimensional vector.
- Training loss is MSE plus cosine distance only. In-batch R@1 is computed under
  `no_grad`, is omitted (`nan`) below four valid examples, and is diagnostic.
- Final `eval_diag.py` stacks the full evaluation set for global R@1/R@5/rank and
  compares no-action, shuffled-action and last-observation controls.

### C2 path integration

- Input is the causal context segment ending at the action-reference frame.
  Consecutive actions describe each waypoint step; exact isolated targets
  supervise every horizon.
- In streaming configs, the action-reference frame is the current prefix tail
  and the input is the corresponding `streaming_prefix` feature. Default
  horizons are
  `[1,4,8,16]`.
- A stacked GRU recurrently updates state and predicts one pooled target feature
  per step. Valid horizons use MSE plus cosine distance.
- Evaluation retrieves poses through target features and reports global
  retrieval, final/step pose error, drift, horizon errors and loop closure.

### C3 counterfactual action

- Input is the same causal context segment ending at the action-reference frame.
  Each action maps that frame directly to one horizon. Horizons with missing
  context/target rows or reference-target overlap below 0.05 are invalid.
- In streaming configs, the input is again the prefix cache ending at the
  current tail, with default horizons `[1,4,8,16]`.
- A Transformer jointly reads context plus alternative action tokens and predicts
  one pooled target per intervention. Loss is MSE plus cosine distance.
- Evaluation reports global retrieval, correct-target cosine, intervention hit
  rate/margin and per-horizon results.

### Sparse autoencoder

- Randomly sample at most 8192 activation tokens; LayerNorm, ReLU encoder and
  top-k masking produce sparse codes; a linear decoder reconstructs activations.
- Base loss is reconstruction MSE plus weighted L1. Optional overlap and ego
  MLP readouts consume frame-mean sparse codes, detached by default.
- This is not a capacity-matched substitute for the direct probes.

## 6. Optimization, validation and output

- AdamW defaults to lr 3e-4, betas `(0.9,0.95)`, weight decay 0.05.
- Scheduler is five warmup epochs followed by cosine decay to 1e-6.
- Diagnostic configs normally use one GPU, bf16 mixed precision, batch size 4,
  50 epochs and no early stopping.
- Training metrics are logged per step. Validation metrics are synchronized and
  averaged by Lightning. Checkpoints and `last.ckpt` are written every five
  epochs; `save_top_k=-1` retains all scheduled checkpoints.
- `train.py test=true` uses `validation_datasets` and the explicitly supplied
  checkpoint. Publication evaluation should instead use `eval_diag.py` with a
  frozen split manifest and explicit checkpoint.

## 7. B1 versus B2, in one table

| Detail | B1 | B2 |
|---|---|---|
| Object specified | past per-frame masks | one mask-pooled object query |
| Camera pose input | no | no |
| Final-frame feature | one global-pooled final-view token; no final object token if hidden | all final patches in Transformer mode |
| Other scene content | final-view global only | all-frame patches in Transformer mode |
| Output | continuous polar regression | joint angle bins + continuous log distance |
| Default head | 2-layer Transformer | 4-layer all-patch Transformer |
| Linear/MLP input | masked tokens + final global + visibility | query + global frame summaries |

Thus B1 asks whether object-conditioned features themselves preserve its
location. B2 asks whether an object query can retrieve/localize that object from
the whole clip representation. B2 is a materially stronger readout and should
not be presented as merely another loss for B1.

## 8. Open issues requiring a decision

These are not silently fixed because several change the scientific task.

1. **B2 final-frame access (high).** B2 reads all final patches while hidden means
   fewer than 200 pixels, not zero pixels. It may directly detect a small residual
   target. Decide whether hidden must be exactly absent and whether B2 should
   exclude the final frame.
2. **Hidden-object centroid (medium).** The centroid uses every valid occurrence,
   including sub-threshold final-frame pixels, while the condition mask excludes
   them. This is label construction, not probe input, but should be stated or
   changed to past-only centroid estimation.
3. **A2 visibility GT (high).** In-bounds projection without a destination-depth
   consistency test counts geometrically occluded points as overlap.
4. **B2 decoder comparison (medium).** Linear/MLP use global frame summaries;
   Transformer uses every patch. This is a useful weak-readout control but not a
   parameter- or information-matched architecture ablation.
5. **Cache temporal alignment (medium).** Original frame indices are mapped
   linearly to cached latent time. This is approximate when model temporal
   downsampling or frame selection is nonuniform and needs a spot-check per VFM.
6. **Resume-level data reproducibility (medium).** Train workers' RNG state is not
   checkpointed, so an interrupted run can replay a different frame-sampling
   stream after resume even though model/optimizer state resumes exactly.
7. **Zero-valid validation batches (medium).** They contribute a graph-connected
   zero with weight one to Lightning epoch aggregation, slightly biasing val loss
   when invalid examples cluster in a batch. Final `eval_diag.py` filters records.
8. **True test through `train.py` (medium).** `test_dataloader` reuses
   `validation_datasets`; only `eval_diag.py` rewrites the split for final test.
9. **Legacy absolute paths (migration).** Main diagnostic YAMLs use environment
    roots, but old launch/resume/evaluation scripts still contain
    `/nas/baiqiao` or `/data/baiqiao` defaults. They are not portable entrypoints.
10. **A3 strength coverage (experimental).** Current caches represent one full
    permutation seed. Multiple seeds and partial/local shuffle strengths remain
    required by the protocol.
11. **Cross-probe capacity matching (experimental).** Decoder selection now
    covers B1/B2 only. A2/A3/C1/C2/C3 retain different task-specific nonlinear
    heads and do not yet have one unified parameter-matched linear/shallow suite.
12. **Naming debt.** `ego_belief_wan_poseonly.yaml` actually activates the
    unconditional zero-feature baseline; the filename is historical and can
    mislead result tables.
13. **Storage optimization deferred.** Node-local scratch staging and packed
    shards are intentionally absent. Reconsider only after measurements attribute
    low GPU utilization or rising loader latency to NAS saturation.
