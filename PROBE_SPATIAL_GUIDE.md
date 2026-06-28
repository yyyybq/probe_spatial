# Probe Spatial — Spatial Representation Diagnostic Suite

> A single-document onboarding guide. After reading this top-to-bottom you should
> understand **what we are studying, why, how the code is organized, what every
> moving part does, and exactly which command to run to reproduce / extend
> any result**.

> **Current-status note.** This guide was originally written before B2, C2/C3,
> and the shared streaming-prefix protocol were added. It remains useful for
> background and code orientation, but the authoritative current protocol is:
> `README.md` for commands, `EXPERIMENT_PROTOCOL.md` for frozen experimental
> decisions, and `TRAINING_LOGIC_AUDIT.md` for the line-by-line training logic.
> Current code supports A1/A2/B1/B2/C1/C2/C3 streaming with prefix lengths
> `4,8,16,32,64`; B1/B2 receive no camera pose, B1 uses object masks plus final
> global feature, B2 uses object query plus prefix patch tokens, and C probes
> use `streaming_prefix` inputs with exact `target_isolated` future targets.

---

## 0. TL;DR

**Goal.** We probe Video Foundation Models (VFMs — Wan2.1, CogVideoX, V-JEPA2,
…) for *spatial understanding* by training small heads on top of their frozen
features and measuring how well the heads can recover quantities that require
3D / temporal / ego-centric reasoning. The result is a **diagnostic suite**: a
handful of orthogonal probes that, taken together, characterize what kind of
"space" a VFM has internalized.

**Inputs.** Pre-extracted, pixel-aligned VFM features over short clips of the
**InsScene-15K** dataset (Infinigen + ScanNet++).
**Outputs.** Per-probe validation/test metrics + per-sample dumps that we
aggregate into a **VFM × probe** comparison table.

**What is *not* in this repo.** Training the VFMs themselves. We never touch
the VFM weights — every probe is read-only on top of a frozen feature cache.

---

## 1. Scientific framing — three layers of "spatial understanding"

We split probes into three families. Each family answers a qualitatively
different question. **All probes share the same dataset and feature cache;
only the labels and heads differ.**

| Family | Question we are asking | Probes |
|---|---|---|
| **A. Global Spatial Perception** | Does the VFM perceive a coherent 3D scene across frames? | A1 depth/camera/identity (already in v3 baseline), **A2 view consistency**, **A3 abnormal video** |
| **B. Ego-Centric Belief** | Does the VFM remember *where things are* even when they leave the field of view? | **B1 hidden-object localization** |
| **C. Action-Conditioned Prediction** | Can the VFM imagine the next observation under a known camera move? | **C1 latent dynamics** |

A1 was already implemented in the upstream VidFM3D project. The new work in
this repo is **A2 + A3 + B1 + C1**.

---

## 2. Repository layout (only the parts that matter)

```
probe_spatial/
├── configs/
│   ├── train.yaml                           # top-level Hydra entry
│   ├── model/
│   │   ├── probe.yaml                       # legacy v3 probe (depth/camera/id)
│   │   └── probe_ext.yaml                   # NEW — A2/A3/B1/C1 probes
│   ├── data/dl3dv.yaml                      # legacy
│   ├── callbacks/v3_checkpoint.yaml         # save_last + every-5-epoch ckpt
│   └── experiment/
│       ├── inscene15k/                      # legacy v3 experiments (A1)
│       └── inscene15k_ext/                  # NEW — 4 probes × 3 VFMs = 12 yamls
│           ├── view_consistency_{wan|cogvideox|vjepa2}_v1.yaml
│           ├── ego_belief_{wan|cogvideox|vjepa2}_v1.yaml
│           ├── action_dynamics_{wan|cogvideox|vjepa2}_v1.yaml
│           └── abnormal_{wan|cogvideox|vjepa2}_v1.yaml
│
├── features/
│   ├── run_inscene15k.py                    # main feature extractor (3 modes!)
│   ├── wan/                                 # per-VFM forwarders
│   ├── cogvideox/
│   └── vjepa2/
│
├── vidfm3d/
│   ├── train.py                             # Hydra entry point (fit + test)
│   ├── eval_diag.py                         # NEW — per-sample dump for one run
│   ├── eval_diag_compare.py                 # NEW — aggregate runs into a CSV
│   ├── data/components/inscene15k_dataset.py    # the only dataset class we use
│   ├── models/
│   │   ├── video_probe_module.py            # legacy v3 LitModule
│   │   ├── probe_ext_module.py              # NEW — LitModule for A2/A3/B1/C1
│   │   └── components/
│   │       ├── probe_pixalign.py            # legacy backbone+DPT (A1)
│   │       ├── probe_view_consistency.py    # NEW — A2 head
│   │       ├── probe_abnormal.py            # NEW — A3 head
│   │       ├── probe_ego_belief.py          # NEW — B1 head
│   │       └── probe_action_dynamics.py     # NEW — C1 head
│   └── utils/
│       └── spatial_diag.py                  # NEW — geometry helpers
│           # compute_overlap_ratio (A2)
│           # compute_hidden_object_target (B1)
│           # encode_relative_pose (C1)  -- 6D rot + 3 trans = 9 dims
│
└── scripts/
    ├── run_diag_sweep.sh                    # NEW — train 4 probes for one VFM
    └── run_diag_eval_sweep.sh               # NEW — eval all runs, write CSV
```

---

## 3. Data

### 3.1 The dataset: `InsScene-15K`
- **Sources**: Infinigen (synthetic, perfect GT) + ScanNet++ v2 (real). Both
  provide RGB, depth, instance masks, intrinsics, and extrinsics per frame.
- **On disk**:
  ```
  ${INSCENE_DATA_ROOT}/
      processed_infinigen/scene_XXX/<hash>/frames/Image/camera_0/*.png
      processed_infinigen/scene_XXX/<hash>/{Depth,ObjectSegmentation,camview}/...
      processed_scannetpp_v2/<scene_id>/{images,depth,refined_ins_ids,metadata.npz}
  ```
- **Window sampling**: long videos are split into overlapping windows of
  length 200 (stride 100). One Lightning sample = one window.
- **Per-sample tensors** (after `_resize_to_target` → 288×512):

| key | shape | meaning |
|---|---|---|
| `image` | (S, 3, H, W) | sampled RGB |
| `identity_ids` | (S, H, W) long | per-pixel instance id (remapped to [0, N)) |
| `intrinsics` | (S, 3, 3) | rescaled to current resolution |
| `extrinsics` | (S, 3, 4) | **world-to-camera**, normalized so frame-0 = identity, scaled by points |
| `cmaps` | (S, 1, H, W) | confidence (1 = depth valid) |
| `dmaps` | (S, 1, H, W) | depth |
| `pmaps` | (S, 3, H, W) | world-coord pointmap (only when `include_pmaps=True`) |
| `vfm_feat` | (S, H_f, W_f, C) | **pixel-aligned** VFM features (see §4) |
| `vfm_idx` | (S,) | feature-time indices (kept for compatibility) |
| `scene_path` | str | for logging |

### 3.2 Frame sampling
- `num_views = 4` per sample (S = 4).
- Sampled inside the window via `_sample_query_frames` with
  `min_view_interval = 5` and `query_idx_divisor = 4` (snapping to a 4-frame
  grid that matches the VFM's temporal stride).
- Predictions are defined in the last view's camera coordinate system. This is
  a target/reference convention, not a pose or current-frame conditioning input.

---

## 4. Feature extraction (`features/run_inscene15k.py`)

**Why we cache features.** VFM forwards are slow (Wan ≈ minutes/clip). We
extract once, store on disk in fp16 .sft, then train probes off the cache.

Each scene is fed **one clip** (default 81 frames evenly spaced) into the VFM.
The VFM returns layer features of shape `(T_feat, H_f, W_f, C)`. We save them
as
```
{out_root}/{vfm_dir}/{source}/{scene}/feature{feat_postfix}.sft   key="feat"
```
For Wan: `(81, 18, 32, 1536)` after reshape; `feat_postfix = _t749_layer20`.

### 4.0 Which layer is currently probed?

The default layer choices are intentionally kept identical to the existing
experiments and are centralized in `vidfm3d/utils/feature_layers.py`.

| Feature backend | Default cache | Layer meaning |
|---|---|---|
| Wan2.1 | `feature_t749_layer20.sft` | diffusion transformer block 20 at timestep 749 |
| CogVideoX | `feature_t749_layer20.sft` | diffusion transformer block 20 at timestep 749 |
| V-JEPA2 ViT-L | `feature_layer23.sft` | last encoder block, 0-based layer 23 |
| Qwen2.5-VL / BAGEL | `feature_layer-1.sft` | current MLLM default visual-token / last-layer cache |

Layer selection is a filename convention, not a probe-head change.  The dataset
loads `feature{feat_postfix}.sft` by default, or can derive the filename from
`feature_layer` plus `feature_timestep`.  The probe heads see the same tensor
shape contract `(S, H_f, W_f, C)`.

To extract extra layers:

```bash
# Wan/CogVideoX: explicit transformer blocks at the same diffusion timestep.
python -m features.run_inscene15k --vfm wan \
  --data-root ${INSCENE_DATA_ROOT} \
  --out-root ${INSCENE_FEAT_ROOT} \
  --t 749 --output-layers 0 5 10 15 20 25 29

# V-JEPA2: all encoder blocks, or aliases such as default/last.
python -m features.run_inscene15k --vfm vjepa2 \
  --data-root ${INSCENE_DATA_ROOT} \
  --out-root ${INSCENE_FEAT_ROOT} \
  --all-layers

# Qwen2.5-VL: -1 is the current visual-merger cache; non-negative layers are
# vision-tower block outputs captured with forward hooks.
python -m features.run_inscene15k_mllm --backend qwen2_5_vl \
  --data-root ${INSCENE_DATA_ROOT} \
  --out-root ${INSCENE_MLLM_FEAT_ROOT} \
  --output-layers -1 0 8 16 24 31
```

To train a layer-wise sweep, reuse one experiment config and override only the
feature layer/run identity:

```bash
VFM=wan PROBE=ego_belief LAYERS="0 5 10 15 20 25 29" DEV=0 \
  bash scripts/run_layer_sweep.sh
```

After `eval_diag.py` has produced `eval/val_summary.json` for each run:

```bash
python scripts/summarize_layer_sweep.py \
  --vfm wan --probe ego_belief \
  --pattern "inscene15k_ext_ego_belief_wan_layer*" \
  --output layer_sweep_ego_belief_wan.csv
```

The summary reports the layer-wise CSV, `best_layer`, the registered default
layer score, and the last-layer score when a static last layer is known.

### 4.1 Three modes

```bash
python -m features.run_inscene15k --vfm wan --mode <MODE> ...
```

| `--mode` | Output dir suffix | Used by | Frame schedule |
|---|---|---|---|
| `normal` (default) | `wan/`        | A1, A2, B1 and other non-causal probes | the actual 81 frames in order |
| `shuffled`         | `wan_shuffled/` | A3 | frames are permuted (per-scene RNG, seed=42) before being passed to the VFM. We then **un-permute** the output, so `shuf[i]` = the feature of frame `i` *as seen under scrambled context*. This is the entire trick that makes A3 nontrivial — VFM features are clip-contextualized, so the only way to expose temporal-coherence info is to recompute them. |
| `context_segment`  | `wan_context/` | C1/C2/C3 inputs | Each cached input segment is forwarded as `[I_start, ..., I_tail]` without future target frames and saved under `context_<start>_<tail>`. Short segments are padded by repeating the tail frame. |
| `target_isolated`  | `wan_target/` | C1/C2/C3 targets | Each cached target frame is replicated to fill the VFM clip; we keep the center-frame feature only. Output: `(M, H_f, W_f, C)` plus `target_indices.npy = (M,)`. Cache every frame with `--num-targets 0` unless you intentionally want sparse valid samples. |

> Why we cannot simply re-use the `normal` features:
> 1. Diffusion-VFM features are **clip-contextualized** (cross-frame attention).
> 2. Permutation-invariant aggregation downstream (BackbonePA uses 2D RoPE only)
>    means a probe that *operates on already-extracted features* cannot
>    distinguish a temporally shuffled clip from a normal one — the answer is
>    *baked into the VFM forward*. Same logic for "predict next frame": a
>    feature of frame *t* extracted alongside frames *t+1…* is leaked.

### 4.2 Resume safety
The script checks per-layer output files before processing each scene; already
done scenes are skipped. CLI: `--start / --end` to slice the scene list; one
GPU per process.

### 4.3 Dataset hookup
- A2 / B1 just consume `vfm_feat` from `normal` mode (and `pmaps` for
  geometry).
- A3 sets `diag_abnormal=True` + `shuffled_feat_root=...` and the dataset
  also loads `vfm_feat_shuffled`.
- C1 sets `diag_action=True`, `context_feat_root=...` and
  `target_feat_root=...`. The dataset loads `input_feat` from the causal
  `context_segment` ending at the last input frame, and `target_feat` from the
  exact `target_isolated` row. The ordinary `vfm_feat` normal cache is not
  passed to the C1 probe.
- C2/C3 also use `context_segment` for `input_feat_seq` and `target_isolated`
  for `target_feat_seq`; horizons with missing context or isolated targets are
  masked invalid.

---

## 5. The four diagnostic probes

All heads live under `vidfm3d/models/components/`. They are **small** (a few
projection MLPs and a 2-layer Transformer encoder); the point is to quantify
*what is already in the VFM features*, not to engineer the head.

Common conventions:
- Input: `vfm_feat` of shape `(B, S, H_f, W_f, C)` (S = 4 views, fp32).
- All heads start with `LayerNorm + Linear(C → hidden)` to stabilize the input.

B1/B2 additionally support `decoder_type={linear,mlp,transformer}`. Linear is
an affine readout; MLP has one hidden GELU layer; Transformer is the existing
default. See `EXPERIMENT_PROTOCOL.md` for the exact summaries exposed to each.

### 5.1 A2 — `ViewConsistencyProbe` (predict pairwise overlap)

- **Signal**: for every ordered pair (i, j) of the 4 views, predict the
  fraction of frame i's valid pixels that re-project into frame j (using GT
  depth + intrinsics + extrinsics).
- **Head**: spatial mean-pool → MLP projection → pair-MLP on
  `[zi, zj, zi*zj, |zi-zj|]` → logit. Output `(B, S, S)`.
- **Loss**: BCE on hard positive (overlap ≥ 0.4) and hard negative
  (overlap ≤ 0.05) pairs, plus 0.5 × soft BCE over the full off-diagonal
  matrix. Hard pairs give clean signal; soft term gives dense gradient.
- **Metrics**: `overlap_mae` (off-diagonal), `overlap_acc` (on hard pairs),
  `overlap_pos_frac/neg_frac` (sanity for class balance).
- **Why no BackbonePA**: BackbonePA's 4 self-attn layers would reconcile
  cross-view geometry on its own, masking what the VFM itself encodes.

### 5.2 A3 — `AbnormalVideoProbe` (binary normal vs. shuffled-context)

- **Signal**: 0 if the clip's features came from `normal` mode, 1 if from
  `shuffled` mode (frames permuted before VFM forward).
- **Head**: spatial pool → linear → CLS token + learned temporal positional
  embedding → 2-layer Transformer encoder → 1-D logit on `[CLS]`.
- **Loss**: standard BCE.
- **Why this works**: positional embedding in the head is the only mechanism
  that lets the model exploit temporal order — but the *information about
  whether the order is plausible* must come from the VFM features. If the VFM
  smears identity across time, it cannot have stored "which frame came first".

### 5.3 B1 — `EgoBeliefProbe` (hidden-object localization)

- **Signal**: pick **one** object per sample that is visible in some past
  frame but no longer in the last frame. Regress its (azimuth, elevation,
  log-distance) in the **last frame's camera coordinates**.
- **Selection**: candidates must have ≥ 200 valid pixels in some past frame
  and < 200 valid pixels (or be absent) in the last frame. Tie-breaker score
  = `num_visible_frames * 1e6 + total_visible_pixels` — so we always pick the
  most informative target. If no candidate, sample is marked `valid=False`
  and skipped during training.
- **Object condition**: B1 receives the object's past masks plus a final-view
  global feature; B2 receives a masked-pooled appearance query. This condition
  is part of the main task.
- **No pose condition**: neither B1 nor B2 receives camera intrinsics,
  extrinsics, relative pose, or an explicit current-frame role token.
- **Head**: B1 masked-pools only the specified object's visible-frame features;
  invisible object frames are ignored. It also receives one global-pooled token
  from the final sampled frame, giving last-view visual context without camera
  pose. B2 instead attends from the object query to all ordered patch tokens and
  therefore measures a broader sequence-level retrieval capability.
  The last-frame reference is implicit in sequence order and the GT definition.
- **Loss**: weighted Smooth-L1 with weights `(1, 1, 0.5)` on (azimuth,
  elevation, log-dist).
- **Metrics**: per-axis errors in degrees for angles and natural-log error
  for distance.
- **Camera convention** (consumed by `compute_hidden_object_target`):
  +X right, +Y down, +Z forward (OpenCV). `azimuth = atan2(x, z)`,
  `elevation = asin(-y / d)`. **Always sanity-check by visualizing the GT
  vector overlaid on the last frame before trusting numbers.**

### 5.4 C1 — `ActionDynamicsProbe` (predict pooled future feature)

- **Signal**: given a causal input video segment feature, e.g. `[I_1..I_48]`
  forwarded together, and the relative camera pose from segment tail to target
  frame (encoded as 9-D action: 6-D rotation Zhou+2019 + 3-D translation),
  predict the spatially-pooled isolated feature of a future target frame such as
  `I_52`, `I_56`, or `I_64`. The input segment forward never contains the
  target frame.
- **Head**: pool input feats → linear → prepend `[query, action]` tokens →
  2-layer Transformer encoder → linear back to `C` dims, returned at
  `[query]`.
- **Loss**: MSE + (1 − cosine).
- **Training diagnostics**: cosine similarity plus explicitly named
  `inbatch_R@1`/mean-rank; these depend on batch size and are not final metrics.
- **Final metrics**: `eval_diag.py` retrieves against the complete evaluation
  set and reports global R@1/R@5/mean-rank plus no-action, shuffled-action and
  last-observation controls.
- **Action variant in use**: **C1a** — replicate target frame to fill clip
  during target extraction. (We considered C1b = noise-pad; rejected because
  C1a's target language is purely a function of the target frame.)

### 5.5 Where invalid samples are masked out

Each probe-step in `ProbeExtensionLitModule` filters by an explicit boolean
flag from the dataset:
- B1: `hidden_obj_valid`
- C1: `dyn_valid` (False when the context segment or isolated target row is missing)
- C2: `path_horizon_valid` (False for horizons with missing context or target rows)
- C3: `counterfactual_valid` (also applies the minimum-overlap filter)
- A3: `abnormal_feat_valid` (False when the shuffled feature file was missing)

If *all* samples in a batch are invalid we return `params.sum()*0.0` (a
graph-connected zero loss) so PyTorch Lightning's autograd does not fail.

---

## 6. The unified Lightning module — `ProbeExtensionLitModule`

One LightningModule for **all four** new probes. Selected at config time via
`probe_type ∈ {view_consistency, abnormal, ego_belief, action_dynamics}`.

```python
loss, metrics = model.model_step(batch, train=...)
# dispatches to _step_view_consistency / _step_abnormal /
#                _step_ego_belief / _step_action_dynamics
```

It also implements `training_step`, `validation_step`, `test_step` (=
validation_step but logs under `test/*`). Optimizer & scheduler are
instantiated from Hydra (`AdamW lr=3e-4, weight_decay=0.05` +
`LinearWarmupCosineAnnealingLR` with 5 warmup epochs).

This module **never touches** the legacy `VideoProbeLitModule`, so v3 training
(A1) keeps working exactly as before.

---

## 7. Config layout (Hydra)

`configs/train.yaml` composes:
```
- data: dl3dv  (overridden by experiment)
- model: probe (overridden by experiment)
- callbacks: default
- trainer: ddp
- experiment: null  (you select with experiment=...)
```

The new `configs/model/probe_ext.yaml`:
```yaml
_target_: vidfm3d.models.probe_ext_module.ProbeExtensionLitModule
probe_type: ???      # set by experiment
optimizer: AdamW lr=3e-4 wd=0.05
scheduler: LinearWarmupCosineAnnealingLR warmup=5 max=${trainer.max_epochs}
probe: ???           # set by experiment
view_overlap_pos_threshold: 0.4
view_overlap_neg_threshold: 0.05
polar_loss_weights: [1.0, 1.0, 0.5]
mse_loss_weight: 1.0
cosine_loss_weight: 1.0
```

Each experiment under `configs/experiment/inscene15k_ext/<probe>_<vfm>_v1.yaml`
overrides:
- `defaults: override /model: probe_ext`
- `defaults: override /callbacks: v3_checkpoint`
- `data.data_module.train/validation_datasets`: a Python expression
  `InsScene15KDataset(...)` with the right `vfm_name`, `feat_postfix`, and
  `diag_*` flag(s).
- `model.probe_type` and `model.probe._target_ + dims`.

| Backbone | Current default feature | Meaning | `in_channels` |
|---|---|---|---|
| Wan2.1-T2V-1.3B | `_t749_layer20` | diffusion transformer block 20 at timestep 749 | 1536 |
| CogVideoX-5b-I2V | `_t749_layer20` | diffusion transformer block 20 at timestep 749 | 3072 |
| V-JEPA2-vitl-fpc64-256 | `_layer23` | final ViT encoder block, 0-indexed | 1024 |
| Qwen2.5-VL-7B | `_layer-1` | historical default: final visual-merger tokens | 3584 |
| Qwen2.5-VL-3B | `_layer-1` | historical default: final visual-merger tokens | 2048 |
| BAGEL HF backend | `_layer-1` | final HF hidden state / visual-token grid when available | 3584 |

Layer metadata lives in `vidfm3d/utils/feature_layers.py`.  The old
`feat_postfix` path is still valid; new experiments can instead override
`feature_layer=<n>` (and, for diffusion VFMs, `feature_timestep=<t>`) to select
another cached layer without editing YAML.

---

## 8. Running things — the canonical sequence

### 8.1 One-time: extract features
For each VFM you want to compare, extract all three modes:
```bash
# A1/A2/B1 inputs + C1 input frames
python -m features.run_inscene15k --vfm wan --mode normal \
    --data-root ${INSCENE_DATA_ROOT} \
    --out-root  ${INSCENE_FEAT_ROOT}
# A3
python -m features.run_inscene15k --vfm wan --mode shuffled \
    --out-root ${INSCENE_SHUFFLED_FEAT_ROOT}
# C1/C2/C3 input contexts
python -m features.run_inscene15k --vfm wan --mode context_segment \
    --out-root ${INSCENE_CONTEXT_FEAT_ROOT} --context-len 76
# C1/C2/C3 targets
python -m features.run_inscene15k --vfm wan --mode target_isolated \
    --out-root ${INSCENE_TARGET_FEAT_ROOT} --num-targets 0
```
Switch `--vfm cogvideox` / `--vfm vjepa2` for the other VFMs.

### 8.1.1 Layer-wise feature sweeps

To probe where spatial information is strongest, extract multiple layers into
the same cache.  Existing default behavior is unchanged when `--output-layers`
is omitted.

```bash
# Wan/CogVideoX: produces feature_t749_layer{L}.sft
python -m features.run_inscene15k --vfm wan --mode normal \
  --data-root ${INSCENE_DATA_ROOT} \
  --out-root ${INSCENE_FEAT_ROOT} \
  --t 749 --output-layers 4 8 12 16 20 24 28

# V-JEPA2 / VLM-style filenames: produces feature_layer{L}.sft
python -m features.run_inscene15k --vfm vjepa2 --mode normal \
  --data-root ${INSCENE_DATA_ROOT} \
  --out-root ${INSCENE_FEAT_ROOT} \
  --output-layers 3 7 11 15 19 23
```

For A3, extract the same layer list for `--mode shuffled`. For C1/C2/C3,
extract the same layer list for both `--mode context_segment` and
`--mode target_isolated`, because inputs and targets intentionally come from
separate VFM forwards.

Training a probe on layer `L` is a Hydra override:

```bash
python vidfm3d/train.py experiment=inscene15k_ext/view_consistency_wan_v1 \
  feature_layer=12 feature_timestep=749 \
  model.probe.in_channels=1536 \
  job_name=view_consistency_wan_layer12
```

For V-JEPA2 use `feature_layer=12` and `model.probe.in_channels=1024`; for
Qwen2.5-VL/BAGEL use `features/run_inscene15k_mllm.py --output-layers ...` and
set the corresponding `model.probe.in_channels` for the cached activation.
Qwen's default `--output-layers -1` keeps the existing final visual-merger
feature; explicit non-`-1` layers are VLM hidden-state activations.

After evaluating each layer with `vidfm3d/eval_diag.py`, summarize best layer,
last-layer score, and the layer-wise curve:

```bash
python scripts/summarize_layer_sweep.py \
  --vfm wan --probe view_consistency \
  --pattern "inscene15k_ext_view_consistency_wan_layer*" \
  --output layer_sweep_view_consistency_wan.csv
```

Or run the whole pipeline with one script. It extracts the feature modes needed
by the selected probe, trains one probe per layer, evaluates all checkpoints,
and writes the summary CSV:

```bash
VFM=wan PROBE=view_consistency LAYERS="0 5 10 15 20 25 29" \
  EXTRA_TRAIN="logger.wandb.offline=true" \
  bash scripts/run_feature_layer_probe_sweep.sh
```

Mode selection is automatic: A2/B1/B2 use normal caches, A3 adds shuffled
caches, and C1/C2/C3 add target-isolated caches. For VLM/UMM experiments whose
config name is not `{probe}_{vfm}_v1`, pass `CFG` explicitly:

```bash
CFG=inscene15k_ext/sae_qwen2_5vl_v1 \
VFM=qwen2_5_vl PROBE=sae SUMMARY_PROBE=sae_spatial \
LAYERS="-1 8 16 24" \
bash scripts/run_feature_layer_probe_sweep.sh
```

### 8.2 Train a single probe
```bash
python vidfm3d/train.py experiment=inscene15k_ext/view_consistency_wan_v1
# resumes automatically from logs/<run>/checkpoints/last.ckpt if present
# (also restores the wandb run id when present)
```

### 8.3 Train all 4 probes for one VFM
```bash
bash scripts/run_diag_sweep.sh wan 0     # GPU=0
bash scripts/run_diag_sweep.sh cogvideox 0
bash scripts/run_diag_sweep.sh vjepa2 0
```

### 8.4 Per-sample evaluation + cross-VFM table
```bash
# 1) per-run dump (writes logs/<run>/eval/{val_predictions.pt, val_summary.json})
python vidfm3d/eval_diag.py \
    experiment=inscene15k_ext/view_consistency_wan_v1 \
    ckpt_path=logs/<run>/checkpoints/last.ckpt \
    eval_split=val train=false test=false

# 2) all runs in one shot + aggregate
bash scripts/run_diag_eval_sweep.sh         # writes comparison_val.csv
```

`comparison_val.csv` is the table you put in slides — one row per run, columns
= probe metrics.

### 8.5 Suggested validation order (from empty cache)
1. `normal` features (already exist for Wan).
2. Train A2 + B1 (no extra features needed).
3. Extract `target_isolated` features → train C1.
4. Extract `shuffled` features → train A3.

(Step 1 is the longest. Steps 3 and 4 each cost ≈ M× and 1× a normal extract.)

---

## 9. Key numerical / engineering decisions, in one place

| Decision | Choice | Reason |
|---|---|---|
| Frames per clip (Wan) | 81 | Wan's native temporal length |
| Clip resolution | 480 × 832 | Wan native; downsamples to 18 × 32 features |
| Probe input resolution | 288 × 512 | Cheap to load, integer ratio to features |
| Views per sample (S) | 4 | Enough for pair tasks, cheap |
| Action encoding | 6-D rot (Zhou+2019, first 2 rows of R_rel) + 3-D trans = 9 | Continuous, no singularity |
| B1 reference frame | camera coordinates of the last sampled view | target convention; final-view global feature is input, final camera pose is not |
| B1 obj selection | single highest-score hidden object per sample | concentrates supervision on a hard, well-posed signal |
| A2 thresholds | pos ≥ 0.4, neg ≤ 0.05 | empirically separates clean pos/neg |
| C1 retrieval | global evaluation set | stable candidate pool; training in-batch score is diagnostic only |
| Loss for B1 distance | weight 0.5 on log-distance | prevents far-object terms dominating |
| Why fp32 in dataset | cast on load (`.float()`) | mixes well with autocast, avoids dtype errors at concat |
| Checkpointing | every 5 epochs + last.ckpt forced | survives SIGKILL; resumable wandb |
| Autoresume | reads `${output_dir}/checkpoints/last.ckpt`; tries to recover wandb run id from `wandb/latest-run/*.wandb` | one-line resume on preempt |

---

## 10. What "good" looks like — interpreting the metrics

| Probe | Worst-case baseline (random VFM) | A "useful" VFM signal |
|---|---|---|
| A2 | overlap_acc ≈ chance (= positive fraction of pairs); MAE high | accuracy clearly above chance, MAE drops below ≈ 0.15 |
| A3 | accuracy ≈ 0.5 | gap > a few percent (any gap indicates temporal-coherence info) |
| B1 | azimuth/elevation errors near uniform-prior baseline (~50°+) | < 30° azimuth is a clearly nontrivial signal on InsScene-15K |
| C1 | cosine ≈ 0; R@1 ≈ 1/B | cosine > 0.3 and R@1 well above 1/B |

These are coarse anchors, not contracts; the comparison row-by-row across VFMs
is what we actually report.

---

## 11. Common pitfalls (read this before debugging)

1. **`feat_postfix` mismatch.** `run_inscene15k.py` and the dataset must agree
   on the file name. Wan/CogVideoX → `_t749_layer20`; V-JEPA2 → `_layer23`.
   The dataset uses the helper `_feat_filename()` to make this consistent.
2. **`include_pmaps`.** A2 and B1 require world-coord pointmaps. The dataset
   force-enables `include_pmaps=True` whenever `diag_overlap` or
   `diag_hidden_obj` is set, but if you read a custom dataset script,
   remember the original v3 default is False.
3. **Camera convention.** All extrinsics are normalized so frame 0 = identity
   (`invert_pose_ref_and_scale`) and points are scaled. B1 polar coordinates
   live in this normalized last-frame coords. Always visualize before
   trusting absolute distances.
4. **C1/C2/C3 contextual leakage.** Do **not** train these probes against
   features sliced from the normal full-clip cache. A bidirectional video VFM can
   mix target-frame information into both input and target hidden states. The
   current protocol uses `context_segment` for inputs and `target_isolated` for
   targets; do not reuse normal full-clip features for either tensor.
5. **A3 trivial leakage.** If the shuffled cache is missing for a sample, the
   dataset emits zeros for `vfm_feat_shuffled` and sets
   `abnormal_feat_valid=False`; the LitModule filters them out. Do not
   bypass the validity flag.
6. **autoresume + wandb.** If `wandb/latest-run/run-*.wandb` is absent, we
   warn and start a new wandb run; training itself still resumes correctly.
7. **OOM on large clips.** Feature extraction (`mode=target_isolated`) does M
   forwards per scene at the same input shape as a normal forward — the
   *peak* memory is unchanged, but wall-time is M× longer. Plan accordingly.

---

## 12. Provenance / what is new

This repo started as a fork of [VidFM3D](https://github.com/zxhuang1698/VidFM3D)
plus the upstream `probe_spatial` extensions (InsScene-15K, V-JEPA2 extractor,
windowed v3 training, ProbeModelPA). The diagnostic suite in §1, the
`probe_ext_module` LitModule in §6, the four heads in §5, the multi-mode
feature extractor in §4.1, the geometry helpers in §3.1 / `spatial_diag.py`,
and the eval pipeline in §8.4 are introduced by this work.

---

## 13. Quick file index

| Need to… | Read this file |
|---|---|
| understand a probe head | `vidfm3d/models/components/probe_*.py` |
| trace a metric back to its formula | `vidfm3d/models/probe_ext_module.py::_step_*` |
| understand the GT for A2 / B1 / C1 | `vidfm3d/utils/spatial_diag.py` |
| change which frames are sampled | `inscene15k_dataset.py::_sample_query_frames` |
| add a new VFM | `features/<vfm>/...` + add a branch in `features/run_inscene15k.py` + new entries to `VFM_DEFAULTS` and `_feat_filename` |
| add a new probe | new head class + new `_step_<x>` in `probe_ext_module.py` + register in `RECORDERS` of `eval_diag.py` + new experiment yaml |
| reproduce slides table | `bash scripts/run_diag_eval_sweep.sh` then open `comparison_val.csv` |

---

## 14. VLM / Unified-Model SAE Extension

This section records the new extension for probing spatial representations in
VLMs and unified multimodal models, beyond the previous video-generation and
video-SSL backbones.

### 14.1 Motivation

The original diagnostic suite asks how much 3D, temporal, and egocentric spatial
information is encoded in frozen video foundation model features such as Wan,
CogVideoX, and V-JEPA2. The new extension asks the analogous question for:

- **VLMs:** Qwen2.5-VL.
- **Unified multimodal models:** ByteDance BAGEL.

The core design remains unchanged: the backbone is frozen, activations are
pre-extracted, and only lightweight probes are trained. The new part is an
SAE-based diagnostic layer that can test whether the internal activations admit
a sparse dictionary whose features still preserve spatial information.

### 14.2 New feature extraction path

New file:

```text
features/run_inscene15k_mllm.py
```

It mirrors `features/run_inscene15k.py`, but targets multimodal language models
whose hidden states are token sequences rather than video diffusion grids.

Output format:

```text
{out_root}/{vfm_name}/{source}/{scene}/feature_layer{layer}.sft
```

Each safetensors file contains:

```text
feat: (S, H_t, W_t, C)
```

where `S` is the sampled frame count and `(H_t, W_t)` is the visual-token grid.
This makes Qwen2.5-VL / BAGEL activations compatible with
`InsScene15KDataset`, which already expects a frame-aligned `(T,H,W,C)` feature
cache.

Example Qwen2.5-VL extraction:

```bash
/data/baiqiao/miniconda3/envs/vidfm3d/bin/python -m features.run_inscene15k_mllm \
  --data-root ${INSCENE_DATA_ROOT} \
  --out-root ${INSCENE_MLLM_FEAT_ROOT} \
  --backend qwen2_5_vl \
  --output-layers -1 \
  --num-frames 8
```

Example BAGEL extraction:

```bash
/data/baiqiao/miniconda3/envs/vidfm3d/bin/python -m features.run_inscene15k_mllm \
  --data-root ${INSCENE_DATA_ROOT} \
  --out-root ${INSCENE_MLLM_FEAT_ROOT} \
  --backend bagel_hf \
  --vfm-name bagel \
  --output-layers -1 \
  --num-frames 8
```

Implementation note: Qwen2.5-VL uses the standard Transformers path. BAGEL is
currently wired through a generic Hugging Face `trust_remote_code` backend,
because the public model card points users to the upstream BAGEL repository
rather than exposing a stable Transformers usage snippet. If a local BAGEL repo
is used later, replace this backend with a native BAGEL adapter while keeping the
same `.sft` output contract.

### 14.3 Dataset support

`vidfm3d/data/components/inscene15k_dataset.py` now treats any 4D feature cache
as a frame-aligned token/grid feature:

```text
feat: (T, H, W, C)
```

This covers Qwen2.5-VL and BAGEL in the same way as Wan / V-JEPA2 after the
features are cached. CogVideoX remains special-cased because its cache has an
extra leading dimension that is flattened before frame alignment.

New default channel hints were added for debug / missing-feature fallbacks:

```text
qwen2_5_vl: 3584
bagel:      3584
```

### 14.4 SAE spatial probe

New file:

```text
vidfm3d/models/components/probe_sae_spatial.py
```

The probe is `TopKSAESpatialProbe`. It trains a Top-k sparse autoencoder on
frozen activations:

```text
x -> sparse code -> reconstructed x
```

It also exposes lightweight readouts from frame-level sparse codes:

- overlap readout: predicts A2-style view overlap from sparse frame codes.
- ego readout: predicts B2-style hidden-object polar target from sparse codes.

By default, `detach_readout=True`, so spatial labels do not shape the SAE
dictionary itself. The readouts evaluate how much spatial information is already
available in the learned sparse codes.

Main metrics logged by `probe_type=sae_spatial`:

```text
sae_recon_mse
sae_rel_mse
sae_l1
sae_l0
sae_active_frac
sae_overlap_bce
sae_overlap_mae
sae_ego_loss
sae_ego_az_err_deg
sae_ego_el_err_deg
sae_ego_logd_err
```

### 14.5 Training configs

New experiment configs:

```text
configs/experiment/inscene15k_ext/sae_qwen2_5vl_v1.yaml
configs/experiment/inscene15k_ext/sae_bagel_v1.yaml
```

Example training:

```bash
/data/baiqiao/miniconda3/envs/vidfm3d/bin/python vidfm3d/train.py \
  experiment=inscene15k_ext/sae_qwen2_5vl_v1
```

```bash
/data/baiqiao/miniconda3/envs/vidfm3d/bin/python vidfm3d/train.py \
  experiment=inscene15k_ext/sae_bagel_v1
```

### 14.6 How to interpret this extension

The SAE extension is not a replacement for A2/A3/B1/B2/C1/C2/C3. It is a
complementary representation diagnostic:

- Direct probes answer: can a lightweight head recover the spatial label from
  frozen features?
- SAE probes answer: can the activation space be decomposed into sparse features
  while retaining spatial information in the sparse code?

Useful comparisons:

1. Qwen2.5-VL SAE vs BAGEL SAE: VLM understanding model vs unified
   understanding-generation model.
2. SAE sparse-code readouts vs direct B2/A2 probes: whether sparse
   decomposition preserves or discards spatial signal.
3. Real features vs scrambled/random controls: whether the spatial readout is
   supported by model activations rather than dataset bias.

### 14.7 Validation already performed

The following checks were run after implementation:

```bash
/data/baiqiao/miniconda3/envs/vidfm3d/bin/python -m py_compile \
  features/run_inscene15k_mllm.py \
  vidfm3d/models/components/probe_sae_spatial.py \
  vidfm3d/models/probe_ext_module.py \
  vidfm3d/data/components/inscene15k_dataset.py
```

Additional smoke checks:

- `TopKSAESpatialProbe` forward / backward on small synthetic tensors.
- `ProbeExtensionLitModule.model_step(probe_type="sae_spatial")` with synthetic
  `vfm_feat`, `overlap_gt`, `hidden_obj_polar`, and `belief_query_feat`.
- Hydra config composition for `sae_qwen2_5vl_v1` and `sae_bagel_v1` via
  `--cfg job`.
