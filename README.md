# probe_spatial: Probing Spatial Understanding in Video Foundation Models

This repository extends [VidFM3D](https://github.com/zxhuang1698/VidFM3D) with a **Spatial Diagnostic Suite** — diagnostic probes that characterize what kind of 3D / temporal / ego-centric understanding a Video Foundation Model (VFM) has internalized, all trained on frozen VFM features without touching VFM weights.

| Family | Probe | Question |
|--------|-------|---------|
| **A. Global Spatial Perception** | A1 depth/camera/identity (VidFM3D baseline) | Does the VFM perceive a coherent 3D scene? |
| **A. Global Spatial Perception** | **A2 view consistency** | Can it tell whether two clips share a viewing region? |
| **A. Global Spatial Perception** | **A3 abnormal detection** | Can it detect temporally shuffled frames? Legacy non-streaming control; not in the default streaming sweep yet. |
| **B. Ego-Centric Belief** | **B1 hidden-object localization** | Does it remember where objects went off-screen? |
| **B. Ego-Centric Belief** | **B2 object-query localization** | Can an object appearance query recover the object's current location? |
| **C. Action-Conditioned Prediction** | **C1 latent dynamics** | Can it predict a future isolated feature given a camera motion? |
| **C. Action-Conditioned Prediction** | **C2 path integration** | Can it roll forward through a sequence of relative camera actions? |
| **C. Action-Conditioned Prediction** | **C3 counterfactual action** | Does changing the action change the predicted future consistently? |

Based on: [VidFM3D: How Much 3D Do Video Foundation Models Encode?](https://arxiv.org/pdf/2512.19949v1)

![Teaser](teaser.png)

---

## Quick-start: reproduce results on a new server

```bash
# 1. Clone repo
git clone https://github.com/yyyybq/probe_spatial.git
cd probe_spatial

# 2. Create env (see Installation section below)
conda create -n vidfm3d python=3.11 cmake=3.14.0 -y
conda activate vidfm3d
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 nvidia/label/cuda-12.4.0::cuda-toolkit -c pytorch -c nvidia
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation
pip install -r requirements.txt
pip install -e .

# 3. Keep code and data separate
export PROJECT_ROOT=$PWD
export DATASET_ROOT=/data/probe_spatial_data/InsScene-15K
mkdir -p ${DATASET_ROOT}

# 4. Download InsScene-15K data outside the git checkout
python data/download_inscene15k.py --step download_scannetpp --base-dir ${DATASET_ROOT}
python data/download_inscene15k.py --step extract_scannetpp  --base-dir ${DATASET_ROOT}

# 5. Extract streaming VFM features outside the git checkout.
# Streaming temporal probes now use ScanNet++ only.  Infinigen frames are
# independent rendered views, not a video trajectory.
INSCENE_DATA=${DATASET_ROOT}
python -m features.run_inscene15k --vfm wan --mode streaming_prefix \
    --model-id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --data-root ${INSCENE_DATA}/data --out-root ${INSCENE_DATA}/FEAT_STREAMING \
    --source scannetpp --t 749 --output-layers 20 \
    --prefix-lengths "8,12,16,24" --prefix-max-len 24

# Layer sweeps: pass multiple layers, then train with feature_layer=<L>.
# Current default sweep models:
#   wan cogvideox vjepa2 dino aether f3r qwen2_5_vl_3b bagel
# Layer defaults are registered in vidfm3d/utils/feature_layers.py.
export INSCENE_DATA_ROOT=${INSCENE_DATA}/data
export INSCENE_STREAMING_FEAT_ROOT=${INSCENE_DATA}/FEAT_STREAMING
export INSCENE_TARGET_FEAT_ROOT=${INSCENE_DATA}/FEAT_TARGET

# 6. Train streaming diagnostic probes for the default model matrix on GPU 0
DEV=0 \
PROBES="streaming_depth view_consistency ego_belief ego_belief_v2 action_dynamics path_integration counterfactual" \
PREFIX_LENGTHS="8 12 16 24" LAYERS="default" \
    bash scripts/run_streaming_probe_sweep.sh

# 7. Test & evaluate
python scripts/summarize_streaming_prefix_sweep.py \
    --runs-root logs/inscene15k_streaming/runs \
    --output streaming_prefix_sweep.csv
```

> **Note**: streaming is the default experimental setting. Legacy normal/full-clip
> scripts now require `ALLOW_NON_STREAMING=1` and should only be used for old
> ablations or historical comparison.
>
> **Temporal validity note**: old streaming caches/runs that include Infinigen
> are not valid evidence for temporal, memory, action, path-integration, or
> counterfactual conclusions.  Infinigen's 100 frames are independent views.
> New temporal experiments default to ScanNet++ only. See
> [TEMPORAL_DATA_VALIDITY.md](TEMPORAL_DATA_VALIDITY.md).
>
> Current configs resolve dataset/cache roots from environment
> variables first. Set the relevant roots before launching jobs, or pass Hydra
> overrides explicitly:
> ```bash
> python vidfm3d/train.py experiment=inscene15k_streaming/action_dynamics_wan_v1 \
>     streaming_feat_root=/data/InsScene-15K/FEAT_STREAMING
> ```

On a shared cluster, the dataset also honors `INSCENE_DATA_ROOT`,
`INSCENE_FEAT_ROOT`, `INSCENE_CONTEXT_FEAT_ROOT`, `INSCENE_TARGET_FEAT_ROOT`,
`INSCENE_STREAMING_FEAT_ROOT`, and `INSCENE_SHUFFLED_FEAT_ROOT`; these take
precedence over YAML paths.
See `EXPERIMENT_PROTOCOL.md` and `TRAINING_LOGIC_AUDIT.md` for the current
causal-cache protocol and task-by-task input contracts.
> See `PROBE_SPATIAL_GUIDE.md` §8.1.1 for layer-wise extraction, probing, and
> `scripts/summarize_layer_sweep.py` best-layer/last-layer reports.
> Legacy non-streaming layer sweeps require explicit opt-in:
> ```bash
> ALLOW_NON_STREAMING=1 VFM=wan PROBE=view_consistency LAYERS="0 5 10 15 20 25 29" \
>   bash scripts/run_feature_layer_probe_sweep.sh
> ```

---

## New Server Layout: Code/Data Separation

Use the git checkout only for source code, configs, scripts, docs, and small
CSV summaries. Put raw data, extracted features, packaged archives, and large
intermediate files under a separate storage root:

```bash
export PROJECT_ROOT=/workspace/probe_spatial
export DATASET_ROOT=/data/probe_spatial_data/InsScene-15K

git clone git@github.com:yyyybq/probe_spatial.git ${PROJECT_ROOT}
cd ${PROJECT_ROOT}
mkdir -p ${DATASET_ROOT}

export INSCENE_DATA_ROOT=${DATASET_ROOT}/data
export INSCENE_TARGET_FEAT_ROOT=${DATASET_ROOT}/FEAT_TARGET
export INSCENE_STREAMING_FEAT_ROOT=${DATASET_ROOT}/FEAT_STREAMING
```

Do not place `InsScene-15K/`, `FEAT*/`, `.sft`, checkpoint, or archive files
inside the repository. The `.gitignore` has guards for common mistakes, but the
clean layout above is the main protection.

To update code without touching data:

```bash
cd ${PROJECT_ROOT}
git pull --ff-only origin main
source /path/to/conda.sh
conda activate vidfm3d
```

To extract all feature caches first and then run many probing experiments:

```bash
cd ${PROJECT_ROOT}
export DATASET_ROOT=/data/probe_spatial_data/InsScene-15K
export INSCENE_DATA_ROOT=${DATASET_ROOT}/data
export INSCENE_STREAMING_FEAT_ROOT=${DATASET_ROOT}/FEAT_STREAMING
export INSCENE_TARGET_FEAT_ROOT=${DATASET_ROOT}/FEAT_TARGET

python data/download_inscene15k.py --step all --base-dir ${DATASET_ROOT}

VFM=wan
MODEL_ID=Wan-AI/Wan2.1-T2V-1.3B-Diffusers
python -m features.run_inscene15k --vfm ${VFM} --model-id ${MODEL_ID} \
    --mode streaming_prefix --data-root ${INSCENE_DATA_ROOT} \
    --out-root ${INSCENE_STREAMING_FEAT_ROOT} --t 749 --output-layers 20 \
    --source scannetpp --prefix-lengths "8,12,16,24" --prefix-max-len 24
python -m features.run_inscene15k --vfm ${VFM} --model-id ${MODEL_ID} \
    --mode target_isolated --data-root ${INSCENE_DATA_ROOT} \
    --out-root ${INSCENE_TARGET_FEAT_ROOT} --t 749 --output-layers 20 \
    --source scannetpp --prefix-lengths "8,12,16,24" --prefix-max-len 24 \
    --target-from-streaming-windows --target-prefix-lengths "8,12,16,24" \
    --target-horizons "1,2,4"
```

After extraction, train from the fixed feature dataset:

```bash
cd ${PROJECT_ROOT}
PROBES="streaming_depth view_consistency ego_belief ego_belief_v2 action_dynamics path_integration counterfactual" \
PREFIX_LENGTHS="8 12 16 24" LAYERS="default" \
EXTRACT_STREAMING=0 EXTRACT_TARGETS=0 \
    bash scripts/run_streaming_probe_sweep.sh
```

Package extracted features as a Hugging Face dataset bundle:

```bash
cd ${PROJECT_ROOT}
DATASET_ROOT=/data/probe_spatial_data/InsScene-15K VFM=wan \
    bash scripts/package_inscene_feature_dataset.sh

export HF_REPO_ID=your-org/inscene15k-wan-probe-spatial-features
python -m pip install -U huggingface_hub
huggingface-cli login
huggingface-cli repo create ${HF_REPO_ID} --type dataset --private
huggingface-cli upload --repo-type dataset ${HF_REPO_ID} \
    /data/probe_spatial_data/InsScene-15K/feature_dataset_bundle/<bundle-name> .
```

When downloading that feature dataset on another machine, extract the archives
back under `DATASET_ROOT`, then `source env.sh` from the bundle or export the
same `INSCENE_*_ROOT` variables manually.

## Installation

```bash
conda create -n vidfm3d python=3.11 cmake=3.14.0 -y
conda activate vidfm3d

# PyTorch (CUDA 12.4)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 \
    nvidia/label/cuda-12.4.0::cuda-toolkit -c pytorch -c nvidia

# PyTorch3D (takes a while to compile; set MAX_JOBS=4 on low-RAM machines)
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation

pip install -r requirements.txt
pip install -e .
```

<details>
<summary>Troubleshooting</summary>

**CUDA Runtime Error** — `fatal error: cuda_runtime.h: No such file or directory`
```bash
export CUDA_HOME=/usr/local/cuda-12.4
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

**PyTorch iJIT_NotifyEvent** — Intel MKL incompatibility:
```bash
conda install "mkl<2024.1" "intel-openmp<2024.1" -c conda-forge -y
```

**GCC too new for CUDA** — Set `CC`/`CXX` to an older GCC or install one into the conda env before building PyTorch3D.

</details>

---

## Dataset Preparation

### InsScene-15K (primary dataset)

InsScene-15K ([HuggingFace: lifuguan/InsScene-15K](https://huggingface.co/datasets/lifuguan/InsScene-15K)) combines **Infinigen** synthetic scenes and **ScanNet++** real-world scenes.

```bash
# Current temporal experiments are ScanNet++ only.
python data/download_inscene15k.py --step download_scannetpp --base-dir /data/InsScene-15K
python data/download_inscene15k.py --step extract_scannetpp  --base-dir /data/InsScene-15K

# Infinigen download/extraction is legacy/debug only and is not used for
# temporal streaming conclusions.
```

Expected structure after extraction:
```
/data/InsScene-15K/
  data/
    processed_scannetpp_v2/   # ScanNet++ real-world scenes
  FEAT_STREAMING/             # default streaming_prefix features
  FEAT_TARGET/                # filled by feature extraction --mode target_isolated (C1/C2/C3 targets)
  FEAT/                       # legacy normal features, only with ALLOW_NON_STREAMING=1
  FEAT_SHUFFLED/              # legacy A3 shuffled features
  FEAT_CONTEXT/               # legacy non-streaming C context_segment features
```

### CO3Dv2 / DL3DV (original VidFM3D datasets)

Follow the [original VidFM3D instructions](https://github.com/zxhuang1698/VidFM3D#dataset-preparation) for CO3Dv2 and DL3DV.

---

## Feature Extraction

Features are pre-extracted once and cached as `.sft` files. The extractor is resume-safe (skips already-done scenes).

### InsScene-15K — extraction modes

| Mode | Output dir | Used by |
|------|-----------|---------|
| `streaming_prefix` | `FEAT_STREAMING/` | Default A1/A2/B1/B2/C1/C2/C3 inputs at selected prefix lengths |
| `target_isolated` | `FEAT_TARGET/` | C1/C2/C3 isolated target features, with exact frame ids |
| `normal` | `FEAT/` | Legacy non-streaming probes only; not the default setting |
| `shuffled` | `FEAT_SHUFFLED/` | Legacy A3 non-streaming branch |
| `context_segment` | `FEAT_CONTEXT/` | Legacy non-streaming C1/C2/C3 causal inputs |

```bash
INSCENE_DATA=/data/InsScene-15K
VFM=wan
MODEL_ID=Wan-AI/Wan2.1-T2V-1.3B-Diffusers

# Streaming-prefix features (default; shared by A1/A2/B1/B2/C1/C2/C3)
python -m features.run_inscene15k --vfm ${VFM} --model-id ${MODEL_ID} \
    --mode streaming_prefix \
    --data-root ${INSCENE_DATA}/data --out-root ${INSCENE_DATA}/FEAT_STREAMING \
    --source scannetpp \
    --t 749 --output-layers 20 --prefix-lengths "8,12,16,24" \
    --prefix-max-len 24

# Isolated target features for C1/C2/C3. These are selected from the same
# ScanNet++ streaming windows, but each target frame is forwarded alone.
python -m features.run_inscene15k --vfm ${VFM} --model-id ${MODEL_ID} \
    --mode target_isolated \
    --data-root ${INSCENE_DATA}/data --out-root ${INSCENE_DATA}/FEAT_TARGET \
    --source scannetpp \
    --t 749 --output-layers 20 --prefix-lengths "8,12,16,24" \
    --prefix-max-len 24 --target-from-streaming-windows \
    --target-prefix-lengths "8,12,16,24" --target-horizons "1,2,4"
```

Streaming prefixes are built from ScanNet++ motion-normalized temporal windows.
The default four cached prefixes are common-history `8`, then `8+4`, `8+8`,
and `8+16` observations.  Long ScanNet++ videos are cut into a capped number of
windows per scene so one recording does not dominate the dataset. The default
observation step is `--streaming-motion-step 0.35` with
`--streaming-rotation-weight 0.5`, which gives visible camera motion without
jumping as far as the earlier `0.5` step.

C1/C2/C3 use the same shared streaming-prefix video cache for their input, but
their prediction targets default to a separate `target_isolated` cache.  This
prevents the target feature from being produced by a VFM forward pass that has
already seen the input history plus the target frame.  The target frame ids are
computed from the same ScanNet++ motion-normalized window, prefix length, and
future horizons used by the C dataset samples.  Streaming prefix length is
orthogonal to probe type: C probes run for each requested prefix
`8/12/16/24`, not only for prefix `8`.

Streaming C-series data flow:

```text
sampled observations: I_0 ... I_7 | I_8 | I_9 | I_11
                      \_________/   target target target
                      common history

input feature:        VFM([I_0, ..., I_7])
                      -> FEAT_STREAMING/.../prefix_000007/...sft

action condition:     relative camera motion from I_7 to each target

target feature:       VFM([I_8]) / VFM([I_9]) / VFM([I_11])
                      -> FEAT_TARGET/.../feature_*.sft + target_indices.npy

probe training:       probe(input_feature, action) predicts target_feature
```

For C, the default target cache covers the union of
`prefix_len + horizon` positions for every C prefix and horizon. With
`prefix_len in {8,12,16,24}` and `horizon in {1,2,4}`, isolated targets are
drawn from observation positions:

```text
prefix 8  -> obs 8, 9, 11
prefix 12 -> obs 12, 13, 15
prefix 16 -> obs 16, 17, 19
prefix 24 -> obs 24, 25, 27
```

Streaming B1/B2 use common-history hidden-object selection.  The object is
selected once per ScanNet++ temporal window from frames `[0..7]`; it must be
visible in at least three history frames, have at least 1024 valid pixels in its
best query frame, and avoid the image border by at least 16 pixels. Selection
prefers objects visible at the common-history tail `obs7`, but falls back to a
history-visible object if no such candidate exists. Hidden B1/B2 train/evaluate
on prefixes `8/12/16/24`: prefix `8` is the visible-current baseline when the
chosen object is visible at `obs7`, while prefixes `12/16/24` require the same
object to be hidden at observation tails `11/15/23`. The same raw object id and
query history are reused across the prefix sweep.

There is also a B2 visible-object sanity probe, `visible_ego_belief_v2`. It
uses the same B2 head and metrics, but selects an object visible in the current
prefix-tail frame and expresses its position in that same tail camera frame.
The object query may therefore be pooled from the current visible view. This is
intended as a lower-bound check: if visible-object localization fails, the
hidden-object B2 result should not be interpreted as purely a memory problem.

Switch `--vfm cogvideox --model-id THUDM/CogVideoX-5b-I2V` or
`--vfm vjepa2` (uses `features/vjepa2/vjepa2_feature.py`) for other VFMs.

| VFM | `--vfm` | `--model-id` | `feat_postfix` | `in_channels` |
|-----|---------|-------------|---------------|--------------|
| Wan2.1-T2V-1.3B | `wan` | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | `_t749_layer20` | 1536 |
| CogVideoX-5B | `cogvideox` | `THUDM/CogVideoX-5b-I2V` | `_t749_layer20` | 3072 |
| V-JEPA2-ViT-L | `vjepa2` | see `features/vjepa2/` | `_layer23` | 1024 |
| DINOv2-Large | `dino` | `facebook/dinov2-large` | `feature.sft` | 1024 |
| Aether | `aether` | local Aether adapter | `_t749_layer1` | 3072 |
| Fast3R ViT-L | `f3r` | `jedyang97/Fast3R_ViT_Large_512` | `_l24` | 1024 |
| Qwen2.5-VL-3B | `qwen2_5_vl_3b` | `Qwen/Qwen2.5-VL-3B-Instruct` | `_layer-1` | 2048 |
| BAGEL | `bagel` | `ByteDance-Seed/BAGEL-7B-MoT` | `_layer-1` | 3584 |

### Feature layer convention

The current diagnostic defaults are now centralized in
`vidfm3d/utils/feature_layers.py` and remain backward compatible:

| Model family | Current default layer | Meaning |
|-----|-----:|-----|
| Wan2.1 | 20 | diffusion transformer block 20 at timestep `t=749` |
| CogVideoX | 20 | diffusion transformer block 20 at timestep `t=749` |
| V-JEPA2 ViT-L | 23 | last encoder block, 0-based |
| DINOv2-Large | 0 | last hidden state patch tokens |
| Aether | 1 | CogVideoX-backbone block at timestep `t=749` |
| Fast3R | 24 | last registered Fast3R ViT-L block |
| Qwen2.5-VL-3B / BAGEL caches | -1 | current default visual-token / last-layer cache |

To probe other layers, extract them by passing explicit layer ids, or use the
aliases `default`, `last`, and `all`:

```bash
# Multiple Wan layers in one pass; writes feature_t749_layer{L}.sft files.
python -m features.run_inscene15k --vfm wan \
    --data-root ${INSCENE_DATA}/data --out-root ${INSCENE_DATA}/FEAT \
    --t 749 --output-layers 0 5 10 15 20 25 29

# V-JEPA2 all registered layers; writes feature_layer{L}.sft files.
python -m features.run_inscene15k --vfm vjepa2 \
    --data-root ${INSCENE_DATA}/data --out-root ${INSCENE_DATA}/FEAT \
    --all-layers

# Qwen2.5-VL: -1 is the default visual-merger cache; non-negative ids are
# vision-tower blocks captured by hooks.
python -m features.run_inscene15k_mllm --backend qwen2_5_vl \
    --data-root ${INSCENE_DATA}/data --out-root ${INSCENE_DATA}/FEAT_MLLM \
    --output-layers -1 0 8 16 24 31
```

Layer-wise streaming probe training uses `run_streaming_probe_sweep.sh` and
sweeps both `prefix_len` and `feature_layer`:

```bash
DRY_RUN=1 VFM=wan PROBES="ego_belief action_dynamics" \
PREFIX_LENGTHS="8 12 16 24" LAYERS="0 5 10 15 20 25 29" \
    bash scripts/run_streaming_probe_sweep.sh
```

The old `run_feature_layer_probe_sweep.sh` is now guarded and requires
`ALLOW_NON_STREAMING=1`.

Direct VLM probing is parallel to the SAE path. By default it uses the same
streaming setting as other models and feeds each selected VLM layer directly
into the ordinary diagnostic probe heads:

```bash
INSCENE_STREAMING_FEAT_ROOT=/data/InsScene-15K/FEAT_STREAMING_MLLM \
INSCENE_TARGET_FEAT_ROOT=/data/InsScene-15K/FEAT_TARGET_MLLM \
VFMS="qwen2_5_vl qwen2_5_vl_3b bagel" \
PROBES="streaming_depth view_consistency ego_belief ego_belief_v2 action_dynamics path_integration counterfactual" \
PREFIX_LENGTHS="8 12 16 24" \
LAYERS="-1 8 16 24 31" \
    bash scripts/run_direct_vlm_probe_sweep.sh
```

The main streaming sweep already includes `qwen2_5_vl_3b` and `bagel` by
default. Set `VFMS=...` to run a smaller subset, or `VFM=wan` for the old
single-model behavior.

For Qwen, layer `-1` is the visual-merger output; non-negative layers are
vision-tower block outputs. The sweep script reads an existing `.sft` cache to
infer `video_channels` per layer, so direct probes can handle layers whose
channel dimension differs from the `-1` cache. Legacy non-streaming direct VLM
requires `STREAMING=0 ALLOW_NON_STREAMING=1`.

---

## Training

### Diagnostic probes (A1 / A2 / B1 / B2 / C1 / C2 / C3) — default streaming

**Train the default shared streaming-prefix sweep:**
```bash
INSCENE_STREAMING_FEAT_ROOT=/data/InsScene-15K/FEAT_STREAMING \
INSCENE_TARGET_FEAT_ROOT=/data/InsScene-15K/FEAT_TARGET \
PROBES="streaming_depth view_consistency ego_belief ego_belief_v2 action_dynamics path_integration counterfactual" \
PREFIX_LENGTHS="8 12 16 24" LAYERS="default" \
    bash scripts/run_streaming_probe_sweep.sh
```

`scripts/run_streaming_probe_sweep.sh` first extracts one shared
`streaming_prefix` cache for the requested prefix lengths, then trains one run
per `(probe, prefix_len, layer)`. C probes additionally require
`target_isolated` caches. By default the script extracts only the exact
streaming C target frame ids, using `--target-from-streaming-windows`,
`--target-prefix-lengths "${C_PREFIX_LENGTHS}"`, and
`--target-horizons "${C_TARGET_HORIZONS}"`.

The same streaming interface works for direct VLM features:

```bash
INSCENE_STREAMING_FEAT_ROOT=/data/InsScene-15K/FEAT_STREAMING_MLLM \
INSCENE_TARGET_FEAT_ROOT=/data/InsScene-15K/FEAT_TARGET_MLLM \
VFM=qwen2_5_vl \
PROBES="streaming_depth view_consistency ego_belief ego_belief_v2 action_dynamics path_integration counterfactual" \
PREFIX_LENGTHS="8 12 16 24" LAYERS="-1 8 16 24 31" \
    bash scripts/run_streaming_probe_sweep.sh
```

For VLM streaming, each `prefix_len` is forwarded as the real image prefix
`[I_0, ..., I_t]`; target frames for C probes are forwarded separately as
single-frame target-isolated caches.

Legacy non-streaming sweep scripts (`run_diag_sweep.sh`,
`run_diag_new_sweep.sh`, `run_diag_new_parallel.sh`,
`run_feature_layer_probe_sweep.sh`) now refuse to run unless
`ALLOW_NON_STREAMING=1` is set.

**Train a single probe:**
```bash
# One streaming probe, one or more prefix lengths.
DEV=0 VFM=wan PROBES="ego_belief" PREFIX_LENGTHS="8 12 16 24" \
LAYERS="default" bash scripts/run_streaming_probe_sweep.sh

# B2 sanity: object is visible in the current prefix-tail view.
DEV=0 VFM=wan PROBES="visible_ego_belief_v2" PREFIX_LENGTHS="8 12 16 24" \
LAYERS="default" bash scripts/run_streaming_probe_sweep.sh

# C probes also need target-isolated caches; the script extracts them by default.
DEV=0 VFM=wan PROBES="action_dynamics" PREFIX_LENGTHS="8 12 16 24" \
LAYERS="default" bash scripts/run_streaming_probe_sweep.sh
```

Training runs for 50 epochs (≈2–4 hours per probe on 1× L40S), auto-resumes from `last.ckpt`.

**Legacy control experiments** (non-streaming scrambled features as baseline):
```bash
ALLOW_NON_STREAMING=1 CUDA_VISIBLE_DEVICES=0 python vidfm3d/train.py \
    experiment=inscene15k_ext/action_dynamics_wan_ctrl
```

### A1 Baseline Probe (depth / camera / identity) — original VidFM3D legacy

```bash
# Wan2.1
ALLOW_NON_STREAMING=1 CUDA_VISIBLE_DEVICES=0 python vidfm3d/train.py experiment=inscene15k/wan_probe_v3

# V-JEPA2
ALLOW_NON_STREAMING=1 CUDA_VISIBLE_DEVICES=1 python vidfm3d/train.py experiment=inscene15k/vjepa2_probe_v3
```

Trains for 100 epochs, checkpointed every 5 epochs.

### Override data paths

If your data is not at the default path, set the environment roots before
launching Hydra:
```bash
INSCENE_DATA_ROOT=/your/InsScene-15K/data \
INSCENE_STREAMING_FEAT_ROOT=/your/InsScene-15K/FEAT_STREAMING \
INSCENE_TARGET_FEAT_ROOT=/your/InsScene-15K/FEAT_TARGET \
bash scripts/run_streaming_probe_sweep.sh
```

---

## Evaluation

### Diagnostic Probes (A2 / A3 / B1 / B2 / C1 / C2 / C3)

```bash
# Evaluate all trained runs and aggregate into a CSV
bash scripts/run_diag_eval_sweep.sh     # writes comparison_val.csv

# Evaluate a single run
python vidfm3d/eval_diag.py \
    experiment=inscene15k_ext/view_consistency_wan_v1 \
    ckpt_path=logs/<run>/checkpoints/last.ckpt \
    eval_split=val train=false test=false
```

**Test mode** (final numbers; requires a frozen three-way split manifest):
```bash
export INSCENE_SPLIT_MANIFEST=$PWD/configs/splits/inscene15k_v1.json
python vidfm3d/eval_diag.py experiment=inscene15k_ext/action_dynamics_wan_v1 \
    ckpt_path=/path/to/last.ckpt +eval_split=test train=false test=false
```

`eval_diag.py` computes C1 retrieval globally, so final R@K no longer depends
on evaluation batch size. It also reports no-action and last-observation
baselines. C2/C3 reuse the same evaluator and report horizon validity,
retrieval/path metrics, and counterfactual intervention metrics.

### A1 Baseline Probe

```bash
# Evaluate depth, camera (Auc_30, Rac_15, Tac_15), and identity
python eval_pertask_v3.py --models wan,vjepa2 --gpu 0
```

Results summary (InsScene-15K val, 953 samples):

| Model | depth↓ | identity↓ | Auc_30↑ | Rac_15↑ | Tac_15↑ |
|-------|--------|-----------|---------|---------|---------|
| Wan2.1 | 0.334 | 5.550 | 1.76% | 7.49% | 7.22% |
| V-JEPA2 | 0.322 | 4.628 | 2.15% | 7.97% | 7.43% |

### New Probe Results Summary

| Probe | Wan v1 | Wan ctrl | V-JEPA2 v1 | Note |
|-------|--------|----------|-----------|------|
| A2 overlap_acc↑ | 85.6% | 78.7% | 85.2% | trivial baseline=84.3% |
| A3 pair_acc↑ | 86.3% | 16.5% | pending re-eval | Wan numbers after dtype/seed fixes |
| B1 az_err↓ | legacy | legacy | legacy | pre no-pose/final-global/streaming task definition; rerun required |
| B1 el_err↓ | legacy | legacy | legacy | pre no-pose/final-global/streaming task definition; rerun required |
| C1 R@1↑ | legacy | legacy | legacy | pre exact-target/global-retrieval/streaming fix; rerun required |
| C2 global_R@1↑ | pending | pending | pending | rerun with context_segment+target_isolated and streaming_prefix variants |
| C3 intervention_validity↑ | pending | pending | pending | rerun with context_segment+target_isolated and streaming_prefix variants |

---

## Repository Structure

```
probe_spatial/
├── data/
│   └── download_inscene15k.py       # Download + extract InsScene-15K
├── configs/
│   ├── train.yaml                   # Hydra top-level config
│   ├── model/
│   │   ├── probe.yaml               # A1 probe (depth/camera/identity)
│   │   └── probe_ext.yaml           # Diagnostic probes
│   └── experiment/
│       ├── inscene15k/              # A1 experiments
│       ├── inscene15k_ext/          # Non-streaming A2/A3/B1/B2/C1/C2/C3 configs
│       └── inscene15k_streaming/    # Shared streaming A1/A2/B1/B2/C1/C2/C3 configs
├── features/
│   ├── run_inscene15k.py            # Feature extractor (normal/shuffled/context/target modes)
│   ├── wan/                         # Wan feature extraction
│   ├── cogvideox/                   # CogVideoX feature extraction
│   └── vjepa2/                      # V-JEPA2 feature extraction
├── vidfm3d/
│   ├── train.py                     # Hydra entry point
│   ├── eval_diag.py                 # Per-sample dump for diagnostic probes
│   ├── eval_diag_compare.py         # Aggregate runs into CSV
│   ├── data/components/
│   │   └── inscene15k_dataset.py    # Dataset (extended with diag flags)
│   ├── models/
│   │   ├── probe_ext_module.py      # LitModule for diagnostic probes
│   │   └── components/
│   │       ├── probe_view_consistency.py   # A2 head
│   │       ├── probe_abnormal.py           # A3 head
│   │       ├── probe_ego_belief.py         # B1 head
│   │       ├── probe_ego_belief_v2.py      # B2 head
│   │       ├── probe_action_dynamics.py    # C1 head
│   │       ├── probe_path_integration.py   # C2 head
│   │       └── probe_counterfactual.py     # C3 head
│   └── utils/
│       └── spatial_diag.py          # Geometry helpers (overlap, polar, pose)
├── scripts/
│   ├── run_streaming_probe_sweep.sh # Default streaming-prefix cache/train/eval sweep
│   ├── run_diag_sweep.sh            # Legacy non-streaming sweep; requires ALLOW_NON_STREAMING=1
│   ├── run_diag_new_sweep.sh        # Legacy non-streaming C2/C3; requires ALLOW_NON_STREAMING=1
│   ├── run_diag_new_parallel.sh     # Legacy non-streaming C2/C3; requires ALLOW_NON_STREAMING=1
│   ├── package_inscene_feature_dataset.sh # Package external feature caches for HF
│   └── run_diag_eval_sweep.sh       # Eval all runs, write CSV
└── eval_pertask_v3.py               # A1 evaluation script
```

---

## Known Issues & Fixes

**PyTorch 2.6 `weights_only` error on checkpoint resume**
`torch.load` defaults to `weights_only=True` in PyTorch 2.6+, which breaks Lightning checkpoint loading with OmegaConf objects. Fixed in `vidfm3d/train.py` via monkeypatch.

**`last.ckpt` not updating when val/loss plateaus**
Fixed in `vidfm3d/train.py` via monkeypatch that unconditionally saves `last.ckpt` every epoch.

**Hydra parse error with `=` in checkpoint path**
`epoch=49-step=104850.ckpt` contains `=` which confuses Hydra's CLI parser. Workaround: symlink to `/tmp/eval_ck.ckpt` first.

**C1 target caches**
The dataset now accepts only exact target-frame matches. Sparse target caches
are sampled directly; use `--num-targets 0` when every frame must be available.

---

## CO3Dv2 and DL3DV (original VidFM3D datasets)

<details>
<summary>Expand</summary>

### Dataset Preparation

**CO3Dv2:**
```bash
python -m vidfm3d.data.processing.co3d.extract_frames \
    --raw_root vidfm3d/data/CO3D/CO3D-data \
    --out_root vidfm3d/data/CO3D/CO3D-raw \
    --stride 1 --num_frames 81 --trunc_thresh 0.25 --resize_to 960 540
python -m vidfm3d.data.processing.process_co3d --root vidfm3d/data/CO3D
```

**DL3DV:**
```bash
python -m vidfm3d.data.processing.process_dl3dv --root vidfm3d/data/DL3DV
```

### Feature Extraction
```bash
python -m features.run_co3d  --vfm wan --model-id Wan-AI/Wan2.1-T2V-1.3B-Diffusers --output-layers 20 --t 749
python -m features.run_dl3dv --vfm wan --model-id Wan-AI/Wan2.1-T2V-1.3B-Diffusers --output-layers 20 --t 749
```

### Training
```bash
python vidfm3d/train.py experiment=co3d/wan job_name=wan
python vidfm3d/train.py experiment=dl3dv/wan job_name=wan
```

### Evaluation
```bash
python scripts/parse_results.py \
    --groups dl3dv,co3d --runs wan \
    --metrics "val/Auc_30,val/pmap_mse_aligned,val/loss_depth"
```

</details>

---

## Citation

```bibtex
@article{huang2025vidfm3d,
  title   = {How Much 3D Do Video Foundation Models Encode?},
  author  = {Huang, Zixuan and Li, Xiang and Lv, Zhaoyang and Rehg, James M.},
  booktitle = {arXiv preprint arXiv:2512.19949},
  year    = {2025}
}
```

## Acknowledgments

This project builds on:
- **[VidFM3D](https://github.com/zxhuang1698/VidFM3D)** — original probing framework (MIT)
- **[Fast3R](https://github.com/facebookresearch/fast3r)** — training & data infrastructure
- **[VGGT](https://github.com/facebookresearch/vggt)** — architecture and data processing

Foundation models evaluated: **Wan 2.1**, **CogVideoX**, **V-JEPA2** (each under its own license).
