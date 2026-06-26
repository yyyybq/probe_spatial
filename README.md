# probe_spatial: Probing Spatial Understanding in Video Foundation Models

This repository extends [VidFM3D](https://github.com/zxhuang1698/VidFM3D) with a **Spatial Diagnostic Suite** — diagnostic probes that characterize what kind of 3D / temporal / ego-centric understanding a Video Foundation Model (VFM) has internalized, all trained on frozen VFM features without touching VFM weights.

| Family | Probe | Question |
|--------|-------|---------|
| **A. Global Spatial Perception** | A1 depth/camera/identity (VidFM3D baseline) | Does the VFM perceive a coherent 3D scene? |
| **A. Global Spatial Perception** | **A2 view consistency** | Can it tell whether two clips share a viewing region? |
| **A. Global Spatial Perception** | **A3 abnormal detection** | Can it detect temporally shuffled frames? |
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
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 \
    nvidia/label/cuda-12.4.0::cuda-toolkit -c pytorch -c nvidia
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation
pip install -r requirements.txt
pip install -e .

# 3. Download InsScene-15K data
python data/download_inscene15k.py --step all --base-dir /data/InsScene-15K

# 4. Extract VFM features (one VFM shown; repeat for cogvideox / vjepa2)
INSCENE_DATA=/data/InsScene-15K
python -m features.run_inscene15k --vfm wan \
    --model-id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --data-root ${INSCENE_DATA}/data --out-root ${INSCENE_DATA}/FEAT \
    --t 749 --output-layers 20
python -m features.run_inscene15k --vfm wan --mode shuffled \
    --model-id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --data-root ${INSCENE_DATA}/data --out-root ${INSCENE_DATA}/FEAT_SHUFFLED \
    --t 749 --output-layers 20
python -m features.run_inscene15k --vfm wan --mode context_segment \
    --model-id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --data-root ${INSCENE_DATA}/data --out-root ${INSCENE_DATA}/FEAT_CONTEXT \
    --t 749 --output-layers 20 --context-len 76
python -m features.run_inscene15k --vfm wan --mode target_isolated \
    --model-id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --data-root ${INSCENE_DATA}/data --out-root ${INSCENE_DATA}/FEAT_TARGET \
    --t 749 --output-layers 20 --num-targets 0

# Layer sweeps: pass multiple layers, then train with feature_layer=<L>.
# Current defaults are Wan/CogVideoX layer20 at t=749, V-JEPA2 layer23,
# and MLLM/VLM layer -1 caches.

# 5. Train diagnostic probes for Wan on GPU 0
bash scripts/run_diag_sweep.sh wan 0
VFMS="wan" bash scripts/run_diag_new_sweep.sh

# 6. Test & evaluate
bash scripts/run_diag_eval_sweep.sh   # writes comparison_val.csv
```

> **Note**: feature paths are hardcoded in `configs/experiment/inscene15k_ext/*.yaml`.
> Update the `root_vfm`, `context_feat_root`, `target_feat_root`, and `shuffled_feat_root` fields to match
> your actual data location before training, or pass overrides via Hydra:
> ```bash
> python vidfm3d/train.py experiment=inscene15k_ext/action_dynamics_wan_v1 \
>     context_feat_root=/data/InsScene-15K/FEAT_CONTEXT \
>     target_feat_root=/data/InsScene-15K/FEAT_TARGET
> ```

On a shared cluster, the dataset also honors `INSCENE_DATA_ROOT`,
`INSCENE_FEAT_ROOT`, `INSCENE_CONTEXT_FEAT_ROOT`, `INSCENE_TARGET_FEAT_ROOT`,
and `INSCENE_SHUFFLED_FEAT_ROOT`; these take precedence over legacy YAML paths.
See `EXPERIMENT_PROTOCOL.md` and `TRAINING_LOGIC_AUDIT.md` for the current
causal-cache protocol and task-by-task input contracts.
> See `PROBE_SPATIAL_GUIDE.md` §8.1.1 for layer-wise extraction, probing, and
> `scripts/summarize_layer_sweep.py` best-layer/last-layer reports.
> End-to-end layer sweeps use:
> ```bash
> VFM=wan PROBE=view_consistency LAYERS="0 5 10 15 20 25 29" \
>   bash scripts/run_feature_layer_probe_sweep.sh
> ```

---

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
# Download + extract in one step (~60 GB)
python data/download_inscene15k.py --step all --base-dir /data/InsScene-15K

# Or separately
python data/download_inscene15k.py --step download --base-dir /data/InsScene-15K
python data/download_inscene15k.py --step extract  --base-dir /data/InsScene-15K
```

Expected structure after extraction:
```
/data/InsScene-15K/
  data/
    processed_infinigen/      # Infinigen synthetic scenes
    processed_scannetpp_v2/   # ScanNet++ real-world scenes
  FEAT/                       # filled by feature extraction (step 4)
  FEAT_SHUFFLED/              # filled by feature extraction --mode shuffled (A3)
  FEAT_CONTEXT/               # filled by feature extraction --mode context_segment (C1/C2/C3 inputs)
  FEAT_TARGET/                # filled by feature extraction --mode target_isolated (C1/C2/C3 targets)
```

### CO3Dv2 / DL3DV (original VidFM3D datasets)

Follow the [original VidFM3D instructions](https://github.com/zxhuang1698/VidFM3D#dataset-preparation) for CO3Dv2 and DL3DV.

---

## Feature Extraction

Features are pre-extracted once and cached as `.sft` files. The extractor is resume-safe (skips already-done scenes).

### InsScene-15K — extraction modes

| Mode | Output dir | Used by |
|------|-----------|---------|
| `normal` (default) | `FEAT/` | A1, A2, B1/B2, non-causal/full-clip inputs |
| `shuffled` | `FEAT_SHUFFLED/` | A3 (temporally shuffled clips) |
| `context_segment` | `FEAT_CONTEXT/` | C1/C2/C3 causal inputs, with no future target frames in the VFM forward |
| `target_isolated` | `FEAT_TARGET/` | C1/C2/C3 isolated target features, with exact frame ids |

```bash
INSCENE_DATA=/data/InsScene-15K
VFM=wan
MODEL_ID=Wan-AI/Wan2.1-T2V-1.3B-Diffusers

# Normal features (required for all probes)
python -m features.run_inscene15k --vfm ${VFM} --model-id ${MODEL_ID} \
    --data-root ${INSCENE_DATA}/data --out-root ${INSCENE_DATA}/FEAT \
    --t 749 --output-layers 20

# Shuffled features (A3 only)
python -m features.run_inscene15k --vfm ${VFM} --model-id ${MODEL_ID} \
    --mode shuffled \
    --data-root ${INSCENE_DATA}/data --out-root ${INSCENE_DATA}/FEAT_SHUFFLED \
    --t 749 --output-layers 20

# Causal context features (C1/C2/C3 inputs)
python -m features.run_inscene15k --vfm ${VFM} --model-id ${MODEL_ID} \
    --mode context_segment \
    --data-root ${INSCENE_DATA}/data --out-root ${INSCENE_DATA}/FEAT_CONTEXT \
    --t 749 --output-layers 20 --context-len 76

# Target-isolated features (C1/C2/C3 targets)
python -m features.run_inscene15k --vfm ${VFM} --model-id ${MODEL_ID} \
    --mode target_isolated \
    --data-root ${INSCENE_DATA}/data --out-root ${INSCENE_DATA}/FEAT_TARGET \
    --t 749 --output-layers 20 --num-targets 0
```

C1/C2/C3 intentionally do not reuse `normal` features for their inputs or
targets. Input features come from causal `context_segment` forwards ending at
the last observed frame; target supervision comes from exact `target_isolated`
rows. This prevents future-frame leakage through clip-contextualized VFM
features.

Switch `--vfm cogvideox --model-id THUDM/CogVideoX-5b-I2V` or
`--vfm vjepa2` (uses `features/vjepa2/vjepa2_feature.py`) for other VFMs.

| VFM | `--vfm` | `--model-id` | `feat_postfix` | `in_channels` |
|-----|---------|-------------|---------------|--------------|
| Wan2.1-T2V-1.3B | `wan` | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | `_t749_layer20` | 1536 |
| CogVideoX-5B | `cogvideox` | `THUDM/CogVideoX-5b-I2V` | `_t749_layer20` | 3072 |
| V-JEPA2-ViT-L | `vjepa2` | see `features/vjepa2/` | `_layer23` | 1024 |

### Feature layer convention

The current diagnostic defaults are now centralized in
`vidfm3d/utils/feature_layers.py` and remain backward compatible:

| Model family | Current default layer | Meaning |
|-----|-----:|-----|
| Wan2.1 | 20 | diffusion transformer block 20 at timestep `t=749` |
| CogVideoX | 20 | diffusion transformer block 20 at timestep `t=749` |
| V-JEPA2 ViT-L | 23 | last encoder block, 0-based |
| Qwen2.5-VL / BAGEL caches | -1 | current default visual-token / last-layer cache |

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

Layer-wise probe training uses the same experiment YAML and only overrides the
feature filename / run name:

```bash
DRY_RUN=1 VFM=wan PROBE=ego_belief LAYERS="0 5 10 15 20 25 29" \
    bash scripts/run_feature_layer_probe_sweep.sh

VFM=vjepa2 PROBE=action_dynamics LAYERS="0 5 11 17 23" DEV=1 \
    bash scripts/run_feature_layer_probe_sweep.sh
```

The script extracts required feature modes for the selected probe, trains one
probe per layer, evaluates checkpoints, and summarizes the layer-wise curve. For
C1/C2/C3, extract the same layer list for both `context_segment` and
`target_isolated`. To summarize existing evaluated runs manually:

```bash
python scripts/summarize_layer_sweep.py \
    --vfm wan --probe ego_belief \
    --pattern "inscene15k_ext_ego_belief_wan_layer*" \
    --output layer_sweep_ego_belief_wan.csv
```

---

## Training

### Diagnostic probes (A2 / A3 / B1 / B2 / C1 / C2 / C3) — this work

**Train the original diagnostic sweep for one VFM on a single GPU:**
```bash
bash scripts/run_diag_sweep.sh wan     0   # GPU 0
bash scripts/run_diag_sweep.sh vjepa2  1   # GPU 1
bash scripts/run_diag_sweep.sh cogvideox 2 # GPU 2
```

**Train the newer C2/C3 probes:**
```bash
# Sequential
VFMS="wan vjepa2 cogvideox" bash scripts/run_diag_new_sweep.sh

# Parallel across GPUs
PYTHON=/path/to/python GPUS="0 1 2 3 4 5" bash scripts/run_diag_new_parallel.sh
```

**Train a single probe:**
```bash
# A2 view consistency
CUDA_VISIBLE_DEVICES=0 python vidfm3d/train.py \
    experiment=inscene15k_ext/view_consistency_wan_v1

# A3 abnormal detection
CUDA_VISIBLE_DEVICES=0 python vidfm3d/train.py \
    experiment=inscene15k_ext/abnormal_wan_v1

# B1 hidden-object localization
CUDA_VISIBLE_DEVICES=0 python vidfm3d/train.py \
    experiment=inscene15k_ext/ego_belief_wan_v1

# B2 object-query localization
CUDA_VISIBLE_DEVICES=0 python vidfm3d/train.py \
    experiment=inscene15k_ext/ego_belief_v2_wan_v1

# C1 latent dynamics
CUDA_VISIBLE_DEVICES=0 python vidfm3d/train.py \
    experiment=inscene15k_ext/action_dynamics_wan_v1

# C2 path integration
CUDA_VISIBLE_DEVICES=0 python vidfm3d/train.py \
    experiment=inscene15k_ext/path_integration_wan_v1

# C3 counterfactual action
CUDA_VISIBLE_DEVICES=0 python vidfm3d/train.py \
    experiment=inscene15k_ext/counterfactual_wan_v1
```

Training runs for 50 epochs (≈2–4 hours per probe on 1× L40S), auto-resumes from `last.ckpt`.

**Control experiments** (scrambled features as baseline):
```bash
CUDA_VISIBLE_DEVICES=0 python vidfm3d/train.py \
    experiment=inscene15k_ext/action_dynamics_wan_ctrl
```

### A1 Baseline Probe (depth / camera / identity) — original VidFM3D

```bash
# Wan2.1
CUDA_VISIBLE_DEVICES=0 python vidfm3d/train.py experiment=inscene15k/wan_probe_v3

# V-JEPA2
CUDA_VISIBLE_DEVICES=1 python vidfm3d/train.py experiment=inscene15k/vjepa2_probe_v3
```

Trains for 100 epochs, checkpointed every 5 epochs.

### Override data paths

If your data is not at the default path, set the environment roots before
launching Hydra:
```bash
INSCENE_DATA_ROOT=/your/InsScene-15K/data \
INSCENE_FEAT_ROOT=/your/InsScene-15K/FEAT \
INSCENE_CONTEXT_FEAT_ROOT=/your/InsScene-15K/FEAT_CONTEXT \
INSCENE_TARGET_FEAT_ROOT=/your/InsScene-15K/FEAT_TARGET \
python vidfm3d/train.py experiment=inscene15k_ext/action_dynamics_wan_v1
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
| B1 az_err↓ | legacy | legacy | legacy | pre no-pose task definition; rerun required |
| B1 el_err↓ | legacy | legacy | legacy | pre no-pose task definition; rerun required |
| C1 R@1↑ | legacy | legacy | legacy | pre exact-target/global-retrieval fix; rerun required |
| C2 global_R@1↑ | pending | pending | pending | rerun with context_segment + target_isolated caches |
| C3 intervention_validity↑ | pending | pending | pending | rerun with context_segment + target_isolated caches |

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
│       └── inscene15k_ext/          # A2/A3/B1/B2/C1/C2/C3 configs
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
│   ├── run_diag_sweep.sh            # Train A2/A3/B1/C1 for one VFM
│   ├── run_diag_new_sweep.sh        # Train C2/C3 sequentially
│   ├── run_diag_new_parallel.sh     # Train C2/C3 in parallel
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
