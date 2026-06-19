# Slurm examples

These examples parallelize independent experiments and extraction shards. They
do not use 100-way DDP for a small probe, because that changes the global batch
and optimizer-step budget.

1. Generate and freeze `configs/splits/inscene15k_v1.json`.
2. Export site-specific `PROJECT_ROOT`, `PYTHON`, `INSCENE_DATA_ROOT`,
   `INSCENE_FEAT_ROOT`, `INSCENE_TARGET_FEAT_ROOT`,
   `INSCENE_SHUFFLED_FEAT_ROOT`, and any required
   `SBATCH_ACCOUNT`/partition settings. Add your cluster's `#SBATCH --account`
   and `#SBATCH --partition` lines locally.
3. Submit extraction arrays per `(VFM, mode, layer set)`.
4. Validate caches with `scripts/validate_feature_cache.py`.
5. Submit `train_diag_array.sbatch`; each array element is one config and seed.

Array concurrency is controlled by `%N`, for example `--array=0-99%32` limits
the extraction to 32 simultaneous GPUs even if the array contains 100 shards.
