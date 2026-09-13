# Normalization and mask audit

- Main labels: uint16 millimetres -> metres -> validity-weighted 72x128 area resize -> divide by each sample's finite-positive mean.
- `target_layout_scale` is label/oracle metadata and is never a readout or predictor input.
- Readout latent normalization is internal per-token LayerNorm; external predictor targets and 3x4 motion entries use train-only channel mean/std saved in the checkpoint.
- All depth metrics use only the GT finite-positive mask; no predicted mask can suppress a pixel.
- `oracle_gt_scale_pointmap_mae_m` is explicitly oracle-scaled and is not a metric-scale prediction claim. Intrinsics are scaled with the direct depth resize.
- V-JEPA2 predictor context masks are applied before encoder attention; target masks have explicit batch dimension and preserve absolute THW RoPE indices.
