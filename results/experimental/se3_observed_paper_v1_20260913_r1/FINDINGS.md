# Findings

All statements below concern the frozen legacy-development split; none is an independent confirmation.

- **DINOv2-L grounding — supported on this development set.** Scene-macro two-step gain versus copy is 0.2182; paired scene-bootstrap CI [0.1340, 0.2857]. See `paper_main_table.csv` and `paired_bootstrap.csv`.
- **DINOv2-L control-adjusted order — inconclusive.** Full normalized margin is 0.0189, tie-aware accuracy 0.5545; paired effects versus action-only/shuffle/additive have CIs [-0.0872,-0.0006], [0.0144,0.0522], and [0.0059,0.0300].
- **Fast3R ViT-L grounding — supported on this development set.** Scene-macro two-step gain versus copy is 0.1474; paired scene-bootstrap CI [0.0743, 0.2072]. See `paper_main_table.csv` and `paired_bootstrap.csv`.
- **Fast3R ViT-L control-adjusted order — inconclusive.** Full normalized margin is 0.0333, tie-aware accuracy 0.6318; paired effects versus action-only/shuffle/additive have CIs [-0.0283,0.0264], [-0.0139,0.0860], and [-0.0008,0.0845].
- **V-JEPA2 ViT-L grounding — supported on this development set.** Scene-macro two-step gain versus copy is 0.1123; paired scene-bootstrap CI [0.0586, 0.1647]. See `paper_main_table.csv` and `paired_bootstrap.csv`.
- **V-JEPA2 ViT-L control-adjusted order — inconclusive.** Full normalized margin is 0.0153, tie-aware accuracy 0.4818; paired effects versus action-only/shuffle/additive have CIs [-0.0728,0.0273], [0.0036,0.0766], and [-0.0150,0.0595].
- **CogVideoX-5B-I2V grounding — inconclusive.** Scene-macro two-step gain versus copy is 0.0861; paired scene-bootstrap CI [-0.2268, 0.3055]. See `paper_main_table.csv` and `paired_bootstrap.csv`.
- **CogVideoX-5B-I2V control-adjusted order — inconclusive.** Full normalized margin is 0.0218, tie-aware accuracy 0.5136; paired effects versus action-only/shuffle/additive have CIs [-0.0503,0.1189], [-0.0633,0.1610], and [-0.0706,0.1430].

The feature-distance and M3/M4 tables are descriptive explanations, not alternate endpoint ground truth. M4 is calibrated only on held-out real feature pairs; no decoded pose is treated as independent GT.

Layer, true-context, diffusion-condition, and forward 1/2/4-step curves remain unavailable under a safe matched artifact contract. Prefix-labelled target-isolated caches do not vary genuine history length, and retracing is not relabelled as forward rollout.
