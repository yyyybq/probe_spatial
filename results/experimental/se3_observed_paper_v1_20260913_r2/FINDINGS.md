# Findings

All statements below concern the frozen legacy-development split; none is an independent confirmation.

- **DINOv2-L grounding — supported on this development set.** Scene-macro two-step gain versus copy is 0.2182; paired scene-bootstrap CI [0.1340, 0.2857]. See `paper_main_table.csv` and `paired_bootstrap.csv`.
- **DINOv2-L control-adjusted order — inconclusive.** Full normalized margin is 0.0189, tie-aware accuracy 0.5545; paired effects versus action-only/shuffle/matched-wrong/additive have CIs [-0.0872,-0.0006], [0.0144,0.0522], [0.0384,0.0654], and [0.0059,0.0300]. Registered claims require positive direction and Holm-adjusted p<.05 for all controls.
- **Fast3R ViT-L grounding — supported on this development set.** Scene-macro two-step gain versus copy is 0.1474; paired scene-bootstrap CI [0.0743, 0.2072]. See `paper_main_table.csv` and `paired_bootstrap.csv`.
- **Fast3R ViT-L control-adjusted order — inconclusive.** Full normalized margin is 0.0333, tie-aware accuracy 0.6318; paired effects versus action-only/shuffle/matched-wrong/additive have CIs [-0.0283,0.0264], [-0.0139,0.0860], [-0.0630,0.0717], and [-0.0008,0.0845]. Registered claims require positive direction and Holm-adjusted p<.05 for all controls.
- **V-JEPA2 ViT-L grounding — supported on this development set.** Scene-macro two-step gain versus copy is 0.1123; paired scene-bootstrap CI [0.0586, 0.1647]. See `paper_main_table.csv` and `paired_bootstrap.csv`.
- **V-JEPA2 ViT-L control-adjusted order — inconclusive.** Full normalized margin is 0.0153, tie-aware accuracy 0.4818; paired effects versus action-only/shuffle/matched-wrong/additive have CIs [-0.0728,0.0273], [0.0036,0.0766], [0.0017,0.0812], and [-0.0150,0.0595]. Registered claims require positive direction and Holm-adjusted p<.05 for all controls.
- **CogVideoX-5B-I2V grounding — inconclusive.** Scene-macro two-step gain versus copy is 0.0861; paired scene-bootstrap CI [-0.2268, 0.3055]. See `paper_main_table.csv` and `paired_bootstrap.csv`.
- **CogVideoX-5B-I2V control-adjusted order — inconclusive.** Full normalized margin is 0.0218, tie-aware accuracy 0.5136; paired effects versus action-only/shuffle/matched-wrong/additive have CIs [-0.0503,0.1189], [-0.0633,0.1610], [-0.0821,0.1541], and [-0.0706,0.1430]. Registered claims require positive direction and Holm-adjusted p<.05 for all controls.

The feature-distance and M3/M4 tables are descriptive explanations, not alternate endpoint ground truth. M4 is calibrated only on held-out real feature pairs; no decoded pose is treated as independent GT.

The M3/M4 labels follow the SeeSE3 readout framing (https://arxiv.org/html/2607.14228v1), but these are protocol-specific readouts and are not claimed as a paper reproduction.

Layer, true-context, diffusion-condition, and forward 1/2/4-step curves remain unavailable under a safe matched artifact contract. Prefix-labelled target-isolated caches do not vary genuine history length, and retracing is not relabelled as forward rollout.
- **DINOv2-L PCA-128 capacity control.** Train-only frozen PCA gives two-step grounding gain 0.2189 and normalized order margin 0.0281; this is a capacity sensitivity check, not a replacement primary cell.
- **CogVideoX-5B-I2V PCA-128 capacity control.** Train-only frozen PCA gives two-step grounding gain 0.0919 and normalized order margin 0.0447; this is a capacity sensitivity check, not a replacement primary cell.
