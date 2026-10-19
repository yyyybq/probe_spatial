# SE3-Observed-v1 study specification

Frozen before the paper-v1 scores in this directory were computed.

Research question: Do frozen visual representations support action-grounded and order-sensitive latent updates under observed camera motion?

The four questions are: (Q1) held-out relative-pose readout from real observed feature pairs; (Q2) whether the correct action improves one- and two-step real-endpoint prediction; (Q3) whether correct action order is closer to the observed endpoint than reversed order and exceeds action-only, shuffled-pose, and commutative-additive controls; and (Q4) how composition error varies with model, available layer/context/diffusion condition, and physical non-commutativity.

The core cells are DINOv2-L export 0, Fast3R export 24, V-JEPA2 encoder block 23, and CogVideoX block 20, all from the existing p8 C artifacts. Seeds are 42--46. Backbones and features are frozen. The residual probe uses hidden width 64, AdamW (lr 1e-3, weight decay 1e-4), MSE plus 0.25 cosine loss, early stopping on development validation, and only real training-scene single-transition targets. No order, composition, cycle, rollout, validation-test, or test loss is used.

Primary aggregation is scene-macro. Tuple-micro is supplementary. Paired scene bootstrap uses 2,000 fixed resamples shared across models and controls. The primary order quantities are raw margin, copy-normalized margin, and tie-aware accuracy with absolute tie tolerance 1e-7 and normalization epsilon 1e-8. Holm correction is applied jointly to the registered control and pairwise model comparisons.

The endpoint convention is `E=inverse(c2w)`, `T_AB=E_B@inverse(E_A)`, and `T_AC=T_BC@T_AB`. Every endpoint is the real observed `z_C`; synthetic-nearest endpoints are excluded.

Conclusion boundary: these are controlled external-probe results, not native model dynamics, full SE(3) equivariance, counterfactual dual-image validation, or evidence that a pretraining objective causes an advantage. This run is a frozen re-analysis of the legacy discovery set, not an independent confirmation.
