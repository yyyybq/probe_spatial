# Goal-Conditioned Continuous Action Output — Relational Subset V2

The frozen evaluation contains 256 trajectories from 108 validation scenes. Results are mean ± SD across five seeds (41–45); intervals use 5,000 scene-cluster paired bootstrap replicates. Validation is not described as a blind test.

## Main results

| Model | Direction error ↓ | Distance MAE ↓ | Rotation error ↓ | Positive progress ↑ | Goal reduction ↑ |
|---|---:|---:|---:|---:|---:|
| DINO | 65.3 ± 1.3° | 0.084 ± 0.002 | 16.1 ± 0.1° | 0.537 ± 0.011 | 0.008 ± 0.006 |
| Fast3R | 64.7 ± 1.3° | 0.081 ± 0.004 | 16.2 ± 0.2° | 0.598 ± 0.034 | 0.025 ± 0.007 |
| V-JEPA2 | 66.2 ± 1.8° | 0.084 ± 0.001 | 15.1 ± 0.2° | 0.537 ± 0.024 | 0.007 ± 0.007 |
| Wan | 70.0 ± 1.5° | 0.089 ± 0.007 | 17.0 ± 0.5° | 0.502 ± 0.035 | -0.001 ± 0.009 |

## Confirmatory controls

- **DINO:** FULL−CURRENT positive progress -0.020 (95% CI -0.045, +0.005; interval includes zero); FULL−GOAL +0.138; FULL−PRIOR +0.069; correct−shuffled goal +0.008 (95% CI -0.021, +0.037; interval includes zero); relational−concat +0.026. Goal shuffling changes predicted direction by 29.8° and rotation by 5.8° on average.
- **Fast3R:** FULL−CURRENT positive progress -0.021 (95% CI -0.054, +0.014; interval includes zero); FULL−GOAL +0.133; FULL−PRIOR +0.129; correct−shuffled goal +0.012 (95% CI -0.027, +0.052; interval includes zero); relational−concat +0.032. Goal shuffling changes predicted direction by 39.1° and rotation by 7.0° on average.
- **V-JEPA2:** FULL−CURRENT positive progress -0.036 (95% CI -0.069, -0.004; favors control/shuffled goal); FULL−GOAL +0.089; FULL−PRIOR +0.069; correct−shuffled goal +0.003 (95% CI -0.029, +0.036; interval includes zero); relational−concat -0.013. Goal shuffling changes predicted direction by 30.2° and rotation by 5.9° on average.
- **Wan:** FULL−CURRENT positive progress -0.025 (95% CI -0.060, +0.010; interval includes zero); FULL−GOAL +0.060; FULL−PRIOR +0.033; correct−shuffled goal -0.023 (95% CI -0.063, +0.017; interval includes zero); relational−concat +0.016. Goal shuffling changes predicted direction by 45.6° and rotation by 8.5° on average.

## Results and interpretation

With no candidate actions provided, the strongest positive-progress rate is 0.598 for Fast3R. FULL beats GOAL_ONLY in positive progress for all four representations (effects +0.060 to +0.138, with every 95% interval above zero), so the current P8 representation contains information about the next local action beyond the endpoint image alone.

That result does not establish useful goal conditioning. FULL fails to beat CURRENT_ONLY in positive progress for every model (effects -0.036 to -0.020), and its goal-distance-reduction effect is negative for all four (-0.011 to -0.005). V-JEPA2 is significantly worse than CURRENT_ONLY in positive progress, while DINO, V-JEPA2, and Wan are significantly worse in goal reduction. More decisively, changing only the goal at inference changes the predicted direction by 29.8–45.6 degrees, yet correct-goal versus shuffled-goal effects on positive progress (-0.023 to +0.012) and goal reduction (-0.006 to +0.001) all include zero. The probe reacts to the goal representation, but the reaction is not reliably aligned with reaching the supplied goal.

The relational head is not a universal remedy. It improves positive progress over concatenation for DINO (+0.026) and Fast3R (+0.032), with intervals above zero, but only Fast3R obtains a clear goal-reduction gain (+0.008). V-JEPA2 and Wan show no reliable utility improvement. Thus the new inductive bias strengthens action readout in selected representations without changing the main negative conclusion about goal-specific control.

These probes perform supervised readout of the first action on observed four-step trajectories. They do not demonstrate online planning, a native policy, or a backbone world model; differences between backbones cannot be attributed solely to pretraining objectives.
