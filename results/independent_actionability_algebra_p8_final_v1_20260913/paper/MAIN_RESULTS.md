# Main Results

All numbers below use the frozen validation set; it is not a blind test. Accuracy and regret are seed means with scene-cluster 95% CIs.

| Model | Direct acc. | Goal+action/no-current acc. | Dynamics acc. | Endpoint reference acc. | Direct regret | Dynamics regret |
|---|---:|---:|---:|---:|---:|---:|
| DINO | 0.762 ± 0.008 [0.709, 0.815] | 0.761 ± 0.017 [0.709, 0.811] | 0.762 ± 0.012 [0.703, 0.815] | 0.855 ± NA [0.810, 0.899] | 0.060 ± 0.002 [0.045, 0.078] | 0.054 ± 0.003 [0.039, 0.070] |
| F3R | 0.799 ± 0.013 [0.745, 0.848] | 0.789 ± 0.004 [0.730, 0.840] | 0.781 ± 0.003 [0.725, 0.833] | 0.780 ± NA [0.724, 0.836] | 0.050 ± 0.002 [0.037, 0.065] | 0.050 ± 0.002 [0.037, 0.065] |
| V-JEPA2 | 0.781 ± 0.007 [0.732, 0.831] | 0.761 ± 0.014 [0.704, 0.814] | 0.767 ± 0.008 [0.712, 0.818] | 0.780 ± NA [0.725, 0.833] | 0.055 ± 0.002 [0.042, 0.070] | 0.055 ± 0.004 [0.041, 0.070] |
| Wan | 0.759 ± 0.009 [0.705, 0.809] | 0.783 ± 0.003 [0.725, 0.833] | 0.636 ± 0.011 [0.577, 0.692] | 0.683 ± NA [0.624, 0.739] | 0.060 ± 0.002 [0.045, 0.078] | 0.101 ± 0.005 [0.084, 0.120] |

Pre-registered questions:

- DINO direct readout over random: **supported** (+0.262 (95% CI +0.204 to +0.315; favored the first method)).
- F3R direct over dynamics-based selection: **inconclusive** (-0.018 (95% CI -0.058 to +0.022; was inconclusive) for dynamics minus direct).
- V-JEPA2 dynamics-based decision advantage over direct: **inconclusive** (-0.015 (95% CI -0.056 to +0.029; was inconclusive)). Algebra evidence remains decomposed in Table 2.
- Wan direct utility beyond action-only and no-goal shortcuts: **supported** (effects +0.081 (95% CI +0.029 to +0.131; favored the first method) and +0.069 (95% CI +0.027 to +0.109; favored the first method)).

Current-state contribution beyond goal and action alone:

- DINO direct minus goal+action/no-current: **inconclusive** (+0.001 (95% CI -0.046 to +0.048; was inconclusive)).
- F3R direct minus goal+action/no-current: **inconclusive** (+0.010 (95% CI -0.029 to +0.050; was inconclusive)).
- V-JEPA2 direct minus goal+action/no-current: **inconclusive** (+0.021 (95% CI -0.019 to +0.066; was inconclusive)).
- Wan direct minus goal+action/no-current: **inconclusive** (-0.023 (95% CI -0.055 to +0.009; was inconclusive)).

The direct scorer does not significantly outperform the goal+action/no-current control for any representation. The experiment therefore supports goal-conditioned candidate-action selection, but does not establish an additional contribution from the current visual state. The endpoint column is privileged post-action reference information, not a theoretical upper bound. Structural diagnostics are intentionally not collapsed into an algebra pass score.
