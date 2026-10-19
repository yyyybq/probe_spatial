# Goal-Conditioned Continuous Action Output — Results

All results use 256 trajectories from 108 validation scenes; this split is not a blind test. Learned values are mean ± sample SD over seeds 41/42/43, and paired intervals use 2,000 scene-cluster bootstrap replicates.

## Main results

| Model | Direction error ↓ | Distance MAE ↓ | Relative distance error ↓ | Rotation error ↓ | Positive progress ↑ | Goal reduction ↑ |
|---|---:|---:|---:|---:|---:|---:|
| DINO | 71.6 ± 4.8° | 0.077 ± 0.003 | 0.503 ± 0.003 | 16.7 ± 0.3° | 0.510 ± 0.009 | -0.003 ± 0.005 |
| Fast3R | 65.4 ± 1.2° | 0.076 ± 0.002 | 0.507 ± 0.021 | 16.7 ± 0.4° | 0.551 ± 0.010 | 0.015 ± 0.003 |
| V-JEPA2 | 69.0 ± 3.1° | 0.076 ± 0.001 | 0.514 ± 0.010 | 15.4 ± 0.3° | 0.543 ± 0.017 | 0.006 ± 0.003 |
| Wan | 72.5 ± 2.2° | 0.075 ± 0.001 | 0.510 ± 0.015 | 16.9 ± 0.4° | 0.488 ± 0.007 | 0.001 ± 0.004 |

## Does the current state add information beyond the goal?

- **DINO:** positive-progress effect +0.151 (95% CI +0.090 to +0.208; favors full); direction improvement +12.1 (95% CI +8.2 to +16.0; favors full) degrees, distance-MAE improvement +0.019 (95% CI +0.014 to +0.024; favors full), and rotation improvement +2.7 (95% CI +1.9 to +3.5; favors full) degrees.
- **Fast3R:** positive-progress effect +0.141 (95% CI +0.073 to +0.207; favors full); direction improvement +13.1 (95% CI +9.2 to +17.0; favors full) degrees, distance-MAE improvement +0.017 (95% CI +0.013 to +0.022; favors full), and rotation improvement +2.7 (95% CI +1.8 to +3.7; favors full) degrees.
- **V-JEPA2:** positive-progress effect +0.109 (95% CI +0.045 to +0.173; favors full); direction improvement +10.9 (95% CI +6.6 to +15.2; favors full) degrees, distance-MAE improvement +0.018 (95% CI +0.014 to +0.023; favors full), and rotation improvement +3.3 (95% CI +2.5 to +4.2; favors full) degrees.
- **Wan:** positive-progress effect +0.107 (95% CI +0.051 to +0.160; favors full); direction improvement +8.1 (95% CI +3.6 to +12.4; favors full) degrees, distance-MAE improvement +0.011 (95% CI +0.006 to +0.016; favors full), and rotation improvement +2.4 (95% CI +1.6 to +3.2; favors full) degrees.

Positive effects are defined so that positive always favors FULL. The current representation contributes additional action information only where the FULL-versus-GOAL_ONLY interval excludes zero.

## Does the goal add information beyond the current state?

- **DINO:** FULL-minus-CURRENT_ONLY positive progress -0.018 (95% CI -0.052 to +0.016; inconclusive) and goal reduction -0.015 (95% CI -0.021 to -0.009; favors control). Its SHUFFLED_GOAL positive-progress rate is 0.514, versus 0.510 for FULL.
- **Fast3R:** FULL-minus-CURRENT_ONLY positive progress -0.095 (95% CI -0.139 to -0.052; favors control) and goal reduction -0.021 (95% CI -0.030 to -0.012; favors control). Its SHUFFLED_GOAL positive-progress rate is 0.603, versus 0.551 for FULL.
- **V-JEPA2:** FULL-minus-CURRENT_ONLY positive progress -0.007 (95% CI -0.047 to +0.033; inconclusive) and goal reduction -0.009 (95% CI -0.015 to -0.002; favors control). Its SHUFFLED_GOAL positive-progress rate is 0.521, versus 0.543 for FULL.
- **Wan:** FULL-minus-CURRENT_ONLY positive progress -0.043 (95% CI -0.083 to -0.003; favors control) and goal reduction -0.009 (95% CI -0.017 to -0.002; favors control). Its SHUFFLED_GOAL positive-progress rate is 0.483, versus 0.488 for FULL.

FULL does not improve goal reduction over CURRENT_ONLY for any representation; all four paired intervals favor CURRENT_ONLY. FULL is also worse in direction and rotation error for all four models. Together with the similar SHUFFLED_GOAL results, this supports readout of trajectory-conditioned motion statistics from the current representation, but not reliable conditioning on the supplied goal image.

## Relation to candidate ranking

The highest continuous-action positive-progress rate is obtained by **Fast3R** (0.551). Previous candidate-ranking accuracies were DINO 0.762, Fast3R 0.799, V-JEPA2 0.781, Wan 0.759. Candidate ranking supplies the actions to be compared, whereas this experiment must generate a local action; agreement or disagreement in model ordering is therefore reported descriptively rather than treated as the same capability.

Wan previously combined strong direct candidate ranking with weak dynamics-based selection. Its continuous-action result is interpreted from its FULL regression and progress metrics together with the FULL-versus-GOAL_ONLY and FULL-versus-prior effects; a high ranking score alone is not evidence that Wan can synthesize a continuous motion command.

## Interpretation

This experiment evaluates whether a shallow probe can read a goal-conditioned first-step motion from frozen features without receiving candidate actions. It remains supervised imitation of the first primitive on an offline observed trajectory, not online planning or proof that the backbone natively implements a policy.

## Paper-ready Results + Interpretation

We next removed the candidate-action set and asked a shallow probe to regress the first observed local motion directly from the frozen p8 current representation and an isolated h4 goal representation. Across the four backbones, FULL probes achieved positive-progress rates of 0.488--0.551, compared with 0.668 for the observed-a1 trajectory reference; Fast3R gave the highest rate (0.551 ± 0.010) and mean goal reduction (0.015 ± 0.003). Adding the current representation to GOAL_ONLY improved positive progress by 0.107--0.151, with all scene-cluster bootstrap intervals excluding zero, and reduced direction error by 8.1--13.1 degrees, distance MAE by 0.011--0.019, and rotation error by 2.4--3.3 degrees. Thus, the p8 current representation contains information predictive of the next local action beyond that available in the endpoint image alone.

This result does not, however, establish goal-conditioned action synthesis. CURRENT_ONLY matched or exceeded FULL: FULL-minus-CURRENT_ONLY goal-reduction effects ranged from -0.009 to -0.021 and excluded zero for every backbone, while direction and rotation errors also favored CURRENT_ONLY for all four models. Shuffling goal images produced similar positive-progress rates to FULL. The most conservative interpretation is therefore that these shallow probes recover trajectory- and state-dependent local-motion regularities, but do not reliably combine the current and goal representations to select a goal-specific continuous command. This also sharpens the earlier candidate-ranking result: supplying two actions makes direct comparison substantially easier than generating an action, and Wan's strong ranking performance does not transfer to continuous output (0.488 positive progress; 0.001 mean goal reduction).

## Rebuild tables and statistics without retraining

Run `OUTPUT_DIR=/tmp/continuous_action_rebuild bash scripts/build_independent_continuous_action_results.sh` from the repository root. The command consumes the frozen per-model outputs and rebuilds all aggregate CSV, LaTeX, integrity, and Results files.
