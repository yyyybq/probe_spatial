# P1 Findings

This follow-up reuses the frozen 436-sample p12 parent experiment and every validation-selected visual checkpoint; no visual head was retrained and no feature was re-extracted.

## Predeclared paired model comparison

Holm-adjusted paired evidence supports: normalized_composition_distance (Wan−V-JEPA2 0.01452, 95% CI [0.01072, 0.01879], Holm p=0).

## Composition and call-count alternative

- wan_l15: original two-update minus direct endpoint gap -7.758e-05 (CI [-0.0001772, 2.549e-05]); identity-before -1.803e-05; identity-after -3.12e-05. Paired sequential-minus-identity contrasts give pre CI [-9.872e-05, -2.062e-05] and post CI [-8.432e-05, -8.626e-06]; the original two-action gap differs from both identity-padded gaps, so update count alone is insufficient. Low endpoint-error similarity by itself is not treated as composition consistency.
- vjepa2_l11: original two-update minus direct endpoint gap 1.277e-07 (CI [-2.749e-05, 2.705e-05]); identity-before 1.575e-05; identity-after 1.816e-05. Paired sequential-minus-identity contrasts give pre CI [-3.46e-05, 3.42e-06] and post CI [-3.812e-05, 2.189e-06]; the update-count explanation is not excluded. Low endpoint-error similarity by itself is not treated as composition consistency.
Exact identity was absent among frozen training actions. Coordinatewise min–max results are reported only as range checks, never as strict distribution certification.

## Order signal and limitations

- High physical-difference bin (rad): wan_l15 margin 0.0005987, CI [-0.001387, 0.002756], tie-aware 0.523; vjepa2_l11 margin -0.0002496, CI [-0.001778, 0.001305], tie-aware 0.505.
- High physical-difference bin (m): wan_l15 margin -0.0008615, CI [-0.004014, 0.002141], tie-aware 0.483; vjepa2_l11 margin -0.001515, CI [-0.003696, 0.0004793], tie-aware 0.483.

## Descriptive layer relationship

Across the seven Wan layers, C1 error correlates with raw sequential–direct distance at r=0.850, but with the copy-normalized composition distance at r=-0.258 and normalized order margin at r=0.210. Across the five V-JEPA2 layers the corresponding values are r=0.989, r=-0.265, and r=0.108. These are descriptive correlations over only 7/5 exported layers: prediction error and structural scores do not move together uniformly, and no best layer is selected post hoc.

## Analytic positive control

The analytic oracle has sequential/direct gap 0 m and positive correct-versus-swapped margin 0.1107 m. The single-step learned GRU has mean one-step RMSE 0.1785 m, sequential RMSE 0.701 m, and order margin -0.0003576 m. The oracle validates the metric implementation, while the learned head's one-step and recursive errors make diagnostic-head limitation an observed alternative explanation rather than a hypothetical one.

Consequently, weak visual order evidence can reflect the physical sample distribution, the single-step-trained recurrent diagnostic, the frozen representation, or an interaction among them. The geometry strata and positive control narrow those alternatives but do not identify representation incapacity, and observed two-step behavior is not a proof of a complete SE(3) group action.
