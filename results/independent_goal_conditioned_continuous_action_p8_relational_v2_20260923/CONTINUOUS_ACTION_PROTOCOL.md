# Goal-Conditioned Continuous Action Output: Relational Subset V2

Status: frozen before formal execution on 2026-09-23.

## Question

Without candidate actions, can a shallow probe read out the first continuous
local motion command from a frozen current P8 representation and a frozen goal
image representation?  The decisive question is whether the goal contributes
information beyond the current observation alone.

## Frozen data

This experiment reuses the independently constructed four-step real
trajectories and their scene-disjoint split without modifying examples or
labels:

- fit: 803 trajectories;
- dev: 221 trajectories;
- validation: 256 trajectories;
- total: 1,280 trajectories from 552 scenes.

For `anchor --a1--> h1 --a2--> h2 --a3--> h3 --a4--> h4`, the inputs are the
anchor's exact ordered P8 streaming feature and h4's exact isolated feature.
The target is the actually executed first primitive action `a1`, represented
by the existing canonical 9D action (rotation 6D plus normalized translation
3D).  Validation remains validation, not a new blind test.

## Frozen representations

The comparison uses the final available representation stage for each frozen
backbone:

- DINOv2-L: project output L0 (the registered final hidden representation);
- Fast3R: L24;
- V-JEPA2: L23;
- Wan: L29 at t=749.

DINOv2-L, Fast3R, and V-JEPA2 reuse the already verified exact independent
features.  Wan L29 is extracted only for the exact current and h4 units required
by these 1,280 trajectories, using the same independent input, padding, seed,
pooling, and raw-ID identity contract as the accepted Wan L20 store.  Features
from a different prompt or preprocessing contract are not mixed in.

## Probe and controls

All learned probes freeze the backbone and use identical optimization budgets.
The primary relational head applies separate current and goal projections and
fuses `[c, g, g-c, c*g]` before a shallow MLP predicts the 9D action.  The old
concatenation MLP is retained as a bridge baseline.

The preregistered variants are:

- `full_relational`: current plus correct goal, relational head;
- `full_concat`: current plus correct goal, concatenation head;
- `current_only`: current plus a zeroed goal branch;
- `goal_only`: goal plus a zeroed current branch;
- `shuffled_goal_train`: current plus a deterministic, different-scene shuffled
  goal during training and evaluation;
- `mean_motion_prior`: train-split mean action, without learned visual input;
- `oracle_a1`: true a1, reported only as a trajectory-label reference.

In addition, each trained `full_relational` checkpoint is evaluated twice on
the same validation anchors: once with the correct goal and once with a fixed,
difficulty-matched, different-scene shuffled goal.  This inference-time
intervention changes no weights and is the cleanest test of goal sensitivity.

## Training and selection

Seeds are 41, 42, 43, 44, and 45.  Feature and target normalization statistics
are fit-split only.  The loss is Huber regression on normalized action plus a
fixed rotation-consistency term.  Learning rate, weight decay, hidden width,
batch size, patience, and loss weights are shared across models and variants.
Early-stopping epoch is selected by scene-macro dev loss; the selected epoch is
then used for the fit+dev refit.  Validation is used exactly once for reporting.

## Metrics and confirmatory effects

Motion regression reports translation-direction angular error, translation
distance MAE and relative error, and rotation geodesic error.  Decision utility
applies the decoded action to the anchor pose using the verified SE(3)
convention and reports normalized goal-distance reduction, positive-progress
rate, and remaining distance to h4.  Oracle a1 progress is the label reference.

The primary paired effects are `full_relational - current_only`, correct-goal
versus inference-shuffled-goal from the same checkpoint, `full_relational -
full_concat`, and `full_relational - mean_motion_prior`.  Five-seed mean and SD
are reported.  Confidence intervals use 5,000 scene-cluster paired bootstrap
replicates; seed repetitions are not treated as independent scenes.  Preset
strata cover translation- versus rotation-dominant motion, goal-distance
tertiles, and first-action-magnitude tertiles, with thresholds derived without
validation tuning.

## Interpretation boundary

Evidence that the current state adds action information requires the relational
full probe to beat current-only.  Evidence that the probe uses the goal further
requires degradation or action change under the same-checkpoint shuffled-goal
intervention.  Better performance than goal-only alone is insufficient.  A
negative result is retained without changing data, layers, labels, or budgets.
Conclusions are limited to shallow probes over frozen representations and do not
establish a native policy or world model in the backbone.

## Leakage exclusions

Learned inputs contain only the current P8 feature and isolated goal feature.
GT pose, goal pose, action labels, future actions, raw frame indices, scene IDs,
and trajectory direction are unavailable to the scorer and are used only for
loss construction or evaluation.  Feature lookup is by exact ordered raw-ID
tuple or exact isolated raw ID; directory order and implicit row order are not
accepted.
