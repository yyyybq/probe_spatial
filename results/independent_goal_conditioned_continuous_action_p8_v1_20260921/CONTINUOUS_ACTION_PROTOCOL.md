# Goal-Conditioned Continuous Action Output — Protocol v1

Status: **FROZEN BEFORE FORMAL PROBE TRAINING**  
Freeze date: 2026-09-21 UTC

## Question and claim boundary

This experiment asks whether a frozen representation can directly emit a
continuous local motion command when given the current p8 observation and a
goal image. No candidate action is provided. The supervised target is the
first real primitive action on an observed four-step trajectory, not the
straight-line transform from the anchor to the goal.

The result concerns a task-trained shallow probe on frozen features. It is not
an online navigation policy, a collision-free planner, or evidence that the
backbone natively emits actions.

## Frozen data and features

The experiment reuses the exact scene-disjoint Algebra splits and feature
request from Independent P8 Final v1:

| split | records | scenes | manifest SHA256 |
|---|---:|---:|---|
| fit | 803 | 354 | `e6190674b145923c512c055f799ffaf4aac52f02351cdd14389f71300d91268e` |
| dev | 221 | 90 | `a67ae524760a25a5c9d5505bf8799c835812683571309a58f35bb86cc6d718d8` |
| validation | 256 | 108 | `947e251d4ff6c75282ba580ec0a00d4a4bbf877077a7386b92b8764a23086688` |

The three scene sets have zero overlap. Validation was touched by the earlier
P8 pilot and remains validation rather than a blind test.

For every record, the only learned-probe inputs are:

- current: the exact ordered eight-raw-ID `streaming_context_p8` feature ending
  at the anchor;
- goal: the exact isolated feature for the observed h4 endpoint.

All four models have both exact units for every record under request SHA256
`fb1dbd2d71878edae7eae9034a16ee9809fc0c0cfb67352e0a301ad0edbed369`.
The fixed representations are DINOv2-L project L0, Fast3R L24, V-JEPA2 L23,
and Wan L20 at scheduler index 749. No feature extraction is permitted unless
an exact lookup fails, in which case only that model branch is blocked.

## Target and SE(3) convention

The label is `primitive_actions_9d[0]`, recomputed and checked from the anchor
and h1 camera-to-world poses. The canonical 9D vector contains the first two
rows of the relative rotation followed by translation. It maps anchor camera
coordinates to h1 camera coordinates. Rotation is decoded by the existing
Gram--Schmidt rule, and a predicted endpoint is obtained as
`predicted_w2c = decoded_action @ anchor_w2c`.

The nine output coordinates are standardized with fit-only target mean and
standard deviation for optimization, then unstandardized before SE(3)
decoding and evaluation. Translation remains in the frozen source-pose units;
no new coordinate convention or per-scene scale is introduced.

## Probe and fixed optimization

Every learned variant uses the same two-layer head:

`concat(current, goal) -> Linear(2D,256) -> GELU -> Linear(256,9)`.

Current and goal features are separately standardized using fit-only
statistics. Missing inputs in ablations are exact zeros after normalization,
so topology and parameter count remain unchanged within a model. Seeds are
41, 42 and 43. AdamW uses learning rate `1e-3`, weight decay `1e-4`, batch size
64, at most 300 epochs and patience 30. The loss is fit-normalized Smooth L1
over all nine coordinates plus `0.1` times decoded rotation geodesic error.
The loss weight is shared by all models and variants.

Epoch selection minimizes scene-macro dev loss. The selected epoch is then
used to refit on the complete 1,024-record original training set, while the
fit-derived normalization remains frozen. Validation is evaluated once.

## Variants and controls

- `full`: current p8 feature plus the correct isolated h4 goal feature;
- `goal_only`: zero current plus the correct goal;
- `current_only`: current plus zero goal;
- `shuffled_goal`: the same head trained and evaluated with a deterministic
  within-split cyclic derangement of goal raw IDs; labels remain unchanged;
- `mean_motion_prior`: the decoded componentwise mean a1 action over the full
  original training set, with no visual input;
- `oracle_a1`: the observed a1 label, reported only as a trajectory reference.

The primary attribution comparison is `full - goal_only`. Failure to improve
on this control forbids a claim that the current visual state contributes
additional action information.

## Metrics

Motion regression reports translation-direction angular error in degrees,
translation-distance MAE, translation relative error, and rotation geodesic
error in degrees. Direction error is defined for nonzero ground-truth
translation and uses a fixed numerical epsilon only for normalization.

Decision utility applies the predicted SE(3) action to the anchor pose and
uses the existing metric `translation + 0.5 * rotation_angle_rad` to report:

- goal-distance reduction (higher is better);
- positive-progress indicator/rate (higher is better);
- remaining distance to the observed h4 goal (lower is better).

The same quantities are reported for the observed a1 oracle. Validation
strata are fixed using original-training thresholds for translation- versus
rotation-dominance, goal-distance tertile and first-action-magnitude tertile.

## Statistics and integrity

Learned variants report three-seed mean and sample SD. Confidence intervals
use 2,000 scene-cluster bootstrap replicates after averaging seeds for each
physical sample. Paired effects are reported for full versus goal-only,
current-only and mean-motion-prior for every primary metric.

The evaluator must verify exact context and goal lookup, recompute a1 from
poses, confirm the frozen split hashes and scene boundaries, and assert that
the learned tensor constructor receives only current and goal features. GT
poses, goal poses, action labels, future actions, raw indices, scene IDs and
trajectory direction are forbidden model inputs.
