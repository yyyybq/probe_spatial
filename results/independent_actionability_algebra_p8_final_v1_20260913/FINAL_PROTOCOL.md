# Independent P8 Actionability + Action Algebra — Final Protocol v1

Status: **FROZEN BEFORE FORMAL FEATURE EXTRACTION AND FINAL-SPLIT EVALUATION**  
Freeze date: 2026-09-13 UTC  
Source study: `independent_actionability_algebra_p8_20260831T091224Z_28bda217`

## Scope and claims

This experiment tests only (1) goal-conditioned one-step action selection from
frozen representations and (2) a small clean dynamics probe trained from
scratch, including limited action-algebra diagnostics. It does not evaluate B2,
relational state, object persistence, online control, collision-free planning,
backbone fine-tuning, or any old C-series checkpoint. Results are claims about
the four settings below plus task-trained probes, not about entire model
families or pretraining objectives.

The fixed settings are DINOv2-L at project output index L0, Fast3R at returned
feature index L24, V-JEPA2 ViT-L at encoder block L23, and Wan2.1-T2V-1.3B at
transformer block L20 and scheduler index 749. DINO `L0` is the output of the
project DINO extraction interface and is not described as transformer block 0.

## Data freeze and evaluation designation

The immutable source manifests are the existing independent-v1 files, with
SHA256 values in `MANIFEST_HASH_INDEX.json`. Formal ranges are:

| component | original train | original held-out | scenes (train / held-out) |
|---|---:|---:|---:|
| Actionability A0 | 844 | 227 | 289 / 77 |
| clean dynamics / primitive algebra | 1,024 | 256 | 444 / 108 |
| observed real-loop control | 0 | 412 matched pairs (824 records) | 0 / 106 |

The 684 value is the scanned raw-scene directory universe, not the number of
eligible scenes for any task. Of 684 candidates, 682 had the required source
artifacts; the selected task manifests cover 501 training scenes and 126
held-out scenes in their union.

The prior 512/128 pilot evaluated 128 Actionability validation records, 128
algebra validation records and 64 loop/non-loop pairs. Its selection touched
all 77 Actionability held-out scenes and all 108 algebra held-out scenes.
Therefore the held-out data are designated **validation**, not a new blind
test. No scene-disjoint confirmatory subset remains in this frozen universe.

All fitting choices use a deterministic shared scene split derived only from
the union of original training scenes: SHA256 rank with seed
`independent-p8-final-dev-v1-20260913`, first `round(20%)` scenes as dev. This
yields Actionability 650 fit / 194 dev and algebra 803 fit / 221 dev. After
selecting the fixed epoch and thresholds on dev, the same configuration is
refit on all original training records and evaluated once on validation. No
architecture, threshold, transform, or stopping rule is changed from
validation performance.

## Frozen representation interface

Every unit is keyed by model contract plus either `(scene, ordered eight raw
IDs)` for context or `(scene, exact raw ID)` for an isolated image. Fallback by
filename, requested role, row number or tail index is forbidden. The complete
feature-request-v2 SHA256 is
`fb1dbd2d71878edae7eae9034a16ee9809fc0c0cfb67352e0a301ad0edbed369`:
12,358 units per model, 49,432 total.

The stored analysis feature is the FP32 mean over all returned temporal and
spatial tokens. DINO processes exactly the ordered 8 context images (one image
for isolated units) framewise and averages every returned token; it does not
pretend to be a temporal backbone. F3R, V-JEPA2 and Wan receive the ordered
eight images followed by tail repeats to their fixed input lengths 64, 64 and
81; an isolated image is repeated to that length. Their returned temporal and
spatial tokens are all averaged. The padding difference and effective context
are reported in `MODEL_CARD.csv` and are a limitation of cross-model
comparison.

Wan uses the empty prompt, ensemble size 1, scheduler index 749 and a
sample-local deterministic noise generator derived only from the extraction
contract, model, scene and ordered raw IDs. The seed is independent of task,
role, label, candidate position and temporal direction. Fast3R receives an
analogously derived generator for deterministic internal view ordering.

For every unit the submit-side verifier requires the exact request identity,
request/source hashes, model/layer/timestep, expanded input policy, extractor
hash, snapshot revision/hash, payload SHA256, finite values, expected shape,
nonzero variance and no duplicates. Coverage passes only when
`requested=verified`, and missing, duplicate, fallback and raw-ID mismatch are
all zero. Pilot payloads may be pooled into the final namespace after verifying
their original payload and sidecar; they are not re-extracted.

## Actionability

Each query contains the frozen p8 context vector, an independently extracted
anchor vector, an independently extracted goal vector and two 9D actions.
Labels and regret come only from frozen GT camera poses. Pose, utilities,
sample/scene IDs, raw indices, temporal direction, difficulty fields and
endpoints are never inputs to a learned deployable scorer.

The direct scorer is shared across candidates:
`s(context, anchor, goal, action)`, a two-layer MLP with one fixed hidden width
of 256 and GELU. Candidate order is deterministically swapped with probability
1/2 during fitting, with labels swapped synchronously. Equivariance is checked
by swapping candidates at evaluation and requiring the selected physical
candidate to swap exactly (up to exact score ties).

The fixed comparisons are:

1. analytic random/identical-score and train-only temporal/candidate priors;
2. learned action-only, current+action/no-goal, and goal+action/no-current;
3. direct shared action scorer;
4. clean-dynamics selection, which predicts each candidate endpoint with the
   frozen clean dynamics probe and chooses the prediction with larger cosine
   similarity to the isolated goal;
5. observed-endpoint feature metric, which chooses the actually observed
   candidate endpoint nearest to the goal in feature space.

Item 5 is a privileged diagnostic reference, not deployable and not a
theoretical upper bound. Shuffled-goal evaluation is a fixed input intervention
on trained direct and dynamics selectors, not a separately trained baseline.

Raw cosine is the primary feature metric. Two robustness conditions are always
reported rather than selected per model: train-only mean centering, and
train-only diagonal standardization. No condition is chosen from validation.

Primary metrics are pairwise accuracy and raw pose-metric regret
`U(best)-U(selected)`. Secondary metrics are scene-macro accuracy,
positive-progress rate, median/P90 regret, and strata for goal distance,
utility margin, translation/rotation dominance, temporal direction, and
endpoint-seen status. Expected random accuracy is exactly 0.5; expected random
regret is the mean of the two candidate regrets and is evaluated analytically.

## Clean dynamics

The backbone is frozen. A single recurrent transition with hidden width 256 is
trained from scratch on the four real primitive actions and exact isolated
waypoint targets. Supervision is applied after each of the four recorded
actions; h=1/2/4 means the first one, first two and all four listed primitive
transitions, not three sparse equally timed inputs. Composed, identity, inverse
and loop examples are evaluation-only.

The fixed learned variants use the same architecture, optimizer, batch budget
and seeds: full visual+action, visual-only/constant identity action, and
action-only/constant zero context. Non-learned comparisons are exact isolated
anchor copy. The frozen full checkpoint is also evaluated with legal identity,
motion/horizon-matched shuffled actions and wrong actions. Shuffling is within
the same horizon and motion-magnitude decile where possible, uses a fixed
derangement, and reports unmatched records.

At h=1/2/4 we report cosine distance, MSE after the registered train-only
transform, retrieval R@1/R@5 from a validation-only candidate bank deduplicated
by exact `(scene, raw ID)`, gain over copy and visual-only, and action
sensitivity. A query target is never drawn from training and duplicate target
IDs are one bank item.

## Action algebra

The camera convention is world-to-camera composition
`T_0to2 = T_1to2 @ T_0to1`. The 9D identity is
`[1,0,0, 0,1,0, 0,0,0]`. Rotation is reconstructed by Gram-Schmidt; actions are
composed and inverted as SE(3) matrices, never by adding/subtracting 9D vectors.
Translations in the frozen manifests are physical ScanNet++ pose units with
implicit `scene_scale=1`; no goal- or candidate-dependent normalization exists.

The frozen clean dynamics checkpoint is evaluated on:

- real primitive sequences versus one exact composed transform;
- four primitives, two composed transforms, and one composed transform;
- same-K composed inputs padded with identity actions, including identity at
  the beginning, middle and end;
- identity output versus exact isolated anchor;
- synthetic inverse cycles versus wrong-inverse and repeated-action controls;
- 412 held-out real-loop/matched-non-loop pairs from evaluation scenes.

Every consistency result is paired with prediction-to-actual-endpoint error,
action dependence and wrong-action separation. Copy/constant predictors are
reported because they can have zero cycle drift. For observed loops we report
prediction-to-endpoint error separately from actual-endpoint-to-anchor
difference. Synthetic inverse is never described as observed motion. Decoder
state, not the full recurrent hidden state, is the object of the consistency
claim. There is no aggregate “passes algebra” score.

## Training, uncertainty and stopping

Learned main methods and key controls use seeds 41, 42 and 43 with paired data,
budgets and preprocessing. AdamW, learning rate `1e-3`, weight decay `1e-4`,
batch size 64 and maximum 200 epochs are fixed. Dev loss is checked every epoch;
the selected epoch is the smallest epoch attaining the minimum scene-macro dev
loss, with patience 20. Final refits use that epoch count and the complete
original training set.

Tables report seed mean ± sample SD. Scene-cluster paired bootstrap uses 2,000
replicates and seed 20260913; paired loop/non-loop records remain paired within
each scene resample. Bootstrap confidence intervals quantify scene sampling;
seed SD is reported separately. The four-model common set is the main table;
native coverage is supplementary.

Negative results, wide intervals, or failure to beat copy/shortcut baselines
are terminal scientific outcomes. Only an implementation, label or lineage bug
can amend this protocol, in a new version with its affected sample and minimum
recomputation scope recorded.

