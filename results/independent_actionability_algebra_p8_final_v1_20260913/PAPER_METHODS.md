# Methods — Independent P8 Actionability and Action Algebra

## Data and evaluation status

We study frozen visual representations on observed ScanNet++ iPhone sequences.
The Actionability split contains 844 training and 227 validation queries from
289 and 77 scenes. The dynamics split contains 1,024 training and 256
validation four-transition paths from 444 and 108 scenes. A separate validation
set contains 412 observed near-return paths paired with 412 same-scene non-loop
paths. Training and validation scenes are disjoint across both tasks. Because a
prior pilot touched every validation scene, we call this split validation, not
a blind test. Model selection uses a shared deterministic scene split derived
only from original training scenes (Actionability 650/194 fit/dev; dynamics
803/221 fit/dev), followed by refitting on all original training records.

Each Actionability record provides an ordered eight-frame context ending at an
anchor, an isolated goal, two observed candidate endpoints, and two relative
camera actions. Labels select the action with greater pose-metric progress
toward the goal. Each dynamics record provides an isolated anchor, four real
primitive relative camera transforms, and an isolated target after every
transition. Camera-to-world source poses are converted to transforms between
camera coordinates. We reconstruct rotations from their first two stored rows
by Gram--Schmidt and compose in SE(3) under
`T_0to2 = T_1to2 @ T_0to1`.

## Frozen representations

We evaluate four preregistered settings without layer selection: DINOv2-L at
the project's L0 output interface, Fast3R at returned feature index L24,
V-JEPA2 ViT-L at encoder block L23, and Wan2.1-T2V-1.3B at transformer block
L20 and scheduler index 749. DINO L0 is an interface index, not transformer
block zero. Every lookup uses either an exact `(scene, ordered eight raw IDs)`
context key or `(scene, exact raw ID)` isolated-image key. Semantic role names
are not lookup keys.

We resize DINO inputs to 728×420 and mean-pool all spatial tokens across the
eight independently encoded frames (one frame for an isolated unit). Fast3R
and V-JEPA2 receive the eight frames followed by tail repetition to 64; Wan
receives tail repetition to 81. An isolated input is repeated to the respective
fixed video length. We mean-pool every returned temporal and spatial token and
store one FP32 vector. Wan uses an empty prompt, one sample, timestep 749 and a
content-derived noise seed that excludes task, semantic role and labels.

## Actionability readouts

Our direct readout applies one shared two-layer candidate scorer
`s(context, anchor, goal, action)` with hidden width 256 and GELU. Candidate
order is swapped with probability 0.5 during training, synchronously with its
label; the physical decision must be equivariant to evaluation-time swapping.
We compare analytic random choice, train-only candidate and temporal priors,
action-only, current+action/no-goal, and goal+action/no-current readouts. A
separately trained recurrent dynamics probe predicts each candidate endpoint;
the dynamics selector chooses the prediction with greatest cosine similarity
to the isolated goal. We also evaluate a privileged observed-endpoint feature
metric using the actual post-action endpoint. This diagnostic is neither
deployable nor a theoretical upper bound. Fixed goal derangements measure goal
dependence without retraining.

Raw-feature cosine is primary. Train-only centering and diagonal
standardization are reported as robustness conditions for every model rather
than selected by validation. Primary outcomes are pairwise accuracy and raw
pose-metric regret; secondary outcomes include scene-macro accuracy,
positive-progress rate, median/P90 regret, and preregistered strata.

## Clean dynamics and algebra diagnostics

The backbone remains frozen. A recurrent probe maps the standardized isolated
anchor to a 256-dimensional state, updates it with a GRUCell receiving each 9D
action, and decodes a residual feature after every step. It is trained from
scratch only on four real primitive transitions, with MSE plus 0.1 times cosine
distance at every waypoint. Matched learned controls replace every action by
identity (visual-only) or every standardized anchor by zero (action-only); an
exact anchor copy is non-trainable. The full model is also evaluated with a
fixed derangement of action sequences within cumulative-motion deciles.
Forward metrics at horizons 1, 2 and 4 are cosine distance, standardized MSE,
and validation-bank retrieval R@1/R@5 after exact `(scene, raw ID)` target
deduplication.

Composition is evaluation-only. We compare four primitives, two composed
transforms, one composed transform, and equal-length composed-plus-identity
sequences with identity at different positions. Each result reports endpoint
correctness, consistency with the four-primitive prediction, and separation
from a magnitude-matched wrong-action prediction. Identity and synthetic
inverse cycles are reported alongside wrong-inverse, repeated-action and exact
copy controls. Observed near-return paths remain separate from synthetic
inverse cycles; their prediction-to-observed-endpoint error and the natural
endpoint-to-anchor difference are both reported. We do not define an aggregate
algebra pass score.

## Optimization and uncertainty

All learned methods use seeds 41, 42 and 43, AdamW, learning rate 0.001, weight
decay 0.0001, batch size 64 and at most 200 epochs. The earliest minimum
scene-macro dev loss selects the epoch, with patience 20, before full-training
refit. We report seed mean ± sample standard deviation separately from 95%
scene-cluster bootstrap intervals using 2,000 replicates and seed 20260913.
Comparisons resample paired records together; loop/non-loop pairs remain in the
same sampled scene cluster.
