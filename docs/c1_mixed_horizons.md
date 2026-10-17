# C1 mixed canonical horizons and the C2 comparison

The new `*_h124_v1` C1 experiments train on approximately equal numbers of
canonical h1, h2, and h4 targets, keeping the original training sample count.
Existing `*_v1` configurations remain the
historical h1-only experiments. Frozen production queues/runtime snapshots
are not modified by this change.

## Data and training

After manifest loading, split selection, and streaming-prefix owner filtering,
training prefixes are randomly partitioned into three horizon groups using the
explicit dataset seed, or the experiment's torch seed when no dataset seed is
specified. Every prefix is used exactly once per dataset pass, with one selected
target. Group sizes differ by at most one (exactly one third each when the prefix
count is divisible by three). The partition stays fixed across epochs and worker
access orders; the normal DataLoader still shuffles the examples each epoch.
Each training example is one of:

```text
(prefix, anchor→h1 action) → isolated h1 feature
(prefix, anchor→h2 action) → isolated h2 feature
(prefix, anchor→h4 action) → isolated h4 feature
```

Targets remain spatially pooled frozen features. The existing C1 Transformer,
action encoding, MSE + cosine loss, optimizer, and schedule are unchanged.
Neither other future targets nor horizon labels are supplied to the predictor.
Validation/test still expand every prefix deterministically into all three
horizons, so per-horizon metrics cover the same complete evaluation prefix set.

An incomplete canonical window or missing mixed-horizon target raises an error;
the loader does not silently replace h2/h4 with h1 or drop those samples. The
existing feature owner/lineage checks still apply. Actual h1/h2/h4 target-cache
availability must be verified for the selected model/layer/feature owner before
launching a full run.

`manifest_sample_id` retains the parent prefix ID; `sample_id` contains the
selected raw target ID. Two separate scalar metadata fields avoid conflating
observation offsets with raw frame offsets:

```text
action_horizon:   1, 2, 4
raw_frame_offset: 3, 4, 12   # one real canonical-window example
```

Training dataset length equals the corresponding h1-only prefix count N. With
the same epochs, batch size, sampler, and drop-last settings, the number of
training batches and optimizer steps is unchanged. Drop-last or distributed
sampler truncation can omit a few examples exactly as before; the 1/3 allocation
describes the full training dataset. Validation/test contain 3N_eval examples;
this increases evaluation work only. No claim of equal wall-clock runtime is
made because target loading and validation costs may differ.

## Experiment selection

There are ten new configs: Aether, CogVideoX, DINO, F3R, LingBot, MiniMax-H3,
VideoMAE, V-JEPA2, Wan, and direct VLM. For example, using the current workspace
and already validated canonical data/owner environment:

```bash
python vidfm3d/train.py \
  experiment=inscene15k_streaming/action_dynamics_dino_h124_v1 \
  prefix_len=8
```

For direct VLM use `direct_vlm_action_dynamics_h124_v1`. The configs inherit the
matching h1 baseline and override `action_horizons`, `job_name`, and
`autoresume=false`. They therefore have separate default output identities;
do not override them back onto an h1 run directory. A new run must use code
containing the mixed-target loader, not an older frozen runtime.

## Evaluation

Use `vidfm3d/eval_diag.py` with the same mixed experiment and its checkpoint.
Per-sample dumps retain both horizon fields. Summaries include:

- Historical pooled C1 metrics over the combined candidate pool.
- `h1_*`, `h2_*`, `h4_*` metrics, each using that horizon's candidate pool.
- `macro_horizon_mean_cos`, `macro_horizon_global_R@1`,
  `macro_horizon_global_R@5`, and `macro_horizon_action_gain_cos`.
- `h1_n`, `h2_n`, `h4_n`, and `n_horizons` for coverage checks.

Macro values give equal weight to each observed canonical horizon. Full formal
evaluation should contain all three. Global mixed-pool R@1 and per-horizon R@1
use different candidate sets and should not be conflated. The existing batch
action-roll baseline is retained; it is a sensitivity diagnostic and does not
constitute a matched-horizon counterfactual control. The current C3 protocol is
evaluation-only: `evaluation_protocol=c3_action_sensitivity` groups these C1
one-step rows by anchor and reports action sensitivity. Mixed h1/h2/h4 rows are
still one observed trajectory, not executed alternative branches. Use a frozen
`C1ActionSetDataset` with restored same-anchor branches (and optional action
equivalence labels) for causal/intervention claims; see `protocols/C_PROBES.md`.

## What current C2 trains

C2 consumes one frozen prefix tensor `[B,T,H,W,C]`. It averages over time and
space, projects that vector to hidden dimension 512, and initializes both GRU
layers with the projected state. For default horizons `[1,2,4]`, it applies the
same two-layer GRU updater three times:

```text
s0 = initialize(prefix)
s1 = update(s0, action(anchor→h1)); decode(s1) → target h1
s2 = update(s1, action(h1→h2));     decode(s2) → target h2
s3 = update(s2, action(h2→h4));     decode(s3) → target h4
```

Each action is GT relative camera motion, encoded as rotation 6D + translation
3D. The final decoded feature has the VFM's channel dimension. The three
isolated GT features are spatially pooled and used only as supervision.
For valid predictions, training minimizes mean feature MSE plus mean
`1 - cosine_similarity`, with default weights 1 and 1. Gradients propagate
through all recurrent updates. There is no future-visual teacher forcing, no
feedback of decoded predictions as inputs, and no pose loss in C2 training.
The recurrent hidden state, rather than the decoded feature, carries history.

Thus h4 receives three recurrent updates, not four: the last action jumps
from h2 to h4. The targets are three selected waypoints along one observed
trajectory, not alternative branches. Mixed C1 supplies a direct endpoint
baseline at the same h1/h2/h4 endpoints. Differences still also include the
head architecture and temporal pooling (C1 retains temporal tokens; C2 pools
them), so this is not a perfectly architecture-matched recurrence ablation.

This implementation does not change C2 training or its previously identified
horizon/pose-metric issues.
