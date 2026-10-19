# Dataset Card — Independent P8 Actionability + Action Algebra v1

The dataset is a deterministic research subset of observed ScanNet++ iPhone
images and camera trajectories. It contains no synthetic branch graph and no
legacy C-series labels, features or checkpoints. Source images remain in the
project's processed ScanNet++ tree; this package stores only raw-ID manifests,
poses, actions, labels and hashes.

## Records

Actionability A0 contains 844 original-training and 227 validation queries from
289 and 77 scenes. Each record has an ordered eight-image context ending at the
anchor, an isolated past or future goal, two observed candidate endpoints, two
relative 9D camera actions and a best-candidate label. The label maximizes
positive progress toward the goal under the frozen pose metric. Thresholds were
computed on the original training split only: minimum goal distance
0.5104432533, maximum local candidate motion 0.3768655756, maximum candidate
motion difference 0.5982680088, and minimum nonzero utility margin
0.09753842086. Both positive progress and a non-near-tie are required.

Clean dynamics contains 1,024 original-training and 256 validation paths from
444 and 108 scenes. Every path lists four real primitive actions, all four
waypoint raw IDs and poses, and exact h1/h2/h4 targets. Maximum primitive motion
0.9103447127 was derived from training only. No algebraic composition,
identity, inverse or real-loop case is used for training.

The loop universe has 824 records: 412 observed near-return paths and 412
same-scene non-loop controls chosen by nearest available path length while
remaining outside four times the loop envelope. All come from validation
scenes; 106 scenes contribute pairs. The loop threshold is pose-motion distance
0.35 after at least eight motion-uniform observations. Pair-level path-length
matching is imperfect: absolute length delta has median 9, P90 23 and maximum
76 frames; the corresponding longer/shorter ratios are 1.277, 3.350 and 10.5.
This residual mismatch is retained and reported rather than hidden.

## Scene splits and development history

The original global split was made by deterministic SHA256 scene ranking before
model evaluation. Across both tasks, training and validation scenes have zero
overlap; Actionability training has zero overlap with algebra validation and
vice versa. Exact duplicate sample IDs and exact raw-ID combinations are zero.
Goals and candidate endpoints never occur inside the corresponding context in
the frozen Actionability records.

The supplementary near-duplicate audit compares task-relevant raw-ID sets only
within the same scene. Actionability has 69/1,015 training and 18/270 validation
pairs at Jaccard 0.833 (and none higher): these are deliberately paired
past/future goals sharing an anchor and context, not duplicate labels. Algebra
has no pairs at Jaccard >=0.8 (four training and three validation pairs fall in
0.5--0.8). Scene-cluster uncertainty therefore remains the primary sampling
interval rather than treating these records as independent scenes.

The previous pilot used validation data and touched every validation scene for
both tasks. Accordingly, this card calls the 227 and 256 records `validation`,
not blind test. A shared train-derived dev partition, recorded in
`manifests/scene_split_final_v1.json`, leaves validation untouched. It has
Actionability 650 fit / 194 dev and algebra 803 fit / 221 dev.

## Actions, poses and scale

Source poses are camera-to-world matrices. The stored 9D action maps anchor
camera coordinates to endpoint camera coordinates using the first two rows of
the relative rotation followed by translation. Algebra first reconstructs a
proper rotation and converts to SE(3); the world-to-camera composition rule is
`T_0to2 = T_1to2 @ T_0to1`. Translation is stored in the physical source-pose
units; no per-query `scene_scale` field or goal-dependent normalization is
present, so the protocol uses implicit scale 1 consistently.

## Feature roles and identity

`streaming_context_p8` means the exact ordered 8-tuple in the record.
`isolated_frame` means only the exact raw image, even when the request aggregates
semantic roles such as anchor, goal, endpoint or horizon. `requested_roles` is
metadata only and is never used to infer a sample-specific lookup. Common sets
are joined by actual sample IDs and raw-ID unit keys.

The immutable feature-request-v2 has 49,432 units (12,358 per model): 10,632
streaming context units and 38,800 isolated units across four models. Its SHA256
is `fb1dbd2d71878edae7eae9034a16ee9809fc0c0cfb67352e0a301ad0edbed369`.
All manifest paths and hashes are in `MANIFEST_HASH_INDEX.json`.

## Filtering and exclusions

From 17,126 raw Actionability candidates, 8,289 passed the frozen eligibility
criteria before balanced/capped selection. Rejections were: candidate motion
mismatch 796, outside local envelope 481, goal too close 2,184, no positive
progress 2,534, and near tie 2,842. From 150,353 primitive paths, 142,993 passed;
7,360 exceeded the local primitive envelope. Selection was deterministic with
a four-record per-scene cap where specified.

## Known biases and permitted interpretation

The data are offline recorded camera trajectories, not interventions in a
branching environment. Candidate endpoints are observed frames, but this does
not establish collision-free straight-line motion between poses. Scene and
trajectory sampling, near-return availability, camera operator behavior and
the fixed local-motion thresholds bias the evaluation. Validation participated
in pilot development, and no untouched scene-disjoint test remains. Results
support only local representation readout and learned-probe analyses under this
protocol.
