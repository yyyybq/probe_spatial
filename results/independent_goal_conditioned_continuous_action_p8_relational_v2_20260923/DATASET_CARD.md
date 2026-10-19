# Dataset Card — Continuous Action Relational Subset V2

This experiment reuses the frozen, independently constructed Algebra
trajectories; it does not create a new raw-pose universe or modify examples.
There are 1,280 real four-step trajectories from 552 ScanNet++ scenes: 803
fit examples from 354 scenes, 221 dev examples from 90 scenes, and 256
validation examples from 108 scenes.  Scene sets are disjoint.  The validation
set participated in the broader research programme and is therefore called
validation, not a new blind test.

Each example has an ordered eight-frame context ending at the anchor, four
observed primitive actions, and observed waypoints h1–h4.  The learned input is
only the exact current P8 streaming feature and exact isolated h4 feature.  The
label is the actually executed first primitive action a1 in the existing
canonical 9D convention; it is not the straight-line anchor-to-h4 displacement.
Pose metadata is used to reconstruct and verify labels and to compute evaluation
utility, never as a learned input.

All 1,280 a1 labels were recomputed from anchor and h1 poses.  The maximum
absolute difference from the stored 9D label was 2.981e-08 at an acceptance
tolerance of 1e-07.  Applying recomputed a1 under the registered SE(3)
convention reproduces h1.  Exact feature lookup uses the full ordered P8 raw-ID
tuple for current and the exact raw ID for h4.
