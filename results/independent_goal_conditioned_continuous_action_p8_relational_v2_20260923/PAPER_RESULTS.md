## Goal-conditioned continuous action output

Candidate ranking does not require a representation to synthesize an action: it
only selects between two actions supplied by the evaluator.  We therefore remove
the candidate set and train a shallow probe to predict the first observed local
motion on a real four-step trajectory.  The probe receives an ordered P8 feature
at the anchor and an isolated feature at the h4 goal, and regresses the canonical
9D action (6D rotation and normalized 3D translation).  We use 803/221/256
scene-disjoint fit/dev/validation trajectories and freeze the final registered
stage of each backbone (DINO project output, Fast3R L24, V-JEPA2 L23, and Wan
L29/t749).  The primary relational probe separately projects current and goal
features and fuses their values, difference, and elementwise product.  All
settings are shared across models; results average five seeds, while confidence
intervals use 5,000 scene-cluster paired bootstrap replicates.

Fast3R gives the strongest direct output, reaching (59.8\%\pm3.4\%\) positive
progress and a mean normalized goal-distance reduction of
(0.025\pm0.007\); DINO and V-JEPA2 each reach (53.8\%\), whereas Wan reaches
(50.2\%\) and has slightly negative mean reduction.  FULL exceeds GOAL_ONLY in
positive progress for all representations by 6.0--13.8 points, with every 95\%
interval above zero.  Thus, the current P8 representation carries information
about the next local motion that is absent from the endpoint image alone.

The decisive controls do not, however, support reliable goal-conditioned action
synthesis.  FULL never exceeds CURRENT_ONLY in positive progress (effects
(-3.6) to (-2.0) points), and its goal-reduction effect is negative for all
four models ((-0.011) to (-0.005)).  Holding each trained checkpoint and
anchor fixed while replacing only the goal changes the predicted direction by
29.8--45.6 degrees, but correct-goal versus shuffled-goal effects on positive
progress ((-2.3) to (+1.2) points) and goal reduction ((-0.006) to
(+0.001)) all include zero.  The probes therefore respond to goal features,
but those responses are not reliably aligned with progress toward the supplied
goal.  A relational inductive bias improves positive progress over concatenation
for DINO and Fast3R, yet only Fast3R obtains a clear improvement in goal
reduction.  This preserves the conclusion suggested by candidate ranking:
frozen features support next-motion and trajectory-statistic readout, but the
present shallow probes do not demonstrate goal-specific continuous control.
In particular, Wan's strong candidate-ranking result does not transfer to action
generation.  These findings concern probes over frozen representations, not a
native policy or world model in the backbone.
