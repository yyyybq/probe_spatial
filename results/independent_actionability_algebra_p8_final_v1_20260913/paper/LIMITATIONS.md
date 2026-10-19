# Limitations

- The 227 Actionability and 256 dynamics records are validation, not blind
  test: the earlier pilot touched every held-out scene. No untouched
  scene-disjoint confirmatory subset remains in the frozen universe.
- This is offline local action comparison on recorded trajectories. Observed
  candidate endpoints do not establish that an arbitrary interpolated path is
  collision-free, and the experiment is not online navigation or control.
- The four representations receive different effective temporal inputs. DINO
  independently encodes eight frames; the three video/3D models receive tail
  repetition to fixed lengths. Comparisons cannot isolate pretraining objective
  from architecture, checkpoint, preprocessing or padding.
- The shallow scorer and recurrent transition are task-supervised modules.
  Their behavior is evidence about readout from a frozen representation, not a
  claim that the backbone contains the same native predictor or learned an
  explicit group action during pretraining.
- The immutable feature request registers isolated dynamics targets at h1, h2
  and h4, but not h3. The recurrent probe always consumes the complete ordered
  four-action sequence and is supervised at rollout steps 1, 2 and 4; it does
  not substitute a contextual token slice or an unregistered h3 feature.
- Actionability sampling is balanced and capped by scene after train-derived
  motion/margin filtering. Same-anchor past/future pairs create 69 training and
  18 validation raw-set overlaps at Jaccard 0.833, although there are no exact
  duplicates. Scene-cluster intervals account for within-scene dependence.
- The observed-loop matching control is imperfect. Pairwise path-length
  differences have median 9, P90 23 and maximum 76 frames; longer/shorter path
  ratios have median 1.277, P90 3.350 and maximum 10.5. Loop/non-loop
  differences may therefore retain path-length confounding.
- Low identity or cycle drift is not sufficient evidence of algebraic
  structure: an anchor-copy predictor has exactly that behavior. Interpretation
  requires endpoint correctness, action sensitivity and wrong-action
  separation together, and remains limited to the evaluated action magnitudes
  and sequence lengths.
- Synthetic inverse sequences are mathematical interventions rather than
  observed physical trajectories. Natural near-return endpoints can differ in
  appearance despite close poses, so their feature difference is reported
  separately from model error.
- Four fixed model settings do not support causal attribution to model family
  or pretraining objective, and no layer/timestep sweep is used to generalize
  beyond these registered checkpoints.
