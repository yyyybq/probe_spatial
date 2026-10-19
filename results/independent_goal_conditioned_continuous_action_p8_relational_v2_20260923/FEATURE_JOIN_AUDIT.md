# Feature and Label Join Audit

Status: **COMPLETE_VALID**.

- DINO, Fast3R, and V-JEPA2 consume their previously accepted independent
  stores, each with 12,358/12,358 verified units and zero missing, duplicate,
  fallback, or raw-ID/contract mismatches.
- Wan L29 uses a restricted request containing exactly 2,560/2,560 verified
  units: 1,280 ordered P8 current contexts and 1,280 isolated h4 goals.  Missing,
  duplicate, fallback, raw-ID/contract mismatch, nonfinite, and constant counts
  are all zero.  All payloads have width 1,536 and share one extraction contract,
  model revision, and snapshot-tree identity.
- Mainline Wan L29 features were not mixed in because their prompt/input contract
  differs from this accepted independent feature contract.
- Consumption is keyed by the complete ordered context raw-ID tuple or the exact
  isolated raw ID.  Roles are not inferred from directory order, requested-role
  labels, or manifest row position.
- The same frozen 803/221/256 trajectories are used by all four models.  Scene
  intersections between fit, dev, and validation are empty.
- Every a1 label was recomputed from anchor and h1 poses; maximum absolute error
  is 2.981e-08.  The registered SE(3) application convention was verified for
  all examples.
- Learned scorers receive no pose, action label, future action sequence, raw
  frame index, scene ID, or trajectory-direction metadata.
