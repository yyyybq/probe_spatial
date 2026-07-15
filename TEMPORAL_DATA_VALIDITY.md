# Temporal Data Validity Note

Date: 2026-07-15

## Decision

Temporal streaming probes are ScanNet++ only.

Infinigen frames in InsScene-15K are independent rendered camera views, not a
continuous video trajectory. They must not be used as time steps for temporal,
memory, action-conditioned, path-integration, or counterfactual conclusions.

## Historical Results

Any old cache, run, table, report, or checkpoint that used streaming prefixes
over mixed `processed_infinigen + processed_scannetpp_v2` data is historical
only. This includes old default prefix sweeps such as `4,8,16,32,64` and the
visible/hidden B2 sanity runs built on Infinigen-only or mixed streaming caches.

These results may still be useful for debugging code paths, but they are not
valid scientific evidence for temporal object permanence or action dynamics.

## Current Default

Current temporal streaming cache extraction uses:

```bash
python -m features.run_inscene15k \
  --mode streaming_prefix \
  --source scannetpp \
  --prefix-lengths "8,12,16,24" \
  --prefix-max-len 24
```

The feature extraction scripts enforce this default for temporal modes:
`streaming_prefix`, `target_isolated`, `context_segment`, and `streaming_target`
reject `--source all` / `--source infinigen` unless
`ALLOW_INFINIGEN_TEMPORAL=1` is set for explicit legacy/debug reproduction.

The four prefixes are cut from ScanNet++ motion-normalized temporal windows:

- `8`: common visible history
- `12`: history + 4 sampled observations
- `16`: history + 8 sampled observations
- `24`: history + 16 sampled observations

The default observation spacing is `motion_step=0.35` and
`rotation_weight=0.5`, i.e. cumulative camera motion is measured as
`translation + 0.5 * rotation_angle_rad`.

Hidden B1/B2 select one object from the common history and evaluate prefixes
`8/12/16/24`. Selection prefers objects visible at observation `7`, so prefix
`8` acts as a visible-current baseline when possible, but obs7 visibility is
not mandatory. Prefixes `12/16/24` are hidden-current settings: the same object
must be hidden at observation tails `11/15/23`. C probes use the shared
streaming-prefix video cache for inputs, but their targets default to
`target_isolated` features. The isolated target frame ids are derived from the
same ScanNet++ temporal windows for every C prefix. With prefixes `8/12/16/24`
and default horizons `1/2/4`, target positions are the union of
`prefix_tail + horizon`, e.g. `8/9/11` for prefix `8` and `24/25/27` for
prefix `24`.
