# Tests

Final local verification:

```bash
MPLCONFIGDIR=/tmp/mpl-p1 \
/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python \
  -m pytest -q tests/test_wan_vjepa2_p1.py \
  tests/test_wan_vjepa2_v2_structural.py
```

Result: `7 passed in 16.37s`.

The checks cover independent additive branches, matrix-product composite
actions, identity-padding update counts, separate rotation/translation
geometry, analytic composition and identity invariance, frozen bin boundaries,
and ratio-of-scene-macro normalization.
