# Strict B2 real smoke

Status: **NOT RUN — prerequisite gate 4 failed.**

The required 1–5 scene, real-data, DINO prefix-8 strict smoke was intentionally not entered. No own scheduler allocation was available: the only local SCO profile resolves the prior created gate job to scheduler owner `guchenyang`, which cannot be used under this run's ownership constraint.

Consequently, none of the following is claimed:

- `n_valid > 0` on real data;
- target raw pixel count equals zero for every accepted sample;
- target-mask mutation leaves the past-only centroid invariant;
- past-visible-mask mutation changes the centroid;
- target depth/object pixels are excluded from centroid construction;
- VFM-independent GT identity;
- real DINO p8 streaming feature readability;
- real target-isolated feature readability;
- real DataLoader sample success.

No strict manifest was generated in this gate tag, no smoke train/eval was run, and no synthetic test was presented as a real experiment. The strict implementation's existing unit tests did pass as part of the repository check, but that is not a substitute for this gate.

