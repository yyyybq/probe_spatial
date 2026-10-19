# Exact feature lineage audit v2

Status: `COMPLETE_WITH_CAVEAT`

- Request units: 49432
- Accepted exact reusable units: 0
- Missing units for independent extraction: 49432
- Scanned sidecars: 54763
- Candidate roots: `/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT`, `/mnt/umm/users/yinbaiqiao/InsScene-15K/FEATURES_VERSIONED`, `/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_STREAMING`, `/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_TARGET`, `/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_STREAMING_CANONICAL_V2`, `/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_TARGET_CANONICAL_V2`, `/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_STREAMING_CANONICAL_V2_SEEDED_V1`, `/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_TARGET_CANONICAL_V2_SEEDED_V1`, `/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1`, `/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_TARGET_CANONICAL_V2_GAP_DELTA_V1`
- Per model: `{'dino': {'requested': 12358, 'reusable': 0, 'missing': 12358}, 'f3r': {'requested': 12358, 'reusable': 0, 'missing': 12358}, 'vjepa2': {'requested': 12358, 'reusable': 0, 'missing': 12358}, 'wan': {'requested': 12358, 'reusable': 0, 'missing': 12358}}`

Only sidecars containing the exact independent raw ID (or the ordered p8 context), model/layer/timestep, extractor and source-manifest provenance, size, dtype, and a matching SHA256 were accepted.  Filename/window/tail similarity and unproved cache entries were rejected.
