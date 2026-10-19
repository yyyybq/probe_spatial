# Feature request v2

Status: `COMPLETE_WITH_CAVEAT`

The original frame-level request manifest is preserved but is `FAILED_VALIDATION` for temporal extraction because it lacks ordered p8 context-unit identity.  It has **not** been edited.  `feature_request_manifest_v2.jsonl` is regenerated directly from the frozen raw-pose manifests and is the only valid input to exact-feature audit/extraction.

- v1 rows: 121748
- v2 units: 49432
- v2 SHA256: `fb1dbd2d71878edae7eae9034a16ee9809fc0c0cfb67352e0a301ad0edbed369`
- Per model: `{'dino': 12358, 'f3r': 12358, 'vjepa2': 12358, 'wan': 12358}`
- Unit kinds: `{'isolated_frame': 38800, 'streaming_context_p8': 10632}`
- Source raw-pose SHA map: `artifacts/independent_actionability_algebra_p8_20260831T091224Z_28bda217/manifests/manifest_summary.json`
