# SeeSE3 replication sanity

Status: **blocked_missing_replication_prerequisites**.  This file is intentionally separate from observed-order feature grounding.

```json
{
  "enough_pairs": false,
  "middle_layer_reason": "DINO C-series artifacts label the only exported last-hidden-state representation as layer 0; no middle-layer C1/C2 artifact was found or substituted.",
  "minimum_eval_pairs": 200,
  "minimum_train_pairs": 1000,
  "multiple_seeds_requested": true,
  "observed_frame_pairs": true,
  "paper_style_temporal_split": true,
  "representative_exported_middle_layer": false,
  "requested_m4_seeds": [
    42,
    43,
    44
  ]
}
```
