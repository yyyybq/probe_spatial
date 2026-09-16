# V-JEPA2 V2 reuse audit

Status: `COMPLETE_VALID`

D-series frozen coverage resolved exactly: **681 scenes, 2,721 windows, 5,442 samples** (4,890 train; 552 dev). Source manifest SHA256: `781e25848de48fa5219aad876bd10cb8f9445328b145a19193a6f6b2ab5efcc9`.

| D-series role | Actual V2 role | Classification | Exact compatible coverage | Missing |
|---|---|---:|---:|---:|
| streaming/context | `streaming_context_p8` | adapted (pooled only) | 0 / 2721 windows | 2721 |
| isolated target | `isolated_frame` | adapted (pooled only) | 425 / 5442 samples | 5017 |
| native target reference | absent | missing | 0 / 5442 samples | 5442 |
| native predictor output | absent | missing | 0 / 5442 samples | 5442 |
| spatial tokens | pooled payload only | incompatible | 0 | all required spatial samples |

The adapter is identity-preserving: it reads safetensors key `feat` as the D-series pooled vector and performs no numeric transform. V2 uses the exact checkpoint revision `b3c1679b7c34d3255ef3547f27c7b226aefab26f`, layer 23, 64-frame repeat-tail/repeat-single inputs, 256 crop, ImageNet normalization, and FP32 mean over all 32x16x16 tokens. It cannot be relabelled as native target or native prediction.

The one-time full V2 coverage receipt verified all **12,358/12,358** V-JEPA2 request units; D-series training should cite that receipt and this mapping rather than rehashing the store. Five deterministic mapped payloads were independently loaded here.
