# Four-model shared science pilot (v1)

This is a deterministic selection from frozen manifests and a verbatim subset of feature-request v2; no protocol or source manifest was changed.

- Seed: `independent-p8-science-pilot-v1-20260831`
- Actionability: train 512, val 128; all available scenes are represented before balance fill.
- Algebra: train 512, val 128; 64 paired real-loop/non-loop examples are retained for frozen-dynamics analysis.
- Shared selected sample IDs: 1408
- Exact DINO L0 units needed now: 6424
- Corresponding all-model request units: 25696

## Strata counts

```json
{
  "actionability_train": {
    "records": 512,
    "sample_id_sha256": "5fe46848dc229e690d7af7f8240160cda1f9ed3b1ec173546e8b584a00c3e419",
    "scenes": 289,
    "strata": {
      "label=1|goal=future|motion=rotation": 65,
      "label=1|goal=future|motion=translation": 65,
      "label=1|goal=past|motion=rotation": 65,
      "label=1|goal=past|motion=translation": 65,
      "label=2|goal=future|motion=rotation": 64,
      "label=2|goal=future|motion=translation": 65,
      "label=2|goal=past|motion=rotation": 65,
      "label=2|goal=past|motion=translation": 58
    }
  },
  "actionability_val": {
    "records": 128,
    "sample_id_sha256": "4f719b1c905656651d6bd03a1c7d0a0095d46a759a627983f70dd46481acba21",
    "scenes": 77,
    "strata": {
      "label=1|goal=future|motion=rotation": 16,
      "label=1|goal=future|motion=translation": 14,
      "label=1|goal=past|motion=rotation": 16,
      "label=1|goal=past|motion=translation": 17,
      "label=2|goal=future|motion=rotation": 16,
      "label=2|goal=future|motion=translation": 16,
      "label=2|goal=past|motion=rotation": 16,
      "label=2|goal=past|motion=translation": 17
    }
  },
  "algebra_real_loop_pairs": {
    "pairs": 64,
    "records": 128,
    "scenes": 51
  },
  "algebra_train": {
    "records": 512,
    "sample_id_sha256": "cd2b9f57d2479b7efebe0c44d4d8fcd176b92a9bed6829dcba7e4757ca7ed197",
    "scenes": 444,
    "strata": {
      "path=4|motion_tertile=0": 170,
      "path=4|motion_tertile=1": 171,
      "path=4|motion_tertile=2": 171
    }
  },
  "algebra_val": {
    "records": 128,
    "sample_id_sha256": "0f8003e3701be0e59fcb4f4dad64c7134e8f6a0fdf18b24836a1d06a4f394f24",
    "scenes": 108,
    "strata": {
      "path=4|motion_tertile=0": 42,
      "path=4|motion_tertile=1": 43,
      "path=4|motion_tertile=2": 43
    }
  },
  "shared_sample_id_count": 1408
}
```

## Immutable request receipts

- all models SHA256: `81d81b8a054bb84b6c693b4ed994374020017f676eeac46710d07fd17fd61aaa`
- DINO L0 SHA256: `4295c4c5824140f1732658b671fcf12574ab14f1f96fdc88f14a3edae54796d4`
