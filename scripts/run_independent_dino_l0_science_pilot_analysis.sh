#!/usr/bin/env bash
# Run clean probes only after the exact DINO pilot coverage receipt exists.
set -euo pipefail

root=/mnt/umm/users/yinbaiqiao/probe_spatial
tag=independent_actionability_algebra_p8_20260831T091224Z_28bda217
python_bin=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
manifest_dir="$root/artifacts/$tag/manifests"
request="$manifest_dir/science_pilot_subset_v1_feature_request_v2_dino_l0.jsonl"
feature_root="$root/features/independent_actionability_algebra_p8/$tag/dino_l0_science_pilot_v1"
coverage="$root/artifacts/$tag/execution/dino_l0_science_pilot_v1/coverage_receipt.json"
output="$root/results/independent_actionability_algebra_p8/$tag/dino_l0_science_pilot_v1"

test -x "$python_bin"
test -f "$request"
test -f "$coverage"
test ! -e "$output"
"$python_bin" - "$coverage" <<'PY'
import json
import sys
receipt = json.load(open(sys.argv[1], encoding="utf-8"))
required = {"status": "COMPLETE_VALID", "requested": 6424, "verified": 6424, "missing_or_invalid": 0}
if any(receipt.get(key) != value for key, value in required.items()):
    raise SystemExit(f"coverage precondition failed: {receipt}")
if receipt.get("duplicate_feature_paths") or receipt.get("problems"):
    raise SystemExit(f"coverage precondition failed: {receipt}")
PY
"$python_bin" "$root/scripts/run_independent_pilot_analysis.py" \
  --science-tag "$tag" \
  --manifest-dir "$manifest_dir" \
  --request-manifest "$request" \
  --feature-root "$feature_root" \
  --output-dir "$output" \
  --device cuda
