#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
STAGE=${STAGE:?STAGE must be d1 or d2d3}
RUN_TAG=${RUN_TAG:?RUN_TAG is required}
FEATURE_ROOT=${FEATURE_ROOT:-/mnt/umm/users/yinbaiqiao/InsScene-15K/D_SERIES_SCANNETPP_V1_FEATURES}
FORMAL_RUN_ROOT=${FORMAL_RUN_ROOT:-${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/formal_runs}
PY=${PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}
REPORT=${PROJECT_ROOT}/experiments/d_series_scannetpp_v1/runs/${RUN_TAG}/${STAGE}
SCENE_ROOT=/mnt/umm/users/yinbaiqiao/InsScene-15K/data/processed_scannetpp_v2/processed_scannetpp_v2

mkdir -p "${REPORT}"
date -u +%FT%TZ > "${REPORT}/started_utc.txt"
nvidia-smi > "${REPORT}/nvidia_smi.txt"
git -c safe.directory="${PROJECT_ROOT}" -C "${PROJECT_ROOT}" rev-parse HEAD > "${REPORT}/base_commit.txt"
git -c safe.directory="${PROJECT_ROOT}" -C "${PROJECT_ROOT}" diff --binary HEAD -- \
  experiments/d_series_scannetpp_v1 scripts/d_series features/vjepa2/native_predictor.py \
  features/wan/wan_feature.py features/noise_seed.py vidfm3d/models/components/probe_layout_readout.py \
  vidfm3d/utils/layout_metrics.py tests/test_d_series.py > "${REPORT}/working_tree.patch"
tar -czf "${REPORT}/source_snapshot.tar.gz" experiments/d_series_scannetpp_v1/README.md \
  experiments/d_series_scannetpp_v1/STATUS.md experiments/d_series_scannetpp_v1/protocol.yaml \
  experiments/d_series_scannetpp_v1/models.yaml experiments/d_series_scannetpp_v1/dataset_card.md \
  experiments/d_series_scannetpp_v1/model_card.md experiments/d_series_scannetpp_v1/frame_token_alignment.csv \
  experiments/d_series_scannetpp_v1/data experiments/d_series_scannetpp_v1/audits scripts/d_series \
  features/vjepa2/native_predictor.py features/wan/wan_feature.py features/noise_seed.py \
  vidfm3d/models/components/probe_layout_readout.py vidfm3d/utils/layout_metrics.py tests/test_d_series.py
cd "${PROJECT_ROOT}"

"${PY}" scripts/d_series/audit_protocol.py --feature-root "${FEATURE_ROOT}" --run-root "${FORMAL_RUN_ROOT}"
"${PY}" - experiments/d_series_scannetpp_v1/audits/protocol_audit.json <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
assert report["feature_status"], "feature audit is empty"
failed = {key: value for key, value in report["feature_status"].items() if value["complete"] != value["expected"]}
assert not failed, failed
PY

"${PY}" scripts/d_series/run_training_campaign.py --stage "${STAGE}" --feature-root "${FEATURE_ROOT}" \
  --run-root "${FORMAL_RUN_ROOT}" --scene-root "${SCENE_ROOT}" --report-dir "${REPORT}"
