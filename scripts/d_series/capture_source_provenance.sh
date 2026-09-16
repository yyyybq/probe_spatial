#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
PROVENANCE_DIR=${PROVENANCE_DIR:?PROVENANCE_DIR is required}
mkdir -p "${PROVENANCE_DIR}"
cd "${PROJECT_ROOT}"

git rev-parse HEAD > "${PROVENANCE_DIR}/base_commit.txt"
if git diff --quiet HEAD -- experiments/d_series_scannetpp_v1 scripts/d_series \
  features/vjepa2/native_predictor.py features/wan/wan_feature.py features/noise_seed.py \
  vidfm3d/models/components/probe_layout_readout.py vidfm3d/utils/layout_metrics.py tests/test_d_series.py; then
  printf 'false\n' > "${PROVENANCE_DIR}/git_dirty.txt"
else
  printf 'true\n' > "${PROVENANCE_DIR}/git_dirty.txt"
fi
git diff --binary HEAD -- experiments/d_series_scannetpp_v1 scripts/d_series \
  features/vjepa2/native_predictor.py features/wan/wan_feature.py features/noise_seed.py \
  vidfm3d/models/components/probe_layout_readout.py vidfm3d/utils/layout_metrics.py \
  tests/test_d_series.py > "${PROVENANCE_DIR}/working_tree.patch"
tar --exclude='*/__pycache__' --exclude='*.pyc' -czf "${PROVENANCE_DIR}/source_snapshot.tar.gz" experiments/d_series_scannetpp_v1/README.md \
  experiments/d_series_scannetpp_v1/STATUS.md experiments/d_series_scannetpp_v1/protocol.yaml \
  experiments/d_series_scannetpp_v1/protocol_revision_20260916.yaml \
  experiments/d_series_scannetpp_v1/models.yaml experiments/d_series_scannetpp_v1/dataset_card.md \
  experiments/d_series_scannetpp_v1/model_card.md experiments/d_series_scannetpp_v1/frame_token_alignment.csv \
  experiments/d_series_scannetpp_v1/v2_reuse_audit.json \
  experiments/d_series_scannetpp_v1/v2_reuse_audit.md \
  experiments/d_series_scannetpp_v1/v2_reuse_mapping.csv \
  results/independent_actionability_algebra_p8_final_v1_20260913/coverage/vjepa2_coverage_receipt.json \
  experiments/d_series_scannetpp_v1/data experiments/d_series_scannetpp_v1/audits scripts/d_series \
  features/vjepa2/native_predictor.py features/wan/wan_feature.py features/noise_seed.py \
  vidfm3d/models/components/probe_layout_readout.py vidfm3d/utils/layout_metrics.py tests/test_d_series.py
sha256sum "${PROVENANCE_DIR}/working_tree.patch" "${PROVENANCE_DIR}/source_snapshot.tar.gz" \
  experiments/d_series_scannetpp_v1/protocol.yaml experiments/d_series_scannetpp_v1/models.yaml \
  experiments/d_series_scannetpp_v1/v2_reuse_audit.json \
  results/independent_actionability_algebra_p8_final_v1_20260913/coverage/vjepa2_coverage_receipt.json \
  > "${PROVENANCE_DIR}/source_hashes.sha256"
