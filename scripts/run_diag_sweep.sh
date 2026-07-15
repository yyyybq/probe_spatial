#!/usr/bin/env bash
# Legacy non-streaming sweep training of the four diagnostic probes across one VFM.
#   ALLOW_NON_STREAMING=1 bash scripts/run_diag_sweep.sh wan
#   ALLOW_NON_STREAMING=1 bash scripts/run_diag_sweep.sh cogvideox
#   ALLOW_NON_STREAMING=1 bash scripts/run_diag_sweep.sh vjepa2
set -euo pipefail
ALLOW_NON_STREAMING=${ALLOW_NON_STREAMING:-0}
if [[ "${ALLOW_NON_STREAMING}" != "1" ]]; then
    cat >&2 <<'EOF'
[err] run_diag_sweep.sh is a legacy non-streaming sweep.
      Streaming is the project default. Use:
        bash scripts/run_streaming_probe_sweep.sh
      To intentionally run this old path, set ALLOW_NON_STREAMING=1.
EOF
    exit 2
fi
VFM=${1:-wan}
DEV=${2:-0}
EXTRA=${3:-}

for probe in view_consistency ego_belief action_dynamics abnormal; do
    cfg="inscene15k_ext/${probe}_${VFM}_v1"
    echo "==== ${cfg} ===="
    CUDA_VISIBLE_DEVICES=${DEV} python vidfm3d/train.py \
        experiment=${cfg} \
        ${EXTRA}
done
