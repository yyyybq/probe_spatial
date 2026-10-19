#!/usr/bin/env bash
# Submit LingBot multilayer A1(depth+pose)+mask + A2 + B2 + C1/C2/C3 via ~/submit.sh.
#
# Equivalent to:
#   AEC2_SPLIT=zoetrope,h800 CAMPAIGN=lingbot bash scripts/submit_probe_fill_sco.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
CAMPAIGN=lingbot \
JOB_PREFIX=${JOB_PREFIX:-bq_lingbot_abc} \
AEC2_SPLIT=${AEC2_SPLIT:-zoetrope,h800} \
  bash scripts/submit_probe_fill_sco.sh "${1:-zoetrope}"
