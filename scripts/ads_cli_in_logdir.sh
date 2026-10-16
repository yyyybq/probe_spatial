#!/usr/bin/env bash
# Run the AOSS sync client from the calling job's LOG_DIR.
#
# `ads-cli sync` creates its transfer record (syncRecord.txt) in the current
# working directory.  The frozen extractor launcher runs the sync loop with its
# working directory set to the frozen source tree, so without this wrapper every
# job writes into that tree (2026-10-16 canary attempt 3: the record broke the
# next step's frozen-tree check) and concurrent jobs share one record file.
# The sync loop honours ADS_CLI and passes absolute paths, so changing the
# working directory is safe.  LOG_DIR is set by the launcher for each job.
set -euo pipefail
cd "${LOG_DIR:?LOG_DIR must be set by the launcher}"
exec /mnt/umm/users/yinbaiqiao/aoss/ads-cli "$@"
