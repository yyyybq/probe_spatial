#!/usr/bin/env bash
# Package extracted InsScene-15K feature caches as an external dataset bundle.
#
# This script intentionally operates on DATASET_ROOT outside the git checkout.
# It creates one archive per feature-cache directory plus a manifest, README,
# and env.sh that can be uploaded to Hugging Face as a dataset repository.
set -euo pipefail

DATASET_ROOT=${DATASET_ROOT:-${INSCENE_BUNDLE_ROOT:-}}
if [[ -z "${DATASET_ROOT}" ]]; then
    echo "[err] set DATASET_ROOT to the external InsScene-15K root" >&2
    exit 1
fi
if [[ ! -d "${DATASET_ROOT}" ]]; then
    echo "[err] DATASET_ROOT does not exist: ${DATASET_ROOT}" >&2
    exit 1
fi

VFM=${VFM:-wan}
FEATURE_DIRS=${FEATURE_DIRS:-"FEAT FEAT_SHUFFLED FEAT_CONTEXT FEAT_TARGET FEAT_STREAMING"}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-"${DATASET_ROOT}/feature_dataset_bundle/inscene15k_${VFM}_features_${STAMP}"}
COMPRESSION=${COMPRESSION:-auto}
INCLUDE_RAW_DATA=${INCLUDE_RAW_DATA:-0}

mkdir -p "${OUT_DIR}"
manifest="${OUT_DIR}/manifest.tsv"
readme="${OUT_DIR}/README.md"
env_file="${OUT_DIR}/env.sh"

archive_ext() {
    case "$1" in
        zstd) printf 'tar.zst' ;;
        gzip) printf 'tar.gz' ;;
        none) printf 'tar' ;;
        auto)
            if command -v zstd >/dev/null 2>&1; then
                printf 'tar.zst'
            else
                printf 'tar.gz'
            fi
            ;;
        *)
            echo "[err] COMPRESSION must be auto, zstd, gzip, or none" >&2
            exit 1
            ;;
    esac
}

make_archive() {
    local src_name="$1"
    local src_path="${DATASET_ROOT}/${src_name}"
    if [[ ! -d "${src_path}" ]]; then
        echo "[warn] skip missing directory: ${src_path}" >&2
        return 0
    fi

    local ext
    ext="$(archive_ext "${COMPRESSION}")"
    local archive="${OUT_DIR}/${src_name}.${ext}"
    echo "[info] archive ${src_name} -> ${archive}"

    case "${ext}" in
        tar.zst)
            tar --zstd -cf "${archive}" -C "${DATASET_ROOT}" "${src_name}"
            ;;
        tar.gz)
            tar -czf "${archive}" -C "${DATASET_ROOT}" "${src_name}"
            ;;
        tar)
            tar -cf "${archive}" -C "${DATASET_ROOT}" "${src_name}"
            ;;
    esac

    local bytes sha
    bytes="$(stat -c '%s' "${archive}")"
    sha="$(sha256sum "${archive}" | awk '{print $1}')"
    printf '%s\t%s\t%s\n' "$(basename "${archive}")" "${bytes}" "${sha}" >> "${manifest}"
}

{
    printf 'file\tbytes\tsha256\n'
} > "${manifest}"

for dir in ${FEATURE_DIRS}; do
    make_archive "${dir}"
done

if [[ "${INCLUDE_RAW_DATA}" == "1" ]]; then
    make_archive "data"
fi

cat > "${env_file}" <<EOF
# Source this after downloading/extracting the feature dataset bundle.
# Edit BASE if you unpack the archives somewhere else.
BASE=\${BASE:-/data/probe_spatial_data/InsScene-15K}
export INSCENE_DATA_ROOT="\${BASE}/data"
export INSCENE_FEAT_ROOT="\${BASE}/FEAT"
export INSCENE_SHUFFLED_FEAT_ROOT="\${BASE}/FEAT_SHUFFLED"
export INSCENE_CONTEXT_FEAT_ROOT="\${BASE}/FEAT_CONTEXT"
export INSCENE_TARGET_FEAT_ROOT="\${BASE}/FEAT_TARGET"
export INSCENE_STREAMING_FEAT_ROOT="\${BASE}/FEAT_STREAMING"
EOF

cat > "${readme}" <<EOF
# InsScene-15K ${VFM} Feature Caches

This dataset bundle contains pre-extracted frozen-VFM feature caches for
\`probe_spatial\`. It is meant to live outside the git checkout.

## Contents

- \`FEAT/\`: normal clip features for non-streaming A/B probes.
- \`FEAT_SHUFFLED/\`: temporally shuffled features for A3.
- \`FEAT_CONTEXT/\`: causal context-segment inputs for non-streaming C1/C2/C3.
- \`FEAT_TARGET/\`: target-isolated single-frame features for C1/C2/C3 labels.
- \`FEAT_STREAMING/\`: streaming-prefix inputs for streaming A1/A2/B1/B2/C1/C2/C3.

The \`manifest.tsv\` file records archive sizes and SHA256 checksums.

## Restore

\`\`\`bash
mkdir -p /data/probe_spatial_data/InsScene-15K
for f in *.tar.zst; do tar --zstd -xf "\${f}" -C /data/probe_spatial_data/InsScene-15K; done
for f in *.tar.gz; do tar -xzf "\${f}" -C /data/probe_spatial_data/InsScene-15K; done
source env.sh
\`\`\`
EOF

echo "[ok] wrote bundle: ${OUT_DIR}"
echo "[ok] manifest: ${manifest}"
