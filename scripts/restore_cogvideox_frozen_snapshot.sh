#!/usr/bin/env bash
# Restore the frozen CogVideoX-5b-I2V snapshot used by seeded-v1 extract.
# Resumes partial files; retries curl 18 / proxy drops until sizes match.
set -u
SNAP="${COGX_SNAPSHOT:-/mnt/umm/users/yinbaiqiao/.cache/huggingface/hub/models--THUDM--CogVideoX-5b-I2V/snapshots/a6f0f4858a8395e7429d82493864ce92bf73af11}"
ROOT="$(dirname "$(dirname "${SNAP}")")"
REV=a6f0f4858a8395e7429d82493864ce92bf73af11
BASE="${COGX_SNAPSHOT_BASE:-https://hf-mirror.com/zai-org/CogVideoX-5b-I2V/resolve/${REV}}"
PARALLEL="${COGX_RESTORE_PARALLEL:-2}"
LOG="${COGX_RESTORE_LOG:-/mnt/umm/users/yinbaiqiao/probe_spatial/logs/tstep_canonical_v2_seeded_v1/restore_cogvideox_snapshot.log}"

pairs=(
  ".gitattributes:1519"
  "LICENSE:5704"
  "README.md:12139"
  "README_zh.md:11212"
  "configuration.json:47"
  "model_index.json:423"
  "scheduler/scheduler_config.json:482"
  "text_encoder/config.json:782"
  "text_encoder/model-00001-of-00002.safetensors:4994582224"
  "text_encoder/model-00002-of-00002.safetensors:4530066360"
  "text_encoder/model.safetensors.index.json:19885"
  "tokenizer/added_tokens.json:2593"
  "tokenizer/special_tokens_map.json:2543"
  "tokenizer/spiece.model:791656"
  "tokenizer/tokenizer_config.json:20617"
  "transformer/config.json:802"
  "transformer/diffusion_pytorch_model-00001-of-00003.safetensors:4992465072"
  "transformer/diffusion_pytorch_model-00002-of-00003.safetensors:4985800640"
  "transformer/diffusion_pytorch_model-00003-of-00003.safetensors:1272025856"
  "transformer/diffusion_pytorch_model.safetensors.index.json:103161"
  "vae/config.json:872"
  "vae/diffusion_pytorch_model.safetensors:862388596"
)

mkdir -p "${SNAP}" "${ROOT}/refs" "$(dirname "${LOG}")"
printf '%s\n' "${REV}" > "${ROOT}/refs/main"

file_ok() {
  local dest="$1" expect="$2"
  [[ -f "${dest}" && "$(stat -c%s "${dest}")" == "${expect}" ]]
}

fetch_one() {
  local rel="$1" expect="$2"
  local dest="${SNAP}/${rel}"
  mkdir -p "$(dirname "${dest}")"
  if file_ok "${dest}" "${expect}"; then
    echo "[$(date -u +%FT%TZ)] SKIP ${rel}"
    return 0
  fi
  local have=0
  [[ -f "${dest}" ]] && have="$(stat -c%s "${dest}")"
  echo "[$(date -u +%FT%TZ)] GET ${rel} have=${have} expect=${expect}"
  if curl -fL --retry 5 --retry-delay 3 --connect-timeout 30 --speed-time 120 --speed-limit 1024 \
      -C - -o "${dest}" "${BASE}/${rel}"; then
    if file_ok "${dest}" "${expect}"; then
      echo "[$(date -u +%FT%TZ)] OK ${rel} ${expect}"
      return 0
    fi
    echo "[$(date -u +%FT%TZ)] SIZE_MISMATCH ${rel} have=$(stat -c%s "${dest}") expect=${expect}"
    return 1
  fi
  echo "[$(date -u +%FT%TZ)] FAIL ${rel} curl=$? have=$(stat -c%s "${dest}" 2>/dev/null || echo 0)"
  return 1
}

echo "[$(date -u +%FT%TZ)] restore start snap=${SNAP} base=${BASE} parallel=${PARALLEL}" | tee -a "${LOG}"
round=0
while true; do
  incomplete=()
  for pair in "${pairs[@]}"; do
    rel="${pair%%:*}"
    expect="${pair##*:}"
    file_ok "${SNAP}/${rel}" "${expect}" || incomplete+=("${pair}")
  done
  if [[ ${#incomplete[@]} -eq 0 ]]; then
    echo "[$(date -u +%FT%TZ)] restore complete $(du -sh "${ROOT}" | awk '{print $1}')" | tee -a "${LOG}"
    exit 0
  fi
  round=$((round + 1))
  echo "[$(date -u +%FT%TZ)] round=${round} remaining=${#incomplete[@]}" | tee -a "${LOG}"
  pids=()
  started=0
  for pair in "${incomplete[@]}"; do
    rel="${pair%%:*}"
    expect="${pair##*:}"
    fetch_one "${rel}" "${expect}" >>"${LOG}" 2>&1 &
    pids+=("$!")
    started=$((started + 1))
    if [[ ${started} -ge ${PARALLEL} ]]; then
      break
    fi
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" || true
  done
done
