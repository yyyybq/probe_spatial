#!/usr/bin/env bash
# Restore VideoMAE / Fast3R / Wan weights into the local HF hub cache.
# AOSS MODEL_MIRRORS has no copies of these three models.
set -u
HF_HOME="${HF_HOME:-/mnt/umm/users/yinbaiqiao/.cache/huggingface}"
STAGE="${STAGE:-${HF_HOME}/restore_snapshots_20260907}"
LOG="${LOG:-/mnt/umm/users/yinbaiqiao/probe_spatial/logs/hf_restore_20260907/curl_restore.log}"
PARALLEL="${PARALLEL:-2}"
mkdir -p "${STAGE}" "$(dirname "${LOG}")"

fetch() {
  local url="$1" dest="$2" expect="$3"
  mkdir -p "$(dirname "${dest}")"
  if [[ -f "${dest}" && "$(stat -c%s "${dest}")" == "${expect}" ]]; then
    echo "[$(date -u +%FT%TZ)] SKIP ${dest##*/} ${expect}"
    return 0
  fi
  echo "[$(date -u +%FT%TZ)] GET ${url} expect=${expect}"
  if curl -fL --retry 8 --retry-delay 4 --connect-timeout 30 \
      --speed-time 180 --speed-limit 1024 -C - -o "${dest}" "${url}"; then
    if [[ "$(stat -c%s "${dest}")" == "${expect}" ]]; then
      echo "[$(date -u +%FT%TZ)] OK ${dest} ${expect}"
      return 0
    fi
    echo "[$(date -u +%FT%TZ)] SIZE_MISMATCH ${dest} have=$(stat -c%s "${dest}") expect=${expect}"
    return 1
  fi
  echo "[$(date -u +%FT%TZ)] FAIL ${dest} curl=$? have=$(stat -c%s "${dest}" 2>/dev/null || echo 0)"
  return 1
}

install_hub() {
  local repo_dir="$1" rev="$2" src_dir="$3"
  local snap="${repo_dir}/snapshots/${rev}"
  mkdir -p "${repo_dir}/refs" "${snap}"
  printf '%s' "${rev}" > "${repo_dir}/refs/main"
  cp -a "${src_dir}/." "${snap}/"
}

# repo|rev|relpath|bytes
pairs=(
  "MCG-NJU/videomae-large|12da269a02d3e1fbbb7011e610e591fea8061dca|config.json|727"
  "MCG-NJU/videomae-large|12da269a02d3e1fbbb7011e610e591fea8061dca|preprocessor_config.json|271"
  "MCG-NJU/videomae-large|12da269a02d3e1fbbb7011e610e591fea8061dca|model.safetensors|1372055288"
  "jedyang97/Fast3R_ViT_Large_512|a2c770b768ceb3a53c36c4f7a3619db0413dc3a1|config.json|941"
  "jedyang97/Fast3R_ViT_Large_512|a2c770b768ceb3a53c36c4f7a3619db0413dc3a1|model.safetensors|2590286416"
  "Wan-AI/Wan2.1-T2V-1.3B-Diffusers|0fad780a534b6463e45facd96134c9f345acfa5b|model_index.json|400"
  "Wan-AI/Wan2.1-T2V-1.3B-Diffusers|0fad780a534b6463e45facd96134c9f345acfa5b|scheduler/scheduler_config.json|751"
  "Wan-AI/Wan2.1-T2V-1.3B-Diffusers|0fad780a534b6463e45facd96134c9f345acfa5b|tokenizer/tokenizer.json|16837459"
  "Wan-AI/Wan2.1-T2V-1.3B-Diffusers|0fad780a534b6463e45facd96134c9f345acfa5b|tokenizer/spiece.model|4548313"
  "Wan-AI/Wan2.1-T2V-1.3B-Diffusers|0fad780a534b6463e45facd96134c9f345acfa5b|tokenizer/tokenizer_config.json|61758"
  "Wan-AI/Wan2.1-T2V-1.3B-Diffusers|0fad780a534b6463e45facd96134c9f345acfa5b|tokenizer/special_tokens_map.json|7079"
  "Wan-AI/Wan2.1-T2V-1.3B-Diffusers|0fad780a534b6463e45facd96134c9f345acfa5b|text_encoder/config.json|854"
  "Wan-AI/Wan2.1-T2V-1.3B-Diffusers|0fad780a534b6463e45facd96134c9f345acfa5b|text_encoder/model.safetensors.index.json|22476"
  "Wan-AI/Wan2.1-T2V-1.3B-Diffusers|0fad780a534b6463e45facd96134c9f345acfa5b|text_encoder/model-00001-of-00005.safetensors|4972389712"
  "Wan-AI/Wan2.1-T2V-1.3B-Diffusers|0fad780a534b6463e45facd96134c9f345acfa5b|text_encoder/model-00002-of-00005.safetensors|4899225672"
  "Wan-AI/Wan2.1-T2V-1.3B-Diffusers|0fad780a534b6463e45facd96134c9f345acfa5b|text_encoder/model-00003-of-00005.safetensors|4966309504"
  "Wan-AI/Wan2.1-T2V-1.3B-Diffusers|0fad780a534b6463e45facd96134c9f345acfa5b|text_encoder/model-00004-of-00005.safetensors|4999880704"
  "Wan-AI/Wan2.1-T2V-1.3B-Diffusers|0fad780a534b6463e45facd96134c9f345acfa5b|text_encoder/model-00005-of-00005.safetensors|2885866152"
  "Wan-AI/Wan2.1-T2V-1.3B-Diffusers|0fad780a534b6463e45facd96134c9f345acfa5b|transformer/config.json|465"
  "Wan-AI/Wan2.1-T2V-1.3B-Diffusers|0fad780a534b6463e45facd96134c9f345acfa5b|transformer/diffusion_pytorch_model.safetensors.index.json|73296"
  "Wan-AI/Wan2.1-T2V-1.3B-Diffusers|0fad780a534b6463e45facd96134c9f345acfa5b|transformer/diffusion_pytorch_model-00001-of-00002.safetensors|4998781576"
  "Wan-AI/Wan2.1-T2V-1.3B-Diffusers|0fad780a534b6463e45facd96134c9f345acfa5b|transformer/diffusion_pytorch_model-00002-of-00002.safetensors|677289072"
  "Wan-AI/Wan2.1-T2V-1.3B-Diffusers|0fad780a534b6463e45facd96134c9f345acfa5b|vae/config.json|724"
  "Wan-AI/Wan2.1-T2V-1.3B-Diffusers|0fad780a534b6463e45facd96134c9f345acfa5b|vae/diffusion_pytorch_model.safetensors|507591892"
)

echo "[$(date -u +%FT%TZ)] curl restore start" | tee -a "${LOG}"
round=0
while true; do
  incomplete=()
  for pair in "${pairs[@]}"; do
    IFS='|' read -r repo rev rel expect <<<"${pair}"
    dest="${STAGE}/${repo}/${rev}/${rel}"
    [[ -f "${dest}" && "$(stat -c%s "${dest}")" == "${expect}" ]] || incomplete+=("${pair}")
  done
  if [[ ${#incomplete[@]} -eq 0 ]]; then
    echo "[$(date -u +%FT%TZ)] downloads complete" | tee -a "${LOG}"
    break
  fi
  round=$((round + 1))
  echo "[$(date -u +%FT%TZ)] round=${round} remaining=${#incomplete[@]}" | tee -a "${LOG}"
  pids=()
  started=0
  for pair in "${incomplete[@]}"; do
    IFS='|' read -r repo rev rel expect <<<"${pair}"
    dest="${STAGE}/${repo}/${rev}/${rel}"
    url="https://huggingface.co/${repo}/resolve/${rev}/${rel}"
    fetch "${url}" "${dest}" "${expect}" >>"${LOG}" 2>&1 &
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

install_hub "${HF_HOME}/hub/models--MCG-NJU--videomae-large" \
  12da269a02d3e1fbbb7011e610e591fea8061dca \
  "${STAGE}/MCG-NJU/videomae-large/12da269a02d3e1fbbb7011e610e591fea8061dca"
install_hub "${HF_HOME}/hub/models--jedyang97--Fast3R_ViT_Large_512" \
  a2c770b768ceb3a53c36c4f7a3619db0413dc3a1 \
  "${STAGE}/jedyang97/Fast3R_ViT_Large_512/a2c770b768ceb3a53c36c4f7a3619db0413dc3a1"
install_hub "${HF_HOME}/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers" \
  0fad780a534b6463e45facd96134c9f345acfa5b \
  "${STAGE}/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/0fad780a534b6463e45facd96134c9f345acfa5b"

echo "[$(date -u +%FT%TZ)] installed into HF hub cache" | tee -a "${LOG}"
