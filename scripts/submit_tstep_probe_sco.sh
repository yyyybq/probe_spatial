#!/usr/bin/env bash
# Build the canonical-v2 gated tstep-probe matrix and submit one 8-GPU SCO job
# per group. Safe default: prepare and validate the complete submit plan only.
set -euo pipefail

export PATH="${HOME}/.sco/bin:${PATH}"
PROJECT_ROOT=${PROJECT_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial}
cd "${PROJECT_ROOT}"

aec2_name=${1:-zoetrope}
PYTHON=${PYTHON:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}
RUN_ID=${RUN_ID:-tstep_probe_seeded_v1_$(date -u +%Y%m%d_%H%M)}
OUT_DIR=${OUT_DIR:-${PROJECT_ROOT}/logs/tstep_probe_campaigns/${RUN_ID}}
PROTOCOL_SPEC=${PROTOCOL_SPEC:-protocols/inscene_streaming_canonical_v2.json}
PROTOCOL_ID=${PROTOCOL_ID:-inscene_streaming_canonical_v2_obs28_h124}
CANONICAL_OBS_LEN=${CANONICAL_OBS_LEN:-28}
TARGET_HORIZONS=${TARGET_HORIZONS:-"1 2 4"}
SOURCE=${SOURCE:-scannetpp}
DONE_INDEX_DIR=${DONE_INDEX_DIR:-logs/tstep_canonical_v2_seeded_v1/aoss_done}
MANIFEST_DIR=${MANIFEST_DIR:-artifacts/tstep_canonical_v2/manifests}
STRICT_AUDIT_DIR=${STRICT_AUDIT_DIR:-artifacts/tstep_canonical_v2_seeded_v1/audits}
STREAMING_NAMESPACE=${STREAMING_NAMESPACE:-FEAT_STREAMING_CANONICAL_V2_SEEDED_V1}
TARGET_NAMESPACE=${TARGET_NAMESPACE:-FEAT_TARGET_CANONICAL_V2_SEEDED_V1}
STREAMING_INDEX_ROOT=${STREAMING_INDEX_ROOT:-/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_STREAMING_CANONICAL_V2_SEEDED_V1}
TARGET_INDEX_ROOT=${TARGET_INDEX_ROOT:-/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_TARGET_CANONICAL_V2_SEEDED_V1}
STREAMING_EXCLUSIONS=${STREAMING_EXCLUSIONS:-configs/experiment/inscene15k_streaming/tstep_canonical_v2_exclusions.txt}
VFMS=${VFMS:-"wan cogvideox minimax_h3"}
LAYERS=${LAYERS:-"15 20 25"}
PREFIXES=${PREFIXES:-"8 12 16 24"}
REFRESH_AOSS_INDEX=${REFRESH_AOSS_INDEX:-1}
STATUS_POLICY=${STATUS_POLICY:-full}
GROUP_REGEX=${GROUP_REGEX:-.*}
MAX_GROUPS=${MAX_GROUPS:-0}
DRY_SUBMIT=${DRY_SUBMIT:-1}
JOB_PREFIX=${JOB_PREFIX:-bq_tp_sv1}
NPROC=${NPROC:-8}
BATCH_SIZE=${BATCH_SIZE:-2}
BATCH_SIZE_VAL=${BATCH_SIZE_VAL:-2}
CACHE_MAX_GIB=${CACHE_MAX_GIB:-900}
CACHE_TARGET_GIB=${CACHE_TARGET_GIB:-700}

EXPECTED_STREAMING_UNITS=10884
EXPECTED_TARGET_UNITS=681

container_image_url="registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617"
storage_mount="019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm"
[[ "${aec2_name}" == "zoetrope" ]] || {
  echo "[err] seeded-v1 probing must use the Zoetrope pool; got aec2_name=${aec2_name}" >&2
  exit 2
}
worker_spec="N4lS.Iq.I80."

required_rel=(
  vidfm3d/train.py
  vidfm3d/data/components/inscene15k_dataset.py
  vidfm3d/utils/aoss_fs.py
  scripts/build_tstep_aoss_done_index.py
  scripts/build_tstep_canonical_manifests.py
  scripts/build_tstep_probe_matrix.py
  scripts/audit_tstep_canonical_alignment.py
  scripts/prune_tstep_probe_cache.py
  scripts/run_tstep_probe_group.sh
  protocols/inscene_streaming_canonical_v2.json
  artifacts/tstep_canonical_v2/manifests/canonical_summary.json
  artifacts/tstep_canonical_v2/manifests/inscene_c_h124_prefix8.jsonl
  artifacts/tstep_canonical_v2/manifests/inscene_c_h124_prefix8.jsonl.meta.json
  artifacts/tstep_canonical_v2/manifests/inscene_c_h124_prefix12.jsonl
  artifacts/tstep_canonical_v2/manifests/inscene_c_h124_prefix12.jsonl.meta.json
  artifacts/tstep_canonical_v2/manifests/inscene_c_h124_prefix16.jsonl
  artifacts/tstep_canonical_v2/manifests/inscene_c_h124_prefix16.jsonl.meta.json
  artifacts/tstep_canonical_v2/manifests/inscene_c_h124_prefix24.jsonl
  artifacts/tstep_canonical_v2/manifests/inscene_c_h124_prefix24.jsonl.meta.json
  configs/experiment/inscene15k_streaming/tstep_canonical_v2_exclusions.txt
  reports/tstep_canonical_v2/INVALID_R1_20260830.md
  logs/feat_extract_tstep_campaign_r7_20260828/INVALID
  logs/tstep_probe_campaigns/tstep_probe_wan_r1_20260828/INVALID
)
for rel in "${required_rel[@]}"; do
  [[ -s "${PROJECT_ROOT}/${rel}" ]] || { echo "[err] missing/empty ${rel}" >&2; exit 3; }
done

resolve_under_project() {
  local value=$1
  if [[ "${value}" == /* ]]; then
    printf '%s\n' "${value}"
  else
    printf '%s\n' "${PROJECT_ROOT}/${value}"
  fi
}

protocol_spec_abs=$(resolve_under_project "${PROTOCOL_SPEC}")
manifest_dir_abs=$(resolve_under_project "${MANIFEST_DIR}")
strict_audit_dir_abs=$(resolve_under_project "${STRICT_AUDIT_DIR}")
streaming_exclusions_abs=$(resolve_under_project "${STREAMING_EXCLUSIONS}")
[[ -s "${protocol_spec_abs}" ]] || { echo "[err] missing/empty protocol spec: ${protocol_spec_abs}" >&2; exit 3; }
[[ -d "${manifest_dir_abs}" ]] || { echo "[err] missing manifest dir: ${manifest_dir_abs}" >&2; exit 3; }
[[ -f "${streaming_exclusions_abs}" ]] || { echo "[err] missing v2 exclusions: ${streaming_exclusions_abs}" >&2; exit 3; }

if [[ "${PROTOCOL_ID}" != "inscene_streaming_canonical_v2_obs28_h124" ||
      "${CANONICAL_OBS_LEN}" != "28" ||
      "${TARGET_HORIZONS}" != "1 2 4" ||
      "${SOURCE}" != "scannetpp" ||
      "${NPROC}" != "8" ]]; then
  echo "[err] canonical-v2 protocol constants may not be downgraded or overridden" >&2
  exit 3
fi
if [[ "${STREAMING_NAMESPACE}" != "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1" ||
      "${TARGET_NAMESPACE}" != "FEAT_TARGET_CANONICAL_V2_SEEDED_V1" ]]; then
  echo "[err] only canonical-v2 seeded-v1 feature namespaces are accepted" >&2
  exit 3
fi
if [[ "$(basename "${STREAMING_INDEX_ROOT%/}")" != "${STREAMING_NAMESPACE}" ||
      "$(basename "${TARGET_INDEX_ROOT%/}")" != "${TARGET_NAMESPACE}" ]]; then
  echo "[err] local semantic-index roots must end in their canonical-v2 namespace" >&2
  exit 3
fi
case "${REFRESH_AOSS_INDEX}:${DRY_SUBMIT}" in
  0:0|0:1|1:0|1:1) ;;
  *) echo "[err] REFRESH_AOSS_INDEX and DRY_SUBMIT must each be 0 or 1" >&2; exit 3 ;;
esac

# The v2 protocol is immutable at submit time. Validate the spec itself and
# make sure the v2 exclusion file contains no active legacy unit.
"${PYTHON}" - "${protocol_spec_abs}" "${streaming_exclusions_abs}" <<'PY'
import json
import sys
from pathlib import Path

spec_path, exclusions_path = map(Path, sys.argv[1:])
spec = json.loads(spec_path.read_text(encoding="utf-8"))
expected_noise = {
    "contract": "inscene_deterministic_noise_v1",
    "wan_cogvideox_seed_scheme": "inscene_noise_seed_v1",
    "seed_payload": "canonical_json_utf8_sorted_compact({noise_seed_scheme,identity})",
    "seed_hash": "sha256_first_8_bytes_big_endian_mask_u63",
    "seed_excludes": ["timestep", "layer", "process_rank", "sample_order"],
    "streaming_identity": [
        "canonical_protocol", "canonical_protocol_sha256", "vfm", "model_id",
        "scene_id", "sample_kind=streaming_prefix", "window_id", "prefix_len",
    ],
    "target_identity": [
        "canonical_protocol", "canonical_protocol_sha256", "vfm", "model_id",
        "scene_id", "sample_kind=target_isolated", "frame_index",
    ],
    "generator_scope": "fresh_torch_generator_per_logical_sample_on_model_device",
    "wan_cogvideox_require_explicit_generator": True,
    "minimax_h3_seed_scheme": "minimax_h3_fixed_seed_v1",
    "minimax_h3_internal_seed": 0,
    "minimax_h3_seed_scope": "reset_per_forward",
}
expected = {
    "canonical_protocol": "inscene_streaming_canonical_v2_obs28_h124",
    "canonical_protocol_sha256": "215c720c8537f4dfae6243bedcc7339fbb7e3ba8a6ca2dcaf04f49cd4b738e4f",
    "source": "scannetpp",
}
for key, value in expected.items():
    if spec.get(key) != value:
        raise SystemExit(f"protocol spec {key}={spec.get(key)!r}, expected {value!r}")
checks = [
    (spec.get("canonical_window", {}).get("observations"), 28, "observations"),
    (spec.get("streaming", {}).get("prefix_lengths"), [8, 12, 16, 24], "prefixes"),
    (spec.get("target", {}).get("future_horizons"), [1, 2, 4], "horizons"),
    (spec.get("target", {}).get("include_anchors"), True, "include_anchors"),
    (
        spec.get("feature_namespaces", {}).get("streaming"),
        "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1",
        "streaming namespace",
    ),
    (
        spec.get("feature_namespaces", {}).get("target"),
        "FEAT_TARGET_CANONICAL_V2_SEEDED_V1",
        "target namespace",
    ),
    (
        spec.get("expected_universe", {}).get("streaming_prefix_units"),
        10884,
        "streaming units",
    ),
    (
        spec.get("expected_universe", {}).get("target_scene_units"),
        681,
        "target units",
    ),
]
for actual, wanted, label in checks:
    if actual != wanted:
        raise SystemExit(f"protocol spec {label}={actual!r}, expected {wanted!r}")
if spec.get("feature_generation") != expected_noise:
    raise SystemExit("protocol spec deterministic-noise contract is missing or mismatched")
active_exclusions = [
    line.strip()
    for line in exclusions_path.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]
if active_exclusions:
    raise SystemExit(
        "canonical-v2 exclusions must be empty; found: " + ", ".join(active_exclusions[:5])
    )
PY

chmod +x \
  scripts/build_tstep_aoss_done_index.py \
  scripts/build_tstep_canonical_manifests.py \
  scripts/build_tstep_probe_matrix.py \
  scripts/audit_tstep_canonical_alignment.py \
  scripts/prune_tstep_probe_cache.py \
  scripts/run_tstep_probe_group.sh

mkdir -p "${OUT_DIR}/launcher"
echo "RUN_ID=${RUN_ID}"
echo "OUT_DIR=${OUT_DIR}"
echo "VFMS=${VFMS} LAYERS=${LAYERS} PREFIXES=${PREFIXES}"
echo "PROTOCOL_ID=${PROTOCOL_ID} OBS_LEN=${CANONICAL_OBS_LEN} HORIZONS=${TARGET_HORIZONS}"
echo "DONE_INDEX_DIR=${DONE_INDEX_DIR} MANIFEST_DIR=${MANIFEST_DIR} STRICT_AUDIT_DIR=${STRICT_AUDIT_DIR}"
echo "STREAMING_NAMESPACE=${STREAMING_NAMESPACE} TARGET_NAMESPACE=${TARGET_NAMESPACE}"
echo "REFRESH_AOSS_INDEX=${REFRESH_AOSS_INDEX} STATUS_POLICY=${STATUS_POLICY}"
echo "GROUP_REGEX=${GROUP_REGEX} MAX_GROUPS=${MAX_GROUPS} DRY_SUBMIT=${DRY_SUBMIT}"

if [[ "${REFRESH_AOSS_INDEX}" == "1" ]]; then
  read -r -a vfm_args <<< "${VFMS}"
  read -r -a layer_args <<< "${LAYERS}"
  "${PYTHON}" scripts/build_tstep_aoss_done_index.py \
    --output-dir "${DONE_INDEX_DIR}" \
    --kinds FEAT_STREAMING FEAT_TARGET \
    --streaming-namespace "${STREAMING_NAMESPACE}" \
    --target-namespace "${TARGET_NAMESPACE}" \
    --expected-streaming-units "${EXPECTED_STREAMING_UNITS}" \
    --expected-target-units "${EXPECTED_TARGET_UNITS}" \
    --vfms "${vfm_args[@]}" \
    --source "${SOURCE}" \
    --layers "${layer_args[@]}" \
    --require-sidecar
fi

"${PYTHON}" scripts/build_tstep_probe_matrix.py \
  --output-dir "${OUT_DIR}/matrix" \
  --done-index-dir "${DONE_INDEX_DIR}" \
  --streaming-namespace "${STREAMING_NAMESPACE}" \
  --target-namespace "${TARGET_NAMESPACE}" \
  --manifest-dir "${MANIFEST_DIR}" \
  --strict-audit-dir "${STRICT_AUDIT_DIR}" \
  --streaming-index-root "${STREAMING_INDEX_ROOT}" \
  --target-index-root "${TARGET_INDEX_ROOT}" \
  --source "${SOURCE}" \
  --protocol-id "${PROTOCOL_ID}" \
  --canonical-obs-len "${CANONICAL_OBS_LEN}" \
  --target-horizons "${TARGET_HORIZONS}" \
  --vfms "${VFMS}" \
  --layers "${LAYERS}" \
  --prefixes "${PREFIXES}" \
  --streaming-exclusions "${STREAMING_EXCLUSIONS}"

groups_tsv="${OUT_DIR}/matrix/groups.tsv"
selected_tsv="${OUT_DIR}/matrix/selected.tsv"
"${PYTHON}" - "${groups_tsv}" "${selected_tsv}" "${STATUS_POLICY}" "${GROUP_REGEX}" "${MAX_GROUPS}" "${PROJECT_ROOT}" "${RUN_ID}" "${MANIFEST_DIR}" "${STRICT_AUDIT_DIR}" "${PROTOCOL_ID}" "${CANONICAL_OBS_LEN}" <<'PY'
import csv, hashlib, json, re, sys
from pathlib import Path
(
    source,
    output,
    policy,
    pattern,
    max_groups,
    project_root,
    run_id,
    manifest_dir,
    strict_audit_dir,
    protocol_id,
    canonical_obs_len,
) = sys.argv[1:]
if policy not in {"full", "available"}:
    raise SystemExit("STATUS_POLICY must be full|available")
matcher = re.compile(pattern)
project_root = Path(project_root).resolve()
manifest_dir = (project_root / manifest_dir).resolve() if not Path(manifest_dir).is_absolute() else Path(manifest_dir).resolve()
strict_audit_dir = (project_root / strict_audit_dir).resolve() if not Path(strict_audit_dir).is_absolute() else Path(strict_audit_dir).resolve()

def within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

required_columns = {
    "group_id", "status", "vfm", "source", "timestep", "layer", "prefix",
    "manifest", "tasks", "canonical_protocol", "canonical_obs_len",
    "streaming_namespace", "target_namespace", "strict_audit",
    "strict_audit_sha256", "strict_audit_ok",
}
with open(source, newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    fields = reader.fieldnames
    missing = sorted(required_columns - set(fields or ()))
    if missing:
        raise SystemExit(f"groups.tsv lacks canonical-v2 columns: {missing}")
    rows = []
    skipped_complete = 0
    for row in reader:
        allowed = row["status"] == "ready_full"
        if policy == "available":
            allowed = allowed or row["status"] == "ready_streaming_only"
        if not (allowed and matcher.search(row["group_id"])):
            continue
        if row["status"] not in {"ready_full", "ready_streaming_only"}:
            raise SystemExit(f"refusing non-ready group {row['group_id']}: {row['status']}")
        if not any(row["tasks"].split(",")):
            raise SystemExit(f"refusing ready group without tasks: {row['group_id']}")
        if row["strict_audit_ok"] != "true":
            raise SystemExit(f"refusing failed strict audit for {row['group_id']}")
        if row["canonical_protocol"] != protocol_id or row["canonical_obs_len"] != canonical_obs_len:
            raise SystemExit(f"refusing non-v2 protocol row {row['group_id']}")
        if row["streaming_namespace"] != "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1":
            raise SystemExit(f"refusing non-seeded streaming namespace for {row['group_id']}")
        if row["target_namespace"] != "FEAT_TARGET_CANONICAL_V2_SEEDED_V1":
            raise SystemExit(f"refusing non-seeded target namespace for {row['group_id']}")

        manifest = Path(row["manifest"]).resolve()
        audit = Path(row["strict_audit"]).resolve()
        if not within(manifest, manifest_dir) or not manifest.is_file():
            raise SystemExit(f"manifest escaped/missing for {row['group_id']}: {manifest}")
        if not within(audit, strict_audit_dir) or not audit.is_file():
            raise SystemExit(f"audit escaped/missing for {row['group_id']}: {audit}")
        if not re.fullmatch(r"[0-9a-f]{64}", row["strict_audit_sha256"]):
            raise SystemExit(f"invalid audit SHA256 for {row['group_id']}")
        if sha256(audit) != row["strict_audit_sha256"]:
            raise SystemExit(f"stale audit SHA256 for {row['group_id']}")
        report = json.loads(audit.read_text(encoding="utf-8"))
        audit_ok = (
            report.get("ok") is True
            and report.get("pass") is True
            and report.get("status") == "pass"
            and report.get("scope") == "full"
            and report.get("problem_count") == 0
            and report.get("protocol_id", report.get("protocol")) == protocol_id
            and str(report.get("canonical_obs_len")) == canonical_obs_len
            and report.get("canonical_protocol_sha256")
            == "215c720c8537f4dfae6243bedcc7339fbb7e3ba8a6ca2dcaf04f49cd4b738e4f"
            and report.get("streaming_namespace")
            == "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1"
            and report.get("target_namespace")
            == "FEAT_TARGET_CANONICAL_V2_SEEDED_V1"
            and report.get("feature_generation_contract")
            == "inscene_deterministic_noise_v1"
            and report.get("vfm") == row["vfm"]
            and report.get("source") == row["source"]
            and Path(str(report.get("manifest", ""))).resolve() == manifest
            and report.get("manifest_sha256") == sha256(manifest)
            and report.get("manifest_streaming_feature_root")
            == "aoss://baiqiao/InsScene-15K/FEAT_STREAMING_CANONICAL_V2_SEEDED_V1"
            and report.get("manifest_target_feature_root")
            == "aoss://baiqiao/InsScene-15K/FEAT_TARGET_CANONICAL_V2_SEEDED_V1"
        )
        if not audit_ok:
            raise SystemExit(
                f"refusing failed, canary, stale, or mismatched audit for {row['group_id']}"
            )
        markers = []
        for task in filter(None, row["tasks"].split(",")):
            tag = (
                f"tp_{row['vfm']}_t{row['timestep']}_l{row['layer']}_"
                f"p{row['prefix']}_{task}"
            )
            markers.append(
                project_root
                / "logs/inscene15k_streaming/runs/tstep_probe_seeded_v1"
                / run_id
                / tag
                / "TRAIN_COMPLETE"
            )
        if markers and all(marker.is_file() for marker in markers):
            skipped_complete += 1
            continue
        rows.append(row)
limit = int(max_groups)
if limit > 0:
    rows = rows[:limit]
with open(output, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
print(
    f"selected_groups={len(rows)} skipped_complete={skipped_complete} "
    f"policy={policy} regex={pattern}"
)
PY

n_groups=$(( $(wc -l < "${selected_tsv}") - 1 ))
if (( n_groups < 1 )); then
  echo "[err] no groups selected; review ${OUT_DIR}/matrix/summary.json and groups.tsv" >&2
  exit 4
fi
echo "selected=${n_groups}:"
cut -f1-6 "${selected_tsv}"

# Compose every selected VFM/task family once before any external mutation.
"${PYTHON}" - "${selected_tsv}" "${PROJECT_ROOT}" <<'PY'
import csv, sys
from pathlib import Path
from hydra import compose, initialize_config_dir

selected, root = sys.argv[1:]
with open(selected, newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
pairs = sorted({(row["vfm"], task) for row in rows for task in row["tasks"].split(",") if task})
with initialize_config_dir(version_base="1.3", config_dir=str(Path(root) / "configs")):
    for vfm, task in pairs:
        cfg = compose(
            config_name="train",
            overrides=[f"experiment=inscene15k_streaming/{task}_{vfm}_v1", "trainer.devices=[0]"],
        )
        assert cfg.vfm_name == vfm
        print("hydra_ok", vfm, task)
PY

# Build an explicit plan in Python so TSV schema additions cannot silently
# shift Bash positional fields again. The strict audit path is taken directly
# from each selected row and shell-quoted into that row's remote command.
submit_plan="${OUT_DIR}/launcher/submit_plan.tsv"
"${PYTHON}" - "${selected_tsv}" "${submit_plan}" "${PROJECT_ROOT}" "${groups_tsv}" "${RUN_ID}" "${JOB_PREFIX}" "${BATCH_SIZE}" "${BATCH_SIZE_VAL}" "${CACHE_MAX_GIB}" "${CACHE_TARGET_GIB}" <<'PY'
import csv
import re
import shlex
import sys

(
    selected,
    output,
    project_root,
    groups_tsv,
    run_id,
    job_prefix,
    batch_size,
    batch_size_val,
    cache_max_gib,
    cache_target_gib,
) = sys.argv[1:]
with open(selected, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))

fields = [
    "group_id", "status", "vfm", "source", "timestep", "layer", "prefix",
    "streaming_namespace", "target_namespace", "strict_audit", "job_name", "command",
]
plans = []
for row in rows:
    if row["status"] not in {"ready_full", "ready_streaming_only"}:
        raise SystemExit(f"refusing non-ready plan row: {row['group_id']} {row['status']}")
    if row.get("strict_audit_ok") != "true":
        raise SystemExit(f"refusing failed audit plan row: {row['group_id']}")
    allow_streaming_only = "1" if row["status"] == "ready_streaming_only" else "0"
    job_name = (
        f"{job_prefix}_{row['vfm']}_t{row['timestep']}_"
        f"l{row['layer']}_p{row['prefix']}"
    )
    if not re.fullmatch(r"[A-Za-z0-9_-]+", job_name):
        raise SystemExit(f"unsafe SCO job name: {job_name!r}")
    command = "cd " + shlex.quote(project_root) + " && " + shlex.join(
        [
            "env",
            f"GROUPS_TSV={groups_tsv}",
            f"GROUP_ID={row['group_id']}",
            f"RUN_ID={run_id}",
            "AEC2_NAME=zoetrope",
            f"ALLOW_STREAMING_ONLY={allow_streaming_only}",
            f"INSCENE_STRICT_AUDIT={row['strict_audit']}",
            f"BATCH_SIZE={batch_size}",
            f"BATCH_SIZE_VAL={batch_size_val}",
            f"CACHE_MAX_GIB={cache_max_gib}",
            f"CACHE_TARGET_GIB={cache_target_gib}",
            "bash",
            "scripts/run_tstep_probe_group.sh",
        ]
    )
    plans.append(
        {
            **{key: row[key] for key in fields[:10]},
            "job_name": job_name,
            "command": command,
        }
    )
with open(output, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerows(plans)
print(f"submit_plan_rows={len(plans)} path={output}")
PY

submit_log="${OUT_DIR}/launcher/sco_submit.log"
: > "${submit_log}"
planned=0
submitted=0
while IFS= read -r -d '' job_name &&
      IFS= read -r -d '' group_id &&
      IFS= read -r -d '' status &&
      IFS= read -r -d '' command; do
  planned=$((planned + 1))
  if [[ "${DRY_SUBMIT}" == "1" ]]; then
    echo "[dry-submit] ${job_name} group=${group_id} status=${status}" | tee -a "${submit_log}"
    continue
  fi
  echo "[submit] ${job_name} group=${group_id} status=${status}" | tee -a "${submit_log}"
  sco acp jobs create \
    --workspace-name=aigc \
    --aec2-name="${aec2_name}" \
    --job-name="${job_name}" \
    --priority=HIGHEST \
    --quota-type=reserved \
    --container-image-url="${container_image_url}" \
    --storage-mount="${storage_mount}" \
    --training-framework=pytorch \
    --worker-nodes=1 \
    --worker-spec="${worker_spec}${NPROC}" \
    --command="${command}" \
    | tee -a "${submit_log}"
  submitted=$((submitted + 1))
done < <("${PYTHON}" - "${submit_plan}" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
out = sys.stdout.buffer
for row in rows:
    for key in ("job_name", "group_id", "status", "command"):
        value = row[key]
        if "\0" in value:
            raise SystemExit(f"NUL in submit-plan field {key}")
        out.write(value.encode("utf-8") + b"\0")
PY
)

if (( planned != n_groups )); then
  echo "[err] submit plan row mismatch: selected=${n_groups} planned=${planned}" >&2
  exit 5
fi
if [[ "${DRY_SUBMIT}" == "1" ]]; then
  echo "DRY_SUBMIT=1: validated ${planned} canonical-v2 jobs; no SCO jobs created."
  echo "review: ${submit_plan}"
  exit 0
fi

grep -oE 'pt-[a-z0-9]+' "${submit_log}" | awk '!seen[$0]++' \
  > "${OUT_DIR}/launcher/sco_job_ids.txt" || true
echo "submitted=${submitted} job_ids=$(tr '\n' ' ' < "${OUT_DIR}/launcher/sco_job_ids.txt")"
