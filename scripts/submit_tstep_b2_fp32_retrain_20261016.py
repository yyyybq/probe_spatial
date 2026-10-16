#!/usr/bin/env python3
"""Re-train the 32 timestep B2 (ego_belief_v2) cells as one-GPU Zoetrope jobs.

Background (docs/MINIMAX_REEXTRACT_RETRAIN_PLAN_20261016.md, stage 4): the
2026-09 timestep campaigns trained every probe with PROBE_FP16_PASSTHROUGH=1
(fp16 features handed to the GPU unchanged).  The ego_belief_v2 heads of
CogVideoX (t499/t999, L15/20/25) and MiniMax-H3 (t500/t1000, L15) diverged to
non-finite weights under that setting; the fp32 loader path
(PROBE_FP16_PASSTHROUGH=0) was proven finite by the 2026-09-28
single_gpu_cog12_ego_fp32 campaign.  This script re-runs exactly those 32 cells
on their original, finite, seeded-v1 fp16 caches.

Cells are taken from metadata/release_inventory/retrain_queue_20261016.csv
(queue_group B2_NONFINITE_WEIGHTS_FP16_PASSTHROUGH, inventory timestep).  Group
rows (manifest, strict audit, namespaces, protocol) are copied verbatim from
the 2026-09 campaign matrices and restricted to ``tasks=ego_belief_v2`` so that
scripts/run_tstep_probe_group.sh trains exactly one probe per job on GPU 0.

Default: dry run.  Writes the derived matrix and submit plan, validates every
strict audit and manifest hash, and prints the plan.  ``--submit`` creates the
SCO jobs (sequentially; pass ``--limit`` for a pilot) and records job ids.

Outputs under logs/tstep_probe_campaigns/<RUN_ID>/:
  matrix/groups.tsv        derived 32-row matrix (same columns as the sources)
  matrix/provenance.json   source matrix, original tasks and hashes per group
  launcher/submit_plan.tsv one row per job with the exact remote command
  launcher/submissions.json ledger (job ids, stdout/stderr, timestamps)
  launcher/sco_submit.log  append-only submission log
  launcher/sco_job_ids.txt one job id per line
Training outputs land in logs/inscene15k_streaming/runs/tstep_probe_seeded_v1/<RUN_ID>/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "tstep_probe_b2_fp32_20261016_r1"
OUT_DIR = ROOT / "logs/tstep_probe_campaigns" / RUN_ID
QUEUE_CSV = ROOT / "metadata/release_inventory/retrain_queue_20261016.csv"
QUEUE_GROUP = "B2_NONFINITE_WEIGHTS_FP16_PASSTHROUGH"
TASK = "ego_belief_v2"
EXPECTED_CELLS = 32
SOURCE_MATRICES = {
    "cogvideox": ROOT
    / "logs/tstep_probe_campaigns/tstep_probe_v2_sv1_wan_cgx_20260907_zoetrope_r1/matrix/groups.tsv",
    "minimax_h3": ROOT
    / "logs/tstep_probe_campaigns/tstep_probe_v2_sv1_minimax_h3_20260909_zoetrope_r1/matrix/groups.tsv",
}
PROTOCOL_ID = "inscene_streaming_canonical_v2_obs28_h124"
PROTOCOL_SHA256 = "215c720c8537f4dfae6243bedcc7339fbb7e3ba8a6ca2dcaf04f49cd4b738e4f"
STREAMING_NAMESPACE = "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1"
TARGET_NAMESPACE = "FEAT_TARGET_CANONICAL_V2_SEEDED_V1"
MANIFEST_DIR = ROOT / "artifacts/tstep_canonical_v2/manifests"
STRICT_AUDIT_DIR = ROOT / "artifacts/tstep_canonical_v2_seeded_v1/audits"
GROUP_SCRIPT = ROOT / "scripts/run_tstep_probe_group.sh"
# Jobs run as root in the container (no ~/.petreloss.conf); the AOSS client
# must be pointed at the project conf or the first remote read fails.
AOSS_CONF = "/mnt/umm/users/yinbaiqiao/aoss/petreloss.conf"
SCO = Path.home() / ".sco/bin/sco"
JOB_PREFIX = "bq_tpb2fp32"
CONTAINER_IMAGE = "registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617"
STORAGE_MOUNT = "019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm"
WORKER_SPEC = "N4lS.Iq.I80.1"
JOB_ID_RE = re.compile(r"pt-[a-z0-9]+")
COLUMNS = [
    "group_id", "status", "vfm", "source", "timestep", "layer", "prefix", "manifest",
    "tasks", "stream_expected", "stream_found", "target_expected", "target_found",
    "target_indices_found", "canonical_protocol", "canonical_obs_len",
    "streaming_namespace", "target_namespace", "strict_audit", "strict_audit_sha256",
    "strict_audit_ok", "strict_audit_problem_count", "reason",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    os.replace(tmp, path)


def within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def load_queue_cells() -> dict[str, dict]:
    cells: dict[str, dict] = {}
    with QUEUE_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["queue_group"] != QUEUE_GROUP or row["inventory"] != "timestep":
                continue
            if row["probe"] != TASK:
                raise SystemExit(f"unexpected probe in queue group: {row}")
            group_id = f"{row['model']}_t{row['timestep']}_l{row['layer']}_p{row['prefix']}"
            if group_id in cells:
                raise SystemExit(f"duplicate queue cell {group_id}")
            cells[group_id] = row
    if len(cells) != EXPECTED_CELLS:
        raise SystemExit(f"expected {EXPECTED_CELLS} timestep B2 cells, found {len(cells)}")
    return cells


def audit_passes(report: dict, row: dict, manifest: Path) -> bool:
    """Same predicate scripts/submit_tstep_probe_sco.sh applies before submitting."""
    return (
        report.get("ok") is True
        and report.get("pass") is True
        and report.get("status") == "pass"
        and report.get("scope") == "full"
        and report.get("problem_count") == 0
        and report.get("protocol_id", report.get("protocol")) == PROTOCOL_ID
        and str(report.get("canonical_obs_len")) == "28"
        and report.get("canonical_protocol_sha256") == PROTOCOL_SHA256
        and report.get("streaming_namespace") == STREAMING_NAMESPACE
        and report.get("target_namespace") == TARGET_NAMESPACE
        and report.get("feature_generation_contract") == "inscene_deterministic_noise_v1"
        and report.get("vfm") == row["vfm"]
        and report.get("source") == row["source"]
        and Path(str(report.get("manifest", ""))).resolve() == manifest
        and report.get("manifest_sha256") == sha256(manifest)
        and report.get("manifest_streaming_feature_root")
        == f"aoss://baiqiao/InsScene-15K/{STREAMING_NAMESPACE}"
        and report.get("manifest_target_feature_root")
        == f"aoss://baiqiao/InsScene-15K/{TARGET_NAMESPACE}"
    )


def build_matrix(cells: dict[str, dict]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    provenance: dict = {}
    for vfm, matrix_path in SOURCE_MATRICES.items():
        with matrix_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames != COLUMNS:
                raise SystemExit(f"{matrix_path}: unexpected columns {reader.fieldnames}")
            for row in reader:
                if row["group_id"] not in cells or row["vfm"] != vfm:
                    continue
                cell = cells[row["group_id"]]
                if (row["timestep"], row["layer"], row["prefix"]) != (
                    cell["timestep"], cell["layer"], cell["prefix"]
                ):
                    raise SystemExit(f"queue/matrix coordinate mismatch for {row['group_id']}")
                if row["status"] != "ready_full":
                    raise SystemExit(f"{row['group_id']} is {row['status']}, not ready_full")
                if TASK not in row["tasks"].split(","):
                    raise SystemExit(f"{row['group_id']} never trained {TASK}")
                if row["strict_audit_ok"] != "true":
                    raise SystemExit(f"{row['group_id']} has no passing strict audit")
                if (
                    row["canonical_protocol"] != PROTOCOL_ID
                    or row["canonical_obs_len"] != "28"
                    or row["streaming_namespace"] != STREAMING_NAMESPACE
                    or row["target_namespace"] != TARGET_NAMESPACE
                ):
                    raise SystemExit(f"{row['group_id']} is not bound to canonical-v2 seeded-v1")
                manifest = Path(row["manifest"]).resolve()
                audit = Path(row["strict_audit"]).resolve()
                if not within(manifest, MANIFEST_DIR.resolve()) or not manifest.is_file():
                    raise SystemExit(f"manifest escaped/missing for {row['group_id']}: {manifest}")
                if not within(audit, STRICT_AUDIT_DIR.resolve()) or not audit.is_file():
                    raise SystemExit(f"audit escaped/missing for {row['group_id']}: {audit}")
                if not re.fullmatch(r"[0-9a-f]{64}", row["strict_audit_sha256"]):
                    raise SystemExit(f"invalid audit SHA256 for {row['group_id']}")
                audit_sha = sha256(audit)
                if audit_sha != row["strict_audit_sha256"]:
                    raise SystemExit(f"stale audit SHA256 for {row['group_id']}")
                report = json.loads(audit.read_text(encoding="utf-8"))
                if not audit_passes(report, row, manifest):
                    raise SystemExit(f"strict audit does not pass for {row['group_id']}")
                derived = dict(row)
                derived["tasks"] = TASK
                rows.append(derived)
                provenance[row["group_id"]] = {
                    "source_matrix": str(matrix_path),
                    "source_matrix_sha256": sha256(matrix_path),
                    "original_tasks": row["tasks"],
                    "strict_audit": str(audit),
                    "strict_audit_sha256": audit_sha,
                    "manifest": str(manifest),
                    "manifest_sha256": report.get("manifest_sha256"),
                    "old_output_path": cell.get("old_output_path"),
                    "old_checkpoint_path": cell.get("old_checkpoint_path"),
                    "queue_reason": cell.get("reason"),
                }
    found = {row["group_id"] for row in rows}
    missing = sorted(set(cells) - found)
    if missing:
        raise SystemExit(f"queue cells without a source matrix row: {missing}")
    rows.sort(key=lambda r: (r["vfm"], int(r["timestep"]), int(r["layer"]), int(r["prefix"])))
    return rows, provenance


def check_hydra(rows: list[dict]) -> None:
    from hydra import compose, initialize_config_dir  # probe-spatial env only

    pairs = sorted({row["vfm"] for row in rows})
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        for vfm in pairs:
            cfg = compose(
                config_name="train",
                overrides=[f"experiment=inscene15k_streaming/{TASK}_{vfm}_v1", "trainer.devices=[0]"],
            )
            if cfg.vfm_name != vfm:
                raise SystemExit(f"hydra vfm mismatch for {vfm}")
            print("hydra_ok", vfm, TASK)


def build_plan(rows: list[dict], groups_tsv: Path, cache_max_gib: int, cache_target_gib: int) -> list[dict]:
    plans = []
    for row in rows:
        job_name = f"{JOB_PREFIX}_{row['vfm']}_t{row['timestep']}_l{row['layer']}_p{row['prefix']}"
        if not re.fullmatch(r"[A-Za-z0-9_-]+", job_name):
            raise SystemExit(f"unsafe SCO job name: {job_name!r}")
        tag = f"tp_{row['vfm']}_t{row['timestep']}_l{row['layer']}_p{row['prefix']}_{TASK}"
        marker = ROOT / "logs/inscene15k_streaming/runs/tstep_probe_seeded_v1" / RUN_ID / tag / "TRAIN_COMPLETE"
        command = "cd " + shlex.quote(str(ROOT)) + " && " + shlex.join(
            [
                "env",
                f"GROUPS_TSV={groups_tsv}",
                f"GROUP_ID={row['group_id']}",
                f"RUN_ID={RUN_ID}",
                "AEC2_NAME=zoetrope",
                "ALLOW_STREAMING_ONLY=0",
                f"INSCENE_STRICT_AUDIT={row['strict_audit']}",
                "BATCH_SIZE=2",
                "BATCH_SIZE_VAL=2",
                f"CACHE_MAX_GIB={cache_max_gib}",
                f"CACHE_TARGET_GIB={cache_target_gib}",
                "GPU_IDS_CSV=0",
                "PROBE_FP16_PASSTHROUGH=0",
                f"AOSS_CONF={AOSS_CONF}",
                "bash",
                "scripts/run_tstep_probe_group.sh",
            ]
        )
        plans.append(
            {
                "group_id": row["group_id"],
                "status": row["status"],
                "vfm": row["vfm"],
                "source": row["source"],
                "timestep": row["timestep"],
                "layer": row["layer"],
                "prefix": row["prefix"],
                "task": TASK,
                "streaming_namespace": row["streaming_namespace"],
                "target_namespace": row["target_namespace"],
                "strict_audit": row["strict_audit"],
                "run_tag": tag,
                "train_complete_marker": str(marker),
                "already_complete": marker.is_file(),
                "job_name": job_name,
                "command": command,
            }
        )
    return plans


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def job_state(job_id: str) -> str:
    completed = subprocess.run(
        [str(SCO), "acp", "jobs", "describe", job_id, "--workspace-name=aigc", "-o", "json"],
        text=True, capture_output=True, timeout=120,
    )
    try:
        return str(json.loads(completed.stdout).get("state", "UNKNOWN"))
    except json.JSONDecodeError:
        return "UNKNOWN"


def submit(plans: list[dict], limit: int, ledger_path: Path, submit_log: Path,
           retry_failed: bool, retry_reason: str) -> None:
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {
        "schema": "TSTEP_B2_FP32_RETRAIN_SUBMISSIONS_V1",
        "run_id": RUN_ID,
        "resource_pool": "zoetrope",
        "worker_spec": WORKER_SPEC,
        "feature_transfer_precision": {"PROBE_FP16_PASSTHROUGH": "0"},
        "group_script_sha256": sha256(GROUP_SCRIPT),
        "jobs": {},
    }
    ledger["group_script_sha256"] = sha256(GROUP_SCRIPT)
    pending = []
    for plan in plans:
        if plan["already_complete"]:
            continue
        previous = ledger["jobs"].get(plan["group_id"])
        if previous is None:
            pending.append(plan)
            continue
        if retry_failed and previous.get("job_id"):
            state = job_state(previous["job_id"])
            if state == "FAILED":
                attempt = {k: previous.get(k) for k in ("job_id", "job_name", "submitted_utc", "returncode", "command")}
                attempt.update({"state": state, "retry_reason": retry_reason})
                plan["previous_attempts"] = list(previous.get("previous_attempts", [])) + [attempt]
                pending.append(plan)
            else:
                print(f"  keep {plan['group_id']}: {previous['job_id']} is {state}")
    if limit > 0:
        pending = pending[:limit]
    print(f"pending={len(pending)} recorded={len(ledger['jobs'])}")
    for plan in pending:
        argv = [
            str(SCO), "acp", "jobs", "create", "--workspace-name=aigc", "--aec2-name=zoetrope",
            f"--job-name={plan['job_name']}", "--priority=HIGHEST", "--quota-type=reserved",
            f"--container-image-url={CONTAINER_IMAGE}", f"--storage-mount={STORAGE_MOUNT}",
            "--training-framework=pytorch", "--worker-nodes=1", f"--worker-spec={WORKER_SPEC}",
            f"--command={plan['command']}",
        ]
        completed = subprocess.run(argv, text=True, capture_output=True, timeout=180)
        text = completed.stdout + "\n" + completed.stderr
        match = JOB_ID_RE.search(text)
        record = {
            **plan,
            "job_id": match.group(0) if match else None,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "submitted_utc": now(),
        }
        ledger["jobs"][plan["group_id"]] = record
        ledger["updated_utc"] = now()
        atomic_json(ledger_path, ledger)
        with submit_log.open("a", encoding="utf-8") as handle:
            handle.write(f"[{record['submitted_utc']}] {plan['job_name']} group={plan['group_id']} "
                         f"job_id={record['job_id']} rc={completed.returncode}\n")
        print(f"{plan['group_id']}\t{record['job_id'] or 'SUBMIT_ERROR'}", flush=True)
        if completed.returncode or not record["job_id"]:
            print(text, flush=True)
            raise SystemExit(f"submission failed for {plan['group_id']}; stopping")
    ids = [job["job_id"] for job in ledger["jobs"].values() if job.get("job_id")]
    (submit_log.parent / "sco_job_ids.txt").write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
    print(json.dumps({"recorded": len(ledger["jobs"]), "job_ids": len(ids)}))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--submit", action="store_true", help="create the SCO jobs (default: dry run)")
    parser.add_argument("--limit", type=int, default=0, help="submit at most N pending jobs (pilot)")
    parser.add_argument("--cache-max-gib", type=int, default=400)
    parser.add_argument("--cache-target-gib", type=int, default=300)
    parser.add_argument("--skip-hydra", action="store_true")
    parser.add_argument("--retry-failed", action="store_true",
                        help="re-submit recorded jobs whose SCO state is FAILED (previous attempts are kept in the ledger)")
    parser.add_argument("--retry-reason", default="", help="why the failed attempts are being retried")
    args = parser.parse_args()
    if args.retry_failed and not args.retry_reason:
        raise SystemExit("--retry-failed requires --retry-reason")
    if not Path(AOSS_CONF).is_file():
        raise SystemExit(f"AOSS conf not found: {AOSS_CONF}")
    if not GROUP_SCRIPT.is_file():
        raise SystemExit(f"missing {GROUP_SCRIPT}")
    if "PROBE_FP16_PASSTHROUGH" not in GROUP_SCRIPT.read_text(encoding="utf-8"):
        raise SystemExit("run_tstep_probe_group.sh does not honour PROBE_FP16_PASSTHROUGH")

    cells = load_queue_cells()
    rows, provenance = build_matrix(cells)
    groups_tsv = OUT_DIR / "matrix/groups.tsv"
    write_tsv(groups_tsv, rows, COLUMNS)
    atomic_json(OUT_DIR / "matrix/provenance.json", {
        "schema": "TSTEP_B2_FP32_RETRAIN_MATRIX_PROVENANCE_V1",
        "run_id": RUN_ID,
        "queue_csv": str(QUEUE_CSV),
        "queue_csv_sha256": sha256(QUEUE_CSV),
        "queue_group": QUEUE_GROUP,
        "task": TASK,
        "groups": provenance,
        "generated_utc": now(),
    })
    if not args.skip_hydra:
        check_hydra(rows)
    plans = build_plan(rows, groups_tsv, args.cache_max_gib, args.cache_target_gib)
    plan_fields = [k for k in plans[0].keys()]
    write_tsv(OUT_DIR / "launcher/submit_plan.tsv", plans, plan_fields)
    print(f"matrix_rows={len(rows)} plan_rows={len(plans)} groups_tsv={groups_tsv}")
    for plan in plans:
        print(f"  {plan['job_name']}  complete={plan['already_complete']}")
    if not args.submit:
        print("dry run: no SCO jobs created (pass --submit)")
        return 0
    if not SCO.is_file():
        raise SystemExit(f"sco CLI not found at {SCO}")
    submit(plans, args.limit, OUT_DIR / "launcher/submissions.json", OUT_DIR / "launcher/sco_submit.log",
           args.retry_failed, args.retry_reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
