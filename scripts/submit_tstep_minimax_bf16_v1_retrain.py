#!/usr/bin/env python3
"""Stage 5: re-train the 128 timestep MiniMax-H3 L20/L25 runs on the bf16 caches.

docs/MINIMAX_REEXTRACT_RETRAIN_PLAN_20261016.md, stage 5: 16 groups
(t500/t1000 x layers 20/25 x prefixes 8/12/16/24) x the 8 probes the 2026-09
campaign trained (streaming_depth, streaming_mask, view_consistency,
ego_belief, ego_belief_v2, action_dynamics, path_integration,
counterfactual), one 8-GPU Zoetrope node per group (one probe per GPU) via
scripts/run_tstep_probe_group.sh.  The runs being replaced were trained by
the tstep_probe_opt_v3_20260912_highest_r1 and tstep_probe_pack_v4_1_20260914_*
campaigns; group rows are checked against the 2026-09-09 MiniMax matrix
(manifests, probe set, expected counts) and every cell's composed Hydra config
against the old run's saved config
(scripts/check_tstep_minimax_bf16_v1_config_parity.py, must PASS).  Intended
differences: the *_SEEDED_V1_MINIMAX_BF16_V1 namespaces, PROBE_FP16_PASSTHROUGH=0
and training from scratch (17 old runs had been resumed from partial runs).

Default: plan + dry run.
  * If the consolidation job's apply report and full verification report are
    not both PASS, the matrix is built into a preview directory only
    (launcher/preview_<UTC>/, audits included) and nothing else happens.
  * Otherwise the matrix is built into matrix/ with the strict audits in
    artifacts/tstep_canonical_v2_seeded_v1_minimax_bf16_v1/audits/, every row
    must be ready_full with the 8 probes and a verified audit, and each group
    is dry-run through run_tstep_probe_group.sh (DRY_RUN=1, 8 GPU ids).
``--layout single`` instead plans 128 one-GPU jobs (the plan's alternative):
matrix/groups_single.tsv repeats each group row once per probe with
``group_id=<group>__<probe>`` and ``tasks=<probe>``; run tags, outputs and
everything else are identical to the 8-GPU layout.
``--submit --approval "<who approved, when, quote>"`` creates the jobs.
User decision (2026-10-16): first ``--submit --limit 1`` (a one-group pilot);
any later ``--submit`` is refused until the newest checkpoint of every probe of
every job already in the ledger loads on CPU with zero non-finite parameters
(launcher/pilot_gate.json).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_tstep_canonical_alignment import (  # noqa: E402
    DEFAULT_PROTOCOL_ID, MINIMAX_BF16_STREAMING_NAMESPACE, MINIMAX_BF16_TARGET_NAMESPACE, sha256_file,
    verify_audit_artifact,
)
from submit_tstep_minimax_bf16_v1_extract import (  # noqa: E402
    AOSS_CONF, CONTAINER_IMAGE, JOB_ID_RE, PY_PROBE, ROOT, SCO, STORAGE_MOUNT, atomic_json, now,
)

RUN_ID = "tstep_probe_minimax_bf16_v1_r1"
OUT_DIR = ROOT / "logs/tstep_probe_campaigns" / RUN_ID
RUNS_ROOT = ROOT / "logs/inscene15k_streaming/runs/tstep_probe_seeded_v1" / RUN_ID
STRICT_AUDIT_DIR = ROOT / "artifacts/tstep_canonical_v2_seeded_v1_minimax_bf16_v1/audits"
CONSOLIDATION_DIR = ROOT / "logs/tstep_minimax_bf16_v1/production_20261016/consolidation"
SOURCE_MATRIX = ROOT / "logs/tstep_probe_campaigns/tstep_probe_v2_sv1_minimax_h3_20260909_zoetrope_r1/matrix/groups.tsv"
GROUP_SCRIPT = ROOT / "scripts/run_tstep_probe_group.sh"
DATASET_ROOT = Path("/mnt/umm/users/yinbaiqiao/InsScene-15K")
TASKS = ("streaming_depth", "streaming_mask", "view_consistency", "ego_belief", "ego_belief_v2",
         "action_dynamics", "path_integration", "counterfactual")
LAYERS, PREFIXES, TIMESTEPS = (20, 25), (8, 12, 16, 24), (500, 1000)
LAYOUTS = {  # layout -> (worker spec, GPU_IDS_CSV)
    "node8": ("N4lS.Iq.I80.8", "0,1,2,3,4,5,6,7"),
    "single": ("N4lS.Iq.I80.1", "0"),
}
JOB_PREFIX = "bq_tp_mmxbf16"
# Same as the 2026-09 MiniMax campaign's submit_plan.tsv.
GROUP_ENV = {"BATCH_SIZE": "2", "BATCH_SIZE_VAL": "2", "CACHE_MAX_GIB": "900", "CACHE_TARGET_GIB": "700"}


def consolidation_ready() -> tuple[bool, str]:
    ledger_path = ROOT / "logs/tstep_minimax_bf16_v1/submissions.json"
    if not ledger_path.is_file():
        return False, f"extraction ledger missing: {ledger_path}"
    rec = json.loads(ledger_path.read_text(encoding="utf-8"))["jobs"].get("consolidation")
    if not rec:
        return False, "consolidation job not submitted"
    reports = {}
    for key in ("apply_report", "verify_full_report"):
        path = Path(rec[key])
        if not path.is_file():
            return False, f"{key} missing: {path}"
        reports[key] = json.loads(path.read_text(encoding="utf-8"))
    if reports["apply_report"].get("status") != "PASS" or reports["apply_report"].get("mode") != "apply":
        return False, f"apply report is {reports['apply_report'].get('status')}"
    if reports["verify_full_report"].get("status") != "PASS":
        return False, f"full verification is {reports['verify_full_report'].get('status')}"
    return True, f"consolidation {rec.get('job_id')} PASS (apply + full verification)"


def build_matrix(output_dir: Path, audit_dir: Path) -> list[dict]:
    command = [
        PY_PROBE, str(ROOT / "scripts/build_tstep_probe_matrix.py"),
        "--output-dir", str(output_dir), "--strict-audit-dir", str(audit_dir),
        "--done-index-dir", "logs/tstep_canonical_v2_seeded_v1/aoss_done",
        "--manifest-dir", "artifacts/tstep_canonical_v2/manifests",
        "--streaming-namespace", MINIMAX_BF16_STREAMING_NAMESPACE,
        "--target-namespace", MINIMAX_BF16_TARGET_NAMESPACE,
        "--streaming-index-root", str(DATASET_ROOT / MINIMAX_BF16_STREAMING_NAMESPACE),
        "--target-index-root", str(DATASET_ROOT / MINIMAX_BF16_TARGET_NAMESPACE),
        "--vfms", "minimax_h3", "--layers", " ".join(map(str, LAYERS)),
        "--prefixes", " ".join(map(str, PREFIXES)), "--tasks", " ".join(TASKS),
    ]
    done = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if done.returncode:
        raise SystemExit(f"matrix build failed:\n{done.stdout}\n{done.stderr}")
    with (output_dir / "groups.tsv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def validate(rows: list[dict]) -> list[str]:
    errors = []
    expected = {f"minimax_h3_t{t}_l{l}_p{p}" for t in TIMESTEPS for l in LAYERS for p in PREFIXES}
    if {row["group_id"] for row in rows} != expected or len(rows) != len(expected):
        errors.append(f"matrix groups {sorted(r['group_id'] for r in rows)} != the 16 stage-5 groups")
    with SOURCE_MATRIX.open(newline="", encoding="utf-8") as handle:
        old = {row["group_id"]: row for row in csv.DictReader(handle, delimiter="\t")}
    for row in rows:
        gid = row["group_id"]
        if row["status"] != "ready_full" or row["tasks"] != ",".join(TASKS):
            errors.append(f"{gid}: status={row['status']} tasks={row['tasks']} reason={row['reason']}")
            continue
        if row["strict_audit_ok"] != "true":
            errors.append(f"{gid}: strict audit not ok")
            continue
        problems = verify_audit_artifact(
            artifact=Path(row["strict_audit"]), manifest=Path(row["manifest"]), vfm="minimax_h3",
            source=row["source"], protocol_id=DEFAULT_PROTOCOL_ID, canonical_obs_len=28,
            prefix=int(row["prefix"]), expected_artifact_sha256=row["strict_audit_sha256"],
            streaming_namespace=row["streaming_namespace"], target_namespace=row["target_namespace"],
        )
        errors += [f"{gid}: {p}" for p in problems]
        prior = old.get(gid)
        if prior is None:
            errors.append(f"{gid}: absent from the 2026-09 source matrix")
        else:
            for key in ("manifest", "tasks", "stream_expected", "target_expected"):
                if prior[key] != row[key]:
                    errors.append(f"{gid}: {key} {row[key]!r} != 2026-09 {prior[key]!r}")
    return errors


def pilot_gate(jobs: dict) -> dict:
    """Newest checkpoint of every probe of every submitted job must exist and be all-finite."""
    import torch  # probe-spatial env

    runs_root = RUNS_ROOT
    checked, problems = [], []
    for key, job in sorted(jobs.items()):
        group = key.split("__", 1)[0]
        for task in job["tasks"].split(","):
            tag = f"tp_{group}_{task}"
            ckpts = sorted((runs_root / tag).glob("VideoProbe3D/*/checkpoints/*.ckpt"), key=lambda p: p.stat().st_mtime)
            if not ckpts:
                problems.append(f"{tag}: no checkpoint yet")
                continue
            rec = {"tag": tag, "ckpt": str(ckpts[-1])}
            try:
                state = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
                tensors = [t for t in state.get("state_dict", state).values()
                           if torch.is_tensor(t) and t.is_floating_point()]
                rec.update({"epoch": state.get("epoch"), "n_params": sum(t.numel() for t in tensors),
                            "n_nonfinite": sum(int((~torch.isfinite(t)).sum()) for t in tensors)})
                if rec["n_params"] == 0 or rec["n_nonfinite"]:
                    problems.append(f"{tag}: {rec['n_nonfinite']} non-finite of {rec['n_params']} parameters")
            except Exception as exc:  # noqa: BLE001 - a checkpoint being rotated or written counts as not ready
                problems.append(f"{tag}: unreadable checkpoint {exc!r}"[:300])
            checked.append(rec)
    return {"schema": "TSTEP_MINIMAX_BF16_V1_PILOT_GATE_V1", "checked_utc": now(), "runs": checked,
            "problems": problems, "status": "PASS" if checked and not problems else "FAIL"}


def config_parity() -> list[str]:
    done = subprocess.run([PY_PROBE, str(ROOT / "scripts/check_tstep_minimax_bf16_v1_config_parity.py"),
                           "--report", str(OUT_DIR / "launcher/config_parity.json")],
                          cwd=ROOT, text=True, capture_output=True)
    if done.returncode:
        return [f"config parity with the replaced runs failed: {done.stdout[-600:]} {done.stderr[-300:]}"]
    return []


def group_command(row: dict, groups_tsv: Path, gpu_ids: str) -> str:
    env = ["env", f"GROUPS_TSV={groups_tsv}", f"GROUP_ID={row['group_id']}", f"RUN_ID={RUN_ID}",
           "AEC2_NAME=zoetrope", "ALLOW_STREAMING_ONLY=0", f"INSCENE_STRICT_AUDIT={row['strict_audit']}",
           *(f"{k}={v}" for k, v in GROUP_ENV.items()), f"GPU_IDS_CSV={gpu_ids}", "PROBE_FP16_PASSTHROUGH=0",
           f"AOSS_CONF={AOSS_CONF}", "bash", "scripts/run_tstep_probe_group.sh"]
    return "cd " + shlex.quote(str(ROOT)) + " && " + shlex.join(env)


def dry_run(row: dict, groups_tsv: Path, log_root: Path, gpu_ids: str) -> dict:
    env = os.environ.copy()
    env.update({"GROUPS_TSV": str(groups_tsv), "GROUP_ID": row["group_id"], "RUN_ID": RUN_ID, "DRY_RUN": "1",
                "AEC2_NAME": "zoetrope", "ALLOW_STREAMING_ONLY": "0", "INSCENE_STRICT_AUDIT": row["strict_audit"],
                **GROUP_ENV, "GPU_IDS_CSV": gpu_ids, "PROBE_FP16_PASSTHROUGH": "0", "AOSS_CONF": AOSS_CONF,
                "LOG_ROOT": str(log_root), "AOSS_CACHE_BASE": str(log_root / "cache")})
    done = subprocess.run(["bash", str(GROUP_SCRIPT)], cwd=ROOT, env=env, text=True, capture_output=True)
    result = {"group_id": row["group_id"], "returncode": done.returncode, "problems": []}
    if done.returncode:
        result["problems"].append(f"rc={done.returncode}: {done.stderr[-800:]}")
        return result
    stream_root = f"streaming_feat_root=aoss://baiqiao/InsScene-15K/{MINIMAX_BF16_STREAMING_NAMESPACE}"
    for task in row["tasks"].split(","):
        text = (log_root / "groups" / row["group_id"] / f"{task}.log").read_text(encoding="utf-8")
        line = next((l for l in text.splitlines() if l.startswith("[dry-run]")), "")
        wants = [stream_root, f"feature_layer={row['layer']}", f"feature_timestep={row['timestep']}",
                 f"prefix_len={row['prefix']}", f"experiment=inscene15k_streaming/{task}_minimax_h3_v1"]
        if task in ("action_dynamics", "path_integration", "counterfactual"):
            wants.append(f"target_feat_root=aoss://baiqiao/InsScene-15K/{MINIMAX_BF16_TARGET_NAMESPACE}")
        missing = [w for w in wants if w not in line]
        if missing or "rc=0" not in text:
            result["problems"].append(f"{task}: dry-run line lacks {missing}")
    if "namespace_family=seeded_v1_minimax_bf16_v1 probe_fp16_passthrough=0" not in done.stdout:
        result["problems"].append("runner did not report the bf16 family with PROBE_FP16_PASSTHROUGH=0")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--approval", default="", help="required with --submit: who approved stage 5, when, quote")
    parser.add_argument("--limit", type=int, default=0, help="submit at most N jobs (pilot)")
    parser.add_argument("--layout", choices=sorted(LAYOUTS), default="node8",
                        help="node8: 16 jobs x 8 GPUs (plan default); single: 128 jobs x 1 GPU")
    args = parser.parse_args()
    worker_spec, gpu_ids = LAYOUTS[args.layout]
    ready, why = consolidation_ready()
    print(f"consolidation: {why}")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if not ready:
        preview = OUT_DIR / "launcher" / f"preview_{stamp}"
        rows = build_matrix(preview / "matrix", preview / "audits")
        for row in rows:
            print(f"  {row['group_id']:28s} {row['status']:22s} stream {row['stream_found']}/{row['stream_expected']} "
                  f"target {row['target_found']}/{row['target_expected']} {row['reason'][:90]}")
        print(f"preview only (written to {preview}); stage 5 cannot be dry-run or submitted yet")
        if args.submit:
            raise SystemExit("refusing --submit: consolidation is not PASS")
        return 0

    if "seeded_v1_minimax_bf16_v1" not in GROUP_SCRIPT.read_text(encoding="utf-8"):
        raise SystemExit(
            f"{GROUP_SCRIPT} does not accept the bf16 namespaces yet; install the staged "
            f"{GROUP_SCRIPT.name}.stage5 (rename, not copy-in-place) once no job runs the old copy"
        )
    rows = build_matrix(OUT_DIR / "matrix", STRICT_AUDIT_DIR)
    errors = validate(rows)
    errors += config_parity()
    rows.sort(key=lambda r: (int(r["timestep"]), int(r["layer"]), int(r["prefix"])))
    groups_tsv = OUT_DIR / "matrix/groups.tsv"
    if args.layout == "single":
        rows = [{**row, "group_id": f"{row['group_id']}__{task}", "tasks": task}
                for row in rows for task in row["tasks"].split(",")]
        groups_tsv = OUT_DIR / "matrix/groups_single.tsv"
        with groups_tsv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    plans = []
    dry_results = []
    for row in rows:
        job_name = f"{JOB_PREFIX}_t{row['timestep']}_l{row['layer']}_p{row['prefix']}"
        if args.layout == "single":
            job_name += f"_{row['tasks']}"
        if not re.fullmatch(r"[A-Za-z0-9_-]+", job_name):
            raise SystemExit(f"unsafe job name {job_name}")
        plans.append({"group_id": row["group_id"], "job_name": job_name, "worker_spec": worker_spec,
                      "tasks": row["tasks"], "strict_audit": row["strict_audit"],
                      "strict_audit_sha256": row["strict_audit_sha256"],
                      "command": group_command(row, groups_tsv, gpu_ids)})
        if not errors:
            dry_results.append(dry_run(row, groups_tsv, OUT_DIR / "launcher" / f"dryrun_{args.layout}_{stamp}",
                                       gpu_ids))
    errors += [f"{r['group_id']}: {p}" for r in dry_results for p in r["problems"]]
    atomic_json(OUT_DIR / f"launcher/plan_{args.layout}.json", {
        "schema": "TSTEP_MINIMAX_BF16_V1_RETRAIN_PLAN_V1", "run_id": RUN_ID, "generated_utc": now(),
        "layout": args.layout, "worker_spec": worker_spec,
        "groups_tsv": str(groups_tsv), "groups_tsv_sha256": sha256_file(groups_tsv),
        "group_script_sha256": sha256_file(GROUP_SCRIPT), "consolidation": why,
        "jobs": plans, "dry_run": dry_results, "errors": errors,
    })
    for plan in plans:
        print(f"  {plan['job_name']}  {worker_spec}  tasks={len(plan['tasks'].split(','))}")
    print(f"groups={len(plans)} runs={sum(len(p['tasks'].split(',')) for p in plans)} "
          f"dry_run_ok={sum(not r['problems'] for r in dry_results)}/{len(dry_results)} errors={len(errors)}")
    for error in errors[:20]:
        print("  ERROR", error)
    if errors:
        raise SystemExit(1)
    if not args.submit:
        print("dry run: no SCO jobs created (pass --submit --approval ...)")
        return 0
    if not args.approval.strip():
        raise SystemExit("--submit requires --approval recording the user's explicit stage-5 approval")
    ledger_path = OUT_DIR / "launcher/submissions.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {
        "schema": "TSTEP_MINIMAX_BF16_V1_RETRAIN_SUBMISSIONS_V1", "run_id": RUN_ID, "resource_pool": "zoetrope",
        "layout": args.layout, "worker_spec": worker_spec,
        "feature_transfer_precision": {"PROBE_FP16_PASSTHROUGH": "0"}, "jobs": {},
    }
    if ledger.get("layout") != args.layout:
        raise SystemExit(f"ledger was started with layout {ledger.get('layout')}; refusing to mix layouts")
    pending = [p for p in plans if p["group_id"] not in ledger["jobs"]]
    if args.limit > 0:
        pending = pending[: args.limit]
    if ledger["jobs"] and pending:
        # Pilot gate (user decision 2026-10-16): the rest is submitted only after
        # every probe of every job already submitted has a finite checkpoint.
        gate = pilot_gate(ledger["jobs"])
        atomic_json(OUT_DIR / "launcher/pilot_gate.json", gate)
        if gate["status"] != "PASS":
            raise SystemExit(f"pilot gate not passed: {gate['problems'][:8]}")
    ledger.setdefault("approvals", []).append({"approval": args.approval, "recorded_utc": now()})
    for plan in pending:
        argv = [str(SCO), "acp", "jobs", "create", "--workspace-name=aigc", "--aec2-name=zoetrope",
                f"--job-name={plan['job_name']}", "--priority=HIGHEST", "--quota-type=reserved",
                f"--container-image-url={CONTAINER_IMAGE}", f"--storage-mount={STORAGE_MOUNT}",
                "--training-framework=pytorch", "--worker-nodes=1", f"--worker-spec={worker_spec}",
                f"--command={plan['command']}"]
        done = subprocess.run(argv, text=True, capture_output=True, timeout=180)
        match = JOB_ID_RE.search(done.stdout + "\n" + done.stderr)
        rec = {**plan, "job_id": match.group(0) if match else None, "returncode": done.returncode,
               "stdout": done.stdout, "stderr": done.stderr, "submitted_utc": now()}
        ledger["jobs"][plan["group_id"]] = rec
        ledger["updated_utc"] = now()
        atomic_json(ledger_path, ledger)
        with (OUT_DIR / "launcher/sco_submit.log").open("a", encoding="utf-8") as handle:
            handle.write(f"[{rec['submitted_utc']}] {plan['job_name']} job_id={rec['job_id']} rc={done.returncode}\n")
        print(f"{plan['group_id']}\t{rec['job_id'] or 'SUBMIT_ERROR'}", flush=True)
        if done.returncode or not rec["job_id"]:
            raise SystemExit(f"submission failed for {plan['group_id']}; stopping")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
