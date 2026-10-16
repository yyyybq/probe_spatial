#!/usr/bin/env python3
"""Re-train the 16 canonical-V2 B2 (ego_belief_v2) cells with PROBE_FP16_PASSTHROUGH=0.

Background (docs/MINIMAX_REEXTRACT_RETRAIN_PLAN_20261016.md, stage 4): the
canonical-V2 B2 heads of CogVideoX L0/L5/L10 (12 cells, t749) and Aether L1
(4 cells, t749) went non-finite under fp16 pass-through in the 2026-09 single-GPU
campaigns.  The 2026-09-28 single_gpu_cog12_ego_fp32 campaign proved that the
same packs train finite heads with PROBE_FP16_PASSTHROUGH=0.

This script assembles a new SHA-bound campaign directory
(artifacts/main_v2_probe_v2_816_20260917/single_gpu_b2_fp32_20261016_r1/):
  items/           the 16 original queue items, copied byte-for-byte
  run_single.py    the one-GPU runner (pack/overlay table, PROBE_FP16_PASSTHROUGH=0)
  plan.json        item hashes, source campaigns, runner hash, job commands
  submissions.json SCO ledger (written by --submit)
Each SCO job runs the runner twice on one GPU: ``--canary`` (2 steps, verifies
pack/overlay/config/item hashes end to end) and then production.  Outputs land in
logs/inscene15k_streaming/runs/main_v2_single_gpu_b2_fp32_20261016_r1/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/main_v2_probe_v2_816_20260917"
CAMPAIGN = BASE / "single_gpu_b2_fp32_20261016_r1"
RUN_ID = "main_v2_single_gpu_b2_fp32_20261016_r1"
RUNNER = CAMPAIGN / "run_single.py"
QUEUE_CSV = ROOT / "metadata/release_inventory/retrain_queue_20261016.csv"
QUEUE_GROUP = "B2_NONFINITE_WEIGHTS_FP16_PASSTHROUGH"
PYTHON = "/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python"
SCO = Path.home() / ".sco/bin/sco"
CONTAINER_IMAGE = "registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617"
STORAGE_MOUNT = "019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm"
WORKER_SPEC = "N4lS.Iq.I80.1"
JOB_ID_RE = re.compile(r"pt-[a-z0-9]+")
SOURCE_ITEMS = {
    "cogvideox": BASE / "single_gpu_cog84_h800_20260928_r1/items",
    "aether": BASE / "single_gpu_aether24_20260923_r1/items",
}
VALIDATION_CASES = {
    "cogvideox": BASE / "probe_validation_cog96_20260924_r1/cases",
    "aether": BASE / "probe552_validation_aether_target8_20260923_r4/cases",
}


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


def load_queue() -> dict[str, dict]:
    cells: dict[str, dict] = {}
    with QUEUE_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["queue_group"] != QUEUE_GROUP or row["inventory"] != "canonical_v2":
                continue
            cells[row["experiment_id"]] = row
    if len(cells) != 16:
        raise SystemExit(f"expected 16 canonical B2 cells in the queue, found {len(cells)}")
    return cells


def collect_items(cells: dict[str, dict]) -> list[dict]:
    items: list[dict] = []
    for model, item_dir in SOURCE_ITEMS.items():
        for path in sorted(item_dir.glob("*.jsonl")):
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(rows) != 1:
                raise SystemExit(f"{path}: not a one-item file")
            item = rows[0]
            if item["probe"] != "ego_belief_v2" or item["model"] != model:
                continue
            key = item["unique_key"]
            if key not in cells:
                raise SystemExit(f"{path}: {key} is not in the retrain queue")
            cell = cells[key]
            if (str(item["layer"]), str(item["prefix"]), str(item["timestep"])) != (
                cell["layer"], cell["prefix"], cell["timestep"]
            ):
                raise SystemExit(f"{path}: queue/item coordinate mismatch for {key}")
            if "PROBE_FP16_PASSTHROUGH" in json.dumps(item):
                raise SystemExit(f"{path}: item pins a transfer precision; refusing")
            case_dir = VALIDATION_CASES[model] / key
            items.append({
                "unique_key": key,
                "model": model,
                "probe": item["probe"],
                "layer": item["layer"],
                "prefix": item["prefix"],
                "timestep": item["timestep"],
                "experiment_config": item["experiment_config"],
                "experiment_config_sha256": item["experiment_config_sha256"],
                "source_item_path": str(path),
                "source_item_sha256": sha256(path),
                "source_campaign": item_dir.parent.name,
                "validation_case": str(case_dir / "case.json") if (case_dir / "case.json").is_file() else None,
                "old_output_path": cell.get("old_output_path"),
                "old_checkpoint_path": cell.get("old_checkpoint_path"),
            })
    found = {it["unique_key"] for it in items}
    missing = sorted(set(cells) - found)
    if missing:
        raise SystemExit(f"queue cells without a source item: {missing}")
    items.sort(key=lambda it: (it["model"], int(it["layer"]), int(it["prefix"])))
    return items


def materialise(items: list[dict]) -> None:
    item_root = CAMPAIGN / "items"
    item_root.mkdir(parents=True, exist_ok=True)
    wanted = set()
    for ordinal, it in enumerate(items):
        dest = item_root / Path(it["source_item_path"]).name
        wanted.add(dest.name)
        if dest.exists():
            if sha256(dest) != it["source_item_sha256"]:
                raise SystemExit(f"{dest} exists with a different hash; refusing to overwrite")
        else:
            shutil.copyfile(it["source_item_path"], dest)
            os.chmod(dest, 0o444)
        it["ordinal"] = ordinal
        it["item_path"] = str(dest)
        it["item_sha256"] = sha256(dest)
        if it["item_sha256"] != it["source_item_sha256"]:
            raise SystemExit(f"copy hash mismatch for {dest}")
    extra = sorted(p.name for p in item_root.glob("*.jsonl") if p.name not in wanted)
    if extra:
        raise SystemExit(f"unexpected files in {item_root}: {extra}")


def build_commands(items: list[dict], runner_sha: str) -> None:
    for it in items:
        single = [PYTHON, str(RUNNER), "--item", it["item_path"], "--item-sha256", it["item_sha256"], "--run-id", RUN_ID]
        canary = " ".join(single + ["--canary"])
        production = " ".join(single)
        it["command"] = f"{canary} && {production}"
        it["job_name"] = f"bq-b2fp32-{it['ordinal']:02d}-{it['item_sha256'][:6]}"
        if not re.fullmatch(r"[A-Za-z0-9_-]+", it["job_name"]):
            raise SystemExit(f"unsafe job name {it['job_name']!r}")
        it["runner_sha256"] = runner_sha


def validate_items(items: list[dict]) -> None:
    for it in items:
        completed = subprocess.run(
            [PYTHON, str(RUNNER), "--item", it["item_path"], "--item-sha256", it["item_sha256"],
             "--run-id", RUN_ID, "--validate-only", "--skip-gpu-check"],
            text=True, capture_output=True,
        )
        if completed.returncode:
            raise SystemExit(f"runner validation failed for {it['unique_key']}:\n{completed.stdout}\n{completed.stderr}")
        it["validate_only"] = json.loads(completed.stdout.strip().splitlines()[-1])


def submit(items: list[dict], limit: int) -> None:
    ledger_path = CAMPAIGN / "submissions.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {
        "schema": "B2_FP32_CANONICAL_RETRAIN_SUBMISSIONS_V1",
        "run_id": RUN_ID,
        "resource_pool": "zoetrope",
        "worker_spec": WORKER_SPEC,
        "feature_transfer_precision": {"PROBE_FP16_PASSTHROUGH": "0"},
        "jobs": {},
    }
    pending = [it for it in items if it["unique_key"] not in ledger["jobs"]]
    if limit > 0:
        pending = pending[:limit]
    print(f"pending={len(pending)} recorded={len(ledger['jobs'])}")
    for it in pending:
        argv = [
            str(SCO), "acp", "jobs", "create", "--workspace-name=aigc", "--aec2-name=zoetrope",
            f"--job-name={it['job_name']}", "--priority=HIGHEST", "--quota-type=reserved",
            f"--container-image-url={CONTAINER_IMAGE}", f"--storage-mount={STORAGE_MOUNT}",
            "--training-framework=pytorch", "--worker-nodes=1", f"--worker-spec={WORKER_SPEC}",
            f"--command={it['command']}",
        ]
        completed = subprocess.run(argv, text=True, capture_output=True, timeout=180)
        text = completed.stdout + "\n" + completed.stderr
        match = JOB_ID_RE.search(text)
        record = {
            **it,
            "job_id": match.group(0) if match else None,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "submitted_utc": now(),
        }
        ledger["jobs"][it["unique_key"]] = record
        ledger["updated_utc"] = now()
        atomic_json(ledger_path, ledger)
        print(f"{it['unique_key']}\t{record['job_id'] or 'SUBMIT_ERROR'}", flush=True)
        if completed.returncode or not record["job_id"]:
            print(text, flush=True)
            raise SystemExit(f"submission failed for {it['unique_key']}; stopping")
    ids = [job["job_id"] for job in ledger["jobs"].values() if job.get("job_id")]
    (CAMPAIGN / "sco_job_ids.txt").write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
    print(json.dumps({"recorded": len(ledger["jobs"]), "job_ids": len(ids)}))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--submit", action="store_true", help="create the SCO jobs (default: dry run)")
    parser.add_argument("--limit", type=int, default=0, help="submit at most N pending jobs (pilot)")
    args = parser.parse_args()
    if not RUNNER.is_file():
        raise SystemExit(f"missing runner {RUNNER}; copy it into the campaign dir first")
    runner_text = RUNNER.read_text(encoding="utf-8")
    if '"PROBE_FP16_PASSTHROUGH": "0"' not in runner_text or f'RUN_ID = "{RUN_ID}"' not in runner_text:
        raise SystemExit("runner does not pin PROBE_FP16_PASSTHROUGH=0 / the campaign RUN_ID")
    runner_sha = sha256(RUNNER)
    cells = load_queue()
    items = collect_items(cells)
    materialise(items)
    build_commands(items, runner_sha)
    validate_items(items)
    plan = {
        "schema": "B2_FP32_CANONICAL_RETRAIN_PLAN_V1",
        "run_id": RUN_ID,
        "generated_utc": now(),
        "queue_csv": str(QUEUE_CSV),
        "queue_csv_sha256": sha256(QUEUE_CSV),
        "runner": str(RUNNER),
        "runner_sha256": runner_sha,
        "feature_transfer_precision": {"PROBE_FP16_PASSTHROUGH": "0"},
        "job_shape": "one GPU; --canary (2 steps) then production in the same job",
        "jobs": items,
    }
    atomic_json(CAMPAIGN / "plan.json", plan)
    print(f"items={len(items)} plan={CAMPAIGN / 'plan.json'} runner_sha256={runner_sha}")
    for it in items:
        print(f"  {it['job_name']}  {it['unique_key']}  case={'yes' if it['validation_case'] else 'no'}")
    if not args.submit:
        print("dry run: no SCO jobs created (pass --submit)")
        return 0
    if not SCO.is_file():
        raise SystemExit(f"sco CLI not found at {SCO}")
    submit(items, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
