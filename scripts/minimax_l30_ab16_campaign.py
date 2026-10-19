#!/usr/bin/env python3
"""Freeze, validate, and release MiniMax-H3 layer-30 A/B probes.

This campaign is deliberately isolated from the historical MiniMax queue.  It
uses only the eight remotely verified layer-30 streaming shards in the BF16 V2
namespace.  C probes are intentionally absent because they require target
features.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/mnt/umm/users/yinbaiqiao/probe_spatial")
BASE = ROOT / "artifacts/main_v2_probe_v2_816_20260917"
OUT = BASE / "minimax_l30_ab16_20260925_r1"
SOURCE = ROOT / "runtime/main_v2_probe_minimax_bf16_v2/source"
SOURCE_MANIFEST = SOURCE.parent / "source.sha256"
PYTHON = "/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python"
SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"
STREAM_ROOT = Path("/mnt/umm/users/yinbaiqiao/InsScene-15K/FEAT_STREAMING_CANONICAL_V2_MINIMAX_BF16_V2")
DATA_ROOT = Path("/mnt/umm/users/yinbaiqiao/InsScene-15K/data")
PRIORITY = BASE / "minimax_layer30_priority_20260924_r1"
ROUTE = BASE / "minimax_bf16_v2_production_route_v2/route_receipt.json"
PREFIXES = (8, 12, 16, 24)
PROBES = ("streaming_depth", "streaming_mask", "view_consistency", "ego_belief_v2")
NAMESPACE = "FEAT_STREAMING_CANONICAL_V2_MINIMAX_BF16_V2"
EXPECTED_UNITS = (1360, 1360, 1360, 1352, 1360, 1360, 1360, 1372)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, value: object, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if exclusive:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def route_manifests() -> dict[int, Path]:
    route = load(ROUTE)
    if route.get("status") != "PASS" or route.get("streaming_records") != 10884:
        raise RuntimeError("MiniMax BF16 V2 route is not the complete PASS route")
    selected: dict[int, Path] = {}
    for prefix in PREFIXES:
        spec = route["frozen_manifests"].get(str(prefix))
        if not spec:
            raise RuntimeError(f"route lacks prefix {prefix}")
        path = Path(spec["path"])
        if sha(path) != spec["sha256"] or int(spec["rows"]) != 2721:
            raise RuntimeError(f"route manifest mismatch for prefix {prefix}")
        selected[prefix] = path
    return selected


def validate_feature_receipts() -> dict:
    receipts = []
    for shard, expected in enumerate(EXPECTED_UNITS):
        completed = load(PRIORITY / f"streaming-{shard}-of-8" / "COMPLETE.json")
        runtime = load(PRIORITY / f"streaming-{shard}-of-8" / "runtime_status.json")
        if (completed.get("status") != "PASS_REMOTE_VERIFIED" or completed.get("mode") != "streaming"
                or int(completed.get("layer", -1)) != 30 or int(completed.get("units", -1)) != expected
                or int(runtime.get("confirmed", -1)) != expected or int(runtime.get("expected", -2)) != expected
                or not runtime.get("extract_finished")):
            raise RuntimeError(f"streaming receipt/runtime mismatch for shard {shard}")
        receipts.append(completed)
    route_sha = sha(ROUTE)
    if any(record.get("route_sha256") != route_sha for record in receipts):
        raise RuntimeError("layer-30 shard route SHA differs from current route")
    return {"route": str(ROUTE), "route_sha256": route_sha, "shards": receipts,
            "streaming_units": sum(EXPECTED_UNITS)}


def inventory_for(prefix: int, manifest: Path) -> list[str]:
    paths: list[str] = []
    for line in manifest.read_text().splitlines():
        row = json.loads(line)
        scene = str(row["scene_id"])
        window = int(row["streaming_window_id"])
        path = STREAM_ROOT / "minimax_h3/scannetpp" / scene / f"window_{window:04d}" / f"prefix_{prefix - 1:06d}" / "feature_t250_layer30.sft"
        paths.append(str(path))
    if len(paths) != 2721 or len(set(paths)) != 2721:
        raise RuntimeError(f"unexpected exact inventory cardinality for prefix {prefix}")
    return paths


def prepare() -> None:
    if OUT.exists():
        raise RuntimeError(f"refusing existing campaign directory: {OUT}")
    evidence = validate_feature_receipts()
    manifests = route_manifests()
    if sha(SOURCE_MANIFEST) != "ab8b6a0d0e8d68f01d35cc37ca5af52be0be6f4007f9557f25a6e8c872c46933":
        raise RuntimeError("immutable MiniMax probe source manifest SHA mismatch")
    cases = []
    for shard, prefix in enumerate(PREFIXES):
        inventory = inventory_for(prefix, manifests[prefix])
        inventory_path = OUT / "groups" / f"{shard:02d}" / "inventory.json"
        write(inventory_path, {"prefix": prefix, "paths": inventory}, exclusive=True)
        for probe in PROBES:
            config = SOURCE / "configs/experiment/inscene15k_streaming" / f"{probe}_minimax_h3_v1.yaml"
            if not config.is_file():
                raise RuntimeError(f"missing immutable experiment config: {config}")
            case_dir = OUT / "cases" / f"p{prefix}" / probe
            output = case_dir / "output"
            command = [PYTHON, "vidfm3d/train.py", f"experiment=inscene15k_streaming/{probe}_minimax_h3_v1",
                       "seed=42", "vfm_name=minimax_h3", "video_channels=5376", "feature_layer=30",
                       "feature_timestep=250", "feat_postfix=_t250_layer30", f"prefix_len={prefix}",
                       f"paths.log_dir={output}", "paths.run_folder_name=run", "logger=canary_csv",
                       "trainer.max_epochs=1", "++trainer.max_steps=3", "++trainer.limit_train_batches=3",
                       "++trainer.limit_val_batches=1", "++trainer.num_sanity_val_steps=0",
                       "++trainer.log_every_n_steps=1", "++trainer.enable_progress_bar=false",
                       "data.data_module.batch_size_per_device=4", "data.data_module.batch_size_per_device_val=2",
                       "data.data_module.num_workers=0", "data.data_module.num_workers_val=0"]
            if probe == "ego_belief_v2":
                command.append(f"streaming_hidden_prefix_lengths=[{prefix}]")
            environment = {
                "PYTHONPATH": str(SOURCE), "INSCENE_DATA_ROOT": str(DATA_ROOT),
                "INSCENE_LOCAL_SCENE_ROOT": str(DATA_ROOT / "processed_scannetpp_v2/processed_scannetpp_v2"),
                "INSCENE_MANIFEST": str(manifests[prefix]), "INSCENE_MANIFEST_SHA256": sha(manifests[prefix]),
                "INSCENE_TRUST_EXPANDED_MANIFEST": "1", "INSCENE_STREAMING_FEAT_ROOT": str(STREAM_ROOT),
                "PROBE_FAIL_ON_NONFINITE_LOSS": "1", "WANDB_MODE": "offline", "OMP_NUM_THREADS": "4",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            }
            case = {"schema": "MINIMAX_L30_AB16_CASE_V1", "unique_key": f"minimax_h3__l30__p{prefix}__{probe}__seed42",
                    "model": "minimax_h3", "layer": 30, "prefix": prefix, "probe": probe, "shard": shard,
                    "requires_target": False, "runtime": str(SOURCE), "config": str(config), "config_sha256": sha(config),
                    "manifest": str(manifests[prefix]), "manifest_sha256": sha(manifests[prefix]),
                    "inventory": str(inventory_path), "inventory_sha256": sha(inventory_path),
                    "environment": environment, "command": command}
            case_path = case_dir / "case.json"
            write(case_path, case, exclusive=True)
            cases.append({"unique_key": case["unique_key"], "prefix": prefix, "probe": probe, "shard": shard,
                          "case_path": str(case_path), "case_sha256": sha(case_path)})
    bundle = {"schema": "MINIMAX_L30_AB16_BUNDLE_V1", "status": "FROZEN_WAITING_VALIDATION", "created_utc": now(),
              "model": "minimax_h3", "layer": 30, "cases": cases, "case_count": len(cases),
              "source": str(SOURCE), "source_manifest": str(SOURCE_MANIFEST), "source_manifest_sha256": sha(SOURCE_MANIFEST),
              "feature_evidence": evidence, "validation_gate": {"inventory": "all 2,721 exact layer*prefix payloads per group",
              "training": "3 optimizer steps and at least 12 train samples with finite loss"}}
    write(OUT / "bundle.json", bundle, exclusive=True)
    write(OUT / "README.json", {"status": bundle["status"], "ready_cases": 16, "target_dependent_cases_excluded": 8,
                                  "streaming_units": 10884, "next": "submit-validation"}, exclusive=True)
    print(json.dumps({"status": bundle["status"], "cases": len(cases), "output": str(OUT)}, indent=2))


def check_header(path: Path) -> None:
    sidecar = Path(f"{path}.manifest.json")
    if not path.is_file() or not sidecar.is_file():
        raise RuntimeError(f"missing payload or sidecar: {path}")
    meta = load(sidecar)
    if (meta.get("feature_namespace") != NAMESPACE or meta.get("feature_storage_contract") != "MINIMAX_H3_FINITE_BF16_V1"
            or meta.get("feature_storage_dtype") != "torch.bfloat16" or int(meta.get("size_bytes", -1)) != path.stat().st_size):
        raise RuntimeError(f"sidecar contract mismatch: {path}")
    with path.open("rb") as handle:
        initial = handle.read(8)
        if len(initial) != 8:
            raise RuntimeError(f"short safetensors header: {path}")
        length = int.from_bytes(initial, "little")
        raw = handle.read(length)
    header = json.loads(raw)
    tensor = header.get("feat", {})
    if tensor.get("dtype") not in {"BF16", "BFLOAT16"} or tensor.get("shape") != [22, 15, 26, 5376]:
        raise RuntimeError(f"safetensors contract mismatch: {path}")


def finite_csv(path: Path) -> tuple[set[int], float | None, bool]:
    matches = list(path.glob("**/metrics.csv"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one metrics CSV, found {matches}")
    steps: set[int] = set()
    samples: float | None = None
    finite_loss = False
    for row in csv.DictReader(matches[0].open()):
        raw_loss = row.get("train/loss")
        if raw_loss not in (None, ""):
            loss = float(raw_loss)
            if not math.isfinite(loss):
                raise RuntimeError(f"nonfinite train loss {loss}")
            finite_loss = True
            steps.add(int(float(row["step"])))
        raw_samples = row.get("trainer/total_samples")
        if raw_samples not in (None, ""):
            samples = max(samples or 0.0, float(raw_samples))
    return steps, samples, finite_loss


def run_validation(shard: int, bundle_sha: str, runner_sha: str) -> None:
    if sha(Path(__file__)) != runner_sha:
        raise RuntimeError("campaign runner SHA mismatch")
    bundle_path = OUT / "bundle.json"
    if sha(bundle_path) != bundle_sha:
        raise RuntimeError("bundle SHA mismatch")
    bundle = load(bundle_path)
    entries = [entry for entry in bundle["cases"] if entry["shard"] == shard]
    if len(entries) != 4:
        raise RuntimeError(f"expected four cases for shard {shard}")
    group = OUT / "groups" / f"{shard:02d}"
    inventory_result = group / "inventory_result.json"
    if not inventory_result.exists():
        inventory = load(group / "inventory.json")
        for index, raw in enumerate(inventory["paths"], 1):
            check_header(Path(raw))
            if index % 250 == 0:
                write(group / "inventory_progress.json", {"checked": index, "total": len(inventory["paths"]), "updated_utc": now()})
        write(inventory_result, {"status": "PASS", "prefix": inventory["prefix"], "objects_checked": len(inventory["paths"]), "completed_utc": now()})
    for entry in entries:
        case_path = Path(entry["case_path"])
        if sha(case_path) != entry["case_sha256"]:
            raise RuntimeError(f"case SHA mismatch: {case_path}")
        case = load(case_path)
        result_path = case_path.parent / "result.json"
        if result_path.exists() and load(result_path).get("status") == "PASS":
            continue
        if sha(Path(case["config"])) != case["config_sha256"] or sha(Path(case["manifest"])) != case["manifest_sha256"]:
            raise RuntimeError(f"binding changed: {case['unique_key']}")
        output = case_path.parent / "output"
        output.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy(); env.update(case["environment"]); env["MPLCONFIGDIR"] = str(case_path.parent / "mpl")
        log = case_path.parent / "train.log"
        with log.open("w") as handle:
            completed = subprocess.run(case["command"], cwd=SOURCE, env=env, stdout=handle, stderr=subprocess.STDOUT, timeout=2400)
        result = {"schema": "MINIMAX_L30_AB16_VALIDATION_V1", "unique_key": case["unique_key"], "status": "FAIL",
                  "returncode": completed.returncode, "train_log": str(log), "train_log_sha256": sha(log), "ended_utc": now()}
        try:
            if completed.returncode:
                raise RuntimeError(f"training exited {completed.returncode}")
            steps, samples, finite_loss = finite_csv(output)
            sample_evidence = samples if samples is not None else len(steps) * 4
            if not ({0, 1, 2} <= steps and sample_evidence >= 12 and finite_loss):
                raise RuntimeError(f"gate failed: steps={sorted(steps)}, samples={sample_evidence}, finite_loss={finite_loss}")
            result.update(status="PASS", optimizer_steps=3, train_samples=sample_evidence,
                          checks={"full_inventory": "PASS", "real_loader": "PASS", "finite_loss": "PASS", "optimizer_steps": 3})
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        write(result_path, result)
    results = [load(Path(entry["case_path"]).parent / "result.json") for entry in entries]
    write(group / "completed.json", {"status": "PASS" if all(item["status"] == "PASS" for item in results) else "FAIL",
                                      "passed": sum(item["status"] == "PASS" for item in results), "total": len(results), "ended_utc": now()})


def submit_validation() -> None:
    bundle = load(OUT / "bundle.json")
    if bundle.get("status") != "FROZEN_WAITING_VALIDATION" or bundle.get("case_count") != 16:
        raise RuntimeError("bundle is not the frozen 16-case validation bundle")
    ledger_path = OUT / "validation_submission_ledger.json"
    ledger = load(ledger_path) if ledger_path.exists() else {"schema": "MINIMAX_L30_AB16_VALIDATION_SUBMISSIONS_V1", "jobs": {}}
    bundle_sha = sha(OUT / "bundle.json"); runner_sha = sha(Path(__file__))
    for shard in range(4):
        key = f"{shard:02d}"
        if key in ledger["jobs"]:
            continue
        command = (f"cd {ROOT} && {PYTHON} {Path(__file__)} run-validation --shard {shard} "
                   f"--bundle-sha {bundle_sha} --runner-sha {runner_sha}")
        intent = {"job_name": f"bq-mm-l30ab16-v{shard}-0925", "command": command, "state": "INTENT", "created_utc": now()}
        ledger["jobs"][key] = intent; write(ledger_path, ledger)
        argv = [SCO, "acp", "jobs", "create", "--workspace-name=aigc", "--aec2-name=zoetrope", f"--job-name={intent['job_name']}",
                "--priority=HIGHEST", "--quota-type=reserved", "--container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617",
                "--storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm", "--training-framework=pytorch", "--worker-nodes=1",
                "--worker-spec=N4lS.Iq.I80.1", f"--command={command}"]
        completed = subprocess.run(argv, text=True, capture_output=True, timeout=120)
        output = completed.stdout + completed.stderr; ids = list(dict.fromkeys(re.findall(r"pt-[a-z0-9]+", output)))
        intent.update(state="SUBMITTED" if completed.returncode == 0 and len(ids) == 1 else "FAILED", job_ids=ids,
                      returncode=completed.returncode, scheduler_output=output, submitted_utc=now())
        write(ledger_path, ledger)
        if intent["state"] != "SUBMITTED":
            raise RuntimeError(f"validation submission failed for shard {shard}: {output}")
    print(json.dumps(ledger, indent=2))


def release() -> None:
    bundle = load(OUT / "bundle.json")
    results = [load(Path(entry["case_path"]).parent / "result.json") for entry in bundle["cases"]]
    if len(results) != 16 or not all(result.get("status") == "PASS" for result in results):
        raise RuntimeError("all sixteen validation results must PASS before formal submission")
    ledger_path = OUT / "formal_submission_ledger.json"
    ledger = load(ledger_path) if ledger_path.exists() else {"schema": "MINIMAX_L30_AB16_FORMAL_SUBMISSIONS_V1", "jobs": {}}
    for entry in bundle["cases"]:
        case_path = Path(entry["case_path"]); case = load(case_path); key = case["unique_key"]
        if key in ledger["jobs"]:
            continue
        formal_output = case_path.parent / "formal_output"
        command_args = [arg for arg in case["command"] if not arg.startswith("paths.log_dir=") and "max_steps" not in arg and "limit_train_batches" not in arg and "limit_val_batches" not in arg and "num_sanity" not in arg and "log_every" not in arg and "enable_progress" not in arg and "batch_size_per_device=" not in arg and "num_workers=" not in arg]
        command_args.extend([f"paths.log_dir={formal_output}", "paths.run_folder_name=run", "logger.wandb.offline=true", "autoresume=false"])
        env = " ".join(f"{name}={value}" for name, value in case["environment"].items() if name not in {"PYTHONPATH", "MPLCONFIGDIR"})
        command = f"cd {SOURCE} && sha256sum --status -c {SOURCE_MANIFEST} && env PYTHONPATH={SOURCE} {env} " + " ".join(command_args)
        job_name = f"bq-mm-l30-{case['probe'][:4]}-p{case['prefix']}-0925"
        record = {"job_name": job_name, "command": command, "state": "INTENT", "created_utc": now(), "case_path": str(case_path)}
        ledger["jobs"][key] = record; write(ledger_path, ledger)
        argv = [SCO, "acp", "jobs", "create", "--workspace-name=aigc", "--aec2-name=zoetrope", f"--job-name={job_name}",
                "--priority=HIGHEST", "--quota-type=reserved", "--container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617",
                "--storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm", "--training-framework=pytorch", "--worker-nodes=1",
                "--worker-spec=N4lS.Iq.I80.1", f"--command={command}"]
        completed = subprocess.run(argv, text=True, capture_output=True, timeout=120)
        output = completed.stdout + completed.stderr; ids = list(dict.fromkeys(re.findall(r"pt-[a-z0-9]+", output)))
        record.update(state="SUBMITTED" if completed.returncode == 0 and len(ids) == 1 else "FAILED", job_ids=ids,
                      returncode=completed.returncode, scheduler_output=output, submitted_utc=now())
        write(ledger_path, ledger)
        if record["state"] != "SUBMITTED":
            raise RuntimeError(f"formal submission failed for {key}: {output}")
    print(json.dumps(ledger, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("prepare")
    sub.add_parser("submit-validation")
    validation = sub.add_parser("run-validation")
    validation.add_argument("--shard", type=int, required=True)
    validation.add_argument("--bundle-sha", required=True)
    validation.add_argument("--runner-sha", required=True)
    sub.add_parser("release")
    args = parser.parse_args()
    if args.action == "prepare": prepare()
    elif args.action == "submit-validation": submit_validation()
    elif args.action == "run-validation": run_validation(args.shard, args.bundle_sha, args.runner_sha)
    else: release()


if __name__ == "__main__":
    main()
