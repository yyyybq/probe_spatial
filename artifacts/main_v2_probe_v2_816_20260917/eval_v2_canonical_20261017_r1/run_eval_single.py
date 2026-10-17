#!/usr/bin/env python3
"""Evaluate one SHA-bound canonical-V2 checkpoint with the v2 evaluators on one GPU.

Each item names the training item, the training runner's environment, the frozen
evaluation pack (training pack + v2 evaluators, runtime/main_v2_eval_v2_from_*),
the checkpoint and the evaluator.  Everything is verified before running:
item hash, evaluation-pack checksum list, experiment-config hash, overlay and
execution-manifest hashes, checkpoint size.  The checkpoint's SHA-256 is
computed here and written next to the result.

Output: logs/inscene15k_streaming/runs/<RUN_ID>/<mode>/<key>/ with the
evaluator's eval/val_summary.json, eval.log and EVAL_COMPLETE.json.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import signal
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path("/mnt/umm/users/yinbaiqiao/probe_spatial")
CAMPAIGN = ROOT / "artifacts/main_v2_probe_v2_816_20260917/eval_v2_canonical_20261017_r1"
ITEM_ROOT = CAMPAIGN / "items"
STATE_ROOT = CAMPAIGN / "state"
RUN_ID = "main_v2_eval_v2_canonical_20261017_r1"
OUTPUT_ROOT = ROOT / "logs/inscene15k_streaming/runs" / RUN_ID
PYTHON = Path("/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python")
AOSS_CONF = "/mnt/umm/users/yinbaiqiao/aoss/petreloss.conf"
ENTRYPOINTS = {"diag": "vidfm3d/eval_diag.py", "a1": "vidfm3d/eval_a1_streaming.py"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def verify(item: dict, *, skip_ckpt_hash: bool) -> dict:
    pack = ROOT / "runtime" / item["eval_pack"]["name"]
    src = pack / "source"
    if sha256(pack / "source.sha256") != item["eval_pack"]["source_sha256_file_sha256"]:
        raise RuntimeError("evaluation pack checksum-file identity mismatch")
    if subprocess.run(["sha256sum", "--status", "-c", str(pack / "source.sha256")], cwd=src).returncode:
        raise RuntimeError("evaluation pack source verification failed")
    if json.loads((pack / "EVAL_PACK.json").read_text()).get("status") != "FROZEN":
        raise RuntimeError("evaluation pack is not FROZEN")
    train_item = Path(item["training_item"]["path"])
    if sha256(train_item) != item["training_item"]["sha256"]:
        raise RuntimeError("training item changed")
    config = src / "configs/experiment" / f"{item['experiment_config']}.yaml"
    if not config.is_file() or sha256(config) != item["experiment_config_sha256"]:
        raise RuntimeError("experiment config SHA mismatch")
    for key in ("overlay", "execution_manifest"):
        ref = item.get(key)
        if ref and sha256(Path(ref["path"])) != ref["sha256"]:
            raise RuntimeError(f"{key} SHA mismatch")
    ckpt = Path(item["checkpoint"]["path"])
    if not ckpt.is_file() or ckpt.stat().st_size != item["checkpoint"]["size_bytes"]:
        raise RuntimeError("checkpoint missing or size changed")
    return {"checkpoint_sha256": None if skip_ckpt_hash else sha256(ckpt), "eval_pack_source": str(src)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--item", type=Path, required=True)
    parser.add_argument("--item-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("canary", "production"), required=True)
    parser.add_argument("--validate-only", action="store_true", help="verify hashes on a login node; no GPU, no checkpoint hash")
    args = parser.parse_args()
    if args.run_id != RUN_ID:
        raise RuntimeError("run ID mismatch")
    item_path = args.item.resolve()
    if item_path.parent != ITEM_ROOT.resolve() or item_path.is_symlink():
        raise RuntimeError("item path is outside the frozen item root")
    if sha256(item_path) != args.item_sha256:
        raise RuntimeError("item SHA mismatch")
    rows = [json.loads(line) for line in item_path.read_text().splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError("one-item file does not contain exactly one row")
    item = rows[0]
    checked = verify(item, skip_ckpt_hash=args.validate_only)
    if args.validate_only:
        print(json.dumps({"status": "PASS", "unique_key": item["unique_key"], **checked}, sort_keys=True))
        return 0
    visible = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
    if not visible:
        probe = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
                               text=True, capture_output=True, check=False)
        visible = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
    if len(visible) != 1:
        raise RuntimeError(f"single-GPU evaluator requires one visible GPU, got {visible!r}")

    key = item["unique_key"]
    claim = STATE_ROOT / "claims" / args.mode / key
    claim.parent.mkdir(parents=True, exist_ok=True)
    claim.mkdir()  # raises if another writer holds it
    output = OUTPUT_ROOT / args.mode / key
    complete = output / "EVAL_COMPLETE.json"
    try:
        if complete.exists():
            raise RuntimeError(f"evaluation receipt already exists for {args.mode}/{key}")
        output.mkdir(parents=True, exist_ok=True)
        src = checked["eval_pack_source"]
        environment = os.environ.copy()
        environment.update(item["env"])
        environment.update({
            "CUDA_VISIBLE_DEVICES": visible[0],
            "PROJECT_ROOT": src,
            "PYTHONPATH": src,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PATH": str(PYTHON.parent) + ":" + environment.get("PATH", ""),
            "AOSS_CONF": AOSS_CONF,
            # Fresh label cache: never read or extend a training campaign's cache.
            "INSCENE_LABEL_CACHE_ROOT": str(STATE_ROOT / "label_cache" / key),
            "AOSS_CACHE_DIR": str(Path("/mnt/local_cache/inscene") / RUN_ID / key),
            "AOSS_FS_METRICS": str(output / "aoss_metrics.jsonl"),
            "AOSS_FS_METRICS_MODE": "aggregate",
            "MPLCONFIGDIR": str(Path("/tmp") / f"mpl-{RUN_ID}"),
            "OMP_NUM_THREADS": "4",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "WANDB_MODE": "offline",
        })
        overrides = [f"experiment={item['experiment_config']}"]
        overrides += [o for o in item["scientific_overrides"] if not o.startswith("autoresume=")]
        ckpt_text = item["checkpoint"]["path"]
        if "'" in ckpt_text:
            raise RuntimeError("checkpoint path contains a single quote")
        overrides += [
            # Lightning names contain '=' (epoch=49-step=...), so the value must be
            # a quoted Hydra string; unquoted it fails the override grammar.
            f"ckpt_path='{ckpt_text}'",
            "++eval_split=val",
            "paths.log_dir=/mnt/umm/users/yinbaiqiao/probe_spatial/logs",
            f"paths.run_folder_name={RUN_ID}/{args.mode}/{key}",
            f"job_name=eval_{key}",
            "data.data_module.num_workers=2",
            "data.data_module.num_workers_val=2",
        ]
        command = [str(PYTHON), ENTRYPOINTS[item["evaluator"]]] + overrides
        start = {
            "schema": "MAIN_V2_EVAL_V2_STATE_V1", "at": now(), "state": "STARTING", "mode": args.mode,
            "unique_key": key, "item_sha256": args.item_sha256, "evaluator": item["evaluator"],
            "eval_pack": item["eval_pack"], "training_code_tag": item["training_code_tag"],
            "checkpoint": {**item["checkpoint"], "sha256": checked["checkpoint_sha256"]},
            "command": command,
        }
        atomic_json(output / "START_RECEIPT.json", start)
        child = None

        def terminate(_signum: int, _frame: Any) -> None:
            if child is not None and child.poll() is None:
                child.send_signal(signal.SIGTERM)

        signal.signal(signal.SIGTERM, terminate)
        with (output / "eval.log").open("a", encoding="utf-8") as log:
            log.write(f"BEGIN {now()} key={key}\n")
            log.flush()
            child = subprocess.Popen(command, cwd=src, env=environment, stdout=log, stderr=subprocess.STDOUT)
            rc = child.wait()
            log.write(f"END {now()} rc={rc}\n")
        summaries = sorted(glob.glob(str(output / "**" / "val_summary.json"), recursive=True))
        receipt = {**start, "at": now(), "returncode": rc, "summary_paths": summaries}
        if rc == 0 and len(summaries) == 1:
            summary = json.loads(Path(summaries[0]).read_text())
            receipt.update({
                "state": "SUCCEEDED",
                "summary_sha256": sha256(Path(summaries[0])),
                "evaluator_version": summary.get("evaluator_version"),
                "n": summary.get("n"),
                "n_nonfinite_pred": summary.get("n_nonfinite_pred"),
                "primary_metric": item["primary_metric"],
                "primary_value": summary.get(item["primary_metric"]),
            })
            atomic_json(complete, receipt)
        else:
            receipt["state"] = "FAILED"
            atomic_json(output / "EVAL_FAILED.json", receipt)
            rc = rc or 2
        return rc
    finally:
        claim.rmdir()


if __name__ == "__main__":
    raise SystemExit(main())
