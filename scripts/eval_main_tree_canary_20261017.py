#!/usr/bin/env python3
"""Evaluate canonical-V2 checkpoints from a committed main-tree checkout.

Prediction canary for releasing from the main tree: the same checkpoints, the
same evaluation items and the same training environment as the frozen-pack
evaluation (eval_v2_canonical_20261017_r2), but the code is a main-tree commit
instead of runtime/main_v2_eval_v2r2_from_*.  `compare` then checks that the
per-sample predictions and the summaries agree with the frozen-pack evaluation.

  run      --key KEY   (on a one-GPU node) evaluate one item from the main tree
  submit   --keys ...  [--submit]          one-GPU Zoetrope job per key
  compare  [--keys ...]                    main tree vs frozen pack, per sample

The main checkout must be clean in vidfm3d/ and configs/ and at the commit
recorded in CAMPAIGN.json; the runner refuses otherwise.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import shlex
import signal
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path("/mnt/umm/users/yinbaiqiao/probe_spatial")
PACK_CAMPAIGN = ROOT / "artifacts/main_v2_probe_v2_816_20260917/eval_v2_canonical_20261017_r2"
PACK_RUNS = ROOT / "logs/inscene15k_streaming/runs/main_v2_eval_v2_canonical_20261017_r2"
CAMPAIGN = ROOT / "artifacts/main_v2_probe_v2_816_20260917/eval_v2_main_tree_canary_20261017"
RUN_ID = "main_v2_eval_v2_main_tree_canary_20261017"
OUTPUT_ROOT = ROOT / "logs/inscene15k_streaming/runs" / RUN_ID
PYTHON = Path("/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python")
AOSS_CONF = "/mnt/umm/users/yinbaiqiao/aoss/petreloss.conf"
ENTRYPOINTS = {"diag": "vidfm3d/eval_diag.py", "a1": "vidfm3d/eval_a1_streaming.py"}
SCO = Path.home() / ".sco/bin/sco"
IMAGE = "registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617"
MOUNT = "019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm"
JOB_ID_RE = re.compile(r"pt-[a-z0-9]+")
# Summary keys that legitimately differ between two runs of the same evaluation.
VOLATILE = re.compile(r"(time|seconds|path|dir|host|device|cuda|gpu|date|started|finished|"
                      r"execution|command|argv|ckpt|checkpoint|git|code|pid)", re.I)


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
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        handle.write("\n")
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def campaign() -> dict:
    return json.loads((CAMPAIGN / "CAMPAIGN.json").read_text())


def load_item(key: str) -> tuple[dict, dict]:
    entry = json.loads((PACK_CAMPAIGN / "plan.json").read_text())["items"][key]
    path = Path(entry["path"])
    if sha256(path) != entry["sha256"]:
        raise RuntimeError(f"item SHA mismatch for {key}")
    return json.loads(path.read_text().splitlines()[0]), entry


CODE_PATHS = ("vidfm3d", "configs")
CODE_SCRIPTS = ("scripts/eval_main_tree_canary_20261017.py",
                "scripts/check_loader_parity_main_vs_pack_20261017.py")


def _code_files(code_root: Path) -> list[str]:
    """Every file that can change behaviour: vidfm3d/, configs/ and the two canary scripts."""
    files = [str(p.relative_to(code_root)) for top in CODE_PATHS for p in (code_root / top).rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    return sorted(files + list(CODE_SCRIPTS))


def verify_code_manifest(code_root: Path, manifest: Path, manifest_sha256: str) -> None:
    """Check the checkout against the SHA-256 list frozen at the campaign commit (no git needed)."""
    if sha256(manifest) != manifest_sha256:
        raise RuntimeError("code manifest changed")
    listed = sorted(line.split("  ", 1)[1] for line in manifest.read_text().splitlines() if line)
    if listed != _code_files(code_root):
        extra = sorted(set(_code_files(code_root)) - set(listed))
        missing = sorted(set(listed) - set(_code_files(code_root)))
        raise RuntimeError(f"code file set differs from the manifest: extra={extra[:5]} missing={missing[:5]}")
    if subprocess.run(["sha256sum", "--status", "-c", str(manifest)], cwd=code_root).returncode:
        raise RuntimeError("main checkout differs from the frozen code manifest")


def verify_main(code_root: Path, camp: dict) -> dict:
    verify_code_manifest(code_root, Path(camp["code_manifest"]), camp["code_manifest_sha256"])
    return {"commit": camp["commit"], "code_root": str(code_root)}


def freeze(args: argparse.Namespace) -> int:
    """Login node: bind the campaign to a commit whose code equals the checkout."""
    code_root = Path(args.code_root).resolve()

    def git(*cmd: str) -> str:
        return subprocess.run(["git", "-C", str(code_root), *cmd], text=True,
                              capture_output=True, check=True).stdout.strip()

    commit = git("rev-parse", args.commit)
    paths = [*CODE_PATHS, *CODE_SCRIPTS]
    if git("diff", "--name-only", commit, "--", *paths) or git(
            "status", "--porcelain", "--untracked-files=all", "--", *paths):
        raise SystemExit("checkout code differs from the commit; commit first")
    tracked = sorted(git("ls-files", "--", *paths).splitlines())
    if tracked != _code_files(code_root):
        raise SystemExit("checkout holds ignored/untracked files under vidfm3d or configs")
    if (CAMPAIGN / "CAMPAIGN.json").exists() and not args.force:
        raise SystemExit("CAMPAIGN.json exists; pass --force to rebind")
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    manifest = CAMPAIGN / f"code_{commit[:12]}.sha256"
    manifest.write_text("".join(f"{sha256(code_root / f)}  {f}\n" for f in tracked))
    keys = [line.strip() for line in Path(args.keys_file).read_text().splitlines() if line.strip()]
    for key in keys:
        load_item(key)
    atomic_json(CAMPAIGN / "CAMPAIGN.json", {
        "schema": "MAIN_TREE_EVAL_CANARY_CAMPAIGN_V1", "created": now(), "commit": commit,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"), "code_root": str(code_root),
        "code_manifest": str(manifest), "code_manifest_sha256": sha256(manifest),
        "n_code_files": len(tracked), "keys": keys,
        "pack_evaluation": str(PACK_CAMPAIGN), "run_id": RUN_ID})
    print(f"frozen {commit} files={len(tracked)} keys={len(keys)}")
    return 0


# ---------------------------------------------------------------------- run
def run(args: argparse.Namespace) -> int:
    camp = campaign()
    code_root = Path(camp["code_root"])
    verify_main(code_root, camp)
    item, entry = load_item(args.key)
    ckpt = Path(item["checkpoint"]["path"])
    if not ckpt.is_file() or ckpt.stat().st_size != item["checkpoint"]["size_bytes"]:
        raise RuntimeError("checkpoint missing or size changed")
    for ref_key in ("overlay", "execution_manifest"):
        ref = item.get(ref_key)
        if ref and sha256(Path(ref["path"])) != ref["sha256"]:
            raise RuntimeError(f"{ref_key} SHA mismatch")
    config = code_root / "configs/experiment" / f"{item['experiment_config']}.yaml"
    config_sha = sha256(config) if config.is_file() else None
    visible = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
    if not visible:
        probe = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
                               text=True, capture_output=True, check=False)
        visible = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
    if len(visible) != 1:
        raise RuntimeError(f"single-GPU evaluator requires one visible GPU, got {visible!r}")
    key = item["unique_key"]
    output = OUTPUT_ROOT / key
    complete = output / "EVAL_COMPLETE.json"
    if complete.exists():
        raise RuntimeError(f"evaluation receipt already exists for {key}")
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(item["env"])
    environment.update({
        "CUDA_VISIBLE_DEVICES": visible[0], "PROJECT_ROOT": str(code_root),
        "PYTHONPATH": str(code_root), "PYTHONDONTWRITEBYTECODE": "1",
        "PATH": str(PYTHON.parent) + ":" + environment.get("PATH", ""), "AOSS_CONF": AOSS_CONF,
        "INSCENE_LABEL_CACHE_ROOT": str(CAMPAIGN / "state/label_cache" / key),
        "AOSS_CACHE_DIR": str(Path("/mnt/local_cache/inscene") / RUN_ID / key),
        "AOSS_FS_METRICS": str(output / "aoss_metrics.jsonl"), "AOSS_FS_METRICS_MODE": "aggregate",
        "MPLCONFIGDIR": str(Path("/tmp") / f"mpl-{RUN_ID}"), "OMP_NUM_THREADS": "4",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "WANDB_MODE": "offline",
    })
    overrides = [f"experiment={item['experiment_config']}"]
    overrides += [o for o in item["scientific_overrides"] if not o.startswith("autoresume=")]
    if "'" in str(ckpt):
        raise RuntimeError("checkpoint path contains a single quote")
    overrides += [f"ckpt_path='{ckpt}'", "++eval_split=val",
                  "paths.log_dir=/mnt/umm/users/yinbaiqiao/probe_spatial/logs",
                  f"paths.run_folder_name={RUN_ID}/{key}", f"job_name=eval_main_{key}",
                  "data.data_module.num_workers=2", "data.data_module.num_workers_val=2"]
    command = [str(PYTHON), ENTRYPOINTS[item["evaluator"]]] + overrides
    start = {"schema": "MAIN_TREE_EVAL_CANARY_STATE_V1", "at": now(), "state": "STARTING",
             "unique_key": key, "item_sha256": entry["sha256"], "evaluator": item["evaluator"],
             "main_commit": camp["commit"], "code_manifest_sha256": camp["code_manifest_sha256"], "code_root": str(code_root),
             "experiment_config_sha256": {"main": config_sha, "pack": item["experiment_config_sha256"]},
             "training_code_tag": item["training_code_tag"],
             "checkpoint": {**item["checkpoint"], "sha256": sha256(ckpt)}, "command": command}
    atomic_json(output / "START_RECEIPT.json", start)
    child = None

    def terminate(_signum: int, _frame: Any) -> None:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, terminate)
    with (output / "eval.log").open("a", encoding="utf-8") as log:
        log.write(f"BEGIN {now()} key={key}\n")
        log.flush()
        child = subprocess.Popen(command, cwd=code_root, env=environment, stdout=log, stderr=subprocess.STDOUT)
        rc = child.wait()
        log.write(f"END {now()} rc={rc}\n")
    summaries = sorted(glob.glob(str(output / "**" / "val_summary.json"), recursive=True))
    receipt = {**start, "at": now(), "returncode": rc, "summary_paths": summaries}
    if rc == 0 and len(summaries) == 1:
        receipt.update({"state": "SUCCEEDED", "summary_sha256": sha256(Path(summaries[0]))})
        atomic_json(complete, receipt)
        return 0
    receipt["state"] = "FAILED"
    atomic_json(output / "EVAL_FAILED.json", receipt)
    return rc or 2


# ------------------------------------------------------------------- submit
def submit(args: argparse.Namespace) -> int:
    camp = campaign()
    verify_main(Path(camp["code_root"]), camp)
    ledger_path = CAMPAIGN / "submissions.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
        "schema": "MAIN_TREE_EVAL_CANARY_SUBMISSIONS_V1", "run_id": RUN_ID, "jobs": {}}
    done = {k for job in ledger["jobs"].values() for k in job["keys"]}
    keys = [k for k in (args.keys or camp["keys"]) if k not in done or args.retry]
    for key in keys:
        load_item(key)
    script = Path(camp["code_root"]) / "scripts" / Path(__file__).name
    batches = [keys[i:i + args.per_job] for i in range(0, len(keys), args.per_job)]
    print(f"items={len(keys)} jobs={len(batches)} commit={camp['commit']}")
    for batch in batches:
        steps = [shlex.join([str(PYTHON), str(script), "run", "--key", key]) + " || rc=1"
                 for key in batch]
        command = f"cd {shlex.quote(camp['code_root'])} && rc=0; " + "; ".join(steps) + "; exit $rc"
        name = f"bq-evalmain-{len(ledger['jobs']):04d}"
        if not args.submit:
            print(f"  {name}: {', '.join(batch)}")
            continue
        argv = [str(SCO), "acp", "jobs", "create", "--workspace-name=aigc", "--aec2-name=zoetrope",
                f"--job-name={name}", "--priority=HIGHEST", "--quota-type=reserved",
                f"--container-image-url={IMAGE}", f"--storage-mount={MOUNT}",
                "--training-framework=pytorch", "--worker-nodes=1", "--worker-spec=N4lS.Iq.I80.1",
                f"--command={command}"]
        result = subprocess.run(argv, text=True, capture_output=True, timeout=180)
        match = JOB_ID_RE.search(result.stdout + result.stderr)
        ledger["jobs"][name] = {"keys": batch, "job_id": match.group(0) if match else None,
                                "returncode": result.returncode, "command": command,
                                "commit": camp["commit"], "submitted_utc": now()}
        atomic_json(ledger_path, ledger)
        print(f"  {name}\t{ledger['jobs'][name]['job_id'] or 'SUBMIT_ERROR'}\t{', '.join(batch)}", flush=True)
        if result.returncode or not match:
            print(result.stdout, result.stderr)
            raise SystemExit("submission failed; stopping")
    return 0


# ------------------------------------------------------------------ compare
def _bits(t):
    import torch

    t = t.contiguous()
    view = {torch.float16: torch.int16, torch.bfloat16: torch.int16,
            torch.float32: torch.int32, torch.float64: torch.int64}.get(t.dtype)
    return t.view(view) if view is not None else t


def _diff(a, b) -> tuple[bool, float]:
    """(bitwise equal, max |a-b| over finite entries; inf if shapes/types differ)."""
    import torch

    if torch.is_tensor(a) or torch.is_tensor(b):
        if not (torch.is_tensor(a) and torch.is_tensor(b)) or a.shape != b.shape or a.dtype != b.dtype:
            return False, math.inf
        if torch.equal(_bits(a), _bits(b)):
            return True, 0.0
        if not a.is_floating_point():
            return False, math.inf
        both = torch.isfinite(a) & torch.isfinite(b)
        if not torch.equal(torch.isfinite(a), torch.isfinite(b)):
            return False, math.inf
        return False, float((a[both].double() - b[both].double()).abs().max()) if both.any() else 0.0
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False, math.inf
        parts = [_diff(x, y) for x, y in zip(a, b)]
        return all(p[0] for p in parts), max((p[1] for p in parts), default=0.0)
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False, math.inf
        parts = [_diff(a[k], b[k]) for k in a]
        return all(p[0] for p in parts), max((p[1] for p in parts), default=0.0)
    if isinstance(a, float) and isinstance(b, float):
        if a == b or (math.isnan(a) and math.isnan(b)):
            return True, 0.0
        return False, abs(a - b) if math.isfinite(a) and math.isfinite(b) else math.inf
    return a == b, (0.0 if a == b else math.inf)


def _pack_output(key: str) -> Path | None:
    for mode in ("production", "canary"):
        out = PACK_RUNS / mode / key
        if (out / "EVAL_COMPLETE.json").is_file():
            return out
    return None


def _one(path: Path, name: str) -> Path:
    found = sorted(path.glob(f"**/{name}"))
    if len(found) != 1:
        raise RuntimeError(f"expected one {name} under {path}, found {len(found)}")
    return found[0]


def compare_one(key: str) -> dict:
    import torch

    main_out, pack_out = OUTPUT_ROOT / key, _pack_output(key)
    result: dict[str, Any] = {"unique_key": key}
    if not (main_out / "EVAL_COMPLETE.json").is_file():
        return {**result, "status": "MAIN_NOT_EVALUATED"}
    if pack_out is None:
        return {**result, "status": "PACK_NOT_EVALUATED"}
    main_receipt = json.loads((main_out / "EVAL_COMPLETE.json").read_text())
    pack_receipt = json.loads((pack_out / "EVAL_COMPLETE.json").read_text())
    result.update({"pack_output": str(pack_out), "main_output": str(main_out),
                   "checkpoint_sha256_equal": main_receipt["checkpoint"]["sha256"]
                   == pack_receipt["checkpoint"]["sha256"]})
    main_records = torch.load(_one(main_out, "val_predictions.pt"), map_location="cpu", weights_only=False)
    pack_records = torch.load(_one(pack_out, "val_predictions.pt"), map_location="cpu", weights_only=False)
    result["n_records"] = {"main": len(main_records), "pack": len(pack_records)}
    mismatched: dict[str, dict] = {}
    only_main, only_pack = set(), set()
    for i, (m, p) in enumerate(zip(main_records, pack_records)):
        only_main |= set(m) - set(p)
        only_pack |= set(p) - set(m)
        for field in set(m) & set(p):
            equal, delta = _diff(m[field], p[field])
            if not equal:
                entry = mismatched.setdefault(field, {"n": 0, "first_record": i, "max_abs_diff": 0.0})
                entry["n"] += 1
                entry["max_abs_diff"] = max(entry["max_abs_diff"], delta)
    # Fields the frozen-pack loader never emitted appear as None there.
    benign_none = {f for f, v in mismatched.items()
                   if all(r.get(f) is None for r in pack_records)}
    # The frozen-pack loader emitted no anchor-isolated feature, so the evaluator
    # recorded its validity flag with the default False.
    if "anchor_isolated_valid" in mismatched and all(r.get("anchor_isolated") is None for r in pack_records) \
            and not any(r.get("anchor_isolated_valid") for r in pack_records):
        benign_none.add("anchor_isolated_valid")
    result.update({"mismatched_fields": mismatched, "fields_only_main": sorted(only_main),
                   "fields_only_pack": sorted(only_pack),
                   "fields_main_adds_where_pack_had_none": sorted(benign_none)})
    main_summary = json.loads(_one(main_out, "val_summary.json").read_text())
    pack_summary = json.loads(_one(pack_out, "val_summary.json").read_text())
    summary_mismatch = {}
    for name in sorted(set(main_summary) & set(pack_summary)):
        if VOLATILE.search(name):
            continue
        equal, delta = _diff(main_summary[name], pack_summary[name])
        if not equal:
            summary_mismatch[name] = {"main": main_summary[name], "pack": pack_summary[name],
                                      "max_abs_diff": delta}
    result.update({"summary_mismatch": summary_mismatch,
                   "summary_only_main": sorted(set(main_summary) - set(pack_summary)),
                   "summary_only_pack": sorted(set(pack_summary) - set(main_summary))})
    hard = {f: v for f, v in mismatched.items() if f not in benign_none}
    exact = (result["checkpoint_sha256_equal"] and len(main_records) == len(pack_records)
             and not hard and not only_pack and not summary_mismatch
             and not result["summary_only_pack"])
    result["status"] = "IDENTICAL" if exact else "DIFFERENT"
    return result


def compare(args: argparse.Namespace) -> int:
    camp = campaign()
    keys = args.keys or camp["keys"]
    rows = {key: compare_one(key) for key in keys}
    for key, row in rows.items():
        print(f"{row['status']}\t{key}")
    counts: dict[str, int] = {}
    for row in rows.values():
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    atomic_json(CAMPAIGN / "COMPARISON.json", {"schema": "MAIN_TREE_VS_PACK_PREDICTIONS_V1",
                                               "at": now(), "main_commit": camp["commit"],
                                               "counts": counts, "items": rows})
    print(json.dumps(counts))
    return 0 if counts.get("IDENTICAL", 0) == len(keys) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--key", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--keys", nargs="*", default=[])
    s.add_argument("--submit", action="store_true")
    s.add_argument("--retry", action="store_true")
    s.add_argument("--per-job", type=int, default=2)
    f = sub.add_parser("freeze")
    f.add_argument("--code-root", default=str(ROOT / ".worktrees/release-main-overlay"))
    f.add_argument("--commit", default="HEAD")
    f.add_argument("--keys-file", required=True)
    f.add_argument("--force", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("--keys", nargs="*", default=[])
    args = parser.parse_args()
    return {"run": run, "submit": submit, "compare": compare, "freeze": freeze}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
