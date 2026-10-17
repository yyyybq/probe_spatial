#!/usr/bin/env python3
"""Sample-level parity of the main-tree data loader against the frozen training packs.

For each canonical-V2 evaluation item (eval_v2_canonical_20261017_r2/items) the
validation and the training dataset are built twice, in separate processes:
once from the item's frozen training pack (runtime/<pack>/source, verified
against its source.sha256) and once from a main-tree checkout.  Both use the
item's training environment (owner overlay, manifest, fp16 setting) and their
own fresh label cache.  The comparison covers

  * the composed Hydra data and model configuration (paths excluded),
  * the full ordered list of kept samples (manifest identity of every sample),
  * every field both loaders emit for a fixed set of sample indices, bit for bit
    (floating tensors are compared as raw bits, so NaN payloads and -0.0 count).

Fields only one loader emits are listed, not compared: the main tree adds
evaluation and audit fields (poses for C2, the anchor-isolated feature, ids).

  run    --items KEY... | --shard i/n   [--out DIR]   orchestrates and writes
         DIR/<key>/PARITY.json and DIR/SUMMARY_<shard>.json
  dump   (internal) one code root, one item -> one .pt file
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/mnt/umm/users/yinbaiqiao/probe_spatial")
CAMPAIGN = ROOT / "artifacts/main_v2_probe_v2_816_20260917/eval_v2_canonical_20261017_r2"
PYTHON = "/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python"
AOSS_CONF = "/mnt/umm/users/yinbaiqiao/aoss/petreloss.conf"
IDENT_KEYS = ("sample_id", "scene_dir", "source", "scene_id", "streaming_window_id",
              "streaming_tail", "streaming_tail_raw", "c1_target_horizon", "window_start",
              "window_end", "num_frames")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------- dump
def _plain(value):
    import numpy as np
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _indices(n: int, k: int) -> list[int]:
    import numpy as np

    if n <= 0 or k <= 0:
        return []
    picked = set(range(min(4, n))) | set(np.linspace(0, n - 1, k).round().astype(int).tolist())
    return sorted(int(i) for i in picked)


def dump(args: argparse.Namespace) -> int:
    import faulthandler

    if os.environ.get("PARITY_STACK_AFTER"):
        faulthandler.dump_traceback_later(int(os.environ["PARITY_STACK_AFTER"]), repeat=True)
    import hydra
    import lightning as L
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    code_root = Path(args.code_root).resolve()
    import vidfm3d
    import vidfm3d.data.video_probe_datamodule as vpdm

    module_file = Path(next(iter(vidfm3d.__path__))).resolve()
    if code_root not in module_file.parents:
        raise RuntimeError(f"vidfm3d imported from {module_file}, not {code_root}")
    item = json.loads(Path(args.item).read_text().splitlines()[0])
    overrides = [f"experiment={item['experiment_config']}"]
    overrides += [o for o in item["scientific_overrides"] if not o.startswith("autoresume=")]
    overrides += ["++eval_split=val", "data.data_module.num_workers=0",
                  "data.data_module.num_workers_val=0"]
    with initialize_config_dir(config_dir=str(code_root / "configs"), version_base="1.3"):
        cfg = compose(config_name="train.yaml", overrides=overrides)
    config_text = {
        name: OmegaConf.to_yaml(cfg[name], resolve=False)
        for name in ("data", "model") if name in cfg
    }
    out = {"code_root": str(code_root), "vidfm3d_file": str(module_file), "overrides": overrides,
           "config_text": config_text, "splits": {}}
    for split in ("val", "train"):
        started = time.time()
        if cfg.get("seed") is not None:
            L.seed_everything(int(cfg.seed), workers=True)
        dm = hydra.utils.instantiate(cfg.data.data_module)
        exprs = dm.hparams.validation_datasets if split == "val" else dm.hparams.train_datasets
        record = {"n_exprs": len(exprs)}
        try:
            expr = dm._dataset_expr(exprs[0]) if hasattr(dm, "_dataset_expr") else exprs[0]
            dataset = eval(expr, vars(vpdm))
        except Exception as exc:  # recorded and compared, never hidden
            record["construct_error"] = f"{type(exc).__name__}: {exc}"
            out["splits"][split] = record
            continue
        if split == "train":
            dataset.seed = int(args.train_sample_seed)  # deterministic frame sampling
        scenes = getattr(dataset, "scenes", [])
        idents = [{k: s.get(k) for k in IDENT_KEYS if k in s} for s in scenes]
        ident_blob = json.dumps(idents, sort_keys=True, default=str).encode()
        record.update({"len": len(dataset), "ident_sha256": sha256_bytes(ident_blob),
                       "idents": idents, "samples": {}, "errors": {}})
        for idx in _indices(len(dataset), args.n_val if split == "val" else args.n_train):
            try:
                record["samples"][idx] = _plain(dataset[idx])
            except Exception as exc:
                record["errors"][idx] = f"{type(exc).__name__}: {exc}"
        record["seconds"] = round(time.time() - started, 1)
        out["splits"][split] = record
    torch.save(out, args.out)
    return 0


# ------------------------------------------------------------------ compare
def _bits(t):
    import torch

    t = t.contiguous()
    view = {torch.float16: torch.int16, torch.bfloat16: torch.int16,
            torch.float32: torch.int32, torch.float64: torch.int64}.get(t.dtype)
    return t.view(view) if view is not None else t


def same(a, b) -> bool:
    import torch

    if torch.is_tensor(a) or torch.is_tensor(b):
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            return False
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(_bits(a), _bits(b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def compare(pack: dict, main: dict) -> dict:
    result = {"config_equal": {k: pack["config_text"].get(k) == main["config_text"].get(k)
                               for k in ("data", "model")}, "splits": {}}
    ok = all(result["config_equal"].values())
    for split in ("val", "train"):
        p, m = pack["splits"].get(split, {}), main["splits"].get(split, {})
        r = {}
        if "construct_error" in p or "construct_error" in m:
            r["construct_error"] = {"pack": p.get("construct_error"), "main": m.get("construct_error")}
            r["ok"] = p.get("construct_error") == m.get("construct_error")
            ok &= r["ok"]
            result["splits"][split] = r
            continue
        r["len"] = {"pack": p["len"], "main": m["len"]}
        r["sample_list_equal"] = p["ident_sha256"] == m["ident_sha256"]
        if not r["sample_list_equal"]:
            first = next((i for i, (x, y) in enumerate(zip(p["idents"], m["idents"])) if x != y), None)
            r["first_sample_list_difference"] = first
        r["errors"] = {"pack": p["errors"], "main": m["errors"]}
        compared, mismatched, only_pack, only_main, n_fields = 0, {}, set(), set(), 0
        for idx in sorted(set(p["samples"]) & set(m["samples"])):
            a, b = p["samples"][idx], m["samples"][idx]
            only_pack |= set(a) - set(b)
            only_main |= set(b) - set(a)
            for key in sorted(set(a) & set(b)):
                n_fields += 1
                if not same(a[key], b[key]):
                    mismatched.setdefault(key, []).append(idx)
            compared += 1
        r.update({"samples_compared": compared, "fields_compared": n_fields,
                  "mismatched_fields": mismatched, "fields_only_pack": sorted(only_pack),
                  "fields_only_main": sorted(only_main),
                  "sample_indices_pack": sorted(p["samples"]), "sample_indices_main": sorted(m["samples"])})
        r["ok"] = (r["sample_list_equal"] and p["len"] == m["len"] and not mismatched
                   and not only_pack and p["errors"] == m["errors"]
                   and sorted(p["samples"]) == sorted(m["samples"]) and compared > 0)
        ok &= r["ok"]
        result["splits"][split] = r
    result["status"] = "PASS" if ok else "FAIL"
    return result


# ---------------------------------------------------------------------- run
def _verify_pack(pack: Path) -> None:
    done = subprocess.run(["sha256sum", "--status", "-c", str(pack / "source.sha256")],
                          cwd=pack / "source")
    if done.returncode:
        raise RuntimeError(f"pack source verification failed: {pack}")


def run(args: argparse.Namespace) -> int:
    import torch

    plan = json.loads((CAMPAIGN / "plan.json").read_text())["items"]
    keys = args.items or sorted(plan)
    if args.shard:
        i, n = (int(v) for v in args.shard.split("/"))
        keys = keys[i::n]
    main_root = Path(args.main_root).resolve()
    if args.campaign_json:
        # On a job node: verify against the SHA-256 list frozen at the commit (no git needed).
        camp = json.loads(Path(args.campaign_json).read_text())
        manifest = Path(camp["code_manifest"])
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != camp["code_manifest_sha256"]:
            raise RuntimeError("code manifest changed")
        if Path(camp["code_root"]).resolve() != main_root:
            raise RuntimeError("campaign code root differs from --main-root")
        if subprocess.run(["sha256sum", "--status", "-c", str(manifest)], cwd=main_root).returncode:
            raise RuntimeError("main tree differs from the frozen code manifest")
        listed = {line.split("  ", 1)[1] for line in manifest.read_text().splitlines() if line}
        present = {str(p.relative_to(main_root)) for top in ("vidfm3d", "configs")
                   for p in (main_root / top).rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
        if present - listed:
            raise RuntimeError(f"unlisted files in the main tree: {sorted(present - listed)[:5]}")
        head, dirty = camp["commit"], ""
    else:
        head = subprocess.run(["git", "-C", str(main_root), "rev-parse", "HEAD"], text=True,
                              capture_output=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(main_root), "status", "--porcelain", "--", "vidfm3d", "configs"],
                               text=True, capture_output=True, check=True).stdout.strip()
        if dirty and not args.allow_dirty:
            raise RuntimeError(f"main tree has uncommitted code/config changes:\n{dirty}")
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {"schema": "MAIN_VS_PACK_LOADER_PARITY_V1", "started": now(), "main_root": str(main_root),
               "main_commit": head, "main_dirty": bool(dirty), "items": {}}
    for key in keys:
        entry = plan[key]
        item = json.loads(Path(entry["path"]).read_text().splitlines()[0])
        pack = ROOT / "runtime" / item["training_pack"]
        _verify_pack(pack)
        work = out_root / key
        work.mkdir(parents=True, exist_ok=True)
        files = {}
        for side, code_root in (("pack", pack / "source"), ("main", main_root)):
            env = os.environ.copy()
            env.update(item["env"])
            env.update({
                "PYTHONPATH": str(code_root), "PROJECT_ROOT": str(code_root),
                "PYTHONDONTWRITEBYTECODE": "1", "AOSS_CONF": AOSS_CONF,
                "INSCENE_LABEL_CACHE_ROOT": str(Path(args.scratch) / key / f"labels_{side}"),
                "AOSS_CACHE_DIR": str(Path(args.scratch) / key / "aoss_cache"),
                "OMP_NUM_THREADS": "4", "WANDB_MODE": "offline", "HYDRA_FULL_ERROR": "1",
            })
            target = Path(args.scratch) / key / f"{side}.pt"  # node-local; can be GBs
            target.parent.mkdir(parents=True, exist_ok=True)
            started = time.time()
            with (work / f"{side}.log").open("w", encoding="utf-8") as log:
                rc = subprocess.run([PYTHON, "-P", str(Path(__file__).resolve()), "dump",
                                     "--code-root", str(code_root), "--item", entry["path"],
                                     "--out", str(target), "--n-val", str(args.n_val),
                                     "--n-train", str(args.n_train),
                                     "--train-sample-seed", str(args.train_sample_seed)],
                                    cwd=code_root, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            files[side] = {"rc": rc, "seconds": round(time.time() - started, 1)}
        if any(v["rc"] for v in files.values()):
            result = {"status": "ERROR", "dumps": files}
        else:
            result = compare(torch.load(Path(args.scratch) / key / "pack.pt", weights_only=False),
                             torch.load(Path(args.scratch) / key / "main.pt", weights_only=False))
            for side in ("pack", "main"):
                (Path(args.scratch) / key / f"{side}.pt").unlink(missing_ok=True)
            result["dumps"] = files
        result.update({"unique_key": key, "training_pack": item["training_pack"],
                       "model": item["model"], "probe": item["probe"], "main_commit": head})
        (work / "PARITY.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n")
        summary["items"][key] = result["status"]
        print(f"{result['status']}\t{key}", flush=True)
    summary["finished"] = now()
    name = f"SUMMARY_{args.tag or (args.shard.replace('/', 'of') if args.shard else 'items')}.json"
    (out_root / name).write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    return 0 if all(v == "PASS" for v in summary["items"].values()) else 1


def submit(args: argparse.Namespace) -> int:
    """One one-GPU Zoetrope job per shard (the loaders need AOSS, not the GPU)."""
    import re
    import shlex

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    ledger_path = out_root / "submissions.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
        "schema": "MAIN_VS_PACK_LOADER_PARITY_SUBMISSIONS_V1", "jobs": {}}
    keys = args.items
    if not keys:
        raise SystemExit("--items is required")
    script = Path(args.main_root).resolve() / "scripts" / Path(__file__).name
    for i in range(args.shards):
        shard_keys = keys[i::args.shards]
        command = (f"cd {shlex.quote(str(ROOT))} && "
                   + shlex.join([PYTHON, str(script), "run", "--items", *shard_keys,
                                 "--main-root", str(Path(args.main_root).resolve()),
                                 "--out", str(out_root), "--scratch", args.scratch,
                                 "--n-val", str(args.n_val), "--n-train", str(args.n_train),
                                 "--campaign-json", args.campaign_json]))
        command += f" --tag shard{i}of{args.shards}"
        name = f"bq-parity-{len(ledger['jobs']):04d}"
        if not args.submit:
            print(f"  {name}: {len(shard_keys)} items")
            continue
        argv = [str(Path.home() / ".sco/bin/sco"), "acp", "jobs", "create", "--workspace-name=aigc",
                "--aec2-name=zoetrope", f"--job-name={name}", "--priority=HIGHEST",
                "--quota-type=reserved",
                "--container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617",
                "--storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm",
                "--training-framework=pytorch", "--worker-nodes=1", "--worker-spec=N4lS.Iq.I80.1",
                f"--command={command}"]
        done = subprocess.run(argv, text=True, capture_output=True, timeout=180)
        match = re.search(r"pt-[a-z0-9]+", done.stdout + done.stderr)
        ledger["jobs"][name] = {"keys": shard_keys, "job_id": match.group(0) if match else None,
                                "returncode": done.returncode, "command": command, "submitted_utc": now()}
        ledger_path.write_text(json.dumps(ledger, indent=1, sort_keys=True) + "\n")
        print(f"  {name}\t{ledger['jobs'][name]['job_id'] or 'SUBMIT_ERROR'}", flush=True)
        if done.returncode or not match:
            print(done.stdout, done.stderr)
            raise SystemExit("submission failed; stopping")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sb = sub.add_parser("submit")
    sb.add_argument("--items", nargs="*", default=[])
    sb.add_argument("--shards", type=int, default=8)
    sb.add_argument("--main-root", default=str(ROOT / ".worktrees/release-main-overlay"))
    sb.add_argument("--out", required=True)
    sb.add_argument("--scratch", default="/tmp/loader_parity")
    sb.add_argument("--n-val", type=int, default=16)
    sb.add_argument("--n-train", type=int, default=8)
    sb.add_argument("--submit", action="store_true")
    sb.add_argument("--campaign-json", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--code-root", required=True)
    d.add_argument("--item", required=True)
    d.add_argument("--out", required=True)
    for p in (d, r := sub.add_parser("run")):
        p.add_argument("--n-val", type=int, default=16)
        p.add_argument("--n-train", type=int, default=8)
        p.add_argument("--train-sample-seed", type=int, default=20261017)
    r.add_argument("--items", nargs="*", default=[])
    r.add_argument("--shard", default="")
    r.add_argument("--main-root", default=str(ROOT / ".worktrees/release-main-overlay"))
    r.add_argument("--out", required=True)
    r.add_argument("--scratch", default="/tmp/loader_parity")
    r.add_argument("--allow-dirty", action="store_true")
    r.add_argument("--tag", default="")
    r.add_argument("--campaign-json", default="")
    args = parser.parse_args()
    return {"dump": dump, "run": run, "submit": submit}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
