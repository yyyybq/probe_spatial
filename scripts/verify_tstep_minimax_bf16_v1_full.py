#!/usr/bin/env python3
"""Full verification of the consolidated MiniMax-H3 bf16 seeded-v1 caches.

Adapted from scripts/verify_tstep_minimax_bf16_v1_canary.py (same per-file
checks, same tolerances) to the whole namespace after
scripts/consolidate_tstep_minimax_bf16_v1_shards.py --apply.  Runs inside an
SCO job (AOSS is not reachable from the login node).  For every
(timestep, mode) step:

  structure  per scene, the canonical bf16 done-index holds exactly the fp16
             seeded-v1 layer-20/25 units; the number of scenes is --scene-count
  sidecars   EVERY indexed file: the payload object exists and its
             .manifest.json carries the storage contract, bf16 dtype,
             finite_check, namespace, protocol SHA, model, vfm, mode, timestep
             and tensors.feat (bf16, shape equal to the fp16 original's
             sidecar); generation provenance equals the fp16 original sidecar
  indices    EVERY scene: the local keep_local structural files are
             byte-identical to the fp16 originals (JSON up to
             feature_namespace) and equal to their AOSS copies
  payloads   a deterministic sample per (step, layer): download, SHA-256 and
             size equal the sidecar, dtype bf16, every value finite, shape
             equal to the fp16 original, value drift vs the fp16 original
             where that is finite (median / p99 tolerances as the canary)

Scope (also written into the report): finiteness of payloads outside the
sample rests on the producer chain -- the extractor checks finiteness before
and after the bf16 conversion and records finite_check in the sidecar, and the
validating sync loop (SYNC_TRUST_EXTRACT=0) re-hashed every file against its
sidecar before upload.  The report is always written; exit 0 only on PASS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from safetensors.torch import load as load_safetensors_bytes

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from verify_tstep_minimax_bf16_v1_canary import (  # noqa: E402
    BUCKET, CONTRACT, DONE_DIR, LAYERS, LOCAL_BASE, MODEL_ID, NAMESPACES, PROTOCOL_SHA256,
    PROVENANCE_KEYS, STRUCTURAL, TIMESTEPS, aoss_fs, compare_values, layer_of, read_index, strip_namespace,
)

SCOPE_NOTE = (
    "structure, sidecars and structural indices: every indexed file/scene; payload SHA-256, dtype, "
    "finiteness and value drift: the deterministic sample listed per step. Finiteness outside the "
    "sample rests on the extractor's finite_check and the validating sync loop's SHA-256 checks."
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def remote(bucket: str, namespace: str, rel: str) -> str:
    return f"{bucket}/{namespace}/minimax_h3/scannetpp/{rel}"


def guarded(fn, label: str, *args) -> dict:
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001 - every failure must land in the report
        return {"rel": label, "problems": [f"exception: {exc!r}"], "traceback": traceback.format_exc(limit=4)}


def check_sidecar(rel: str, ns_new: str, ns_old: str, timestep: int, mode_name: str, bucket: str) -> dict:
    rec = {"rel": rel, "problems": []}
    p_new, p_old = remote(bucket, ns_new, rel), remote(bucket, ns_old, rel)
    if not aoss_fs.exists(p_new):
        rec["problems"].append("new payload missing on AOSS")
    side = aoss_fs.load_json(p_new + ".manifest.json")
    old_side = aoss_fs.load_json(p_old + ".manifest.json")
    expectations = {
        "feature_storage_contract": CONTRACT, "feature_storage_dtype": "torch.bfloat16",
        "feature_namespace": ns_new, "canonical_protocol_sha256": PROTOCOL_SHA256,
        "model_id": MODEL_ID, "vfm": "minimax_h3", "mode": mode_name,
    }
    for key, want in expectations.items():
        if side.get(key) != want:
            rec["problems"].append(f"sidecar {key}={side.get(key)!r} != {want!r}")
    if int(side.get("timestep", -1)) != timestep:
        rec["problems"].append(f"sidecar timestep {side.get('timestep')} != {timestep}")
    if not side.get("finite_check"):
        rec["problems"].append("sidecar lacks finite_check")
    if not isinstance(side.get("size_bytes"), int) or side["size_bytes"] <= 0:
        rec["problems"].append(f"sidecar size_bytes {side.get('size_bytes')!r}")
    feat = side.get("tensors", {}).get("feat", {})
    old_shape = old_side.get("tensors", {}).get("feat", {}).get("shape")
    if feat.get("dtype") != "torch.bfloat16":
        rec["problems"].append(f"sidecar tensors.feat dtype {feat.get('dtype')!r}")
    if old_shape is None:
        rec["old_shape_unavailable"] = True
    elif feat.get("shape") != old_shape:
        rec["problems"].append(f"sidecar shape {feat.get('shape')} != fp16 original {old_shape}")
    diffs = {k: [side.get(k), old_side.get(k)] for k in PROVENANCE_KEYS if side.get(k) != old_side.get(k)}
    if diffs:
        rec["problems"].append(f"provenance differs from fp16 original: {diffs}")
    return rec


def check_payload(rel: str, ns_new: str, ns_old: str, eps: float, bucket: str) -> dict:
    rec = {"rel": rel, "problems": []}
    p_new, p_old = remote(bucket, ns_new, rel), remote(bucket, ns_old, rel)
    raw = aoss_fs.read_bytes(p_new)
    side = aoss_fs.load_json(p_new + ".manifest.json")
    rec["size_bytes"] = len(raw)
    if side.get("sha256") != sha256_bytes(raw) or side.get("size_bytes") != len(raw):
        rec["problems"].append("payload SHA-256/size differ from sidecar")
    new = load_safetensors_bytes(raw)["feat"]
    del raw
    rec.update({"new_dtype": str(new.dtype), "new_shape": list(new.shape)})
    if new.dtype != torch.bfloat16:
        rec["problems"].append(f"new dtype {new.dtype} != bfloat16")
    n_bad = int((~torch.isfinite(new)).sum())
    if n_bad:
        rec["problems"].append(f"new payload has {n_bad} non-finite values")
    old = load_safetensors_bytes(aoss_fs.read_bytes(p_old))["feat"]
    rec["old_shape"] = list(old.shape)
    if list(old.shape) != list(new.shape):
        rec["problems"].append(f"shape {list(new.shape)} != fp16 original {list(old.shape)}")
    else:
        rec.update(compare_values(new, old, eps))
    return rec


def check_structural(scene: str, ns_new: str, ns_old: str, mode: str, local_base: Path, bucket: str) -> dict:
    rec = {"rel": scene, "problems": []}
    for name in STRUCTURAL[mode]:
        p_new = local_base / ns_new / "minimax_h3/scannetpp" / scene / name
        p_old = local_base / ns_old / "minimax_h3/scannetpp" / scene / name
        if not p_new.is_file() or not p_old.is_file():
            rec["problems"].append(f"{name}: missing locally (new={p_new.is_file()}, fp16={p_old.is_file()})")
            continue
        b_new, b_old = p_new.read_bytes(), p_old.read_bytes()
        if name.endswith(".json"):
            j_new, j_old = json.loads(b_new), json.loads(b_old)
            if j_new.get("feature_namespace") != ns_new:
                rec["problems"].append(f"{name}: feature_namespace {j_new.get('feature_namespace')!r}")
            a, b = strip_namespace(j_new), strip_namespace(j_old)
            diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            if diff:
                rec["problems"].append(f"{name}: differs from fp16 original in {diff}")
        elif b_new != b_old:
            rec["problems"].append(f"{name}: not byte-identical to fp16 original")
        target = remote(bucket, ns_new, f"{scene}/{name}")
        if not aoss_fs.exists(target):
            rec["problems"].append(f"{name}: missing on AOSS")
        elif aoss_fs.read_bytes(target) != b_new:
            rec["problems"].append(f"{name}: AOSS copy differs from local copy")
    return rec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--scene-count", type=int, default=681, help="scenes per step (fp16 index has 681)")
    parser.add_argument("--sample-stream", type=int, default=16, help="payloads per (stream step, layer)")
    parser.add_argument("--sample-target", type=int, default=8, help="payloads per (target step, layer)")
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--workers", type=int, default=32, help="threads for sidecar/index checks")
    parser.add_argument("--payload-workers", type=int, default=4)
    parser.add_argument("--eps", type=float, default=1e-2)
    parser.add_argument("--median-tol", type=float, default=0.01)
    parser.add_argument("--p99-tol", type=float, default=0.05)
    parser.add_argument("--steps", default="all", help="comma list like t500_stream,t1000_target")
    parser.add_argument("--done-dir", type=Path, default=DONE_DIR)
    parser.add_argument("--local-base", type=Path, default=LOCAL_BASE)
    parser.add_argument("--bucket", default=BUCKET)
    args = parser.parse_args()
    os.environ.setdefault("AOSS_CACHE_DIR", "/tmp/aoss_verify_full")
    report = {
        "schema": "TSTEP_MINIMAX_BF16_V1_FULL_VERIFY_V1",
        "verifier": str(Path(__file__).resolve()),
        "verifier_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "tolerances": {"eps": args.eps, "median": args.median_tol, "p99": args.p99_tol},
        "sampling": {"stream_per_layer": args.sample_stream, "target_per_layer": args.sample_target,
                     "seed": args.seed},
        "scope": SCOPE_NOTE, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename, "steps": {}, "problems": [],
    }
    wanted = None if args.steps == "all" else set(args.steps.split(","))
    total_sidecars = total_payloads = 0
    for mode, (ns_new, ns_old) in NAMESPACES.items():
        mode_name = "streaming_prefix" if mode == "stream" else "target_isolated"
        for t in TIMESTEPS:
            key = f"t{t}_{mode}"
            if wanted is not None and key not in wanted:
                continue
            t0 = time.time()
            new_idx = read_index(args.done_dir / f"{ns_new}_minimax_h3_t{t}.txt")
            old_idx = {rel for rel in read_index(args.done_dir / f"{ns_old}_minimax_h3_t{t}.txt")
                       if layer_of(rel) in LAYERS}
            step = {"new_index_entries": len(new_idx), "old_index_entries_l20_l25": len(old_idx), "problems": []}
            outside = sorted(rel for rel in new_idx if layer_of(rel) not in LAYERS)
            if outside:
                step["problems"].append(f"{len(outside)} entries outside layers {LAYERS}: {outside[:3]}")
            scenes = sorted({rel.split("/")[0] for rel in new_idx})
            step["scenes"] = len(scenes)
            if len(scenes) != args.scene_count:
                step["problems"].append(f"expected {args.scene_count} scenes, found {len(scenes)}")
            if new_idx != old_idx:
                step["problems"].append(f"unit set differs from fp16 (missing={len(old_idx - new_idx)} "
                                        f"e.g. {sorted(old_idx - new_idx)[:3]}, extra={len(new_idx - old_idx)} "
                                        f"e.g. {sorted(new_idx - old_idx)[:3]})")
            rels = sorted(new_idx)
            with ThreadPoolExecutor(args.workers) as pool:
                sidecars = list(pool.map(
                    lambda rel: guarded(check_sidecar, rel, rel, ns_new, ns_old, t, mode_name, args.bucket), rels))
                indices = list(pool.map(
                    lambda s: guarded(check_structural, s, s, ns_new, ns_old, mode, args.local_base, args.bucket),
                    scenes))
            rng = random.Random(f"{args.seed}:{key}")
            per_layer = args.sample_stream if mode == "stream" else args.sample_target
            sample = []
            for layer in LAYERS:
                pool_rels = [rel for rel in rels if layer_of(rel) == layer]
                sample += sorted(rng.sample(pool_rels, min(per_layer, len(pool_rels))))
            with ThreadPoolExecutor(args.payload_workers) as pool:
                payloads = list(pool.map(
                    lambda rel: guarded(check_payload, rel, rel, ns_new, ns_old, args.eps, args.bucket), sample))
            for rec in payloads:
                if "rel_err_median" in rec and (rec["rel_err_median"] > args.median_tol
                                                or rec["rel_err_p99"] > args.p99_tol):
                    rec["problems"].append(f"value drift vs fp16: median={rec['rel_err_median']:.4g} "
                                           f"p99={rec['rel_err_p99']:.4g}")
            for kind, recs in (("sidecar", sidecars), ("index", indices), ("payload", payloads)):
                for rec in recs:
                    step["problems"].extend(f"{kind} {rec['rel']}: {p}" for p in rec["problems"])
            step.update({
                "sidecars_checked": len(sidecars), "indices_checked": len(indices), "payloads_checked": len(payloads),
                "old_shape_unavailable": sum(bool(r.get("old_shape_unavailable")) for r in sidecars),
                "payload_sample": payloads, "seconds": round(time.time() - t0, 1),
                "max_rel_err_p99": max((r.get("rel_err_p99", 0.0) for r in payloads), default=None),
                "old_nonfinite_values_in_sample": sum(r.get("old_n_nonfinite", 0) for r in payloads),
                "new_absmax_in_sample": max((r.get("new_absmax", 0.0) for r in payloads), default=None),
            })
            total_sidecars += len(sidecars)
            total_payloads += len(payloads)
            n_problems = len(step["problems"])
            step["problem_count"] = n_problems
            step["problems"] = step["problems"][:200]
            report["steps"][key] = step
            report["problems"].extend(f"{key}: {p}" for p in step["problems"][:50])
            report.setdefault("problem_count", 0)
            report["problem_count"] += n_problems
            print(json.dumps({"step": key, "entries": len(new_idx), "scenes": len(scenes),
                              "payloads": len(payloads), "problems": n_problems,
                              "seconds": step["seconds"]}), flush=True)
    report.setdefault("problem_count", 0)
    report["sidecars_checked"] = total_sidecars
    report["payloads_checked"] = total_payloads
    report["status"] = "PASS" if report["problem_count"] == 0 and total_sidecars > 0 else "FAIL"
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.report.with_name(f".{args.report.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(tmp, 0o644)
    os.replace(tmp, args.report)
    print(json.dumps({"status": report["status"], "sidecars": total_sidecars, "payloads": total_payloads,
                      "problem_count": report["problem_count"], "first_problems": report["problems"][:10]},
                     indent=1))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
