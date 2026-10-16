#!/usr/bin/env python3
"""Verify bf16 seeded-v1 MiniMax caches published by the frozen extractor.

Runs inside an SCO job (AOSS is reachable there, not from the login node) after
scripts/run_tstep_minimax_bf16_v1_extract.sh has finished for every
(timestep, mode) step of a run tag.  For every entry the new done-indices hold:

  structure   the set of (scene, window, prefix, layer) units equals the fp16
              seeded-v1 done-index entries for the same scenes and layers
  payload     safetensors dtype is bf16, every value is finite, shape matches
              the fp16 original
  values      where the fp16 original is finite, |new - old| / max(|old|, eps)
              is reported (median / p99 / max); the fp16 non-finite fraction is
              reported per file (this is the overflow being repaired)
  sidecar     storage contract, dtype, finite_check, namespace, protocol SHA,
              payload SHA-256, and the generation provenance (model, prompt,
              resize, frames, timestep, noise seed) equal to the fp16 original
  indices     structural files (prefix_index.npy for streaming; target_indices.npy
              and target_indices.meta.json for targets) are byte-identical to the
              fp16 originals (local keep_local copies), their sidecars match up to
              feature_namespace, and the AOSS copy equals the local copy

Every check failure, including an exception, is recorded as a problem; the
report is always written.  Exit 0 and status PASS only if nothing failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT = Path("/mnt/umm/users/yinbaiqiao/probe_spatial")
sys.path.insert(0, str(ROOT))
from vidfm3d.utils import aoss_fs  # noqa: E402

DONE_DIR = ROOT / "logs/tstep_canonical_v2_seeded_v1/aoss_done"
LOCAL_BASE = Path("/mnt/umm/users/yinbaiqiao/InsScene-15K")
BUCKET = "aoss://baiqiao/InsScene-15K"
NAMESPACES = {
    "stream": ("FEAT_STREAMING_CANONICAL_V2_SEEDED_V1_MINIMAX_BF16_V1", "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1"),
    "target": ("FEAT_TARGET_CANONICAL_V2_SEEDED_V1_MINIMAX_BF16_V1", "FEAT_TARGET_CANONICAL_V2_SEEDED_V1"),
}
STRUCTURAL = {
    "stream": ["prefix_index.npy", "prefix_index.npy.manifest.json"],
    "target": ["target_indices.npy", "target_indices.npy.manifest.json", "target_indices.meta.json"],
}
TIMESTEPS = (500, 1000)
LAYERS = (20, 25)
CONTRACT = "MINIMAX_H3_FINITE_BF16_V1"
PROTOCOL_SHA256 = "215c720c8537f4dfae6243bedcc7339fbb7e3ba8a6ca2dcaf04f49cd4b738e4f"
MODEL_ID = "/mnt/umm/users/yinbaiqiao/Consistency/minimax-h3/models/MiniMax-H3"
PROVENANCE_KEYS = (
    "vfm", "model_id", "mode", "timestep", "prompt", "resize", "num_frames",
    "canonical_protocol", "canonical_protocol_sha256", "canonical_obs_len",
    "noise_seed", "noise_seed_scheme", "feature_generation_contract",
)
LAYER_RE = re.compile(r"feature_t(\d+)_layer(\d+)\.sft$")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_index(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def layer_of(rel: str) -> int | None:
    match = LAYER_RE.search(rel)
    return int(match.group(2)) if match else None


def strip_namespace(value):
    """Drop feature_namespace keys at any depth (the one field that must differ)."""
    if isinstance(value, dict):
        return {k: strip_namespace(v) for k, v in value.items() if k != "feature_namespace"}
    if isinstance(value, list):
        return [strip_namespace(v) for v in value]
    return value


def compare_values(new: torch.Tensor, old: torch.Tensor, eps: float) -> dict:
    old32 = old.float()
    finite = torch.isfinite(old32)
    out = {"old_nonfinite_fraction": float((~finite).float().mean()), "old_n_nonfinite": int((~finite).sum())}
    if finite.any():
        a = new.float()[finite]
        b = old32[finite]
        rel = (a - b).abs() / torch.clamp(b.abs(), min=eps)
        sample = rel[:: max(1, rel.numel() // 2_000_000)]
        q = torch.quantile(sample, torch.tensor([0.5, 0.99]))
        out.update({
            "rel_err_median": float(q[0]), "rel_err_p99": float(q[1]), "rel_err_max": float(rel.max()),
            "frac_rel_err_gt_5pct": float((rel > 0.05).float().mean()),
            "new_absmax": float(new.float().abs().max()), "old_absmax_finite": float(b.abs().max()),
        })
    return out


def check_file(rel: str, ns_new: str, ns_old: str, timestep: int, mode_name: str, eps: float) -> dict:
    rec = {"rel": rel, "problems": []}
    p_new = f"{BUCKET}/{ns_new}/minimax_h3/scannetpp/{rel}"
    p_old = f"{BUCKET}/{ns_old}/minimax_h3/scannetpp/{rel}"
    if not aoss_fs.exists(p_new):
        rec["problems"].append("new payload missing on AOSS")
        return rec
    raw = aoss_fs.read_bytes(p_new)
    rec["size_bytes"] = len(raw)
    payload_sha = sha256_bytes(raw)
    new = aoss_fs.load_safetensors(p_new)["feat"]
    rec["new_dtype"] = str(new.dtype)
    rec["new_shape"] = list(new.shape)
    if new.dtype != torch.bfloat16:
        rec["problems"].append(f"new dtype {new.dtype} != bfloat16")
    if not bool(torch.isfinite(new).all()):
        rec["problems"].append(f"new payload has {int((~torch.isfinite(new)).sum())} non-finite values")
    side = aoss_fs.load_json(p_new + ".manifest.json")
    expectations = {
        "feature_storage_contract": CONTRACT,
        "feature_storage_dtype": "torch.bfloat16",
        "feature_namespace": ns_new,
        "canonical_protocol_sha256": PROTOCOL_SHA256,
        "model_id": MODEL_ID,
        "vfm": "minimax_h3",
        "mode": mode_name,
        "sha256": payload_sha,
        "size_bytes": len(raw),
    }
    for key, want in expectations.items():
        if side.get(key) != want:
            rec["problems"].append(f"sidecar {key}={side.get(key)!r} != {want!r}")
    if int(side.get("timestep", -1)) != timestep:
        rec["problems"].append(f"sidecar timestep {side.get('timestep')} != {timestep}")
    if not side.get("finite_check"):
        rec["problems"].append("sidecar lacks finite_check")
    feat_meta = side.get("tensors", {}).get("feat", {})
    if feat_meta.get("dtype") != "torch.bfloat16" or feat_meta.get("shape") != list(new.shape):
        rec["problems"].append(f"sidecar tensors.feat {feat_meta} != bf16 {list(new.shape)}")
    if aoss_fs.exists(p_old):
        old = aoss_fs.load_safetensors(p_old)["feat"]
        rec["old_dtype"] = str(old.dtype)
        rec["old_shape"] = list(old.shape)
        if list(old.shape) != list(new.shape):
            rec["problems"].append(f"shape {list(new.shape)} != fp16 original {list(old.shape)}")
        else:
            rec.update(compare_values(new, old, eps))
        old_side = aoss_fs.load_json(p_old + ".manifest.json")
        diffs = {k: [side.get(k), old_side.get(k)] for k in PROVENANCE_KEYS if side.get(k) != old_side.get(k)}
        if diffs:
            rec["problems"].append(f"provenance differs from fp16 original: {diffs}")
        rec["old_sidecar_keys_not_in_new"] = sorted(set(old_side) - set(side))
    else:
        rec["problems"].append("fp16 original missing on AOSS (cannot cross-check)")
    return rec


def check_structural(scene: str, ns_new: str, ns_old: str, mode: str) -> dict:
    """Compare keep_local structural files with the fp16 originals and their AOSS copies."""
    rec = {"scene": scene, "problems": [], "compared": []}
    new_dir = LOCAL_BASE / ns_new / "minimax_h3/scannetpp" / scene
    old_dir = LOCAL_BASE / ns_old / "minimax_h3/scannetpp" / scene
    for name in STRUCTURAL[mode]:
        entry = {"file": name}
        p_new, p_old = new_dir / name, old_dir / name
        if not p_new.is_file():
            rec["problems"].append(f"{name}: missing locally in new namespace ({p_new})")
            rec["compared"].append(entry)
            continue
        if not p_old.is_file():
            rec["problems"].append(f"{name}: fp16 original missing locally ({p_old})")
            rec["compared"].append(entry)
            continue
        b_new, b_old = p_new.read_bytes(), p_old.read_bytes()
        entry["new_sha256"] = sha256_bytes(b_new)
        entry["old_sha256"] = sha256_bytes(b_old)
        if name.endswith(".json"):
            j_new, j_old = json.loads(b_new), json.loads(b_old)
            if j_new.get("feature_namespace") != ns_new:
                rec["problems"].append(f"{name}: feature_namespace {j_new.get('feature_namespace')!r} != {ns_new!r}")
            a, b = strip_namespace(j_new), strip_namespace(j_old)
            diff_keys = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            entry["diff_keys_excluding_namespace"] = diff_keys
            if diff_keys:
                rec["problems"].append(f"{name}: differs from fp16 original in {diff_keys}")
        else:
            entry["byte_identical"] = b_new == b_old
            if b_new != b_old:
                rec["problems"].append(f"{name}: not byte-identical to fp16 original")
        remote = f"{BUCKET}/{ns_new}/minimax_h3/scannetpp/{scene}/{name}"
        try:
            if aoss_fs.exists(remote):
                entry["aoss_equals_local"] = aoss_fs.read_bytes(remote) == b_new
                if not entry["aoss_equals_local"]:
                    rec["problems"].append(f"{name}: AOSS copy differs from local copy")
            else:
                entry["aoss_equals_local"] = None
                rec["problems"].append(f"{name}: missing on AOSS")
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            rec["problems"].append(f"{name}: AOSS check raised {exc!r}")
        rec["compared"].append(entry)
    return rec


def guarded(fn, label: str, *args) -> dict:
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001 - every failure must land in the report
        return {"rel": label, "scene": label, "problems": [f"exception: {exc!r}"],
                "traceback": traceback.format_exc(limit=6)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--scene-count", type=int, required=True, help="distinct scenes expected per step")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--eps", type=float, default=1e-2, help="denominator floor for relative error")
    parser.add_argument("--median-tol", type=float, default=0.01)
    parser.add_argument("--p99-tol", type=float, default=0.05)
    parser.add_argument("--max-value-compares", type=int, default=0, help="0 = compare every file")
    args = parser.parse_args()
    os.environ.setdefault("AOSS_CACHE_DIR", f"/tmp/aoss_verify_{args.run_tag}")
    report = {
        "schema": "TSTEP_MINIMAX_BF16_V1_CANARY_VERIFY_V2", "run_tag": args.run_tag,
        "verifier": str(Path(__file__).resolve()),
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tolerances": {"eps": args.eps, "median": args.median_tol, "p99": args.p99_tol},
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": os.uname().nodename,
        "steps": {}, "problems": [],
    }
    compared = 0
    for mode, (ns_new, ns_old) in NAMESPACES.items():
        mode_name = "streaming_prefix" if mode == "stream" else "target_isolated"
        for t in TIMESTEPS:
            key = f"t{t}_{mode}"
            new_idx = read_index(DONE_DIR / f"{ns_new}_minimax_h3_t{t}.txt")
            old_idx = read_index(DONE_DIR / f"{ns_old}_minimax_h3_t{t}.txt")
            step = {"new_index_entries": len(new_idx), "old_index_entries": len(old_idx),
                    "files": [], "indices": [], "problems": []}
            outside = {rel for rel in new_idx if layer_of(rel) not in LAYERS}
            if outside:
                step["problems"].append(f"{len(outside)} entries outside layers {LAYERS}: {sorted(outside)[:3]}")
            scenes = sorted({rel.split("/")[0] for rel in new_idx})
            step["scenes"] = scenes
            if len(scenes) != args.scene_count:
                step["problems"].append(f"expected {args.scene_count} scenes, found {len(scenes)}")
            for scene in scenes:
                expected = {rel for rel in old_idx if rel.startswith(scene + "/") and layer_of(rel) in LAYERS}
                got = {rel for rel in new_idx if rel.startswith(scene + "/")}
                if expected != got:
                    step["problems"].append(
                        f"{scene}: unit set differs from fp16 original "
                        f"(missing={sorted(expected - got)[:5]}, extra={sorted(got - expected)[:5]})"
                    )
                step["indices"].append(guarded(check_structural, scene, scene, ns_new, ns_old, mode))
            for rel in sorted(new_idx):
                if args.max_value_compares and compared >= args.max_value_compares:
                    break
                rec = guarded(check_file, rel, rel, ns_new, ns_old, t, mode_name, args.eps)
                compared += 1
                if "rel_err_median" in rec and (rec["rel_err_median"] > args.median_tol or rec["rel_err_p99"] > args.p99_tol):
                    rec["problems"].append(
                        f"value drift vs fp16 original: median={rec['rel_err_median']:.4g} p99={rec['rel_err_p99']:.4g}"
                    )
                step["files"].append(rec)
                print(json.dumps({k: rec.get(k) for k in ("rel", "new_dtype", "new_shape", "old_nonfinite_fraction",
                                                         "rel_err_median", "rel_err_p99", "problems")}), flush=True)
            for rec in step["files"] + step["indices"]:
                for problem in rec["problems"]:
                    step["problems"].append(f"{rec.get('rel') or rec.get('scene')}: {problem}")
            report["steps"][key] = step
            report["problems"].extend(f"{key}: {p}" for p in step["problems"])
    report["files_checked"] = compared
    report["status"] = "PASS" if not report["problems"] and compared > 0 else "FAIL"
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "files_checked": compared, "problems": report["problems"][:20]}, indent=1))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
