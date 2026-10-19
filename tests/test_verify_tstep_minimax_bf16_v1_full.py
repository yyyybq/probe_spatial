"""scripts/verify_tstep_minimax_bf16_v1_full.py against a local stand-in for AOSS."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save

pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/verify_tstep_minimax_bf16_v1_full.py"
NS = {
    "stream": ("FEAT_STREAMING_CANONICAL_V2_SEEDED_V1_MINIMAX_BF16_V1", "FEAT_STREAMING_CANONICAL_V2_SEEDED_V1"),
    "target": ("FEAT_TARGET_CANONICAL_V2_SEEDED_V1_MINIMAX_BF16_V1", "FEAT_TARGET_CANONICAL_V2_SEEDED_V1"),
}
STRUCT = {
    "stream": ("prefix_index.npy", "prefix_index.npy.manifest.json"),
    "target": ("target_indices.npy", "target_indices.npy.manifest.json", "target_indices.meta.json"),
}
SCENE = "s00"
PROVENANCE = {
    "vfm": "minimax_h3", "model_id": "/mnt/umm/users/yinbaiqiao/Consistency/minimax-h3/models/MiniMax-H3",
    "prompt": "", "resize": [208, 120], "num_frames": 85,
    "canonical_protocol": "inscene_streaming_canonical_v2_obs28_h124",
    "canonical_protocol_sha256": "215c720c8537f4dfae6243bedcc7339fbb7e3ba8a6ca2dcaf04f49cd4b738e4f",
    "canonical_obs_len": 28, "noise_seed": 0, "noise_seed_scheme": "minimax_h3_fixed_seed_v1",
    "feature_generation_contract": "inscene_deterministic_noise_v1",
}


def rels(mode: str, t: int) -> list[str]:
    if mode == "target":
        return [f"{SCENE}/feature_t{t}_layer{layer}.sft" for layer in (20, 25)]
    return [f"{SCENE}/window_0000/prefix_000007/feature_t{t}_layer{layer}.sft" for layer in (20, 25)]


def put(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def put_feature(bucket: Path, ns: str, rel: str, tensor: torch.Tensor, mode: str, t: int, bf16: bool) -> None:
    raw = save({"feat": tensor})
    side = {**PROVENANCE, "mode": "streaming_prefix" if mode == "stream" else "target_isolated", "timestep": t,
            "feature_namespace": ns, "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw),
            "tensors": {"feat": {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}}}
    if bf16:
        side.update({"feature_storage_contract": "MINIMAX_H3_FINITE_BF16_V1",
                     "feature_storage_dtype": "torch.bfloat16", "finite_check": {"ok": True}})
    target = bucket / ns / "minimax_h3/scannetpp" / rel
    put(target, raw)
    put(Path(f"{target}.manifest.json"), json.dumps(side).encode())


def build(tmp_path: Path) -> dict:
    bucket, local, done = tmp_path / "bucket", tmp_path / "local", tmp_path / "done"
    done.mkdir()
    base = torch.linspace(-3, 3, 2 * 3 * 4).reshape(2, 3, 4)
    old = base.half()
    old[0, 0, 0] = float("inf")  # the fp16 overflow being repaired
    new = base.bfloat16()
    for mode, (ns_new, ns_old) in NS.items():
        for t in (500,):
            (done / f"{ns_new}_minimax_h3_t{t}.txt").write_text("\n".join(rels(mode, t)) + "\n")
            (done / f"{ns_old}_minimax_h3_t{t}.txt").write_text(
                "\n".join(rels(mode, t) + [rels(mode, t)[0].replace("layer20", "layer15")]) + "\n")
            for rel in rels(mode, t):
                put_feature(bucket, ns_new, rel, new, mode, t, True)
                put_feature(bucket, ns_old, rel, old, mode, t, False)
        for name in STRUCT[mode]:
            for ns in (ns_new, ns_old):
                body = (json.dumps({"feature_namespace": ns, "file": name}) if name.endswith(".json")
                        else f"npy:{name}").encode()
                put(local / ns / "minimax_h3/scannetpp" / SCENE / name, body)
                if ns == ns_new:
                    put(bucket / ns / "minimax_h3/scannetpp" / SCENE / name, body)
    return {"bucket": bucket, "local": local, "done": done}


def run(tmp_path: Path, tree: dict):
    report = tmp_path / "report.json"
    cmd = [sys.executable, str(SCRIPT), "--report", str(report), "--scene-count", "1",
           "--steps", "t500_stream,t500_target", "--done-dir", str(tree["done"]),
           "--local-base", str(tree["local"]), "--bucket", str(tree["bucket"]), "--workers", "2"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    return result, json.loads(report.read_text())


def test_full_verify_passes_on_finite_bf16_copy(tmp_path: Path) -> None:
    result, report = run(tmp_path, build(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr
    assert report["status"] == "PASS"
    assert report["sidecars_checked"] == 4 and report["payloads_checked"] == 4
    assert report["steps"]["t500_stream"]["old_nonfinite_values_in_sample"] == 2


def test_full_verify_fails_on_nonfinite_payload_and_wrong_sidecar(tmp_path: Path) -> None:
    tree = build(tmp_path)
    ns_new = NS["stream"][0]
    bad = torch.linspace(-3, 3, 24).reshape(2, 3, 4).bfloat16()
    bad[1, 1, 1] = float("nan")
    put_feature(tree["bucket"], ns_new, rels("stream", 500)[0], bad, "stream", 500, True)
    side_path = Path(f"{tree['bucket'] / NS['target'][0] / 'minimax_h3/scannetpp' / rels('target', 500)[1]}.manifest.json")
    side = json.loads(side_path.read_text())
    side["feature_storage_dtype"] = "torch.float16"
    side_path.write_text(json.dumps(side))
    result, report = run(tmp_path, tree)
    assert result.returncode == 1
    text = "\n".join(report["problems"])
    assert "non-finite values" in text
    assert "feature_storage_dtype" in text
