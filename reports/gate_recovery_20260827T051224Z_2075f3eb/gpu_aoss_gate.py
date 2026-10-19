#!/usr/bin/env python3
"""Recovery-only GPU, AOSS, and one-sample DataLoader gate."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path("/mnt/umm/users/yinbaiqiao/probe_spatial")
REPORT_DIR = PROJECT_ROOT / "reports/gate_recovery_20260827T051224Z_2075f3eb"
CANONICAL_MANIFEST = PROJECT_ROOT / "manifests/c_retrain_20260722/inscene_c_h124_prefix8.jsonl"
MINI_MANIFEST = REPORT_DIR / "manifests/dataloader_one_sample.jsonl"
DATA_ROOT = "/mnt/umm/users/yinbaiqiao/InsScene-15K/data"
LOCAL_SCENE = (
    "/mnt/umm/users/yinbaiqiao/InsScene-15K/data/processed_scannetpp_v2/"
    "processed_scannetpp_v2/00777c41d4"
)
STREAM_ROOT = (
    "fj:s3://baiqiao/InsScene-15K/FEAT_STREAMING_BY_PREFIX/"
    "dino_prefix_8_m0p35_r0p5"
)
TARGET_ROOT = (
    "fj:s3://baiqiao/InsScene-15K/FEAT_TARGET_BY_PREFIX/"
    "dino_prefix_8_m0p35_r0p5"
)
TARGET_INDICES = f"{TARGET_ROOT}/dino/scannetpp/00777c41d4/target_indices.npy"
STREAM_FEATURE = (
    f"{STREAM_ROOT}/dino/scannetpp/00777c41d4/"
    "window_0000/prefix_000007/feature.sft"
)
TARGET_FEATURE = f"{TARGET_ROOT}/dino/scannetpp/00777c41d4/feature.sft"
AUDIT_CSV = REPORT_DIR / "AOSS_READ_AUDIT.csv"
ENV_JSON = REPORT_DIR / "ENV_GATE.json"


rows: list[dict[str, object]] = []
gate: dict[str, object] = {
    "schema_version": 1,
    "recovery_tag": "gate_recovery_20260827T051224Z_2075f3eb",
    "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "project_root": str(PROJECT_ROOT),
    "conda_python": sys.executable,
    "paths": {
        "data_root": DATA_ROOT,
        "streaming_feature_root": STREAM_ROOT,
        "target_feature_root": TARGET_ROOT,
        "split_manifest": str(PROJECT_ROOT / "configs/splits/inscene15k_v1.json"),
        "sample_manifest": str(CANONICAL_MANIFEST),
    },
    "gpu": {"status": "NOT_RUN"},
    "aoss": {"status": "NOT_RUN"},
    "dataloader": {"status": "NOT_RUN"},
    "strict_b2_smoke": {"status": "NOT_RUN"},
}


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def flush() -> None:
    fields = ["phase", "step", "path", "status", "elapsed_seconds", "bytes", "detail"]
    tmp = AUDIT_CSV.with_name(f".{AUDIT_CSV.name}.tmp-{os.getpid()}")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, AUDIT_CSV)
    atomic_text(ENV_JSON, json.dumps(gate, indent=2, sort_keys=True) + "\n")


def timed(phase: str, step: str, path: str, fn):
    row: dict[str, object] = {
        "phase": phase,
        "step": step,
        "path": path,
        "status": "STARTED",
        "elapsed_seconds": "",
        "bytes": "",
        "detail": "",
    }
    rows.append(row)
    flush()
    started = time.perf_counter()
    try:
        value, nbytes, detail = fn()
        row.update(
            status="PASS",
            elapsed_seconds=f"{time.perf_counter() - started:.6f}",
            bytes=nbytes,
            detail=detail,
        )
        flush()
        return value
    except Exception as exc:
        row.update(
            status="FAIL",
            elapsed_seconds=f"{time.perf_counter() - started:.6f}",
            detail=f"{type(exc).__name__}: {exc}",
        )
        flush()
        raise


def local_manifest_stat():
    st = CANONICAL_MANIFEST.stat()
    return st, st.st_size, f"mode={oct(st.st_mode & 0o777)} mtime_ns={st.st_mtime_ns}"


def local_manifest_head():
    with CANONICAL_MANIFEST.open("rb") as handle:
        payload = handle.read(4096)
    return payload, len(payload), f"sha256={hashlib.sha256(payload).hexdigest()}"


def load_target_indices(aoss_fs):
    value = aoss_fs.load_npy(TARGET_INDICES)
    return value, int(value.nbytes), "loaded through vidfm3d.utils.aoss_fs"


def main() -> int:
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))
    flush()

    smi = timed(
        "gpu",
        "nvidia_smi",
        "local GPU allocation",
        lambda: (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used",
                    "--format=csv,noheader",
                ],
                text=True,
                timeout=20,
            ).strip(),
            "",
            "nvidia-smi query succeeded",
        ),
    )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False on allocated GPU node")
    started = time.perf_counter()
    x = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
    y = torch.relu(x @ x.T)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if tuple(y.shape) != (4, 4) or not bool(torch.isfinite(y).all().item()):
        raise RuntimeError("CUDA tensor forward returned an invalid tensor")
    rows.append(
        {
            "phase": "gpu",
            "step": "torch_cuda_forward",
            "path": sys.executable,
            "status": "PASS",
            "elapsed_seconds": f"{elapsed:.6f}",
            "bytes": y.numel() * y.element_size(),
            "detail": f"torch={torch.__version__} device={torch.cuda.get_device_name(0)} shape={tuple(y.shape)}",
        }
    )
    gate["gpu"] = {
        "status": "PASS",
        "nvidia_smi": smi,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": True,
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_forward_shape": list(y.shape),
    }
    flush()

    timed("manifest", "exists_stat_known_manifest", str(CANONICAL_MANIFEST), local_manifest_stat)
    timed("manifest", "read_first_4096_bytes", str(CANONICAL_MANIFEST), local_manifest_head)

    from vidfm3d.utils import aoss_fs

    exists = timed(
        "aoss",
        "project_aoss_fs_exists_target_indices",
        TARGET_INDICES,
        lambda: (aoss_fs.exists(TARGET_INDICES), "", "aoss_fs.exists"),
    )
    if not exists:
        raise FileNotFoundError(TARGET_INDICES)
    target_indices = timed(
        "aoss",
        "load_small_target_indices_npy",
        TARGET_INDICES,
        lambda: load_target_indices(aoss_fs),
    )

    stream_exists = timed(
        "aoss",
        "exists_real_dino_p8_streaming_feature",
        STREAM_FEATURE,
        lambda: (aoss_fs.exists(STREAM_FEATURE), "", "aoss_fs.exists"),
    )
    if not stream_exists:
        raise FileNotFoundError(STREAM_FEATURE)
    stream_data = timed(
        "aoss",
        "load_real_dino_p8_streaming_feature",
        STREAM_FEATURE,
        lambda: (
            aoss_fs.load_safetensors(STREAM_FEATURE),
            Path(aoss_fs.materialize(STREAM_FEATURE)).stat().st_size,
            "loaded through aoss_fs.load_safetensors",
        ),
    )
    target_exists = timed(
        "aoss",
        "exists_real_dino_p8_target_feature",
        TARGET_FEATURE,
        lambda: (aoss_fs.exists(TARGET_FEATURE), "", "aoss_fs.exists"),
    )
    if not target_exists:
        raise FileNotFoundError(TARGET_FEATURE)
    target_data = timed(
        "aoss",
        "load_real_dino_p8_target_feature",
        TARGET_FEATURE,
        lambda: (
            aoss_fs.load_safetensors(TARGET_FEATURE),
            Path(aoss_fs.materialize(TARGET_FEATURE)).stat().st_size,
            "loaded through aoss_fs.load_safetensors",
        ),
    )
    gate["aoss"] = {
        "status": "PASS",
        "target_indices_count": int(target_indices.size),
        "target_indices_first": [int(v) for v in target_indices.reshape(-1)[:8]],
        "stream_feature_keys": sorted(stream_data),
        "stream_feature_shapes": {k: list(v.shape) for k, v in stream_data.items()},
        "target_feature_keys": sorted(target_data),
        "target_feature_shapes": {k: list(v.shape) for k, v in target_data.items()},
    }
    flush()

    first = json.loads(CANONICAL_MANIFEST.open().readline())
    first["scene_dir"] = LOCAL_SCENE
    atomic_text(MINI_MANIFEST, json.dumps(first, separators=(",", ":")) + "\n")

    from torch.utils.data import DataLoader
    from vidfm3d.data.components.inscene15k_dataset import InsScene15KDataset

    def load_one_sample():
        dataset = InsScene15KDataset(
            root=DATA_ROOT,
            root_vfm=STREAM_ROOT,
            streaming_feat_root=STREAM_ROOT,
            target_feat_root=TARGET_ROOT,
            scene_manifest=str(MINI_MANIFEST),
            sources=["processed_scannetpp_v2"],
            split="train",
            vfm_name="dino",
            feature_layer=0,
            feature_timestep=749,
            feat_pixalign=True,
            num_views=8,
            min_view_interval=5,
            context_len=8,
            query_idx_divisor=None,
            target_h=288,
            target_w=512,
            window_size=0,
            include_pmaps=False,
            streaming_prefix=True,
            prefix_min_len=8,
            prefix_max_len=8,
            diag_action=True,
            action_horizons=[1],
            allow_missing_vfm=False,
        )
        if len(dataset) != 1:
            raise RuntimeError(f"expected one sample, got {len(dataset)}")
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        batch = next(iter(loader))
        detail = {
            key: list(value.shape) if hasattr(value, "shape") else type(value).__name__
            for key, value in batch.items()
        }
        return batch, "", json.dumps(detail, sort_keys=True)

    batch = timed(
        "dataloader",
        "one_real_dino_p8_streaming_target_sample",
        str(MINI_MANIFEST),
        load_one_sample,
    )
    gate["dataloader"] = {
        "status": "PASS",
        "manifest": str(MINI_MANIFEST),
        "batch_keys": sorted(batch),
        "vfm_feat_shape": list(batch["vfm_feat"].shape),
        "target_feat_shape": list(batch["target_feat"].shape),
        "dyn_valid": bool(batch["dyn_valid"].item()),
        "target_frame_idx": int(batch["target_frame_idx"].item()),
    }
    if not gate["dataloader"]["dyn_valid"]:
        raise RuntimeError("DataLoader returned dyn_valid=False for exact DINO p8 sample")
    gate["pre_smoke_gate_pass"] = True
    gate["completed_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    flush()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        gate["failure"] = f"{type(exc).__name__}: {exc}"
        gate["completed_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        flush()
        raise
