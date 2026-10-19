#!/usr/bin/env python3
"""Read one real dataset sample per receipt-bound backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/umm/users/yinbaiqiao/probe_spatial")
SOURCE = ROOT / "runtime/main_v2_probe_v2_r5/source"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_summary(value: Any) -> Any:
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    return type(value).__name__


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--models", nargs="*", default=[])
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    selection = json.loads((bundle / "selection.json").read_text(encoding="utf-8"))
    overlay_path = Path(selection["overlay_manifest"]["path"])
    if sha256_file(overlay_path) != selection["overlay_manifest"]["sha256"]:
        raise RuntimeError("overlay manifest changed after selection")

    sys.path.insert(0, str(SOURCE))
    os.environ.update({
        "PROJECT_ROOT": str(SOURCE),
        "PYTHONPATH": str(SOURCE),
        "PROBE_OWNER_OVERLAY_MANIFEST": str(overlay_path),
        "INSCENE_DATA_ROOT": "/mnt/umm/users/yinbaiqiao/InsScene-15K/data",
        "INSCENE_LOCAL_SCENE_ROOT": "/mnt/umm/users/yinbaiqiao/InsScene-15K/data/processed_scannetpp_v2/processed_scannetpp_v2",
        "INSCENE_STREAMING_FEAT_ROOT": "aoss://baiqiao/PROBE816_RECEIPT_ONLY",
        "INSCENE_TRUST_EXPANDED_MANIFEST": "1",
        "INSCENE_PROBE_TASK": "streaming_depth",
        "AOSS_CACHE_DIR": str(bundle / "aoss_cache"),
        "AOSS_FS_METRICS": str(bundle / "aoss_metrics.jsonl"),
        "AOSS_FS_METRICS_MODE": "aggregate",
    })
    from vidfm3d.data.components.inscene15k_dataset import InsScene15KDataset

    wanted = set(args.models)
    results = []
    for row in selection["rows"]:
        model = row["model"]
        if wanted and model not in wanted:
            continue
        os.environ["INSCENE_MANIFEST"] = row["manifest_path"]
        os.environ["INSCENE_MANIFEST_SHA256"] = row["manifest_sha256"]
        started = time.monotonic()
        dataset = InsScene15KDataset(
            root=os.environ["INSCENE_DATA_ROOT"],
            root_vfm=os.environ["INSCENE_STREAMING_FEAT_ROOT"],
            streaming_feat_root=os.environ["INSCENE_STREAMING_FEAT_ROOT"],
            sources=["processed_scannetpp_v2"],
            split="all",
            vfm_name=model,
            feature_layer=int(row["layer"]),
            feature_timestep=int(row["timestep"]),
            feat_pixalign=True,
            num_views=int(row["prefix"]),
            query_idx_divisor=None,
            target_h=288,
            target_w=512,
            window_size=0,
            include_pmaps=False,
            streaming_prefix=True,
            prefix_min_len=int(row["prefix"]),
            prefix_max_len=int(row["prefix"]),
            prefix_stride=1,
        )
        sample = dataset[0]
        # The optimized pixel-probe return intentionally omits prefix_indices;
        # the exact IDs remain available on the normalized dataset record that
        # drove the successful feature lookup.
        observed_ids = [int(v) for v in dataset.scenes[0]["streaming_indices"]]
        if observed_ids != row["raw_ids"]:
            raise RuntimeError(f"raw ID mismatch for {model}: {observed_ids} != {row['raw_ids']}")
        results.append({
            "model": model,
            "key": {
                "scene_id": row["scene_id"], "window_id": row["window_id"],
                "prefix": row["prefix"], "layer": row["layer"], "timestep": row["timestep"],
            },
            "source_uri": row["source_uri"],
            "receipt_sha256": row["receipt_sha256"],
            "raw_ids": observed_ids,
            "manifest_sample_id": sample["manifest_sample_id"],
            "outputs": {key: tensor_summary(value) for key, value in sample.items()},
            "elapsed_seconds": time.monotonic() - started,
            "status": "PASS",
        })
    report_path = bundle / "loader_canary_report.json"
    if report_path.is_file():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        by_model = {row["model"]: row for row in previous.get("results", [])}
        by_model.update({row["model"]: row for row in results})
        results = [by_model[key] for key in sorted(by_model)]
    report = {
        "schema": "PROBE816_RECEIPT_REAL_LOADER_CANARY_V1",
        "status": "PASS" if results else "FAIL",
        "count": len(results),
        "results": results,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if results else 2


if __name__ == "__main__":
    raise SystemExit(main())
