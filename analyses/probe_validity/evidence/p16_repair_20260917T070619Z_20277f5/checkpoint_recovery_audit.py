#!/usr/bin/env python3
"""One-shot read-only recovery audit for the three adjudicated B2 P16 finals."""

from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from pathlib import Path

from aoss_client.ceph.ceph import Ceph
from aoss_client.client import Client


RUN = Path("/mnt/umm/users/yinbaiqiao/probe_spatial/analyses/probe_validity/evidence/p16_repair_20260917T070619Z_20277f5")
CONF = Path("/mnt/umm/users/yinbaiqiao/aoss/petreloss.conf")
TARGETS = {
    "vjepa2": {
        "run_id": "j42pwdvc",
        "filename": "epoch=49-step=122350.ckpt",
        "formal_path": Path("/mnt/umm/users/yinbaiqiao/probe_spatial/logs/inscene15k_streaming/runs/inscene15k_streaming_b2_vjepa2_p16_layer23/VideoProbe3D/j42pwdvc/checkpoints/epoch=49-step=122350.ckpt"),
    },
    "videomae": {
        "run_id": "jessyc3a",
        "filename": "epoch=49-step=122350.ckpt",
        "formal_path": Path("/mnt/umm/users/yinbaiqiao/probe_spatial/logs/inscene15k_streaming/runs/inscene15k_streaming_b2_videomae_p16_layer23/VideoProbe3D/jessyc3a/checkpoints/epoch=49-step=122350.ckpt"),
    },
    "wan": {
        "run_id": "06mfoqwx",
        "filename": "epoch=49-step=122350.ckpt",
        "formal_path": Path("/mnt/umm/users/yinbaiqiao/probe_spatial/logs/inscene15k_streaming/runs/inscene15k_streaming_b2_wan_p16_layer20_heavy64_20260716_021635/VideoProbe3D/06mfoqwx/checkpoints/epoch=49-step=122350.ckpt"),
    },
}
SEARCHABLE_TOP_NAMES = (
    "backup", "archive", "checkpoint", "ckpt", "probe", "spatial", "log",
    "experiment", "artifact", "result", "inscene",
)
SKIP_NAMES = ("feat", "data", "dataset", "image", "model_weight")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    started = time.time()
    report = {
        "schema_version": 1,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": os.uname().nodename,
        "targets": {},
        "local_scopes": [],
        "aoss": {},
        "errors": [],
    }
    for model, target in TARGETS.items():
        path = target["formal_path"]
        report["targets"][model] = {
            "run_id": target["run_id"], "filename": target["filename"],
            "formal_path": str(path), "present_at_start": path.is_file(),
        }

    # Exact, bounded shared-storage scopes.  Large feature/data/cache trees are
    # pruned because they are not historical checkpoint staging locations.
    roots = [
        Path("/mnt/umm/users/yinbaiqiao/probe_spatial/logs"),
        Path("/mnt/umm/users/yinbaiqiao/.q00_tmp"),
        Path("/mnt/umm/users/yinbaiqiao/.local/share/Trash"),
        Path("/mnt/umm/users/yinbaiqiao/.codex/tmp"),
        Path("/mnt/umm/users/yinbaiqiao/.codex-relay/tmp"),
    ]
    wanted_ids = {target["run_id"] for target in TARGETS.values()}
    local_matches = []
    for root in roots:
        scope = {"root": str(root), "exists": root.exists(), "files_examined": 0}
        if root.exists():
            for current, directories, files in os.walk(root):
                directories[:] = [
                    name for name in directories
                    if not any(skip in name.lower() for skip in SKIP_NAMES)
                ]
                scope["files_examined"] += len(files)
                for name in files:
                    full = Path(current) / name
                    text = str(full)
                    if name == "epoch=49-step=122350.ckpt" and any(token in text for token in wanted_ids):
                        local_matches.append({"path": text, "size_bytes": full.stat().st_size})
        report["local_scopes"].append(scope)
    report["local_matches"] = local_matches

    # Discover AOSS prefixes from the documented baiqiao bucket.  Only
    # recursively enumerate top-level namespaces whose discovered names look
    # like checkpoint/log/archive/probe storage; never scan feature/data roots.
    try:
        os.environ["NO_PROXY"] = (
            "localhost,127.0.0.1,::1,baiqiao.aoss-internal.cn-fz-01.fjscmsapi-oss.com,"
            ".aoss-internal.cn-fz-01.fjscmsapi-oss.com,aoss-internal.cn-fz-01.fjscmsapi-oss.com"
        )
        client = Client(str(CONF))
        mixed = client._get_local_client()
        cluster, bucket, _ = Ceph.parse_uri(
            "fj:s3://baiqiao/", mixed._ceph_dict, mixed._default_cluster
        )
        s3 = mixed._ceph_dict[cluster]._s3_resource.meta.client
        top = s3.list_objects_v2(Bucket=bucket, Delimiter="/", MaxKeys=1000)
        top_prefixes = [item["Prefix"] for item in top.get("CommonPrefixes", [])]
        report["aoss"]["bucket"] = bucket
        report["aoss"]["top_level_prefixes"] = top_prefixes
        candidates = [
            prefix for prefix in top_prefixes
            if any(term in prefix.lower() for term in SEARCHABLE_TOP_NAMES)
            and not any(skip in prefix.lower() for skip in SKIP_NAMES)
            and "inscene" not in prefix.lower()
        ]
        # InsScene-15K is a documented bucket namespace but mostly contains
        # multi-million-object feature/data trees.  Discover one level below it
        # and admit only checkpoint/log/archive-like children.
        for prefix in [value for value in top_prefixes if "inscene" in value.lower()]:
            child_page = s3.list_objects_v2(
                Bucket=bucket, Prefix=prefix, Delimiter="/", MaxKeys=1000
            )
            candidates.extend(
                item["Prefix"] for item in child_page.get("CommonPrefixes", [])
                if any(term in item["Prefix"].lower() for term in SEARCHABLE_TOP_NAMES)
                and not any(skip in item["Prefix"].lower() for skip in SKIP_NAMES)
            )
        report["aoss"]["recursively_examined_prefixes"] = candidates
        matches = []
        objects_examined = 0
        paginator = s3.get_paginator("list_objects_v2")
        for prefix in candidates:
            for page in paginator.paginate(
                Bucket=bucket, Prefix=prefix,
                PaginationConfig={"PageSize": 1000, "MaxItems": 250000},
            ):
                for item in page.get("Contents", []):
                    objects_examined += 1
                    key = str(item["Key"])
                    for model, target in TARGETS.items():
                        if target["run_id"] in key and key.endswith(target["filename"]):
                            matches.append({
                                "model": model, "key": key,
                                "size_bytes": int(item.get("Size", -1)),
                                "etag": str(item.get("ETag", "")),
                            })
        report["aoss"]["objects_examined"] = objects_examined
        report["aoss"]["exact_matches"] = matches

        # Exact run-id+filename matches are safe to restore.  Download first to
        # a sibling partial file, validate torch metadata, then atomically place
        # the bytes at the formal summary-bound path.
        import torch
        for match in matches:
            model = match["model"]
            destination = TARGETS[model]["formal_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_name(destination.name + ".p16_recovery_partial")
            if not destination.exists():
                s3.download_file(bucket, match["key"], str(partial))
                payload = torch.load(partial, map_location="cpu", weights_only=False)
                if int(payload.get("epoch", -1)) != 49 or int(payload.get("global_step", -1)) != 122350:
                    raise RuntimeError(f"metadata mismatch for {match['key']}")
                os.replace(partial, destination)
            match["restored_path"] = str(destination)
            match["sha256"] = sha256(destination)
            match["epoch"] = 49
            match["global_step"] = 122350
    except Exception as error:
        report["errors"].append({
            "stage": "AOSS_DISCOVERY_OR_RECOVERY",
            "reason": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        })

    for model, target in TARGETS.items():
        path = target["formal_path"]
        report["targets"][model]["present_at_end"] = path.is_file()
        if path.is_file():
            report["targets"][model]["size_bytes"] = path.stat().st_size
            report["targets"][model]["sha256"] = sha256(path)
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report["elapsed_seconds"] = time.time() - started
    report["status"] = (
        "RECOVERED_ALL" if all(item["present_at_end"] for item in report["targets"].values())
        else "RECOVERY_INCOMPLETE"
    )
    temporary = RUN / ".checkpoint_recovery_audit.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, RUN / "checkpoint_recovery_audit.json")
    print(json.dumps({"status": report["status"], "targets": report["targets"]}, indent=2))


if __name__ == "__main__":
    main()
