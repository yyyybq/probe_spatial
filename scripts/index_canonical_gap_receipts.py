#!/usr/bin/env python3
"""Build an audit-only index of verified canonical-v2 gap receipts."""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

CONTRACT = "CANONICAL_V2_GAP_AOSS_COMPLETION_RECEIPT_V1"
STATUS = "PASS_REMOTE_VERIFIED"
PROTOCOL_SHA = "215c720c8537f4dfae6243bedcc7339fbb7e3ba8a6ca2dcaf04f49cd4b738e4f"
SOURCE_SHA = "ebc496eb0587cda372ccf29a7c5244ef454096151f73c8eed9511967ded1902d"


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--models", default="", help="comma-separated VFM allowlist")
    args = p.parse_args()
    selected_models = {value.strip() for value in args.models.split(",") if value.strip()}
    owners = {}
    index_versions = {}
    receipts = []
    unreadable = []
    for root in args.root:
        for name in sorted(glob.glob(str(root / "*" / "remote_completion.json"))):
            path = Path(name)
            if path.is_symlink() or not path.is_file():
                raise SystemExit(f"unsafe receipt: {path}")
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except PermissionError:
                unreadable.append(str(path.resolve()))
                continue
            unhashed = dict(value); recorded = unhashed.pop("content_sha256", None)
            lineage = value.get("lineage", {})
            if value.get("contract") != CONTRACT or value.get("status") != STATUS:
                raise SystemExit(f"invalid receipt contract/status: {path}")
            if recorded != canonical_sha(unhashed):
                raise SystemExit(f"receipt content SHA mismatch: {path}")
            if lineage.get("protocol_sha256") != PROTOCOL_SHA or lineage.get("canonical_source_sha256") != SOURCE_SHA:
                raise SystemExit(f"receipt canonical lineage mismatch: {path}")
            vfm, mode = str(lineage.get("vfm")), str(lineage.get("mode"))
            if selected_models and vfm not in selected_models:
                continue
            remote_base = str(value.get("remote", {}).get("base_uri", "")).rstrip("/")
            if not remote_base.startswith(("aoss://", "s3://", "fj:s3://")):
                raise SystemExit(f"invalid receipt remote base: {path}")
            receipt_binding = {"path": str(path.resolve()), "sha256": file_sha(path), "content_sha256": recorded}
            receipts.append(receipt_binding)
            for relative, item in value.get("objects", {}).items():
                relative_path = Path(relative)
                if (not relative or relative_path.is_absolute()
                        or ".." in relative_path.parts):
                    raise SystemExit(f"unsafe object path: {path}:{relative}")
                identity = item.get("canonical_unit_identity")
                if not isinstance(identity, dict) or item.get("canonical_unit_identity_sha256") != canonical_sha(identity):
                    raise SystemExit(f"invalid object identity: {path}:{relative}")
                key = (remote_base, relative)
                candidate = {
                    "root": remote_base, "relative_path": relative, "vfm": vfm, "mode": mode,
                    "identity": identity, "identity_sha256": item["canonical_unit_identity_sha256"],
                    "payload_sha256": item.get("payload_sha256"), "payload_size_bytes": item.get("payload_size_bytes"),
                }
                if relative.endswith(("prefix_index.npy", "target_indices.npy")):
                    index_versions.setdefault(key, {})[candidate["payload_sha256"]] = candidate
                    continue
                if key in owners and owners[key] != candidate:
                    raise SystemExit(f"owner conflict: {key}")
                owners[key] = candidate
    counts = Counter((owner["mode"], owner["vfm"]) for owner in owners.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8") as f:
        for key in sorted(owners):
            f.write(json.dumps(owners[key], sort_keys=True, separators=(",", ":")) + "\n")
    summary = {
        "schema": "canonical_v2_gap_receipt_index_v1",
        "audit_only": True,
        "status": "PASS" if not unreadable else "INCOMPLETE_UNREADABLE_RECEIPTS",
        "protocol_sha256": PROTOCOL_SHA, "canonical_source_sha256": SOURCE_SHA,
        "selected_models": sorted(selected_models),
        "receipts": len(receipts), "objects": len(owners),
        "mutable_index_paths": len(index_versions),
        "mutable_index_versions": sum(len(value) for value in index_versions.values()),
        "counts": {f"{m}:{v}": n for (m, v), n in sorted(counts.items())},
        "output": {"path": str(args.output.resolve()), "sha256": file_sha(args.output)},
        "receipt_bindings_sha256": canonical_sha(receipts),
        "unreadable_receipts": unreadable,
    }
    summary_path = args.output.with_suffix("").with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
