#!/usr/bin/env python3
"""Read-only AOSS check for legacy target roots still used by probe528."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from aoss_client.client import Client


ROOTS = (
    "fj:s3://baiqiao/InsScene-15K/FEAT_TARGET/",
    "fj:s3://baiqiao/InsScene-15K/FEAT_TARGET_BY_PREFIX/",
    "fj:s3://baiqiao/InsScene-15K/FEAT_TARGET_CANONICAL_V2_GAP_DELTA_V1/",
)
OBJECTS = (
    "fj:s3://baiqiao/InsScene-15K/FEAT_TARGET/lingbot/scannetpp/00777c41d4/feature_t250_layer10.sft",
    "fj:s3://baiqiao/InsScene-15K/FEAT_TARGET/videomae/scannetpp/00777c41d4/feature_layer5.sft",
    "fj:s3://baiqiao/InsScene-15K/FEAT_TARGET/vjepa2/scannetpp/00777c41d4/feature_layer5.sft",
    "fj:s3://baiqiao/InsScene-15K/FEAT_TARGET_CANONICAL_V2_GAP_DELTA_V1/dino/scannetpp/00a231a370/.canonical_v2_target_delta/d71d3262ae1bdba4/feature_t749_layer0.delta.sft",
)


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    client = Client("/mnt/umm/users/yinbaiqiao/aoss/petreloss.conf")
    roots = []
    for root in ROOTS:
        entries = []
        try:
            for i, value in enumerate(client.list(root)):
                entries.append(str(value))
                if i >= 49: break
            error = None
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        roots.append({"root": root, "isdir": bool(client.isdir(root)), "first_entries": entries, "error": error})
    objects = []
    for uri in OBJECTS:
        objects.append({"uri": uri, "contains": bool(client.contains(uri))})
    receipt = {
        "schema": "PROBE528_TARGET_ROOT_QUERY_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "three explicit roots and four exact representative objects; read-only",
        "roots": roots, "objects": objects,
    }
    receipt["content_sha256"] = canonical_sha(receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{args.output.name}.", dir=args.output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(temp, args.output); os.chmod(args.output, 0o644)
    finally:
        try: os.unlink(temp)
        except FileNotFoundError: pass
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
