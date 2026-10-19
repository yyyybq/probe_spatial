#!/usr/bin/env python3
"""Create the start snapshot and final canonical-v2 canary receipt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from scripts.verify_tstep_canonical_v2_canary_receipt import (
        SCENE_ID,
        ReceiptError,
        build_receipt,
        create_start_snapshot,
        sha256_file,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from verify_tstep_canonical_v2_canary_receipt import (  # type: ignore[no-redef]
        SCENE_ID,
        ReceiptError,
        build_receipt,
        create_start_snapshot,
        sha256_file,
    )


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--protocol-spec", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--done-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--expected-scene", default=SCENE_ID)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser("snapshot", help="freeze code/protocol/manifest hashes")
    add_common(snapshot)
    snapshot.add_argument("--output", required=True)
    finalize = commands.add_parser("finalize", help="validate evidence and write receipt")
    add_common(finalize)
    finalize.add_argument("--start-snapshot", required=True)
    finalize.add_argument("--receipt", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "snapshot":
            output = Path(args.output)
            if output.exists() or output.is_symlink():
                raise ReceiptError(f"refusing to overwrite start snapshot: {output}")
            value = create_start_snapshot(
                output=output,
                project_root=Path(args.project_root),
                protocol_spec=Path(args.protocol_spec),
                manifest_dir=Path(args.manifest_dir),
                done_dir=Path(args.done_dir),
                dataset_root=Path(args.dataset_root),
                scene_id=args.expected_scene,
            )
            result = {
                "status": "SNAPSHOT_WRITTEN",
                "path": str(output.resolve()),
                "sha256": sha256_file(output),
                "source_count": len(value["source_sha256"]),
                "manifest_file_count": sum(len(item) for item in value["manifest_sha256"].values()),
            }
        else:
            receipt = Path(args.receipt)
            if receipt.exists() or receipt.is_symlink():
                raise ReceiptError(f"refusing to overwrite canary receipt: {receipt}")
            value = build_receipt(
                receipt=receipt,
                start_snapshot=Path(args.start_snapshot),
                project_root=Path(args.project_root),
                protocol_spec=Path(args.protocol_spec),
                manifest_dir=Path(args.manifest_dir),
                done_dir=Path(args.done_dir),
                dataset_root=Path(args.dataset_root),
                scene_id=args.expected_scene,
            )
            result = {
                "status": value["status"],
                "path": str(receipt.resolve()),
                "sha256": sha256_file(receipt),
                "stream_keys": value["completion"]["stream_keys"],
                "target_keys": value["completion"]["target_keys"],
                "local_sft": value["completion"]["local_sft"],
            }
    except ReceiptError as exc:
        print(f"CANARY_RECEIPT_BUILD_FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
