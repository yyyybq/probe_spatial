#!/usr/bin/env python3
"""Local fail-closed gate for official Wan/Aether GAP_DELTA requests.

Reads the same request schema as the AOSS reference gate, maps
``aoss://baiqiao/InsScene-15K/...`` onto the local InsScene-15K mount, and
emits ``LEGACY_EXACT_REFERENCE_REMOTE_GATE_RECEIPT_V1`` receipts. Missing
files are FAIL, not PENDING. PASS additionally requires the sidecar's
feature namespace and canonical source SHA.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyses.windowfix_v2.gate_aoss_streaming_references import (  # noqa: E402
    FAIL,
    PASS,
    RemoteReader,
    canonical_json,
    check_request,
    sha256_file,
)

GAP_NAMESPACE = "FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1"
CANONICAL_PARENT_SOURCE_SHA256 = (
    "ebc496eb0587cda372ccf29a7c5244ef454096151f73c8eed9511967ded1902d"
)
AOSS_PREFIX = "aoss://baiqiao/InsScene-15K/"
FJ_PREFIX = "fj:s3://baiqiao/InsScene-15K/"
DEFAULT_LOCAL_ROOT = Path("/mnt/umm/users/yinbaiqiao/InsScene-15K")


class LocalClient:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def _path(self, uri: str) -> Path:
        text = str(uri)
        if text.startswith("fj:s3://"):
            text = "aoss://" + text[len("fj:s3://") :]
        if not text.startswith(AOSS_PREFIX):
            raise FileNotFoundError(uri)
        relative = text[len(AOSS_PREFIX) :]
        path = (self.root / relative).resolve()
        if path != self.root and not str(path).startswith(str(self.root) + "/"):
            raise FileNotFoundError(uri)
        return path

    def contains(self, uri: str) -> bool:
        path = self._path(uri)
        return path.is_file() and not path.is_symlink()

    def size(self, uri: str) -> int:
        path = self._path(uri)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(uri)
        return int(path.stat().st_size)

    def get(self, uri: str, range: str | None = None) -> bytes:
        path = self._path(uri)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(uri)
        if range is None:
            return path.read_bytes()
        start_text, end_text = range.split("-", 1)
        start = int(start_text)
        end = int(end_text)
        if start < 0 or end < start:
            raise ValueError(f"invalid range: {range}")
        with path.open("rb") as handle:
            handle.seek(start)
            data = handle.read(end - start + 1)
        if len(data) != end - start + 1:
            raise ValueError(f"short local range read: {uri} {range}")
        return data


def enforce_gap_identity(receipt: dict[str, Any], request: dict[str, Any], client: LocalClient) -> dict[str, Any]:
    if receipt.get("status") != PASS:
        return receipt
    sidecar_uri = str(request["sidecar_uri"])
    sidecar = json.loads(client.get(sidecar_uri))
    expected = request["expected"]
    failed: list[str] = []
    if sidecar.get("feature_namespace") != GAP_NAMESPACE:
        failed.append("sidecar_feature_namespace")
    if sidecar.get("canonical_source_sha256") != CANONICAL_PARENT_SOURCE_SHA256:
        failed.append("sidecar_canonical_source_sha256")
    gap_key = sidecar.get("gap_record_key") or {}
    if (
        str(gap_key.get("scene")) != str(expected["scene_id"])
        or int(gap_key.get("window_id", -1)) != int(expected["window_id"])
        or int(gap_key.get("prefix", -1)) != int(expected["prefix"])
    ):
        failed.append("sidecar_gap_record_key")
    if int(sidecar.get("layer", -1)) != int(expected["feature_layer"]):
        failed.append("sidecar_layer")
    if failed:
        receipt["status"] = FAIL
        receipt["reason_codes"] = [f"CHECK_FAILED:{name}" for name in failed]
        extra = dict(receipt.get("checks") or {})
        extra.update({name: False for name in failed})
        receipt["checks"] = extra
    return receipt


def gate_shard(
    *,
    requests_path: str,
    output_dir: str,
    local_root: str,
    shard_index: int,
    shard_count: int,
    max_metadata_bytes: int,
) -> dict[str, Any]:
    requests = Path(requests_path)
    destination = Path(output_dir)
    if destination.exists():
        raise ValueError(f"shard output exists: {destination}")
    destination.mkdir(parents=True)
    reader = RemoteReader(LocalClient(Path(local_root)), max_metadata_bytes)
    client = reader.client
    counts: Counter[str] = Counter()
    processed = 0
    receipts_path = destination / "receipts.jsonl.gz"
    with gzip.GzipFile(filename=str(receipts_path), mode="wb", mtime=0) as sink:
        opener = gzip.open if requests.name.endswith(".gz") else open
        with opener(requests, "rt", encoding="utf-8") as handle:
            for offset, line in enumerate(handle):
                if offset % shard_count != shard_index:
                    continue
                request = json.loads(line)
                receipt = check_request(reader, request)
                receipt = enforce_gap_identity(receipt, request, client)
                receipt["expected"] = request["expected"]
                receipt["local_root"] = str(Path(local_root).resolve())
                counts[str(receipt["status"])] += 1
                processed += 1
                sink.write((canonical_json(receipt) + "\n").encode("utf-8"))
    summary = {
        "schema": "LEGACY_EXACT_REFERENCE_REMOTE_GATE_SUMMARY_V1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "payload_full_read_performed": False,
        "processed": processed,
        "receipts_file": receipts_path.name,
        "receipts_sha256": sha256_file(receipts_path),
        "requests": str(requests.resolve()),
        "requests_sha256": sha256_file(requests),
        "shard_count": shard_count,
        "shard_index": shard_index,
        "status_counts": dict(counts),
        "gate_mode": "local_filesystem_mapped_from_aoss_uri",
        "feature_namespace": GAP_NAMESPACE,
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _worker(payload: dict[str, Any]) -> dict[str, Any]:
    return gate_shard(**payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    parser.add_argument("--shard-count", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-metadata-bytes", type=int, default=4 * 1024 * 1024)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"output already exists: {output_dir}")
    if args.shard_count <= 0:
        raise SystemExit("shard count must be positive")
    output_dir.mkdir(parents=True)
    payloads = [
        {
            "requests_path": str(args.requests.resolve()),
            "output_dir": str(output_dir / f"gate-{index:04d}-of-{args.shard_count:04d}"),
            "local_root": str(args.local_root.resolve()),
            "shard_index": index,
            "shard_count": args.shard_count,
            "max_metadata_bytes": args.max_metadata_bytes,
        }
        for index in range(args.shard_count)
    ]
    summaries = []
    totals: Counter[str] = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for summary in pool.map(_worker, payloads):
            summaries.append(summary)
            totals.update(summary.get("status_counts") or {})
            print(
                f"[gate] shard {summary['shard_index']:02d} "
                f"processed={summary['processed']} {summary['status_counts']}",
                flush=True,
            )
    rollup = {
        "schema": "CANONICAL_V2_WAN_AETHER_LOCAL_GATE_ROLLUP_V1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "requests": str(args.requests.resolve()),
        "requests_sha256": sha256_file(args.requests),
        "local_root": str(args.local_root.resolve()),
        "shard_count": args.shard_count,
        "processed": sum(int(item["processed"]) for item in summaries),
        "status_counts": dict(totals),
        "shards": summaries,
        "frozen_816_request_sha256": "5e0854c11f5141780ae797da3f1368cffec270a42b9ec7ab2d4cf31908597cd5",
        "note": "Disjoint from frozen 816; local header/sidecar/raw-id gate only.",
    }
    (output_dir / "rollup.json").write_text(
        json.dumps(rollup, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: rollup[k] for k in ("processed", "status_counts", "requests_sha256", "shard_count")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
