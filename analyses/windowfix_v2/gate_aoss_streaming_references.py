#!/usr/bin/env python3
"""Range-read AOSS gate for canonical-v2 legacy streaming references.

The program consumes ``remote_gate_requests.jsonl.gz`` produced by
``build_streaming_reference_catalog.py``.  It downloads only the small
``prefix_index.npy`` and JSON sidecar, then two byte ranges from each
safetensors object (8-byte length and JSON header).  It never downloads or
hashes tensor data.  Network/client failures remain pending; definitive
identity/header failures are fail-closed re-extraction decisions.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import signal
import struct
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REQUESTS = (
    ROOT
    / "artifacts/windowfix_v2/reference_overlay_02e17c40b59a_r11/"
    "remote_gate_requests.jsonl.gz"
)
DEFAULT_CONF = Path("/mnt/umm/users/yinbaiqiao/aoss/petreloss.conf")
SCHEMA = "LEGACY_EXACT_REFERENCE_REMOTE_GATE_RECEIPT_V1"
PASS = "REFERENCE_SOURCE_READ_ONLY"
FAIL = "REEXTRACT_UNVERIFIED"
PENDING = "PENDING_REMOTE_GATE"
HEX64 = set("0123456789abcdef")
DTYPE_ALIASES = {
    "F64": "float64",
    "F32": "float32",
    "F16": "float16",
    "BF16": "bfloat16",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U8": "uint8",
    "BOOL": "bool",
}
DTYPE_BYTES = {
    "float64": 8,
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "uint8": 1,
    "bool": 1,
}
NO_TIMESTEP_VFMS = frozenset(
    {"dino", "f3r", "qwen2_5_vl", "videomae", "vjepa2"}
)


class GateInputError(RuntimeError):
    pass


class RequestTimeout(BaseException):
    """Raised by the per-request wall-clock deadline.

    This deliberately derives directly from :class:`BaseException`.  Some
    HTTP client retry stacks catch every ordinary ``Exception`` and would
    otherwise swallow the alarm once, then resume an unbounded retry.  The
    outer gate loop catches this one control-flow exception explicitly and
    converts it into a fail-closed PENDING receipt.
    """


class request_timeout:
    """SIGALRM guard around one request; a timeout is classified as PENDING."""

    def __init__(self, seconds: int) -> None:
        self.seconds = int(seconds)
        self.previous_handler: Any = None

    def __enter__(self):
        if self.seconds <= 0:
            raise GateInputError("request timeout must be positive")
        self.previous_handler = signal.getsignal(signal.SIGALRM)

        def expired(_signum, _frame):
            raise RequestTimeout(f"remote gate exceeded {self.seconds}s")

        signal.signal(signal.SIGALRM, expired)
        signal.siginterrupt(signal.SIGALRM, True)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, _kind, _value, _traceback):
        signal.alarm(0)
        signal.siginterrupt(signal.SIGALRM, False)
        signal.signal(signal.SIGALRM, self.previous_handler)
        return False


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_uri(uri: str) -> str:
    if uri.startswith("aoss://"):
        return "fj:s3://" + uri[len("aoss://") :]
    if uri.startswith("s3://"):
        return "fj:" + uri
    if uri.startswith("fj:s3://"):
        return uri
    raise GateInputError(f"not an AOSS URI: {uri!r}")


def normalize_dtype(value: Any) -> str:
    text = str(value).strip()
    text = text.removeprefix("torch.").removeprefix("numpy.").removeprefix("np.")
    return DTYPE_ALIASES.get(text.upper(), text.lower())


def valid_sha(value: Any) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and set(text) <= HEX64


def read_requests(path: Path, shard_index: int, shard_count: int) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for offset, line in enumerate(handle):
            if offset % shard_count != shard_index:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise GateInputError(f"request line {offset + 1} is not an object")
            yield value


def index_key(record: Mapping[str, Any]) -> tuple[int, int]:
    window = record.get("window_id", record.get("streaming_window_id"))
    tail = record.get("tail", record.get("streaming_tail"))
    if tail is None:
        tail = int(record["prefix_length"]) - 1
    return int(window), int(tail)


def raw_ids(record: Mapping[str, Any]) -> tuple[int, ...]:
    value = record.get(
        "actual_input_raw_ids", record.get("streaming_indices", record.get("indices"))
    )
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError("index record lacks raw IDs")
    return tuple(int(item) for item in value)


def parse_index(data: bytes) -> dict[tuple[int, int], dict[str, Any]]:
    values = np.load(io.BytesIO(data), allow_pickle=True).tolist()
    if isinstance(values, dict):
        values = [values]
    if not isinstance(values, list):
        raise ValueError("prefix index is not list-like")
    result: dict[tuple[int, int], dict[str, Any]] = {}
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("prefix index row is not an object")
        key = index_key(value)
        if key in result:
            raise ValueError(f"duplicate prefix index key {key}")
        result[key] = value
    return result


def model_matches(expected: Any, observed: Any) -> bool:
    expected_text = str(expected or "").strip().rstrip("/")
    observed_text = str(observed or "").strip().rstrip("/")
    if expected_text == observed_text:
        return True
    if "/" not in expected_text:
        return False
    organisation, repo = expected_text.split("/", 1)
    marker = f"models--{organisation}--{repo}/snapshots/"
    return marker in observed_text and bool(observed_text.rsplit("/", 1)[-1])


class RemoteReader:
    def __init__(self, client: Any, max_metadata_bytes: int) -> None:
        self.client = client
        self.max_metadata_bytes = max_metadata_bytes
        self.index_cache: dict[str, tuple[dict[tuple[int, int], dict[str, Any]], dict[str, Any]]] = {}

    def contains(self, uri: str) -> bool:
        return bool(self.client.contains(normalize_uri(uri)))

    def size(self, uri: str) -> int:
        return int(self.client.size(normalize_uri(uri)))

    def get_small(self, uri: str) -> tuple[bytes, int]:
        if not self.contains(uri):
            raise FileNotFoundError(uri)
        size = self.size(uri)
        if size <= 0 or size > self.max_metadata_bytes:
            raise ValueError(f"metadata object size outside limit: {size}")
        data = self.client.get(normalize_uri(uri))
        if not isinstance(data, bytes) or len(data) != size:
            raise ValueError(f"short metadata read: expected {size}, got {len(data)}")
        return data, size

    def index(self, uri: str) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[str, Any]]:
        if uri not in self.index_cache:
            data, size = self.get_small(uri)
            self.index_cache[uri] = (
                parse_index(data),
                {"uri": uri, "size_bytes": size, "sha256": sha256_bytes(data)},
            )
        return self.index_cache[uri]

    def header(self, uri: str, tensor_key: str) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self.contains(uri):
            raise FileNotFoundError(uri)
        object_size = self.size(uri)
        first = self.client.get(normalize_uri(uri), range="0-7")
        if not isinstance(first, bytes) or len(first) != 8:
            raise ValueError("range read did not return exactly 8 bytes")
        header_size = struct.unpack("<Q", first)[0]
        if header_size <= 1 or header_size > min(object_size - 8, self.max_metadata_bytes):
            raise ValueError(f"invalid safetensors header length: {header_size}")
        body = self.client.get(normalize_uri(uri), range=f"8-{7 + header_size}")
        if not isinstance(body, bytes) or len(body) != header_size:
            raise ValueError("short safetensors header range read")
        value = json.loads(body.decode("utf-8"))
        tensor = value.get(tensor_key)
        if not isinstance(tensor, dict):
            raise ValueError(f"tensor key {tensor_key!r} absent")
        shape = [int(item) for item in tensor.get("shape", ())]
        if not shape or any(item < 0 for item in shape):
            raise ValueError("invalid tensor shape")
        dtype = normalize_dtype(tensor.get("dtype"))
        if dtype not in DTYPE_BYTES:
            raise ValueError("unsupported tensor dtype")
        offsets = tensor.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError("invalid tensor data offsets")
        start, end = int(offsets[0]), int(offsets[1])
        payload_size = object_size - 8 - header_size
        if start < 0 or end < start or end > payload_size:
            raise ValueError("tensor offsets exceed object")
        elements = 1
        for dimension in shape:
            elements *= dimension
        if end - start != elements * DTYPE_BYTES[dtype]:
            raise ValueError("tensor bytes disagree with shape/dtype")
        return (
            {"shape": shape, "dtype": dtype, "data_offsets": [start, end]},
            {
                "uri": uri,
                "object_size_bytes": object_size,
                "header_size_bytes": header_size,
                "header_sha256": sha256_bytes(first + body),
                "payload_body_read": False,
            },
        )


def check_request(reader: RemoteReader, request: Mapping[str, Any]) -> dict[str, Any]:
    checked_at = datetime.now(timezone.utc).isoformat()
    unit_id = str(request.get("unit_id", ""))
    request_sha = sha256_bytes(canonical_json(request).encode("utf-8"))
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "unit_id": unit_id,
        "request_sha256": request_sha,
        "source_uri": request.get("source_uri"),
        "source_index_uri": request.get("source_index_uri"),
        "checked_at": checked_at,
        "payload_full_read_performed": False,
        "byte_level_certified": False,
    }
    try:
        expected = request["expected"]
        expected_shape = expected.get("expected_shape")
        # A missing exact shape is not a remote object failure.  It is an
        # unresolved extractor contract, so keep the unit pending without
        # touching AOSS and never promote an internally self-consistent but
        # wrong-shaped legacy tensor.
        if (
            not isinstance(expected_shape, list)
            or not expected_shape
            or any(not isinstance(item, int) or item <= 0 for item in expected_shape)
        ):
            receipt.update(
                status=PENDING,
                reason_codes=["BACKEND_EXACT_SHAPE_CANARY_REQUIRED"],
                error="request has no canary-bound exact expected_shape",
            )
            return receipt
        source_uri = str(request["source_uri"])
        sidecar_uri = str(request["sidecar_uri"])
        index_uri = str(request["source_index_uri"])
        record_key = (
            int(request["source_index_record"]["window_id"]),
            int(request["source_index_record"]["tail"]),
        )
        records, index_evidence = reader.index(index_uri)
        record = records.get(record_key)
        expected_raw = tuple(int(item) for item in expected["raw_ids"])
        checks: dict[str, bool] = {
            "source_index_record_exists": record is not None,
            "source_index_raw_ids_exact": record is not None
            and raw_ids(record) == expected_raw,
        }

        sidecar_bytes, sidecar_size = reader.get_small(sidecar_uri)
        sidecar = json.loads(sidecar_bytes)
        if not isinstance(sidecar, dict):
            raise ValueError("feature sidecar is not an object")
        tensor_key = str(expected.get("tensor_key", "feat"))
        header_tensor, header_evidence = reader.header(source_uri, tensor_key)
        source_size = int(header_evidence["object_size_bytes"])
        sidecar_tensor = sidecar.get("tensors", {}).get(tensor_key, {})
        sidecar_shape = [int(item) for item in sidecar_tensor.get("shape", ())]
        sidecar_dtype = normalize_dtype(sidecar_tensor.get("dtype"))
        prompt_contract = expected["prompt_contract"]
        preprocess = expected["preprocess_identity"]
        vfm = str(expected["vfm"])
        checks.update(
            {
                "sidecar_file": sidecar.get("file") == expected["feature_filename"],
                "sidecar_size": int(sidecar.get("size_bytes", -1)) == source_size,
                "sidecar_vfm": sidecar.get("vfm") == vfm,
                "sidecar_model": model_matches(expected["model_id"], sidecar.get("model_id")),
                "sidecar_mode": sidecar.get("mode") == expected["mode"],
                "sidecar_frame_index_schema": sidecar.get("frame_index_schema")
                == expected["frame_index_schema"],
                "sidecar_resize": sidecar.get("resize") == preprocess["resize"],
                "sidecar_num_frames": int(sidecar.get("num_frames", -1))
                == int(preprocess["num_frames"]),
                "sidecar_timestep": vfm in NO_TIMESTEP_VFMS
                or int(sidecar.get("timestep", -1)) == int(expected["timestep"]),
                "sidecar_prompt": prompt_contract["mode"] == "not_applicable"
                or sidecar.get("prompt") == prompt_contract["value"],
                "sidecar_tensor_shape_present": bool(sidecar_shape),
                "sidecar_tensor_shape_equals_header": sidecar_shape
                == header_tensor["shape"],
                "header_shape_equals_expected": header_tensor["shape"]
                == expected_shape,
                "sidecar_tensor_dtype_equals_header": sidecar_dtype
                == header_tensor["dtype"],
                "header_dtype_equals_expected": header_tensor["dtype"]
                == normalize_dtype(expected["expected_dtype"]),
            }
        )
        receipt["checks"] = checks
        receipt["evidence"] = {
            "source_index": index_evidence,
            "sidecar": {
                "uri": sidecar_uri,
                "size_bytes": sidecar_size,
                "sha256": sha256_bytes(sidecar_bytes),
                "declared_payload_sha256_unverified": (
                    str(sidecar.get("sha256")).lower()
                    if valid_sha(sidecar.get("sha256"))
                    else None
                ),
            },
            "safetensors": header_evidence,
            "observed_tensor": header_tensor,
        }
        failed = sorted(key for key, passed in checks.items() if not passed)
        if failed:
            receipt.update(status=FAIL, reason_codes=[f"CHECK_FAILED:{x}" for x in failed])
        else:
            receipt.update(
                status=PASS,
                reason_codes=["LEGACY_EXACT_REFERENCE_V1_REMOTE_RANGE_GATE_PASS"],
            )
    except (GateInputError, KeyError, TypeError, ValueError, FileNotFoundError) as exc:
        receipt.update(
            status=FAIL,
            reason_codes=[f"DEFINITIVE_GATE_FAILURE:{type(exc).__name__}"],
            error=str(exc),
        )
    except Exception as exc:  # network/backend errors are retriable, never a PASS
        receipt.update(
            status=PENDING,
            reason_codes=[f"REMOTE_GATE_ERROR:{type(exc).__name__}"],
            error=str(exc),
        )
    return receipt


def request_timeout_receipt(
    request: Mapping[str, Any], *, seconds: int, error: RequestTimeout
) -> dict[str, Any]:
    """Build the canonical retriable receipt for a hard wall-clock timeout."""

    return {
        "schema": SCHEMA,
        "unit_id": str(request.get("unit_id", "")),
        "request_sha256": sha256_bytes(canonical_json(request).encode("utf-8")),
        "source_uri": request.get("source_uri"),
        "source_index_uri": request.get("source_index_uri"),
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "payload_full_read_performed": False,
        "byte_level_certified": False,
        "status": PENDING,
        "reason_codes": ["REMOTE_GATE_REQUEST_TIMEOUT"],
        "error": str(error),
        "request_timeout_seconds": int(seconds),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise GateInputError("invalid shard index/count")
    if not args.requests.is_file() or not args.conf.is_file():
        raise GateInputError("request catalogue or AOSS config missing")
    if args.output_dir.exists():
        raise GateInputError(f"output already exists: {args.output_dir}")
    from aoss_client.client import Client

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    client = Client(str(args.conf))
    reader = RemoteReader(client, args.max_metadata_bytes)
    requests_sha = sha256_file(args.requests)
    counts: Counter[str] = Counter()
    processed = 0
    with tempfile.TemporaryDirectory(
        prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent
    ) as tmp:
        stage = Path(tmp)
        receipts_path = stage / "receipts.jsonl.gz"
        with receipts_path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
                with io.TextIOWrapper(zipped, encoding="utf-8") as handle:
                    for request in read_requests(
                        args.requests, args.shard_index, args.shard_count
                    ):
                        if args.max_records is not None and processed >= args.max_records:
                            break
                        try:
                            with request_timeout(args.request_timeout_seconds):
                                receipt = check_request(reader, request)
                        except RequestTimeout as exc:
                            receipt = request_timeout_receipt(
                                request,
                                seconds=args.request_timeout_seconds,
                                error=exc,
                            )
                        receipt["request_timeout_seconds"] = args.request_timeout_seconds
                        handle.write(canonical_json(receipt) + "\n")
                        counts[receipt["status"]] += 1
                        processed += 1
        summary = {
            "schema": "LEGACY_EXACT_REFERENCE_REMOTE_GATE_SUMMARY_V1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "requests": str(args.requests),
            "requests_sha256": requests_sha,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "processed": processed,
            "status_counts": dict(sorted(counts.items())),
            "payload_full_read_performed": False,
            "receipts_file": receipts_path.name,
            "receipts_sha256": sha256_file(receipts_path),
        }
        write_json(stage / "summary.json", summary)
        os.rename(stage, args.output_dir)
    return args.output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, default=DEFAULT_REQUESTS)
    parser.add_argument("--conf", type=Path, default=DEFAULT_CONF)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-metadata-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--request-timeout-seconds", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    try:
        output = run(parse_args())
    except GateInputError as exc:
        raise SystemExit(f"ERROR: {exc}")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
