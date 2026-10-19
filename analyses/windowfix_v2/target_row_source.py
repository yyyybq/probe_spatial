"""Local and AOSS byte-range readers for aggregate safetensors target rows."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch


REMOTE_RE = re.compile(r"^(?:(?:[A-Za-z0-9_-]+:)?s3|aoss)://")
DTYPE_BYTES = {
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}
TORCH_DTYPES = {
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}
NORMAL_DTYPES = {
    "F16": "torch.float16",
    "BF16": "torch.bfloat16",
    "F32": "torch.float32",
    "F64": "torch.float64",
    "I64": "torch.int64",
    "I32": "torch.int32",
    "I16": "torch.int16",
    "I8": "torch.int8",
    "U8": "torch.uint8",
    "BOOL": "torch.bool",
}


class SourceReadError(ValueError):
    pass


class RemoteSourcePending(SourceReadError):
    """A remote gate could not run to a definitive result (retry elsewhere)."""


class RemoteSourceMissing(SourceReadError):
    """A remote endpoint definitively reported that an object is absent."""


@dataclass(frozen=True)
class TensorLayout:
    location: str
    tensor_key: str
    shape: tuple[int, ...]
    dtype: str
    safetensors_dtype: str
    file_size: int
    header_size: int
    header_sha256: str
    tensor_data_start: int
    tensor_data_end: int
    row_nbytes: int


def is_remote(location: str | Path) -> bool:
    return bool(REMOTE_RE.match(str(location)))


def basename(location: str | Path) -> str:
    return PurePosixPath(str(location).rstrip("/")).name


def _client_path(location: str) -> str:
    if location.startswith("aoss://"):
        return "fj:s3://" + location[len("aoss://") :]
    if location.startswith("s3://"):
        return "fj:s3://" + location[len("s3://") :]
    return location


def _get_client():
    from vidfm3d.utils.aoss_fs import get_client

    return get_client()


def require_local_regular(location: str | Path, label: str) -> Path:
    path = Path(str(location))
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise SourceReadError(f"{label} must be an absolute regular non-symlink file: {path}")
    return path


def read_range(location: str | Path, start: int, length: int) -> bytes:
    if start < 0 or length < 0:
        raise SourceReadError("negative byte range")
    if length == 0:
        return b""
    text = str(location)
    if not is_remote(text):
        path = require_local_regular(text, "source object")
        with path.open("rb") as handle:
            handle.seek(start)
            data = handle.read(length)
    else:
        end = start + length - 1
        try:
            value = _get_client().get(_client_path(text), range=f"{start}-{end}")
        except Exception as exc:
            raise RemoteSourcePending(f"AOSS range read failed for {text}: {exc}") from exc
        if value is None:
            raise RemoteSourceMissing(f"AOSS object is missing: {text}")
        raw = bytes(value)
        if len(raw) == length:
            data = raw
        elif start == 0 and len(raw) >= length:
            data = raw[:length]
        elif len(raw) >= start + length:
            # Defensive fallback for clients/endpoints that ignored Range.
            data = raw[start : start + length]
        elif len(raw) > length:
            data = raw[:length]
        else:
            data = raw
    if len(data) != length:
        raise SourceReadError(
            f"short byte-range read for {text}: offset={start} wanted={length} got={len(data)}"
        )
    return data


def read_all(location: str | Path) -> bytes:
    text = str(location)
    if not is_remote(text):
        return require_local_regular(text, "source object").read_bytes()
    try:
        value = _get_client().get(_client_path(text))
    except Exception as exc:
        raise RemoteSourcePending(f"AOSS read failed for {text}: {exc}") from exc
    if value is None:
        raise RemoteSourceMissing(f"AOSS object is missing: {text}")
    return bytes(value)


def read_json(location: str | Path, label: str) -> dict[str, Any]:
    try:
        raw = read_all(location)
    except SourceReadError:
        raise
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceReadError(f"cannot read {label}: {location}: {exc}") from exc
    if not isinstance(value, dict):
        raise SourceReadError(f"{label} must contain an object: {location}")
    return value


def read_npy_int64(location: str | Path) -> tuple[int, ...]:
    try:
        raw = read_all(location)
    except SourceReadError:
        raise
    try:
        array = np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception as exc:
        raise SourceReadError(f"cannot read target index {location}: {exc}") from exc
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise SourceReadError("target index must be a one-dimensional integer array")
    values = tuple(int(value) for value in array.tolist())
    if len(values) != len(set(values)):
        raise SourceReadError("target index contains duplicate raw IDs")
    if any(value < 0 for value in values):
        raise SourceReadError("target index contains negative raw IDs")
    return values


def object_size(location: str | Path, declared_size: Any = None) -> int:
    text = str(location)
    if not is_remote(text):
        return require_local_regular(text, "source object").stat().st_size
    try:
        declared = int(declared_size)
    except (TypeError, ValueError) as exc:
        raise SourceReadError(
            f"remote source requires sidecar size_bytes: {text}"
        ) from exc
    if declared <= 8:
        raise SourceReadError(f"invalid remote source size_bytes: {text}")
    try:
        actual = int(_get_client().size(_client_path(text)))
    except Exception as exc:
        raise RemoteSourcePending(f"cannot stat AOSS object {text}: {exc}") from exc
    if actual != declared:
        raise SourceReadError(
            f"AOSS object size differs from sidecar: {text}: {actual} != {declared}"
        )
    return actual


def sha256_object(location: str | Path) -> str:
    text = str(location)
    if is_remote(text):
        return hashlib.sha256(read_all(text)).hexdigest()
    digest = hashlib.sha256()
    with require_local_regular(text, "source object").open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_layout(
    location: str | Path,
    tensor_key: str,
    *,
    declared_size: Any = None,
) -> TensorLayout:
    text = str(location)
    size = object_size(text, declared_size)
    prefix = read_range(text, 0, 8)
    header_size = struct.unpack("<Q", prefix)[0]
    if header_size <= 1 or header_size > min(size - 8, 128 * 1024 * 1024):
        raise SourceReadError(f"invalid safetensors header length: {text}")
    header_bytes = read_range(text, 8, header_size)
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceReadError(f"invalid safetensors header JSON: {text}") from exc
    tensor = header.get(tensor_key)
    if not isinstance(tensor, dict):
        raise SourceReadError(f"safetensors tensor {tensor_key!r} is missing: {text}")
    shape_value = tensor.get("shape")
    if not isinstance(shape_value, list) or not shape_value:
        raise SourceReadError(f"safetensors shape is missing: {text}")
    shape = tuple(int(value) for value in shape_value)
    if any(value < 0 for value in shape):
        raise SourceReadError(f"safetensors shape contains a negative dimension: {text}")
    dtype_code = str(tensor.get("dtype", ""))
    if dtype_code not in DTYPE_BYTES:
        raise SourceReadError(f"unsupported safetensors dtype {dtype_code!r}: {text}")
    offsets = tensor.get("data_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or any(not isinstance(value, int) for value in offsets)
        or offsets[0] < 0
        or offsets[1] < offsets[0]
    ):
        raise SourceReadError(f"invalid safetensors data_offsets: {text}")
    data_size = size - 8 - header_size
    if offsets[1] > data_size:
        raise SourceReadError(f"safetensors data_offsets exceed payload: {text}")
    elements = int(np.prod(shape, dtype=np.int64))
    expected_bytes = elements * DTYPE_BYTES[dtype_code]
    if offsets[1] - offsets[0] != expected_bytes:
        raise SourceReadError(f"safetensors byte length disagrees with shape/dtype: {text}")
    if shape[0] <= 0 or expected_bytes % shape[0]:
        raise SourceReadError(f"safetensors first dimension cannot define rows: {text}")
    return TensorLayout(
        location=text,
        tensor_key=tensor_key,
        shape=shape,
        dtype=NORMAL_DTYPES[dtype_code],
        safetensors_dtype=dtype_code,
        file_size=size,
        header_size=header_size,
        header_sha256=hashlib.sha256(header_bytes).hexdigest(),
        tensor_data_start=8 + header_size + int(offsets[0]),
        tensor_data_end=8 + header_size + int(offsets[1]),
        row_nbytes=expected_bytes // shape[0],
    )


def read_tensor_row(layout: TensorLayout, row_index: int) -> tuple[torch.Tensor, str]:
    if row_index < 0 or row_index >= layout.shape[0]:
        raise SourceReadError(
            f"row {row_index} out of bounds for shape {layout.shape}: {layout.location}"
        )
    offset = layout.tensor_data_start + row_index * layout.row_nbytes
    payload = read_range(layout.location, offset, layout.row_nbytes)
    digest = hashlib.sha256(payload).hexdigest()
    # bytearray owns writable storage, avoiding torch.frombuffer warnings and
    # keeping the returned tensor independent of the response buffer.
    tensor = torch.frombuffer(
        bytearray(payload), dtype=TORCH_DTYPES[layout.safetensors_dtype]
    ).clone()
    return tensor.reshape(layout.shape[1:]), digest
