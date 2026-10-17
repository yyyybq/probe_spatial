"""Local/AOSS filesystem helpers for InsScene-15K training I/O.

Paths starting with ``cluster:s3://`` (e.g. ``fj:s3://baiqiao/...``), bare
``s3://``, or ``aoss://`` are served via ``aoss_client``. Everything else uses
the local FS.
"""

from __future__ import annotations

import fnmatch
import hashlib
import io
import json
import os
import re
import threading
import time
import uuid
from typing import Any, Iterable, List, Optional

import numpy as np
from PIL import Image

_CLIENT = None
_CLIENT_LOCK = threading.Lock()
_CLIENT_PID = None
_METRICS_LOCK = threading.Lock()

# Per-process cache of remote ``exists()`` results (True and False). Safe
# because dataset readers only probe static pre-extracted inputs that do not
# appear/disappear mid-run; never used for checkpoints/outputs.
_EXISTS_RESULT_CACHE: dict[str, bool] = {}
_EXISTS_RESULT_CACHE_LOCK = threading.Lock()
_EXISTS_RESULT_CACHE_MAX = 200_000

_REMOTE_RE = re.compile(r"^(?:(?:[A-Za-z0-9_-]+:)?s3|aoss)://")


def is_remote(path: str) -> bool:
    return bool(path) and bool(_REMOTE_RE.match(path))


def _client_path(path: str) -> str:
    """Map ``aoss://bucket/key`` / ``s3://bucket/key`` to ``fj:s3://bucket/key``."""
    if path.startswith("aoss://"):
        return "fj:s3://" + path[len("aoss://") :]
    if path.startswith("s3://"):
        return "fj:s3://" + path[len("s3://") :]
    return path


def default_conf_path() -> str:
    return os.environ.get(
        "AOSS_CONF",
        os.path.expanduser("~/.petreloss.conf"),
    )


def get_client():
    global _CLIENT, _CLIENT_PID
    pid = os.getpid()
    with _CLIENT_LOCK:
        if _CLIENT is None or _CLIENT_PID != pid:
            from aoss_client.client import Client

            _CLIENT = Client(default_conf_path())
            _CLIENT_PID = pid
        return _CLIENT


def join(*parts: str) -> str:
    if not parts:
        return ""
    base = parts[0]
    rest = [p.strip("/") for p in parts[1:] if p is not None and str(p) != ""]
    if not rest:
        return base
    if base.endswith("/"):
        return base + "/".join(rest)
    return base.rstrip("/") + "/" + "/".join(rest)


def cache_dir() -> str:
    root = os.environ.get("AOSS_CACHE_DIR") or os.environ.get("AOSS_CACHE_ROOT")
    if not root:
        for candidate in ("/mnt/local_cache/inscene/aoss_cache", "/tmp/inscene_aoss_cache"):
            try:
                os.makedirs(candidate, exist_ok=True)
                root = candidate
                break
            except Exception:
                continue
        if not root:
            root = "/tmp/inscene_aoss_cache"
            os.makedirs(root, exist_ok=True)
    else:
        os.makedirs(root, exist_ok=True)
    return root


def _emit_metric(payload: dict[str, Any]) -> None:
    path = os.environ.get("AOSS_FS_METRICS")
    if not path:
        return
    try:
        with _METRICS_LOCK:
            with open(path, "a", encoding="utf-8") as fh:
                payload = dict(payload)
                payload.setdefault("ts", time.time())
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _local_cache_path(path: str) -> str:
    digest = hashlib.sha1(path.encode("utf-8")).hexdigest()
    base = os.path.basename(path.rstrip("/")) or "object"
    return os.path.join(cache_dir(), f"{digest}_{base}")


def _remember_exists(path: str, found: bool) -> None:
    with _EXISTS_RESULT_CACHE_LOCK:
        if len(_EXISTS_RESULT_CACHE) >= _EXISTS_RESULT_CACHE_MAX:
            _EXISTS_RESULT_CACHE.clear()
        _EXISTS_RESULT_CACHE[path] = found


def exists(path: str) -> bool:
    if not path:
        return False
    if is_remote(path):
        local = _local_cache_path(path)
        if os.path.exists(local) and os.path.getsize(local) > 0:
            _emit_metric({"event": "exists_local_hit", "path": path})
            return True
        with _EXISTS_RESULT_CACHE_LOCK:
            if path in _EXISTS_RESULT_CACHE:
                _emit_metric({"event": "exists_cache_hit", "path": path})
                return _EXISTS_RESULT_CACHE[path]
        _emit_metric({"event": "exists_remote_check", "path": path})
        try:
            client_path = _client_path(path)
            found = bool(get_client().contains(client_path)) or bool(
                get_client().isdir(client_path)
            )
        except Exception as query_error:
            try:
                prefix = path if path.endswith("/") else path + "/"
                found = any(True for _ in get_client().list(_client_path(prefix)))
            except Exception as exc:
                # A transport/authentication failure is not evidence that an
                # input is missing, and must never enter the negative cache.
                raise OSError(f"Unable to query remote path: {path}") from exc
            if not found:
                # An object can exist without child keys. An empty prefix
                # listing cannot resolve a failed lookup of the object itself.
                raise OSError(f"Unable to query remote path: {path}") from query_error
        _remember_exists(path, found)
        return found
    return os.path.exists(path)


def isdir(path: str) -> bool:
    if not path:
        return False
    if is_remote(path):
        try:
            return bool(get_client().isdir(_client_path(path)))
        except Exception as query_error:
            try:
                prefix = path if path.endswith("/") else path + "/"
                found = any(True for _ in get_client().list(_client_path(prefix)))
            except Exception as exc:
                raise OSError(f"Unable to query remote directory: {path}") from exc
            if not found:
                raise OSError(f"Unable to query remote directory: {path}") from query_error
            return True
    return os.path.isdir(path)


def listdir(path: str) -> List[str]:
    if is_remote(path):
        prefix = path if path.endswith("/") else path + "/"
        names: List[str] = []
        for item in get_client().list(_client_path(prefix)):
            name = str(item).rstrip("/")
            if "/" in name:
                name = name.split("/")[-1]
            if name:
                names.append(name)
        return names
    return os.listdir(path)


def read_bytes(path: str) -> bytes:
    if is_remote(path):
        data = get_client().get(_client_path(path))
        if data is None:
            raise FileNotFoundError(path)
        return bytes(data)
    with open(path, "rb") as fh:
        return fh.read()


def open_binary(path: str):
    return io.BytesIO(read_bytes(path))


def load_json(path: str) -> Any:
    return json.loads(read_bytes(path).decode("utf-8"))


def load_npy(path: str, *args: Any, **kwargs: Any) -> Any:
    if "allow_pickle" not in kwargs:
        kwargs["allow_pickle"] = True
    return np.load(io.BytesIO(read_bytes(path)), *args, **kwargs)


def load_image(path: str) -> Image.Image:
    return Image.open(io.BytesIO(read_bytes(path)))


def materialize(path: str) -> str:
    """Download a remote object into AOSS_CACHE_DIR and return the local path."""
    if not is_remote(path):
        return path
    local = _local_cache_path(path)
    if os.path.exists(local) and os.path.getsize(local) > 0:
        _emit_metric({"event": "materialize_hit", "path": path})
        return local
    os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
    tmp = f"{local}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    _emit_metric({"event": "materialize_fetch", "path": path})
    data = read_bytes(path)
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, local)
    return local


def load_safetensors(path: str) -> dict[str, Any]:
    from safetensors.torch import load_file

    try:
        local = materialize(path)
        return load_file(local)
    except Exception:
        if not is_remote(path):
            raise
        # Corrupt / truncated cache: refetch once.
        local = _local_cache_path(path)
        try:
            if os.path.exists(local):
                os.remove(local)
        except Exception:
            pass
        local = materialize(path)
        return load_file(local)


def glob(pattern: str) -> List[str]:
    """Limited glob: supports trailing ``*`` / ``scene_*`` style under one directory."""
    if not is_remote(pattern):
        import glob as _glob

        return sorted(_glob.glob(pattern))

    # Split into directory prefix + name pattern.
    if pattern.endswith("/"):
        directory, name_pat = pattern.rstrip("/"), "*"
    else:
        directory, name_pat = pattern.rsplit("/", 1) if "/" in pattern else ("", pattern)
    if not directory:
        raise ValueError(f"unsupported remote glob pattern: {pattern}")
    names = listdir(directory)
    return sorted(join(directory, name) for name in names if fnmatch.fnmatch(name, name_pat))
