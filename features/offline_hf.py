"""Fail-closed resolution of cached Hugging Face model snapshots.

Canonical feature jobs keep the public Hub identifier as their logical model
identity.  Some Diffusers loaders nevertheless call ``model_info`` when that
Hub identifier is supplied, even when all checkpoint files are already
mounted locally and ``HF_HUB_OFFLINE=1`` is set.  This helper resolves only a
revision pinned by ``refs/main`` to its local snapshot directory, so loaders
never need to contact the Hub while sidecars can retain the logical identity.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable


_REVISION = re.compile(r"^[0-9a-f]{40,64}$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hf_hub_cache() -> Path:
    """Resolve the Hub cache using the same environment precedence as the Hub.

    An explicit cache must not silently fall back to another user's snapshot.
    Keep this helper dependency-free so extraction preflight can run before
    importing the model's backend.
    """
    cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if cache:
        return Path(cache).expanduser()
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home).expanduser() / "hub"
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
    return cache_home / "huggingface" / "hub"


def _offline_requested() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _complete_snapshot(snapshot: Path, required_paths: list[str]) -> tuple[bool, list[str]]:
    """Validate required files and every shard named by a safetensors index."""
    missing: list[str] = []
    queue = list(required_paths)
    visited: set[str] = set()
    while queue:
        relative = str(queue.pop())
        if relative in visited:
            continue
        visited.add(relative)
        candidate = snapshot / relative
        # Hugging Face snapshot entries normally are symlinks to blobs.  They
        # are valid here only when the link resolves to a non-empty regular
        # file; a dangling link must not pass an offline preflight.
        if not candidate.exists() or not candidate.is_file() or candidate.stat().st_size <= 0:
            missing.append(relative)
            continue
        if relative.endswith(".safetensors.index.json"):
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
                values = payload.get("weight_map", {}).values()
                shards = sorted({str(value) for value in values})
                if not shards:
                    raise ValueError("empty weight_map")
            except (OSError, ValueError, AttributeError, TypeError):
                missing.append(f"{relative}:invalid_weight_map")
                continue
            queue.extend(str(Path(relative).parent / shard) for shard in shards)
    return not missing, sorted(missing)


def resolve_offline_snapshot(
    model_id: str | None,
    *,
    required_paths: Iterable[str] = (),
) -> tuple[str | None, dict[str, object] | None]:
    """Return a physical local load path plus immutable provenance.

    The first tuple element is suitable for ``from_pretrained``.  The second
    retains the original logical identifier and a content binding for the
    cached revision reference.  For a direct absolute path, the path is
    retained unchanged.  A Hub identifier is never silently passed through
    under offline mode: a missing or incomplete cached snapshot is an error.
    """
    if model_id is None:
        return None, None
    required = [str(value) for value in required_paths]
    text = str(model_id)
    direct = Path(text)
    if direct.is_absolute():
        if not direct.is_dir():
            raise RuntimeError(f"local model path is missing: {direct}")
        complete, missing = _complete_snapshot(direct, required)
        if not complete:
            raise RuntimeError(
                f"local model path is incomplete: {direct}; missing={','.join(missing)}"
            )
        return text, {
            "logical_model_id": text,
            "load_kind": "direct_local_path",
            "runtime_model_id": text,
            "required_paths": required,
        }

    if text.count("/") != 1 or text.startswith("/") or ".." in text.split("/"):
        raise RuntimeError(f"invalid Hugging Face model identifier: {text!r}")
    cache_name = "models--" + text.replace("/", "--")
    hub_cache = _hf_hub_cache()
    root = hub_cache / cache_name
    reference = root / "refs" / "main"
    if not reference.is_symlink() and reference.is_file():
        revision = reference.read_text(encoding="utf-8").strip()
        snapshot = root / "snapshots" / revision
        if _REVISION.fullmatch(revision) and snapshot.is_dir():
            complete, _missing = _complete_snapshot(snapshot, required)
            if complete:
                return str(snapshot), {
                    "logical_model_id": text,
                    "load_kind": "huggingface_cached_snapshot",
                    "runtime_model_id": str(snapshot),
                    "hf_home": str(hub_cache.parent),
                    "hf_hub_cache": str(hub_cache),
                    "resolved_revision": revision,
                    "refs_main_path": str(reference),
                    "refs_main_sha256": _sha256_file(reference),
                    "required_paths": required,
                }

    if _offline_requested():
        raise RuntimeError(
            "offline canonical extraction has no complete revision-pinned local "
            f"snapshot for {text!r}; required={required}"
        )
    return text, {
        "logical_model_id": text,
        "load_kind": "hub_identifier_network_allowed",
        "runtime_model_id": text,
        "required_paths": required,
    }
