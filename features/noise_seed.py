"""Stable, sample-local noise for canonical InsScene feature extraction.

The seed deliberately excludes diffusion timestep and output layer.  A logical
sample therefore sees the same base noise at every requested timestep and when
only a subset of layers must be resumed.  The generator is local, so sample
ordering, multi-GPU sharding, and unrelated torch RNG calls cannot change it.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import torch


DETERMINISTIC_NOISE_CONTRACT = "inscene_deterministic_noise_v1"
NOISE_SEED_SCHEME = "inscene_noise_seed_v1"
MINIMAX_NOISE_SEED_SCHEME = "minimax_h3_fixed_seed_v1"
MINIMAX_FIXED_NOISE_SEED = 0
NO_NOISE_SEED_SCHEME = "deterministic_forward_no_noise_v1"
_MAX_TORCH_SEED = (1 << 63) - 1


class NoiseSeedError(ValueError):
    """Raised when a canonical sample identity is incomplete or ambiguous."""


def _required_text(value: Any, field: str) -> str:
    if value is None:
        raise NoiseSeedError(f"{field} must be non-empty")
    text = str(value).strip()
    if not text:
        raise NoiseSeedError(f"{field} must be non-empty")
    return text


def canonical_noise_identity(
    *,
    protocol: Mapping[str, Any],
    vfm: str,
    model_id: str,
    scene: str,
    sample_kind: str,
    window_id: int | None = None,
    prefix_len: int | None = None,
    frame_index: int | None = None,
) -> dict[str, Any]:
    """Build the exact identity used by :data:`NOISE_SEED_SCHEME`.

    ``streaming_prefix`` identities contain ``window_id`` and ``prefix_len``;
    ``target_isolated`` identities contain the scene-local ``frame_index``.
    Timestep and layer are intentionally not accepted by this API.
    """

    identity: dict[str, Any] = {
        "canonical_protocol": _required_text(
            protocol.get("protocol"), "canonical protocol"
        ),
        "canonical_protocol_sha256": _required_text(
            protocol.get("protocol_sha256"), "canonical protocol sha256"
        ),
        "vfm": _required_text(vfm, "vfm"),
        "model_id": _required_text(model_id, "model_id"),
        "scene_id": _required_text(scene, "scene"),
        "sample_kind": _required_text(sample_kind, "sample_kind"),
    }
    if len(identity["canonical_protocol_sha256"]) != 64:
        raise NoiseSeedError("canonical protocol sha256 must contain 64 hex digits")
    try:
        int(identity["canonical_protocol_sha256"], 16)
    except ValueError as exc:
        raise NoiseSeedError("canonical protocol sha256 must contain 64 hex digits") from exc

    if sample_kind == "streaming_prefix":
        if window_id is None or prefix_len is None or frame_index is not None:
            raise NoiseSeedError(
                "streaming_prefix requires window_id/prefix_len and forbids frame_index"
            )
        if int(window_id) < 0 or int(prefix_len) <= 0:
            raise NoiseSeedError("window_id must be >= 0 and prefix_len must be > 0")
        identity.update({"window_id": int(window_id), "prefix_len": int(prefix_len)})
    elif sample_kind == "target_isolated":
        if frame_index is None or window_id is not None or prefix_len is not None:
            raise NoiseSeedError(
                "target_isolated requires frame_index and forbids window_id/prefix_len"
            )
        if int(frame_index) < 0:
            raise NoiseSeedError("frame_index must be >= 0")
        identity["frame_index"] = int(frame_index)
    else:
        raise NoiseSeedError(f"unsupported canonical sample_kind: {sample_kind!r}")
    return identity


def noise_seed_from_identity(identity: Mapping[str, Any]) -> int:
    """Hash a canonical identity into torch's non-negative 63-bit seed range."""

    payload = {
        "noise_seed_scheme": NOISE_SEED_SCHEME,
        "identity": dict(identity),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & _MAX_TORCH_SEED


def canonical_noise_sample(**identity_kwargs: Any) -> dict[str, Any]:
    """Return sidecar-ready identity and seed for one logical sample."""

    identity = canonical_noise_identity(**identity_kwargs)
    return {
        "noise_seed": noise_seed_from_identity(identity),
        "noise_seed_identity": identity,
    }


def canonical_noise_metadata(samples: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Return strict sidecar fields for one or more ordered logical samples."""

    normalized = [dict(sample) for sample in samples]
    if not normalized:
        raise NoiseSeedError("canonical noise provenance requires at least one sample")
    for sample in normalized:
        identity = sample.get("noise_seed_identity")
        seed = sample.get("noise_seed")
        if not isinstance(identity, Mapping) or not isinstance(seed, int):
            raise NoiseSeedError("malformed canonical noise sample provenance")
        if seed != noise_seed_from_identity(identity):
            raise NoiseSeedError("canonical noise seed does not match its identity")
    metadata: dict[str, Any] = {
        "feature_generation_contract": DETERMINISTIC_NOISE_CONTRACT,
        "noise_seed_scheme": NOISE_SEED_SCHEME,
        "noise_seed_samples": normalized,
    }
    if len(normalized) == 1:
        metadata.update(normalized[0])
    return metadata


def deterministic_no_noise_metadata() -> dict[str, Any]:
    """Return explicit provenance for deterministic backends with no RNG draw.

    A concrete dictionary is intentional: seeded-v1 cache validation is
    fail-closed and must distinguish a certified deterministic forward from a
    missing noise/provenance contract.
    """

    return {
        "feature_generation_contract": DETERMINISTIC_NOISE_CONTRACT,
        "noise_seed_scheme": NO_NOISE_SEED_SCHEME,
        "noise_seed_scope": "no_random_draws",
    }


def noise_metadata_matches(
    metadata: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    """Exact subset comparison used by fail-closed cache validation."""

    return all(metadata.get(key) == value for key, value in expected.items())


def make_noise_generator(seed: int, device: torch.device | str) -> torch.Generator:
    """Create a fresh generator without modifying torch's global RNG state."""

    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(int(seed))
    return generator


def randn_like_on_generator_device(
    tensor: torch.Tensor,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Draw N(0, 1) on the generator's device in float32, then cast/move.

    CUDA ``Tensor.normal_`` on bf16 (and CUDA ``normal_`` with a CPU
    generator) can silently consume the process-global CUDA RNG.  Aether's
    reconstruction DPM steps and feature noise must not: the same logical
    sample has to hash identically regardless of which other samples ran
    first in the process.
    """

    if generator is None:
        return torch.randn_like(tensor)
    noise = torch.randn(
        tuple(tensor.shape),
        generator=generator,
        device=generator.device,
        dtype=torch.float32,
    )
    return noise.to(device=tensor.device, dtype=tensor.dtype)


def randn_like_local(
    tensor: torch.Tensor,
    generator: torch.Generator | Sequence[torch.Generator] | None = None,
) -> torch.Tensor:
    """Draw like ``randn_like`` while honoring an explicit local generator.

    PyTorch 2.6 does not accept ``generator=`` in ``torch.randn_like``.  Using
    ``empty_like().normal_`` preserves tensor layout and supports both CPU and
    CUDA generators.  The ``None`` branch retains the historical global-RNG
    behavior for non-canonical callers.

    Do not use this helper for Aether: its latents are bf16 and its sample
    generator is CPU-local.  Use :func:`randn_like_on_generator_device`.
    """

    if generator is None:
        return torch.randn_like(tensor)
    if isinstance(generator, (list, tuple)):
        if len(generator) != tensor.shape[0]:
            raise NoiseSeedError(
                "generator list length must match the tensor batch dimension"
            )
        return torch.cat(
            [
                torch.empty_like(tensor[index : index + 1]).normal_(
                    generator=sample_generator
                )
                for index, sample_generator in enumerate(generator)
            ],
            dim=0,
        )
    return torch.empty_like(tensor).normal_(generator=generator)
