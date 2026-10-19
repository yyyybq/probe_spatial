#!/usr/bin/env python3
"""Build the canonical-v2 feature inventory for the frozen 1,096 experiments.

This tool is deliberately inventory-only.  It reads the frozen experiment
ledger, the canonical-v2 protocol, and historical ``run_manifest.json`` files.
It does not inspect remote caches, submit jobs, or decide that a legacy tensor
is reusable.  Every emitted row therefore starts in ``PENDING_CACHE_AUDIT``.

The feature identity is deduplicated across probing tasks and campaigns by

    (canonical protocol, VFM, model id, physical feature layer, timestep,
     prefix length).

``layer`` in an experiment name is not always the physical cache layer.  The
resolved run manifests prove two historical aliases: MiniMax-H3 ``l50`` reads
physical layer 49, and LingBot ``l48`` reads physical layer 47.  Both identities
are retained in the output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = ROOT / "experiments/task_status_1096_before_expand_20260822.csv"
DEFAULT_PROTOCOL = ROOT / "protocols/inscene_streaming_canonical_v2.json"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts/windowfix_v2/feature_inventory"

SCHEMA_VERSION = 1
EXPECTED_LEDGER_SHA256 = "9c7d6807b701de4b052819971a5c011e29c717e1c3e958ec9ae23116cdcfe779"
EXPECTED_CAMPAIGN_COUNTS = {
    "default8": 224,
    "wan_cog_layer_sweep": 292,
    "videomae_vjepa_layer_sweep": 280,
    "minimax_h3_layer_benchmark": 160,
    "lingbot_video_layer_benchmark": 140,
}
TARGET_PROBE_FAMILIES = frozenset({"C1", "C2", "C3"})

# These are the concrete backbone identities used by the historical campaigns
# and by the current extraction entrypoints.  Aether loads its checkpoint
# internally, while the other values are CLI model ids (or the two checked-in
# cluster-local model locations).  Keeping this mapping explicit prevents a
# logical VFM name from silently changing checkpoints between inventory runs.
MODEL_IDS = {
    "aether": "AetherWorldModel/AetherV1",
    "cogvideox": "THUDM/CogVideoX-5b-I2V",
    "dino": "facebook/dinov2-large",
    "f3r": "jedyang97/Fast3R_ViT_Large_512",
    "lingbot": "/mnt/umm/users/yinbaiqiao/Consistency/lingbot-video/models/lingbot-video-moe-30b-a3b",
    "minimax_h3": "/mnt/umm/users/yinbaiqiao/Consistency/minimax-h3/models/MiniMax-H3",
    "qwen2_5_vl": "Qwen/Qwen2.5-VL-7B-Instruct",
    "videomae": "MCG-NJU/videomae-large",
    "vjepa2": "facebook/vjepa2-vitl-fpc64-256",
    "wan": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
}

MODEL_ID_SOURCES = {
    "aether": "features/aether/aether_feature.py:internal_checkpoint",
    "qwen2_5_vl": "historical_features/run_inscene15k_mllm.py:default_model",
}

TIMESTEP_SEMANTICS = {
    "wan": "diffusion_scheduler_index",
    "cogvideox": "diffusion_scheduler_index",
    "aether": "diffusion_scheduler_index",
    "minimax_h3": "continuous_t_x1000",
    "lingbot": "flow_sigma_x1000",
    "dino": "config_compatibility_only",
    "f3r": "config_compatibility_only",
    "qwen2_5_vl": "config_compatibility_only",
    "videomae": "config_compatibility_only",
    "vjepa2": "config_compatibility_only",
}

CSV_FIELDS = [
    "spec_id",
    "canonical_protocol",
    "protocol_sha256",
    "protocol_file_sha256",
    "vfm",
    "model",
    "model_variant",
    "model_id",
    "model_id_source",
    "ledger_layers",
    "feature_layer",
    "timestep",
    "timestep_semantics",
    "prefix",
    "feature_filename",
    "extractor_entrypoint",
    "extractor_status",
    "campaigns",
    "probe_families",
    "tasks",
    "experiment_count",
    "source_experiment_ids_sha256",
    "requires_streaming",
    "requires_target",
    "canonical_windows",
    "target_scenes",
    "run_manifest_count",
    "missing_run_manifest_count",
    "resolution_sources",
    "status",
    "action",
]


class InventoryError(RuntimeError):
    """Raised when an input cannot define an unambiguous feature inventory."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_int(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise InventoryError(f"{field} must be an integer, got {value!r}") from exc


def _resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _run_manifest_path(row: Mapping[str, str], project_root: Path) -> Path | None:
    run_artifact = str(row.get("run_artifact_path", "")).strip()
    if not run_artifact:
        return None
    return _resolve_path(run_artifact, project_root) / "run_manifest.json"


def _read_manifest_resolution(
    row: Mapping[str, str], project_root: Path
) -> tuple[tuple[str, int, int] | None, str, str]:
    """Return ``((vfm, physical_layer, timestep), source, path)``."""

    path = _run_manifest_path(row, project_root)
    if path is None or not path.is_file():
        return None, "missing_run_manifest", "" if path is None else str(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InventoryError(f"cannot read run manifest {path}: {exc}") from exc
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise InventoryError(f"run manifest lacks an object-valued config: {path}")
    fields = (config.get("vfm_name"), config.get("feature_layer"), config.get("feature_timestep"))
    if any(value is None or str(value).strip() == "" for value in fields):
        return None, "incomplete_run_manifest", str(path)
    resolved = (
        str(fields[0]).strip(),
        _as_int(fields[1], f"{path}:config.feature_layer"),
        _as_int(fields[2], f"{path}:config.feature_timestep"),
    )
    return resolved, "run_manifest", str(path)


def _validate_ledger(
    rows: Sequence[Mapping[str, str]],
    *,
    ledger_path: Path,
    require_frozen_ledger: bool,
    expected_rows: int | None,
) -> None:
    required = {
        "experiment_id",
        "campaign",
        "model",
        "model_variant",
        "layer",
        "prefix",
        "probe_config",
        "probe_family",
        "run_artifact_path",
    }
    if not rows:
        raise InventoryError(f"ledger is empty: {ledger_path}")
    missing = required - set(rows[0])
    if missing:
        raise InventoryError(f"ledger lacks required columns: {sorted(missing)}")
    if expected_rows is not None and len(rows) != expected_rows:
        raise InventoryError(f"expected {expected_rows} ledger rows, found {len(rows)}")
    identifiers = [str(row["experiment_id"]).strip() for row in rows]
    duplicate_ids = sorted(key for key, count in Counter(identifiers).items() if count > 1)
    if duplicate_ids:
        raise InventoryError(f"duplicate experiment_id values: {duplicate_ids[:5]}")
    if any(not identifier for identifier in identifiers):
        raise InventoryError("experiment_id must not be empty")
    if require_frozen_ledger:
        actual_sha = sha256_file(ledger_path)
        if actual_sha != EXPECTED_LEDGER_SHA256:
            raise InventoryError(
                "frozen ledger SHA256 changed: "
                f"expected {EXPECTED_LEDGER_SHA256}, found {actual_sha}"
            )
        campaign_counts = Counter(str(row["campaign"]) for row in rows)
        if dict(campaign_counts) != EXPECTED_CAMPAIGN_COUNTS:
            raise InventoryError(
                "frozen campaign counts changed: "
                f"expected {EXPECTED_CAMPAIGN_COUNTS}, found {dict(campaign_counts)}"
            )


def extractor_protocol_payload(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Mirror ``features.run_inscene15k.canonical_protocol_metadata``."""

    window = protocol.get("canonical_window", {})
    streaming = protocol.get("streaming", {})
    target = protocol.get("target", {})
    return {
        "schema_version": 2,
        "protocol": str(protocol["canonical_protocol"]),
        "sampling": str(window["sampling"]),
        "canonical_obs_len": _as_int(window["observations"], "canonical observations"),
        "prefixes": [_as_int(value, "streaming prefix") for value in streaming["prefix_lengths"]],
        "target_prefixes": [_as_int(value, "target prefix") for value in target["prefix_lengths"]],
        "horizons": [_as_int(value, "target horizon") for value in target["future_horizons"]],
        "include_anchors": bool(target["include_anchors"]),
        "motion_step": float(window["motion_step"]),
        "rotation_weight": float(window["rotation_weight"]),
        "window_stride": _as_int(window["window_stride"], "window stride"),
        "max_windows_per_scene": _as_int(
            window["max_windows_per_scene"], "max windows per scene"
        ),
    }


def _feature_filename(vfm: str, layer: int, timestep: int) -> str:
    if vfm == "dino":
        return "feature.sft"
    if vfm == "f3r":
        return f"feature_l{layer}.sft"
    if vfm in {"vjepa2", "videomae", "qwen2_5_vl"}:
        return f"feature_layer{layer}.sft"
    return f"feature_t{timestep}_layer{layer}.sft"


def _extractor_entrypoint(vfm: str, project_root: Path) -> tuple[str, str]:
    relative = (
        "features/run_inscene15k_mllm.py"
        if vfm == "qwen2_5_vl"
        else "features/run_inscene15k.py"
    )
    status = "PRESENT" if (project_root / relative).is_file() else "MISSING_IN_WORKTREE"
    return relative, status


def _safe_number(value: int) -> str:
    return f"m{abs(value)}" if value < 0 else str(value)


def _spec_id(
    *, protocol_id: str, vfm: str, model_id: str, layer: int, timestep: int, prefix: int
) -> str:
    identity = {
        "protocol": protocol_id,
        "vfm": vfm,
        "model_id": model_id,
        "feature_layer": layer,
        "timestep": timestep,
        "prefix": prefix,
    }
    suffix = sha256_json(identity)[:10]
    return (
        f"cv2_{re.sub(r'[^a-z0-9]+', '_', vfm.lower()).strip('_')}"
        f"_t{_safe_number(timestep)}_l{_safe_number(layer)}_p{prefix}_{suffix}"
    )


def _joined(values: Iterable[Any]) -> str:
    return ";".join(str(value) for value in sorted(set(values), key=lambda item: str(item)))


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def build_inventory(
    *,
    ledger_path: Path = DEFAULT_LEDGER,
    protocol_path: Path = DEFAULT_PROTOCOL,
    project_root: Path = ROOT,
    require_frozen_ledger: bool = True,
    expected_rows: int | None = 1096,
) -> dict[str, Any]:
    """Return a deterministic inventory document without writing any files."""

    ledger_path = ledger_path.resolve()
    protocol_path = protocol_path.resolve()
    project_root = project_root.resolve()
    rows = read_csv(ledger_path)
    _validate_ledger(
        rows,
        ledger_path=ledger_path,
        require_frozen_ledger=require_frozen_ledger,
        expected_rows=expected_rows,
    )
    try:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InventoryError(f"cannot read canonical protocol {protocol_path}: {exc}") from exc

    protocol_id = str(protocol.get("canonical_protocol", "")).strip()
    if not protocol_id:
        raise InventoryError("canonical protocol id is empty")
    protocol_payload = extractor_protocol_payload(protocol)
    protocol_sha = sha256_json(protocol_payload)
    protocol_file_sha = sha256_file(protocol_path)
    allowed_prefixes = set(protocol_payload["prefixes"])
    canonical_windows = _as_int(
        protocol["expected_universe"]["canonical_windows"], "canonical windows"
    )
    target_scenes = _as_int(
        protocol["expected_universe"]["target_scene_units"], "target scenes"
    )

    raw_resolutions: list[dict[str, Any]] = []
    consensus_values: dict[tuple[str, int], set[tuple[str, int, int]]] = defaultdict(set)
    for row in rows:
        model = str(row["model"]).strip()
        ledger_layer = _as_int(row["layer"], f"{row['experiment_id']}:layer")
        prefix = _as_int(row["prefix"], f"{row['experiment_id']}:prefix")
        if prefix not in allowed_prefixes:
            raise InventoryError(
                f"{row['experiment_id']} uses prefix {prefix}, outside canonical {sorted(allowed_prefixes)}"
            )
        resolved, source, manifest_path = _read_manifest_resolution(row, project_root)
        if resolved is not None:
            if resolved[0] != model:
                raise InventoryError(
                    f"{row['experiment_id']} ledger model {model!r} conflicts with "
                    f"run-manifest VFM {resolved[0]!r}"
                )
            consensus_values[(model, ledger_layer)].add(resolved)
        raw_resolutions.append(
            {
                "row": row,
                "model": model,
                "ledger_layer": ledger_layer,
                "prefix": prefix,
                "resolved": resolved,
                "resolution_source": source,
                "run_manifest_path": manifest_path,
            }
        )

    consensus: dict[tuple[str, int], tuple[str, int, int]] = {}
    for key, values in consensus_values.items():
        if len(values) != 1:
            raise InventoryError(
                f"conflicting run-manifest feature identities for {key}: {sorted(values)}"
            )
        consensus[key] = next(iter(values))

    resolved_rows: list[dict[str, Any]] = []
    missing_manifest_rows: list[dict[str, str]] = []
    for item in raw_resolutions:
        resolved = item["resolved"]
        if resolved is None:
            key = (item["model"], item["ledger_layer"])
            if key not in consensus:
                raise InventoryError(
                    f"cannot resolve feature identity for {item['row']['experiment_id']}: "
                    f"no complete run manifest or sibling consensus for {key}"
                )
            resolved = consensus[key]
            item["resolution_source"] = "sibling_run_manifest_consensus"
            missing_manifest_rows.append(
                {
                    "experiment_id": item["row"]["experiment_id"],
                    "expected_run_manifest": item["run_manifest_path"],
                    "resolved_from": f"model={key[0]};ledger_layer={key[1]}",
                }
            )
        vfm, feature_layer, timestep = resolved
        if vfm not in MODEL_IDS:
            raise InventoryError(f"no frozen model id registered for VFM {vfm!r}")
        resolved_rows.append(
            {
                **item,
                "vfm": vfm,
                "feature_layer": feature_layer,
                "timestep": timestep,
                "model_id": MODEL_IDS[vfm],
            }
        )

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in resolved_rows:
        key = (
            protocol_id,
            item["vfm"],
            item["model_id"],
            item["feature_layer"],
            item["timestep"],
            item["prefix"],
        )
        grouped[key].append(item)

    specs: list[dict[str, Any]] = []
    for key, members in grouped.items():
        _, vfm, model_id, feature_layer, timestep, prefix = key
        source_rows = [member["row"] for member in members]
        experiment_ids = sorted(str(row["experiment_id"]) for row in source_rows)
        families = {str(row["probe_family"]) for row in source_rows}
        requires_target = bool(families & TARGET_PROBE_FAMILIES)
        extractor_entrypoint, extractor_status = _extractor_entrypoint(vfm, project_root)
        manifest_count = sum(
            member["resolution_source"] == "run_manifest" for member in members
        )
        spec = {
            "spec_id": _spec_id(
                protocol_id=protocol_id,
                vfm=vfm,
                model_id=model_id,
                layer=feature_layer,
                timestep=timestep,
                prefix=prefix,
            ),
            "canonical_protocol": protocol_id,
            "protocol_sha256": protocol_sha,
            "protocol_file_sha256": protocol_file_sha,
            "vfm": vfm,
            "model": vfm,
            "model_variant": _joined(row["model_variant"] for row in source_rows),
            "model_id": model_id,
            "model_id_source": MODEL_ID_SOURCES.get(
                vfm, "features/run_inscene15k.py:VFM_DEFAULTS"
            ),
            "ledger_layers": _joined(member["ledger_layer"] for member in members),
            "feature_layer": feature_layer,
            "timestep": timestep,
            "timestep_semantics": TIMESTEP_SEMANTICS[vfm],
            "prefix": prefix,
            "feature_filename": _feature_filename(vfm, feature_layer, timestep),
            "extractor_entrypoint": extractor_entrypoint,
            "extractor_status": extractor_status,
            "campaigns": _joined(row["campaign"] for row in source_rows),
            "probe_families": _joined(families),
            "tasks": _joined(row["probe_config"] for row in source_rows),
            "experiment_count": len(members),
            "source_experiment_ids_sha256": sha256_json(experiment_ids),
            "requires_streaming": "true",
            "requires_target": _bool_text(requires_target),
            "canonical_windows": canonical_windows,
            "target_scenes": target_scenes if requires_target else 0,
            "run_manifest_count": manifest_count,
            "missing_run_manifest_count": len(members) - manifest_count,
            "resolution_sources": _joined(
                member["resolution_source"] for member in members
            ),
            "status": "REQUIRED",
            "action": "PENDING_CACHE_AUDIT",
            # JSON retains complete lineage; CSV intentionally uses only its
            # digest so it remains practical to inspect in a spreadsheet.
            "source_experiment_ids": experiment_ids,
            "run_manifest_paths": sorted(
                {
                    member["run_manifest_path"]
                    for member in members
                    if member["run_manifest_path"]
                }
            ),
        }
        specs.append(spec)

    specs.sort(
        key=lambda row: (
            str(row["vfm"]),
            int(row["timestep"]),
            int(row["feature_layer"]),
            int(row["prefix"]),
        )
    )
    if len({row["spec_id"] for row in specs}) != len(specs):
        raise InventoryError("spec_id collision")

    stream_groups = {
        (row["vfm"], row["model_id"], row["feature_layer"], row["timestep"])
        for row in specs
    }
    target_groups = {
        (row["vfm"], row["model_id"], row["feature_layer"], row["timestep"])
        for row in specs
        if row["requires_target"] == "true"
    }
    aliases = sorted(
        {
            (row["model"], int(row["ledger_layer"]), int(row["feature_layer"]))
            for row in resolved_rows
            if int(row["ledger_layer"]) != int(row["feature_layer"])
        }
    )
    spec_identity_payload = [
        {field: row[field] for field in CSV_FIELDS if field not in {"status", "action"}}
        for row in specs
    ]
    inventory_sha = sha256_json(spec_identity_payload)
    inventory_version = f"windowfix_v2_{inventory_sha[:12]}"
    counts_by_vfm = Counter(str(row["vfm"]) for row in specs)
    target_specs_by_vfm = Counter(
        str(row["vfm"]) for row in specs if row["requires_target"] == "true"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "inventory_version": inventory_version,
        "inventory_sha256": inventory_sha,
        "scope": "canonical_v2_features_for_frozen_1096",
        "source": {
            "ledger": str(ledger_path),
            "ledger_sha256": sha256_file(ledger_path),
            "ledger_rows": len(rows),
            "protocol": str(protocol_path),
            "protocol_file_sha256": protocol_file_sha,
        },
        "canonical_protocol": {
            **protocol_payload,
            "protocol_sha256": protocol_sha,
        },
        "counts": {
            "experiments": len(rows),
            "feature_specs": len(specs),
            "streaming_specs": len(specs),
            "target_required_specs": sum(
                row["requires_target"] == "true" for row in specs
            ),
            "streaming_extraction_groups": len(stream_groups),
            "target_extraction_groups": len(target_groups),
            "run_manifest_resolved_experiments": len(rows) - len(missing_manifest_rows),
            "sibling_consensus_resolved_experiments": len(missing_manifest_rows),
            "feature_specs_by_vfm": dict(sorted(counts_by_vfm.items())),
            "target_required_specs_by_vfm": dict(sorted(target_specs_by_vfm.items())),
        },
        "physical_layer_aliases": [
            {
                "model": model,
                "ledger_layer": ledger_layer,
                "feature_layer": feature_layer,
            }
            for model, ledger_layer, feature_layer in aliases
        ],
        "missing_run_manifests": missing_manifest_rows,
        "policy": {
            "dedup_key": [
                "canonical_protocol",
                "vfm",
                "model_id",
                "feature_layer",
                "timestep",
                "prefix",
            ],
            "cache_decision": "not_performed",
            "initial_action": "PENDING_CACHE_AUDIT",
            "note": (
                "A legacy cache may be reused only after a separate raw-frame, "
                "model/layer/timestep, tensor, checksum, and provenance audit."
            ),
        },
        "specs": specs,
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_inventory(document: Mapping[str, Any], output_dir: Path) -> tuple[Path, Path]:
    version = str(document["inventory_version"])
    csv_path = output_dir / f"canonical_v2_feature_specs.{version}.csv"
    json_path = output_dir / f"canonical_v2_feature_specs.{version}.json"

    # Produce CSV bytes in memory so publication of each artifact is atomic.
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for spec in document["specs"]:
        writer.writerow({field: spec.get(field, "") for field in CSV_FIELDS})
    _atomic_write(csv_path, buffer.getvalue().encode("utf-8"))
    _atomic_write(
        json_path,
        (json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
            "utf-8"
        ),
    )
    return csv_path, json_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--allow-unfrozen-ledger",
        action="store_true",
        help="Allow a noncanonical ledger hash/count (intended only for development fixtures).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    document = build_inventory(
        ledger_path=args.ledger,
        protocol_path=args.protocol,
        project_root=args.project_root,
        require_frozen_ledger=not args.allow_unfrozen_ledger,
        expected_rows=None if args.allow_unfrozen_ledger else 1096,
    )
    csv_path, json_path = write_inventory(document, args.output_dir.resolve())
    counts = document["counts"]
    print(f"inventory_version={document['inventory_version']}")
    print(f"experiments={counts['experiments']}")
    print(f"feature_specs={counts['feature_specs']}")
    print(f"streaming_extraction_groups={counts['streaming_extraction_groups']}")
    print(f"target_extraction_groups={counts['target_extraction_groups']}")
    print(f"csv={csv_path}")
    print(f"json={json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

