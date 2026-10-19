#!/usr/bin/env python3
"""Read-only audit for the five requested Active Spatial checkpoint labels.

The tool never resolves a missing checkpoint to a similarly named directory.
That invariant is intentional: a comparison with an accidental replacement is
less useful than a blocked comparison.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .io_utils import sha256_file, stable_hash, write_json
from .protocol import freeze_protocol


REQUESTED = (
    "qwen_pretrain_baseline",
    "v46_step200",
    "v50_step150",
    "cambrian_pretrain_baseline",
    "c8_step300",
)


def _registry_entries(path: Path) -> dict[str, dict[str, str]]:
    """Parse the small checkpoint block without depending on PyYAML."""
    entries: dict[str, dict[str, str]] = {}
    current: str | None = None
    in_checkpoints = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw == "checkpoints:":
            in_checkpoints = True
            continue
        if in_checkpoints and raw and not raw.startswith(" "):
            break
        match = re.match(r"^  ([A-Za-z0-9_]+):\s*$", raw)
        if match:
            current = match.group(1)
            entries[current] = {}
            continue
        match = re.match(r"^    ([A-Za-z0-9_]+):\s*(.*)$", raw)
        if current and match:
            entries[current][match.group(1)] = match.group(2).strip().strip('"').strip("'")
    return entries


def _resolve(base: Path, value: str) -> Path:
    return Path(value) if Path(value).is_absolute() else base / value


def _config_summary(path: Path) -> dict[str, Any]:
    config_path = path / "config.json"
    if not config_path.is_file():
        return {"status": "UNCERTAIN", "reason": "config.json is absent"}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    vision = config.get("vision_config") if isinstance(config.get("vision_config"), dict) else {}
    return {
        "status": "VERIFIED",
        "config_sha256": sha256_file(config_path),
        "architecture": config.get("architectures"),
        "model_type": config.get("model_type"),
        "hidden_size": text.get("hidden_size"),
        "num_hidden_layers": text.get("num_hidden_layers"),
        "vision_hidden_size": vision.get("hidden_size"),
        "vision_depth": vision.get("depth") or vision.get("num_hidden_layers"),
    }


def _checkpoint_record(name: str, entry: dict[str, str], base: Path) -> dict[str, Any]:
    raw_path = entry.get("path", "")
    path = _resolve(base, raw_path)
    record: dict[str, Any] = {
        "name": name,
        "registered_path": raw_path,
        "resolved_path": str(path),
        "registry_model_type": entry.get("model_type", "UNSPECIFIED"),
        "registry_description": entry.get("desc", ""),
    }
    if not path.exists():
        record.update({"status": "BLOCKED", "reason": "registered checkpoint path does not exist on this host"})
        return record
    record["config"] = _config_summary(path)
    if record["config"]["status"] != "VERIFIED":
        record.update({"status": "BLOCKED", "reason": record["config"]["reason"]})
        return record
    files = [path / "model.safetensors.index.json", path / "model.safetensors", path / "pytorch_model.bin.index.json"]
    weight_index = next((candidate for candidate in files if candidate.is_file()), None)
    record["weight_identity"] = {
        "index_or_weight": str(weight_index) if weight_index else None,
        "sha256": sha256_file(weight_index) if weight_index else None,
    }
    record["status"] = "VERIFIED"
    return record


def _script_evidence(vagen_root: Path) -> dict[str, Any]:
    paths = {
        "v46": vagen_root / "examples/train/active_spatial/experiments/v46_7b_nodelta_w3_rerun.sh",
        "c8": vagen_root / "examples/train/active_spatial/experiments/c8_fwdfirst_rewscale_lfp_server.sh",
    }
    evidence: dict[str, Any] = {}
    for name, path in paths.items():
        if not path.is_file():
            evidence[name] = {"status": "BLOCKED", "reason": f"missing training script: {path}"}
            continue
        text = path.read_text(encoding="utf-8")
        model = re.search(r'^MODEL_PATH="?([^"\n]+)', text, flags=re.MULTILINE)
        experiment = re.search(r'^EXPERIMENT_NAME="?([^"\n]+)', text, flags=re.MULTILINE)
        evidence[name] = {
            "status": "VERIFIED",
            "script": str(path),
            "model_path_declared": model.group(1) if model else None,
            "experiment_name": experiment.group(1) if experiment else None,
            "uses_external_cambrian_adapter": "cambrian_register" in text,
        }
    return evidence


def _validity_evidence(vagen_root: Path) -> dict[str, Any]:
    path = vagen_root / "docs/diagnosis/ACTIVE_SPATIAL_HISTORICAL_VALIDITY.md"
    if not path.is_file():
        return {"status": "BLOCKED", "reason": "historical validity document missing"}
    text = path.read_text(encoding="utf-8")
    return {
        "status": "VERIFIED",
        "source": str(path),
        "v46_affected_mentioned": "Qwen v46 baselines" in text and "AFFECTED" in text,
        "c8_invalidated_mentioned": "Cambrian C8/B5" in text and "INVALIDATED" in text,
        "scope_note": "This document labels historical training-method evidence; it does not make a fixed checkpoint unloadable.",
    }


def _pair_status(checkpoints: dict[str, dict[str, Any]], baseline: str, trained: str) -> dict[str, Any]:
    left, right = checkpoints[baseline], checkpoints[trained]
    if left["status"] != "VERIFIED" or right["status"] != "VERIFIED":
        return {"status": "BLOCKED", "reason": "one or both registered paths are unavailable"}
    lc, rc = left["config"], right["config"]
    keys = ("model_type", "hidden_size", "num_hidden_layers")
    mismatch = {key: [lc.get(key), rc.get(key)] for key in keys if lc.get(key) != rc.get(key)}
    if mismatch:
        return {"status": "BLOCKED", "reason": "architecture/size mismatch", "mismatch": mismatch}
    return {"status": "VERIFIED", "reason": "registered configs agree on type, hidden size, and layer count"}


def _module_delta_status(pairs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Do not claim module deltas until both exact weight files are present.

    A tensor-by-tensor comparison belongs on the GPU host because the files can
    be large.  This host-level audit records whether it is legal to run it.
    """
    blocked = [name for name, pair in pairs.items() if pair["status"] != "VERIFIED"]
    if blocked:
        return {"status": "BLOCKED", "reason": "exact paired checkpoints unavailable", "blocked_pairs": blocked}
    return {"status": "NOT_RUN", "reason": "eligible; run tensor module-delta audit on the checkpoint host before canary"}


def markdown(payload: dict[str, Any]) -> str:
    lines = ["# Active Spatial Representation Probe Audit", "", "This is a read-only host audit. `BLOCKED` means the requested registered identity cannot be loaded from this host; it does not authorize substitution.", "", "## Checkpoints", "", "| Name | Status | Model type | Resolved path |", "| --- | --- | --- | --- |"]
    for row in payload["checkpoints"].values():
        typ = row.get("config", {}).get("model_type", row.get("registry_model_type", ""))
        lines.append(f"| {row['name']} | {row['status']} | {typ} | `{row['resolved_path']}` |")
    lines += ["", "## Pair eligibility", ""]
    for name, result in payload["pairs"].items():
        lines.append(f"- **{name}: {result['status']}** — {result['reason']}")
    lines += ["", "## Validity evidence", "", f"- Historical-document status: {payload['historical_validity']['status']}.", "- v46/c8 historical labels are retained as provenance limits, not treated as a replacement for checking the exact checkpoint and forward path.", "", "## Result", "", "No feature extraction or model training was run by this audit."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vagen-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    registry = args.vagen_root / "scripts/easi_registry.yaml"
    if not registry.is_file():
        raise SystemExit(f"BLOCKED: registry not found: {registry}")
    entries = _registry_entries(registry)
    missing = [name for name in REQUESTED if name not in entries]
    if missing:
        raise SystemExit(f"BLOCKED: requested registry keys missing: {missing}")
    checkpoints = {name: _checkpoint_record(name, entries[name], args.vagen_root) for name in REQUESTED}
    pairs = {
        "qwen_pretrain_baseline__v46_step200": _pair_status(checkpoints, "qwen_pretrain_baseline", "v46_step200"),
        "cambrian_pretrain_baseline__c8_step300": _pair_status(checkpoints, "cambrian_pretrain_baseline", "c8_step300"),
    }
    payload = {
        "audit_version": "active_repr_checkpoint_audit_v1",
        "vagen_root": str(args.vagen_root),
        "registry": str(registry),
        "checkpoints": checkpoints,
        "pairs": pairs,
        "module_delta_audit": _module_delta_status(pairs),
        "training_scripts": _script_evidence(args.vagen_root),
        "historical_validity": _validity_evidence(args.vagen_root),
    }
    payload["audit_hash"] = stable_hash(payload)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "checkpoints.json", payload)
    (args.output_dir / "AUDIT.md").write_text(markdown(payload), encoding="utf-8")
    freeze_protocol(args.output_dir, {"checkpoint_audit_hash": payload["audit_hash"]})
    print(json.dumps({"audit": str(args.output_dir / "AUDIT.md"), "checkpoints": str(args.output_dir / "checkpoints.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
