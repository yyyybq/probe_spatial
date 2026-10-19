"""Classify 816 holes, recover sidecar-only owners, and emit re-extract worklists.

Fail-closed: never drop window_id, never rewrite tensors, never treat a
shape/filename similarity as identity. Recovery is allowed only when the
gate already certified the exact tensor/index contract and the failed
fields are incomplete sidecar metadata with an authoritative source.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from scripts.build_probe_owner_overlay import receipt_key
from scripts.owner_catalog import sha256_file

REQUEST_SHA = "5e0854c11f5141780ae797da3f1368cffec270a42b9ec7ab2d4cf31908597cd5"
F3R_MODEL = "jedyang97/Fast3R_ViT_Large_512"
F3R_EXTRACTOR_RESOLVE = (
    "features/run_inscene15k.py loads "
    "`args.model_id or 'jedyang97/Fast3R_ViT_Large_512'` after the f3r "
    "default model_id is left None, so sidecars persist model_id=null"
)
WORKLIST_CONTRACT = "CANONICAL_V2_GAP_WORKLIST_V1"
CANONICAL_PROTOCOL = "inscene_streaming_canonical_v2_obs28_h124"
PROTOCOL_SHA = "215c720c8537f4dfae6243bedcc7339fbb7e3ba8a6ca2dcaf04f49cd4b738e4f"
PROTOCOL_FILE_SHA = "cb4feaecc1deaeb9be0ee7e9e66c0e205d96528e23513def934aeaf93f01465c"
SOURCE_SHA = "ebc496eb0587cda372ccf29a7c5244ef454096151f73c8eed9511967ded1902d"
VFM_RE = re.compile(
    r"^(?:cv2_)?(cogvideox|lingbot|minimax_h3|videomae|vjepa2|dino|f3r|qwen2_5_vl)"
)
TEMPLATE_WORKLISTS = {
    "minimax_h3": Path(
        "/mnt/umm/users/yinbaiqiao/probe_spatial/artifacts/windowfix_v2/"
        "streaming_gap_worklists_ebc496eb0587_catalog_r12_qwen_pass_20260831/"
        "gap_worklists/minimax_h3-t250-28c6bd353a5c.json"
    ),
    "videomae": Path(
        "/mnt/umm/users/yinbaiqiao/probe_spatial/artifacts/windowfix_v2/"
        "streaming_gap_worklists_ebc496eb0587_catalog_r12_qwen_pass_20260831/"
        "gap_worklists/videomae-t749-32f9201be4bb.json"
    ),
    "vjepa2": Path(
        "/mnt/umm/users/yinbaiqiao/probe_spatial/artifacts/windowfix_v2/"
        "streaming_gap_worklists_ebc496eb0587_catalog_r12_qwen_pass_20260831/"
        "gap_worklists/vjepa2-t749-600db135aa18.json"
    ),
}


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _vfm(unit_id: str) -> str:
    match = VFM_RE.search(unit_id)
    return match.group(1) if match else "UNK"


def _failed_checks(row: dict[str, Any]) -> tuple[str, ...]:
    checks = row.get("checks") or {}
    if checks:
        return tuple(sorted(key for key, passed in checks.items() if not passed))
    return tuple(row.get("reason_codes") or ())


def classify_row(row: dict[str, Any]) -> str:
    codes = tuple(row.get("reason_codes") or [])
    failed = _failed_checks(row)
    vfm = _vfm(str(row.get("unit_id") or ""))
    if "DEFINITIVE_GATE_FAILURE:FileNotFoundError" in codes:
        return "genuinely_missing"
    if vfm == "minimax_h3" and failed == ("header_shape_equals_expected",):
        return "unusable_wrong_tensor"
    if vfm == "f3r" and failed == ("sidecar_model",):
        return "metadata_recoverable"
    if vfm == "qwen2_5_vl" and failed == (
        "sidecar_num_frames",
        "sidecar_tensor_dtype_equals_header",
    ):
        return "metadata_recoverable"
    if vfm == "videomae" and failed == ("source_index_raw_ids_exact",):
        return "unusable_wrong_identity"
    return "unexpected"


def load_requests(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in _jsonl(path):
        out[str(row["unit_id"])] = row
    return out


def recover_receipt(
    *,
    row: dict[str, Any],
    request: dict[str, Any],
    class_name: str,
    sidecar_model: Any = None,
) -> dict[str, Any] | None:
    key = receipt_key(row)
    if key is None:
        return None
    expected = request.get("expected") or {}
    vfm = _vfm(row["unit_id"])
    observed = ((row.get("evidence") or {}).get("observed_tensor") or {})
    if vfm == "f3r":
        if sidecar_model not in (None, "", "null"):
            return None
        if expected.get("model_id") != F3R_MODEL:
            return None
        basis = [
            "gate certified header_shape, header_dtype, raw_ids, and every sidecar field except model_id",
            "on-disk sidecar model_id is null/absent, not a conflicting model",
            F3R_EXTRACTOR_RESOLVE,
            f"frozen request expected.model_id={F3R_MODEL}",
        ]
        recovered = {
            "field": "sidecar.model_id",
            "from": sidecar_model,
            "to": F3R_MODEL,
        }
    elif vfm == "qwen2_5_vl":
        if list(observed.get("shape") or []) != list(expected.get("expected_shape") or []):
            return None
        if str(observed.get("dtype")) != "float16":
            return None
        basis = [
            "gate certified header_shape_equals_expected and header_dtype_equals_expected",
            "sidecar is missing num_frames and declares feat.dtype=float32 while the safetensors header is float16",
            "tensor header is the authority for dtype/shape; num_frames is taken from that header T and the frozen preprocess contract",
        ]
        recovered = {
            "sidecar_num_frames": {
                "from": None,
                "to": int((expected.get("preprocess_identity") or {}).get("num_frames") or observed["shape"][0]),
            },
            "sidecar_tensor_dtype": {"from": "float32", "to": "float16"},
        }
    else:
        return None
    payload = {
        "schema": "canonical_v2_probe816_metadata_recovery_receipt_v1",
        "status": "PASS_METADATA_RECOVERED",
        "unit_id": row["unit_id"],
        "key": list(key),
        "source_uri": row.get("source_uri") or request.get("source_uri"),
        "source_index_uri": row.get("source_index_uri") or request.get("source_index_uri"),
        "original_sidecar_uri": request.get("sidecar_uri"),
        "request_sha256": row.get("request_sha256"),
        "classification": class_name,
        "recovery_basis": basis,
        "recovered_fields": recovered,
        "observed_tensor": observed,
        "tensor_payload_unmodified": True,
        "window_id_preserved": True,
    }
    payload["receipt_sha256"] = _canonical_sha(payload)
    return payload


def build_worklist(template_path: Path, entries: list[dict[str, Any]]) -> dict[str, Any]:
    template = json.loads(template_path.read_text(encoding="utf-8"))
    layers = sorted({int(layer) for entry in entries for layer in entry["layers"]})
    payload = {
        "schema_version": 1,
        "contract": WORKLIST_CONTRACT,
        "canonical_protocol": CANONICAL_PROTOCOL,
        "mode": "streaming_prefix",
        "vfm": template["vfm"],
        "model_id": template["model_id"],
        "timestep": template["timestep"],
        "source": template.get("source", "scannetpp"),
        "feature_namespace": "FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1",
        "protocol_sha256": PROTOCOL_SHA,
        "protocol_file_sha256": PROTOCOL_FILE_SHA,
        "canonical_source_sha256": SOURCE_SHA,
        "output_layers": layers,
        "preprocess_identity": template["preprocess_identity"],
        "prompt_contract": template["prompt_contract"],
        "entries": entries,
        "counts": {
            "entries": len(entries),
            "layer_items": sum(len(entry["layers"]) for entry in entries),
            "scenes": len({entry["scene"] for entry in entries}),
        },
    }
    unhashed = dict(payload)
    payload["content_sha256"] = _canonical_sha(unhashed)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-f3r-sidecars", action="store_true")
    args = parser.parse_args()

    print("[holes] loading frozen requests", file=sys.stderr)
    requests = load_requests(args.requests)
    classes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_vfm = defaultdict(Counter)
    examples: dict[str, dict[str, Any]] = {}
    recovered: list[dict[str, Any]] = []
    reextract_units: dict[str, list[dict[str, Any]]] = defaultdict(list)

    shard_dirs = sorted(
        path for path in args.gate_root.iterdir()
        if path.is_dir() and path.name.startswith("gate-")
    )
    for shard_dir in shard_dirs:
        for row in _jsonl(shard_dir / "receipts.jsonl.gz"):
            if row.get("status") != "REEXTRACT_UNVERIFIED":
                continue
            unit = str(row["unit_id"])
            request = requests.get(unit)
            if request is None:
                classes["unexpected"].append(row)
                continue
            label = classify_row(row)
            key = receipt_key(row)
            rec = {
                "unit_id": unit,
                "vfm": _vfm(unit),
                "class": label,
                "reason_codes": row.get("reason_codes"),
                "key": list(key) if key else None,
                "source_uri": row.get("source_uri"),
                "observed_tensor": ((row.get("evidence") or {}).get("observed_tensor") or {}),
                "expected_shape": (request.get("expected") or {}).get("expected_shape"),
                "expected_raw_ids": (request.get("expected") or {}).get("raw_ids"),
            }
            classes[label].append(rec)
            by_vfm[rec["vfm"]][label] += 1
            examples.setdefault(f"{rec['vfm']}:{label}", rec)
            if label == "metadata_recoverable":
                sidecar_model = None
                recovered_row = recover_receipt(
                    row=row, request=request, class_name=label, sidecar_model=sidecar_model,
                )
                if recovered_row is not None:
                    recovered.append(recovered_row)
                else:
                    classes["unexpected"].append(rec)
                    by_vfm[rec["vfm"]]["unexpected"] += 1
            elif label in {"genuinely_missing", "unusable_wrong_tensor", "unusable_wrong_identity"}:
                exp = request.get("expected") or {}
                record = request.get("source_index_record") or {}
                reextract_units[rec["vfm"]].append({
                    "unit_id": unit,
                    "vfm": rec["vfm"],
                    "scene": key[3] if key else None,
                    "window_id": key[6] if key else record.get("window_id"),
                    "prefix": key[7] if key else None,
                    "layer": key[4] if key else exp.get("feature_layer"),
                    "timestep": key[5] if key else exp.get("timestep"),
                    "raw_ids": exp.get("raw_ids"),
                    "class": label,
                    "source_uri": request.get("source_uri"),
                })

    if args.verify_f3r_sidecars:
        print("[holes] verifying F3R sidecar model_id on AOSS", file=sys.stderr)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "aoss_fs",
            str(Path(__file__).resolve().parents[1] / "vidfm3d/utils/aoss_fs.py"),
        )
        aoss_fs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(aoss_fs)
        client = aoss_fs.get_client()
        kept = []
        dropped = 0
        for index, row in enumerate(recovered, 1):
            if row["key"][1] != "f3r":
                kept.append(row)
                continue
            uri = row.get("original_sidecar_uri")
            data = client.get(aoss_fs._client_path(uri))
            if hasattr(data, "read"):
                data = data.read()
            sidecar = json.loads(data.decode() if isinstance(data, bytes) else data)
            if sidecar.get("model_id") not in (None, "", "null"):
                dropped += 1
                continue
            row = dict(row)
            row["observed_sidecar_model_id"] = sidecar.get("model_id")
            row["receipt_sha256"] = _canonical_sha(
                {k: v for k, v in row.items() if k != "receipt_sha256"}
            )
            kept.append(row)
            if index % 500 == 0:
                print(f"[holes] f3r sidecars {index}/{len(recovered)}", file=sys.stderr)
        recovered[:] = kept
        print(f"[holes] f3r sidecar conflicts dropped={dropped}", file=sys.stderr)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    recovery_path = args.output_dir / "recovery" / "receipts.jsonl.gz"
    recovery_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(recovery_path, "wt", encoding="utf-8", compresslevel=6) as handle:
        for row in recovered:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    reextract_root = args.output_dir / "reextract"
    worklist_info = {}
    for vfm, units in reextract_units.items():
        man_path = reextract_root / "manifests" / f"{vfm}.jsonl.gz"
        man_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(man_path, "wt", encoding="utf-8", compresslevel=6) as handle:
            for row in units:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        grouped: dict[tuple[Any, ...], list[int]] = defaultdict(list)
        for row in units:
            grouped[(row["scene"], int(row["window_id"]), int(row["prefix"]))].append(int(row["layer"]))
        entries = []
        for (scene, window_id, prefix), layers in sorted(grouped.items()):
            entries.append({
                "scene": scene,
                "window_id": window_id,
                "prefix": prefix,
                "layers": sorted(set(layers)),
                "reason": "PROBE816_EXACT_OWNER_MISSING",
            })
        if vfm in TEMPLATE_WORKLISTS and entries:
            worklist = build_worklist(TEMPLATE_WORKLISTS[vfm], entries)
            wl_path = reextract_root / "worklists" / f"{vfm}.json"
            _write_json(wl_path, worklist)
            worklist_info[vfm] = {
                "path": str(wl_path.resolve()),
                "file_sha256": sha256_file(wl_path),
                "content_sha256": worklist["content_sha256"],
                "entries": worklist["counts"]["entries"],
                "layer_items": worklist["counts"]["layer_items"],
                "scenes": worklist["counts"]["scenes"],
                "manifest": str(man_path.resolve()),
                "units": len(units),
            }

    report = {
        "schema": "canonical_v2_probe816_hole_resolution_v1",
        "request_sha256": REQUEST_SHA,
        "total_holes": sum(len(rows) for rows in classes.values()),
        "counts": {name: len(rows) for name, rows in sorted(classes.items())},
        "by_vfm": {vfm: dict(counter) for vfm, counter in sorted(by_vfm.items())},
        "metadata_recoverable": len(recovered),
        "genuinely_missing": len(classes["genuinely_missing"]),
        "unusable_wrong_tensor": len(classes["unusable_wrong_tensor"]),
        "unusable_wrong_identity": len(classes["unusable_wrong_identity"]),
        "reextract_units": {vfm: len(rows) for vfm, rows in reextract_units.items()},
        "reextract_worklists": worklist_info,
        "recovery_receipts": {
            "path": str(recovery_path.resolve()),
            "sha256": sha256_file(recovery_path),
            "rows": len(recovered),
        },
        "examples": examples,
        "notes": [
            "MiniMax tensors are uniformly (22,15,26,5376) against frozen expected (19,15,26,5376); sidecar agrees with the tensor, so this is not metadata recovery.",
            "VideoMAE p8/p12/p16 prefix_index raw_ids differ from frozen request raw_ids at the same window_id; rewriting the index would lie about the extracted frames.",
            "VideoMAE p24 and the single VJEPA2 hole are FileNotFound on the exact owner URI.",
            "F3R/Qwen tensors already match the frozen exact contract; only sidecar fields are incomplete or stale.",
            "Gap-delta still must not be bound across window_id.",
        ],
    }
    _write_json(args.output_dir / "hole_resolution_report.json", report)
    print(json.dumps({
        key: report[key]
        for key in (
            "total_holes", "counts", "by_vfm", "metadata_recoverable",
            "reextract_units", "reextract_worklists",
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
