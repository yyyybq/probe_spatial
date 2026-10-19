#!/usr/bin/env python3
"""Build immutable CogVideoX/LingBot target recovery queues from PASS receipts.

Only frozen V2 target entries without a validated remote-completion receipt are
emitted.  LingBot L30 is deliberately excluded: its existing jobs are paused,
not unstarted.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/main_v2_probe_v2_816_20260917"
DATA = Path("/mnt/umm/users/yinbaiqiao/InsScene-15K")
OUT = BASE / "raised_priority_target_recovery_20260930_r1"
GAPS = BASE / "targeted_reextract_worklists_20260922_r2/gap_worklists"
SOURCE_SHA = "ebc496eb0587cda372ccf29a7c5244ef454096151f73c8eed9511967ded1902d"
PROTOCOL_SHA = "215c720c8537f4dfae6243bedcc7339fbb7e3ba8a6ca2dcaf04f49cd4b738e4f"


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scene_indices() -> dict[str, int]:
    result: dict[str, int] = {}
    for line in (ROOT / "manifests/c_retrain_20260722/inscene_c_h124_prefix24.jsonl").read_text().splitlines():
        scene = str(json.loads(line)["scene_id"])
        result.setdefault(scene, len(result))
    if len(result) != 681:
        raise RuntimeError(f"expected 681 scenes, got {len(result)}")
    return result


def remote_target_keys(model: str, allowed_layers: set[int]) -> set[tuple[int, str, int]]:
    root = DATA / "FEAT_TARGET_CANONICAL_V2_GAP_DELTA_V1/.windowfix_v2_gap_logs" / model
    receipt_paths: set[Path] = set()
    for local in root.glob("*/*/receipt.json"):
        try:
            receipt_paths.add(Path(json.loads(local.read_text())["remote_completion_receipt"]))
        except (OSError, ValueError, KeyError):
            continue
    result: set[tuple[int, str, int]] = set()
    for receipt in sorted(receipt_paths):
        try:
            raw = receipt.read_bytes(); value = json.loads(raw)
            unsigned = dict(value)
            if unsigned.pop("content_sha256") != canonical(unsigned):
                continue
            lineage = value["lineage"]
            if (lineage.get("vfm") != model or lineage.get("canonical_source_sha256") != SOURCE_SHA
                    or lineage.get("protocol_sha256") != PROTOCOL_SHA or value.get("status") != "PASS_REMOTE_VERIFIED"):
                continue
            for obj in value["objects"].values():
                identity = obj["canonical_unit_identity"]
                if canonical(identity) != obj["canonical_unit_identity_sha256"]:
                    continue
                if (identity.get("kind") == "target_delta_layer" and identity.get("payload_role") == "feature"
                        and int(identity.get("layer", -1)) in allowed_layers):
                    result.update((int(identity["layer"]), str(identity["scene"]), int(frame))
                                  for frame in identity["target_indices"])
        except (OSError, ValueError, KeyError, AssertionError, TypeError):
            continue
    return result


def rows_for(model: str, layers: list[int], parent_path: Path, index: dict[str, int]) -> tuple[list[dict], dict]:
    parent = json.loads(parent_path.read_text())
    reference = {"path": str(parent_path.resolve()), "file_sha256": file_sha(parent_path),
                 "content_sha256": parent["content_sha256"]}
    confirmed = remote_target_keys(model, set(layers))
    rows: list[dict] = []
    missing_by_layer: dict[int, int] = defaultdict(int)
    for layer in layers:
        for entry in parent["entries"]:
            scene, frame = str(entry["scene"]), int(entry["frame_index"])
            if (layer, scene, frame) in confirmed:
                continue
            copied = {key: value for key, value in entry.items() if key != "layers"}
            copied["layers"] = [layer]
            row = {"schema": "PROBE528_ATOMIC_FEATURE_TASK_V1", "model": model,
                   "mode": "target_isolated", "layer": layer, "timestep": int(parent["timestep"]),
                   "scene_id": scene, "scene_index": index[scene], "prefix": None,
                   "entry_count": 1, "layer_items": 1, "entries": [copied],
                   "parent_worklist": reference}
            row["unique_key"] = canonical({key: row[key] for key in ("model", "mode", "layer", "timestep", "scene_id", "entries", "parent_worklist")})
            rows.append(row); missing_by_layer[layer] += 1
    return rows, {"remote_confirmed_in_frozen_scope": {str(layer): 32350 - missing_by_layer[layer] for layer in layers},
                  "emitted_missing": dict(missing_by_layer)}


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite immutable output: {OUT}")
    OUT.mkdir(parents=True)
    index = scene_indices()
    definitions = [
        ("cogvideox", [0, 5, 10, 30, 35, 40], GAPS / "cogvideox-t749-target_isolated-eaa19f1aba6a.json"),
        ("lingbot", [10, 20, 40, 47], GAPS / "lingbot-t250-target_isolated-dddc629938e8.json"),
    ]
    rows, evidence = [], {}
    for model, layers, path in definitions:
        emitted, summary = rows_for(model, layers, path, index)
        rows.extend(emitted); evidence[model] = summary
    queue = OUT / "target_atomic_queue.jsonl"
    with queue.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    qsha = file_sha(queue)
    templates = [
        BASE / "atomic_feature_workers_four_models_20260923_r3/shards/p528-cogvideox-target-00.json",
        BASE / "atomic_feature_workers_lingbot_20260923_r2/shards/p528-lingbot-target-00.json",
    ]
    command = ["/usr/bin/python3", str(ROOT / "scripts/build_probe528_atomic_worker_plan.py"),
               "--queue", str(queue), "--queue-sha256", qsha, "--output", str(OUT / "workers")]
    for template in templates:
        command.extend(["--template", str(template)])
    subprocess.run(command, check=True, cwd=ROOT)
    report = {"status": "READY_NOT_SUBMITTED", "queue": str(queue), "queue_sha256": qsha,
              "atomic_tasks": len(rows), "evidence": evidence,
              "excluded": {"lingbot_l30": "existing paused jobs pt-7jmv0yrc and pt-hitrv7fv"},
              "workers_plan": str(OUT / "workers/manifest.json")}
    (OUT / "submission_plan.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
