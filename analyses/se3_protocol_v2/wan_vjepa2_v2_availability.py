#!/usr/bin/env python3
"""Live object-level availability audit restricted to the frozen Wan cells."""

from __future__ import annotations

from collections import defaultdict

from analyses.se3_protocol_v2.wan_vjepa2_v2_data import RESULT, atomic_json, read_jsonl
from vidfm3d.utils import aoss_fs


def main() -> None:
    routes = [row for row in read_jsonl(RESULT / "data/route_plan.jsonl") if row["model"] == "wan"]
    existence: dict[str, bool] = {}

    def present(uri: str) -> bool:
        if uri not in existence:
            existence[uri] = bool(aoss_fs.exists(uri))
        return existence[uri]

    by_cell: dict[str, list[dict]] = defaultdict(list)
    for row in routes:
        by_cell[row["cell"]].append(row)
    cells = {}
    for cell, rows in sorted(by_cell.items()):
        stream_pass = 0
        target_role_pass = 0
        stream_components = defaultdict(int)
        target_components = defaultdict(lambda: defaultdict(int))
        missing_streaming = []
        missing_targets = []
        for row in rows:
            stream_uri = row["streaming"]["uri"]
            stream_payload_ok = present(stream_uri)
            stream_sidecar_ok = present(stream_uri + ".manifest.json")
            stream_components["payload_available"] += int(stream_payload_ok)
            stream_components["sidecar_available"] += int(stream_sidecar_ok)
            stream_ok = stream_payload_ok and stream_sidecar_ok
            stream_pass += int(stream_ok)
            if not stream_ok and len(missing_streaming) < 10:
                missing_streaming.append({
                    "sample_id": row["sample_id"], "scene_id": row["scene_id"],
                    "role": "streaming_context_A", "uri": stream_uri,
                    "payload_available": stream_payload_ok, "sidecar_available": stream_sidecar_ok,
                })
            for role, target in row["targets"].items():
                payload_ok = present(target["uri"])
                sidecar_ok = present(target["uri"] + ".manifest.json")
                index_ok = present(target["index_uri"])
                target_components[role]["payload_available"] += int(payload_ok)
                target_components[role]["sidecar_available"] += int(sidecar_ok)
                target_components[role]["index_available"] += int(index_ok)
                target_ok = payload_ok and sidecar_ok and index_ok
                target_components[role]["complete_available"] += int(target_ok)
                target_role_pass += int(target_ok)
                if not target_ok and len(missing_targets) < 30:
                    missing_targets.append({
                        "sample_id": row["sample_id"], "scene_id": row["scene_id"],
                        "role": f"target_isolated_{role}", "target_position": target["target_position"],
                        "uri": target["uri"], "index_uri": target["index_uri"],
                        "payload_available": payload_ok, "sidecar_available": sidecar_ok,
                        "index_available": index_ok,
                    })
        target_component_summary = {}
        for role in sorted(target_components):
            counts = dict(target_components[role])
            counts.update({
                "required": len(rows),
                "payload_missing": len(rows) - counts["payload_available"],
                "sidecar_missing": len(rows) - counts["sidecar_available"],
                "index_missing": len(rows) - counts["index_available"],
                "complete_missing": len(rows) - counts["complete_available"],
            })
            target_component_summary[role] = counts
        cells[cell] = {
            "streaming_samples_required": len(rows), "streaming_samples_available": stream_pass,
            "streaming_samples_missing": len(rows) - stream_pass,
            "streaming_components": {
                "payload_available": stream_components["payload_available"],
                "payload_missing": len(rows) - stream_components["payload_available"],
                "sidecar_available": stream_components["sidecar_available"],
                "sidecar_missing": len(rows) - stream_components["sidecar_available"],
            },
            "target_roles_required": 3 * len(rows), "target_roles_available": target_role_pass,
            "target_roles_missing": 3 * len(rows) - target_role_pass,
            "target_components_by_role": target_component_summary,
            "runnable_on_immutable_common_set": stream_pass == len(rows) and target_role_pass == 3 * len(rows),
            "missing_streaming_examples": missing_streaming,
            "missing_target_examples": missing_targets,
        }
    atomic_json(RESULT / "inventory/wan_live_availability.json", {
        "scope": "frozen Wan cells and common p12 samples only",
        "cells": cells,
        "unique_object_paths_checked": len(existence),
        "objects_available": sum(existence.values()),
        "objects_missing": len(existence) - sum(existence.values()),
    })


if __name__ == "__main__":
    main()
