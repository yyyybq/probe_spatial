from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.windowfix_v2.prepare_canonical_gap_lifecycle_smoke import (
    DEFAULT_STREAMING_SHARD,
    DEFAULT_TARGET_SHARD,
    main as render_main,
)
from scripts.windowfix_v2.run_canonical_gap_shard import run_shard, validate_batch, validate_shard


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_lifecycle_smoke_derives_exact_one_unit_from_ready_parent_plans(tmp_path: Path) -> None:
    """The smoke is a real bounded derivative, not a broad synthetic plan."""

    if not DEFAULT_STREAMING_SHARD.exists() or not DEFAULT_TARGET_SHARD.exists():
        pytest.skip("authoritative r17/r8 smoke parent plans are unavailable")
    output = tmp_path / "lifecycle-smoke"
    assert render_main(["--output-dir", str(output)]) == 0

    summary = json.loads((output / "summary.json").read_text())
    lock = json.loads((output / "PLAN.lock.json").read_text())
    assert summary["submission_performed"] is False
    assert summary["counts"] == {
        "physical_scenes": 1,
        "smoke_jobs": 2,
        "streaming_layer_items": 1,
        "target_materialization_rows": 0,
        "target_row_layer_items": 1,
    }
    assert summary["scene"] == "00777c41d4"
    assert summary["scene_range"] == {"start": 0, "end": 1}
    assert lock["summary_sha256"] == _sha(output / "summary.json")

    parent_stream = json.loads(DEFAULT_STREAMING_SHARD.read_text())
    parent_target = json.loads(DEFAULT_TARGET_SHARD.read_text())
    shards = sorted((output / "shards").glob("*.json"))
    assert len(shards) == 2
    by_mode = {json.loads(path.read_text())["mode"]: path for path in shards}
    assert set(by_mode) == {"streaming_prefix", "target_isolated"}
    for mode, parent in (("streaming_prefix", parent_stream), ("target_isolated", parent_target)):
        shard = validate_shard(by_mode[mode])
        assert shard["feature_namespace"] == parent["feature_namespace"]
        assert shard["feature_root"] == parent["feature_root"]
        assert shard["layers"] == [20]
        assert shard["counts"] == {"entries": 1, "layer_items": 1, "scenes": 1}
        assert (shard["scene_start"], shard["scene_end"], shard["n_gpus"]) == (0, 1, 1)
        assert shard["source_bindings"] == parent["source_bindings"]
        smoke = shard["lifecycle_smoke"]
        assert smoke["target_materialization"] == "FORBIDDEN"
        assert smoke["parent_plan"]["source_shard"]["sha256"] == _sha(
            DEFAULT_STREAMING_SHARD if mode == "streaming_prefix" else DEFAULT_TARGET_SHARD
        )
        preview = run_shard(shard, execute=False)
        assert len(preview["commands"]) == 1
        assert preview["commands"][0]["start"] == 0
        assert preview["commands"][0]["end"] == 1
        assert "--canonical-gap-worklist-sha256" in preview["commands"][0]["argv"]

    target_worklist = json.loads(
        Path(json.loads(by_mode["target_isolated"].read_text())["worklist_path"]).read_text()
    )
    assert target_worklist["entries"] == [
        {
            "frame_index": 48,
            "layers": [20],
            "reason": "canonical_target_raw_id_unrouted",
            "scene": "00777c41d4",
        }
    ]
    for batch_path in (output / "batches").glob("*.json"):
        _batch, bound_shards = validate_batch(batch_path)
        assert len(bound_shards) == 1
    dry_run = (output / "dry_run_commands.txt").read_text()
    assert all(
        "--execute" not in line
        for line in dry_run.splitlines()
        if line and not line.startswith("#")
    )
    assert "run_canonical_gap_shard.py --batch" in dry_run
    assert "--execute" in (output / "sco_create_commands.txt").read_text()
