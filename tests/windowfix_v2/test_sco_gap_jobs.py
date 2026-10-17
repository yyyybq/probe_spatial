from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from features.canonical_gap_worklist import (
    canonical_json_sha,
    streaming_worklist_preprocess_identity,
)
from scripts.windowfix_v2.prepare_sco_gap_jobs import main as prepare_main
from scripts.windowfix_v2.run_canonical_gap_shard import (
    run_shard,
    validate_batch,
    validate_shard,
)


PROJECT = Path(__file__).resolve().parents[2]
PROTOCOL = PROJECT / "protocols/inscene_streaming_canonical_v2.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _worklist(
    path: Path,
    *,
    mode: str,
    source_sha: str,
    protocol: dict,
) -> dict:
    is_stream = mode == "streaming_prefix"
    entry = (
        {
            "scene": "scene-a",
            "window_id": 0,
            "prefix": 8,
            "layers": [5, 10],
            "reason": "RAW_ID_MISMATCH",
        }
        if is_stream
        else {
            "scene": "scene-a",
            "frame_index": 7,
            "layers": [5, 10],
            "reason": "CANONICAL_TARGET_RAW_ID_UNROUTED",
        }
    )
    value = {
        "schema_version": 1,
        "contract": "CANONICAL_V2_GAP_WORKLIST_V1",
        "canonical_protocol": "inscene_streaming_canonical_v2_obs28_h124",
        "protocol_sha256": protocol["canonical_protocol_sha256"],
        "protocol_file_sha256": _sha(PROTOCOL),
        "canonical_source_sha256": source_sha,
        "mode": mode,
        "vfm": "wan",
        "model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "timestep": 749,
        "feature_namespace": (
            "FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1"
            if is_stream
            else "FEAT_TARGET_CANONICAL_V2_GAP_DELTA_V1"
        ),
        "output_layers": [5, 10],
        "source": "scannetpp",
        "prompt_contract": {"mode": "exact", "value": "."},
        "preprocess_identity": (
            streaming_worklist_preprocess_identity("wan", (480, 832), 81)
            if is_stream
            else {"resize": [480, 832], "num_frames": 81}
        ),
        "entries": [entry],
        "counts": {"entries": 1, "layer_items": 2, "scenes": 1},
    }
    value["content_sha256"] = canonical_json_sha(value)
    _json(path, value)
    return value


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "canonical.jsonl"
    source.write_text("frozen fixture\n")
    source_sha = _sha(source)
    protocol = json.loads(PROTOCOL.read_text())
    worklists = []
    for mode in ("streaming_prefix", "target_isolated"):
        path = tmp_path / f"{mode}.json"
        value = _worklist(path, mode=mode, source_sha=source_sha, protocol=protocol)
        worklists.append(
            {
                "path": str(path.resolve()),
                "file_sha256": _sha(path),
                "content_sha256": value["content_sha256"],
                "counts": value["counts"],
            }
        )
    audit = {
        "schema_version": 1,
        "audit_type": "fixture_route_audit",
        "status": "INCOMPLETE",
        "protocol_file_sha256": _sha(PROTOCOL),
        "canonical_source": {"path": str(source), "sha256": source_sha},
        "counts": {"hard_errors": 0},
        "hard_errors": [],
        "outputs": {"gap_worklists": worklists},
    }
    audit["content_sha256"] = canonical_json_sha(audit)
    audit_path = tmp_path / "route_audit.json"
    _json(audit_path, audit)
    return source, audit_path


def test_renderer_schedules_only_audit_bound_records_and_serializes_vfm(tmp_path: Path) -> None:
    source, audit = _fixture(tmp_path)
    output = tmp_path / "plan"
    assert prepare_main(
        [
            "--source-audit", str(audit),
            "--protocol", str(PROTOCOL),
            "--canonical-source", str(source),
            "--output-dir", str(output),
            "--dataset-root", str(tmp_path / "dataset"),
            "--project-root", str(PROJECT),
            "--n-gpus", "2",
        ]
    ) == 0

    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "READY_FOR_REVIEW"
    assert summary["submission_performed"] is False
    assert summary["counts"]["ready_shards"] == 2
    assert summary["counts"]["ready_sco_batches"] == 1
    assert summary["counts"]["ready_layer_items"] == 4
    jobs = list((output / "jobspecs").glob("*.json"))
    assert len(jobs) == 1
    command_text = (output / "sco_create_commands.txt").read_text()
    batch_scripts = list((output / "scripts").glob("*.sh"))
    assert len(batch_scripts) == 1
    assert str(batch_scripts[0]) in command_text
    assert "run_canonical_gap_shard.py" in batch_scripts[0].read_text()

    batch, shards = validate_batch(next((output / "batches").glob("*.json")))
    assert batch["vfm"] == "wan"
    assert [item["mode"] for item in shards] == ["streaming_prefix", "target_isolated"]
    assert {
        "aoss_sync_loop",
        "aoss_gap_adapter",
    }.issubset(shards[0]["source_bindings"])
    preview = run_shard(shards[0], execute=False)
    assert len(preview["commands"]) == 2
    assert all("--canonical-gap-worklist-sha256" in item["argv"] for item in preview["commands"])


def test_renderer_refuses_audit_without_positive_gap_bindings(tmp_path: Path) -> None:
    source, audit_path = _fixture(tmp_path)
    audit = json.loads(audit_path.read_text())
    audit["outputs"]["gap_worklists"] = []
    audit["content_sha256"] = canonical_json_sha(
        {key: value for key, value in audit.items() if key != "content_sha256"}
    )
    _json(audit_path, audit)
    with pytest.raises(ValueError, match="zero gap worklists"):
        prepare_main(
            [
                "--source-audit", str(audit_path),
                "--protocol", str(PROTOCOL),
                "--canonical-source", str(source),
                "--output-dir", str(tmp_path / "bad-plan"),
                "--project-root", str(PROJECT),
            ]
        )


def test_renderer_accepts_only_bound_definite_gaps_from_pending_audit(
    tmp_path: Path,
) -> None:
    source, audit_path = _fixture(tmp_path)
    audit = json.loads(audit_path.read_text())
    audit["status"] = "PENDING"
    audit["counts"]["pending_rows"] = 999
    audit["content_sha256"] = canonical_json_sha(
        {key: value for key, value in audit.items() if key != "content_sha256"}
    )
    _json(audit_path, audit)
    output = tmp_path / "pending-plan"
    assert prepare_main(
        [
            "--source-audit", str(audit_path),
            "--protocol", str(PROTOCOL),
            "--canonical-source", str(source),
            "--output-dir", str(output),
            "--dataset-root", str(tmp_path / "dataset"),
            "--project-root", str(PROJECT),
        ]
    ) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["counts"]["ready_layer_items"] == 4
    assert summary["submission_performed"] is False


def test_renderer_can_render_a_positive_vfm_subset(tmp_path: Path) -> None:
    source, audit = _fixture(tmp_path)
    output = tmp_path / "subset-plan"
    assert prepare_main(
        [
            "--source-audit", str(audit),
            "--protocol", str(PROTOCOL),
            "--canonical-source", str(source),
            "--output-dir", str(output),
            "--project-root", str(PROJECT),
            "--include-vfm", "wan",
        ]
    ) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["counts"]["audited_worklists"] == 2
    assert summary["counts"]["selected_audited_worklists"] == 2
    assert "VFM subset: wan" in summary["notes"]


def test_runner_rejects_worklist_after_shard_render(tmp_path: Path) -> None:
    source, audit = _fixture(tmp_path)
    output = tmp_path / "plan"
    prepare_main(
        [
            "--source-audit", str(audit),
            "--protocol", str(PROTOCOL),
            "--canonical-source", str(source),
            "--output-dir", str(output),
            "--dataset-root", str(tmp_path / "dataset"),
            "--project-root", str(PROJECT),
        ]
    )
    shard_path = next((output / "shards").glob("*.json"))
    shard = json.loads(shard_path.read_text())
    Path(shard["worklist_path"]).write_text("{}\n")
    with pytest.raises(SystemExit, match="worklist SHA-256 mismatch"):
        validate_shard(shard_path)


def test_runner_rejects_symlink_shard_before_resolve(tmp_path: Path) -> None:
    source, audit = _fixture(tmp_path)
    output = tmp_path / "plan"
    prepare_main(
        [
            "--source-audit", str(audit),
            "--protocol", str(PROTOCOL),
            "--canonical-source", str(source),
            "--output-dir", str(output),
            "--dataset-root", str(tmp_path / "dataset"),
            "--project-root", str(PROJECT),
        ]
    )
    shard_path = next((output / "shards").glob("*.json"))
    link = tmp_path / "shard-link.json"
    link.symlink_to(shard_path)
    with pytest.raises(SystemExit, match="gap shard must be a regular non-symlink"):
        validate_shard(link)


def test_runner_rejects_symlink_worklist_before_resolve(tmp_path: Path) -> None:
    source, audit = _fixture(tmp_path)
    output = tmp_path / "plan"
    prepare_main(
        [
            "--source-audit", str(audit),
            "--protocol", str(PROTOCOL),
            "--canonical-source", str(source),
            "--output-dir", str(output),
            "--dataset-root", str(tmp_path / "dataset"),
            "--project-root", str(PROJECT),
        ]
    )
    shard_path = next((output / "shards").glob("*.json"))
    shard = json.loads(shard_path.read_text())
    original = Path(shard["worklist_path"])
    link = tmp_path / "worklist-link.json"
    link.symlink_to(original)
    shard["worklist_path"] = str(link)
    shard["content_sha256"] = canonical_json_sha(
        {key: value for key, value in shard.items() if key != "content_sha256"}
    )
    shard_path.write_text(json.dumps(shard, sort_keys=True) + "\n")
    with pytest.raises(SystemExit, match="gap worklist must be a regular non-symlink"):
        validate_shard(shard_path)


def test_runner_rejects_tampered_backend_dependency_binding(tmp_path: Path) -> None:
    source, audit = _fixture(tmp_path)
    output = tmp_path / "plan"
    prepare_main(
        [
            "--source-audit", str(audit),
            "--protocol", str(PROTOCOL),
            "--canonical-source", str(source),
            "--output-dir", str(output),
            "--dataset-root", str(tmp_path / "dataset"),
            "--project-root", str(PROJECT),
        ]
    )
    shard_path = next((output / "shards").glob("*.json"))
    shard = json.loads(shard_path.read_text())
    dependency = shard["source_bindings"]["backend_dependencies"]["features/noise_seed.py"]
    dependency["sha256"] = "0" * 64
    shard["content_sha256"] = canonical_json_sha(
        {key: value for key, value in shard.items() if key != "content_sha256"}
    )
    shard_path.write_text(json.dumps(shard, sort_keys=True) + "\n")
    with pytest.raises(SystemExit, match="backend dependency features/noise_seed.py SHA-256"):
        validate_shard(shard_path)
