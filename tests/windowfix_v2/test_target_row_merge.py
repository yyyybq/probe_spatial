from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from analyses.windowfix_v2.audit_target_row_routes import (
    CONTRACT,
    GAP_CONTRACT,
    GAP_TARGET_NAMESPACE,
    MIXED_TARGET_NAMESPACE,
    audit_plan,
    canonical_json_sha,
    canonical_target_unions,
    build_gap_worklists,
    write_outputs,
)
from analyses.windowfix_v2.augment_target_merge_plan import AugmentError, augment_plan
from analyses.windowfix_v2 import target_row_source
from scripts.windowfix_v2.materialize_target_row_merge import MergeError, materialize


MODEL = "model/target"
LAYER = 5
TIMESTEP = 749


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(tmp_path: Path) -> tuple[Path, tuple[int, ...]]:
    path = tmp_path / "p24.jsonl"
    window = list(range(100, 128))
    row = {
        "sample_id": "scene-a_w000_p24",
        "source": "scannetpp",
        "scene_id": "scene-a",
        "streaming_window_id": 0,
        "streaming_obs_start": 0,
        "streaming_window_indices": window,
        "streaming_indices": window[:24],
    }
    path.write_text(json.dumps(row) + "\n")
    targets, _meta = canonical_target_unions(path.resolve(), _sha(path))
    return path, targets[("scannetpp", "scene-a")]


def _protocol(tmp_path: Path) -> Path:
    path = tmp_path / "protocol.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "canonical_protocol": "inscene_streaming_canonical_v2_obs28_h124",
                "canonical_protocol_sha256": "a" * 64,
                "source": "scannetpp",
                "frame_index_schema": "image_mask_intersection_v1",
            }
        )
        + "\n"
    )
    return path


def _values(raw_ids: tuple[int, ...]) -> torch.Tensor:
    return torch.stack(
        [torch.full((2, 3), float(raw_id), dtype=torch.float16) for raw_id in raw_ids]
    )


def _write_gap_worklist(
    path: Path,
    *,
    canonical_sha: str,
    protocol_file_sha: str,
    raw_ids: tuple[int, ...],
) -> tuple[str, str]:
    payload = {
        "schema_version": 1,
        "contract": GAP_CONTRACT,
        "canonical_protocol": "inscene_streaming_canonical_v2_obs28_h124",
        "protocol_sha256": "a" * 64,
        "protocol_file_sha256": protocol_file_sha,
        "canonical_source_sha256": canonical_sha,
        "mode": "target_isolated",
        "vfm": "wan",
        "model_id": MODEL,
        "timestep": TIMESTEP,
        "feature_namespace": "FEAT_TARGET_CANONICAL_V2_GAP_DELTA_V1",
        "output_layers": [LAYER],
        "source": "scannetpp",
        "prompt_contract": {"mode": "exact", "value": "."},
        "preprocess_identity": {"resize": [4, 6], "num_frames": 3},
        "entries": [
            {
                "scene": "scene-a",
                "frame_index": raw_id,
                "layers": [LAYER],
                "reason": "CANONICAL_TARGET_RAW_ID_UNROUTED",
            }
            for raw_id in raw_ids
        ],
        "counts": {
            "entries": len(raw_ids),
            "layer_items": len(raw_ids),
            "scenes": 1,
        },
    }
    payload["content_sha256"] = canonical_json_sha(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return _sha(path), payload["content_sha256"]


def _write_source(
    root: Path,
    name: str,
    raw_ids: tuple[int, ...],
    *,
    kind: str,
    canonical_sha: str,
    worklist: Path | None = None,
    worklist_file_sha: str | None = None,
    worklist_content_sha: str | None = None,
    prompt: str = ".",
    include_layer: bool | None = None,
) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    feature = root / name
    index = root / f"{name}.indices.npy"
    save_file({"feat": _values(raw_ids)}, feature)
    with index.open("wb") as handle:
        np.save(handle, np.asarray(raw_ids, dtype=np.int64), allow_pickle=False)
    index_sha = _sha(index)
    feature_meta = {
        "schema_version": 1,
        "frame_index_schema": "image_mask_intersection_v1",
        "vfm": "wan",
        "model_id": MODEL,
        "mode": "target_isolated",
        "timestep": TIMESTEP,
        "prompt": prompt,
        "resize": [4, 6],
        "num_frames": 3,
        "file": feature.name,
        "size_bytes": feature.stat().st_size,
        "sha256": None if kind == "legacy" else _sha(feature),
        "tensors": {
            "feat": {"shape": [len(raw_ids), 2, 3], "dtype": "torch.float16"}
        },
    }
    index_meta = {
        "schema_version": 1,
        "file": index.name,
        "size_bytes": index.stat().st_size,
        "sha256": index_sha,
        "shape": [len(raw_ids)],
        "dtype": "int64",
        "target_raw_ids": list(raw_ids),
    }
    if include_layer is None:
        include_layer = kind == "gap_delta"
    if include_layer:
        feature_meta["layer"] = LAYER
        index_meta["layer"] = LAYER
    source = {
        "source_id": kind,
        "kind": kind,
        "feature_path": str(feature.resolve()),
        "index_path": str(index.resolve()),
    }
    if kind == "legacy":
        source["expected_source_filename"] = feature.name
    else:
        shared = {
            "feature_namespace": "FEAT_TARGET_CANONICAL_V2_GAP_DELTA_V1",
            "gap_contract": GAP_CONTRACT,
            "gap_worklist_file_sha256": worklist_file_sha,
            "gap_worklist_content_sha256": worklist_content_sha,
            "canonical_source_sha256": canonical_sha,
            "vfm": "wan",
            "model_id": MODEL,
            "timestep": TIMESTEP,
            "layer": LAYER,
            "prompt": prompt,
            "resize": [4, 6],
            "num_frames": 3,
            "target_indices_sha256": index_sha,
        }
        feature_meta.update(shared)
        index_meta.update(shared)
        source.update(
            {
                "gap_worklist_path": str(worklist.resolve()),
                "gap_worklist_file_sha256": worklist_file_sha,
                "gap_worklist_content_sha256": worklist_content_sha,
                "expected_payload_sha256": _sha(feature),
                "expected_index_sha256": index_sha,
            }
        )
    Path(f"{feature}.manifest.json").write_text(json.dumps(feature_meta) + "\n")
    Path(f"{index}.manifest.json").write_text(json.dumps(index_meta) + "\n")
    return source


def _plan(
    tmp_path: Path,
    sources: list[dict],
    *,
    canonical: Path,
    protocol: Path,
) -> Path:
    root = tmp_path / MIXED_TARGET_NAMESPACE
    payload = {
        "schema_version": 1,
        "plan_type": "canonical_v2_target_row_merge_plan",
        "contract": CONTRACT,
        "canonical_protocol": "inscene_streaming_canonical_v2_obs28_h124",
        "protocol_file": str(protocol.resolve()),
        "protocol_file_sha256": _sha(protocol),
        "canonical_source": {"path": str(canonical.resolve()), "sha256": _sha(canonical)},
        "destination_root": str(root.resolve()),
        "units": [
            {
                "spec_id": "wan-t749-l5",
                "vfm": "wan",
                "model_id": MODEL,
                "accepted_model_ids": [MODEL],
                "layer": LAYER,
                "timestep": TIMESTEP,
                "tensor_key": "feat",
                "source": "scannetpp",
                "scene_id": "scene-a",
                "expected_tail_shape": [2, 3],
                "expected_dtype": "torch.float16",
                "accepted_modes": ["target_isolated"],
                "prompt_contract": {"mode": "exact", "value": "."},
                "preprocess_identity": {"resize": [4, 6], "num_frames": 3},
                "destination_filename": "feature_t749_layer5.sft",
                "sources": sources,
            }
        ],
    }
    payload["content_sha256"] = canonical_json_sha(payload)
    path = tmp_path / "merge_plan.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _fixture(tmp_path: Path, *, duplicate: bool = False, wrong_prompt: bool = False):
    canonical, expected = _canonical(tmp_path)
    protocol = _protocol(tmp_path)
    split = len(expected) // 2
    legacy_ids = expected[:split]
    gap_ids = expected[split:]
    if duplicate:
        gap_ids = (legacy_ids[-1], *gap_ids)
    worklist = tmp_path / "gap_worklist.json"
    worklist_file_sha, worklist_content_sha = _write_gap_worklist(
        worklist,
        canonical_sha=_sha(canonical),
        protocol_file_sha=_sha(protocol),
        raw_ids=gap_ids,
    )
    legacy = _write_source(
        tmp_path / "legacy",
        "feature_t749_layer5.sft",
        legacy_ids,
        kind="legacy",
        canonical_sha=_sha(canonical),
        prompt="" if wrong_prompt else ".",
    )
    gap = _write_source(
        tmp_path / "gap",
        "feature_t749_layer5.delta.sft",
        gap_ids,
        kind="gap_delta",
        canonical_sha=_sha(canonical),
        worklist=worklist,
        worklist_file_sha=worklist_file_sha,
        worklist_content_sha=worklist_content_sha,
    )
    return _plan(tmp_path, [legacy, gap], canonical=canonical, protocol=protocol), expected


def _augment_fixture(
    tmp_path: Path,
    *,
    delta_mode: str = "exact",
    dino_conditional: bool = False,
):
    canonical, expected = _canonical(tmp_path)
    protocol = _protocol(tmp_path)
    split = len(expected) // 2
    legacy = _write_source(
        tmp_path / "legacy",
        "feature_t749_layer5.sft",
        expected[:split],
        kind="legacy",
        canonical_sha=_sha(canonical),
    )
    initial = _plan(tmp_path, [legacy], canonical=canonical, protocol=protocol)
    if dino_conditional:
        payload = json.loads(initial.read_text())
        unit = payload["units"][0]
        unit["vfm"] = "dino"
        unit["spec_id"] = "dino-t749-l5"
        unit["duplicate_owner_policy"] = "require_bitwise_equal_then_priority"
        unit["sources"][0]["source_id"] = "dino_prefix24_aoss"
        unit["sources"][0]["priority"] = 0
        payload["content_sha256"] = canonical_json_sha(
            {key: value for key, value in payload.items() if key != "content_sha256"}
        )
        initial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    initial_sha = _sha(initial)
    audit_result = audit_plan(initial, verify_source_sha=False, remote_policy="pending")
    assert audit_result["report"]["status"] == "INCOMPLETE"
    audit_dir = tmp_path / "route_audit"
    report = write_outputs(audit_dir, audit_result)
    audit_path = audit_dir / "target_row_route_audit.json"
    binding = report["outputs"]["gap_worklists"][0]
    worklist_path = Path(binding["path"])
    worklist = json.loads(worklist_path.read_text())
    missing = tuple(
        entry["frame_index"]
        for entry in worklist["entries"]
        if entry["scene"] == "scene-a" and LAYER in entry["layers"]
    )

    delta_root = tmp_path / GAP_TARGET_NAMESPACE
    delta_root.mkdir()
    if not dino_conditional:
        if delta_mode == "exact":
            delta_ids = missing
        elif delta_mode == "missing":
            delta_ids = missing[:-1]
        elif delta_mode == "duplicate":
            delta_ids = tuple(sorted((*missing, missing[-1])))
        elif delta_mode == "extra":
            delta_ids = tuple(sorted((*missing, expected[0])))
        else:
            raise AssertionError(delta_mode)
        delta_dir = (
            delta_root
            / "wan"
            / "scannetpp"
            / "scene-a"
            / ".canonical_v2_target_delta"
            / binding["file_sha256"][:16]
        )
        source = _write_source(
            delta_dir,
            "feature_t749_layer5.delta.sft",
            delta_ids,
            kind="gap_delta",
            canonical_sha=_sha(canonical),
            worklist=worklist_path,
            worklist_file_sha=binding["file_sha256"],
            worklist_content_sha=binding["content_sha256"],
        )
        old_index = Path(source["index_path"])
        old_sidecar = Path(f"{old_index}.manifest.json")
        new_index = delta_dir / "target_indices_layer5.npy"
        new_sidecar = Path(f"{new_index}.manifest.json")
        old_index.replace(new_index)
        index_meta = json.loads(old_sidecar.read_text())
        index_meta["file"] = new_index.name
        new_sidecar.write_text(json.dumps(index_meta) + "\n")
        old_sidecar.unlink()

    shard = {
        "schema_version": 1,
        "status": "READY",
        "contract": GAP_CONTRACT,
        "mode": "target_isolated",
        "feature_namespace": GAP_TARGET_NAMESPACE,
        "feature_root": str(delta_root.resolve()),
        "canonical_source_sha256": _sha(canonical),
        "worklist_file_sha256": binding["file_sha256"],
        "worklist_content_sha256": binding["content_sha256"],
        "source_audit_sha256": _sha(audit_path),
        "source_audit_content_sha256": report["content_sha256"],
        "vfm": worklist["vfm"],
        "model_id": worklist["model_id"],
        "timestep": worklist["timestep"],
        "source": worklist["source"],
        "layers": worklist["output_layers"],
        "scene_start": 0,
        "scene_end": 1,
        "shard_id": "fixture-target-gap",
    }
    shard["content_sha256"] = canonical_json_sha(shard)
    shard_path = tmp_path / "gap_shard.json"
    shard_path.write_text(json.dumps(shard, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema_version": 1,
        "status": "COMPLETE",
        "shard_id": shard["shard_id"],
        "shard_content_sha256": shard["content_sha256"],
        "worklist_file_sha256": binding["file_sha256"],
        "worklist_content_sha256": binding["content_sha256"],
        "canonical_source_sha256": _sha(canonical),
        "processes": [{"gpu": 0, "start": 0, "end": 1, "returncode": 0}],
    }
    receipt_path = tmp_path / "gap_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return {
        "initial": initial,
        "initial_sha": initial_sha,
        "audit": audit_path,
        "delta_root": delta_root,
        "binding": (shard_path, receipt_path),
        "output": tmp_path / "augmented_plan.json",
        "expected": expected,
        "missing": missing,
    }


def _enable_dino_conditional_duplicates(plan: Path) -> dict:
    payload = json.loads(plan.read_text())
    unit = payload["units"][0]
    unit["vfm"] = "dino"
    unit["spec_id"] = "dino-t749-l5"
    unit["duplicate_owner_policy"] = "require_bitwise_equal_then_priority"
    identities = (("dino_prefix24_aoss", 0), ("dino_prefix16_aoss", 1))
    for source, (source_id, priority) in zip(unit["sources"], identities):
        source["source_id"] = source_id
        source["kind"] = "legacy"
        source["priority"] = priority
        source["expected_source_filename"] = Path(source["feature_path"]).name
        for key in (
            "gap_worklist_path",
            "gap_worklist_file_sha256",
            "gap_worklist_content_sha256",
        ):
            source.pop(key, None)
        sidecar_path = Path(f"{source['feature_path']}.manifest.json")
        sidecar = json.loads(sidecar_path.read_text())
        sidecar["vfm"] = "dino"
        sidecar_path.write_text(json.dumps(sidecar) + "\n")
    payload["content_sha256"] = canonical_json_sha(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    plan.write_text(json.dumps(payload) + "\n")
    return payload


def test_auditor_emits_hash_bound_gap_worklist(tmp_path: Path) -> None:
    canonical, expected = _canonical(tmp_path)
    protocol = _protocol(tmp_path)
    legacy = _write_source(
        tmp_path / "legacy",
        "feature_t749_layer5.sft",
        expected[:5],
        kind="legacy",
        canonical_sha=_sha(canonical),
    )
    plan = _plan(tmp_path, [legacy], canonical=canonical, protocol=protocol)
    result = audit_plan(plan, verify_source_sha=False)

    assert result["report"]["status"] == "INCOMPLETE"
    assert result["report"]["counts"]["routed_rows"] == 5
    assert result["missing_units"][0]["missing_raw_ids"] == list(expected[5:])
    report = write_outputs(tmp_path / "audit", result)
    binding = report["outputs"]["gap_worklists"][0]
    worklist = json.loads(Path(binding["path"]).read_text())
    assert worklist["contract"] == GAP_CONTRACT
    assert worklist["prompt_contract"] == {"mode": "exact", "value": "."}
    assert worklist["preprocess_identity"] == {"resize": [4, 6], "num_frames": 3}
    assert hashlib.sha256(Path(binding["path"]).read_bytes()).hexdigest() == binding["file_sha256"]


def test_writer_dry_run_then_merges_legacy_and_gap_in_canonical_order(tmp_path: Path) -> None:
    plan, expected = _fixture(tmp_path)
    destination_root = tmp_path / MIXED_TARGET_NAMESPACE
    dry_run = materialize(plan, execute=False)
    assert dry_run["status"] == "DRY_RUN_PASS"
    assert not destination_root.exists()

    receipt = materialize(plan, execute=True)
    assert receipt["status"] == "PASS"
    destination = (
        destination_root
        / "wan"
        / "scannetpp"
        / "scene-a"
        / "feature_t749_layer5.sft"
    )
    indices = np.load(destination.parent / "target_indices.npy", allow_pickle=False)
    tensor = load_file(destination)["feat"]
    assert indices.tolist() == list(expected)
    assert tensor.shape == (len(expected), 2, 3)
    assert [float(tensor[index, 0, 0]) for index in range(len(expected))] == list(expected)
    feature_receipt = json.loads(Path(f"{destination}.merge_receipt.json").read_text())
    assert feature_receipt["status"] == "PASS"
    assert feature_receipt["bitwise_rows_verified"] == len(expected)
    assert feature_receipt["rows_by_source_kind"] == {
        "gap_delta": len(expected) - len(expected) // 2,
        "legacy": len(expected) // 2,
    }


def test_augmenter_builds_new_complete_plan_and_dry_materializer_passes(
    tmp_path: Path,
) -> None:
    case = _augment_fixture(tmp_path)
    result = augment_plan(
        initial_plan_path=case["initial"],
        route_audit_path=case["audit"],
        delta_root=case["delta_root"],
        completion_bindings=[case["binding"]],
        output_path=case["output"],
    )
    assert result["status"] == "AUGMENTED_PLAN_WRITTEN_NOT_MATERIALIZED"
    assert result["gap_row_layer_items_added"] == len(case["missing"])
    assert _sha(case["initial"]) == case["initial_sha"]
    complete = audit_plan(case["output"], verify_source_sha=True)
    assert complete["report"]["status"] == "PASS"
    assert complete["report"]["counts"]["routed_rows"] == len(case["expected"])
    preview = materialize(case["output"], execute=False)
    assert preview["status"] == "DRY_RUN_PASS"


@pytest.mark.parametrize("delta_mode", ["missing", "duplicate", "extra"])
def test_augmenter_rejects_nonexact_delta_rows_without_publishing(
    tmp_path: Path, delta_mode: str
) -> None:
    case = _augment_fixture(tmp_path, delta_mode=delta_mode)
    with pytest.raises(AugmentError, match="gap delta"):
        augment_plan(
            initial_plan_path=case["initial"],
            route_audit_path=case["audit"],
            delta_root=case["delta_root"],
            completion_bindings=[case["binding"]],
            output_path=case["output"],
        )
    assert not case["output"].exists()


def test_augmenter_requires_complete_receipt_without_publishing(tmp_path: Path) -> None:
    case = _augment_fixture(tmp_path)
    with pytest.raises(AugmentError, match="at least one completed"):
        augment_plan(
            initial_plan_path=case["initial"],
            route_audit_path=case["audit"],
            delta_root=case["delta_root"],
            completion_bindings=[],
            output_path=case["output"],
        )
    assert not case["output"].exists()


def test_materializer_rejects_same_size_delta_mutation_after_augmentation(
    tmp_path: Path,
) -> None:
    case = _augment_fixture(tmp_path)
    augment_plan(
        initial_plan_path=case["initial"],
        route_audit_path=case["audit"],
        delta_root=case["delta_root"],
        completion_bindings=[case["binding"]],
        output_path=case["output"],
    )
    plan = json.loads(case["output"].read_text())
    delta = next(
        source
        for source in plan["units"][0]["sources"]
        if source["kind"] == "gap_delta"
    )
    feature = Path(delta["feature_path"])
    payload = bytearray(feature.read_bytes())
    payload[-1] ^= 1
    feature.write_bytes(payload)
    with pytest.raises(MergeError, match="not complete and unambiguous"):
        materialize(case["output"], execute=True)
    assert not Path(plan["destination_root"]).exists()


def test_augmenter_blocks_dino_conditional_plan_until_routes_are_frozen(
    tmp_path: Path,
) -> None:
    case = _augment_fixture(tmp_path, dino_conditional=True)
    with pytest.raises(AugmentError, match="frozen into one unambiguous"):
        augment_plan(
            initial_plan_path=case["initial"],
            route_audit_path=case["audit"],
            delta_root=case["delta_root"],
            completion_bindings=[case["binding"]],
            output_path=case["output"],
        )
    assert not case["output"].exists()


def test_duplicate_raw_id_owner_is_fatal(tmp_path: Path) -> None:
    plan, _expected = _fixture(tmp_path, duplicate=True)
    result = audit_plan(plan, verify_source_sha=False)
    assert result["report"]["status"] == "FAIL"
    assert result["report"]["counts"]["duplicate_rows"] == 1
    with pytest.raises(MergeError, match="not complete and unambiguous"):
        materialize(plan, execute=True)


def test_explicit_duplicate_policy_requires_bitwise_equal_rows(tmp_path: Path) -> None:
    plan, _expected = _fixture(tmp_path, duplicate=True)
    _enable_dino_conditional_duplicates(plan)
    result = audit_plan(plan, verify_source_sha=False)
    assert result["report"]["status"] == "PASS"
    assert result["report"]["counts"]["duplicate_rows_requiring_bitwise_check"] == 1
    receipt = materialize(plan, execute=True)
    assert receipt["status"] == "PASS"
    feature_receipt = json.loads(
        next((tmp_path / MIXED_TARGET_NAMESPACE).rglob("*.merge_receipt.json")).read_text()
    )
    assert feature_receipt["duplicate_rows_bitwise_verified"] == 1


def test_explicit_duplicate_policy_rejects_nonidentical_rows(tmp_path: Path) -> None:
    plan, _expected = _fixture(tmp_path, duplicate=True)
    payload = _enable_dino_conditional_duplicates(plan)
    gap_path = Path(payload["units"][0]["sources"][1]["feature_path"])
    tensors = load_file(str(gap_path))
    tensors["feat"][0].add_(1)
    save_file(tensors, str(gap_path))
    sidecar_path = Path(f"{gap_path}.manifest.json")
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["size_bytes"] = gap_path.stat().st_size
    sidecar["sha256"] = _sha(gap_path)
    sidecar["tensors"]["feat"]["shape"] = list(tensors["feat"].shape)
    sidecar_path.write_text(json.dumps(sidecar) + "\n")
    payload["content_sha256"] = canonical_json_sha(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    plan.write_text(json.dumps(payload) + "\n")
    with pytest.raises(MergeError, match="not bitwise equal"):
        materialize(plan, execute=True)


def test_conditional_duplicate_policy_is_rejected_for_non_dino(tmp_path: Path) -> None:
    plan, _expected = _fixture(tmp_path, duplicate=True)
    payload = json.loads(plan.read_text())
    payload["units"][0]["duplicate_owner_policy"] = (
        "require_bitwise_equal_then_priority"
    )
    payload["content_sha256"] = canonical_json_sha(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    plan.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="restricted to vfm=dino"):
        audit_plan(plan, verify_source_sha=False)


def test_conditional_duplicate_policy_rejects_forged_dino_source(tmp_path: Path) -> None:
    plan, _expected = _fixture(tmp_path, duplicate=True)
    payload = _enable_dino_conditional_duplicates(plan)
    payload["units"][0]["sources"][0]["source_id"] = "forged_dino_cache"
    payload["content_sha256"] = canonical_json_sha(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    plan.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="forged/non-prefix"):
        audit_plan(plan, verify_source_sha=False)
    with pytest.raises(MergeError, match="forged/non-prefix"):
        materialize(plan, execute=False)


def test_conditional_duplicate_policy_rejects_nonlegacy_source(tmp_path: Path) -> None:
    plan, _expected = _fixture(tmp_path, duplicate=True)
    payload = _enable_dino_conditional_duplicates(plan)
    payload["units"][0]["sources"][1]["kind"] = "gap_delta"
    payload["content_sha256"] = canonical_json_sha(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    plan.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="only legacy prefix sources"):
        audit_plan(plan, verify_source_sha=False)


def test_full_coverage_source_plus_pending_alternate_keeps_whole_unit_pending(
    tmp_path: Path,
) -> None:
    canonical, expected = _canonical(tmp_path)
    protocol = _protocol(tmp_path)
    first = _write_source(
        tmp_path / "first",
        "feature_t749_layer5.sft",
        expected,
        kind="legacy",
        canonical_sha=_sha(canonical),
    )
    alternate = _write_source(
        tmp_path / "alternate",
        "feature_t749_layer5.sft",
        expected[:1],
        kind="legacy",
        canonical_sha=_sha(canonical),
    )
    plan = _plan(tmp_path, [first, alternate], canonical=canonical, protocol=protocol)
    payload = _enable_dino_conditional_duplicates(plan)
    payload["units"][0]["sources"][1]["feature_path"] = (
        "aoss://example/dino/feature.sft"
    )
    payload["units"][0]["sources"][1]["index_path"] = (
        "aoss://example/dino/target_indices.npy"
    )
    payload["content_sha256"] = canonical_json_sha(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    plan.write_text(json.dumps(payload) + "\n")

    result = audit_plan(
        plan,
        verify_source_sha=False,
        remote_policy="pending",
    )
    assert result["report"]["status"] == "PENDING"
    assert result["report"]["counts"]["routed_rows"] == 0
    assert result["report"]["counts"]["pending_rows"] == len(expected)
    assert result["report"]["units"][0]["multi_source_gate_pending"] is True


def test_gap_source_declaration_requires_both_worklist_shas(tmp_path: Path) -> None:
    plan, expected = _fixture(tmp_path)
    payload = json.loads(plan.read_text())
    payload["units"][0]["sources"][1].pop("gap_worklist_content_sha256")
    payload["content_sha256"] = canonical_json_sha(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    plan.write_text(json.dumps(payload) + "\n")
    result = audit_plan(plan, verify_source_sha=False)
    assert result["report"]["status"] == "INCOMPLETE"
    assert result["report"]["counts"]["missing_rows"] == len(expected) - len(expected) // 2
    assert "requires worklist file/content SHA256" in (
        result["report"]["units"][0]["source_failures"][0]["reason"]
    )


@pytest.mark.parametrize("field", ["feature_namespace", "gap_contract"])
def test_gap_sidecars_require_delta_namespace_and_contract(
    tmp_path: Path, field: str
) -> None:
    plan, expected = _fixture(tmp_path)
    payload = json.loads(plan.read_text())
    source = payload["units"][0]["sources"][1]
    for sidecar_path in (
        Path(f"{source['feature_path']}.manifest.json"),
        Path(f"{source['index_path']}.manifest.json"),
    ):
        sidecar = json.loads(sidecar_path.read_text())
        sidecar.pop(field)
        sidecar_path.write_text(json.dumps(sidecar) + "\n")
    result = audit_plan(plan, verify_source_sha=False)
    assert result["report"]["status"] == "INCOMPLETE"
    assert result["report"]["counts"]["missing_rows"] == len(expected) - len(expected) // 2
    assert field in result["report"]["units"][0]["source_failures"][0]["reason"]


def test_prompt_mismatch_rejects_legacy_rows_and_routes_them_to_gap(tmp_path: Path) -> None:
    plan, expected = _fixture(tmp_path, wrong_prompt=True)
    result = audit_plan(plan, verify_source_sha=False)
    assert result["report"]["status"] == "INCOMPLETE"
    assert result["report"]["counts"]["source_payloads_failed"] == 1
    assert result["report"]["counts"]["missing_rows"] == len(expected) // 2
    assert "prompt mismatch" in result["missing_units"][0]["source_gate_failures"][0]["reason"]


def test_seeded_destination_is_rejected(tmp_path: Path) -> None:
    plan, _expected = _fixture(tmp_path)
    payload = json.loads(plan.read_text())
    payload["destination_root"] = str(
        (tmp_path / "FEAT_TARGET_CANONICAL_V2_SEEDED_V1").resolve()
    )
    payload["content_sha256"] = canonical_json_sha(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    plan.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="must end in"):
        audit_plan(plan, verify_source_sha=False)


def test_remote_feature_uses_header_and_row_ranges_not_full_download(
    tmp_path: Path, monkeypatch
) -> None:
    canonical, expected = _canonical(tmp_path)
    protocol = _protocol(tmp_path)
    local = _write_source(
        tmp_path / "remote_source",
        "feature_t749_layer5.sft",
        expected,
        kind="legacy",
        canonical_sha=_sha(canonical),
    )
    locations = {
        "feature_path": "aoss://bucket/feature_t749_layer5.sft",
        "index_path": "aoss://bucket/feature_t749_layer5.sft.indices.npy",
        "feature_sidecar_path": "aoss://bucket/feature_t749_layer5.sft.manifest.json",
        "index_sidecar_path": "aoss://bucket/feature_t749_layer5.sft.indices.npy.manifest.json",
    }
    blobs = {
        "fj:s3://bucket/feature_t749_layer5.sft": Path(local["feature_path"]).read_bytes(),
        "fj:s3://bucket/feature_t749_layer5.sft.indices.npy": Path(local["index_path"]).read_bytes(),
        "fj:s3://bucket/feature_t749_layer5.sft.manifest.json": Path(
            f"{local['feature_path']}.manifest.json"
        ).read_bytes(),
        "fj:s3://bucket/feature_t749_layer5.sft.indices.npy.manifest.json": Path(
            f"{local['index_path']}.manifest.json"
        ).read_bytes(),
    }

    class FakeClient:
        def __init__(self):
            self.calls: list[tuple[str, str | None]] = []

        def size(self, uri: str) -> int:
            return len(blobs[uri])

        def get(self, uri: str, range: str | None = None):
            self.calls.append((uri, range))
            data = blobs[uri]
            if range is None:
                return data
            start, end = (int(value) for value in range.split("-"))
            return data[start : end + 1]

    client = FakeClient()
    monkeypatch.setattr(target_row_source, "_get_client", lambda: client)
    remote_source = {
        **local,
        **locations,
        "expected_source_filename": "feature_t749_layer5.sft",
    }
    plan = _plan(tmp_path, [remote_source], canonical=canonical, protocol=protocol)
    result = materialize(plan, execute=True)
    assert result["status"] == "PASS"
    feature_uri = "fj:s3://bucket/feature_t749_layer5.sft"
    feature_calls = [byte_range for uri, byte_range in client.calls if uri == feature_uri]
    assert feature_calls
    assert all(byte_range is not None for byte_range in feature_calls)
    assert any(byte_range == "0-7" for byte_range in feature_calls)


def test_remote_policy_pending_never_becomes_a_gap(
    tmp_path: Path, monkeypatch
) -> None:
    canonical, expected = _canonical(tmp_path)
    protocol = _protocol(tmp_path)
    local = _write_source(
        tmp_path / "remote_source",
        "feature_t749_layer5.sft",
        expected,
        kind="legacy",
        canonical_sha=_sha(canonical),
    )
    remote = {
        **local,
        "feature_path": "aoss://bucket/feature_t749_layer5.sft",
        "index_path": "aoss://bucket/target_indices.npy",
        "feature_sidecar_path": "aoss://bucket/feature_t749_layer5.sft.manifest.json",
        "index_sidecar_path": "aoss://bucket/target_indices.npy.manifest.json",
    }
    monkeypatch.setattr(
        target_row_source,
        "_get_client",
        lambda: pytest.fail("remote_policy=pending contacted AOSS"),
    )
    plan = _plan(tmp_path, [remote], canonical=canonical, protocol=protocol)
    result = audit_plan(plan, verify_source_sha=False, remote_policy="pending")
    assert result["report"]["status"] == "PENDING"
    assert result["report"]["counts"]["pending_rows"] == len(expected)
    assert result["report"]["counts"]["missing_rows"] == 0
    assert not result["missing_units"]
    assert len(result["pending_units"]) == 1
    assert build_gap_worklists(result) == []


def test_remote_gate_timeout_stays_pending_not_reextract(
    tmp_path: Path, monkeypatch
) -> None:
    canonical, expected = _canonical(tmp_path)
    protocol = _protocol(tmp_path)
    remote = {
        "source_id": "slow_remote",
        "kind": "legacy",
        "feature_path": "aoss://bucket/feature_t749_layer5.sft",
        "index_path": "aoss://bucket/target_indices.npy",
        "expected_source_filename": "feature_t749_layer5.sft",
    }

    class SlowClient:
        def get(self, *_args, **_kwargs):
            time.sleep(0.2)
            return None

        def size(self, *_args, **_kwargs):
            time.sleep(0.2)
            return 1

    monkeypatch.setattr(target_row_source, "_get_client", lambda: SlowClient())
    plan = _plan(tmp_path, [remote], canonical=canonical, protocol=protocol)
    result = audit_plan(
        plan,
        verify_source_sha=False,
        remote_policy="gate",
        source_timeout_seconds=0.01,
    )
    assert result["report"]["status"] == "PENDING"
    assert result["report"]["counts"]["source_payloads_pending"] == 1
    assert result["report"]["counts"]["missing_rows"] == 0
    assert build_gap_worklists(result) == []


def test_worklist_identity_contract_is_strict(tmp_path: Path) -> None:
    plan, _expected = _fixture(tmp_path)
    payload = json.loads(plan.read_text())
    payload["units"][0]["prompt_contract"] = {
        "mode": "not_applicable",
        "value": "not-null",
    }
    payload["content_sha256"] = canonical_json_sha(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    plan.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="requires value=null"):
        audit_plan(plan, verify_source_sha=False)
