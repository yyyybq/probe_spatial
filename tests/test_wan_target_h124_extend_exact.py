import json
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import extract_wan_manifest_exact as base
from scripts import extract_wan_target_h124_extend_exact as extend


def _rows(scene="scene"):
    rows = []
    for window_id, start in enumerate((0, 5, 10, 15)):
        window = list(range(start, start + 6))
        rows.append(
            {
                "sample_id": f"{scene}_w{window_id:03d}_p2",
                "scene_id": scene,
                "scene_dir": f"/data/{scene}",
                "source": "scannetpp",
                "split": "train",
                "streaming_window_id": window_id,
                "streaming_tail": 1,
                "streaming_indices": window[:2],
                "streaming_window_indices": window,
                "valid_frames": [f"frame_{index:04d}.jpg" for index in range(32)],
                # This is intentionally wrong. Direct derivation must ignore it.
                "target_raw_id": 31,
            }
        )
    return rows


def _contract():
    return extend.Contract(
        layers=(0, 5),
        feature_shape=(2, 2, 3),
        input_frames=81,
    )


def _git_identity():
    return extend.GitIdentity(
        repo_root="/fixture/repo",
        commit=extend.EXPECTED_GIT_COMMIT,
        verification_source="direct:.git/HEAD->.git/refs/heads/main",
    )


def _write_direct_git_repo(root: Path):
    reference = root / ".git/refs/heads/main"
    reference.parent.mkdir(parents=True)
    (root / ".git/HEAD").write_text("ref: refs/heads/main\n")
    reference.write_text(f"{extend.EXPECTED_GIT_COMMIT}\n")


def _write_input_records(data_root: Path, rows, contract):
    mapping = extend.validate_scene_rows("scene", rows, contract)
    records = {}
    for raw_id, (horizon, row, _) in mapping.items():
        if horizon not in contract.extract_horizons:
            continue
        path = (
            base.local_scene_dir(data_root, row)
            / "images"
            / row["valid_frames"][raw_id]
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture-image-{raw_id}".encode())
        records[str(raw_id)] = extend.snapshot_extracted_input_image(
            data_root, "scene", row, raw_id, horizon
        )
    return records


def _write_source(root: Path, rows, model_id: Path, contract):
    scene = rows[0]["scene_id"]
    mapping = extend.validate_scene_rows(scene, rows, contract)
    h1_ids = sorted(raw for raw, value in mapping.items() if value[0] == 1)
    scene_root = root / "wan" / "scannetpp" / scene
    common = {
        "schema_version": 2,
        "canonical_manifest_sha256": contract.manifest_sha256,
        "extractor_sha256": contract.base_extractor_sha256,
        "mode": "manifest_exact_target",
        "model_id": str(model_id.resolve()),
        "prompt": contract.prompt,
        "timestep": contract.timestep,
        "layers": list(contract.layers),
        "num_frames": contract.input_frames,
        "base_seed": contract.seed,
        "target_raw_ids": h1_ids,
        "scene_id": scene,
        "sample_ids": sorted(row["sample_id"] for row in rows),
    }
    base.atomic_npy(
        scene_root / "target_indices.npy",
        np.asarray(h1_ids, dtype=np.int64),
        common,
    )
    tensors = {}
    for layer in contract.layers:
        tensor = torch.stack(
            [
                torch.full(contract.feature_shape, raw_id + layer, dtype=torch.float16)
                for raw_id in h1_ids
            ]
        )
        tensors[layer] = tensor
        base.atomic_safetensors(
            scene_root / f"feature_t{contract.timestep}_layer{layer}.sft",
            tensor,
            {
                **common,
                "target_frame_names": [
                    mapping[raw_id][1]["valid_frames"][raw_id] for raw_id in h1_ids
                ],
                "target_id_sources": {
                    str(raw_id): mapping[raw_id][2] for raw_id in h1_ids
                },
                "sample_seeds": {
                    str(raw_id): base.deterministic_seed(
                        contract.seed, f"target:{scene}:{raw_id}"
                    )
                    for raw_id in h1_ids
                },
            },
        )
    return tensors


def test_direct_horizon_derivation_ignores_explicit_target_shortcut():
    row = _rows()[0]
    assert row["target_raw_id"] == 31
    assert extend.direct_target_raw_id(row, 1) == (2, "streaming_window_indices[2]")
    assert extend.direct_target_raw_id(row, 2) == (3, "streaming_window_indices[3]")
    assert extend.direct_target_raw_id(row, 4) == (5, "streaming_window_indices[5]")


def test_scene_contract_requires_disjoint_h124_and_complete_scene():
    rows = _rows()
    mapping = extend.validate_scene_rows("scene", rows, _contract())
    assert len(mapping) == 12
    assert {value[0] for value in mapping.values()} == {1, 2, 4}
    with pytest.raises(extend.ContractError, match="four manifest rows"):
        extend.validate_scene_rows("scene", rows[:3], _contract())


def test_h1_validation_checks_sidecar_sha_shape_identity_and_finite(tmp_path):
    rows = _rows()
    source_root = tmp_path / "source"
    model_id = tmp_path / "model"
    model_id.mkdir()
    contract = _contract()
    _write_source(source_root, rows, model_id, contract)
    source = extend.validate_source_h1_scene(
        source_root, "scene", rows, model_id, contract
    )
    assert source.target_ids == [2, 7, 12, 17]
    assert set(source.tensors) == {0, 5}

    sidecar = (
        source_root
        / "wan/scannetpp/scene/feature_t749_layer0.sft.manifest.json"
    )
    metadata = json.loads(sidecar.read_text())
    metadata["sha256"] = "0" * 64
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(extend.ContractError, match="SHA does not match"):
        extend.validate_source_h1_scene(
            source_root, "scene", rows, model_id, contract
        )


def test_collect_only_forwards_h2_h4_and_merge_preserves_h1(tmp_path):
    rows = _rows()
    source_root = tmp_path / "source"
    model_id = tmp_path / "model"
    model_id.mkdir()
    contract = _contract()
    _write_source(source_root, rows, model_id, contract)
    source = extend.validate_source_h1_scene(
        source_root, "scene", rows, model_id, contract
    )
    called = []

    def forward_one(row, raw_id):
        called.append(raw_id)
        return {
            layer: torch.full(contract.feature_shape, 1000 + raw_id + layer)
            for layer in contract.layers
        }

    new = extend.collect_new_features("scene", rows, forward_one, contract)
    mapping = extend.validate_scene_rows("scene", rows, contract)
    expected_new = sorted(raw for raw, value in mapping.items() if value[0] in (2, 4))
    assert called == expected_new
    assert not set(called) & set(source.target_ids)

    all_ids = sorted(mapping)
    merged = extend.merged_layer_tensor(source, new, all_ids, 0)
    for source_row, raw_id in enumerate(source.target_ids):
        out_row = all_ids.index(raw_id)
        assert torch.equal(merged[out_row], source.tensors[0][source_row])


def test_atomic_new_scene_is_valid_and_second_writer_fails_closed(tmp_path):
    rows = _rows()
    source_root = tmp_path / "source"
    out_root = tmp_path / "new"
    model_id = tmp_path / "model"
    model_id.mkdir()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("test fixture; pure-function test does not validate global SHA\n")
    source_audit = tmp_path / "source_audit.json"
    source_audit.write_text("{}\n")
    contract = _contract()
    input_records = _write_input_records(tmp_path / "data", rows, contract)
    _write_source(source_root, rows, model_id, contract)
    source = extend.validate_source_h1_scene(
        source_root, "scene", rows, model_id, contract
    )

    def forward_one(row, raw_id):
        return {
            layer: torch.full(contract.feature_shape, 1000 + raw_id + layer)
            for layer in contract.layers
        }

    new = extend.collect_new_features("scene", rows, forward_one, contract)
    inventory = extend.write_scene(
        out_root,
        "fixture",
        "scene",
        rows,
        source,
        new,
        manifest,
        source_root,
        source_audit,
        model_id,
        contract,
        input_records,
        _git_identity(),
    )
    assert len(inventory) == 6
    assert all(".incomplete-" not in item["path"] for item in inventory)
    assert (out_root / "wan/scannetpp/scene/target_indices.npy").is_file()
    index_sidecar = json.loads(
        (out_root / "wan/scannetpp/scene/target_indices.npy.manifest.json").read_text()
    )
    assert index_sidecar["git_commit"] == extend.EXPECTED_GIT_COMMIT
    assert index_sidecar["git_commit_verified"] is True
    assert index_sidecar["repo_root"] == _git_identity().repo_root
    assert (
        index_sidecar["git_commit_verification_source"]
        == _git_identity().verification_source
    )
    assert index_sidecar["extracted_input_images_by_raw_id"] == input_records
    completion_path = out_root / "provenance/scene_completions/scene.json"
    completion = json.loads(completion_path.read_text())
    assert completion["output_inventory_sha256"] == extend.canonical_json_sha256(
        inventory
    )
    assert extend.validate_scene_markers(
        out_root, "scene", inventory, contract
    ) == "fixture"
    nested = out_root / "wan/scannetpp/scene/unexpected_nested"
    nested.mkdir()
    with pytest.raises(extend.ContractError, match="output entries differ"):
        extend.validate_output_scene(
            out_root / "wan/scannetpp/scene",
            "scene",
            rows,
            source,
            model_id,
            contract,
            _git_identity(),
            input_records,
        )
    nested.rmdir()
    with pytest.raises(extend.ContractError, match="existing scene output"):
        extend.write_scene(
            out_root,
            "fixture",
            "scene",
            rows,
            source,
            new,
            manifest,
            source_root,
            source_audit,
            model_id,
            contract,
            input_records,
            _git_identity(),
        )

    base.atomic_json(completion_path, {"status": "COMPLETE"})
    with pytest.raises(extend.ContractError, match="missing required field"):
        extend.validate_scene_markers(out_root, "scene", inventory, contract)


def test_native_target_indices_reject_int32_without_coercion(tmp_path):
    path = tmp_path / "target_indices.npy"
    np.save(path, np.asarray([1, 2, 3, 4], dtype=np.int32), allow_pickle=False)
    with pytest.raises(extend.ContractError, match="native dtype mismatch"):
        extend.load_native_int64(path, 4)


def test_extracted_input_byte_gate_rejects_same_size_mutation(tmp_path):
    rows = _rows()
    contract = _contract()
    records = _write_input_records(tmp_path / "data", rows, contract)
    first = records[sorted(records)[0]]
    path = Path(first["path"])
    original = path.read_bytes()
    path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    mapping = extend.validate_scene_rows("scene", rows, contract)
    with pytest.raises(extend.ContractError, match="input SHA256 changed"):
        extend.validate_extracted_input_images(records, mapping, "scene", contract)


def test_rename_noreplace_has_one_winner_under_race(tmp_path):
    destination = tmp_path / "published"
    sources = [tmp_path / "writer-a", tmp_path / "writer-b"]
    for label, source in zip(("a", "b"), sources):
        source.mkdir()
        (source / "writer.txt").write_text(label)
    barrier = threading.Barrier(2)

    def publish(source):
        barrier.wait()
        try:
            extend.rename_noreplace(source, destination)
            return "published"
        except extend.ContractError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(publish, sources))
    assert sorted(outcomes) == ["published", "rejected"]
    assert (destination / "writer.txt").read_text() in {"a", "b"}
    assert sum(source.exists() for source in sources) == 1


def test_directory_publish_link_failure_preserves_fail_closed_evidence(
    tmp_path, monkeypatch
):
    source = tmp_path / "private-scene"
    source.mkdir()
    for name in ("a.sft", "b.sft", "indices.npy"):
        (source / name).write_text(name)
    destination = tmp_path / "published-scene"
    original = extend.os.link
    calls = 0

    def fail_second_link(src, dst, *, follow_symlinks=True):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected hard-link failure")
        return original(src, dst, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(extend.os, "link", fail_second_link)
    with pytest.raises(OSError, match="injected hard-link failure"):
        extend.rename_noreplace(source, destination)
    assert source.is_dir()
    assert {path.name for path in source.iterdir()} == {
        "a.sft",
        "b.sft",
        "indices.npy",
    }
    assert destination.is_dir()
    assert {path.name for path in destination.iterdir()} == {"a.sft"}


def test_atomic_json_partial_stage_never_exposes_final_marker(tmp_path, monkeypatch):
    destination = tmp_path / "immutable.json"

    def partial_then_fail(payload, handle, **kwargs):
        handle.write('{"status":')
        handle.flush()
        raise OSError("injected JSON staging failure")

    monkeypatch.setattr(extend.json, "dump", partial_then_fail)
    with pytest.raises(OSError, match="injected JSON staging failure"):
        extend.atomic_json_noreplace(destination, {"status": "OURS"})
    assert not destination.exists()
    assert not list(tmp_path.glob(".immutable.json.incomplete-*"))


def test_atomic_json_noreplace_preserves_injected_foreign_race(
    tmp_path, monkeypatch
):
    destination = tmp_path / "immutable.json"
    original = extend.rename_noreplace

    def inject_foreign_then_publish(source, final):
        final.write_text("foreign-writer\n")
        original(source, final)

    monkeypatch.setattr(extend, "rename_noreplace", inject_foreign_then_publish)
    with pytest.raises(extend.ContractError, match="no-replace publication target"):
        extend.atomic_json_noreplace(destination, {"status": "OURS"})
    assert destination.read_text() == "foreign-writer\n"
    assert not list(tmp_path.glob(".immutable.json.incomplete-*"))


def test_explicit_git_identity_uses_real_repo_from_frozen_wrapper_path(
    tmp_path, monkeypatch
):
    repo = tmp_path / "real-repo"
    _write_direct_git_repo(repo)
    frozen_runtime = tmp_path / "bundle/runtime_tree"
    frozen_runtime.mkdir(parents=True)
    monkeypatch.setattr(extend, "REPO_ROOT", frozen_runtime)
    identity = extend.verify_git_identity(repo, extend.EXPECTED_GIT_COMMIT)
    assert identity.repo_root == str(repo.resolve())
    assert identity.commit == extend.EXPECTED_GIT_COMMIT
    assert identity.verification_source == "direct:.git/HEAD->.git/refs/heads/main"
    with pytest.raises(extend.ContractError, match="not pinned"):
        extend.verify_git_identity(repo, "0" * 40)


def test_extract_path_reaches_model_boundary_with_explicit_git_identity(
    tmp_path, monkeypatch
):
    repo = tmp_path / "real-repo"
    _write_direct_git_repo(repo)
    frozen_runtime = tmp_path / "bundle/runtime_tree"
    frozen_runtime.mkdir(parents=True)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("fixture\n")
    data_root = tmp_path / "data"
    source_root = tmp_path / "source"
    model_id = tmp_path / "model"
    for directory in (data_root, source_root, model_id):
        directory.mkdir()
    source_audit = tmp_path / "audit.json"
    source_audit.write_text("{}\n")
    out_root = tmp_path / "out"
    args = extend.parse_args(
        [
            "--manifest",
            str(manifest),
            "--data-root",
            str(data_root),
            "--source-h1-root",
            str(source_root),
            "--source-h1-audit",
            str(source_audit),
            "--out-root",
            str(out_root),
            "--model-id",
            str(model_id),
            "--repo-root",
            str(repo),
            "--expected-git-commit",
            extend.EXPECTED_GIT_COMMIT,
            "--shard-name",
            "fixture",
            "--fail-on-existing",
        ]
    )
    rows = _rows()
    contract = _contract()
    _write_input_records(data_root, rows, contract)
    seen = {"model_loads": 0, "write_identity": None, "input_records": None}
    fake_module = types.ModuleType("features.wan.wan_feature")

    def fake_model_loader(**kwargs):
        seen["model_loads"] += 1
        return object()

    fake_module.get_wan_featurizer = fake_model_loader
    monkeypatch.setitem(sys.modules, "features.wan.wan_feature", fake_module)
    monkeypatch.setattr(extend, "REPO_ROOT", frozen_runtime)
    monkeypatch.setattr(extend, "MODEL_SNAPSHOT", model_id)
    monkeypatch.setattr(extend, "ensure_pinned_base_extractor", lambda contract: None)
    monkeypatch.setattr(
        extend,
        "load_manifest_selection",
        lambda manifest, selection, contract: (rows, {"scene": rows}),
    )
    monkeypatch.setattr(extend, "validate_source_audit", lambda *args: {})
    monkeypatch.setattr(
        extend,
        "validate_source_h1_scene",
        lambda *args: extend.SourceScene(
            scene="scene",
            source_root=source_root.resolve(),
            target_ids=[],
            tensors={},
            artifact_sha256={},
            feature_sha256={},
            index_sha256="fixture",
        ),
    )
    monkeypatch.setattr(extend, "assert_source_unchanged", lambda source: None)
    monkeypatch.setattr(base, "load_frame", lambda *args: object())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(
        base,
        "forward_target",
        lambda model, image, layers, seed, key: {
            layer: torch.zeros(contract.feature_shape) for layer in layers
        },
    )

    def fake_write_scene(*arguments):
        seen["input_records"] = arguments[-2]
        seen["write_identity"] = arguments[-1]
        return []

    monkeypatch.setattr(extend, "write_scene", fake_write_scene)
    extend.extract_shard(args, contract)
    assert seen["model_loads"] == 1
    assert seen["write_identity"].repo_root == str(repo.resolve())
    assert len(seen["input_records"]) == 8
    assert all(
        set(record) == extend.INPUT_IMAGE_RECORD_FIELDS
        for record in seen["input_records"].values()
    )
    claim = json.loads(
        (out_root / "provenance/shards/fixture.claim.json").read_text()
    )
    assert claim["runtime"]["git_commit"] == extend.EXPECTED_GIT_COMMIT
    assert claim["git_commit"] == extend.EXPECTED_GIT_COMMIT
    assert claim["git_commit_verified"] is True
    assert claim["runtime"]["git_commit_verified"] is True
    assert claim["runtime"]["repo_root"] == str(repo.resolve())
    assert claim["runtime"]["git_commit_verification_source"].startswith("direct:")


def test_shard_marker_audit_rejects_minimal_and_foreign_markers(tmp_path):
    out_root = tmp_path / "out"
    shard_dir = out_root / "provenance/shards"
    shard_dir.mkdir(parents=True)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("fixture\n")
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_audit = tmp_path / "audit.json"
    source_audit.write_text("{}\n")
    output_artifact = out_root / "wan/scannetpp/scene/artifact"
    output_artifact.parent.mkdir(parents=True)
    output_artifact.write_text("output")
    source_artifact = source_root / "artifact"
    source_artifact.write_text("source")
    output_inventory = [
        {"path": str(output_artifact.resolve()), "sha256": extend.sha256(output_artifact)}
    ]
    source_inventory = [
        {
            "path": str(source_artifact.resolve()),
            "sha256": extend.sha256(source_artifact),
            "scene": "scene",
        }
    ]
    output_inventory_path = shard_dir / "fixture.output_inventory.json"
    source_inventory_path = shard_dir / "fixture.source_inventory.json"
    base.atomic_json(
        output_inventory_path, {"schema_version": 1, "artifacts": output_inventory}
    )
    base.atomic_json(
        source_inventory_path, {"schema_version": 1, "artifacts": source_inventory}
    )
    contract = _contract()
    claim = {
        "status": "CLAIMED",
        "shard": "fixture",
        "scenes": ["scene"],
        "manifest_sha256": contract.manifest_sha256,
        "manifest": str(manifest.resolve()),
        "source_h1_root": str(source_root.resolve()),
        "source_h1_audit": str(source_audit.resolve()),
        "source_h1_audit_sha256": contract.source_audit_sha256,
        "selection_file": None,
        "selection_file_sha256": None,
        "wrapper_sha256": extend.wrapper_sha256(),
        "repo_root": _git_identity().repo_root,
        "git_commit": _git_identity().commit,
        "git_commit_verified": True,
        "git_commit_verification_source": _git_identity().verification_source,
        "runtime": {
            "repo_root": _git_identity().repo_root,
            "git_commit": _git_identity().commit,
            "git_commit_verified": True,
            "git_commit_verification_source": _git_identity().verification_source,
        },
        "created_unix": 1.0,
    }
    completion = {
        "status": "COMPLETE",
        "shard": "fixture",
        "manifest_sha256": contract.manifest_sha256,
        "selection_file": None,
        "selection_file_sha256": None,
        "wrapper_sha256": extend.wrapper_sha256(),
        "scene_count": 1,
        "sample_count": 4,
        "new_forward_count": 8,
        "output_inventory": str(output_inventory_path.resolve()),
        "output_inventory_sha256": extend.sha256(output_inventory_path),
        "source_inventory": str(source_inventory_path.resolve()),
        "source_inventory_sha256": extend.sha256(source_inventory_path),
        "completed_unix": 2.0,
    }
    base.atomic_json(shard_dir / "fixture.claim.json", claim)
    base.atomic_json(shard_dir / "fixture.complete.json", completion)
    arguments = (
        out_root,
        manifest,
        source_root,
        source_audit,
        {"scene": _rows()},
        {"scene": "fixture"},
        {"scene": output_inventory},
        {"scene": source_inventory},
        contract,
        _git_identity(),
    )
    extend.validate_shard_markers(*arguments)

    base.atomic_json(shard_dir / "fixture.complete.json", {"status": "COMPLETE"})
    with pytest.raises(extend.ContractError, match="missing required field"):
        extend.validate_shard_markers(*arguments)
    base.atomic_json(shard_dir / "fixture.complete.json", completion)
    (shard_dir / "foreign.complete.json").write_text("{}\n")
    with pytest.raises(extend.ContractError, match="marker set mismatch"):
        extend.validate_shard_markers(*arguments)
