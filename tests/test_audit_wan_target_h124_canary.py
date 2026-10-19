import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from scripts import audit_wan_target_h124_canary as audit


def _sha(path: Path) -> str:
    return audit.sha256(path)


def _rows(scene: str = "scene"):
    rows = []
    for window_id, start in enumerate((0, 8, 16, 24)):
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
                "valid_frames": [f"frame_{index:04d}.jpg" for index in range(40)],
                # Deliberately false: the independent auditor must ignore it.
                "target_raw_id": 39,
            }
        )
    return rows


def _artifact(path: Path, value, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".npy":
        with path.open("wb") as handle:
            np.save(handle, value, allow_pickle=False)
        shape = list(value.shape)
        dtype = str(value.dtype)
        extra = {"shape": shape, "dtype": dtype}
    else:
        save_file({"feat": value.contiguous()}, str(path))
        extra = {
            "tensors": {
                "feat": {"shape": list(value.shape), "dtype": "torch.float16"}
            }
        }
    sidecar = {**metadata, **extra, "sha256": _sha(path), "size_bytes": path.stat().st_size}
    Path(f"{path}.manifest.json").write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")


def _fixture(tmp_path: Path):
    rows = _rows()
    base_extractor = tmp_path / "extract_wan_manifest_exact.py"
    extractor = tmp_path / "extract_wan_target_h124_extend_exact.py"
    base_extractor.write_text("# immutable base fixture\n")
    extractor.write_text("# wrapper fixture\n")
    model_id = tmp_path / "model"
    model_id.mkdir()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("fixture\n")
    source_audit = tmp_path / "source_audit.json"
    source_audit.write_text("{}\n")
    source_root = tmp_path / "source"
    out_root = tmp_path / "output"
    contract = audit.AuditContract(
        base_extractor_sha256=_sha(base_extractor),
        source_audit_sha256=_sha(source_audit),
        layers=(0, 5),
        feature_shape=(2, 2, 3),
        manifest_rows=4,
        manifest_scenes=1,
        split_counts=(("train", 4),),
        canary_scenes=1,
        canary_rows=4,
    )
    mapping = audit.derive_scene_mapping("scene", rows, contract)
    all_ids = sorted(mapping)
    h1_ids = [raw_id for raw_id in all_ids if mapping[raw_id]["horizon"] == 1]
    data_root = tmp_path / "data"
    extracted_images = {}
    for raw_id in all_ids:
        identity = mapping[raw_id]
        if identity["horizon"] == 1:
            continue
        image = (
            data_root / "processed_scannetpp_v2/processed_scannetpp_v2"
            / "scene/images" / identity["frame_name"]
        )
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(f"image-{raw_id}".encode())
        extracted_images[str(raw_id)] = {
            "path": str(image.absolute()),
            "realpath": str(image.resolve()),
            "is_symlink": False,
            "size_bytes": image.stat().st_size,
            "sha256": _sha(image),
            "scene": "scene",
            "sample_id": identity["sample_id"],
            "horizon": identity["horizon"],
            "raw_id": raw_id,
            "frame_name": identity["frame_name"],
        }
    source_common = audit.source_common_required("scene", rows, h1_ids, model_id, contract)
    source_scene = source_root / "wan/scannetpp/scene"
    _artifact(source_scene / "target_indices.npy", np.asarray(h1_ids, dtype=np.int64), source_common)
    output_common = audit.output_common_required(
        "scene", rows, mapping, manifest, source_root, source_audit,
        model_id, base_extractor, extractor, "a" * 40, extracted_images, contract,
    )
    output_common.update(
        git_dirty=False,
        git_commit_verification_source="fixture immutable git snapshot",
        legacy_config_discrepancy="fixture discrepancy annotation",
        source_h1_target_indices_sha256=_sha(source_scene / "target_indices.npy"),
    )
    output_scene = out_root / "wan/scannetpp/scene"
    _artifact(output_scene / "target_indices.npy", np.asarray(all_ids, dtype=np.int64), output_common)
    for layer in contract.layers:
        source_tensor = torch.stack(
            [torch.full(contract.feature_shape, raw_id + layer, dtype=torch.float16) for raw_id in h1_ids]
        )
        source_metadata = {
            **source_common,
            "target_frame_names": [mapping[raw_id]["frame_name"] for raw_id in h1_ids],
            "target_id_sources": {str(raw_id): mapping[raw_id]["source"] for raw_id in h1_ids},
            "sample_seeds": {str(raw_id): mapping[raw_id]["seed"] for raw_id in h1_ids},
        }
        source_path = source_scene / f"feature_t749_layer{layer}.sft"
        _artifact(source_path, source_tensor, source_metadata)
        source_by_raw = {raw_id: source_tensor[pos] for pos, raw_id in enumerate(h1_ids)}
        output_tensor = torch.stack(
            [
                source_by_raw.get(raw_id, torch.full(contract.feature_shape, 1000 + raw_id + layer, dtype=torch.float16))
                for raw_id in all_ids
            ]
        )
        _artifact(
            output_scene / f"feature_t749_layer{layer}.sft",
            output_tensor,
            {**output_common, "source_h1_feature_sha256": _sha(source_path)},
        )
    return (
        rows, out_root, source_root, source_audit, manifest, model_id,
        base_extractor, extractor, "a" * 40, extracted_images, contract,
    )


def test_independent_scene_audit_derives_h124_and_requires_exact_h1(tmp_path):
    fixture = _fixture(tmp_path)
    result = audit.audit_scene("scene", *fixture)
    assert result["target_ids"] == [2, 3, 5, 10, 11, 13, 18, 19, 21, 26, 27, 29]
    assert result["horizon_counts"] == {1: 4, 2: 4, 4: 4}
    assert result["h1_exact_rows"] == 8

    rows, out_root, source_root, source_audit, manifest, model_id, base_extractor, extractor, git_commit, extracted_images, contract = fixture
    path = out_root / "wan/scannetpp/scene/feature_t749_layer0.sft"
    from safetensors.torch import load_file

    value = load_file(str(path))["feat"]
    value[all_ids_index(rows, contract)] += 1
    metadata = json.loads(Path(f"{path}.manifest.json").read_text())
    metadata.pop("sha256")
    metadata.pop("size_bytes")
    metadata.pop("tensors")
    _artifact(path, value, metadata)
    with pytest.raises(audit.AuditError, match="h1 tensor row is not exact"):
        audit.audit_scene(
            "scene", rows, out_root, source_root, source_audit, manifest,
            model_id, base_extractor, extractor, git_commit, extracted_images, contract,
        )


def all_ids_index(rows, contract):
    mapping = audit.derive_scene_mapping("scene", rows, contract)
    all_ids = sorted(mapping)
    first_h1 = next(raw_id for raw_id in all_ids if mapping[raw_id]["horizon"] == 1)
    return all_ids.index(first_h1)


def test_report_is_exclusive(tmp_path):
    path = tmp_path / "audit.json"
    audit.exclusive_report(path, {"ok": True})
    with pytest.raises(audit.AuditError, match="refusing existing report"):
        audit.exclusive_report(path, {"ok": False})


def test_output_rejects_unknown_git_verification_source(tmp_path):
    fixture = _fixture(tmp_path)
    rows, out_root, source_root, source_audit, manifest, model_id, base_extractor, extractor, git_commit, extracted_images, contract = fixture
    sidecar = out_root / "wan/scannetpp/scene/target_indices.npy.manifest.json"
    metadata = json.loads(sidecar.read_text())
    metadata["git_commit_verification_source"] = "unknown"
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    with pytest.raises(audit.AuditError, match="verification_source is absent"):
        audit.audit_scene(
            "scene", rows, out_root, source_root, source_audit, manifest,
            model_id, base_extractor, extractor, git_commit, extracted_images, contract,
        )


def test_output_rejects_input_image_provenance_mismatch(tmp_path):
    fixture = _fixture(tmp_path)
    (
        rows, out_root, source_root, source_audit, manifest, model_id,
        base_extractor, extractor, git_commit, extracted_images, contract,
    ) = fixture
    sidecar = out_root / "wan/scannetpp/scene/target_indices.npy.manifest.json"
    metadata = json.loads(sidecar.read_text())
    raw_id = sorted(metadata["extracted_input_images_by_raw_id"])[0]
    metadata["extracted_input_images_by_raw_id"][raw_id]["sha256"] = "0" * 64
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    with pytest.raises(audit.AuditError, match="extracted_input_images_by_raw_id"):
        audit.audit_scene(
            "scene", rows, out_root, source_root, source_audit, manifest,
            model_id, base_extractor, extractor, git_commit, extracted_images, contract,
        )


def test_final_output_rehash_rejects_mutation(tmp_path):
    fixture = _fixture(tmp_path)
    result = audit.audit_scene("scene", *fixture)
    sidecar = Path(result["output_inventory"][1]["path"])
    sidecar.write_text(sidecar.read_text() + " ")
    with pytest.raises(audit.AuditError, match="final output (SHA|size) changed"):
        audit.verify_final_output_inventory(result["output_inventory"], fixture[1])


def test_output_scene_rejects_symlink_entry(tmp_path):
    fixture = _fixture(tmp_path)
    path = fixture[1] / "wan/scannetpp/scene/target_indices.npy.manifest.json"
    outside = tmp_path / "outside-sidecar.json"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(audit.AuditError, match="regular non-symlink contained"):
        audit.audit_scene("scene", *fixture)


def test_frozen_runtime_uses_explicit_external_repo_for_protected(
    tmp_path, monkeypatch
):
    frozen_runtime = tmp_path / "bundle/runtime_tree"
    (frozen_runtime / "scripts").mkdir(parents=True)
    external_repo = tmp_path / "real_repo"
    (external_repo / ".git").mkdir(parents=True)
    protected_file = external_repo / "experiments/registry.csv"
    protected_file.parent.mkdir(parents=True)
    protected_file.write_text("id,status\nfixture,COMPLETE\n")
    protected_before = tmp_path / "bundle/inputs/protected_before.json"
    protected_before.parent.mkdir(parents=True)
    protected_before.write_text(
        json.dumps({"experiments/registry.csv": _sha(protected_file)}) + "\n"
    )

    # Simulate importing/running the auditor from bundle/runtime_tree/scripts.
    # Protected resolution must still use the external repo, not this value.
    monkeypatch.setattr(audit, "REPO_ROOT", frozen_runtime, raising=False)
    key, resolved = audit.frozen_repo_root(
        {"external_roots": {"repo_root": str(external_repo)}}
    )
    assert key == "external_roots.repo_root"
    assert resolved == external_repo
    result = audit.audit_protected(protected_before, resolved)
    assert result["ok"] is True
    assert result["checked"][0]["path"] == str(protected_file)

    relative_inventory = tmp_path / "relative_inventory.json"
    relative_inventory.write_text(
        json.dumps(
            {
                "artifacts": [
                    {"path": "experiments/registry.csv", "sha256": _sha(protected_file)}
                ]
            }
        )
        + "\n"
    )
    with pytest.raises(audit.AuditError, match="explicit frozen base"):
        audit.parse_inventory(relative_inventory)
