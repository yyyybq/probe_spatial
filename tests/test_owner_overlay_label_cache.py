"""Owner overlay, C-probe label cache and probe-batch transfer in the main tree.

These paths were restored from the frozen canonical-V2 training packs
(code/main_v2_probe_v2_r7_single_v3, code/main_v2_probe_v2_aetherfix_r1).
"""

import hashlib
import json
import sqlite3
import struct
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from vidfm3d.data.components.inscene15k_dataset import InsScene15KDataset
from vidfm3d.utils import probe_owner_overlay
from vidfm3d.utils.owner_catalog import OwnerCatalogError

PREFIX = "aoss://baiqiao/InsScene-15K/"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _header_sha(path: Path, envelope: bool) -> str:
    raw = path.read_bytes()
    size = struct.unpack("<Q", raw[:8])[0]
    return hashlib.sha256(raw[: 8 + size] if envelope else raw[8 : 8 + size]).hexdigest()


def _db(path: Path, table: str, columns: list[str], rows: list[dict]) -> None:
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
    con.execute("INSERT INTO metadata VALUES ('status', 'READY')")
    con.execute(f"CREATE TABLE {table} ({', '.join(columns)})")
    for row in rows:
        con.execute(
            f"INSERT INTO {table} ({', '.join(row)}) VALUES ({', '.join('?' for _ in row)})",
            list(row.values()),
        )
    con.commit()
    con.close()


@pytest.fixture()
def overlay(tmp_path, monkeypatch):
    """A two-table READY overlay whose payloads live in an exact local mirror."""
    mirror = tmp_path / "mirror"
    stream_rel = "FEAT_STREAM/wan/scannetpp/sceneA/window_0000/prefix_000004/feature.sft"
    target_rel = "FEAT_TARGET/wan/scannetpp/sceneA/feature.sft"
    for rel, tensor in (
        (stream_rel, torch.arange(4 * 2 * 2 * 3, dtype=torch.float32).reshape(4, 2, 2, 3)),
        (target_rel, torch.arange(3 * 2 * 2 * 3, dtype=torch.float32).reshape(3, 2, 2, 3) + 100),
    ):
        path = mirror / rel
        path.parent.mkdir(parents=True)
        save_file({"feat": tensor}, str(path))
    stream_file, target_file = mirror / stream_rel, mirror / target_rel
    common = ["vfm", "source", "scene_id", "layer", "timestep", "source_uri", "origin",
              "payload_size_bytes", "payload_header_sha256", "payload_sha256"]
    _db(tmp_path / "stream.sqlite", "streaming_owners", common + ["window_id", "prefix"], [{
        "vfm": "wan", "source": "scannetpp", "scene_id": "sceneA", "layer": 5, "timestep": 749,
        "window_id": 0, "prefix": 4, "source_uri": PREFIX + stream_rel, "origin": "gate_pass",
        "payload_size_bytes": stream_file.stat().st_size,
        "payload_header_sha256": _header_sha(stream_file, envelope=True),
        "payload_sha256": _sha256(stream_file),
    }])
    _db(tmp_path / "target.sqlite", "target_owners",
        common + ["target_raw_id", "status", "tensor_key", "source_row_index"], [
            {"vfm": "wan", "source": "scannetpp", "scene_id": "sceneA", "layer": 5, "timestep": 749,
             "target_raw_id": raw, "status": "READY", "tensor_key": "feat", "source_row_index": row,
             "source_uri": PREFIX + target_rel, "origin": "target_route_audit",
             "payload_size_bytes": target_file.stat().st_size,
             "payload_header_sha256": _header_sha(target_file, envelope=False),
             "payload_sha256": _sha256(target_file)}
            for row, raw in enumerate([10, 11, 12])
        ])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": "CANONICAL_V2_PROBE816_MIXED_OWNER_OVERLAY_V1", "status": "READY",
        "fail_closed": True,
        "streaming_database": {"path": str(tmp_path / "stream.sqlite"),
                               "sha256": _sha256(tmp_path / "stream.sqlite")},
        "target_database": {"path": str(tmp_path / "target.sqlite"),
                            "sha256": _sha256(tmp_path / "target.sqlite")},
    }))
    monkeypatch.setenv("PROBE_OWNER_OVERLAY_MANIFEST", str(manifest))
    monkeypatch.setenv("PROBE_OWNER_LOCAL_MIRROR_ROOT", str(mirror))
    probe_owner_overlay._STATE.update({"pid": None, "manifest": None})
    yield {"manifest": manifest, "mirror": mirror, "stream": stream_file, "target": target_file}
    probe_owner_overlay._STATE.update({"pid": None, "manifest": None})


def test_overlay_resolves_exact_owners_and_verifies_payloads(overlay):
    row = probe_owner_overlay.resolve_streaming("wan", "scannetpp", "sceneA", 5, 749, 0, 4)
    assert probe_owner_overlay.materialize_verified(row) == str(overlay["stream"].resolve())
    rows = probe_owner_overlay.resolve_targets("wan", "scannetpp", "sceneA", 5, 749, [12, 10])
    assert [r["source_row_index"] for r in rows] == [2, 0]
    with pytest.raises(probe_owner_overlay.OwnerOverlayError, match="no exact streaming owner"):
        probe_owner_overlay.resolve_streaming("wan", "scannetpp", "sceneA", 5, 749, 0, 8)
    with pytest.raises(probe_owner_overlay.OwnerOverlayError, match="no exact target owner"):
        probe_owner_overlay.resolve_targets("wan", "scannetpp", "sceneA", 5, 749, [13])


def test_overlay_fails_closed_on_changed_payload_or_database(overlay):
    row = probe_owner_overlay.resolve_streaming("wan", "scannetpp", "sceneA", 5, 749, 0, 4)
    data = bytearray(overlay["stream"].read_bytes())
    data[-1] ^= 1
    overlay["stream"].write_bytes(bytes(data))
    with pytest.raises(probe_owner_overlay.OwnerOverlayError, match="payload full SHA mismatch"):
        probe_owner_overlay.materialize_verified(row)
    manifest = json.loads(overlay["manifest"].read_text())
    manifest["streaming_database"]["sha256"] = "0" * 64
    overlay["manifest"].write_text(json.dumps(manifest))
    probe_owner_overlay._STATE.update({"pid": None, "manifest": None})
    with pytest.raises(probe_owner_overlay.OwnerOverlayError, match="SHA mismatch"):
        probe_owner_overlay.load_manifest()


def test_overlay_and_owner_catalog_are_mutually_exclusive(overlay, tmp_path, monkeypatch):
    monkeypatch.setenv("INSCENE_OWNER_CATALOG", str(tmp_path / "catalog.json"))
    with pytest.raises(OwnerCatalogError, match="mutually exclusive"):
        InsScene15KDataset(root=str(tmp_path), sources=[], split="train")


def _overlay_dataset() -> InsScene15KDataset:
    ds = InsScene15KDataset.__new__(InsScene15KDataset)
    ds.owner_overlay_enabled = True
    ds.vfm_name = "wan"
    ds.feature_layer = 5
    ds.feature_timestep = 749
    ds.feat_postfix = "_t749_layer5"
    ds.target_feat_root = None
    return ds


def test_overlay_streaming_prefix_and_target_features(overlay):
    ds = _overlay_dataset()
    ds.streaming_feat_root = None
    ds.root_vfm = None
    ds.owner_catalog = None
    ds.allow_missing_vfm = False
    ds._streaming_prefix_meta_cache = {}
    ds.scramble_feat = False
    ds.feat_pixalign = False
    ds.pose_only = False
    scene = {"source": "scannetpp", "scene_id": "sceneA", "scene_dir": "/x/sceneA",
             "streaming_window_id": 0, "streaming_indices": [0, 1, 2, 3]}
    ds._feat_scene_name = lambda scene_info: scene_info["scene_id"]
    ds._streaming_prefix_record = lambda scene_info: None
    feat, idx = ds._load_streaming_prefix_feat(scene, torch.arange(4))
    assert feat.shape == (4, 2, 2, 3) and torch.equal(idx, torch.arange(4))
    targets = ds._load_target_feats(scene, [11, 12], num_frames=40)
    assert torch.equal(targets[0], torch.arange(12, dtype=torch.float32).reshape(2, 2, 3) + 112)
    assert ds._load_anchor_isolated_feat(scene, 11, 40) is not None
    assert ds._load_anchor_isolated_feat(scene, 99, 40) is None


def _c_dataset(tmp_path: Path, *, path_integration: bool) -> InsScene15KDataset:
    ds = InsScene15KDataset.__new__(InsScene15KDataset)
    ds._rng = np.random.default_rng(0)
    ds.diag_action = not path_integration
    ds.diag_path_integration = path_integration
    ds.diag_counterfactual = False
    ds.counterfactual_min_overlap = 0.05
    ds.owner_overlay_enabled = False
    ds.vfm_name = "wan"
    ds.context_len = 4
    ds.target_h = ds.target_w = 2
    ds.streaming_prefix = False
    ds.streaming_target_from_prefix = False
    ds.action_horizons = [1]
    ds._label_cache_root = str(tmp_path / "labels")
    ds._label_cache_manifest_sha256 = "0" * 64
    ds._label_cache_version = "tstep-label-v1"
    ds._feat_scene_name = lambda scene_info: "scene"
    generator = torch.Generator().manual_seed(7)
    poses = torch.eye(4)[:3].repeat(3, 1, 1)
    poses[:, :, 3] = torch.randn(3, 3, generator=generator)
    depth = torch.rand(3, 2, 2, generator=generator) + 0.5
    ds._loads = 0

    def load_camera_depth(scene, sel):
        ds._loads += 1
        n = len(sel)
        return depth[:n].clone(), torch.eye(3).repeat(n, 1, 1), poses[:n].clone()

    ds._load_camera_depth_scene = load_camera_depth
    ds._resize_depth_to_target = lambda d, k: (d, k)
    ds._load_vfm_feat_for_selection = lambda scene, n, sel: (
        torch.zeros(len(sel), 1, 1, 3), torch.arange(len(sel)))
    ds._load_target_feats = lambda scene, indices, n: [
        torch.full((1, 1, 3), float(i)) for i in indices]
    ds._load_context_segment_feat = lambda scene, start, tail, select_indices=None: (
        torch.full((4, 1, 1, 3), 3.0), torch.arange(4))
    return ds


@pytest.mark.parametrize("path_integration", [False, True])
def test_c_probe_label_cache_hit_equals_recomputation(tmp_path, path_integration):
    ds = _c_dataset(tmp_path, path_integration=path_integration)
    scene = {"source": "scannetpp", "scene_id": "scene", "scene_dir": "/x/scene",
             "sample_id": "s0"}
    sel = torch.tensor([4, 5, 7]) if path_integration else torch.tensor([4, 5])
    first = ds._getitem_feature_action_diag(scene, 40, sel)
    ds._rng = np.random.default_rng(0)
    second = ds._getitem_feature_action_diag(scene, 40, sel)
    assert ds._loads == 1, "second call must be served by the label cache"
    assert set(first) == set(second)
    for key, value in first.items():
        if torch.is_tensor(value):
            assert value.dtype == second[key].dtype and torch.equal(value, second[key]), key
        else:
            assert value == second[key], key
    fields = {"target_extrinsics", "start_extrinsic", "action_horizons", "path_actions"}
    assert fields <= set(first) if path_integration else {"action", "action_horizon"} <= set(first)


def test_probe_batch_transfer_keeps_only_consumed_tensors():
    from vidfm3d.models.probe_ext_module import ProbeExtensionLitModule

    module = ProbeExtensionLitModule.__new__(ProbeExtensionLitModule)
    module.probe_type = "path_integration"
    captured = {}

    def parent(self, batch, device, dataloader_idx):
        captured.update(batch)
        return batch

    batch = {"input_feat_seq": torch.zeros(1), "path_actions": torch.zeros(1),
             "target_feat_seq": torch.zeros(1), "path_horizon_valid": torch.ones(1, dtype=torch.bool),
             "vfm_feat": torch.zeros(1), "target_extrinsics": torch.zeros(1), "scene_path": ["/x"]}
    original = ProbeExtensionLitModule.__mro__[1].transfer_batch_to_device
    try:
        ProbeExtensionLitModule.__mro__[1].transfer_batch_to_device = parent
        module.transfer_batch_to_device(batch, "cpu", 0)
    finally:
        ProbeExtensionLitModule.__mro__[1].transfer_batch_to_device = original
    assert set(captured) == {"input_feat_seq", "path_actions", "target_feat_seq",
                             "path_horizon_valid", "scene_path"}
    del batch["path_actions"]
    captured.clear()
    try:
        ProbeExtensionLitModule.__mro__[1].transfer_batch_to_device = parent
        module.transfer_batch_to_device(batch, "cpu", 0)
    finally:
        ProbeExtensionLitModule.__mro__[1].transfer_batch_to_device = original
    assert set(captured) == set(batch), "incomplete batches are moved unchanged"
