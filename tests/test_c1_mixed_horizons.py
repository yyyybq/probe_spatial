"""C1 mixed targets must stay independent, canonical, and equally represented."""

import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from torch.utils.data import default_collate

from vidfm3d.data.components.inscene15k_dataset import InsScene15KDataset
from vidfm3d.eval_diag import _record_action_dynamics, _summarize
from vidfm3d.models.components.probe_action_dynamics import ActionDynamicsProbe
from vidfm3d.models.probe_ext_module import ProbeExtensionLitModule


WINDOW = [0, 18, 25, 29, 34, 37, 41, 45, 48, 49, 53, 57]


def dataset(tmp_path, monkeypatch, horizons=(1, 2, 4), split="train", window=None,
            count=1, seed=42):
    for key in ("INSCENE_MANIFEST", "INSCENE_OWNER_CATALOG", "INSCENE_STREAMING_EXCLUSIONS",
                "INSCENE_SPLIT_MANIFEST", "INSCENE_LOCAL_SCENE_ROOT"):
        monkeypatch.delenv(key, raising=False)
    manifest = tmp_path / "samples.jsonl"
    row = {
        "sample_id": "scene_w000_p8", "scene_id": "scene", "scene_dir": "/synthetic/scene",
        "source": "scannetpp", "split": split, "num_frames": 100,
        "streaming_window_id": 0, "streaming_tail": 7, "streaming_tail_raw": 45,
        "streaming_indices": WINDOW[:8],
        "streaming_window_indices": WINDOW if window is None else window,
    }
    rows = [row] if count == 1 else [
        {**row, "sample_id": f"scene_w{i:03d}_p8", "streaming_window_id": i}
        for i in range(count)
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    ds = InsScene15KDataset(
        root=str(tmp_path), sources=[], split=split, scene_manifest=str(manifest),
        streaming_prefix=True, diag_action=True, action_horizons=list(horizons), seed=seed,
        prefix_min_len=8, prefix_max_len=8, num_views=8, target_h=2, target_w=2,
    )
    selected = []

    def geometry(scene, indices):
        selected.append(indices.tolist())
        extrinsics = torch.eye(4)[:3].repeat(len(indices), 1, 1)
        extrinsics[:, 0, 3] = indices.float() / 100
        return torch.ones(len(indices), 2, 2), torch.eye(3).repeat(len(indices), 1, 1), extrinsics

    ds._load_camera_depth_scene = geometry
    ds._resize_depth_to_target = lambda depth, intrinsics: (depth, intrinsics)
    ds._load_streaming_prefix_feat = lambda scene, indices: (
        torch.ones(8, 1, 1, 8), torch.arange(8)
    )
    ds._load_target_feats = lambda scene, indices, count: [
        torch.full((1, 1, 8), float(index)) for index in indices
    ]
    return ds, selected


def test_validation_has_three_distinct_correct_targets(tmp_path, monkeypatch):
    ds, selected = dataset(tmp_path, monkeypatch, split="val")
    assert len(ds) == 3
    rows = [ds[i] for i in range(3)]
    assert selected == [[45, 48], [45, 49], [45, 57]]
    assert [r["action_horizon"].item() for r in rows] == [1, 2, 4]
    assert [r["raw_frame_offset"].item() for r in rows] == [3, 4, 12]
    assert [r["target_feat"][0, 0, 0].item() for r in rows] == [48, 49, 57]
    assert len({r["sample_id"] for r in rows}) == 3
    assert {r["manifest_sample_id"] for r in rows} == {"scene_w000_p8"}
    assert torch.equal(rows[0]["input_feat"], rows[2]["input_feat"])
    assert rows[0]["action"][6] < rows[1]["action"][6] < rows[2]["action"][6]
    assert ds[2]["sample_id"] == rows[2]["sample_id"]


@pytest.mark.parametrize("count", [2, 9, 10])
def test_training_keeps_original_size_with_balanced_reproducible_assignment(tmp_path, monkeypatch, count):
    ds, _ = dataset(tmp_path, monkeypatch, count=count)
    assert len(ds) == count
    rows = [ds[i] for i in range(count)]
    counts = [sum(r["action_horizon"].item() == h for r in rows) for h in (1, 2, 4)]
    assert sum(counts) == count
    assert max(counts) - min(counts) <= 1
    assert len({r["manifest_sample_id"] for r in rows}) == count
    expected_target = {1: 48, 2: 49, 4: 57}
    for row in rows:
        assert row["target_frame_idx"].item() == expected_target[row["action_horizon"].item()]
    same, _ = dataset(tmp_path, monkeypatch, count=count)
    assert [same[i]["sample_id"] for i in range(count)] == [r["sample_id"] for r in rows]
    ds.set_epoch(5)
    # Access order and epoch changes do not alter the frozen partition.
    assert [ds[i]["sample_id"] for i in reversed(range(count))] == [r["sample_id"] for r in reversed(rows)]
    h1, _ = dataset(tmp_path, monkeypatch, count=count, horizons=(1,))
    assert len(ds) == len(h1)


def test_training_partition_seed_changes_assignment_not_budget(tmp_path, monkeypatch):
    first, _ = dataset(tmp_path, monkeypatch, count=30, seed=42)
    second, _ = dataset(tmp_path, monkeypatch, count=30, seed=43)
    a = [s["c1_target_horizon"] for s in first.scenes]
    b = [s["c1_target_horizon"] for s in second.scenes]
    assert a != b
    assert sorted(a) == sorted(b) == [1] * 10 + [2] * 10 + [4] * 10


def test_single_horizon_keeps_one_example(tmp_path, monkeypatch):
    ds, selected = dataset(tmp_path, monkeypatch, horizons=(2,))
    assert len(ds) == 1
    row = ds[0]
    assert selected == [[45, 49]]
    assert row["action_horizon"].item() == 2


def test_mixed_horizons_reject_short_windows(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="covering all horizons"):
        dataset(tmp_path, monkeypatch, window=WINDOW[:10])


def test_missing_horizon_target_is_not_silently_dropped(tmp_path, monkeypatch):
    ds, _ = dataset(tmp_path, monkeypatch, count=3)
    loader = ds._load_target_feats
    ds._load_target_feats = lambda scene, indices, n: [
        None if index == 49 else loader(scene, [index], n)[0] for index in indices
    ]
    index = next(i for i, scene in enumerate(ds.scenes) if scene["c1_target_horizon"] == 2)
    with pytest.raises(FileNotFoundError, match="horizon=2"):
        ds[index]


def test_mixed_batch_trains_and_eval_keeps_canonical_horizons(tmp_path, monkeypatch):
    ds, _ = dataset(tmp_path, monkeypatch, count=3)
    batch = default_collate([ds[i] for i in range(3)])
    probe = ActionDynamicsProbe(in_channels=8, hidden_dim=16, num_layers=1,
                                num_heads=4, max_seq_len=8, dropout=0)
    module = ProbeExtensionLitModule(probe=probe, probe_type="action_dynamics",
                                     optimizer=None, scheduler=None)
    loss, metrics = module._step_action_dynamics(batch, train=True)
    assert torch.isfinite(loss)
    assert metrics["dyn_n_valid"] == 3
    loss.backward()
    assert probe.action_proj[0].weight.grad is not None
    assert torch.isfinite(probe.action_proj[0].weight.grad).all()
    module.eval()
    records = _record_action_dynamics(module, batch)
    assert sorted((r["action_horizon"], r["raw_frame_offset"]) for r in records) == [(1, 3), (2, 4), (4, 12)]


def test_summary_uses_separate_horizon_pools_and_equal_macro_weight():
    records = []
    # Unequal counts deliberately distinguish macro from sample-weighted mean.
    for horizon, count in [(1, 1), (2, 2), (4, 3)]:
        for index in range(count):
            target = torch.eye(4)[index]
            prediction = target if horizon == 1 else -target
            records.append({
                "action_horizon": horizon, "raw_frame_offset": horizon * 10,
                "pred": prediction, "target": target,
                "pred_no_action": torch.zeros(4),
                "pred_shuffled_action": prediction, "last_observation": prediction,
            })
    summary = _summarize(records, "action_dynamics", compute_calibrated=False)
    assert [summary[f"h{h}_n"] for h in (1, 2, 4)] == [1, 2, 3]
    assert summary["macro_horizon_mean_cos"] == pytest.approx(-1 / 3)
    assert summary["mean_cos"] == pytest.approx(-2 / 3)
    assert "h10_n" not in summary
    legacy = _summarize([{k: v for k, v in records[0].items() if k != "action_horizon"}],
                        "action_dynamics", compute_calibrated=False)
    assert "n_horizons" not in legacy


def test_all_mixed_configs_compose_with_separate_job_identity(monkeypatch):
    monkeypatch.setenv("INSCENE_DATA_ROOT", "/synthetic/data")
    monkeypatch.setenv("INSCENE_STREAMING_FEAT_ROOT", "/synthetic/features")
    root = Path(__file__).resolve().parents[1]
    configs = sorted((root / "configs/experiment/inscene15k_streaming").glob("action_dynamics_*_h124_v1.yaml"))
    assert len(configs) == 10
    with initialize_config_dir(config_dir=str(root / "configs"), version_base="1.3"):
        for path in configs:
            cfg = compose(config_name="train", overrides=[f"experiment=inscene15k_streaming/{path.stem}"])
            assert list(cfg.action_horizons) == [1, 2, 4]
            assert cfg.job_name == path.stem
            assert cfg.autoresume is False
            assert "action_horizons=[1, 2, 4]" in cfg.data.data_module.train_datasets[0]
            assert "action_horizons=[1, 2, 4]" in cfg.data.data_module.validation_datasets[0]
