import math
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from vidfm3d.eval_diag import _summarize
from vidfm3d.models.components.probe_ego_belief import EgoBeliefProbe
from vidfm3d.models.components.probe_ego_belief_v2 import EgoBeliefProbeV2
from vidfm3d.data.components.inscene15k_dataset import InsScene15KDataset
from vidfm3d.train import _guard_probe_world_size
from omegaconf import OmegaConf


def test_b1_is_object_conditioned_without_pose():
    probe = EgoBeliefProbe(in_channels=8, hidden_dim=16, num_layers=1, num_heads=4)
    feat = torch.randn(2, 4, 3, 5, 8)
    object_mask = torch.zeros(2, 4, 3, 5, dtype=torch.bool)
    object_mask[:, :3, 1, 2] = True
    assert probe(feat, object_mask).shape == (2, 3)
    assert not any("pose" in name for name, _ in probe.named_parameters())


def test_b1_retains_last_reference_frame_when_object_is_hidden():
    probe = EgoBeliefProbe(in_channels=8, hidden_dim=16, num_layers=1, num_heads=4)
    feat = torch.randn(1, 4, 3, 5, 8)
    object_mask = torch.zeros(1, 4, 3, 5, dtype=torch.bool)
    object_mask[:, :3, 1, 2] = True
    pooled = probe._mask_pool(feat, object_mask)
    assert torch.allclose(pooled[:, -1], feat[:, -1].mean(dim=(1, 2)))


def test_b2_has_object_query_without_current_role_embedding():
    probe = EgoBeliefProbeV2(
        in_channels=8, hidden_dim=16, num_layers=1, num_heads=4,
        max_seq_len=4, max_h=4, max_w=6, n_az_bins=4, n_el_bins=2,
    )
    out = probe(torch.randn(2, 4, 3, 5, 8), torch.randn(2, 8))
    assert out["logits"].shape == (2, 4, 2)
    assert not hasattr(probe, "role_embed")


def test_b1_summary_wraps_azimuth():
    records = [{
        "scene_path": "processed_infinigen/scene",
        "polar_pred": torch.tensor([math.pi - 0.01, 0.0, 0.0]),
        "polar_gt": torch.tensor([-math.pi + 0.01, 0.0, 0.0]),
    }]
    assert _summarize(records, "ego_belief")["mean_az_err_deg"] < 2.0


def test_a3_reports_paired_ranking_and_auc():
    records = [{
        "scene_path": "processed_scannetpp_v2/scene",
        "prob_normal": 0.2,
        "prob_shuffled": 0.8,
        "valid": True,
    }]
    summary = _summarize(records, "abnormal")
    assert summary["paired_rank_acc"] == 1.0
    assert summary["roc_auc"] == 1.0


def test_c1_target_lookup_requires_exact_frame():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cache = root / "wan" / "infinigen" / "scene"
        cache.mkdir(parents=True)
        save_file({"feat": torch.stack([torch.zeros(2, 2, 3), torch.ones(2, 2, 3)])},
                  cache / "feature.sft")
        np.save(cache / "target_indices.npy", np.array([1, 5], dtype=np.int64))
        ds = InsScene15KDataset.__new__(InsScene15KDataset)
        ds.target_feat_root = str(root)
        ds.vfm_name = "wan"
        ds._target_index_cache = {}
        ds._feat_scene_name = lambda scene: "scene"
        ds._feat_filename = lambda: "feature.sft"
        scene = {"scene_dir": "/data/scene", "source": "infinigen"}
        assert torch.equal(ds._load_target_feat(scene, 5, 10), torch.ones(2, 2, 3))
        assert ds._load_target_feat(scene, 4, 10) is None


def test_large_probe_ddp_requires_explicit_override():
    cfg = OmegaConf.create({
        "model": {"_target_": "vidfm3d.models.probe_ext_module.ProbeExtensionLitModule"},
        "trainer": {"devices": 100, "num_nodes": 1},
        "allow_large_probe_ddp": False,
    })
    try:
        _guard_probe_world_size(cfg)
    except ValueError as exc:
        assert "job array" in str(exc)
    else:
        raise AssertionError("100-way probe DDP was not rejected")
