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


def test_b1_supports_linear_mlp_and_transformer_decoders():
    feat = torch.randn(2, 4, 3, 5, 8)
    object_mask = torch.zeros(2, 4, 3, 5, dtype=torch.bool)
    object_mask[:, :3, 1, 2] = True
    for decoder_type in ("linear", "mlp", "transformer"):
        probe = EgoBeliefProbe(
            in_channels=8, hidden_dim=16, num_layers=1, num_heads=4,
            decoder_type=decoder_type, num_frames=4,
        )
        assert probe(feat, object_mask).shape == (2, 3)
        if decoder_type == "linear":
            assert isinstance(probe.flat_readout, torch.nn.Linear)


def test_b1_ignores_frames_where_conditioned_object_is_hidden():
    probe = EgoBeliefProbe(in_channels=8, hidden_dim=16, num_layers=1, num_heads=4)
    feat = torch.randn(1, 4, 3, 5, 8)
    object_mask = torch.zeros(1, 4, 3, 5, dtype=torch.bool)
    object_mask[:, :3, 1, 2] = True
    pooled = probe._mask_pool(feat, object_mask)
    assert torch.equal(pooled[:, -1], torch.zeros_like(pooled[:, -1]))


def test_b1_final_global_feature_is_an_input():
    probe = EgoBeliefProbe(
        in_channels=2, hidden_dim=8, num_layers=1, num_heads=2,
        decoder_type="linear", num_frames=4, use_final_global_feature=True,
    )
    with torch.no_grad():
        probe.flat_readout.weight.zero_()
        probe.flat_readout.bias.zero_()
        # Linear flat layout is object tokens, final-global token, visibility bits.
        probe.flat_readout.weight[0, 4 * 2] = 1.0

    feat = torch.zeros(1, 4, 2, 2, 2)
    object_mask = torch.zeros(1, 4, 2, 2, dtype=torch.bool)
    object_mask[:, :3, 0, 0] = True
    out0 = probe(feat, object_mask)
    feat[:, -1, :, :, 0] = 3.0
    out1 = probe(feat, object_mask)
    assert out0[0, 0] == 0
    assert out1[0, 0] == 3


def test_b2_has_object_query_without_current_role_embedding():
    probe = EgoBeliefProbeV2(
        in_channels=8, hidden_dim=16, num_layers=1, num_heads=4,
        max_seq_len=4, max_h=4, max_w=6, n_az_bins=4, n_el_bins=2,
    )
    out = probe(torch.randn(2, 4, 3, 5, 8), torch.randn(2, 8))
    assert out["logits"].shape == (2, 4, 2)
    assert not hasattr(probe, "role_embed")


def test_b2_supports_linear_mlp_and_transformer_decoders():
    feat = torch.randn(2, 4, 3, 5, 8)
    query = torch.randn(2, 8)
    for decoder_type in ("linear", "mlp", "transformer"):
        probe = EgoBeliefProbeV2(
            in_channels=8, hidden_dim=16, num_layers=1, num_heads=4,
            decoder_type=decoder_type, num_frames=4, max_seq_len=4,
            max_h=4, max_w=6, n_az_bins=4, n_el_bins=2,
        )
        out = probe(feat, query)
        assert out["logits"].shape == (2, 4, 2)
        assert out["log_dist"].shape == (2,)
        if decoder_type == "linear":
            assert isinstance(probe.flat_readout, torch.nn.Linear)


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


def test_c2_fast_path_uses_context_segment_not_normal_or_isolated_anchor():
    ds = InsScene15KDataset.__new__(InsScene15KDataset)
    ds._rng = np.random.default_rng(0)
    ds.diag_counterfactual = False
    ds.counterfactual_min_overlap = 0.05
    ds.vfm_name = "wan"
    ds.context_len = 4
    ds._load_camera_depth_scene = lambda scene, sel: (
        torch.ones(2, 2, 2),
        torch.eye(3).repeat(2, 1, 1),
        torch.eye(4)[:3].repeat(2, 1, 1),
    )
    ds._resize_depth_to_target = lambda depth, intr: (depth, intr)
    normal_feat = torch.full((2, 1, 1, 3), -1.0)
    ds._load_vfm_feat_for_selection = lambda scene, n, sel: (
        normal_feat.clone(),
        torch.arange(2),
    )

    def load_isolated(scene, indices, num_frames):
        values = {0: 7.0, 1: 9.0}
        return [torch.full((1, 1, 3), values[int(i)]) for i in indices]

    ds._load_target_feats = load_isolated
    ds._load_context_segment_feat = lambda scene, start, tail, select_indices=None: (
        torch.full((1, 1, 1, 3), 11.0),
        torch.arange(1),
    )
    out = ds._getitem_feature_action_diag(
        {"source": "infinigen", "scene_dir": "/scene"}, 2, torch.tensor([0, 1])
    )
    assert torch.equal(out["vfm_feat"][0], normal_feat[0])
    assert torch.equal(out["input_feat_seq"][0], torch.full((1, 1, 3), 11.0))
    assert torch.equal(out["target_feat_seq"][0], torch.full((1, 1, 3), 9.0))


def test_context_select_indices_are_fixed_length_with_tail_padding():
    ds = InsScene15KDataset.__new__(InsScene15KDataset)
    ds.context_len = 5
    assert torch.equal(
        ds._context_select_indices(2, 6),
        torch.tensor([2, 3, 4, 5, 6]),
    )
    assert torch.equal(
        ds._context_select_indices(0, 2),
        torch.tensor([0, 1, 2, 2, 2]),
    )


def test_inscene_validation_sampling_defaults_to_fixed_seed():
    with tempfile.TemporaryDirectory() as tmp:
        val = InsScene15KDataset(root=tmp, sources=[], split="val")
        train = InsScene15KDataset(root=tmp, sources=[], split="train")
        explicit_zero = InsScene15KDataset(root=tmp, sources=[], split="train", seed=0)
        assert val.seed == 0
        assert train.seed is None
        assert explicit_zero.seed == 0


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
