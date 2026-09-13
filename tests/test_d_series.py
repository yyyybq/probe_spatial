import math
import json

import torch

from features.vjepa2.native_predictor import (
    make_prefix_safe_clip,
    observation_to_tubelet,
    predict_future_tubelets,
    tubelet_token_indices,
)
from scripts.d_series.feature_store import load_verified, write_feature
from scripts.d_series.run_d2 import bin_index, motion_bin_thresholds, shuffled_motion_map

from vidfm3d.models.components.probe_layout_readout import (
    ExternalPooledPredictor,
    PooledDepthReadout,
    SpatialDepthReadout,
    TokenwiseLinearAdapter,
    parameter_count,
)
from vidfm3d.utils.layout_metrics import (
    coarse_layout_mae,
    depth_metrics,
    detail_gradient_mae,
    metric_depth_to_camera_pointmap,
    normalize_depth_per_sample,
    resize_depth_validity_weighted,
    scene_cluster_paired_bootstrap,
)


def test_depth_targets_keep_metric_scale_out_of_normalized_input():
    depth = torch.tensor([[[1.0, 3.0], [0.0, 2.0]]])
    valid = depth > 0
    normalized, scale = normalize_depth_per_sample(depth, valid)
    assert torch.allclose(scale, torch.tensor([2.0]))
    assert torch.allclose(normalized[valid], torch.tensor([0.5, 1.5, 1.0]))
    assert normalized[~valid].eq(0).all()


def test_metric_pointmap_is_target_camera_xyz():
    depth = torch.full((2, 2), 2.0)
    K = torch.eye(3)
    points = metric_depth_to_camera_pointmap(depth, K)
    assert torch.allclose(points[0, 0], torch.tensor([0.0, 0.0, 2.0]))
    assert torch.allclose(points[1, 1], torch.tensor([2.0, 2.0, 2.0]))
    K = torch.tensor([[2.0, 0.0, 1.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]])
    point = metric_depth_to_camera_pointmap(torch.full((3, 2), 2.0), K)
    assert torch.allclose(point[2, 1], torch.tensor([0.0, 0.0, 2.0]))


def test_validity_weighted_resize_does_not_blend_missing_depth_as_zero():
    depth = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
    valid = depth > 0
    resized, resized_valid = resize_depth_validity_weighted(depth, valid, (1, 1))
    assert resized_valid.item()
    assert resized.item() == 2.0


def test_readout_shapes_and_capacity_contract():
    z = torch.randn(2, 3, 4, 8)
    spatial = SpatialDepthReadout(8, hidden_dim=6, output_size=(12, 16))
    pooled = PooledDepthReadout(8, hidden_dim=6, grid_size=(3, 4), output_size=(12, 16))
    assert spatial(z).shape == (2, 1, 12, 16)
    assert pooled(z).shape == (2, 1, 12, 16)
    forbidden_dense = 8 * 12 * 16 * 6
    assert max(p.numel() for p in pooled.parameters()) < forbidden_dense
    assert parameter_count(spatial) > 0


def test_tokenwise_adapter_is_shared_over_positions():
    adapter = TokenwiseLinearAdapter(3)
    z = torch.randn(2, 4, 5, 3)
    assert torch.allclose(adapter(z), z)


def test_context_only_is_independently_parameterized_and_ignores_test_motion():
    model = ExternalPooledPredictor(8, motion_dim=12, hidden_dim=16, num_layers=1,
                                    num_heads=4, conditioning="context_only", dropout=0.0)
    model.eval()
    context = torch.randn(2, 5, 3, 4, 8)
    y0 = model(context, torch.zeros(2, 12))
    y1 = model(context, torch.randn(2, 12))
    assert torch.equal(y0, y1)


def test_motion_bins_and_shuffle_are_fit_on_train_and_horizon_matched():
    train = []
    dev = []
    for horizon in (2, 4):
        for index in range(6):
            row = {"horizon": horizon, "translation_m": float(index), "rotation_deg": float(index * 2),
                   "directed_depth_overlap": float(index) / 10, "sample_id": f"tr-{horizon}-{index}"}
            train.append(row)
        for index, value in enumerate((.2, .3, 2.2, 2.3, 4.2, 4.3)):
            dev.append({"horizon": horizon, "translation_m": value, "rotation_deg": value * 2,
                        "directed_depth_overlap": value / 10, "sample_id": f"dv-{horizon}-{index}"})
    thresholds = motion_bin_thresholds(train)
    shuffled = shuffled_motion_map(train, dev, thresholds)
    assert set(shuffled) == {row["sample_id"] for row in dev}
    for row in dev:
        other = shuffled[row["sample_id"]]
        assert other["horizon"] == row["horizon"]
        cuts = thresholds[row["horizon"]]["translation_m"]
        assert bin_index(other["translation_m"], cuts) == bin_index(row["translation_m"], cuts)


def test_depth_metrics_use_gt_mask_and_mechanism_metrics_are_zero_on_identity():
    target = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4) + 1
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[:, 0, 0] = False
    pred = target.clone()
    pred[:, 0, 0] = 1e6
    row = depth_metrics(pred, target, valid)[0]
    assert row["mae"] == 0 and row["absrel"] == 0 and row["delta1"] == 1
    assert coarse_layout_mae(pred, target, valid, (2, 2)).item() == 0
    assert detail_gradient_mae(pred, target, valid).item() == 0


def test_scene_cluster_bootstrap_pairs_samples_then_resamples_scenes():
    rows = []
    for scene, delta in (("a", 1.0), ("b", 3.0)):
        for i in range(2):
            rows += [
                {"scene_id": scene, "sample_id": str(i), "condition": "a", "mae": delta},
                {"scene_id": scene, "sample_id": str(i), "condition": "b", "mae": 0.0},
            ]
    out = scene_cluster_paired_bootstrap(rows, "a", "b", n_boot=2000)
    assert math.isclose(out["delta"], 2.0)
    assert out["n_scenes"] == 2
    assert out["ci_low"] <= out["delta"] <= out["ci_high"]


def test_vjepa_native_masks_use_time_height_width_flattening():
    idx = tubelet_token_indices([1, 3], spatial_size=(2, 3))
    assert torch.equal(idx, torch.tensor([6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23]))
    assert observation_to_tubelet(9) == 4
    assert observation_to_tubelet(11) == 5


def test_true_future_invariance_starts_before_the_encoder():
    prefix = torch.randn(1, 8, 3, 2, 2)
    a = torch.cat((prefix, torch.randn(1, 20, 3, 2, 2)), dim=1)
    b = torch.cat((prefix, torch.zeros(1, 20, 3, 2, 2)), dim=1)
    c = torch.cat((prefix, torch.full((1, 20, 3, 2, 2), 99.0)), dim=1)
    safe_a = make_prefix_safe_clip(a, 8, total_frames=16)
    safe_b = make_prefix_safe_clip(b, 8, total_frames=16)
    safe_c = make_prefix_safe_clip(c, 8, total_frames=16)
    assert torch.equal(safe_a, safe_b)
    assert torch.equal(safe_a, safe_c)
    assert torch.equal(safe_a[:, 8:], prefix[:, -1:].expand(-1, 8, -1, -1, -1))


def test_tiny_full_native_predictor_is_invariant_to_deliberate_future_changes():
    from transformers import VJEPA2Config, VJEPA2Model

    config = VJEPA2Config(
        image_size=4, crop_size=4, frames_per_clip=12, patch_size=2, tubelet_size=2,
        hidden_size=8, num_hidden_layers=1, num_attention_heads=2, mlp_ratio=2,
        pred_hidden_size=8, pred_num_hidden_layers=1, pred_num_attention_heads=2,
        pred_mlp_ratio=2, pred_num_mask_tokens=2,
    )
    model = VJEPA2Model(config).eval()
    prefix = torch.randn(1, 8, 3, 4, 4)
    a = torch.cat((prefix, torch.randn(1, 4, 3, 4, 4)), 1)
    b = torch.cat((prefix, torch.full((1, 4, 3, 4, 4), 50.0)), 1)
    pred_a = predict_future_tubelets(model, a)
    pred_b = predict_future_tubelets(model, b)
    assert set(pred_a) == {9, 11}
    assert pred_a[9].shape == (1, 2, 2, 8)
    assert torch.equal(pred_a[9], pred_b[9])
    assert torch.equal(pred_a[11], pred_b[11])


def test_feature_store_is_resume_safe_and_fails_closed(tmp_path):
    revision = "abc123"
    tensor = torch.arange(6, dtype=torch.float16).reshape(2, 3)
    write_feature(tmp_path, "model", "space", "sample", {"spatial": tensor}, {"model_revision": revision})
    loaded, receipt = load_verified(tmp_path, "model", "space", "sample", revision)
    assert torch.equal(loaded["spatial"], tensor)
    write_feature(tmp_path, "model", "space", "sample", {"spatial": tensor}, {"model_revision": revision})
    sidecar = tmp_path / "model/space/sample.json"
    data = json.loads(sidecar.read_text()); data["model_revision"] = "wrong"; sidecar.write_text(json.dumps(data))
    try:
        load_verified(tmp_path, "model", "space", "sample", revision)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched feature receipt was accepted")


def test_evaluator_identity_gate_uses_same_frozen_readout():
    readout = SpatialDepthReadout(8, hidden_dim=4, output_size=(6, 8)).eval()
    target_latent = torch.randn(2, 3, 4, 8)
    predicted_latent = target_latent.clone()
    with torch.inference_mode():
        target_depth = readout(target_latent)
        predicted_depth = readout(predicted_latent)
    assert torch.equal(target_depth, predicted_depth)
    valid = torch.ones(2, 6, 8, dtype=torch.bool)
    target_metrics = depth_metrics(target_depth, target_depth[:, 0], valid)
    predicted_metrics = depth_metrics(predicted_depth, target_depth[:, 0], valid)
    assert target_metrics == predicted_metrics
