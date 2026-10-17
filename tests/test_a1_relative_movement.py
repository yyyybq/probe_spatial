import torch

from vidfm3d.models.components.movement_head import RelativeMovementHead
from vidfm3d.vggt.heads.camera_head import CameraHead
from vidfm3d.utils.loss import (
    movement_6d_to_rotation_matrix,
    relative_movement_loss,
    relative_movements_from_extrinsics,
)
from vidfm3d.utils.spatial_diag import encode_relative_pose


def _trajectory() -> torch.Tensor:
    extrinsics = torch.eye(4)[:3].repeat(3, 1, 1)
    extrinsics[1, :3, 3] = torch.tensor([1.0, 0.0, 0.0])
    extrinsics[2, :3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    extrinsics[2, :3, 3] = torch.tensor([1.0, 2.0, 0.0])
    return extrinsics


def test_relative_movement_targets_match_c_series_action_convention():
    extrinsics = _trajectory()
    target = relative_movements_from_extrinsics(extrinsics.unsqueeze(0))[0]
    expected = torch.stack(
        [
            encode_relative_pose(extrinsics[0], extrinsics[1]),
            encode_relative_pose(extrinsics[1], extrinsics[2]),
        ]
    )
    assert target.shape == (2, 9)
    assert torch.allclose(target, expected, atol=1e-6, rtol=1e-6)


def test_row_major_rotation_6d_round_trip():
    extrinsics = _trajectory()
    target = relative_movements_from_extrinsics(extrinsics.unsqueeze(0))
    decoded = movement_6d_to_rotation_matrix(target[..., :6])

    rotation_from = extrinsics[:-1, :3, :3]
    rotation_to = extrinsics[1:, :3, :3]
    expected = rotation_to @ rotation_from.transpose(-1, -2)
    assert torch.allclose(decoded[0], expected, atol=1e-6, rtol=1e-6)


def test_movement_head_outputs_one_prediction_per_adjacent_edge():
    torch.manual_seed(0)
    head = RelativeMovementHead(dim_in=32, trunk_depth=1, num_heads=4)
    tokens = torch.randn(2, 5, 6, 32, requires_grad=True)
    predictions = head([tokens], num_iterations=2)

    assert len(predictions) == 2
    assert predictions[-1].shape == (2, 4, 9)
    predictions[-1].sum().backward()
    assert tokens.grad is not None


def test_movement_and_camera_heads_have_matched_parameter_capacity():
    movement = RelativeMovementHead(dim_in=32, trunk_depth=1, num_heads=4)
    camera = CameraHead(dim_in=32, trunk_depth=1, num_heads=4)
    assert sum(p.numel() for p in movement.parameters()) == sum(
        p.numel() for p in camera.parameters()
    )


def test_perfect_relative_movement_has_zero_loss_and_near_zero_error():
    extrinsics = _trajectory().unsqueeze(0)
    target = relative_movements_from_extrinsics(extrinsics)
    result = relative_movement_loss(
        [target.clone()], extrinsics, return_metrics=True
    )

    assert result["loss_movement"].item() == 0.0
    assert result["adjacent_movement_RotErr_deg"].item() < 0.1
    assert result["adjacent_movement_TransAngle_deg"].item() < 0.1
    assert result["adjacent_movement_TransL2"].item() < 1e-7
    assert result["adjacent_movement_Auc_30"].item() > 0.95
