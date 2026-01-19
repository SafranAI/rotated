from copy import deepcopy
import math

import pytest
import torch

from rotated.boxes.decode import decode_ppyoloer_boxes
from rotated.losses.ppyoloer_criterion import RotatedDetectionLoss

DEFAULT_CONFIG = {
    "num_classes": 3,
    "angle_bins": 90,
    "use_angle_bins": True,
    "use_dfl": False,
    "loss_weights": {"cls": 1.0, "box": 1.0, "angle": 1.0, "dfl": 1.0},
}


@pytest.fixture(scope="function", params=["probiou", "mgiou"])
def criterion(request) -> RotatedDetectionLoss:
    loss_cfg = deepcopy(DEFAULT_CONFIG)
    loss_cfg["box_loss_type"] = request.param
    return RotatedDetectionLoss(**loss_cfg)


def test_criterion_perfect_prediction(criterion: RotatedDetectionLoss):
    """Test criterion with predictions that match targets exactly."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = criterion.to(device)
    batch_size, num_anchors, num_classes = 1, 4, 3

    anchor_points = torch.tensor([[[50.0, 50.0], [100.0, 50.0], [50.0, 100.0], [100.0, 100.0]]], device=device)

    stride_tensor = torch.ones(1, num_anchors, 1, device=device) * 8

    targets = {
        "labels": torch.tensor([[[1]]], device=device),
        "boxes": torch.tensor([[[50.0, 50.0, 20.0, 10.0, 0.0]]], device=device),
        "valid_mask": torch.ones(1, 1, 1, device=device),
    }

    cls_logits = torch.full((batch_size, num_anchors, num_classes), -5.0, device=device, requires_grad=True)
    cls_logits.data[0, 0, 1] = 5.0

    reg_dist = torch.zeros(batch_size, num_anchors, 4, device=device, requires_grad=True)
    reg_dist.data[0, 0] = torch.tensor([0.0, 0.0, 1.5, 0.25])

    raw_angles = torch.full((batch_size, num_anchors, 91), -3.0, device=device, requires_grad=True)
    raw_angles.data[0, 0, 0] = 5.0

    angle_proj = torch.linspace(0, math.pi / 2, 91, device=device)

    decoded_boxes = decode_ppyoloer_boxes(
        anchor_points,
        reg_dist,
        raw_angles,
        stride_tensor,
        angle_proj,
        use_angle_bins=DEFAULT_CONFIG["use_angle_bins"],
        use_dfl=DEFAULT_CONFIG["use_dfl"],
    )

    losses = criterion(
        cls_logits,
        reg_dist,
        raw_angles,
        decoded_boxes=decoded_boxes,
        targets=targets,
        anchor_points=anchor_points,
        stride_tensor=stride_tensor,
    )

    assert losses.box.item() > 0
    assert losses.angle.item() >= 0
    assert losses.cls.item() > 0

    losses.total.backward()
    assert cls_logits.grad is not None
    assert reg_dist.grad is not None
    assert raw_angles.grad is not None


def test_criterion_empty_targets(criterion: RotatedDetectionLoss):
    """Test criterion with no ground truth targets."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = criterion.to(device)

    batch_size, num_anchors, num_classes = 1, 4, 3

    cls_logits = torch.zeros(batch_size, num_anchors, num_classes, device=device, requires_grad=True)
    reg_dist = torch.zeros(batch_size, num_anchors, 4, device=device, requires_grad=True)
    raw_angles = torch.zeros(batch_size, num_anchors, 91, device=device, requires_grad=True)

    targets = {
        "labels": torch.zeros(batch_size, 0, 1, dtype=torch.long, device=device),
        "boxes": torch.zeros(batch_size, 0, 5, device=device),
        "valid_mask": torch.zeros(batch_size, 0, 1, device=device),
    }

    anchor_points = torch.zeros(1, num_anchors, 2, device=device)
    stride_tensor = torch.ones(1, num_anchors, 1, device=device) * 8
    angle_proj = torch.linspace(0, math.pi / 2, 91, device=device)

    decoded_boxes = decode_ppyoloer_boxes(
        anchor_points,
        reg_dist,
        raw_angles,
        stride_tensor,
        angle_proj,
        use_angle_bins=DEFAULT_CONFIG["use_angle_bins"],
        use_dfl=DEFAULT_CONFIG["use_dfl"],
    )

    losses = criterion(
        cls_logits,
        reg_dist,
        raw_angles,
        decoded_boxes=decoded_boxes,
        targets=targets,
        anchor_points=anchor_points,
        stride_tensor=stride_tensor,
    )

    assert losses.box.item() == 0.0
    assert losses.angle.item() == 0.0
    assert losses.cls.item() >= 0

    losses.total.backward()
    assert cls_logits.grad is not None


def test_criterion_no_spatial_overlap(criterion: RotatedDetectionLoss):
    """Test criterion with targets spatially distant from anchors."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = criterion.to(device)

    batch_size, num_anchors, num_classes = 1, 4, 3

    anchor_points = torch.tensor([[[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]]], device=device)

    targets = {
        "labels": torch.tensor([[[1]]], device=device),
        "boxes": torch.tensor([[[1000.0, 1000.0, 20.0, 10.0, 0.0]]], device=device),
        "valid_mask": torch.ones(1, 1, 1, device=device),
    }

    cls_logits = torch.zeros(batch_size, num_anchors, num_classes, device=device, requires_grad=True)
    reg_dist = torch.zeros(batch_size, num_anchors, 4, device=device, requires_grad=True)
    raw_angles = torch.zeros(batch_size, num_anchors, 91, device=device, requires_grad=True)

    stride_tensor = torch.ones(1, num_anchors, 1, device=device) * 8
    angle_proj = torch.linspace(0, math.pi / 2, 91, device=device)

    decoded_boxes = decode_ppyoloer_boxes(
        anchor_points,
        reg_dist,
        raw_angles,
        stride_tensor,
        angle_proj,
        use_angle_bins=DEFAULT_CONFIG["use_angle_bins"],
        use_dfl=DEFAULT_CONFIG["use_dfl"],
    )

    losses = criterion(
        cls_logits,
        reg_dist,
        raw_angles,
        decoded_boxes=decoded_boxes,
        targets=targets,
        anchor_points=anchor_points,
        stride_tensor=stride_tensor,
    )

    assert losses.box.item() == 0.0
    assert losses.angle.item() == 0.0

    losses.total.backward()
    assert cls_logits.grad is not None


def test_criterion_multiple_targets(criterion: RotatedDetectionLoss):
    """Test criterion with multiple targets assigned to different anchors."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = criterion.to(device)

    batch_size, num_anchors, num_classes = 1, 4, 3

    anchor_points = torch.tensor([[[25.0, 25.0], [75.0, 25.0], [25.0, 75.0], [75.0, 75.0]]], device=device)

    targets = {
        "labels": torch.tensor([[[1], [2]]], device=device),
        "boxes": torch.tensor([[[25.0, 25.0, 20.0, 10.0, 0.0], [75.0, 75.0, 15.0, 15.0, 0.5]]], device=device),
        "valid_mask": torch.ones(1, 2, 1, device=device),
    }

    cls_logits = torch.full((batch_size, num_anchors, num_classes), -3.0, device=device, requires_grad=True)
    cls_logits.data[0, 0, 1] = 3.0
    cls_logits.data[0, 3, 2] = 3.0

    reg_dist = torch.zeros(batch_size, num_anchors, 4, device=device, requires_grad=True)
    reg_dist.data[0, 0] = torch.tensor([0.0, 0.0, 1.0, 0.5])
    reg_dist.data[0, 3] = torch.tensor([0.0, 0.0, 0.8, 0.8])

    raw_angles = torch.full((batch_size, num_anchors, 91), -2.0, device=device, requires_grad=True)
    raw_angles.data[0, :, 0] = 2.0

    stride_tensor = torch.ones(1, num_anchors, 1, device=device) * 8
    angle_proj = torch.linspace(0, math.pi / 2, 91, device=device)

    decoded_boxes = decode_ppyoloer_boxes(
        anchor_points,
        reg_dist,
        raw_angles,
        stride_tensor,
        angle_proj,
        use_angle_bins=DEFAULT_CONFIG["use_angle_bins"],
        use_dfl=DEFAULT_CONFIG["use_dfl"],
    )

    losses = criterion(
        cls_logits,
        reg_dist,
        raw_angles,
        decoded_boxes=decoded_boxes,
        targets=targets,
        anchor_points=anchor_points,
        stride_tensor=stride_tensor,
    )

    assert losses.box.item() > 0
    assert losses.angle.item() >= 0

    losses.total.backward()
    assert cls_logits.grad is not None
    assert reg_dist.grad is not None
    assert raw_angles.grad is not None
