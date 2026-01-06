# Modified from PaddleDetection (https://github.com/PaddlePaddle/PaddleDetection)
# Copyright (c) 2024 PaddlePaddle Authors. Apache 2.0 License.

"""Loss criterion for PP-YOLOE-R rotated object detection.

Provides flexible classification losses (BCE, focal, varifocal) and optional
Distribution Focal Loss for bbox regression.
"""

from collections import namedtuple
import math
from typing import Literal, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from rotated.assigners import RotatedTaskAlignedAssigner
from rotated.losses.prob_iou import ProbIoULoss


class LossWeights(TypedDict):
    """Weight configuration for different loss components.

    Attributes:
        cls: Weight for classification loss
        box: Weight for bounding box regression loss
        angle: Weight for angle prediction loss
        dfl: Weight for distribution focal loss (if enabled)
    """

    cls: float
    box: float
    angle: float
    dfl: float


class AssignerConfig(TypedDict):
    """Configuration for the task-aligned assigner.

    Attributes:
        topk: Number of top candidates to consider
        alpha: Weight for classification score in assignment
        beta: Weight for IoU score in assignment
    """

    topk: float
    alpha: float
    beta: float


class FocalConfig(TypedDict):
    """Configuration for focal loss parameters.

    Attributes:
        alpha: Balancing factor between positive/negative samples
        gamma: Focusing parameter to reduce easy examples' weight
    """

    alpha: float
    gamma: float


class LossComponents(namedtuple("LossComponents", ["total", "cls", "box", "angle", "dfl"])):
    """Named tuple for PP-YOLOE-R loss components."""

    @classmethod
    def empty(cls, device: torch.device | str | None = None) -> "LossComponents":
        """Create empty loss components."""
        device = device or torch.device("cpu")
        return cls(
            total=torch.tensor(0.0, device=device),
            cls=torch.tensor(0.0, device=device),
            box=torch.tensor(0.0, device=device),
            angle=torch.tensor(0.0, device=device),
            dfl=torch.tensor(0.0, device=device),
        )


class RotatedDetectionLoss(nn.Module):
    """Loss criterion for rotated object detection.

    Args:
        num_classes: Number of object classes
        loss_weights: Weights for each component
        cls_loss_type: Classification loss type
        assigner_config: Task-aligned assigner configuration
        focal_config: Focal loss parameters
        use_angle_bins: Enable binned angle prediction
        angle_bins: Number of angle bins
        use_dfl: Enable Distribution Focal Loss for bbox regression
        reg_max: Number of DFL bins
    """

    def __init__(
        self,
        num_classes: int = 15,
        loss_weights: LossWeights | None = None,
        cls_loss_type: Literal["varifocal", "focal", "bce"] = "varifocal",
        assigner_config: AssignerConfig | None = None,
        focal_config: FocalConfig | None = None,
        use_angle_bins: bool = True,
        angle_bins: int = 90,
        use_dfl: bool = False,
        reg_max: int = 16,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.angle_bins = angle_bins
        self.cls_loss_type = cls_loss_type
        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.use_angle_bins = use_angle_bins

        self.loss_weights = loss_weights or {"cls": 1.0, "box": 2.5, "angle": 0.05, "dfl": 0.5}
        assigner_config = assigner_config or {"topk": 13, "alpha": 1.0, "beta": 6.0}
        focal_config = focal_config or {"alpha": 0.75, "gamma": 2.0}

        self.focal_alpha = focal_config["alpha"]
        self.focal_gamma = focal_config["gamma"]
        self.angle_scale = math.pi / 2 / angle_bins

        self.assigner = RotatedTaskAlignedAssigner(**assigner_config)
        self.box_loss_fn = ProbIoULoss()

    def _sync_config(self, use_dfl: bool, reg_max: int, use_angle_bins: bool) -> None:
        """Sync configuration from head."""
        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.use_angle_bins = use_angle_bins

    def forward(
        self,
        cls_logits: torch.Tensor,
        reg_dist: torch.Tensor,
        raw_angles: torch.Tensor,
        decoded_boxes: torch.Tensor,
        targets: dict[str, torch.Tensor],
        anchor_points: torch.Tensor,
        stride_tensor: torch.Tensor,
    ) -> LossComponents:
        """Compute detection losses.

        Args:
            cls_logits: Classification logits [B, N, C]
            reg_dist: Raw regression predictions [B, N, 4*reg_max] or [B, N, 4]
            raw_angles: Raw angle predictions [B, N, angle_bins+1] or [B, N, 1]
            decoded_boxes: Decoded boxes in pixels [B, N, 5]
            targets: Ground truth data:
                - labels: [B, M, 1] - Class labels [0, num_classes-1]
                - boxes: [B, M, 5] - Rotated GT boxes [cx, cy, w, h, angle]
                    * cx, cy, w, h: in absolute pixels
                    * angle: in radians, should be in range [0, π/2)
                - valid_mask: [B, M, 1] - Valid target mask
            anchor_points: Anchor points in pixels [1, N, 2]
            stride_tensor: Stride values [1, N, 1]

        Returns:
            Loss components
        """
        gt_labels = targets["labels"]
        gt_boxes = targets["boxes"]
        valid_mask = targets["valid_mask"]

        cls_scores = torch.sigmoid(cls_logits)

        # Task-aligned assignment
        assigned_labels, assigned_boxes, assigned_scores = self.assigner(
            pred_scores=cls_scores.detach(),
            pred_boxes=decoded_boxes.detach(),
            anchor_points=anchor_points,
            gt_labels=gt_labels,
            gt_boxes=gt_boxes,
            pad_gt_mask=valid_mask,
            bg_index=self.num_classes,
        )

        # Compute losses
        loss_cls = self._classification_loss(cls_logits, assigned_scores, assigned_labels)
        loss_box = self._box_loss(decoded_boxes, assigned_boxes, assigned_labels, assigned_scores, stride_tensor)

        if self.use_angle_bins:
            loss_angle = self._angle_loss(raw_angles, assigned_boxes, assigned_labels)
        else:
            loss_angle = torch.tensor(0.0, device=cls_logits.device)

        if self.use_dfl:
            loss_dfl = self._dfl_loss(
                reg_dist, assigned_boxes, assigned_labels, assigned_scores, anchor_points, stride_tensor
            )
        else:
            loss_dfl = torch.tensor(0.0, device=cls_logits.device)

        total_loss = (
            self.loss_weights["cls"] * loss_cls
            + self.loss_weights["box"] * loss_box
            + self.loss_weights["angle"] * loss_angle
            + self.loss_weights["dfl"] * loss_dfl
        )

        return LossComponents(total=total_loss, cls=loss_cls, box=loss_box, angle=loss_angle, dfl=loss_dfl)

    def _classification_loss(
        self, pred_logits: torch.Tensor, target_scores: torch.Tensor, target_labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute classification loss."""
        if self.cls_loss_type == "bce":
            return self._bce_loss(pred_logits, target_scores)
        elif self.cls_loss_type == "varifocal":
            return self._varifocal_loss(pred_logits, target_scores, target_labels)
        return self._focal_loss(pred_logits, target_scores)

    def _bce_loss(self, pred_logits: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
        """Binary cross-entropy loss."""
        assigned_scores_sum = torch.clamp(target_scores.sum(), min=1.0)
        base_loss = F.binary_cross_entropy_with_logits(pred_logits, target_scores, reduction="none")
        return base_loss.sum() / assigned_scores_sum

    def _varifocal_loss(
        self, pred_logits: torch.Tensor, target_scores: torch.Tensor, target_labels: torch.Tensor
    ) -> torch.Tensor:
        """Varifocal loss."""
        assigned_scores_sum = torch.clamp(target_scores.sum(), min=1.0)
        pred_sigmoid = torch.sigmoid(pred_logits)

        target_labels_onehot = F.one_hot(target_labels, self.num_classes + 1)[..., :-1].float()

        focal_weight = (
            self.focal_alpha * pred_sigmoid.pow(self.focal_gamma) * (1.0 - target_labels_onehot)
            + target_scores * target_labels_onehot
        )

        base_loss = F.binary_cross_entropy_with_logits(pred_logits, target_scores, reduction="none")
        return (base_loss * focal_weight).sum() / assigned_scores_sum

    def _focal_loss(self, pred_logits: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
        """Focal loss."""
        assigned_scores_sum = torch.clamp(target_scores.sum(), min=1.0)
        pred_sigmoid = torch.sigmoid(pred_logits)

        base_loss = F.binary_cross_entropy_with_logits(pred_logits, target_scores, reduction="none")
        focal_weight = (pred_sigmoid - target_scores).pow(self.focal_gamma)

        if self.focal_alpha > 0:
            alpha_weight = self.focal_alpha * target_scores + (1 - self.focal_alpha) * (1 - target_scores)
            focal_weight = focal_weight * alpha_weight

        return (base_loss * focal_weight).sum() / assigned_scores_sum

    def _box_loss(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        target_labels: torch.Tensor,
        target_scores: torch.Tensor,
        stride_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Compute box regression loss in stride-normalized space."""
        positive_mask = target_labels != self.num_classes
        num_positives = positive_mask.sum()

        if num_positives == 0:
            return pred_boxes.sum() * 0.0

        # Convert to stride-normalized space
        pred_boxes_s = pred_boxes.clone()
        pred_boxes_s[..., :4] = pred_boxes[..., :4] / stride_tensor

        target_boxes_s = target_boxes.clone()
        target_boxes_s[..., :4] = target_boxes[..., :4] / stride_tensor

        pred_pos = pred_boxes_s[positive_mask]
        target_pos = target_boxes_s[positive_mask]

        iou_loss = self.box_loss_fn(pred_pos, target_pos)

        box_weights = target_scores.sum(-1)[positive_mask]
        weighted_loss = (iou_loss * box_weights).sum()

        normalizer = torch.clamp(target_scores.sum(), min=1.0)
        return weighted_loss / normalizer

    def _angle_loss(
        self,
        pred_angles: torch.Tensor,
        target_boxes: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute angle distribution loss."""
        positive_mask = target_labels != self.num_classes
        num_positives = positive_mask.sum()

        if num_positives == 0:
            return pred_angles.sum() * 0.0

        pred_angle_pos = pred_angles[positive_mask]
        target_angle_pos = target_boxes[positive_mask][:, 4]

        target_bins = target_angle_pos / self.angle_scale
        target_bins = torch.clamp(target_bins, 0, self.angle_bins - 0.01)

        left_idx = target_bins.long()
        right_idx = torch.clamp(left_idx + 1, max=self.angle_bins)

        left_weight = right_idx.float() - target_bins
        right_weight = 1.0 - left_weight

        loss_left = F.cross_entropy(pred_angle_pos, left_idx, reduction="none") * left_weight
        loss_right = F.cross_entropy(pred_angle_pos, right_idx, reduction="none") * right_weight
        return (loss_left + loss_right).mean()

    def _dfl_loss(
        self,
        pred_dist: torch.Tensor,
        target_boxes: torch.Tensor,
        target_labels: torch.Tensor,
        target_scores: torch.Tensor,
        anchor_points: torch.Tensor,
        stride_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Distribution Focal Loss."""
        positive_mask = target_labels != self.num_classes
        num_positives = positive_mask.sum()

        if num_positives == 0:
            return pred_dist.sum() * 0.0

        # Extract positive samples
        pred_dist_pos = pred_dist[positive_mask]
        target_boxes_pos = target_boxes[positive_mask]
        anchor_points_pos = anchor_points.expand_as(target_boxes[:, :, :2])[positive_mask]
        stride_tensor_pos = stride_tensor.expand_as(target_boxes[:, :, :1])[positive_mask]

        weight = target_scores.sum(-1)[positive_mask].unsqueeze(-1)

        # Reshape to [P, 4, reg_max]
        batch_size = pred_dist_pos.shape[0]
        pred_dist_pos = pred_dist_pos.view(batch_size, 4, self.reg_max)

        # Convert boxes to stride-normalized xyxy
        target_boxes_s = target_boxes_pos.clone()
        target_boxes_s[:, :4] = target_boxes_pos[:, :4] / stride_tensor_pos

        anchor_points_s = anchor_points_pos / stride_tensor_pos

        target_centers = target_boxes_s[:, :2]
        target_sizes = target_boxes_s[:, 2:4]
        target_x1y1 = target_centers - target_sizes / 2
        target_x2y2 = target_centers + target_sizes / 2

        # Convert to LTRB distances
        lt = anchor_points_s - target_x1y1
        rb = target_x2y2 - anchor_points_s
        target_ltrb = torch.cat([lt, rb], dim=-1)

        target_ltrb = torch.clamp(target_ltrb, min=0.0, max=self.reg_max - 1.0 - 0.01)

        # Linear interpolation
        left_idx = target_ltrb.long()
        right_idx = torch.clamp(left_idx + 1, max=self.reg_max - 1)

        left_weight_ltrb = right_idx.float() - target_ltrb
        right_weight_ltrb = 1.0 - left_weight_ltrb

        # Compute loss
        pred_dist_flat = pred_dist_pos.view(-1, self.reg_max)
        left_idx_flat = left_idx.view(-1)
        right_idx_flat = right_idx.view(-1)
        left_weight_flat = left_weight_ltrb.view(-1)
        right_weight_flat = right_weight_ltrb.view(-1)

        loss_left = F.cross_entropy(pred_dist_flat, left_idx_flat, reduction="none") * left_weight_flat
        loss_right = F.cross_entropy(pred_dist_flat, right_idx_flat, reduction="none") * right_weight_flat
        loss_per_ltrb = (loss_left + loss_right).view(batch_size, 4)

        loss_per_sample = loss_per_ltrb.mean(-1, keepdim=True)
        weighted_loss = (loss_per_sample * weight).sum()

        target_scores_sum = torch.clamp(target_scores.sum(), min=1.0)
        return weighted_loss / target_scores_sum
