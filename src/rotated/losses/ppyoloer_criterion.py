# Modified from PaddleDetection (https://github.com/PaddlePaddle/PaddleDetection)
# Copyright (c) 2024 PaddlePaddle Authors. Apache 2.0 License.

"""Loss criterion for PP-YOLOE-R rotated object detection.

This module implements a flexible multi-component loss function for training rotated object
detection models. It combines classification, bounding box regression, and angle prediction
losses with task-aligned assignment for optimal anchor-to-ground-truth matching.

Key Features:

    - Task-aligned assignment: Dynamically assigns ground truth objects to anchors based on
      alignment metrics that combine classification confidence and localization quality.
    - Flexible classification losses: Supports BCE, focal loss, and varifocal loss to handle
      class imbalance and focus on hard examples.
    - Dual bbox regression modes: Standard regression or Distribution Focal Loss (DFL) for
      finer-grained localization through distribution modeling.
    - Dual angle prediction modes: Binned (distribution over discrete angles) or continuous
      (direct angle regression) with appropriate loss functions for each.
    - ProbIoU loss: Uses probabilistic IoU for rotated boxes that handles varying uncertainty
      in box predictions.
    - Optionally, can use MGIoU loss instead of ProbIoU loss. Note: angle loss is then disabled,
        because it is not needed.

Components:

    - RotatedDetectionLoss: Main loss module that orchestrates all loss components
    - LossComponents: Named tuple container for individual loss values
    - LossWeights: Type definition for loss component weights
    - AssignerConfig: Configuration for task-aligned assignment
    - FocalConfig: Configuration for focal loss parameters
    - AngleConfig: Configuration for angle loss parameters

Example:
    >>> criterion = RotatedDetectionLoss(
    ...     num_classes=15,
    ...     use_angle_bins=True,
    ...     angle_bins=90,
    ...     use_dfl=False
    ... )
    >>> losses = criterion(
    ...     cls_logits=cls_logits,
    ...     reg_dist=reg_dist,
    ...     raw_angles=raw_angles,
    ...     decoded_boxes=decoded_boxes,
    ...     targets=targets,
    ...     anchor_points=anchor_points,
    ...     stride_tensor=stride_tensor
    ... )
    >>> total_loss = losses.total
"""

from collections import namedtuple
import math
from typing import Literal, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from rotated.assigners import RotatedTaskAlignedAssigner
from rotated.losses.mgiou import MGIoU2DLoss
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

    topk: int
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


class AngleConfig(TypedDict):
    """Configuration for angle loss parameters.

    Used for consistency with other configuration dictionaries (FocalConfig, AssignerConfig).

    Attributes:
        beta: Smooth L1 threshold for continuous angle loss (only used when use_angle_bins=False)
    """

    beta: float


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

    Supports two bounding box regression modes:

    - Standard mode (use_dfl=False): Direct regression of box parameters with IoU-based loss.
    - DFL mode (use_dfl=True): Distribution Focal Loss that models box coordinates as
      distributions over discrete bins, providing finer-grained localization.

    Supports two angle prediction modes:

    - Binned mode (use_angle_bins=True): Predicts angle as a distribution over bins,
      using soft cross-entropy loss with linear interpolation between adjacent bins.
    - Continuous mode (use_angle_bins=False): Predicts angle directly as a continuous value,
      using circular smooth L1 loss that respects angle periodicity.

    Args:
        num_classes: Number of object classes
        loss_weights: Weights for each loss component (cls, box, angle, dfl)
        cls_loss_type: Classification loss type ('varifocal', 'focal', or 'bce')
        box_loss_type: Bounding box loss type ('probiou', 'mgiou')
        assigner_config: Task-aligned assigner configuration (topk, alpha, beta)
        focal_config: Focal loss parameters (alpha, gamma)
        angle_config: Angle loss parameters (beta for smooth L1 threshold when use_angle_bins=False)
        use_angle_bins: If True, use binned angle prediction; if False, use continuous angle
        angle_bins: Number of angle bins (only used when use_angle_bins=True)
        use_dfl: If True, use Distribution Focal Loss for bbox regression; if False, use standard regression
        reg_max: Number of DFL bins (only used when use_dfl=True)

    Raises:
        ValueError: if `box_loss_type` is unknown, ie not "probiou" or "mgiou".
    """

    def __init__(
        self,
        num_classes: int = 15,
        loss_weights: LossWeights | None = None,
        cls_loss_type: Literal["varifocal", "focal", "bce"] = "varifocal",
        box_loss_type: Literal["probiou", "mgiou"] = "probiou",
        assigner_config: AssignerConfig | None = None,
        focal_config: FocalConfig | None = None,
        angle_config: AngleConfig | None = None,
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
        angle_config = angle_config or {"beta": 0.1}

        self.focal_alpha = focal_config["alpha"]
        self.focal_gamma = focal_config["gamma"]
        self.angle_beta = angle_config["beta"]
        self.angle_scale = math.pi / 2 / angle_bins

        self.assigner = RotatedTaskAlignedAssigner(**assigner_config)

        if box_loss_type not in ["probiou", "mgiou"]:
            raise ValueError(f"Unknown `bbox_loss_type`: {box_loss_type}")
        self.box_loss_type = box_loss_type
        if box_loss_type == "mgiou":
            self.box_loss_fn = MGIoU2DLoss()
        else:
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
            targets: Ground truth with 'labels', 'boxes' (pixels), 'valid_mask'
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

        if self.box_loss_type == "mgiou":
            # no loss_angle needed for MGIoU
            loss_angle = torch.tensor(0.0, device=cls_logits.device)
        elif self.use_angle_bins:
            loss_angle = self._angle_loss_binned(raw_angles, assigned_boxes, assigned_labels)
        else:
            loss_angle = self._angle_loss_continuous(raw_angles, assigned_boxes, assigned_labels)

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

    def _weighted_focal_loss(
        self, pred_logits: torch.Tensor, target_scores: torch.Tensor, focal_weight: torch.Tensor
    ) -> torch.Tensor:
        """Compute focal loss with given weights."""
        assigned_scores_sum = torch.clamp(target_scores.sum(), min=1.0)
        base_loss = F.binary_cross_entropy_with_logits(pred_logits, target_scores, reduction="none")
        return (base_loss * focal_weight).sum() / assigned_scores_sum

    def _compute_varifocal_weight(
        self, pred_sigmoid: torch.Tensor, target_scores: torch.Tensor, target_labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute varifocal weighting.

        Varifocal loss down-weights negative samples with focal term and uses target IoU scores as weights for
        positive samples.
        """
        target_labels_onehot = F.one_hot(target_labels, self.num_classes + 1)[..., :-1].float()

        focal_weight = (
            self.focal_alpha * pred_sigmoid.pow(self.focal_gamma) * (1.0 - target_labels_onehot)
            + target_scores * target_labels_onehot
        )
        return focal_weight

    def _compute_focal_weight(self, pred_sigmoid: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
        """Compute standard focal weighting.

        Standard focal loss weights based on prediction error magnitude.
        """
        focal_weight = (pred_sigmoid - target_scores).abs().pow(self.focal_gamma)

        if self.focal_alpha > 0:
            alpha_weight = self.focal_alpha * target_scores + (1 - self.focal_alpha) * (1 - target_scores)
            focal_weight = focal_weight * alpha_weight

        return focal_weight

    def _varifocal_loss(
        self, pred_logits: torch.Tensor, target_scores: torch.Tensor, target_labels: torch.Tensor
    ) -> torch.Tensor:
        """Varifocal loss."""
        pred_sigmoid = torch.sigmoid(pred_logits)
        focal_weight = self._compute_varifocal_weight(pred_sigmoid, target_scores, target_labels)
        return self._weighted_focal_loss(pred_logits, target_scores, focal_weight)

    def _focal_loss(self, pred_logits: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
        """Focal loss."""
        pred_sigmoid = torch.sigmoid(pred_logits)
        focal_weight = self._compute_focal_weight(pred_sigmoid, target_scores)
        return self._weighted_focal_loss(pred_logits, target_scores, focal_weight)

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

    def _angle_loss_binned(
        self,
        pred_angles: torch.Tensor,
        target_boxes: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute binned angle prediction loss using soft cross-entropy.

        Converts continuous target angles to soft labels over adjacent bins using linear interpolation, then
        computes weighted cross-entropy loss.

        Args:
            pred_angles: Predicted angle logits [B, N, angle_bins+1]
            target_boxes: Ground truth boxes [B, N, 5] where [..., 4] is angle in radians
            target_labels: Assigned class labels [B, N]

        Returns:
            Scalar loss value
        """
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

    def _angle_loss_continuous(
        self,
        pred_angles: torch.Tensor,
        target_boxes: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute continuous angle prediction loss using circular smooth L1.

        Uses smooth L1 loss that respects angle periodicity at the [0, π/2] boundaries.
        This handles cases where angles near 0 and π/2 should be considered close.

        Args:
            pred_angles: Predicted angles [B, N, 1] in radians, range [0, π/2]
            target_boxes: Ground truth boxes [B, N, 5] where [..., 4] is angle in radians
            target_labels: Assigned class labels [B, N]

        Returns:
            Scalar loss value
        """
        positive_mask = target_labels != self.num_classes
        num_positives = positive_mask.sum()

        if num_positives == 0:
            return pred_angles.sum() * 0.0

        pred_angle_pos = pred_angles[positive_mask].squeeze(-1)
        target_angle_pos = target_boxes[positive_mask][:, 4]

        # Compute angular difference respecting [0, π/2] periodicity
        diff = torch.abs(pred_angle_pos - target_angle_pos)
        half_pi = math.pi / 2
        diff = torch.minimum(diff, half_pi - diff)

        # Smooth L1 loss with configurable beta threshold
        loss = torch.where(diff < self.angle_beta, 0.5 * diff.pow(2) / self.angle_beta, diff - 0.5 * self.angle_beta)

        return loss.mean()
