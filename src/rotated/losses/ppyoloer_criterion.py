# Modified from PaddleDetection (https://github.com/PaddlePaddle/PaddleDetection)
# Copyright (c) 2024 PaddlePaddle Authors. Apache 2.0 License.

from collections import namedtuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rotated.assigners import RotatedTaskAlignedAssigner
from rotated.boxes.decode import decode_ppyoloer_boxes
from rotated.losses.prob_iou import ProbIoULoss


class LossComponents(namedtuple("LossComponents", ["total", "cls", "box", "angle"])):
    """Named tuple for PP-YOLOE-R loss components with helper methods."""

    @classmethod
    def empty(cls, device: torch.device | str | None = None) -> "LossComponents":
        """Create empty loss components with proper device placement."""
        device = device or torch.device("cpu")
        return cls(
            total=torch.tensor(0.0, device=device),
            cls=torch.tensor(0.0, device=device),
            box=torch.tensor(0.0, device=device),
            angle=torch.tensor(0.0, device=device),
        )


class RotatedDetectionLoss(nn.Module):
    """Rotated detection loss with clean raw prediction interface.

    Receives only raw predictions and handles its own processing for
    assignment and loss computation. Uses shared decoder for assignment.
    """

    def __init__(
        self,
        num_classes: int = 15,
        angle_bins: int = 90,
        loss_weights: dict[str, float] = None,
        use_varifocal: bool = True,
        assigner_config: dict[str, float] = None,
        focal_config: dict[str, float] = None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.angle_bins = angle_bins
        self.use_varifocal = use_varifocal

        # Default configurations
        self.loss_weights = loss_weights or {"cls": 1.0, "box": 2.5, "angle": 0.05}
        assigner_config = assigner_config or {"topk": 13, "alpha": 1.0, "beta": 6.0}
        focal_config = focal_config or {"alpha": 0.75, "gamma": 2.0}

        self.focal_alpha = focal_config["alpha"]
        self.focal_gamma = focal_config["gamma"]

        # Angle encoding parameters
        self.angle_scale = math.pi / 2 / angle_bins

        # Loss components
        self.assigner = RotatedTaskAlignedAssigner(**assigner_config)
        self.box_loss_fn = ProbIoULoss()

    def forward(
        self,
        cls_logits: torch.Tensor,
        reg_dist: torch.Tensor,
        raw_angles: torch.Tensor,
        targets: dict[str, torch.Tensor],
        anchor_points: torch.Tensor,
        stride_tensor: torch.Tensor,
        angle_proj: torch.Tensor,
    ) -> LossComponents:
        """Compute detection losses using only raw predictions.

        Args:
            cls_logits: [B, N, C] - Raw classification logits
            reg_dist: [B, N, 4] - Raw distance predictions
            raw_angles: [B, N, angle_bins+1] - Raw angle logits
            targets: Ground truth data:
                - labels: [B, M, 1] - Class labels [0, num_classes-1]
                - boxes: [B, M, 5] - Rotated GT boxes [cx, cy, w, h, angle]
                    * cx, cy, w, h: in absolute pixels
                    * angle: in radians, should be in range [0, π/2)
                - valid_mask: [B, M, 1] - Valid target mask
            anchor_points: [1, N, 2] - Anchor point coordinates
            stride_tensor: [1, N, 1] - Stride values
            angle_proj: [angle_bins+1] - Angle projection weights

        Returns:
            Dictionary with loss components:
                - total: Combined weighted loss
                - cls: Classification loss
                - box: Box regression loss
                - angle: Angle distribution loss
        """
        gt_labels = targets["labels"]  # [B, M, 1]
        gt_boxes = targets["boxes"]  # [B, M, 5]
        valid_mask = targets["valid_mask"]  # [B, M, 1]

        batch_size, num_anchors = cls_logits.shape[:2]
        num_targets = gt_boxes.shape[1]

        # Handle empty targets
        if num_targets == 0:
            return self._empty_losses(cls_logits, reg_dist, raw_angles)

        # Process raw predictions for assignment
        cls_scores = torch.sigmoid(cls_logits)
        decoded_boxes = decode_ppyoloer_boxes(anchor_points, reg_dist, raw_angles, stride_tensor, angle_proj)

        # Task-aligned assignment using processed outputs
        assigned_labels, assigned_boxes, assigned_scores = self.assigner(
            cls_scores.detach(),
            decoded_boxes.detach(),
            anchor_points,
            gt_labels,
            gt_boxes,
            valid_mask,
            self.num_classes,
        )

        # Compute individual losses using raw predictions
        loss_cls = self._classification_loss(cls_logits, assigned_scores, assigned_labels)
        loss_box = self._box_loss(decoded_boxes, assigned_boxes, assigned_labels, assigned_scores)
        loss_angle = self._angle_loss(raw_angles, assigned_boxes, assigned_labels)

        # Weighted total loss
        total_loss = (
            self.loss_weights["cls"] * loss_cls
            + self.loss_weights["box"] * loss_box
            + self.loss_weights["angle"] * loss_angle
        )

        return LossComponents(total=total_loss, cls=loss_cls, box=loss_box, angle=loss_angle)

    def _classification_loss(
        self, pred_logits: torch.Tensor, target_scores: torch.Tensor, target_labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute classification loss using focal or varifocal loss.

        Args:
            pred_logits: [B, N, C] - Raw classification logits
            target_scores: [B, N, C] - Assigned target scores from assigner
            target_labels: [B, N] - Assigned class labels

        Returns:
            Scalar classification loss normalized by sum of assigned scores
        """
        # Normalize by sum of assigned scores (quality-aware normalization)
        assigned_scores_sum = torch.clamp(target_scores.sum(), min=1.0)
        pred_sigmoid = torch.sigmoid(pred_logits)

        if self.use_varifocal:
            # Varifocal loss
            # Create one-hot label from assigned_labels
            target_labels_onehot = F.one_hot(target_labels, self.num_classes + 1)[..., :-1].float()

            # Varifocal loss weighting: alpha * p^gamma * (1-q) + t * q
            # where p=pred_sigmoid, q=target_labels_onehot, t=target_scores
            focal_weight = (
                self.focal_alpha * pred_sigmoid.pow(self.focal_gamma) * (1.0 - target_labels_onehot)
                + target_scores * target_labels_onehot
            )

            # Compute BCE loss first, then apply weighting
            base_loss = F.binary_cross_entropy_with_logits(pred_logits, target_scores, reduction="none")
            loss = (base_loss * focal_weight).sum()
        else:
            # Standard focal loss - compute BCE first, then apply focal weighting
            base_loss = F.binary_cross_entropy_with_logits(pred_logits, target_scores, reduction="none")
            focal_weight = (pred_sigmoid - target_scores).pow(self.focal_gamma)
            if self.focal_alpha > 0:
                alpha_weight = self.focal_alpha * target_scores + (1 - self.focal_alpha) * (1 - target_scores)
                focal_weight = focal_weight * alpha_weight

            loss = (base_loss * focal_weight).sum()

        return loss / assigned_scores_sum

    def _box_loss(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        target_labels: torch.Tensor,
        target_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Compute box regression loss using rotated IoU.

        Args:
            pred_boxes: [B, N, 5] - Predicted rotated boxes (already decoded)
            target_boxes: [B, N, 5] - Assigned target boxes
            target_labels: [B, N] - Assigned class labels
            target_scores: [B, N, C] - Assigned target scores

        Returns:
            Scalar box regression loss weighted by target scores
        """
        positive_mask = target_labels != self.num_classes
        num_positives = positive_mask.sum()

        if num_positives == 0:
            return pred_boxes.sum() * 0.0

        # Extract positive predictions and targets
        pred_pos = pred_boxes[positive_mask]
        target_pos = target_boxes[positive_mask]

        # Compute IoU loss
        iou_loss = self.box_loss_fn(pred_pos, target_pos)

        # Weight by target scores
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
        """Compute angle distribution loss using distribution focal loss.

        Args:
            pred_angles: [B, N, angle_bins+1] - Raw angle logits
            target_boxes: [B, N, 5] - Assigned target boxes
            target_labels: [B, N] - Assigned class labels

        Returns:
            Scalar angle distribution loss using linear interpolation between bins
        """
        positive_mask = target_labels != self.num_classes
        num_positives = positive_mask.sum()

        if num_positives == 0:
            return pred_angles.sum() * 0.0

        # Extract positive samples
        pred_angle_pos = pred_angles[positive_mask]  # [P, bins+1]
        target_angle_pos = target_boxes[positive_mask][:, 4]  # [P]

        # Convert target angles to bin indices (continuous)
        target_bins = target_angle_pos / self.angle_scale
        target_bins = torch.clamp(target_bins, 0, self.angle_bins - 0.01)

        # Distribution focal loss with linear interpolation
        left_idx = target_bins.long()
        right_idx = torch.clamp(left_idx + 1, max=self.angle_bins)

        left_weight = right_idx.float() - target_bins
        right_weight = 1.0 - left_weight

        loss_left = F.cross_entropy(pred_angle_pos, left_idx, reduction="none") * left_weight
        loss_right = F.cross_entropy(pred_angle_pos, right_idx, reduction="none") * right_weight

        angle_loss = (loss_left + loss_right).mean()

        return angle_loss

    def _empty_losses(
        self, cls_logits: torch.Tensor, reg_dist: torch.Tensor, raw_angles: torch.Tensor
    ) -> LossComponents:
        """Return zero losses maintaining gradient flow for empty targets.

        Args:
            cls_logits: Classification logits
            reg_dist: Distance predictions
            raw_angles: Raw angle logits

        Returns:
            Named tuple with zero losses that maintain gradient computation
        """
        loss_cls = cls_logits.sum() * 0.0
        loss_box = reg_dist.sum() * 0.0
        loss_angle = raw_angles.sum() * 0.0

        total_loss = loss_cls + loss_box + loss_angle

        return LossComponents(total=total_loss, cls=loss_cls, box=loss_box, angle=loss_angle)
