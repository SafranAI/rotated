import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rotated.assigner import RotatedTaskAlignedAssigner
from rotated.iou.prob_iou import ProbIoULoss


class RotatedDetectionLoss(nn.Module):
    """Rotated detection loss with optimal separation of concerns.

    Receives exactly what each component needs (no conversions)
    Scores for assignment, logits for loss computation
    Decoded boxes for assignment, raw angles for loss
    Clean separation: assignment → loss computation

    The criterion expects both raw and processed predictions from the head.
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
        focal_config = focal_config or {"alpha": 0.25, "gamma": 2.0}

        self.focal_alpha = focal_config["alpha"]
        self.focal_gamma = focal_config["gamma"]

        # Angle encoding parameters
        self.angle_scale = math.pi / 2 / angle_bins

        # Loss components
        self.assigner = RotatedTaskAlignedAssigner(**assigner_config)
        self.box_loss_fn = ProbIoULoss()

    def forward(
        self,
        cls_scores: torch.Tensor,
        cls_logits: torch.Tensor,
        decoded_boxes: torch.Tensor,
        raw_angles: torch.Tensor,
        targets: dict[str, torch.Tensor],
        anchor_points: torch.Tensor,
        stride_tensor: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute detection losses with clean separation of assignment and loss computation.

        Args:
            cls_scores: (B, N, C) - Classification scores (post-sigmoid) for assignment
            cls_logits: (B, N, C) - Raw classification logits (pre-sigmoid) for loss computation
            decoded_boxes: (B, N, 5) - Decoded rotated boxes for assignment [cx, cy, w, h, angle]
            raw_angles: (B, N, angle_bins+1) - Raw angle logits (pre-softmax) for loss computation
            targets: Dictionary with ground truth data:
                - labels: (B, M, 1) - Class labels [0, num_classes-1]
                - boxes: (B, M, 5) - Rotated GT boxes [cx, cy, w, h, angle]
                - valid_mask: (B, M, 1) - Valid target mask
            anchor_points: (1, N, 2) - Anchor point coordinates from head
            stride_tensor: (1, N, 1) - Stride values from head

        Returns:
            Dictionary with loss components:
                - total: Combined weighted loss
                - cls: Classification loss
                - box: Box regression loss
                - angle: Angle distribution loss

        Shape:
            - B: Batch size
            - N: Number of anchor points (sum across all FPN levels)
            - M: Maximum number of ground truth objects per image
            - C: Number of classes
        """
        gt_labels = targets["labels"]  # [B, M, 1]
        gt_boxes = targets["boxes"]  # [B, M, 5]
        valid_mask = targets["valid_mask"]  # [B, M, 1]

        batch_size, num_anchors = cls_scores.shape[:2]
        num_targets = gt_boxes.shape[1]

        # Handle empty targets
        if num_targets == 0:
            return self._empty_losses(cls_scores, cls_logits, decoded_boxes, raw_angles)

        # Task-aligned assignment using scores and decoded boxes (no conversions!)
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
        loss_angle = self._angle_loss(raw_angles, assigned_boxes, assigned_labels, assigned_scores)

        # Weighted total loss
        total_loss = (
            self.loss_weights["cls"] * loss_cls
            + self.loss_weights["box"] * loss_box
            + self.loss_weights["angle"] * loss_angle
        )

        return {"total": total_loss, "cls": loss_cls, "box": loss_box, "angle": loss_angle}

    def _classification_loss(
        self, pred_logits: torch.Tensor, target_scores: torch.Tensor, target_labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute classification loss using focal or varifocal loss.

        Args:
            pred_logits: (B, N, C) - Raw classification logits (pre-sigmoid)
            target_scores: (B, N, C) - Assigned target scores from assigner
            target_labels: (B, N) - Assigned class labels

        Returns:
            Scalar classification loss normalized by number of positive samples
        """
        num_positives = torch.clamp(target_scores.sum(), min=1.0)

        if self.use_varifocal:
            # Varifocal loss
            target_labels_onehot = F.one_hot(target_labels, self.num_classes + 1)[..., :-1].float()
            pred_sigmoid = torch.sigmoid(pred_logits)

            focal_weight = target_scores * target_labels_onehot + (
                pred_sigmoid.pow(self.focal_gamma) * (1 - target_labels_onehot)
            )

            loss = F.binary_cross_entropy_with_logits(
                pred_logits, target_scores, weight=focal_weight.detach(), reduction="sum"
            )
        else:
            # Standard focal loss
            pred_sigmoid = torch.sigmoid(pred_logits)
            weight = (pred_sigmoid - target_scores).abs().pow(self.focal_gamma)
            if self.focal_alpha > 0:
                alpha_weight = self.focal_alpha * target_scores + (1 - self.focal_alpha) * (1 - target_scores)
                weight = weight * alpha_weight

            loss = F.binary_cross_entropy_with_logits(pred_logits, target_scores, weight=weight, reduction="sum")

        return loss / num_positives

    def _box_loss(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        target_labels: torch.Tensor,
        target_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Compute box regression loss using rotated IoU.

        Args:
            pred_boxes: (B, N, 5) - Predicted rotated boxes (already decoded)
            target_boxes: (B, N, 5) - Assigned target boxes
            target_labels: (B, N) - Assigned class labels
            target_scores: (B, N, C) - Assigned target scores

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
        target_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Compute angle distribution loss using distribution focal loss.

        Uses the original PP-YOLOE-R angle loss design with distribution
        focal loss and linear interpolation between angle bins, following
        the original PaddlePaddle implementation pattern.

        Args:
            pred_angles: (B, N, angle_bins+1) - Raw angle logits (pre-softmax)
            target_boxes: (B, N, 5) - Assigned target boxes
            target_labels: (B, N) - Assigned class labels
            target_scores: (B, N, C) - Assigned target scores

        Returns:
            Scalar angle distribution loss using linear interpolation between bins
        """
        positive_mask = target_labels != self.num_classes
        num_positives = positive_mask.sum()

        if num_positives == 0:
            # Following original pattern for empty case
            return torch.zeros([1], device=pred_angles.device, dtype=pred_angles.dtype)

        # Extract positive samples
        pred_angle_pos = pred_angles[positive_mask]  # [P, bins+1]
        target_angle_pos = target_boxes[positive_mask][:, 4]  # [P]

        # Convert target angles to bin indices (continuous)
        # Following original: angle / half_pi_bin, clipped to valid range
        target_bins = target_angle_pos / self.angle_scale
        target_bins = torch.clamp(target_bins, 0, self.angle_bins - 0.01)

        # Distribution focal loss with linear interpolation (following original _df_loss)
        left_idx = target_bins.long()
        right_idx = torch.clamp(left_idx + 1, max=self.angle_bins)

        left_weight = right_idx.float() - target_bins
        right_weight = 1.0 - left_weight

        loss_left = F.cross_entropy(pred_angle_pos, left_idx, reduction="none") * left_weight
        loss_right = F.cross_entropy(pred_angle_pos, right_idx, reduction="none") * right_weight

        # Following original pattern: mean over the last dimension, then sum
        angle_loss = (loss_left + loss_right).mean()

        return angle_loss

    def _empty_losses(
        self, cls_scores: torch.Tensor, cls_logits: torch.Tensor, decoded_boxes: torch.Tensor, raw_angles: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return zero losses maintaining gradient flow for empty targets.

        Following the original PaddlePaddle implementation pattern for robust
        zero loss computation that preserves gradients properly.

        Args:
            cls_scores: Classification scores
            cls_logits: Classification logits
            decoded_boxes: Decoded box predictions
            raw_angles: Raw angle logits

        Returns:
            Dictionary with zero losses that maintain gradient computation
        """
        # Use more robust zero loss computation following original implementation
        # This ensures proper gradient flow even with empty targets
        loss_cls = cls_logits.sum() * 0.0
        loss_box = decoded_boxes.sum() * 0.0
        loss_angle = raw_angles.sum() * 0.0

        total_loss = loss_cls + loss_box + loss_angle

        return {"total": total_loss, "cls": loss_cls, "box": loss_box, "angle": loss_angle}


if __name__ == "__main__":
    print("Testing RotatedDetectionLoss")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize loss function
    criterion = RotatedDetectionLoss(
        num_classes=15, angle_bins=90, use_varifocal=True, loss_weights={"cls": 1.0, "box": 2.5, "angle": 0.1}
    ).to(device)

    # Test parameters
    batch_size, num_anchors, num_classes = 2, 1000, 15
    num_targets = 5

    # Create test inputs (as would come from head)
    cls_logits = torch.randn(batch_size, num_anchors, num_classes, device=device, requires_grad=True)
    cls_scores = torch.sigmoid(cls_logits.detach())  # Derived from logits

    decoded_boxes = (
        torch.randn(batch_size, num_anchors, 5, device=device, requires_grad=True) * 50 + 100
    )  # Reasonable coordinate ranges

    raw_angles = torch.randn(
        batch_size,
        num_anchors,
        91,  # 90 bins + 1
        device=device,
        requires_grad=True,
    )

    # Create test targets
    targets = {
        "labels": torch.randint(0, num_classes, (batch_size, num_targets, 1), device=device),
        "boxes": torch.randn(batch_size, num_targets, 5, device=device) * 50 + 100,
        "valid_mask": torch.ones(batch_size, num_targets, 1, device=device),
    }

    # Create anchor info (as would come from head)
    anchor_points = torch.randn(1, num_anchors, 2, device=device) * 100
    stride_tensor = torch.ones(1, num_anchors, 1, device=device) * 16

    # Test forward pass
    losses = criterion(cls_scores, cls_logits, decoded_boxes, raw_angles, targets, anchor_points, stride_tensor)

    # Test backward pass
    losses["total"].backward()

    # Test empty targets
    empty_targets = {
        "labels": torch.zeros(batch_size, 0, 1, dtype=torch.long, device=device),
        "boxes": torch.zeros(batch_size, 0, 5, device=device),
        "valid_mask": torch.zeros(batch_size, 0, 1, device=device),
    }

    # Fresh inputs for empty test
    empty_cls_logits = torch.randn(batch_size, num_anchors, num_classes, device=device, requires_grad=True)
    empty_cls_scores = torch.sigmoid(empty_cls_logits.detach())
    empty_boxes = torch.randn(batch_size, num_anchors, 5, device=device, requires_grad=True) * 50
    empty_angles = torch.randn(batch_size, num_anchors, 91, device=device, requires_grad=True)

    empty_losses = criterion(
        empty_cls_scores, empty_cls_logits, empty_boxes, empty_angles, empty_targets, anchor_points, stride_tensor
    )

    print("Forward and backward pass successful")
