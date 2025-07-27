import torch
import torch.nn as nn


class ProbIoU:
    """Probabilistic IoU for Rotated Bounding Boxes.

    Computes IoU between rotated bounding boxes by treating them as
    2D Gaussian distributions. This provides differentiable IoU computation with
    better gradient flow compared to discrete polygon IoU.

    Reference: "Gaussian Bounding Boxes and Probabilistic Intersection-over-Union for Object Detection"
    """

    def __init__(self, eps: float = 1e-3):
        self.eps = eps

    def __call__(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute ProbIoU between predicted and target rotated boxes.

        Args:
            pred_boxes: Predicted boxes [N, 5] - (x, y, w, h, angle_in_radians)
            target_boxes: Target boxes [N, 5] - (x, y, w, h, angle_in_radians)

        Returns:
            IoU tensor [N] - ProbIoU values for each box pair (0 to 1)
        """
        # Convert to Gaussian Bounding Box form
        gbboxes1 = self._gbb_form(pred_boxes)
        gbboxes2 = self._gbb_form(target_boxes)

        center_x1, center_y1, variance_a1, variance_b1, angle1 = gbboxes1.unbind(-1)
        center_x2, center_y2, variance_a2, variance_b2, angle2 = gbboxes2.unbind(-1)

        # Get rotated covariance elements
        covariance_a1, covariance_b1, covariance_c1 = self._rotated_form(variance_a1, variance_b1, angle1)
        covariance_a2, covariance_b2, covariance_c2 = self._rotated_form(variance_a2, variance_b2, angle2)

        # Compute Bhattacharyya distance terms
        denominator = (
            (covariance_a1 + covariance_a2) * (covariance_b1 + covariance_b2)
            - (covariance_c1 + covariance_c2).pow(2)
            + self.eps
        )

        term1 = (
            (
                (covariance_a1 + covariance_a2) * (center_y1 - center_y2).pow(2)
                + (covariance_b1 + covariance_b2) * (center_x1 - center_x2).pow(2)
            )
            / denominator
        ) * 0.25

        term2 = (
            ((covariance_c1 + covariance_c2) * (center_x2 - center_x1) * (center_y1 - center_y2)) / denominator
        ) * 0.5

        term3 = 0.5 * torch.log(
            ((covariance_a1 + covariance_a2) * (covariance_b1 + covariance_b2) - (covariance_c1 + covariance_c2).pow(2))
            / (
                4
                * torch.sqrt(
                    (covariance_a1 * covariance_b1 - covariance_c1.pow(2))
                    * (covariance_a2 * covariance_b2 - covariance_c2.pow(2))
                )
                + self.eps
            )
            + self.eps
        )

        # Bhattacharyya distance and conversion to IoU
        bhattacharyya_distance = (term1 + term2 + term3).clamp(self.eps, 100.0)
        l1_loss = torch.sqrt(1.0 - torch.exp(-bhattacharyya_distance) + self.eps)
        iou = 1.0 - l1_loss

        return iou

    def _gbb_form(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert rotated bounding boxes to Gaussian Bounding Box form."""
        center_x, center_y, width, height, angle = boxes.unbind(-1)
        # Convert w,h to variance: σ² = (w²/12, h²/12) for uniform distribution
        variance_a = width.pow(2) / 12.0
        variance_b = height.pow(2) / 12.0
        return torch.stack([center_x, center_y, variance_a, variance_b, angle], dim=-1)

    def _rotated_form(
        self, variance_a: torch.Tensor, variance_b: torch.Tensor, angles: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert box variance to rotated covariance matrix elements."""
        cos_angle = torch.cos(angles)
        sin_angle = torch.sin(angles)

        cos_squared = cos_angle.pow(2)
        sin_squared = sin_angle.pow(2)
        cos_sin = cos_angle * sin_angle

        covariance_a = variance_a * cos_squared + variance_b * sin_squared
        covariance_b = variance_a * sin_squared + variance_b * cos_squared
        covariance_c = (variance_a - variance_b) * cos_sin

        return covariance_a, covariance_b, covariance_c


class ProbIoULoss(nn.Module):
    """ProbIoU Loss for training object detectors."""

    def __init__(self, eps: float = 1e-3, mode: str = "l1", reduction: str = "mean"):
        super().__init__()
        self.probiou = ProbIoU(eps=eps)
        self.mode = mode
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute ProbIoU loss.

        Args:
            pred_boxes: Predicted boxes [N, 5] - (x, y, w, h, angle_in_radians)
            target_boxes: Target boxes [N, 5] - (x, y, w, h, angle_in_radians)

        Returns:
            Loss tensor

        Raises:
            ValueError: If unsupported loss mode
        """
        iou = self.probiou(pred_boxes, target_boxes)

        if self.mode == "iou" or self.mode == "l1":
            loss = 1.0 - iou
        elif self.mode == "l2":
            l1_loss = 1.0 - iou
            l_squared = l1_loss.pow(2)
            loss = -torch.log(1.0 - l_squared + self.probiou.eps)
        else:
            raise ValueError(f"Unsupported loss mode: {self.mode}")

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()

        return loss
