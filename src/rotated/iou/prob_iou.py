# Adapted from original paper implementation:
# https://github.com/ProbIOU/PROBIOU-SSD/blob/8ea89792881c61c98ea1ba01a6efc69566d467b7/utils/box/box_utils.py#L163

import torch


class ProbIoU:
    """Probabilistic IoU for Rotated Bounding Boxes.

    Computes IoU between rotated bounding boxes by treating them as 2D Gaussian distributions.
    This provides differentiable IoU computation with better gradient flow compared to discrete polygon IoU.

    Reference:
        Title: "Gaussian Bounding Boxes and Probabilistic Intersection-over-Union for Object Detection"
        Authors: Jeffri M. Llerena, Luis Felipe Zeni, Lucas N. Kristen, Claudio Jung
        Paper link: https://arxiv.org/pdf/2106.06072
    """

    def __init__(self, eps: float = 1e-7):
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
        center_x1, center_y1, variance_a1, variance_b1, angle1 = self._gbb_form(pred_boxes)
        center_x2, center_y2, variance_a2, variance_b2, angle2 = self._gbb_form(target_boxes)

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
                    (covariance_a1 * covariance_b1 - covariance_c1.pow(2)).clamp_(0)
                    * (covariance_a2 * covariance_b2 - covariance_c2.pow(2)).clamp_(0)
                )
                + self.eps
            )
            + self.eps
        )

        # Bhattacharyya distance and conversion to IoU
        bhattacharyya_distance = (term1 + term2 + term3).clamp(self.eps, 100.0)
        l1_loss = torch.sqrt(1.0 - torch.exp(-bhattacharyya_distance) + self.eps)
        iou = 1.0 - l1_loss

        return iou.clamp(0.0, 1.0)

    def _gbb_form(
        self, boxes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert rotated bounding boxes to Gaussian Bounding Box form."""
        center_x, center_y, width, height, angle = boxes.unbind(-1)
        # Convert w,h to variance: σ² = (w²/12, h²/12) for uniform distribution
        variance_a = width.pow(2) / 12.0
        variance_b = height.pow(2) / 12.0
        return center_x, center_y, variance_a, variance_b, angle

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
