import torch


class RotatedIoUCalculator:
    """IoU calculator for rotated bounding boxes using ProbIoU.

    Args:
        eps: Small constant for numerical stability in ProbIoU computation
    """

    def __init__(self, eps: float = 1e-3):
        from rotated.iou.prob_iou import ProbIoU

        self.prob_iou = ProbIoU(eps=eps)

    def __call__(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute pairwise IoU matrix between two sets of rotated boxes.

        Args:
            boxes1: First set of rotated boxes [N, 5] - (cx, cy, w, h, angle)
            boxes2: Second set of rotated boxes [M, 5] - (cx, cy, w, h, angle)

        Returns:
            IoU matrix [N, M] with values in [0, 1]
        """
        N, M = boxes1.shape[0], boxes2.shape[0]

        if N == 0 or M == 0:
            return torch.zeros(N, M, device=boxes1.device, dtype=boxes1.dtype)

        # Expand for pairwise computation
        # boxes1: [N, 1, 5] -> [N, M, 5]
        boxes1_expanded = boxes1.unsqueeze(1).expand(-1, M, -1)
        # boxes2: [1, M, 5] -> [N, M, 5]
        boxes2_expanded = boxes2.unsqueeze(0).expand(N, -1, -1)

        # Flatten for element-wise computation: [N*M, 5]
        boxes1_flat = boxes1_expanded.reshape(-1, 5)
        boxes2_flat = boxes2_expanded.reshape(-1, 5)

        # Compute element-wise IoU: [N*M]
        iou_flat = self.prob_iou(boxes1_flat, boxes2_flat)

        # Reshape back to matrix: [N, M]
        iou_matrix = iou_flat.view(N, M)

        return iou_matrix


class HorizontalIoUCalculator:
    """IoU calculator for horizontal (axis-aligned) bounding boxes.

    Args:
        eps: Small constant to prevent division by zero
    """

    def __init__(self, eps: float = 1e-7):
        self.eps = eps

    def __call__(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute pairwise IoU matrix between two sets of horizontal boxes.

        Args:
            boxes1: First set of boxes [N, 4] - (cx, cy, w, h)
            boxes2: Second set of boxes [M, 4] - (cx, cy, w, h)

        Returns:
            IoU matrix [N, M] with values in [0, 1]
        """
        N, M = boxes1.shape[0], boxes2.shape[0]

        if N == 0 or M == 0:
            return torch.zeros(N, M, device=boxes1.device, dtype=boxes1.dtype)

        boxes1_xyxy = self._cxcywh_to_xyxy(boxes1)
        boxes2_xyxy = self._cxcywh_to_xyxy(boxes2)

        # Compute intersection
        x1_inter = torch.max(boxes1_xyxy[:, None, 0], boxes2_xyxy[None, :, 0])
        y1_inter = torch.max(boxes1_xyxy[:, None, 1], boxes2_xyxy[None, :, 1])
        x2_inter = torch.min(boxes1_xyxy[:, None, 2], boxes2_xyxy[None, :, 2])
        y2_inter = torch.min(boxes1_xyxy[:, None, 3], boxes2_xyxy[None, :, 3])

        # Intersection area
        inter_area = torch.clamp(x2_inter - x1_inter, min=0) * torch.clamp(y2_inter - y1_inter, min=0)

        # Box areas
        area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1])
        area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1])

        # Union area
        union_area = area1[:, None] + area2[None, :] - inter_area

        # IoU
        iou = inter_area / (union_area + self.eps)
        return torch.clamp(iou, 0.0, 1.0)

    def _cxcywh_to_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert center format to corner format."""
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - w * 0.5
        y1 = cy - h * 0.5
        x2 = cx + w * 0.5
        y2 = cy + h * 0.5
        return torch.stack([x1, y1, x2, y2], dim=-1)


if __name__ == "__main__":
    print("Testing IoU calculators")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test rotated IoU calculator
    rotated_calc = RotatedIoUCalculator()

    boxes1 = torch.tensor([[50, 50, 30, 20, 0.0], [100, 100, 40, 30, 0.785]], device=device)
    boxes2 = torch.tensor([[50, 50, 30, 20, 0.0], [55, 55, 30, 20, 0.785]], device=device)

    rotated_iou = rotated_calc(boxes1, boxes2)
    print(f"Rotated IoU matrix:\n{rotated_iou}")

    # Test horizontal IoU calculator
    horizontal_calc = HorizontalIoUCalculator()

    h_boxes1 = torch.tensor([[50, 50, 30, 20], [100, 100, 40, 30]], device=device)
    h_boxes2 = torch.tensor([[50, 50, 30, 20], [55, 55, 30, 20]], device=device)

    horizontal_iou = horizontal_calc(h_boxes1, h_boxes2)
    print(f"Horizontal IoU matrix:\n{horizontal_iou}")

    # Test identical boxes (should be 1.0)
    identical_iou = horizontal_calc(h_boxes1[:1], h_boxes1[:1])
    print(f"Identical boxes IoU: {identical_iou.item():.6f}")

    # Test empty cases
    empty_boxes = torch.empty(0, 5, device=device)
    non_empty_boxes = torch.tensor([[50, 50, 30, 20, 0.0]], device=device)

    empty_iou = rotated_calc(empty_boxes, non_empty_boxes)
    print(f"Empty vs non-empty IoU shape: {empty_iou.shape}")

    # Test different sizes
    boxes_3 = torch.tensor([[30, 30, 20, 15, 0.0], [70, 70, 25, 18, 0.5], [110, 110, 30, 20, 1.0]], device=device)
    mixed_iou = rotated_calc(boxes1, boxes_3)  # [2, 3] matrix
    print(f"Mixed size IoU matrix shape: {mixed_iou.shape}")
    print(f"Mixed size IoU matrix:\n{mixed_iou}")
