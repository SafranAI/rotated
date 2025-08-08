from rotated.assigners.base import BaseTaskAlignedAssigner
from rotated.assigners.calculators import RotatedIoUCalculator


class RotatedTaskAlignedAssigner(BaseTaskAlignedAssigner):
    """Task-aligned assigner specifically for rotated bounding boxes.

    Expects rotated boxes in format [cx, cy, w, h, angle] where:
    - cx, cy: Center coordinates in absolute pixels
    - w, h: Width and height in absolute pixels
    - angle: Rotation angle in radians

    Args:
        topk: Number of top candidates to select per GT object
        alpha: Exponent for classification score in alignment computation
        beta: Exponent for IoU score in alignment computation
        eps: Small value to prevent division by zero in assignment
        iou_eps: Small constant for numerical stability in IoU computation
    """

    def __init__(
        self,
        topk: int = 13,
        alpha: float = 1.0,
        beta: float = 6.0,
        eps: float = 1e-9,
        iou_eps: float = 1e-3,
    ):
        super().__init__(
            iou_calculator=RotatedIoUCalculator(eps=iou_eps),
            box_format='rotated',
            topk=topk,
            alpha=alpha,
            beta=beta,
            eps=eps
        )
