"""Post-processing module for rotated object detection."""

import torch
import torch.nn as nn

from rotated.boxes.nms import postprocess_detections


class DetectionPostProcessor(nn.Module):
    """Post-processing module for rotated object detection.

    Wraps the postprocess_detections function in a nn.Module.

    Args:
        score_thresh: Score threshold for filtering detections
        nms_thresh: IoU threshold for NMS
        detections_per_img: Maximum number of detections to keep per image
        topk_candidates: Number of top candidates to consider before NMS
    """

    def __init__(
        self,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        detections_per_img: int = 300,
        topk_candidates: int = 1000,
    ):
        super().__init__()
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates

    def forward(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply post-processing pipeline.

        Args:
            boxes: Predicted boxes [N, 5] or [B, N, 5]
            scores: Predicted scores [N] or [B, N]
            labels: Predicted labels [N] or [B, N]

        Returns:
            Tuple of (filtered_boxes, filtered_labels, filtered_scores)
        """
        return postprocess_detections(
            boxes=boxes,
            scores=scores,
            labels=labels,
            score_thresh=self.score_thresh,
            nms_thresh=self.nms_thresh,
            detections_per_img=self.detections_per_img,
            topk_candidates=self.topk_candidates,
        )

    def extra_repr(self) -> str:
        return (
            f"score_thresh={self.score_thresh}, "
            f"nms_thresh={self.nms_thresh}, "
            f"detections_per_img={self.detections_per_img}, "
            f"topk_candidates={self.topk_candidates}, "
        )
