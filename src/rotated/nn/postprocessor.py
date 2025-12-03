"""Post-processing module for rotated object detection."""

import torch
import torch.nn as nn

from rotated.boxes.nms import NMS, IoUMethods


class DetectionPostProcessor(nn.Module):
    """Post-processing module for rotated object detection.

    Wraps the postprocess_detections function in a nn.Module.

    Args:
        score_thresh: Score threshold for filtering detections
        nms_thresh: IoU threshold for NMS
        detections_per_img: Maximum number of detections to keep per image
        topk_candidates: Number of top candidates to consider before NMS
        iou_method: Method name to compute Intersection Over Union
    """

    def __init__(
        self,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        detections_per_img: int = 300,
        topk_candidates: int = 1000,
        iou_method: IoUMethods = "approx_sdf_l1",
    ):
        super().__init__()
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates
        self.nms = NMS(iou_method=iou_method)

    @torch.jit.script_if_tracing
    def forward(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply detection post-processing pipeline.

        Pipeline: score filtering → topk selection → NMS → detections_per_img limit
        Handles both single and batched inputs automatically.

        Args:
            boxes: Boxes [N, 5] or [B, N, 5]
            scores: Scores [N] or [B, N]
            labels: Labels [N] or [B, N]

        Returns:
            For single input: (boxes, scores, labels) [detections_per_img, 5], [detections_per_img], [detections_per_img]
            For batched input: (boxes, scores, labels) [B, detections_per_img, 5], [B, detections_per_img], [B, detections_per_img]

        Raises:
            ValueError: If input tensors have incompatible dimensions
        """
        # Normalize to batched format
        is_single = boxes.dim() == 2
        if is_single:
            boxes = boxes.unsqueeze(0)
            scores = scores.unsqueeze(0)
            labels = labels.unsqueeze(0)
        elif boxes.dim() != 3:
            raise ValueError(f"Expected 2D or 3D input, got {boxes.dim()}D")

        # Process with unified logic
        output_boxes, output_scores, output_labels = self.nms._postprocess(
            boxes,
            scores,
            labels,
            score_thresh=self.score_thresh,
            nms_thresh=self.nms_thresh,
            detections_per_img=self.detections_per_img,
            topk_candidates=self.topk_candidates,
        )

        # Squeeze back if single sample
        if is_single:
            output_boxes = output_boxes.squeeze(0)
            output_scores = output_scores.squeeze(0)
            output_labels = output_labels.squeeze(0)

        return output_boxes, output_scores, output_labels

    def extra_repr(self) -> str:
        return (
            f"score_thresh={self.score_thresh}, "
            f"nms_thresh={self.nms_thresh}, "
            f"detections_per_img={self.detections_per_img}, "
            f"topk_candidates={self.topk_candidates}, "
        )
