"""Post-processing module for rotated object detection."""

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from rotated.boxes.nms import NMS

if TYPE_CHECKING:
    from rotated.iou import IoUKwargs, IoUMethodName


class DetectionPostProcessor(nn.Module):
    """Post-processing module for rotated object detection.

    Apply the following post-processing steps:
        1. Score filtering
        2. Top-k selection
        3. Non Maximum Suppression
        4. Limit detections per image

    Args:
        score_thresh: Score threshold for filtering detections
        nms_thresh: IoU threshold for NMS
        detections_per_img: Maximum number of detections to keep per image
        topk_candidates: Number of top candidates to consider before NMS
        iou_method: Method name to compute Intersection Over Union
        iou_kwargs: Dictionary with parameters for the IoU method.
    """

    def __init__(
        self,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        detections_per_img: int = 300,
        topk_candidates: int = 1000,
        iou_method: "IoUMethodName" = "approx_sdf_l1",
        iou_kwargs: "IoUKwargs" = None,
    ):
        super().__init__()
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates
        self.nms = NMS(nms_thresh=nms_thresh, iou_method=iou_method, iou_kwargs=iou_kwargs)

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
        output_boxes, output_scores, output_labels = self._postprocess(
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

    @torch.jit.script_if_tracing
    def _postprocess(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
        score_thresh: float,
        nms_thresh: float,
        detections_per_img: int,
        topk_candidates: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unified post-processing logic for batched inputs."""
        batch_size = boxes.size(0)
        device = boxes.device

        # Pre-allocate output tensors
        output_boxes = torch.zeros((batch_size, detections_per_img, 5), device=device, dtype=boxes.dtype)
        output_labels = torch.full((batch_size, detections_per_img), -1, device=device, dtype=labels.dtype)
        output_scores = torch.zeros((batch_size, detections_per_img), device=device, dtype=scores.dtype)

        # Process each batch element
        for batch_idx in range(batch_size):
            batch_boxes = boxes[batch_idx]
            batch_scores = scores[batch_idx]
            batch_labels = labels[batch_idx]

            # Step 1: Score filtering
            keep_mask = batch_scores > score_thresh
            if not keep_mask.any():
                continue

            filtered_scores = batch_scores[keep_mask]
            filtered_boxes = batch_boxes[keep_mask]
            filtered_labels = batch_labels[keep_mask]

            # Step 2: Topk selection
            num_filtered = filtered_scores.size(0)
            if num_filtered > topk_candidates:
                top_scores, top_idxs = filtered_scores.topk(topk_candidates)
                filtered_scores = top_scores
                filtered_boxes = filtered_boxes[top_idxs]
                filtered_labels = filtered_labels[top_idxs]

            # Step 3: NMS
            keep_indices = self.nms.forward(filtered_boxes, filtered_scores, filtered_labels, nms_thresh)
            if keep_indices.numel() == 0:
                continue

            # Step 4: Limit to detections_per_img
            keep_indices = keep_indices[:detections_per_img]

            # Extract and store results
            num_keep = keep_indices.size(0)
            output_boxes[batch_idx, :num_keep] = filtered_boxes[keep_indices]
            output_scores[batch_idx, :num_keep] = filtered_scores[keep_indices]
            output_labels[batch_idx, :num_keep] = filtered_labels[keep_indices]

        return output_boxes, output_scores, output_labels

    def extra_repr(self) -> str:
        return (
            f"score_thresh={self.score_thresh}, "
            f"nms_thresh={self.nms_thresh}, "
            f"detections_per_img={self.detections_per_img}, "
            f"topk_candidates={self.topk_candidates}, "
        )
