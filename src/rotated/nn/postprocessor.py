"""Post-processing module for rotated object detection with torch.jit.trace compatibility."""

import torch
import torch.nn as nn

from rotated.boxes.nms import multiclass_nms


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
        nms_mode: NMS algorithm mode. Options:
            - "sequential": Original implementation (lowest memory, slowest)
            - "vectorized": Standard NMS with vectorized IoU (default, ~20-30x faster)
            - "fast": Fast-NMS algorithm (~50-100x faster, slightly more aggressive)
        n_samples: Number of samples for IoU computation, using approx SDF-L1 method
        eps: Epsilon for numerical stability

    Note: This module expects batched input [B, N, 5] for boxes and [B, N] for scores/labels.
    """

    def __init__(
        self,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        detections_per_img: int = 300,
        topk_candidates: int = 1000,
        nms_mode: str = "vectorized",
        n_samples: int = 40,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates
        self.nms_mode = nms_mode
        self.n_samples = n_samples
        self.eps = eps

    def forward(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply detection post-processing pipeline.

        Pipeline: score filtering → topk selection → NMS → detections_per_img limit

        Args:
            boxes: Batched boxes [B, N, 5] format [cx, cy, w, h, angle]
            scores: Batched scores [B, N]
            labels: Batched labels [B, N]

        Returns:
            Tuple of (boxes, scores, labels):
            - boxes: [B, detections_per_img, 5]
            - scores: [B, detections_per_img]
            - labels: [B, detections_per_img]

        Raises:
            ValueError: If input tensors are not 3D batched tensors
        """
        if boxes.dim() != 3:
            raise ValueError(
                f"Expected 3D batched input [B, N, 5], got {boxes.dim()}D. For single samples, use boxes.unsqueeze(0)"
            )

        return _postprocess_batch(
            boxes=boxes,
            scores=scores,
            labels=labels,
            score_thresh=self.score_thresh,
            nms_thresh=self.nms_thresh,
            detections_per_img=self.detections_per_img,
            topk_candidates=self.topk_candidates,
            nms_mode=self.nms_mode,
            n_samples=self.n_samples,
            eps=self.eps,
        )

    def extra_repr(self) -> str:
        return (
            f"score_thresh={self.score_thresh}, "
            f"nms_thresh={self.nms_thresh}, "
            f"detections_per_img={self.detections_per_img}, "
            f"topk_candidates={self.topk_candidates}, "
            f"nms_mode={self.nms_mode}"
        )


@torch.jit.script_if_tracing
def _postprocess_batch(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    score_thresh: float,
    nms_thresh: float,
    detections_per_img: int,
    topk_candidates: int,
    nms_mode: str,
    n_samples: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Process a batch of detections with dynamic batch size.

    This function is scripted when tracing, allowing dynamic batch sizes.

    Args:
        boxes: Batched boxes [B, N, 5]
        scores: Batched scores [B, N]
        labels: Batched labels [B, N]
        score_thresh: Score threshold
        nms_thresh: NMS threshold
        detections_per_img: Max detections per image
        topk_candidates: Top-k before NMS
        nms_mode: NMS mode
        n_samples: IoU samples
        eps: Epsilon

    Returns:
        Tuple of (boxes, scores, labels) for the batch
    """
    batch_size = boxes.size(0)
    device = boxes.device
    dtype = boxes.dtype

    output_boxes = torch.zeros((batch_size, detections_per_img, 5), device=device, dtype=dtype)
    output_scores = torch.zeros((batch_size, detections_per_img), device=device, dtype=dtype)
    output_labels = torch.full((batch_size, detections_per_img), -1, device=device, dtype=labels.dtype)

    for batch_idx in range(batch_size):
        batch_boxes, batch_scores, batch_labels = _postprocess_single(
            boxes=boxes[batch_idx],
            scores=scores[batch_idx],
            labels=labels[batch_idx],
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            detections_per_img=detections_per_img,
            topk_candidates=topk_candidates,
            nms_mode=nms_mode,
            n_samples=n_samples,
            eps=eps,
        )
        output_boxes[batch_idx] = batch_boxes
        output_scores[batch_idx] = batch_scores
        output_labels[batch_idx] = batch_labels

    return output_boxes, output_scores, output_labels


@torch.jit.script_if_tracing
def _postprocess_single(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    score_thresh: float,
    nms_thresh: float,
    detections_per_img: int,
    topk_candidates: int,
    nms_mode: str,
    n_samples: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Post-process detections for a single batch element.

    Pipeline: score filtering → topk selection → NMS → detections_per_img limit

    Args:
        boxes: Boxes [N, 5] format [cx, cy, w, h, angle]
        scores: Scores [N]
        labels: Labels [N]
        score_thresh: Score threshold for filtering
        nms_thresh: IoU threshold for NMS
        detections_per_img: Maximum detections to keep
        topk_candidates: Number of top candidates before NMS
        nms_mode: NMS algorithm mode
        n_samples: Number of samples for IoU computation
        eps: Epsilon for numerical stability

    Returns:
        Tuple of (boxes, scores, labels) after post-processing
        - boxes: [detections_per_img, 5] (padded with zeros)
        - scores: [detections_per_img] (padded with zeros)
        - labels: [detections_per_img] (padded with -1)
    """
    device = boxes.device
    dtype = boxes.dtype

    output_boxes = torch.zeros((detections_per_img, 5), device=device, dtype=dtype)
    output_scores = torch.zeros((detections_per_img,), device=device, dtype=dtype)
    output_labels = torch.full((detections_per_img,), -1, device=device, dtype=labels.dtype)

    keep_mask = scores > score_thresh
    if not keep_mask.any():
        return output_boxes, output_scores, output_labels

    filtered_scores = scores[keep_mask]
    filtered_boxes = boxes[keep_mask]
    filtered_labels = labels[keep_mask]

    num_filtered = filtered_scores.size(0)
    if num_filtered > topk_candidates:
        top_scores, top_idxs = filtered_scores.topk(topk_candidates)
        filtered_scores = top_scores
        filtered_boxes = filtered_boxes[top_idxs]
        filtered_labels = filtered_labels[top_idxs]

    keep_indices = multiclass_nms(
        boxes=filtered_boxes,
        scores=filtered_scores,
        labels=filtered_labels,
        iou_threshold=nms_thresh,
        nms_mode=nms_mode,
        n_samples=n_samples,
        eps=eps,
    )

    if keep_indices.numel() == 0:
        return output_boxes, output_scores, output_labels

    keep_indices = keep_indices[:detections_per_img]

    num_keep = keep_indices.size(0)
    output_boxes[:num_keep] = filtered_boxes[keep_indices]
    output_scores[:num_keep] = filtered_scores[keep_indices]
    output_labels[:num_keep] = filtered_labels[keep_indices]

    return output_boxes, output_scores, output_labels
