"""Module to run the Non Maximum Suppression algorithm with torch.jit.trace compatibility."""

from typing import Literal

import torch
import torch.nn as nn

from rotated.iou import compute_rotated_iou_approx_sdf

NMS_MODE = Literal["sequential", "vectorized", "fast"]


class NMS(nn.Module):
    """Non-Maximum Suppression module for rotated object detection.

    Args:
        nms_thresh: threshold below which detections are removed
        nms_mode: NMS algorithm mode. Options:
            - "sequential": Original implementation (lowest memory, slowest)
            - "vectorized": Standard NMS with vectorized IoU (default, ~20-30x faster)
            - "fast": Fast-NMS algorithm (~50-100x faster, slightly more aggressive)
        n_samples: Number of samples for IoU computation, using approx SDF-L1 method
        eps: Epsilon for numerical stability

    Reference (Fast-NMS):
        Title: "YOLACT Real-time Instance Segmentation"
        Authors: Daniel Bolya, Chong Zhou, Fanyi Xiao, Yong Jae Lee
        Paper link: https://arxiv.org/abs/1904.02689
    """

    def __init__(self, nms_thresh: float = 0.5, nms_mode: str = "vectorized", n_samples: int = 40, eps: float = 1e-7):
        super().__init__()
        self.nms_thresh = nms_thresh
        self.nms_mode = nms_mode
        self.n_samples = n_samples
        self.eps = eps

    def forward(
        self, boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, iou_threshold: float
    ) -> torch.Tensor:
        """Multi-class non-maximum suppression for rotated boxes.

        Args:
            boxes: Rotated boxes [N, 5] format [cx, cy, w, h, angle]
            scores: Confidence scores [N]
            labels: Class labels [N]
            iou_threshold: IoU threshold for suppression

        Returns:
            Indices of boxes to keep, sorted by score descending
        """
        return multiclass_nms(
            boxes=boxes,
            scores=scores,
            labels=labels,
            iou_threshold=iou_threshold,
            nms_mode=self.nms_mode,
            n_samples=self.n_samples,
            eps=self.eps,
        )

    def rotated_nms(self, boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
        """Single-class NMS for rotated bounding boxes.

        Args:
            boxes: Rotated boxes [N, 5] format [cx, cy, w, h, angle]
            scores: Confidence scores [N]
            iou_threshold: IoU threshold for suppression

        Returns:
            Indices of boxes to keep, sorted by score descending
        """
        return rotated_nms(
            boxes=boxes,
            scores=scores,
            iou_threshold=iou_threshold,
            nms_mode=self.nms_mode,
            n_samples=self.n_samples,
            eps=self.eps,
        )

    def batched_multiclass_rotated_nms(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
        iou_threshold: float,
        max_output_per_batch: int = 300,
    ) -> torch.Tensor:
        """Batched multi-class NMS with consistent output shapes.

        Processes multiple samples in parallel and returns padded outputs.
        Note: This function filters boxes with scores <= 0.0.

        Args:
            boxes: Batched boxes [B, N, 5]
            scores: Batched scores [B, N]
            labels: Batched labels [B, N]
            iou_threshold: IoU threshold for suppression
            max_output_per_batch: Maximum detections per batch element

        Returns:
            Indices tensor [B, max_output_per_batch] with -1 padding
        """
        batch_size = boxes.size(0)
        device = boxes.device
        output = torch.full((batch_size, max_output_per_batch), -1, dtype=torch.long, device=device)

        for batch_idx in range(batch_size):
            batch_boxes = boxes[batch_idx]
            batch_scores = scores[batch_idx]
            batch_labels = labels[batch_idx]

            valid_mask = batch_scores > 0.0

            if valid_mask.any():
                valid_boxes = batch_boxes[valid_mask]
                valid_scores = batch_scores[valid_mask]
                valid_labels = batch_labels[valid_mask]

                keep_indices = self.forward(
                    boxes=valid_boxes,
                    scores=valid_scores,
                    labels=valid_labels,
                    iou_threshold=iou_threshold,
                )

                if keep_indices.size(0) > 0:
                    valid_original_indices = torch.where(valid_mask)[0]
                    final_indices = valid_original_indices[keep_indices]
                    num_to_keep = min(final_indices.size(0), max_output_per_batch)
                    output[batch_idx, :num_to_keep] = final_indices[:num_to_keep]

        return output


@torch.jit.script_if_tracing
def _compute_iou_matrix(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    n_samples: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Compute IoU matrix for all box pairs.

    Args:
        boxes: Rotated boxes [N, 5]
        scores: Confidence scores [N]
        n_samples: Number of samples for IoU computation
        eps: Epsilon for numerical stability

    Returns:
        Tuple of (iou_matrix [N, N], order [N], N)
    """
    N = boxes.size(0)
    order = torch.argsort(scores, descending=True)
    boxes_sorted = boxes[order]

    boxes_i = boxes_sorted.unsqueeze(1).expand(N, N, 5).reshape(-1, 5)
    boxes_j = boxes_sorted.unsqueeze(0).expand(N, N, 5).reshape(-1, 5)

    ious_flat = compute_rotated_iou_approx_sdf(pred_boxes=boxes_i, target_boxes=boxes_j, n_samples=n_samples, eps=eps)
    iou_matrix = ious_flat.reshape(N, N)

    iou_matrix = torch.triu(iou_matrix, diagonal=1)

    return iou_matrix, order, N


@torch.jit.script_if_tracing
def _rotated_nms_sequential(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
    n_samples: int,
    eps: float,
) -> torch.Tensor:
    """Sequential NMS without pre-computed IoU matrix.

    Original implementation - slowest but uses minimal memory O(N).

    Args:
        boxes: Rotated boxes [N, 5] format [cx, cy, w, h, angle]
        scores: Confidence scores [N]
        iou_threshold: IoU threshold for suppression
        n_samples: Number of samples for IoU computation
        eps: Epsilon for numerical stability

    Returns:
        Indices of boxes to keep, sorted by score descending
    """
    order = torch.argsort(scores, descending=True)
    keep_mask = torch.ones(boxes.size(0), dtype=torch.bool, device=boxes.device)

    for i in range(boxes.size(0)):
        if not keep_mask[order[i]]:
            continue

        current_idx = order[i]
        current_box = boxes[current_idx : current_idx + 1]
        remaining_indices = order[i + 1 :]
        remaining_mask = keep_mask[remaining_indices]

        if not remaining_mask.any():
            break

        remaining_boxes = boxes[remaining_indices[remaining_mask]]
        current_boxes_expanded = current_box.expand(remaining_boxes.size(0), -1)

        ious = compute_rotated_iou_approx_sdf(
            pred_boxes=current_boxes_expanded,
            target_boxes=remaining_boxes,
            n_samples=n_samples,
            eps=eps,
        )

        suppress_mask = ious > iou_threshold
        remaining_indices_to_suppress = remaining_indices[remaining_mask][suppress_mask]
        keep_mask[remaining_indices_to_suppress] = False

    return order[keep_mask[order]]


@torch.jit.script_if_tracing
def _rotated_nms_vectorized(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
    n_samples: int,
    eps: float,
) -> torch.Tensor:
    """Vectorized standard NMS with pre-computed IoU matrix.

    Computes full IoU matrix once, then applies sequential suppression logic.

    Args:
        boxes: Rotated boxes [N, 5] format [cx, cy, w, h, angle]
        scores: Confidence scores [N]
        iou_threshold: IoU threshold for suppression
        n_samples: Number of samples for IoU computation
        eps: Epsilon for numerical stability

    Returns:
        Indices of boxes to keep, sorted by score descending
    """
    iou_matrix, order, N = _compute_iou_matrix(boxes=boxes, scores=scores, n_samples=n_samples, eps=eps)

    suppress = iou_matrix > iou_threshold
    keep_mask = torch.ones(N, dtype=torch.bool, device=boxes.device)

    for i in range(N - 1):
        if keep_mask[i]:
            keep_mask[suppress[i]] = False

    return order[keep_mask]


@torch.jit.script_if_tracing
def _rotated_nms_fast(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
    n_samples: int,
    eps: float,
) -> torch.Tensor:
    """Fast-NMS with parallel suppression.

    Computes full IoU matrix once, then applies parallel suppression.
    More aggressive than standard NMS.

    Args:
        boxes: Rotated boxes [N, 5] format [cx, cy, w, h, angle]
        scores: Confidence scores [N]
        iou_threshold: IoU threshold for suppression
        n_samples: Number of samples for IoU computation
        eps: Epsilon for numerical stability

    Returns:
        Indices of boxes to keep, sorted by score descending
    """
    iou_matrix, order, N = _compute_iou_matrix(boxes=boxes, scores=scores, n_samples=n_samples, eps=eps)

    suppress_mask = iou_matrix > iou_threshold
    keep_mask = suppress_mask.sum(dim=0) == 0

    return order[keep_mask]


@torch.jit.script_if_tracing
def rotated_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
    nms_mode: str,
    n_samples: int,
    eps: float,
) -> torch.Tensor:
    """Non-maximum suppression for rotated bounding boxes.

    Supports three modes:
    - "sequential": Original implementation, lowest memory (O(N)), slowest
    - "vectorized": Standard NMS with vectorized IoU (O(N²) memory, ~20-30x faster)
    - "fast": Fast-NMS with parallel suppression (O(N²) memory, ~50-100x faster)

    Args:
        boxes: Rotated boxes [N, 5] format [cx, cy, w, h, angle]
        scores: Confidence scores [N]
        iou_threshold: IoU threshold for suppression
        nms_mode: NMS algorithm mode
        n_samples: Number of samples for IoU computation
        eps: Epsilon for numerical stability

    Returns:
        Indices of boxes to keep, sorted by score descending
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    if nms_mode == "fast":
        return _rotated_nms_fast(
            boxes=boxes,
            scores=scores,
            iou_threshold=iou_threshold,
            n_samples=n_samples,
            eps=eps,
        )
    elif nms_mode == "vectorized":
        return _rotated_nms_vectorized(
            boxes=boxes,
            scores=scores,
            iou_threshold=iou_threshold,
            n_samples=n_samples,
            eps=eps,
        )
    else:
        return _rotated_nms_sequential(
            boxes=boxes,
            scores=scores,
            iou_threshold=iou_threshold,
            n_samples=n_samples,
            eps=eps,
        )


@torch.jit.script_if_tracing
def _multiclass_nms_vanilla(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
    nms_mode: str,
    n_samples: int,
    eps: float,
) -> torch.Tensor:
    """Multi-class NMS using per-class processing.

    Args:
        boxes: Rotated boxes [N, 5] format [cx, cy, w, h, angle]
        scores: Confidence scores [N]
        labels: Class labels [N]
        iou_threshold: IoU threshold for suppression
        nms_mode: NMS algorithm mode
        n_samples: Number of samples for IoU computation
        eps: Epsilon for numerical stability

    Returns:
        Indices of boxes to keep, sorted by score descending
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    unique_labels = torch.unique(labels)
    all_keep_indices = []

    for label in unique_labels:
        class_mask = labels == label
        class_indices = torch.where(class_mask)[0]

        if class_indices.size(0) == 0:
            continue

        class_boxes = boxes[class_indices]
        class_scores = scores[class_indices]
        class_keep = rotated_nms(
            boxes=class_boxes,
            scores=class_scores,
            iou_threshold=iou_threshold,
            nms_mode=nms_mode,
            n_samples=n_samples,
            eps=eps,
        )

        if class_keep.size(0) > 0:
            original_indices = class_indices[class_keep]
            all_keep_indices.append(original_indices)

    if len(all_keep_indices) == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    keep_indices = torch.cat(all_keep_indices)

    _, sort_order = scores[keep_indices].sort(descending=True)
    return keep_indices[sort_order]


@torch.jit.script_if_tracing
def _multiclass_nms_coordinate_trick(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
    nms_mode: str,
    n_samples: int,
    eps: float,
) -> torch.Tensor:
    """Multi-class NMS using coordinate offset trick for efficiency.

    Args:
        boxes: Rotated boxes [N, 5] format [cx, cy, w, h, angle]
        scores: Confidence scores [N]
        labels: Class labels [N]
        iou_threshold: IoU threshold for suppression
        nms_mode: NMS algorithm mode
        n_samples: Number of samples for IoU computation
        eps: Epsilon for numerical stability

    Returns:
        Indices of boxes to keep, sorted by score descending
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    max_radius = torch.sqrt(w**2 + h**2).max() / 2.0
    max_coord = torch.max(cx.max(), cy.max())
    offset_scale = (max_coord + max_radius) * 2.0 + 1.0
    offsets = labels.to(boxes.dtype) * offset_scale

    boxes_for_nms = boxes.clone()
    boxes_for_nms[:, 0] += offsets
    boxes_for_nms[:, 1] += offsets

    return rotated_nms(
        boxes=boxes_for_nms,
        scores=scores,
        iou_threshold=iou_threshold,
        nms_mode=nms_mode,
        n_samples=n_samples,
        eps=eps,
    )


@torch.jit.script_if_tracing
def multiclass_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
    nms_mode: str,
    n_samples: int,
    eps: float,
) -> torch.Tensor:
    """Multi-class non-maximum suppression for rotated boxes.

    Performs NMS separately for each class to prevent cross-class suppression.
    Automatically selects the most efficient algorithm based on input size.

    Algorithm selection:
    - Coordinate offset trick for N <= 20,000 (faster for moderate sizes)
    - Per-class processing for N > 20,000 (avoids numerical issues)

    Args:
        boxes: Rotated boxes [N, 5] format [cx, cy, w, h, angle]
        scores: Confidence scores [N]
        labels: Class labels [N]
        iou_threshold: IoU threshold for suppression
        nms_mode: NMS algorithm mode
        n_samples: Number of samples for IoU computation
        eps: Epsilon for numerical stability

    Returns:
        Indices of boxes to keep, sorted by score descending
    """
    if boxes.numel() > 20000:
        return _multiclass_nms_vanilla(
            boxes=boxes,
            scores=scores,
            labels=labels,
            iou_threshold=iou_threshold,
            nms_mode=nms_mode,
            n_samples=n_samples,
            eps=eps,
        )
    return _multiclass_nms_coordinate_trick(
        boxes=boxes,
        scores=scores,
        labels=labels,
        iou_threshold=iou_threshold,
        nms_mode=nms_mode,
        n_samples=n_samples,
        eps=eps,
    )
