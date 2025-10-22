"""Non-maximum suppression for rotated bounding boxes.

Pure PyTorch implementation with TorchScript compatibility.
For CUDA acceleration, consider integrating with detectron2's rotated_iou operators.
"""

import torch

from rotated.iou.precise_iou import PreciseRotatedIoU


@torch.jit.script_if_tracing
def rotated_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """Non-maximum suppression for rotated bounding boxes.

    Args:
        boxes: Rotated boxes [N, 5] format [cx, cy, w, h, angle] in absolute pixels, angle in radians
        scores: Confidence scores [N]
        iou_threshold: IoU threshold for suppression

    Returns:
        Indices of boxes to keep, sorted by score descending
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    _, order = torch.sort(scores, descending=True)
    keep_mask = torch.ones(boxes.size(0), dtype=torch.bool, device=boxes.device)

    iou_calculator = PreciseRotatedIoU()

    for i in range(boxes.size(0)):
        if not keep_mask[order[i]]:
            continue

        # Get current box
        current_idx = order[i]
        current_box = boxes[current_idx : current_idx + 1]

        # Get all remaining boxes that haven't been suppressed yet
        remaining_indices = order[i + 1 :]
        remaining_mask = keep_mask[remaining_indices]

        if not remaining_mask.any():
            break

        # Filter remaining_indices to only include boxes that haven't been suppressed yet
        # Note: keep_mask gets updated at every iteration with more False values,
        # so we need to filter remaining_indices using the current state of keep_mask
        remaining_boxes = boxes[remaining_indices[remaining_mask]]
        current_boxes_expanded = current_box.expand(remaining_boxes.size(0), -1)

        ious = iou_calculator(current_boxes_expanded, remaining_boxes)

        # Suppress boxes with IoU > threshold
        suppress_mask = ious > iou_threshold
        remaining_indices_to_suppress = remaining_indices[remaining_mask][suppress_mask]
        keep_mask[remaining_indices_to_suppress] = False

    return order[keep_mask[order]]


@torch.jit.script_if_tracing
def _multiclass_nms_coordinate_trick(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Multi-class NMS using coordinate offset trick for efficiency."""
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

    return rotated_nms(boxes_for_nms, scores, iou_threshold)


@torch.jit.script_if_tracing
def _multiclass_nms_vanilla(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Multi-class NMS using per-class processing."""
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
        class_keep = rotated_nms(class_boxes, class_scores, iou_threshold)

        if class_keep.size(0) > 0:
            original_indices = class_indices[class_keep]
            all_keep_indices.append(original_indices)

    if not all_keep_indices:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    keep_indices = torch.cat(all_keep_indices)

    # Sort by score descending
    _, sort_order = scores[keep_indices].sort(descending=True)
    return keep_indices[sort_order]


@torch.jit.script_if_tracing
def multiclass_rotated_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Multi-class non-maximum suppression for rotated boxes.

    Performs NMS separately for each class to prevent cross-class suppression.
    Automatically selects the most efficient algorithm based on input size.

    Algorithm selection:
    - Coordinate offset trick for N <= 20,000 (faster for moderate sizes)
    - Per-class processing for N > 20,000 (avoids numerical issues with large offsets)

    Args:
        boxes: Rotated boxes [N, 5] format [cx, cy, w, h, angle]
        scores: Confidence scores [N]
        labels: Class labels [N]
        iou_threshold: IoU threshold for suppression

    Returns:
        Indices of boxes to keep, sorted by score descending
    """
    if boxes.numel() > 20000:
        return _multiclass_nms_vanilla(boxes, scores, labels, iou_threshold)
    return _multiclass_nms_coordinate_trick(boxes, scores, labels, iou_threshold)


@torch.jit.script_if_tracing
def batched_multiclass_rotated_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
    max_output_per_batch: int = 300,
) -> torch.Tensor:
    """Batched multi-class NMS with consistent output shapes.

    Processes multiple samples in parallel and returns padded outputs for consistent tensor shapes across the batch.
    Note: This function filters boxes with scores <= 0.0.

    Args:
        boxes: Batched boxes [B, N, 5]
        scores: Batched scores [B, N]
        labels: Batched labels [B, N]
        iou_threshold: IoU threshold for suppression
        max_output_per_batch: Maximum detections per batch element

    Returns:
        Indices tensor [B, max_output_per_batch] with -1 padding for invalid detections
    """
    batch_size = boxes.size(0)
    device = boxes.device
    output = torch.full((batch_size, max_output_per_batch), -1, dtype=torch.long, device=device)

    for batch_idx in range(batch_size):
        batch_boxes = boxes[batch_idx]
        batch_scores = scores[batch_idx]
        batch_labels = labels[batch_idx]

        # Filter meaningful scores
        valid_mask = batch_scores > 0.0

        if valid_mask.any():
            valid_boxes = batch_boxes[valid_mask]
            valid_scores = batch_scores[valid_mask]
            valid_labels = batch_labels[valid_mask]

            keep_indices = multiclass_rotated_nms(valid_boxes, valid_scores, valid_labels, iou_threshold)

            if keep_indices.size(0) > 0:
                valid_original_indices = torch.where(valid_mask)[0]
                final_indices = valid_original_indices[keep_indices]
                num_to_keep = min(final_indices.size(0), max_output_per_batch)
                output[batch_idx, :num_to_keep] = final_indices[:num_to_keep]

    return output


@torch.jit.script_if_tracing
def postprocess_detections(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    score_thresh: float = 0.05,
    nms_thresh: float = 0.5,
    detections_per_img: int = 300,
    topk_candidates: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply detection post-processing pipeline.

    Pipeline: score filtering → topk selection → NMS → detections_per_img limit
    Handles both single and batched inputs automatically.

    Args:
        boxes: Boxes [N, 5] or [B, N, 5]
        scores: Scores [N] or [B, N]
        labels: Labels [N] or [B, N]
        score_thresh: Score threshold used for postprocessing the detections
        nms_thresh: NMS threshold used for postprocessing the detections
        detections_per_img: Number of best detections to keep after NMS
        topk_candidates: Number of best detections to keep before NMS

    Returns:
        For single input: (boxes, labels, scores) [detections_per_img, 5], [detections_per_img], [detections_per_img]
        For batched input: (boxes, labels, scores) [B, detections_per_img, 5], [B, detections_per_img], [B, detections_per_img]

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
    output_boxes, output_labels, output_scores = _postprocess(
        boxes,
        scores,
        labels,
        score_thresh=score_thresh,
        nms_thresh=nms_thresh,
        detections_per_img=detections_per_img,
        topk_candidates=topk_candidates,
    )

    # Squeeze back if single sample
    if is_single:
        output_boxes = output_boxes.squeeze(0)
        output_labels = output_labels.squeeze(0)
        output_scores = output_scores.squeeze(0)

    return output_boxes, output_labels, output_scores


@torch.jit.script_if_tracing
def _postprocess(
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
        keep_indices = multiclass_rotated_nms(filtered_boxes, filtered_scores, filtered_labels, nms_thresh)
        if keep_indices.numel() == 0:
            continue

        # Step 4: Limit to detections_per_img
        keep_indices = keep_indices[:detections_per_img]

        # Extract and store results
        num_keep = keep_indices.size(0)
        output_boxes[batch_idx, :num_keep] = filtered_boxes[keep_indices]
        output_labels[batch_idx, :num_keep] = filtered_labels[keep_indices]
        output_scores[batch_idx, :num_keep] = filtered_scores[keep_indices]

    return output_boxes, output_labels, output_scores
