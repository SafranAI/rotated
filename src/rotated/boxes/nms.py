"""Module to run the Non Maximum Suppression algorithm."""

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from rotated.iou import iou_picker

if TYPE_CHECKING:
    from rotated.iou import IoUKwargs, IoUMethodName


class NMS(nn.Module):
    """Non-Maximum Suppression module for rotated object detection.

    Args:
        nms_thresh: threshold below which detections are removed
        iou_method: Method name to compute Intersection Over Union
        iou_kwargs: Dictionary with parameters for the IoU method.
    """

    def __init__(
        self,
        nms_thresh: float = 0.5,
        iou_method: "IoUMethodName" = "approx_sdf_l1",
        iou_kwargs: "IoUKwargs" = None,
    ):
        super().__init__()
        self.nms_thresh = nms_thresh
        self.iou_calculator = iou_picker(iou_method=iou_method, iou_kwargs=iou_kwargs)

    @torch.jit.script_if_tracing
    def forward(
        self,
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
            return self._multiclass_nms_vanilla(boxes, scores, labels, iou_threshold)
        return self._multiclass_nms_coordinate_trick(boxes, scores, labels, iou_threshold)

    @torch.jit.script_if_tracing
    def _multiclass_nms_vanilla(
        self,
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
            class_keep = self.rotated_nms(class_boxes, class_scores, iou_threshold)

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
    def _multiclass_nms_coordinate_trick(
        self,
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

        return self.rotated_nms(boxes_for_nms, scores, iou_threshold)

    @torch.jit.script_if_tracing
    def rotated_nms(self, boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
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

            ious = self.iou_calculator(current_boxes_expanded, remaining_boxes)

            # Suppress boxes with IoU > threshold
            suppress_mask = ious > iou_threshold
            remaining_indices_to_suppress = remaining_indices[remaining_mask][suppress_mask]
            keep_mask[remaining_indices_to_suppress] = False

        return order[keep_mask[order]]

    @torch.jit.script_if_tracing
    def batched_multiclass_rotated_nms(
        self,
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

                keep_indices = self.forward(valid_boxes, valid_scores, valid_labels, iou_threshold)

                if keep_indices.size(0) > 0:
                    valid_original_indices = torch.where(valid_mask)[0]
                    final_indices = valid_original_indices[keep_indices]
                    num_to_keep = min(final_indices.size(0), max_output_per_batch)
                    output[batch_idx, :num_to_keep] = final_indices[:num_to_keep]

        return output
