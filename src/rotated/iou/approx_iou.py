"""Precise Rotated IoU Implementation.

Computes exact IoU between rotated bounding boxes using polygon intersection
with the Sutherland-Hodgman clipping algorithm.
"""

import torch


class PreciseRotatedIoU:
    """Exact rotated IoU computation using polygon intersection."""

    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    def __call__(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute exact IoU between rotated boxes.

        Args:
            pred_boxes: [N, 5] (x, y, w, h, angle)
            target_boxes: [N, 5] (x, y, w, h, angle)

        Returns:
            [N] IoU values
        """
        num_boxes = pred_boxes.shape[0]
        if num_boxes == 0:
            return torch.empty(0, device=pred_boxes.device, dtype=pred_boxes.dtype)

        # Step 1: Filter out obviously non-overlapping boxes
        overlap_candidates = self._find_overlap_candidates(pred_boxes, target_boxes)
        ious = torch.zeros(num_boxes, device=pred_boxes.device, dtype=pred_boxes.dtype)

        if not overlap_candidates.any():
            return ious

        # Step 2: Process only the candidates
        candidate_indices = torch.where(overlap_candidates)[0]
        pred_candidates = pred_boxes[candidate_indices]
        target_candidates = target_boxes[candidate_indices]

        # Step 3: Convert boxes to polygons
        pred_polygons = self._boxes_to_polygons(pred_candidates)
        target_polygons = self._boxes_to_polygons(target_candidates)

        # Step 4: Compute individual polygon areas
        pred_areas = self._compute_polygon_areas(pred_polygons)
        target_areas = self._compute_polygon_areas(target_polygons)

        # Step 5: Compute intersection areas
        intersection_areas = self._compute_intersection_areas(pred_polygons, target_polygons)

        # Step 6: Calculate IoU = intersection / union
        union_areas = pred_areas + target_areas - intersection_areas
        candidate_ious = self._safe_divide(intersection_areas, union_areas)

        ious[candidate_indices] = candidate_ious
        return ious

    def _find_overlap_candidates(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Find box pairs that might overlap using axis-aligned bounding boxes."""
        # Extract box parameters
        center_x1, center_y1, width1, height1, angle1 = boxes1.unbind(-1)
        center_x2, center_y2, width2, height2, angle2 = boxes2.unbind(-1)

        # Compute the axis-aligned bounding box extents for boxes1
        cos_angle1, sin_angle1 = torch.cos(angle1), torch.sin(angle1)
        extent1_width = 0.5 * width1 * torch.abs(cos_angle1) + 0.5 * height1 * torch.abs(sin_angle1)
        extent1_height = 0.5 * width1 * torch.abs(sin_angle1) + 0.5 * height1 * torch.abs(cos_angle1)

        # Compute the axis-aligned bounding box extents for boxes2
        cos_angle2, sin_angle2 = torch.cos(angle2), torch.sin(angle2)
        extent2_width = 0.5 * width2 * torch.abs(cos_angle2) + 0.5 * height2 * torch.abs(sin_angle2)
        extent2_height = 0.5 * width2 * torch.abs(sin_angle2) + 0.5 * height2 * torch.abs(cos_angle2)

        # Check if AABBs overlap
        x_overlap = torch.abs(center_x1 - center_x2) < (extent1_width + extent2_width)
        y_overlap = torch.abs(center_y1 - center_y2) < (extent1_height + extent2_height)

        return x_overlap & y_overlap

    def _boxes_to_polygons(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert rotated boxes to 4-vertex polygons.

        Returns: [N, 4, 2] tensor of polygon vertices (counter-clockwise)
        """
        center_x, center_y, width, height, angle = boxes.unbind(-1)

        # Define the 4 corners of a unit box (counter-clockwise from bottom-left)
        unit_corners = torch.tensor(
            [
                [-0.5, -0.5],  # bottom-left
                [0.5, -0.5],  # bottom-right
                [0.5, 0.5],  # top-right
                [-0.5, 0.5],  # top-left
            ],
            device=boxes.device,
            dtype=boxes.dtype,
        )

        # Scale corners by box dimensions
        scaled_corners = unit_corners.unsqueeze(0) * torch.stack([width, height], dim=-1).unsqueeze(-2)

        # Rotate corners
        cos_angle, sin_angle = torch.cos(angle), torch.sin(angle)
        x_local, y_local = scaled_corners[..., 0], scaled_corners[..., 1]

        x_rotated = x_local * cos_angle.unsqueeze(-1) - y_local * sin_angle.unsqueeze(-1)
        y_rotated = x_local * sin_angle.unsqueeze(-1) + y_local * cos_angle.unsqueeze(-1)

        # Translate to final position
        final_corners = torch.stack([x_rotated + center_x.unsqueeze(-1), y_rotated + center_y.unsqueeze(-1)], dim=-1)

        return final_corners

    def _compute_polygon_areas(self, polygons: torch.Tensor) -> torch.Tensor:
        """Compute areas using the shoelace formula.

        Args: polygons [N, 4, 2]
        Returns: areas [N]
        """
        x_coords = polygons[..., 0]  # [N, 4]
        y_coords = polygons[..., 1]  # [N, 4]

        # Get next vertex coordinates (with wraparound)
        x_next = torch.roll(x_coords, -1, dims=-1)
        y_next = torch.roll(y_coords, -1, dims=-1)

        # Shoelace formula: 0.5 * |sum(x_i * y_{i+1} - x_{i+1} * y_i)|
        cross_products = x_coords * y_next - x_next * y_coords
        areas = 0.5 * torch.abs(cross_products.sum(dim=-1))

        return areas

    def _compute_intersection_areas(self, polygons1: torch.Tensor, polygons2: torch.Tensor) -> torch.Tensor:
        """Compute intersection areas for pairs of polygons."""
        num_boxes = polygons1.shape[0]
        intersection_areas = torch.zeros(num_boxes, device=polygons1.device, dtype=polygons1.dtype)

        # Process each polygon pair individually
        for idx in range(num_boxes):
            clipped_polygon = self._clip_polygon_by_polygon(polygons1[idx], polygons2[idx])

            if clipped_polygon.shape[0] >= 3:  # Valid polygon needs at least 3 vertices
                area = self._compute_polygon_areas(clipped_polygon.unsqueeze(0)).squeeze(0)
                intersection_areas[idx] = area

        return intersection_areas

    def _clip_polygon_by_polygon(self, subject_polygon: torch.Tensor, clip_polygon: torch.Tensor) -> torch.Tensor:
        """Clip subject polygon by clip polygon using Sutherland-Hodgman algorithm.

        The algorithm works by clipping the subject polygon against each edge of the
        clipping polygon one by one.

        Args:
            subject_polygon: [4, 2] vertices of polygon to be clipped
            clip_polygon: [4, 2] vertices of clipping polygon

        Returns:
            [K, 2] vertices of clipped polygon (K can be 0 if no intersection)
        """
        current_polygon = subject_polygon

        # Clip against each of the 4 edges of the clipping polygon
        for edge_index in range(4):
            if current_polygon.shape[0] == 0:
                break  # No vertices left, completely clipped

            # Define the current clipping edge
            edge_start = clip_polygon[edge_index]
            edge_end = clip_polygon[(edge_index + 1) % 4]

            # Clip current polygon by this edge
            current_polygon = self._clip_polygon_by_edge(current_polygon, edge_start, edge_end)

        return current_polygon

    def _clip_polygon_by_edge(
        self, polygon: torch.Tensor, edge_start: torch.Tensor, edge_end: torch.Tensor
    ) -> torch.Tensor:
        """Clip polygon by a single edge using Sutherland-Hodgman algorithm.

        For each edge of the input polygon:
        - If both vertices are inside: keep the second vertex
        - If first inside, second outside: keep intersection point
        - If first outside, second inside: keep intersection + second vertex
        - If both outside: keep nothing
        """
        if polygon.shape[0] == 0:
            return polygon

        output_vertices = []

        # Process each edge of the input polygon
        for vertex_idx in range(polygon.shape[0]):
            current_vertex = polygon[vertex_idx]
            previous_vertex = polygon[vertex_idx - 1]  # -1 wraps to last vertex

            current_inside = self._is_vertex_inside_edge(current_vertex, edge_start, edge_end)
            previous_inside = self._is_vertex_inside_edge(previous_vertex, edge_start, edge_end)

            if current_inside:
                if not previous_inside:
                    # Entering the inside region: add intersection point
                    intersection = self._compute_line_intersection(
                        previous_vertex, current_vertex, edge_start, edge_end
                    )
                    if intersection is not None:
                        output_vertices.append(intersection)

                # Current vertex is inside: always add it
                output_vertices.append(current_vertex)

            elif previous_inside:
                # Leaving the inside region: add intersection point only
                intersection = self._compute_line_intersection(previous_vertex, current_vertex, edge_start, edge_end)
                if intersection is not None:
                    output_vertices.append(intersection)

        # Convert list back to tensor
        if output_vertices:
            return torch.stack(output_vertices)
        else:
            return torch.zeros(0, 2, device=polygon.device, dtype=polygon.dtype)

    def _is_vertex_inside_edge(self, vertex: torch.Tensor, edge_start: torch.Tensor, edge_end: torch.Tensor) -> bool:
        """Check if vertex is on the inside (left) side of a directed edge.

        Uses the cross product to determine which side of the line the point is on.
        Positive cross product means the point is to the left (inside).
        """
        edge_vector = edge_end - edge_start
        vertex_vector = vertex - edge_start

        # 2D cross product: edge_vector × vertex_vector
        cross_product = edge_vector[0] * vertex_vector[1] - edge_vector[1] * vertex_vector[0]

        return cross_product >= -self.eps  # Allow small numerical errors

    def _compute_line_intersection(
        self, point1: torch.Tensor, point2: torch.Tensor, point3: torch.Tensor, point4: torch.Tensor
    ):
        """Compute intersection point of two line segments.

        Line 1: point1 -> point2
        Line 2: point3 -> point4

        Returns intersection point or None if lines are parallel.
        """
        line1_direction = point2 - point1
        line2_direction = point4 - point3

        # Check if lines are parallel using cross product
        cross_product = line1_direction[0] * line2_direction[1] - line1_direction[1] * line2_direction[0]

        if torch.abs(cross_product) < self.eps:
            return None  # Lines are parallel

        # Solve for intersection using parametric line equations
        # point1 + t * line1_direction = point3 + s * line2_direction
        start_difference = point3 - point1
        parameter_t = (
            start_difference[0] * line2_direction[1] - start_difference[1] * line2_direction[0]
        ) / cross_product

        intersection_point = point1 + parameter_t * line1_direction
        return intersection_point

    def _safe_divide(self, numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        """Safely divide, returning 0 when denominator is too small."""
        valid_denominator = denominator > self.eps
        result = torch.where(valid_denominator, numerator / denominator, torch.zeros_like(numerator))
        return torch.clamp(result, 0.0, 1.0)
