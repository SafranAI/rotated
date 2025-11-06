"""Precise Rotated IoU Implementation.

Using the Sutherland-Hodgman clipping algorithm which is the core of exact polygon intersection.
"""

import torch

from rotated.iou.base import BaseRotatedIoU


class PreciseRotatedIoU(BaseRotatedIoU):
    """Exact rotated IoU computation."""

    def __init__(self, eps: float = 1e-12):
        super().__init__(eps=eps)

    def _compute_candidate_ious(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU for candidate box pairs using exact polygon intersection.

        Args:
            pred_boxes: [M, 5] candidate predicted boxes
            target_boxes: [M, 5] candidate target boxes

        Returns:
            [M] IoU values for candidates
        """
        # Convert boxes to polygons
        pred_polygons = self._boxes_to_polygons(pred_boxes)
        target_polygons = self._boxes_to_polygons(target_boxes)

        # Compute individual polygon areas
        pred_areas = self._compute_polygon_areas(pred_polygons)
        target_areas = self._compute_polygon_areas(target_polygons)

        # Compute intersection areas (the expensive part)
        intersection_areas = self._compute_intersection_areas(pred_polygons, target_polygons)

        # Calculate IoU = intersection / union
        union_areas = pred_areas + target_areas - intersection_areas
        candidate_ious = self._safe_divide(intersection_areas, union_areas)

        return candidate_ious

    def _boxes_to_polygons(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert rotated boxes to 4-vertex polygons.

        Returns: [N, 4, 2] tensor of polygon vertices (counter-clockwise)
        """
        x, y, w, h, angle = boxes.unbind(-1)

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
        scaled_corners = unit_corners.unsqueeze(0) * torch.stack([w, h], dim=-1).unsqueeze(-2)

        # Rotate corners
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        x_local, y_local = scaled_corners[..., 0], scaled_corners[..., 1]

        x_rotated = x_local * cos_a.unsqueeze(-1) - y_local * sin_a.unsqueeze(-1)
        y_rotated = x_local * sin_a.unsqueeze(-1) + y_local * cos_a.unsqueeze(-1)

        # Translate to final position
        final_corners = torch.stack([x_rotated + x.unsqueeze(-1), y_rotated + y.unsqueeze(-1)], dim=-1)

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
        N = polygons1.shape[0]
        intersection_areas = torch.zeros(N, device=polygons1.device, dtype=polygons1.dtype)

        # Process each polygon pair individually
        for i in range(N):
            clipped_polygon = self._clip_polygon_by_polygon(polygons1[i], polygons2[i])

            if clipped_polygon.shape[0] >= 3:  # Valid polygon needs at least 3 vertices
                area = self._compute_polygon_areas(clipped_polygon.unsqueeze(0)).squeeze(0)
                intersection_areas[i] = area

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
        for i in range(polygon.shape[0]):
            current_vertex = polygon[i]
            previous_vertex = polygon[i - 1]  # -1 wraps to last vertex

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

        return bool((cross_product >= -self.eps).item())

    def _compute_line_intersection(self, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor):
        """Compute intersection point of two line segments.

        Line 1: p1 -> p2
        Line 2: p3 -> p4

        Returns intersection point or None if lines are parallel.
        """
        line1_direction = p2 - p1
        line2_direction = p4 - p3

        # Check if lines are parallel using cross product
        cross_product = line1_direction[0] * line2_direction[1] - line1_direction[1] * line2_direction[0]

        if torch.abs(cross_product) < self.eps:
            return None  # Lines are parallel

        # Solve for intersection using parametric line equations
        # p1 + t * line1_direction = p3 + s * line2_direction
        start_difference = p3 - p1
        t = (start_difference[0] * line2_direction[1] - start_difference[1] * line2_direction[0]) / cross_product

        intersection_point = p1 + t * line1_direction
        return intersection_point

    def _safe_divide(self, numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        """Safely divide, returning 0 when denominator is too small."""
        valid_denominator = denominator > self.eps
        result = torch.where(valid_denominator, numerator / denominator, torch.zeros_like(numerator))
        return torch.clamp(result, 0.0, 1.0)
