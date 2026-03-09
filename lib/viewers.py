import logging
import math
import time
from dataclasses import dataclass, field
from typing import Callable

from farsounder.proto import nav_api_pb2

from lib.config import (
    GRIDDED_LOG_EVERY_MESSAGES,
    GRIDDED_SURFACE_LOG_EVERY_MESSAGES,
    TIN_MIN_EDGE,
    TIN_TUNING_FACTOR,
    UNGRIDDED_LOG_EVERY_MESSAGES,
)
from lib.depth_colors import depth_colors
from lib.geometry import delaunay_triangle_indices, gridded_cell_vertices, mesh_vertex_normals
from lib.models import (
    ClearLogger,
    GriddedCell,
    LiveVertex,
    MeshLogger,
    MeshRender,
    Point3D,
    PointLogger,
    PointsRender,
    Vertex,
)
from lib.navigation import get_horizontal_angle_spacing_rad
from lib.time import time_it


@dataclass
class UnGriddedBottomViewer:
    point_logger: PointLogger
    entity_path: str = "world/bottom/ungridded"
    point_radius: float = 0.2
    log_every_messages: int = UNGRIDDED_LOG_EVERY_MESSAGES

    def __post_init__(self) -> None:
        if self.log_every_messages <= 0:
            raise ValueError("UnGriddedBottomViewer log_every_messages must be positive")

    def should_log(self, current_message: int | None) -> bool:
        return current_message is None or current_message % self.log_every_messages == 0

    def reset(self) -> None:
        pass

    @time_it(name="UnGriddedBottomViewer.log_points")
    def log_points(self, points: list[Point3D], current_message: int | None = None) -> None:
        if not points:
            return

        if not self.should_log(current_message):
            return

        self.point_logger(
            PointsRender(
                entity_path=f"{self.entity_path}/ping{current_message}",
                points=points,
                colors=depth_colors(points),
                radii=self.point_radius,
            )
        )
        logging.info(f"Logged {len(points)} raw bottom points")


@dataclass
class GriddedBottomViewer:
    interval_m: float
    point_logger: PointLogger
    entity_path: str = "world/bottom/gridded"
    point_radius: float = 0.3
    log_every_messages: int = GRIDDED_LOG_EVERY_MESSAGES
    cells: dict[tuple[int, int], GriddedCell] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.interval_m <= 0.0:
            raise ValueError("GriddedBottomViewer interval must be positive")
        if self.log_every_messages <= 0:
            raise ValueError("GriddedBottomViewer log_every_messages must be positive")

    def reset(self) -> None:
        self.cells.clear()

    def should_log(self, current_message: int | None) -> bool:
        return current_message is None or current_message % self.log_every_messages == 0

    def add_points(self, points: list[Point3D]) -> None:
        for x_m, y_m, depth_m in points:
            cell_key = (
                math.floor(x_m / self.interval_m),
                math.floor(y_m / self.interval_m),
            )
            cell = self.cells.setdefault(cell_key, GriddedCell())
            cell.add_sample(depth_m)

    def averaged_points(self) -> list[Point3D]:
        averaged_points: list[Point3D] = []
        for (cell_x, cell_y), cell in sorted(self.cells.items()):
            averaged_points.append(
                (
                    (cell_x + 0.5) * self.interval_m,
                    (cell_y + 0.5) * self.interval_m,
                    cell.average_depth,
                )
            )
        return averaged_points

    @time_it(name="GriddedBottomViewer.log_points")
    def log_points(self, points: list[Point3D], current_message: int | None = None) -> None:
        self.add_points(points)
        if not self.should_log(current_message):
            return

        if not self.cells:
            return

        averaged_points = self.averaged_points()

        self.point_logger(
            PointsRender(
                entity_path=self.entity_path,
                points=averaged_points,
                colors=depth_colors(averaged_points),
                radii=self.point_radius,
            )
        )
        logging.info(f"Logged {len(averaged_points)} gridded bottom cells")


@dataclass
class GriddedBottomSurfaceViewer:
    mesh_logger: MeshLogger
    clear_logger: ClearLogger
    entity_path: str = "world/bottom/gridded_surface"
    log_every_messages: int = GRIDDED_SURFACE_LOG_EVERY_MESSAGES
    min_edge: float = TIN_MIN_EDGE

    def __post_init__(self) -> None:
        if self.log_every_messages <= 0:
            raise ValueError("GriddedBottomSurfaceViewer log_every_messages must be positive")

    def should_log(self, current_message: int | None) -> bool:
        return current_message is None or current_message % self.log_every_messages == 0

    def reset(self) -> None:
        self.clear_logger(self.entity_path)

    @time_it(name="GriddedBottomSurfaceViewer.log_points")
    def log_points(
        self,
        cells: dict[tuple[int, int], GriddedCell],
        interval_m: float,
        current_message: int | None = None,
    ) -> None:
        if not self.should_log(current_message):
            return

        if not cells:
            self.reset()
            return

        vertices = gridded_cell_vertices(cells, interval_m)

        def skip_triangle(a: Vertex, b: Vertex, c: Vertex) -> bool:
            edge_lengths = (
                math.dist(a.position, b.position),
                math.dist(b.position, c.position),
                math.dist(c.position, a.position),
            )
            return any(edge_length > self.min_edge for edge_length in edge_lengths)

        triangle_indices = delaunay_triangle_indices(
            vertices,
            cull_triangle_func=skip_triangle,
        )
        if not triangle_indices:
            self.reset()
            return

        vertex_positions = [vertex.position for vertex in vertices]
        vertex_normals = mesh_vertex_normals(vertex_positions, triangle_indices)

        self.mesh_logger(
            MeshRender(
                entity_path=self.entity_path,
                vertex_positions=vertex_positions,
                triangle_indices=triangle_indices,
                vertex_normals=vertex_normals,
                vertex_colors=depth_colors(vertex_positions),
            )
        )
        logging.info(f"Logged gridded bottom surface with {len(triangle_indices)} triangles")


@dataclass
class LiveBottomSurfaceViewer:
    mesh_logger: MeshLogger
    clear_logger: ClearLogger
    entity_path: str = "world/bottom/live_surface"
    log_every_messages: int = UNGRIDDED_LOG_EVERY_MESSAGES
    tuning_factor: float = TIN_TUNING_FACTOR
    min_edge: float = TIN_MIN_EDGE

    def __post_init__(self) -> None:
        if self.log_every_messages <= 0:
            raise ValueError("LiveBottomSurfaceViewer log_every_messages must be positive")
        if self.tuning_factor <= 0.0:
            raise ValueError("LiveBottomSurfaceViewer tuning_factor must be positive")

    def should_log(self, current_message: int | None) -> bool:
        return current_message is None or current_message % self.log_every_messages == 0

    def reset(self) -> None:
        self.clear_logger(self.entity_path)

    @time_it(name="LiveBottomSurfaceViewer.log_points")
    def log_points(
        self,
        vertices: list[LiveVertex],
        message: nav_api_pb2.TargetData,
        current_message: int | None = None,
    ) -> None:
        if not self.should_log(current_message):
            return

        if len(vertices) < 3:
            self.reset()
            return

        if not (hor_angle_spacing_rad := get_horizontal_angle_spacing_rad(message)):
            self.reset()
            return

        def skip_triangle(a: LiveVertex, b: LiveVertex, c: LiveVertex) -> bool:
            max_down_range = max(a.down_range_m, b.down_range_m, c.down_range_m)
            max_edge_m = max_down_range * math.tan(hor_angle_spacing_rad) * self.tuning_factor
            max_edge_m = max(max_edge_m, self.min_edge)
            edge_lengths = (
                math.dist(a.position, b.position),
                math.dist(b.position, c.position),
                math.dist(c.position, a.position),
            )
            return any(edge_length > max_edge_m for edge_length in edge_lengths)

        triangle_indices = delaunay_triangle_indices(
            vertices,
            cull_triangle_func=skip_triangle,
        )
        if not triangle_indices:
            self.reset()
            return

        vertex_positions = [vertex.position for vertex in vertices]
        vertex_normals = mesh_vertex_normals(vertex_positions, triangle_indices)

        self.mesh_logger(
            MeshRender(
                entity_path=self.entity_path,
                vertex_positions=vertex_positions,
                triangle_indices=triangle_indices,
                vertex_normals=vertex_normals,
                vertex_colors=depth_colors(vertex_positions),
            )
        )
        logging.info(f"Logged live bottom surface with {len(triangle_indices)} triangles")
