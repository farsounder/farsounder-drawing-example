from dataclasses import dataclass
from typing import Callable

Point3D = tuple[float, float, float]
ZoneId = tuple[int, str]
ColorRGB = tuple[int, int, int]
TriangleIndices = tuple[int, int, int]


@dataclass(frozen=True)
class Vertex:
    position: Point3D


@dataclass(frozen=True)
class LiveVertex(Vertex):
    down_range_m: float


@dataclass
class GriddedCell:
    depth_sum: float = 0.0
    sample_count: int = 0

    def add_sample(self, depth: float) -> None:
        self.depth_sum += depth
        self.sample_count += 1

    @property
    def average_depth(self) -> float:
        return self.depth_sum / self.sample_count


@dataclass(frozen=True)
class PointsRender:
    entity_path: str
    points: list[Point3D]
    colors: list[ColorRGB]
    radii: float


@dataclass(frozen=True)
class MeshRender:
    entity_path: str
    vertex_positions: list[Point3D]
    triangle_indices: list[TriangleIndices]
    vertex_normals: list[Point3D]
    vertex_colors: list[ColorRGB]


PointLogger = Callable[[PointsRender], None]
MeshLogger = Callable[[MeshRender], None]
ClearLogger = Callable[[str], None]


@dataclass(frozen=True)
class ViewerBackend:
    init: Callable[[], None]
    log_points: PointLogger
    log_mesh: MeshLogger
    clear: ClearLogger
