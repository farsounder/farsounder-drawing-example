import math
from typing import Callable

from scipy.spatial import Delaunay, QhullError

from lib.models import GriddedCell, Point3D, TriangleIndices, Vertex


def subtract_points(a: Point3D, b: Point3D) -> Point3D:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def cross_product(a: Point3D, b: Point3D) -> Point3D:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def normalize_vector(vector: Point3D) -> Point3D:
    magnitude = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
    if math.isclose(magnitude, 0.0):
        return (0.0, 0.0, -1.0)
    return (vector[0] / magnitude, vector[1] / magnitude, vector[2] / magnitude)


def mesh_vertex_normals(
    vertex_positions: list[Point3D],
    triangle_indices: list[TriangleIndices],
) -> list[Point3D]:
    normals = [(0.0, 0.0, 0.0) for _ in vertex_positions]
    for index_a, index_b, index_c in triangle_indices:
        point_a = vertex_positions[index_a]
        point_b = vertex_positions[index_b]
        point_c = vertex_positions[index_c]
        edge_ab = subtract_points(point_b, point_a)
        edge_ac = subtract_points(point_c, point_a)
        face_normal = cross_product(edge_ab, edge_ac)

        for vertex_index in (index_a, index_b, index_c):
            normal_x, normal_y, normal_z = normals[vertex_index]
            normals[vertex_index] = (
                normal_x + face_normal[0],
                normal_y + face_normal[1],
                normal_z + face_normal[2],
            )

    return [
        normalize_vector((-normal_x, -normal_y, -normal_z))
        for normal_x, normal_y, normal_z in normals
    ]


def delaunay_triangle_indices(
    vertices: list[Vertex],
    cull_triangle_func: Callable[[Vertex, Vertex, Vertex], bool],
) -> list[TriangleIndices]:
    if len(vertices) < 3:
        return []

    projected_points = [(vertex.position[0], vertex.position[1]) for vertex in vertices]
    try:
        triangulation = Delaunay(projected_points)
    except QhullError:
        return []

    triangle_indices: list[TriangleIndices] = []
    for simplex in triangulation.simplices:
        index_a, index_b, index_c = (int(simplex[0]), int(simplex[1]), int(simplex[2]))

        if cull_triangle_func(vertices[index_a], vertices[index_b], vertices[index_c]):
            continue
        triangle_indices.append((index_a, index_b, index_c))

    return triangle_indices


def gridded_cell_vertices(
    cells: dict[tuple[int, int], GriddedCell],
    interval_m: float,
) -> list[Vertex]:
    vertices: list[Vertex] = []
    for (cell_x, cell_y), cell in sorted(cells.items()):
        position = (
            (cell_x + 0.5) * interval_m,
            (cell_y + 0.5) * interval_m,
            cell.average_depth,
        )
        vertices.append(Vertex(position=position))
    return vertices
