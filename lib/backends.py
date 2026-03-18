from typing import Literal, TypeAlias
from typing import Callable
import rerun as rr

from lib.models import MeshRender, PointsRender, ViewerBackend


BackendName: TypeAlias = Literal["rerun"]


def _build_rerun_viewer_backend() -> ViewerBackend:
    def init() -> None:
        rr.init("Argos SDK Example", spawn=True)
        rr.spawn()

    def log_points(render: PointsRender) -> None:
        rr.log(
            render.entity_path,
            rr.Points3D(
                render.points,
                colors=render.colors,
                radii=render.radii,
            ),
        )

    def log_mesh(render: MeshRender) -> None:
        rr.log(
            render.entity_path,
            rr.Mesh3D(
                vertex_positions=render.vertex_positions,
                triangle_indices=render.triangle_indices,
                vertex_normals=render.vertex_normals,
                vertex_colors=render.vertex_colors,
            ),
        )

    def clear(entity_path: str) -> None:
        rr.log(entity_path, rr.Clear(recursive=False))

    return ViewerBackend(
        init=init,
        log_points=log_points,
        log_mesh=log_mesh,
        clear=clear,
    )


BACKENDS: dict[BackendName, Callable[[], ViewerBackend]] = {
    "rerun": _build_rerun_viewer_backend,
}


def get_viewer_backend(ui: BackendName) -> ViewerBackend:
    return BACKENDS[ui]()
