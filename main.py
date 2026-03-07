import colorsys
import math
import time
from dataclasses import dataclass, field

import rerun as rr
import utm

from farsounder import config, subscriber
from farsounder.proto import nav_api_pb2

Point3D = tuple[float, float, float]
ZoneId = tuple[int, str]
DEPTH_MIN_M = 0.0
DEPTH_MAX_M = 25.0
GRID_INTERVAL_M = 1.0


def boat_position_to_utm(message: nav_api_pb2.TargetData) -> tuple[float, float, ZoneId]:
    easting, northing, zone_number, zone_letter = utm.from_latlon(
        message.position.lat,
        message.position.lon,
    )
    return easting, northing, (zone_number, zone_letter)


def rotate_boat_offset_to_world(
    forward_m: float,
    right_m: float,
    heading_deg: float,
) -> tuple[float, float]:
    heading_rad = math.radians(heading_deg)
    east_offset = forward_m * math.sin(heading_rad) + right_m * math.cos(heading_rad)
    north_offset = forward_m * math.cos(heading_rad) - right_m * math.sin(heading_rad)
    return east_offset, north_offset


def depth_to_color(depth_m: float, shallowest_m: float, deepest_m: float) -> tuple[int, int, int]:
    if math.isclose(shallowest_m, deepest_m):
        return (80, 180, 255)

    # Shallower points trend warm, deeper points trend cool.
    t = (depth_m - shallowest_m) / (deepest_m - shallowest_m)
    hue = (40.0 + (220.0 - 40.0) * t) / 360.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return (int(red * 255), int(green * 255), int(blue * 255))


def depth_colors(points: list[Point3D]) -> list[tuple[int, int, int]]:
    return [depth_to_color(abs(point[2]), DEPTH_MIN_M, DEPTH_MAX_M) for point in points]


@dataclass
class BottomGeoReference:
    current_zone: ZoneId | None = None
    origin_xy: tuple[float, float] | None = None

    def update(self, boat_xy: tuple[float, float], zone_id: ZoneId) -> bool:
        if self.current_zone is None:
            self.current_zone = zone_id
            self.origin_xy = boat_xy
            return False

        if self.current_zone != zone_id:
            print(f"UTM zone changed from {self.current_zone} to {zone_id}; resetting bottom view")
            self.current_zone = zone_id
            self.origin_xy = boat_xy
            return True

        if self.origin_xy is None:
            self.origin_xy = boat_xy

        return False

    def to_local_xy(self, world_xy: tuple[float, float]) -> tuple[float, float]:
        if self.origin_xy is None:
            raise ValueError("BottomGeoReference origin is not initialized")

        world_easting, world_northing = world_xy
        origin_easting, origin_northing = self.origin_xy
        return world_easting - origin_easting, world_northing - origin_northing


def has_valid_navigation(message: nav_api_pb2.TargetData) -> bool:
    nav_values = (
        message.position.lat,
        message.position.lon,
        message.heading.heading,
    )
    return all(math.isfinite(value) for value in nav_values)


def local_bottom_points(
    message: nav_api_pb2.TargetData,
    geo_reference: BottomGeoReference,
) -> tuple[list[Point3D], bool]:
    boat_easting, boat_northing, zone_id = boat_position_to_utm(message)
    zone_changed = geo_reference.update((boat_easting, boat_northing), zone_id)
    points, _ = bottom_points_in_world(message, geo_reference)
    return points, zone_changed


def bottom_points_in_world(
    message: nav_api_pb2.TargetData,
    geo_reference: BottomGeoReference,
) -> tuple[list[Point3D], ZoneId]:
    easting, northing, zone_id = boat_position_to_utm(message)
    heading_deg = message.heading.heading
    local_easting, local_northing = geo_reference.to_local_xy((easting, northing))

    points: list[Point3D] = []
    for bottom_bin in message.bottom:
        values = (
            bottom_bin.cross_range,
            bottom_bin.down_range,
            bottom_bin.depth,
            heading_deg,
            easting,
            northing,
        )
        if not all(math.isfinite(value) for value in values):
            continue

        # Positive cross-range is port/left, so flip it into a conventional right axis.
        right_m = -bottom_bin.cross_range
        east_offset, north_offset = rotate_boat_offset_to_world(
            forward_m=bottom_bin.down_range,
            right_m=right_m,
            heading_deg=heading_deg,
        )
        points.append(
            (
                local_easting + east_offset,
                local_northing + north_offset,
                bottom_bin.depth,
            )
        )

    return points, zone_id


@dataclass
class UnGriddedBottomViewer:
    entity_path: str = "world/bottom/ungridded"
    point_radius: float = 0.2
    accumulated_points: list[Point3D] = field(default_factory=list)

    def reset(self) -> None:
        self.accumulated_points.clear()

    def log_points(self, points: list[Point3D]) -> None:
        if not points:
            return

        self.accumulated_points.extend(points)
        rr.log(
            self.entity_path,
            rr.Points3D(
                self.accumulated_points,
                colors=depth_colors(self.accumulated_points),
                radii=self.point_radius,
            ),
        )
        print(f"Logged {len(points)} bottom points ({len(self.accumulated_points)} total)")


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


@dataclass
class GriddedBottomViewer:
    interval_m: float
    entity_path: str = "world/bottom/gridded"
    point_radius: float = 0.3
    cells: dict[tuple[int, int], GriddedCell] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.interval_m <= 0.0:
            raise ValueError("GriddedBottomViewer interval must be positive")

    def reset(self) -> None:
        self.cells.clear()

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

    def log_points(self, points: list[Point3D]) -> None:
        self.add_points(points)

        if not self.cells:
            return

        averaged_points = self.averaged_points()

        rr.log(
            self.entity_path,
            rr.Points3D(
                averaged_points,
                colors=depth_colors(averaged_points),
                radii=self.point_radius,
            ),
        )
        print(f"Logged {len(averaged_points)} gridded bottom cells")


@dataclass
class GriddedBottomSurfaceViewer:
    entity_path: str = "world/bottom/gridded_surface"

    def log_surface(self, cells: dict[tuple[int, int], GriddedCell], interval_m: float) -> None:
        if not cells:
            return

        cell_keys = sorted(cells)
        vertex_indices = {cell_key: idx for idx, cell_key in enumerate(cell_keys)}
        vertex_positions: list[Point3D] = []
        for cell_x, cell_y in cell_keys:
            cell = cells[(cell_x, cell_y)]
            vertex_positions.append(
                (
                    (cell_x + 0.5) * interval_m,
                    (cell_y + 0.5) * interval_m,
                    cell.average_depth,
                )
            )

        triangle_indices: list[tuple[int, int, int]] = []
        for cell_x, cell_y in cell_keys:
            quad_keys = (
                (cell_x, cell_y),
                (cell_x + 1, cell_y),
                (cell_x, cell_y + 1),
                (cell_x + 1, cell_y + 1),
            )
            if not all(key in vertex_indices for key in quad_keys):
                continue

            lower_left = vertex_indices[(cell_x, cell_y)]
            lower_right = vertex_indices[(cell_x + 1, cell_y)]
            upper_left = vertex_indices[(cell_x, cell_y + 1)]
            upper_right = vertex_indices[(cell_x + 1, cell_y + 1)]
            triangle_indices.append((lower_left, lower_right, upper_left))
            triangle_indices.append((lower_right, upper_right, upper_left))

        if not triangle_indices:
            return

        rr.log(
            self.entity_path,
            rr.Mesh3D(
                vertex_positions=vertex_positions,
                triangle_indices=triangle_indices,
                vertex_colors=depth_colors(vertex_positions),
            ),
        )
        print(f"Logged gridded surface with {len(triangle_indices)} triangles")


def main() -> None:
    rr.init("grid-example")
    rr.spawn()

    cfg = config.build_config(
        host="127.0.0.1",
        subscribe=["TargetData"],
    )

    sub = subscriber.subscribe(cfg)
    geo_reference = BottomGeoReference()
    ungridded_viewer = UnGriddedBottomViewer()
    gridded_viewer = GriddedBottomViewer(interval_m=GRID_INTERVAL_M)
    gridded_surface_viewer = GriddedBottomSurfaceViewer()

    def on_targets(message: nav_api_pb2.TargetData) -> None:
        if not has_valid_navigation(message):
            print("Skipping TargetData with invalid navigation values")
            return

        points, zone_changed = local_bottom_points(message, geo_reference)

        if zone_changed:
            ungridded_viewer.reset()
            gridded_viewer.reset()

        if not points:
            return

        ungridded_viewer.log_points(points)
        gridded_viewer.log_points(points)
        gridded_surface_viewer.log_surface(gridded_viewer.cells, gridded_viewer.interval_m)

    sub.on("TargetData", on_targets)


    try:
        sub.start()
        print("Running... Press Ctrl+C to stop")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        sub.stop()


if __name__ == "__main__":
    main()