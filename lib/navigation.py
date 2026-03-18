import logging
import math
from dataclasses import dataclass

import utm

from farsounder.proto import nav_api_pb2

from lib.models import LiveVertex, ZoneId
from lib.time import time_it


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
            logging.info(
                f"UTM zone changed from {self.current_zone} to {zone_id}; resetting bottom view"
            )
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


def boat_position_to_utm(
    message: nav_api_pb2.TargetData,
) -> tuple[float, float, ZoneId]:
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


def has_valid_navigation(message: nav_api_pb2.TargetData) -> bool:
    nav_values = (
        message.position.lat,
        message.position.lon,
        message.heading.heading,
    )
    return all(math.isfinite(value) for value in nav_values)


@time_it(name="local_bottom_vertices")
def local_bottom_vertices(
    message: nav_api_pb2.TargetData,
    geo_reference: BottomGeoReference,
) -> tuple[list[LiveVertex], bool]:
    boat_easting, boat_northing, zone_id = boat_position_to_utm(message)
    zone_changed = geo_reference.update((boat_easting, boat_northing), zone_id)
    heading_deg = message.heading.heading
    local_easting, local_northing = geo_reference.to_local_xy(
        (boat_easting, boat_northing)
    )

    vertices: list[LiveVertex] = []
    for bottom_bin in message.bottom:
        values = (
            bottom_bin.cross_range,
            bottom_bin.down_range,
            bottom_bin.depth,
            heading_deg,
            boat_easting,
            boat_northing,
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
        vertices.append(
            LiveVertex(
                position=(
                    local_easting + east_offset,
                    local_northing + north_offset,
                    bottom_bin.depth,
                ),
                down_range_m=bottom_bin.down_range,
            )
        )

    return vertices, zone_changed


@time_it(name="local_iwt_vertices")
def local_iwt_vertices(
    message: nav_api_pb2.TargetData,
    geo_reference: BottomGeoReference,
) -> tuple[list[LiveVertex], bool]:
    boat_easting, boat_northing, zone_id = boat_position_to_utm(message)
    zone_changed = geo_reference.update((boat_easting, boat_northing), zone_id)
    heading_deg = message.heading.heading
    local_easting, local_northing = geo_reference.to_local_xy(
        (boat_easting, boat_northing)
    )

    vertices: list[LiveVertex] = []
    for group in message.groups:
        for target_bin in group.bins:
            values = (
                target_bin.cross_range,
                target_bin.down_range,
                target_bin.depth,
                heading_deg,
                boat_easting,
                boat_northing,
            )
            if not all(math.isfinite(value) for value in values):
                continue

            # Target bins use the same boat-relative axes as bottom bins.
            right_m = -target_bin.cross_range
            east_offset, north_offset = rotate_boat_offset_to_world(
                forward_m=target_bin.down_range,
                right_m=right_m,
                heading_deg=heading_deg,
            )
            vertices.append(
                LiveVertex(
                    position=(
                        local_easting + east_offset,
                        local_northing + north_offset,
                        target_bin.depth,
                    ),
                    down_range_m=target_bin.down_range,
                )
            )

    return vertices, zone_changed


def get_horizontal_angle_spacing_rad(message: nav_api_pb2.TargetData) -> float | None:
    hor_angles = [
        math.radians(angle)
        for angle in message.grid_description.hor_angles
        if math.isfinite(angle)
    ]
    return abs(hor_angles[1] - hor_angles[0])
