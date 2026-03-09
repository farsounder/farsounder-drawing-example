# Simple example to demo getting and visualizing Argos data using the Python
# SDK - so far ReRun is the only supported UI.
import argparse
import logging
import time
from typing import Callable

import rerun as rr

from farsounder import config, subscriber
from farsounder.proto import nav_api_pb2

from lib.backends import build_rerun_viewer_backend
from lib.config import GRID_INTERVAL_M
from lib.navigation import BottomGeoReference, has_valid_navigation, local_bottom_vertices
from lib.viewers import (
    GriddedBottomSurfaceViewer,
    GriddedBottomViewer,
    LiveBottomSurfaceViewer,
    UnGriddedBottomViewer,
)


def get_message_counter() -> Callable[[], int]:
    message_count = 0

    def increment() -> int:
        nonlocal message_count
        message_count += 1
        return message_count

    return increment


def handle_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Argos data using the Python SDK")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Log level")
    parser.add_argument("--ui", type=str, default="rerun", choices=["rerun"], help="UI to use")
    return parser.parse_args()


def main() -> None:
    args = handle_arguments()
    logging.basicConfig(level=args.log_level)

    if args.ui != "rerun":
        raise ValueError(f"Unsupported UI: {args.ui}")

    logging.info("Initializing ReRun viewer")
    rr.init("grid-example")
    rr.spawn()
    viewer_backend = build_rerun_viewer_backend()

    cfg = config.build_config(
        host="127.0.0.1",
        subscribe=["TargetData"],
    )
    sub = subscriber.subscribe(cfg)

    geo_reference = BottomGeoReference()
    ungridded_viewer = UnGriddedBottomViewer(point_logger=viewer_backend.log_points)
    gridded_viewer = GriddedBottomViewer(
        interval_m=GRID_INTERVAL_M,
        point_logger=viewer_backend.log_points,
    )
    gridded_surface_viewer = GriddedBottomSurfaceViewer(
        mesh_logger=viewer_backend.log_mesh,
        clear_logger=viewer_backend.clear,
    )
    live_surface_viewer = LiveBottomSurfaceViewer(
        mesh_logger=viewer_backend.log_mesh,
        clear_logger=viewer_backend.clear,
    )
    message_counter = get_message_counter()

    def on_targets(message: nav_api_pb2.TargetData) -> None:
        if not has_valid_navigation(message):
            logging.warning("Skipping TargetData with invalid navigation values")
            return

        msg_num = message_counter()
        live_vertices, zone_changed = local_bottom_vertices(message, geo_reference)

        if zone_changed:
            ungridded_viewer.reset()
            gridded_viewer.reset()

        if not live_vertices:
            live_surface_viewer.reset()
            return

        points = [vertex.position for vertex in live_vertices]
        msg_num = None if zone_changed else msg_num
        ungridded_viewer.log_points(points, current_message=msg_num)
        gridded_viewer.log_points(points, current_message=msg_num)
        gridded_surface_viewer.log_points(
            gridded_viewer.cells,
            gridded_viewer.interval_m,
            current_message=msg_num,
        )
        live_surface_viewer.log_points(
            live_vertices,
            message,
            current_message=msg_num,
        )

    sub.on("TargetData", on_targets)

    try:
        sub.start()
        logging.info("Running... Press Ctrl+C to stop")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logging.info("Stopping...")
    finally:
        sub.stop()


if __name__ == "__main__":
    main()