# Simple example to demo getting and visualizing Argos data using the Python
# SDK - so far ReRun is the only supported UI.
import argparse
import logging
import time
from typing import Callable

from farsounder import config, subscriber
from farsounder.proto import nav_api_pb2

from lib.backends import get_viewer_backend, BACKENDS
from lib.config import (
    GRID_INTERVAL_M,
)
from lib.navigation import (
    BottomGeoReference,
    has_valid_navigation,
    local_bottom_vertices,
    local_iwt_vertices,
)
import lib.viewers


def get_message_counter() -> Callable[[], int]:
    message_count = 0

    def increment() -> int:
        nonlocal message_count
        message_count += 1
        return message_count

    return increment


def handle_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Argos data using the Python SDK"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level",
    )
    parser.add_argument(
        "--ui", type=str, default="rerun", choices=BACKENDS.keys(), help="UI to use"
    )
    return parser.parse_args()


def main() -> None:
    args = handle_arguments()
    logging.basicConfig(level=args.log_level)

    logging.info("Initializing viewer")
    viewer_backend = get_viewer_backend(args.ui)
    viewer_backend.init()

    cfg = config.build_config(
        host="127.0.0.1",
        subscribe=["TargetData"],
    )
    sub = subscriber.subscribe(cfg)

    geo_reference = BottomGeoReference()
    ungridded_viewer = lib.viewers.UnGriddedBottomViewer(
        point_logger=viewer_backend.log_points
    )
    raw_target_viewer = lib.viewers.RawTargetViewer(
        point_logger=viewer_backend.log_points
    )
    gridded_target_viewer = lib.viewers.GriddedTargetViewer(
        interval_m=GRID_INTERVAL_M,
        point_logger=viewer_backend.log_points,
    )

    gridded_viewer = lib.viewers.GriddedBottomViewer(
        interval_m=GRID_INTERVAL_M,
        point_logger=viewer_backend.log_points,
        mesh_logger=viewer_backend.log_mesh,
        clear_logger=viewer_backend.clear,
        show_surface=True,
    )
    live_surface_viewer = lib.viewers.LiveBottomSurfaceViewer(
        mesh_logger=viewer_backend.log_mesh,
        clear_logger=viewer_backend.clear,
    )
    bottom_viewers = {
        "ungridded": ungridded_viewer,
        "gridded": gridded_viewer,
        "live_surface": live_surface_viewer,
    }
    target_viewers = {
        "raw": raw_target_viewer,
        "gridded": gridded_target_viewer,
    }
    message_counter = get_message_counter()

    def on_targets(message: nav_api_pb2.TargetData) -> None:
        if not has_valid_navigation(message):
            logging.warning("Skipping TargetData with invalid navigation values")
            return

        msg_num = message_counter()
        live_bottom_vertices, zone_changed = local_bottom_vertices(
            message, geo_reference
        )
        live_iwt_vertices, _ = local_iwt_vertices(message, geo_reference)

        if zone_changed:
            for viewer in bottom_viewers.values():
                viewer.reset()
            for viewer in target_viewers.values():
                viewer.reset()

        msg_num = None if zone_changed else msg_num
        for viewer in target_viewers.values():
            viewer.log_points(live_iwt_vertices, current_message=msg_num)

        if not live_bottom_vertices:
            return

        for viewer in bottom_viewers.values():
            viewer.log_points(
                live_bottom_vertices, current_message=msg_num, message=message
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
