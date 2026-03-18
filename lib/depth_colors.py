import logging
import math
from functools import lru_cache

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from lib.config import COLOR_MAP, DEPTH_MAX_M, DEPTH_MIN_M
from lib.models import ColorRGB, Point3D


@lru_cache(maxsize=1)
def _get_cmap(color_map: str) -> mcolors.Colormap:
    if not (cmap := plt.get_cmap(color_map)):
        raise ValueError(f"Color map {color_map} not found")
    return cmap


def depth_to_color(depth_m: float, shallowest_m: float, deepest_m: float) -> ColorRGB:
    cmap = _get_cmap(COLOR_MAP)

    if math.isclose(shallowest_m, deepest_m):
        logging.warning(
            f"Shallowest and deepest are the same: {shallowest_m} {deepest_m}"
        )
        return (255, 255, 255)

    t = 1.0 - mcolors.Normalize(vmin=shallowest_m, vmax=deepest_m)(depth_m)
    red, green, blue, _ = cmap(t)
    return (round(red * 255), round(green * 255), round(blue * 255))


def depth_colors(points: list[Point3D]) -> list[ColorRGB]:
    return [depth_to_color(abs(point[2]), DEPTH_MIN_M, DEPTH_MAX_M) for point in points]
