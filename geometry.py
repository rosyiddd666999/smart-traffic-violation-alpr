"""Utility geometri zona (polygon) berbasis OpenCV + Supervision."""

from __future__ import annotations

import json
import math
from typing import Sequence

import cv2
import numpy as np


class PolygonError(ValueError):
    """Input poligon tidak valid."""


Point = Sequence[float]


def parse_polygon(raw: str) -> list[list[float]]:
    """Parse JSON string `[[x, y], ...]` menjadi list titik tervalidasi."""
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise PolygonError("zone_coordinates harus berupa JSON yang valid") from exc

    if not isinstance(data, list) or len(data) < 3:
        raise PolygonError("zone_coordinates minimal berisi 3 titik")

    points: list[list[float]] = []
    for index, item in enumerate(data):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise PolygonError(f"Titik ke-{index + 1} harus berformat [x, y]")
        try:
            x = float(item[0])
            y = float(item[1])
        except (TypeError, ValueError) as exc:
            raise PolygonError(f"Koordinat titik ke-{index + 1} harus berupa angka") from exc
        if not (math.isfinite(x) and math.isfinite(y)):
            raise PolygonError(f"Koordinat titik ke-{index + 1} tidak valid")
        if x < 0 or y < 0:
            raise PolygonError("Koordinat zona tidak boleh negatif")
        points.append([round(x, 3), round(y, 3)])

    if polygon_area(points) < 100:
        raise PolygonError("Luas zona terlalu kecil, gambar poligon yang lebih besar")

    return points


def clamp_polygon(points: Sequence[Point], width: int, height: int) -> list[list[float]]:
    """Batasi titik agar berada di dalam resolusi video."""
    max_x = max(0.0, float(width) - 1.0)
    max_y = max(0.0, float(height) - 1.0)
    return [
        [min(max(0.0, float(x)), max_x), min(max(0.0, float(y)), max_y)]
        for x, y in points
    ]


def polygon_area(points: Sequence[Point]) -> float:
    if len(points) < 3:
        return 0.0
    contour = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    return float(abs(cv2.contourArea(contour)))


def polygon_contains(points: Sequence[Point], point: Point) -> bool:
    """True bila titik berada di dalam/tepi poligon."""
    if len(points) < 3:
        return False
    contour = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0


def build_zone(points: Sequence[Point]):
    """Buat PolygonZone Supervision dengan anchor titik bawah kendaraan."""
    import supervision as sv

    polygon = np.array(points, dtype=np.int32)
    return sv.PolygonZone(
        polygon=polygon,
        triggering_anchors=[sv.Position.BOTTOM_CENTER],
    )
