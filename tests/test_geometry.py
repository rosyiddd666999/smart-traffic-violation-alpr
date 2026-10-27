import numpy as np
import pytest
import supervision as sv

from geometry import (
    PolygonError,
    build_zone,
    clamp_polygon,
    parse_polygon,
    polygon_area,
    polygon_contains,
)

SQUARE = [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]


def test_parse_polygon_valid():
    points = parse_polygon("[[10, 20], [110, 20], [110, 120], [10, 120]]")
    assert points == [[10.0, 20.0], [110.0, 20.0], [110.0, 120.0], [10.0, 120.0]]
    assert polygon_area(points) == pytest.approx(10000.0)


def test_parse_polygon_rejects_too_few_points():
    with pytest.raises(PolygonError):
        parse_polygon("[[0, 0], [10, 10]]")


def test_parse_polygon_rejects_degenerate_area():
    with pytest.raises(PolygonError):
        parse_polygon("[[0, 0], [10, 0], [20, 0]]")


def test_parse_polygon_rejects_non_numeric():
    with pytest.raises(PolygonError):
        parse_polygon("[[0, 0], [100, 0], [100, 'x']]")


def test_parse_polygon_rejects_invalid_json():
    with pytest.raises(PolygonError):
        parse_polygon("bukan json")


def test_clamp_polygon():
    points = clamp_polygon([[-10, -5], [500, 50], [50, 500]], width=200, height=100)
    assert points[0] == [0.0, 0.0]
    assert points[1] == [199.0, 50.0]
    assert points[2] == [50.0, 99.0]


def test_polygon_contains():
    assert polygon_contains(SQUARE, [50, 50])
    assert polygon_contains(SQUARE, [0, 0])
    assert not polygon_contains(SQUARE, [150, 50])
    assert not polygon_contains(SQUARE, [50, -1])


def test_zone_uses_bottom_center_anchor():
    zone = build_zone([[0, 50], [100, 50], [100, 150], [0, 150]])

    inside = sv.Detections(
        xyxy=np.array([[10, 0, 50, 80]], dtype=np.float32),
        tracker_id=np.array([1]),
    )
    outside = sv.Detections(
        xyxy=np.array([[10, 0, 50, 40]], dtype=np.float32),
        tracker_id=np.array([2]),
    )

    assert bool(zone.trigger(inside)[0]) is True
    assert bool(zone.trigger(outside)[0]) is False
