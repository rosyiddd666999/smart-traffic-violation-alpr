"""Regresi untuk optimasi komposisi frame: blending hanya di area yang berubah."""

from __future__ import annotations

import numpy as np
import supervision as sv

from pipeline import _annotate_frame

FRAME_HEIGHT = 240
FRAME_WIDTH = 320
ZONE = [[40, 120], [280, 120], [300, 230], [20, 230]]


def _zone_bbox_and_header_mask(height: int, width: int) -> np.ndarray:
    """Mask piksel yang memang boleh berubah (zona + bar judul).

    Kotak zona dilebihkan 2 px karena `cv2.polylines` bertepi `LINE_AA`
    masih boleh menggambar di sekitar bounding box poligon.
    """
    touch = np.ones((height, width), dtype=bool)
    touch[max(0, 120 - 2) : min(height, 231 + 2), max(0, 20 - 2) : min(width, 301 + 2)] = False
    touch[0 : max(46, int(height * 0.075)) + 1, :] = False
    return touch


def test_annotation_keeps_pixels_outside_zone_and_header() -> None:
    frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 90, dtype=np.uint8)

    annotated = _annotate_frame(
        frame, sv.Detections.empty(), ZONE, set(), {}, "job", 0, 10, 0
    )

    assert annotated.shape == frame.shape
    assert annotated.dtype == frame.dtype
    assert not np.shares_memory(annotated, frame), "frame masukan tidak boleh diubah"

    untouched = _zone_bbox_and_header_mask(FRAME_HEIGHT, FRAME_WIDTH)
    assert np.array_equal(annotated[untouched], frame[untouched])

    changed = ~untouched
    assert not np.array_equal(annotated[changed], frame[changed])


def test_annotation_header_band_is_rendered() -> None:
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

    annotated = _annotate_frame(
        frame, sv.Detections.empty(), ZONE, set(), {}, "job", 3, 10, 2
    )

    header = annotated[0:10, :]
    assert header.any(), "bar judul harus tergambar"


def test_annotation_clips_zone_to_frame() -> None:
    frame = np.full((120, 160, 3), 40, dtype=np.uint8)
    edge_zone = [[0, 60], [159, 60], [159, 119], [0, 119]]

    annotated = _annotate_frame(
        frame, sv.Detections.empty(), edge_zone, set(), {}, "job", 0, 1, 0
    )

    assert annotated.shape == frame.shape
    # Baris 50 masih di antara bar judul (47 baris) dan zona (mulai y=60).
    assert np.array_equal(annotated[50, 0], frame[50, 0])
