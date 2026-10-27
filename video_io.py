"""Pembacaan metadata video dan penulisan video output (H.264)."""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float


def probe_video(path: Path) -> Optional[VideoInfo]:
    """Buka video dan baca frame pertama; None jika tidak bisa didekode."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return None
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if not (0.1 <= fps <= 240):
            fps = 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ok, _frame = capture.read()
        if not ok or width <= 0 or height <= 0:
            return None
        duration = total_frames / fps if total_frames > 0 else 0.0
        return VideoInfo(
            width=width,
            height=height,
            fps=fps,
            total_frames=max(0, total_frames),
            duration=duration,
        )
    finally:
        capture.release()


def _ffmpeg_executable() -> Optional[str]:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # pragma: no cover - bergantung paket opsional
        logger.warning("imageio-ffmpeg tidak tersedia, memakai codec mp4v internal OpenCV.")
        return None


class VideoWriter:
    """Encoder H.264 agar hasil bisa diputar di browser.

    Jika imageio-ffmpeg tidak tersedia, fallback ke cv2.VideoWriter (mp4v).
    """

    def __init__(self, path: Path, width: int, height: int, fps: float) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width = max(2, width - (width % 2))
        self.height = max(2, height - (height % 2))
        self.fps = max(1.0, float(fps))
        self._process: Optional[subprocess.Popen] = None
        self._cv_writer: Optional[cv2.VideoWriter] = None
        self._started = False
        self._failed = False

    def _start(self) -> None:
        executable = _ffmpeg_executable()
        if executable:
            command = [
                executable,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{self.width}x{self.height}",
                "-r",
                f"{self.fps:.6f}",
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self.path),
            ]
            kwargs = {}
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
            try:
                self._process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    **kwargs,
                )
                logger.info("Encoder H.264 aktif untuk %s", self.path.name)
                return
            except OSError:
                logger.exception("Gagal menjalankan ffmpeg, fallback ke mp4v.")
                self._process = None

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._cv_writer = cv2.VideoWriter(
            str(self.path), fourcc, self.fps, (self.width, self.height)
        )
        if not self._cv_writer.isOpened():
            self._cv_writer = None
            raise RuntimeError("Tidak ada encoder video yang tersedia")

    def write(self, frame: np.ndarray) -> None:
        if self._failed or frame is None or frame.size == 0:
            return
        if not self._started:
            self._start()
            self._started = True

        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        if self._process is not None and self._process.stdin is not None:
            try:
                self._process.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError):
                logger.exception("Encoder H.264 berhenti lebih awal.")
                self._failed = True
        elif self._cv_writer is not None:
            self._cv_writer.write(frame)

    def close(self) -> None:
        if self._process is not None:
            if self._process.stdin is not None:
                try:
                    self._process.stdin.close()
                except OSError:
                    pass
            try:
                return_code = self._process.wait(timeout=120)
                if return_code != 0:
                    logger.error("ffmpeg keluar dengan kode %s", return_code)
            except subprocess.TimeoutExpired:  # pragma: no cover
                self._process.kill()
            finally:
                self._process = None
        if self._cv_writer is not None:
            self._cv_writer.release()
            self._cv_writer = None
