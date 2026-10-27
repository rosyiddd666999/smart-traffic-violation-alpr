"""Potong segmen video dan encode ulang ke H.264 agar ringan diuji.

Contoh:
    # 8 detik mulai menit 02:00, perkecil ke lebar 960 (tinggi otomatis genap)
    python scripts/cut_video.py input.mp4 -o segmen.mp4 --start 00:02:00 --duration 8 --width 960

    # tanpa resize (hanya potong + H.264)
    python scripts/cut_video.py input.mp4 -o segmen.mp4 --start 10 --duration 6
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from video_io import probe_video  # noqa: E402


def _ffmpeg() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - tergantung environment
        raise RuntimeError(
            "imageio-ffmpeg tidak tersedia. Jalankan: pip install imageio-ffmpeg"
        ) from exc


def cut_video(
    source: Path,
    destination: Path,
    *,
    start: str | float = 0,
    duration: str | float | None = None,
    width: int | None = None,
    crf: int = 23,
    preset: str = "veryfast",
) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"Video sumber tidak ditemukan: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        _ffmpeg(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        str(start),
        "-i",
        str(source),
    ]
    if duration:
        command += ["-t", str(duration)]
    if width:
        # -2 menjaga tinggi tetap genap (syarat encoder H.264).
        command += ["-vf", f"scale={max(2, width)}:-2"]
    command += [
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg gagal memotong video:\n" + (result.stderr or "").strip()
        )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Potong segmen video dan encode H.264 untuk pengujian."
    )
    parser.add_argument("input", help="Path video sumber.")
    parser.add_argument("-o", "--output", help="Path video hasil.")
    parser.add_argument(
        "--start",
        default="0",
        help="Waktu mulai (detik atau HH:MM:SS). Default 0.",
    )
    parser.add_argument(
        "--duration",
        default=None,
        help="Durasi (detik atau HH:MM:SS). Default sampai akhir video.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=960,
        help="Lebar output; tinggi mengikuti rasio. 0 = tanpa resize. Default 960.",
    )
    parser.add_argument("--crf", type=int, default=23, help="Kualitas H.264 (default 23).")
    args = parser.parse_args()

    source = Path(args.input)
    destination = Path(args.output) if args.output else source.with_name(
        f"{source.stem}_seg.mp4"
    )

    output = cut_video(
        source,
        destination,
        start=args.start,
        duration=args.duration,
        width=args.width or None,
        crf=args.crf,
    )

    print(f"Video hasil: {output}")
    info = probe_video(output)
    if info is None:
        raise SystemExit("Hasil tidak bisa dibaca OpenCV. Periksa log ffmpeg di atas.")
    print(
        f"Resolusi   : {info.width}x{info.height} @ {info.fps:.2f} fps\n"
        f"Durasi     : {info.duration:.2f} detik ({info.total_frames} frame)\n"
        f"Ukuran     : {output.stat().st_size / (1024 * 1024):.2f} MB"
    )


if __name__ == "__main__":
    main()
