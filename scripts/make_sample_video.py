"""Generator video CCTV sintetis untuk demo dan pengujian pipeline.

Jalankan:
    python scripts/make_sample_video.py
Output default: samples/sample_traffic.mp4
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

# Izinkan eksekusi langsung: python scripts/make_sample_video.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from video_io import VideoWriter  # noqa: E402

# Zona terlarang default (relatif terhadap 1280x720) yang kompatibel
# dengan lintasan kendaraan pada video sintetis.
DEFAULT_ZONE_1280 = [[250, 500], [1030, 500], [1120, 700], [160, 700]]


def _draw_road(frame: np.ndarray, width: int, height: int) -> None:
    frame[:] = (38, 40, 44)

    # Aspal dengan gradasi halus.
    gradient = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
    frame[:] = (frame * (0.75 + 0.35 * gradient)).astype(np.uint8)

    horizon = int(height * 0.42)
    # Trotoar / area atas.
    cv2.rectangle(frame, (0, 0), (width, horizon), (58, 60, 64), -1)

    # Marka putus-putus.
    dash_length = max(18, height // 30)
    gap = dash_length
    lane_xs = [int(width * 0.22), int(width * 0.78)]
    for lane_x in lane_xs:
        y = horizon
        while y < height:
            cv2.rectangle(
                frame,
                (lane_x - 3, y),
                (lane_x + 3, min(height, y + dash_length)),
                (210, 210, 210),
                -1,
            )
            y += dash_length + gap

    # Garis utuh yang dilanggar.
    line_y = int(height * 0.72)
    cv2.line(frame, (0, line_y), (width, line_y), (0, 215, 255), 6, cv2.LINE_AA)

    # Area zig-zag / zona terlarang.
    zone_top = line_y + 6
    zone_bottom = height - 20
    left, right = int(width * 0.10), int(width * 0.90)
    cv2.rectangle(
        frame,
        (left, zone_top),
        (right, zone_bottom),
        (52, 60, 72),
        -1,
    )
    step = max(24, width // 40)
    for x in range(left - (zone_bottom - zone_top), right, step * 2):
        cv2.line(
            frame,
            (x, zone_bottom),
            (x + (zone_bottom - zone_top), zone_top),
            (0, 190, 230),
            2,
            cv2.LINE_AA,
        )
    cv2.rectangle(frame, (left, zone_top), (right, zone_bottom), (0, 190, 230), 2)

    cv2.putText(
        frame,
        "ZONA TERLARANG (MARKA GARIS UTUH)",
        (left + 12, zone_top + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.5, width / 2200),
        (0, 200, 240),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "SIMULASI CCTV - SMART TRAFFIC MONITORING",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.55, width / 1900),
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )


def _draw_plate(
    frame: np.ndarray,
    center_x: int,
    bottom_y: int,
    plate_width: int,
    plate_text: str,
) -> None:
    plate_width = max(46, plate_width)
    plate_height = max(18, int(plate_width * 0.34))
    x1 = int(center_x - plate_width / 2)
    x2 = int(center_x + plate_width / 2)
    y2 = int(bottom_y - max(4, plate_height * 0.25))
    y1 = y2 - plate_height

    cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (20, 20, 20), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (245, 245, 245), -1)

    font_scale = max(0.30, plate_height / 62.0)
    thickness = max(1, int(plate_height / 34))
    (text_w, text_h), baseline = cv2.getTextSize(
        plate_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    while text_w > (x2 - x1) - 8 and font_scale > 0.2:
        font_scale -= 0.04
        (text_w, text_h), baseline = cv2.getTextSize(
            plate_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
    text_x = int(center_x - text_w / 2)
    text_y = int((y1 + y2) / 2 + text_h / 2)
    cv2.putText(
        frame,
        plate_text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (15, 15, 15),
        thickness,
        cv2.LINE_AA,
    )


def _draw_car(
    frame: np.ndarray,
    center_x: int,
    bottom_y: int,
    car_width: int,
    car_height: int,
    color: tuple[int, int, int],
    plate_text: str,
) -> None:
    x1 = int(center_x - car_width / 2)
    x2 = int(center_x + car_width / 2)
    y2 = int(bottom_y)
    y1 = int(bottom_y - car_height)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)

    roof_h = max(6, int(car_height * 0.42))
    roof_x1 = x1 + int(car_width * 0.20)
    roof_x2 = x2 - int(car_width * 0.20)
    cv2.rectangle(frame, (roof_x1, y1 - roof_h), (roof_x2, y1 + 4), color, -1)

    glass = (150, 170, 185)
    cv2.rectangle(
        frame,
        (roof_x1 + 4, y1 - roof_h + 5),
        (roof_x2 - 4, y1 - 2),
        glass,
        -1,
    )
    cv2.line(
        frame,
        (int(center_x), y1 - roof_h + 5),
        (int(center_x), y1 - 2),
        tuple(int(value * 0.7) for value in color),
        2,
    )

    headlight = (220, 240, 255)
    light_w = max(4, int(car_width * 0.10))
    light_h = max(3, int(car_height * 0.08))
    cv2.rectangle(
        frame,
        (x1 + 4, y1 + int(car_height * 0.12)),
        (x1 + 4 + light_w, y1 + int(car_height * 0.12) + light_h),
        headlight,
        -1,
    )
    cv2.rectangle(
        frame,
        (x2 - 4 - light_w, y1 + int(car_height * 0.12)),
        (x2 - 4, y1 + int(car_height * 0.12) + light_h),
        headlight,
        -1,
    )

    wheel_w = max(6, int(car_width * 0.12))
    wheel_h = max(4, int(car_height * 0.12))
    cv2.rectangle(
        frame,
        (x1 + int(car_width * 0.12), y2 - wheel_h),
        (x1 + int(car_width * 0.12) + wheel_w, y2),
        (18, 18, 18),
        -1,
    )
    cv2.rectangle(
        frame,
        (x2 - int(car_width * 0.12) - wheel_w, y2 - wheel_h),
        (x2 - int(car_width * 0.12), y2),
        (18, 18, 18),
        -1,
    )

    _draw_plate(
        frame,
        center_x=int(center_x),
        bottom_y=y2,
        plate_width=int(car_width * 0.34),
        plate_text=plate_text,
    )


def _draw_motorcycle(
    frame: np.ndarray,
    center_x: int,
    bottom_y: int,
    vehicle_width: int,
    vehicle_height: int,
    color: tuple[int, int, int],
    plate_text: str,
) -> None:
    x1 = int(center_x - vehicle_width / 2)
    x2 = int(center_x + vehicle_width / 2)
    y2 = int(bottom_y)
    y1 = int(bottom_y - vehicle_height)
    wheel_radius = max(4, int(vehicle_width * 0.18))

    cv2.circle(
        frame,
        (x1 + wheel_radius, y2 - wheel_radius),
        wheel_radius,
        (20, 20, 20),
        -1,
    )
    cv2.circle(
        frame,
        (x2 - wheel_radius, y2 - wheel_radius),
        wheel_radius,
        (20, 20, 20),
        -1,
    )

    body_top = int(y1 + vehicle_height * 0.42)
    body_bottom = int(y2 - wheel_radius)
    cv2.rectangle(
        frame,
        (x1 + int(vehicle_width * 0.18), body_top),
        (x2 - int(vehicle_width * 0.18), body_bottom),
        color,
        -1,
    )

    rider_x = int(center_x)
    rider_y = int(y1 + vehicle_height * 0.24)
    rider_r = max(4, int(vehicle_width * 0.14))
    cv2.circle(frame, (rider_x, rider_y), rider_r, (190, 200, 210), -1)
    cv2.rectangle(
        frame,
        (rider_x - rider_r, rider_y + rider_r),
        (rider_x + rider_r, body_top + 2),
        (30, 80, 190),
        -1,
    )

    _draw_plate(
        frame,
        center_x=center_x,
        bottom_y=y2,
        plate_width=int(vehicle_width * 0.55),
        plate_text=plate_text,
    )


def generate_video(
    output: str | Path,
    *,
    width: int = 1280,
    height: int = 720,
    fps: int = 15,
    seconds: float = 8.0,
) -> dict:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_frames = max(2, int(fps * seconds))
    # VideoWriter memakai H.264 via imageio-ffmpeg agar bisa diputar
    # langsung di browser (mp4v tidak didukung Chrome/Edge).
    writer = VideoWriter(output_path, width, height, float(fps))

    try:
        for index in range(total_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            _draw_road(frame, width, height)

            progress = index / max(1, total_frames - 1)

            # Mobil: menuju kamera, membesar.
            car_width = int(width * (0.085 + 0.120 * progress))
            car_height = int(car_width * 0.72)
            car_center = int(
                width * 0.40 + width * 0.01 * math.sin(progress * math.tau)
            )
            car_bottom = int(height * (0.68 + 0.29 * progress))
            _draw_car(
                frame,
                car_center,
                car_bottom,
                car_width,
                car_height,
                (40, 60, 190),
                "B 1234 XYZ",
            )

            # Motor: jalur kanan.
            motor_width = int(width * (0.050 + 0.070 * progress))
            motor_height = int(motor_width * 1.05)
            motor_center = int(
                width * 0.63 + width * 0.008 * math.sin(progress * math.tau + 1.0)
            )
            motor_bottom = int(height * (0.70 + 0.27 * progress))
            _draw_motorcycle(
                frame,
                motor_center,
                motor_bottom,
                motor_width,
                motor_height,
                (30, 140, 210),
                "D 5678 AB",
            )

            timestamp = f"00:{int(index / fps):02d}:{int((index % fps) * 1000 / fps):03d}"
            cv2.putText(
                frame,
                f"2026-10-26 {timestamp}",
                (width - int(width * 0.24), height - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.45, width / 2400),
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        writer.close()

    zone = [
        [
            round(x * width / 1280.0, 2),
            round(y * height / 720.0, 2),
        ]
        for x, y in DEFAULT_ZONE_1280
    ]
    return {
        "path": str(output_path),
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
        "zone_coordinates": zone,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Buat video CCTV sintetis.")
    parser.add_argument(
        "--output",
        default="samples/sample_traffic.mp4",
        help="Path output video.",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--seconds", type=float, default=8.0)
    args = parser.parse_args()

    info = generate_video(
        args.output,
        width=args.width,
        height=args.height,
        fps=args.fps,
        seconds=args.seconds,
    )
    print(f"Video sampel dibuat: {info['path']}")
    print(f"Resolusi: {info['width']}x{info['height']} @ {info['fps']} fps")
    print("Zona yang disarankan:")
    print(info["zone_coordinates"])


if __name__ == "__main__":
    main()
