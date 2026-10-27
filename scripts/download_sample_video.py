"""Unduh video lalu lintas publik untuk smoke test model asli.

Sumber: contoh video dari proyek Supervision (Roboflow).
Jalankan: python scripts/download_sample_video.py
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

CANDIDATES = [
    "https://media.roboflow.com/supervision/video-examples/vehicles.mp4",
    "https://media.roboflow.com/supervision/videos/vehicles.mp4",
]

DESTINATION = Path("samples/vehicles.mp4")


def download() -> Path:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (traffic-violation-project)"}

    last_error: Exception | None = None
    for url in CANDIDATES:
        print(f"Mencoba: {url}")
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                total = int(response.headers.get("Content-Length") or 0)
                downloaded = 0
                with DESTINATION.open("wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            percent = downloaded * 100 / total
                            print(f"\r  {percent:5.1f}% ({downloaded}/{total} bytes)", end="")
                        else:
                            print(f"\r  {downloaded} bytes", end="")
            print()
            if DESTINATION.stat().st_size < 10_000:
                raise RuntimeError("File terlalu kecil, kemungkinan bukan video.")
            print(f"Berhasil: {DESTINATION} ({DESTINATION.stat().st_size} bytes)")
            return DESTINATION
        except (urllib.error.URLError, OSError, RuntimeError) as exc:
            last_error = exc
            print(f"  gagal: {exc}")
            DESTINATION.unlink(missing_ok=True)

    print("Gagal mengunduh video contoh.")
    if last_error:
        print(f"Error terakhir: {last_error}")
    sys.exit(1)


if __name__ == "__main__":
    download()
