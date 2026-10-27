import os
import tempfile
from pathlib import Path

# Konfigurasi environment HARUS di-set sebelum modul aplikasi diimpor.
_TEST_ROOT = Path(tempfile.mkdtemp(prefix="traffic-violation-tests-"))
os.environ["STORAGE_DIR"] = str(_TEST_ROOT / "storage")
os.environ["DATABASE_URL"] = f"sqlite:///{(_TEST_ROOT / 'test.db').as_posix()}"
os.environ["MOCK_PIPELINE"] = "1"
os.environ["MOCK_OCR"] = "1"
os.environ["MAX_UPLOAD_MB"] = "50"
os.environ["FRAME_STRIDE"] = "1"
os.environ["CORS_ORIGINS"] = "*"
os.environ["PROGRESS_INTERVAL_FRAMES"] = "2"

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from api import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory):
    from scripts.make_sample_video import generate_video

    path = tmp_path_factory.mktemp("media") / "sample.mp4"
    info = generate_video(
        path,
        width=480,
        height=320,
        fps=12,
        seconds=3.0,
    )
    return path, info
