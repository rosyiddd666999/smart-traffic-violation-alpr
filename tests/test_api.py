import json
import time

import pytest

TERMINAL = {"COMPLETED", "FAILED"}


def _upload(client, sample_video, *, zone=None, classes=None, filename=None, payload=None):
    path, info = sample_video
    content = payload if payload is not None else path.read_bytes()
    return client.post(
        "/api/v1/jobs",
        files={"file": (filename or path.name, content, "video/mp4")},
        data={
            "zone_coordinates": json.dumps(zone or info["zone_coordinates"]),
            "target_classes": json.dumps(classes or [2, 3, 5, 7]),
        },
    )


def _wait_for_job(client, job_id: str, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    latest = {}
    while time.time() < deadline:
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        latest = response.json()
        if latest["status"] in TERMINAL:
            return latest
        time.sleep(0.15)
    pytest.fail(f"Job {job_id} tidak selesai. Status terakhir: {latest}")


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["database"] is True
    assert payload["worker"] is True
    assert payload["mock_pipeline"] is True


def test_rejects_invalid_zone(client, sample_video):
    response = _upload(client, sample_video, zone=[[0, 0], [10, 0]])
    assert response.status_code == 400
    assert "3 titik" in response.json()["detail"]


def test_rejects_non_video_file(client, sample_video):
    _path, info = sample_video
    response = _upload(client, sample_video, payload=b"ini bukan video")
    assert response.status_code == 400
    assert "video" in response.json()["detail"].lower()


def test_rejects_unsupported_extension(client, sample_video):
    _path, info = sample_video
    response = _upload(client, sample_video, filename="catatan.txt")
    assert response.status_code == 400
    assert "tidak didukung" in response.json()["detail"]


def test_job_not_found(client):
    response = client.get("/api/v1/jobs/job_tidak_ada")
    assert response.status_code == 404


def test_upload_process_and_evidence(client, sample_video):
    response = _upload(client, sample_video)
    assert response.status_code == 202, response.text
    payload = response.json()
    assert payload["job_id"].startswith("job_")
    assert payload["status"] == "PENDING"

    job = _wait_for_job(client, payload["job_id"])
    assert job["status"] == "COMPLETED", job.get("error_message")
    assert job["progress_percentage"] == 100.0
    assert job["processed_frames"] > 0
    assert job["total_frames"] > 0
    assert job["violations_detected"] >= 1

    assert job["output_video_url"]
    video_response = client.get(job["output_video_url"])
    assert video_response.status_code == 200
    assert len(video_response.content) > 1000
    assert video_response.headers["content-type"].startswith("video/")

    violations_response = client.get(
        "/api/v1/violations", params={"job_id": payload["job_id"]}
    )
    assert violations_response.status_code == 200
    violations = violations_response.json()
    assert len(violations) == job["violations_detected"]
    assert len(violations) >= 1

    first = violations[0]
    assert first["job_id"] == payload["job_id"]
    assert first["violation_type"] == "Melanggar Marka Garis Utuh"
    assert first["license_plate"]
    assert first["ocr_confidence"] > 0
    assert first["evidence_image_url"]
    # Tes berjalan di MOCK_OCR=1, jadi plat wajib ditandai simulasi.
    assert first["plate_source"] == "simulated"
    assert first["needs_verification"] is False

    evidence_response = client.get(first["evidence_image_url"])
    assert evidence_response.status_code == 200
    assert len(evidence_response.content) > 500
    assert evidence_response.headers["content-type"].startswith("image/")

    detail_response = client.get(f"/api/v1/violations/{first['id']}")
    assert detail_response.status_code == 200
    assert detail_response.json()["id"] == first["id"]

    assert job["mode"] == "simulated"
    assert job["mock_ocr"] is True

    stats = client.get("/api/v1/stats").json()
    # Statistik utama hanya menghitung data non-simulasi.
    assert stats["total_violations"] == (
        stats["readable_plates"] + stats["unreadable_plates"]
    )
    assert stats["simulated_violations"] >= len(violations)
    assert stats["completed_jobs"] >= 1
    assert stats["pending_jobs"] == 0
    assert sum(stats["vehicle_breakdown"].values()) == stats["total_violations"]

    jobs = client.get("/api/v1/jobs", params={"limit": 5}).json()
    assert any(item["job_id"] == payload["job_id"] for item in jobs)


def test_dashboard_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Smart Traffic Violation Monitor" in response.text
