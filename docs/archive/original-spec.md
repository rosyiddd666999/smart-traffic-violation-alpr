> **Archived original specification (Indonesian).**
>
> This document describes the first iteration of the project (EasyOCR, Celery,
> simulated plates). It is kept for history only and is **superseded** by
> [`../end-to-end.md`](../end-to-end.md) and the current [`README.md`](../../README.md).
> Several claims below — especially the OCR engine, task queue, and mock plates —
> no longer describe the running system.

---

# 🚦 Smart Traffic Violation & ALPR Surveillance Pipeline

Sistem pemantauan lalu lintas otomatis berbasis video rekaman CCTV/kamera pengawas untuk kepolisian lalu lintas (ETLE/Electronic Traffic Law Enforcement). Petugas cukup mengunggah file video rekaman dan mendefinisikan titik koordinat zona terlarang; sistem akan memproses video secara asinkron di latar belakang, melacak pergerakan kendaraan, mengidentifikasi pelanggaran marka jalan, mengambil bukti foto beresolusi optimal (*smart-crop*), mengekstrak nomor plat menggunakan ALPR, dan menyimpan seluruh log ke database.

---

## 🛠️ Tech Stack & Architecture

- **Backend & API:** FastAPI (Python 3.10+)
- **Computer Vision:** YOLOv11 / YOLOv8 (Ultralytics) untuk deteksi objek kendaraan
- **Multi-Object Tracking (MOT):** ByteTrack (`persist=True`)
- **Spatial Geometry:** Supervision Toolkit (Polygon Point-in-Polygon logic)
- **Optical Character Recognition (OCR):** EasyOCR + OpenCV Preprocessing (CLAHE, Bilateral Filter, Super-Resolution Resizing)
- **Database:** SQLite (Dapat dimigrasikan ke PostgreSQL)
- **Task Management:** BackgroundTasks / Celery Worker (Asynchronous processing)

---

## 🏗️ System Workflow

```text
[Polisi / Client]
       │
       ▼  1. POST /api/v1/jobs (Upload Video + Parameter Poligon Zona)
[FastAPI Server] ──► Simpan file sementara di storage (/uploads)
       │
       ▼  2. Buat Task ID & Dispatch Job ke Background Worker
[Processing Pipeline]
       │
       ├──► Baca stream video frame-by-frame
       ├──► YOLOv11 Engine: Deteksi [Car, Motorcycle, Bus, Truck]
       ├──► ByteTrack: Penetapan ID persisten tiap kendaraan
       ├──► PolygonZone: Pengecekan posisi jangkar ban (BOTTOM_CENTER) terhadap marka jalan
       ├──► Jarak Terdekat Engine: Update & pilih frame saat dimensi bodi (w * h) maksimum
       └──► ALPR & Evidence Extractor:
               ├── Crop bodi bawah kendaraan
               ├── Image Enhancement (CLAHE + Upscaling)
               ├── EasyOCR Character Recognition
               └── Simpan bukti visual ke /bukti_pelanggaran
       │
       ▼  3. Ekspor & Rekapitulasi Data
[Database SQLite] ◄── Rekam metadata: ID, Jenis, Waktu, Nomor Plat, Path Foto
       │
       ▼  4. GET /api/v1/violations
[Dashboard Monitoring] ◄── Menampilkan daftar pelanggaran terverifikasi beserta foto bukti
```

---

## 📁 Struktur Direktori

```text
traffic-violation-surveillance/
│
├── api.py                    # Server FastAPI, endpoint upload, status job, dan query data
├── pipeline.py               # Modul logika computer vision (YOLO + ByteTrack + ALPR)
├── database.py               # Konfigurasi SQLAlchemy / SQLite connection
├── models.py                 # Skema tabel database (Violations & ProcessingJobs)
├── requirements.txt          # Dependensi Python
├── README.md                 # Dokumentasi proyek
│
├── storage/
│   ├── uploads/              # Direktori video mentah yang diunggah petugas
│   ├── output_videos/        # Video hasil render dengan bounding box & counter
│   ├── evidence/             # Potongan foto plat dan mobil pelanggar
│   └── violations.db         # Database SQLite lokal
```

---

## 🔌 API Specification & Contracts

### 1. Submit Video Analysis Job
Membuat antrean analisis video baru.

- **Endpoint:** `POST /api/v1/jobs`
- **Content-Type:** `multipart/form-data`
- **Payload:**
  - `file`: Video file (`.mp4`, `.avi`, `.mkv`)
  - `zone_coordinates`: JSON String titik poligon (misal: `[[1250, 1050], [1650, 1050], [1650, 1850], [850, 1850]]`)
  - `target_classes`: JSON Array kelas COCO (Default: `[2, 3, 5, 7]`)

**Sample Response (`202 Accepted`):**
```json
{
  "job_id": "job_9b83f0c1",
  "status": "PROCESSING",
  "message": "Video berhasil diunggah dan sedang diproses di latar belakang."
}
```

---

### 2. Check Job Processing Status
Melihat kemajuan rendering frame video.

- **Endpoint:** `GET /api/v1/jobs/{job_id}`
- **Sample Response (`200 OK`):**
```json
{
  "job_id": "job_9b83f0c1",
  "status": "COMPLETED",
  "progress_percentage": 100,
  "total_frames": 900,
  "processed_frames": 900,
  "violations_detected": 4,
  "output_video_url": "/static/output_videos/job_9b83f0c1.mp4"
}
```

---

### 3. List All Detected Violations
Mengambil data log pelanggaran untuk kebutuhan bukti tilang.

- **Endpoint:** `GET /api/v1/violations?limit=50&offset=0`
- **Sample Response (`200 OK`):**
```json
[
  {
    "id": 1,
    "job_id": "job_9b83f0c1",
    "tracker_id": 3,
    "vehicle_type": "Mobil",
    "violation_type": "Melanggar Marka Garis Utuh",
    "time_seconds": 2.45,
    "frame_number": 74,
    "license_plate": "B1234XYZ",
    "ocr_confidence": 0.88,
    "evidence_image_url": "/static/evidence/job_9b83f0c1_id_3.jpg",
    "created_at": "2026-10-26T13:40:00"
  }
]
```

---

## 🗄️ Database Schema (`traffic_violations.db`)

### Tabel: `jobs`
| Kolom | Tipe | Deskripsi |
| :--- | :--- | :--- |
| `id` | VARCHAR(64) | Primary Key (UUID / Job ID) |
| `original_filename`| TEXT | Nama file video yang diunggah |
| `status` | VARCHAR(20) | `PENDING`, `PROCESSING`, `COMPLETED`, `FAILED` |
| `progress` | FLOAT | Persentase eksekusi (0.0 - 100.0) |
| `total_violations` | INTEGER | Jumlah pelanggaran terverifikasi |
| `created_at` | TIMESTAMP | Waktu unggah |

### Tabel: `violations`
| Kolom | Tipe | Deskripsi |
| :--- | :--- | :--- |
| `id` | INTEGER | Primary Key (Auto Increment) |
| `job_id` | VARCHAR(64) | Foreign Key ke tabel `jobs.id` |
| `tracker_id` | INTEGER | ID persisten kendaraan dari tracker |
| `vehicle_type` | VARCHAR(32) | Jenis kendaraan (Mobil, Truk, Bus, Motor) |
| `frame_number` | INTEGER | Frame saat kendaraan mencapai titik terdekat |
| `time_seconds` | FLOAT | Timestamp detik video kejadian pelanggaran |
| `license_plate` | VARCHAR(20) | Hasil ekstraksi teks nomor polisi |
| `ocr_confidence` | FLOAT | Tingkat akurasi pembacaan ALPR |
| `proof_image_path`| TEXT | Lokasi file gambar hasil pemotongan |
| `created_at` | TIMESTAMP | Waktu pencatatan sistem |

---

## 🚀 Setup & Panduan Menjalankan

### 1. Persiapan Environment
```bash
git clone https://github.com/username/traffic-violation-surveillance.git
cd traffic-violation-surveillance

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

*Isi `requirements.txt`:*
```text
fastapi>=0.115.0
uvicorn>=0.30.0
ultralytics>=8.3.0
supervision>=0.24.0
opencv-python>=4.10.0
numpy>=1.26.0
easyocr>=1.7.0
python-multipart>=0.0.9
pydantic>=2.8.0
```

### 2. Jalankan Server API
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Dokumentasi API interaktif dapat diakses melalui browser di:
`http://localhost:8000/docs`

---

## 💡 Instruksi Pengembangan Lanjutan untuk AI Agent

Jika kode ini dikembangkan lebih lanjut menggunakan AI Agent:
1. **Pisahkan Worker:** Gunakan `FastAPI.BackgroundTasks` untuk skenario prototipe mandiri, atau ganti dengan Celery/Redis jika sistem dipersiapkan menangani antrean video berukuran di atas 1 GB secara simultan.
2. **Kamera Kalibrasi:** Tambahkan modul Perspective Transform (*Bird's-Eye View Matrix*) jika petugas menginginkan kalkulasi estimasi kecepatan kendaraan (km/jam) selain pelanggaran garis marka.
3. **Dedicated Plate Detector:** Jika EasyOCR mengalami degradasi pembacaan karakter pada citra sudut miring, tambahkan model deteksi sekunder (misal YOLO khusus lisensi plat nomor) sebelum melempar potongan gambar ke OCR.