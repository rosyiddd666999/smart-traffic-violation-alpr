# Deploy Docker + Cloudflare Tunnel

> **Untuk `nakamaserver`, gunakan [deploy-homelab.md](deploy-homelab.md).**
> Server tersebut memakai `cloudflare-tunnel` + Traefik yang sudah ada, bukan
> cloudflared systemd di host seperti panduan generik ini.

Panduan ini memasang **Smart Traffic Violation & ALPR** di server pribadi:
Cloudflare Tunnel (di host) → Caddy (basic auth) → aplikasi FastAPI/ALPR (GPU
atau CPU) dengan data persisten di volume Docker.

```
Internet
   │ HTTPS (Cloudflare edge)
   ▼
cloudflared (systemd di host, outbound-only)
   │ http://127.0.0.1:8011
   ▼
Caddy container :80  (basic auth, body <=100MB, CSP, no-store)
   │ http://app:8000
   ▼
app container (GPU) ── volume traffic-alpr_app_data:/data
   ├── violations.db (SQLite + WAL)
   ├── uploads/
   ├── output_videos/
   ├── evidence/
   └── backups/
```

Tidak ada port 80/443 yang dibuka ke internet. Server hanya perlu SSH keluar
untuk build dan koneksi tunnel.

---

## 1. Prasyarat server

| Kebutuhan | Keterangan |
|---|---|
| OS | Linux x86_64 (Ubuntu 22.04/24.04, Debian 12) |
| Docker | Docker Engine 24+ dan Docker Compose v2.24+ (`!override` di overlay direct) |
| GPU (opsional) | Driver NVIDIA + `nvidia-container-toolkit` |
| Disk | Minimal 20 GB bebas (≈1 GB per job video 4K) |
| RAM | 4 GB minimum, 8 GB disarankan |
| cloudflared | Sudah terpasang dan login di host (systemd) |

Verifikasi cepat di server:

```bash
docker --version
docker compose version
nvidia-smi
cloudflared tunnel list
systemctl status cloudflared --no-pager
```

Uji akses GPU ke container (wajib bila memakai overlay GPU):

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

Jika perintah itu gagal, perbaiki `nvidia-container-toolkit` dulu — image GPU
tidak akan bisa memakai CUDA tanpa itu.

---

## 2. Kirim project ke server

Dari laptop (Windows/PowerShell atau WSL):

```bash
rsync -av \
  --exclude .venv --exclude storage --exclude samples \
  --exclude __pycache__ --exclude .pytest_cache \
  ./ user@SERVER:/opt/traffic-alpr/
```

Atau lewat `scp -r`. Yang **tidak** boleh ikut: `.venv/`, `storage/` (data
dikirim terpisah di langkah 6), `samples/`.

---

## 3. Siapkan `.env`

```bash
cd /opt/traffic-alpr
cp .env.docker.example .env
```

Buat hash password basic auth:

```bash
docker run --rm caddy:2-alpine caddy hash-password --plaintext 'passwordRahasiaAnda'
```

Salin hasilnya (diawali `$2a$14$...`) ke `.env`:

```ini
BASIC_AUTH_USER=operator
BASIC_AUTH_HASH='$2a$14$HASIL_HASH_DI_SINI'
```

> **Penting:** hash harus diapit **kutip tunggal**. Tanpa itu Docker Compose
> menganggap `$2a` sebagai variabel dan hash menjadi rusak.

Sesuaikan juga:

```ini
PUBLIC_HOSTNAME=trafic-detection.abdulrosyid.my.id
CORS_ORIGINS=https://trafic-detection.abdulrosyid.my.id
APP_BIND=127.0.0.1      # cloudflared ada di host
APP_PORT=8011           # port yang dipakai tunnel
MAX_UPLOAD_MB=95        # batas Cloudflare Free 100 MB
DEVICE=0                # GPU 0
```

Cek `.env` tidak terbaca publik:

```bash
chmod 600 .env
```

---

## 4. Build & jalankan

### Dengan GPU (sesuai server Anda)

```bash
cd /opt/traffic-alpr
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

Build pertama mengunduh CUDA base, torch, dan model AI: **±10–20 menit** dan
butuh internet. Build berikutnya memakai cache.

### Tanpa GPU (fallback)

```bash
docker compose up -d --build
```

### Pantau proses

```bash
docker compose ps
docker compose logs -f app
docker compose logs -f caddy
```

Tunggu sampai `app` berstatus `healthy` (healthcheck memanggil `/health`).

---

## 5. Verifikasi lokal (sebelum menyentuh Cloudflare)

```bash
# Lewat Caddy (loopback)
curl -i http://127.0.0.1:8011/health                 # 401 tanpa kredensial
curl -u operator:passwordRahasiaAnda http://127.0.0.1:8011/health

# Container berjalan sebagai non-root
docker compose exec app id                           # uid=10001(appuser)

# Hanya loopback yang terbuka
ss -tlnp | grep -E ':8011|:80 |:443'
```

`/health` harus memuat:

```json
{
  "status": "ok",
  "database": true,
  "worker": true,
  "mock_pipeline": false,
  "mock_ocr": false,
  "device": "0",
  "imgsz_auto": true,
  "ocr_engine": "fast-alpr"
}
```

### Verifikasi GPU

```bash
docker compose exec app python -c \
  "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

docker compose exec app python -c \
  "import onnxruntime as ort; print(ort.get_available_providers())"
```

Yang diharapkan:

- `cuda: True <nama GPU>`
- `['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`
  (TensorRT boleh tidak ada)
- Saat job berjalan: log ALPR memuat `ALPR memakai CUDAExecutionProvider`
- `nvidia-smi` menunjukkan pemakaian memori GPU

Jika `cuda: False`, lihat bagian [Troubleshooting](#10-troubleshooting).

---

## 6. Migrasi data lama (15 job, termasuk job dengan plat terbaca)

Data di laptop: `violations.db`, `uploads/`, `output_videos/`, `evidence/`
(total ±344 MB). Path di database relatif (`evidence/...`), jadi tidak ada
yang perlu diubah setelah disalin.

### 6.1 Rapikan WAL di laptop (sekali)

```powershell
.\.venv\Scripts\python.exe -c "import sqlite3; c=sqlite3.connect('storage/violations.db'); c.execute('PRAGMA wal_checkpoint(TRUNCATE)'); c.close()"
```

### 6.2 Kirim ke server

```bash
scp -r storage user@SERVER:/opt/traffic-alpr/seed
```

### 6.3 Salin ke volume Docker

```bash
cd /opt/traffic-alpr
docker compose stop app          # pastikan app tidak menulis saat migrasi

docker run --rm \
  -v traffic-alpr_app_data:/data \
  -v /opt/traffic-alpr/seed:/seed:ro \
  alpine:3.20 \
  sh -c "cp -a /seed/. /data/ && chown -R 10001:10001 /data"

docker compose start app
```

> Nama volume mengikuti `name: traffic-alpr` di compose, yaitu
> `traffic-alpr_app_data`. Cek dengan `docker volume ls`.

### 6.4 Verifikasi

```bash
curl -u operator:password https://trafic-detection.abdulrosyid.my.id/api/v1/stats
curl -u operator:password https://trafic-detection.abdulrosyid.my.id/api/v1/jobs?limit=3
```

Job lama harus muncul, plat hasil OCR tetap `plate_source=real`, dan thumbnail
evidence tetap terbuka. Hapus `seed/` setelah yakin:

```bash
rm -rf /opt/traffic-alpr/seed
```

---

## 7. Cloudflare Tunnel (cloudflared di host)

### 7.1 Arahkan tunnel ke Caddy

**Dashboard-managed** (Zero Trust → Networks → Tunnels → tunnel Anda →
Public Hostnames):

```
Subdomain : trafic-detection
Domain    : abdulrosyid.my.id
Service   : http://localhost:8011
```

**Locally-managed** (`/etc/cloudflared/config.yml`):

```yaml
tunnel: <UUID-TUNNEL>
credentials-file: /etc/cloudflared/<UUID-TUNNEL>.json
ingress:
  - hostname: trafic-detection.abdulrosyid.my.id
    service: http://localhost:8011
  - service: http_status:404
```

```bash
sudo cloudflared tunnel ingress validate
sudo systemctl restart cloudflared
```

> Jika tunnel sebelumnya diarahkan ke `localhost:8000` (uvicorn lama), ubah ke
> **8011** agar basic auth, batas upload, dan header keamanan Caddy ikut aktif.
> Hentikan uvicorn lama di port 8011 agar tidak bentrok:
> `sudo systemctl stop traffic-alpr` (atau `pkill -f uvicorn`).

### 7.2 DNS

```bash
dig +short trafic-detection.abdulrosyid.my.id
# harus CNAME ke <uuid>.cfargotunnel.com
```

### 7.3 SSL/TLS

Dashboard Cloudflare → SSL/TLS → Overview → mode **Full**. Jangan `Flexible`
(tidak relevan untuk tunnel, tapi memicu loop redirect bila nanti ada origin
HTTP lain).

### 7.4 Cache Rule — WAJIB

Tanpa rule ini, Cloudflare dapat meng-cache gambar evidence/video dan
menyajikannya **tanpa melewati basic auth**.

Dashboard → Caching → Cache Rules → Create rule:

- **If**: `Hostname equals trafic-detection.abdulrosyid.my.id`
- **Then**: *Bypass cache*

Aplikasi juga mengirim `Cache-Control: private, no-store` untuk semua respons
sebagai lapisan kedua.

### 7.5 Pengaturan lain yang disarankan

| Pengaturan | Nilai | Alasan |
|---|---|---|
| Bot Fight Mode | Nonaktif untuk hostname ini | Polling `/api/v1/jobs` bisa kena challenge |
| Cloudflare Access | Pilih salah satu dengan basic auth | Dua lapis login membingungkan |
| Brotli | Aktif (default) | Aset CSS/HTML lebih kecil |

### 7.6 Verifikasi publik

```bash
curl -i https://trafic-detection.abdulrosyid.my.id/health
# HTTP/2 401 + WWW-Authenticate: Basic

curl -u operator:password https://trafic-detection.abdulrosyid.my.id/health
# 200 {"status":"ok",...}
```

Dashboard: `https://trafic-detection.abdulrosyid.my.id/` → browser meminta
username/password.

---

## 8. Batas upload Cloudflare Free

| Lapisan | Batas | File |
|---|---|---|
| Cloudflare | 100 MB per request | tidak bisa diubah di Free |
| Caddy | `CADDY_MAX_BODY` (default 100MB) | `.env` |
| Aplikasi | `MAX_UPLOAD_MB=95` | `.env` |

Konsekuensi:

- Video 4K ±35 MB aman.
- Video >100 MB **ditolak Cloudflare** dengan 413 sebelum mencapai server.
  Opsi: upgrade plan, potong video, atau unggah lewat LAN
  (`docker-compose.direct.yml`).
- Timeout origin Cloudflare ±100 detik. Upload 95 MB pada uplink <8 Mbps bisa
  berakhir `524`. Bila sering terjadi, turunkan `MAX_UPLOAD_MB=50` dan
  `CADDY_MAX_BODY=60MB`, lalu `docker compose up -d` (Caddy membaca ulang
  `.env`).

Pemrosesan video berjalan asinkron (`202 Accepted`), jadi durasi analisis tidak
terkena timeout Cloudflare.

---

## 9. Operasional

### Log

```bash
docker compose logs -f app
docker compose logs -f caddy
docker compose logs --tail=100 app | grep -i -E "cuda|provider|error"
```

Rotasi otomatis: 10 MB × 3 file per container.

### Backup

```bash
# Di dalam container (aman terhadap WAL)
docker compose exec app python scripts/backup_db.py

# Salin ke host
mkdir -p /opt/backups
docker run --rm \
  -v traffic-alpr_app_data:/data \
  -v /opt/backups:/backup \
  alpine:3.20 \
  sh -c "tar czf /backup/traffic-alpr-$(date +%F-%H%M).tar.gz -C /data backups"
```

Hasil: `/data/backups/<timestamp>/violations.db` + `evidence.tar.gz`.

Restore:

```bash
docker compose stop app
docker run --rm -v traffic-alpr_app_data:/data -v /opt/backups:/backup alpine:3.20 \
  sh -c "cd /data && tar xzf /backup/traffic-alpr-XXXX.tar.gz && chown -R 10001:10001 /data"
docker compose start app
```

### Update aplikasi

```bash
rsync -av --exclude .venv --exclude storage --exclude samples ./ user@SERVER:/opt/traffic-alpr/
cd /opt/traffic-alpr
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

Volume `/data` tidak tersentuh. Model di-download ulang hanya saat layer build
berubah (atau tetap memakai cache BuildKit).

### Restart saat ada job berjalan

Job yang sedang diproses akan ditandai `FAILED` oleh `recover_stale_jobs()`
saat startup. Video perlu diunggah ulang. Ini perilaku aplikasi saat ini
(bukan bug Docker); jadwalkan restart saat antrean kosong.

### Skala

**Jangan** memakai `--scale app=2` atau `--workers 2`:

- worker pipeline adalah thread in-process;
- SQLite tidak cocok untuk dua penulis;
- hasilnya job ganda/race.

### Retensi disk

±1 GB per job (upload + output). Pantau:

```bash
docker system df -v
du -sh /var/lib/docker/volumes/traffic-alpr_app_data/_data
```

Hapus job lama lewat dashboard/aplikasi, bukan dengan `rm` langsung ke DB.

---

## 10. Troubleshooting

| Gejala | Penyebab umum | Solusi |
|---|---|---|
| `401` terus setelah login | `BASIC_AUTH_HASH` rusak karena `$` tidak dikutip | Regenerasi hash, tulis dengan kutip tunggal, `docker compose up -d caddy` |
| Caddy crash: `invalid hash` | Hash bukan bcrypt / kosong | `caddy hash-password` ulang |
| `502 Bad Gateway` | App belum healthy / crash | `docker compose logs app`; cek `docker compose ps` |
| `413` saat upload 40 MB | `MAX_UPLOAD_MB`/`CADDY_MAX_BODY` terlalu kecil | Naikkan (tetap ≤95 MB untuk Cloudflare Free) |
| `413` saat upload >100 MB | Batas Cloudflare Free | Potong video / upgrade plan / unggah via LAN |
| `524` saat upload | Uplink lambat, >100 detik | Turunkan batas upload, ulangi dari jaringan lebih cepat |
| `Permission denied` di `/data` | Ownership volume salah | `docker run --rm -v traffic-alpr_app_data:/data alpine chown -R 10001:10001 /data` |
| `torch.cuda.is_available() == False` | Container tidak dapat GPU | Pastikan build memakai `docker-compose.gpu.yml`; cek `nvidia-container-toolkit`; jalankan `docker compose exec app nvidia-smi` |
| Log: `CUDAExecutionProvider tidak ada` | Memakai image CPU, atau cuDNN tidak ditemukan | Pakai image GPU + base CUDA 12.8; cek `LD_LIBRARY_PATH` |
| Log: `libcudnn.so.9: cannot open` | Base image bukan cuDNN 9 | Jangan ganti base ke `python:slim`; pakai `CUDA_IMAGE` resmi NVIDIA |
| Model diunduh ulang tiap start | Cache tidak terbawa / `HOME` berubah | Pastikan `HOME=/home/appuser`; jangan mount volume di atas `/home/appuser` |
| Dashboard blank / tanpa styling | `/static/assets` tidak termuat | `curl -I http://127.0.0.1:8011/static/assets/tailwind.css -u ...` → harus 200 |
| CSS kelas baru tidak muncul | Tailwind perlu di-build ulang | Di laptop: `powershell -File scripts/build_css.ps1`, lalu kirim ulang `static/assets/tailwind.css` |
| Tunnel 502 | Service tunnel menunjuk `localhost:8000` | Ubah ke `http://localhost:8011` |

### Kumpulkan diagnostik

```bash
docker compose ps
docker compose logs --tail=200 app
docker compose exec app python -c "import torch, onnxruntime as ort; print(torch.cuda.is_available(), ort.get_available_providers())"
docker compose exec app sh -c "id; ls -la /data | head"
ss -tlnp | grep 8011
journalctl -u cloudflared -n 100 --no-pager
```

---

## 11. Pilihan versi CUDA / driver

`Dockerfile.gpu` menerima build arg. Default: CUDA 12.8 + torch cu126
(torch 2.14.1 belum tersedia di index cu128).

| Driver NVIDIA | `CUDA_IMAGE` | `TORCH_INDEX_URL` |
|---|---|---|
| ≥ 570 | `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04` (default) | `https://download.pytorch.org/whl/cu126` |
| 550–569 | `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` | `https://download.pytorch.org/whl/cu124` + turunkan `TORCH_VERSION` ke versi yang tersedia di index tersebut (maks 2.6.0) |
| < 550 | Gunakan image CPU | — |

Ubah lewat `.env`:

```ini
CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
```

Lalu build ulang tanpa cache untuk layer torch:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build --no-cache app
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

> `onnxruntime-gpu` 1.31 membutuhkan **CUDA 12.8+** dan **cuDNN 9**. Bila base
> diturunkan ke CUDA 12.4, turunkan juga `onnxruntime-gpu` ke versi yang cocok
> (mis. 1.22.x) — atau tetap di CUDA 12.8 bila driver mendukung.

---

## 12. Akses LAN (opsional, untuk uji tanpa tunnel)

```bash
docker compose -f docker-compose.yml -f docker-compose.direct.yml up -d
# http://<IP-server>:8011  (basic auth tetap aktif)
```

Jangan membuka port ini ke internet. Setelah selesai, kembali ke mode normal:

```bash
docker compose -f docker-compose.yml -f docker-compose.direct.yml down
docker compose up -d
```

---

## 13. Batasan yang perlu diketahui

- Satu job diproses pada satu waktu (antrean in-process). Belum bisa
  scale-out tanpa pindah ke Celery/PostgreSQL.
- Batas 100 MB request Cloudflare Free tidak bisa dilewati tanpa upgrade plan
  atau perubahan arsitektur upload (R2 presigned URL).
- Restart saat job berjalan membuat job tersebut `FAILED`.
- GPU hanya terpakai bila memakai overlay GPU; image CPU tetap tersedia
  sebagai fallback.
- Tailwind CSS dibangun saat pengembangan (`scripts/build_css.ps1`); image
  Docker tidak memerlukan Node.
