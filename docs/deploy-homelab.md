# Deploy Homelab `nakamaserver`

Panduan ini khusus untuk server `nakamaserver` yang sudah memiliki stack
Traefik + Cloudflare Tunnel. Topologi publik tetap **satu pintu**:

```text
Internet
  │ HTTPS
  ▼
Cloudflare Edge
  │
  ▼
cloudflare-tunnel (container, tunnel homelab)
  │ http://traefik:80
  ▼
Traefik :80
  │ http://caddy:80
  ▼
Caddy (basic auth, body limit, CSP, no-store)
  │ http://app:8000
  ▼
app (FastAPI + YOLO GPU + ALPR CPU)
  └── volume traffic-alpr_app_data:/data
```

Tidak ada tunnel kedua dan tidak ada port publik baru. Caddy tetap menjadi
lapisan autentikasi; Traefik hanya meneruskan request.

## 1. Konfigurasi server ini

| Komponen | Nilai |
|---|---|
| Hostname | `nakamaserver` |
| Domain | `trafic-detection.abdulrosyid.my.id` |
| Tunnel | `cloudflare-tunnel` (homelab, bukan `primegate-tunnel`) |
| Target tunnel | `http://traefik:80` |
| Project | `/opt/traffic-alpr` |
| GPU | NVIDIA GTX 1050 Ti, Pascal `sm_61`, 4 GB VRAM |
| CPU | Intel i7 860, 8 thread, tanpa AVX/AVX2 |
| Mode | Hybrid: YOLO GPU, ALPR/OCR CPU |
| Data | 15 job, 58 violation, ±344 MB |

Desktop GUI tetap aktif. GNOME/Xorg ikut memakai VRAM, tetapi YOLO11n hanya
memakai ~100 MB tensor pada `imgsz=1280`, jadi headroom-nya masih longgar.
Biarkan `IMGSZ` kosong agar resolusinya menyesuaikan lebar video (640 untuk
sumber kecil, 960 untuk 1080p, 1280 untuk 4K); paksa `640` hanya bila benar-benar
terjadi OOM.

## 2. File yang dipakai

```bash
docker-compose.yml           # app + Caddy
docker-compose.gpu.yml       # overlay GPU (DEVICE/OCR_GPU bisa dioverride)
docker-compose.homelab.yml   # network homelab + label Traefik
.env                         # rahasia + konfigurasi server (mode 600)
```

Jangan menambahkan port publik ke Compose. Caddy hanya bind ke
`127.0.0.1:8011` untuk verifikasi lokal.

## 3. Prasyarat

- Docker Engine + Compose v2.24+ (dipakai `!override`)
- Driver NVIDIA 580 dan `nvidia-container-toolkit`
- Akses ke network external `homelab-network`
- Tunnel homelab sudah aktif

Cek:

```bash
docker network inspect homelab-network --format '{{.Name}}'
docker ps --format '{{.Names}}' | grep -E 'traefik|cloudflare-tunnel'
```

## 4. Instal driver NVIDIA (sekali)

Di `nakamaserver`, Legacy BIOS dan tanpa Secure Boot:

```bash
sudo apt update
sudo apt install -y nvidia-driver-580 ca-certificates curl gnupg

# Repo NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker

sudo reboot
```

Setelah reboot:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

Keduanya harus menampilkan GTX 1050 Ti. Driver 470 tidak dipakai karena terlalu
tua untuk CUDA 12.8.

## 5. Konfigurasi `.env`

Nilai penting:

```ini
PUBLIC_HOSTNAME=trafic-detection.abdulrosyid.my.id
CORS_ORIGINS=https://trafic-detection.abdulrosyid.my.id

APP_BIND=127.0.0.1
APP_PORT=8011

STORAGE_DIR=/data
DATABASE_URL=sqlite:////data/violations.db

DEVICE=0
OCR_GPU=false
# Kosong = adaptive: 640 / 960 / 1280 mengikuti lebar video.
IMGSZ=
FRAME_STRIDE=1

MAX_UPLOAD_MB=95
CADDY_MAX_BODY=100MB
```

Buat hash basic auth:

```bash
docker run --rm -it caddy:2-alpine caddy hash-password
```

Simpan hasilnya dengan kutip tunggal:

```ini
BASIC_AUTH_USER=operator
BASIC_AUTH_HASH='$2a$14$...'
```

Lalu:

```bash
chmod 600 .env
```

## 6. Build

Build tidak membutuhkan GPU dan bisa dilakukan sebelum maintenance:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  -f docker-compose.homelab.yml \
  build app
```

Validasi konfigurasi:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  -f docker-compose.homelab.yml \
  config >/dev/null
```

## 7. Migrasi data lama

Data ada di `storage/` proyek. Lakukan saat app belum berjalan.

```bash
# checkpoint WAL
python3 -c "import sqlite3; c=sqlite3.connect('storage/violations.db'); c.execute('PRAGMA wal_checkpoint(TRUNCATE)'); c.close()"

# salin ke volume
docker run --rm \
  -v traffic-alpr_app_data:/data \
  -v "$PWD/storage:/seed:ro" \
  alpine:3.20 \
  sh -c "cp -a /seed/. /data/ && rm -f /data/violations.db-wal /data/violations.db-shm && chown -R 10001:10001 /data"
```

Jangan menyalin `violations.db-wal`/`-shm` setelah checkpoint.

Jika `docker pull alpine:3.20` gagal karena timeout registry, gunakan image yang
sudah ada:

```bash
docker run --rm --entrypoint sh \
  -v traffic-alpr_app_data:/data \
  -v "$PWD/storage:/seed:ro" \
  caddy:2-alpine \
  -c "cp -a /seed/. /data/ && rm -f /data/violations.db-wal /data/violations.db-shm && chown -R 10001:10001 /data"
```

Verifikasi dengan checksum, bukan `diff -r` (owner `1000` vs `10001` akan
membuat `diff` mengembalikan selisih metadata walau isi file identik):

```bash
docker run --rm --entrypoint sh \
  -v traffic-alpr_app_data:/data \
  -v "$PWD/storage:/seed:ro" \
  caddy:2-alpine \
  -c '(cd /seed && find . -type f -print0 | sort -z | xargs -0 md5sum) > /tmp/a;
      (cd /data && find . -type f -print0 | sort -z | xargs -0 md5sum) > /tmp/b;
      diff /tmp/a /tmp/b && echo checksum_ok'
```

## 8. Jalankan stack

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  -f docker-compose.homelab.yml \
  up -d
```

Cek status:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  -f docker-compose.homelab.yml \
  ps

docker compose \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  -f docker-compose.homelab.yml \
  logs -f app
```

## 9. Verifikasi

### 9.1 GPU dan ALPR

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  -f docker-compose.homelab.yml \
  exec app python -c \
  "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Harus:

```text
True NVIDIA GeForce GTX 1050 Ti
```

Log ALPR harus menampilkan:

```text
ALPR dipaksa memakai CPUExecutionProvider (OCR_GPU=false)
```

### 9.2 Lokal

```bash
curl -i http://127.0.0.1:8011/health
curl -u operator:PASSWORD http://127.0.0.1:8011/health
```

Tanpa kredensial harus `401`; dengan kredensial harus `200`.

### 9.3 Traefik tanpa Cloudflare

```bash
curl -i -H 'Host: trafic-detection.abdulrosyid.my.id' http://127.0.0.1/health
curl -i -u operator:PASSWORD \
  -H 'Host: trafic-detection.abdulrosyid.my.id' \
  http://127.0.0.1/health
```

### 9.4 Data lama

```bash
curl -u operator:PASSWORD \
  https://trafic-detection.abdulrosyid.my.id/api/v1/stats

curl -u operator:PASSWORD \
  'https://trafic-detection.abdulrosyid.my.id/api/v1/jobs?limit=3'

curl -u operator:PASSWORD -I \
  https://trafic-detection.abdulrosyid.my.id/static/evidence/<job_id>_id_<track>.jpg
```

Target:

- 15 job, 58 violation
- plat contoh (`B 1234 XYZ`) tetap `plate_source=real`
- evidence `200`

### 9.5 Publik lewat Cloudflare

```bash
# tanpa kredensial: harus 401 dari Caddy
curl -sS -o /dev/null -w '%{http_code}\n' \
  https://trafic-detection.abdulrosyid.my.id/health

# dengan kredensial: harus 200
curl -sS -u operator https://trafic-detection.abdulrosyid.my.id/health
```

Ciri respons benar: header `www-authenticate: Basic realm="restricted"` dan
`server: cloudflare`. Kalau yang muncul halaman 404 Cloudflare, berarti ingress
tunnel belum memuat hostname tersebut.

### 9.6 Performa acuan (GTX 1050 Ti, `IMGSZ` adaptive)

Ada dua jenis angka yang berbeda dan sering tertukar. Jangan bandingkan
keduanya.

**Inferensi satu frame** (sumber 4K, `predict()` tanpa tracker, `imgsz=960`):

| Metrik | Nilai |
|---|---|
| YOLO warm inference | ±0,025 s/frame |
| VRAM tensor YOLO | ±92 MB |
| VRAM total GPU (termasuk GUI) | ±660 MiB dari 4096 MiB |
| Latensi publik dari edge CGK | ±160 ms |
| ALPR | CPU, plat contoh `B 1234 XYZ` confidence 0,98 |

**Job penuh** (klip 4K 3840×2160, 538 frame, `imgsz=1280`, 10 pelanggaran,
cache model hangat):

| Tahap | Total | Per frame | Porsi |
|---|---|---|---|
| Decode | 9,9 s | 18,3 ms | 13 % |
| Deteksi (`track` + ByteTrack) | 27,4 s | 50,7 ms | 36 % |
| Anotasi frame | 11,6 s | 21,4 ms | 15 % |
| Encoder `libx264` 4K | 13,0 s | 24,1 ms | 17 % |
| ALPR CPU (10 kendaraan) | 9,6 s | — | 12 % |
| DB + evidence + load model | 5,7 s | — | 7 % |
| **Total** | **77–82 s** | 143–153 ms | 100 % |

Headroom VRAM masih ±3,3 GB, jadi `imgsz=1280` tidak perlu diturunkan. Deteksi
adalah tahap terbesar (±36 %); menaikkan ke `1536` hanya menambah ±13 ms/frame
untuk objek yang jauh lebih kecil daripada kendaraan di garis stop, jadi
`1280` sudah titik seimbangnya. Job pertama setelah container dibuat akan
terasa lebih lama karena model plat diunduh dan dimuat ke cache.

## 10. Cloudflare Tunnel

Di Zero Trust → Networks → Tunnels → `cloudflare-tunnel` →
Public Hostnames, tambahkan:

```text
Subdomain : trafic-detection
Domain    : abdulrosyid.my.id
Service   : http://traefik:80
```

Lalu:

- Cache Rule: **Bypass cache** untuk hostname ini
- Bot Fight Mode: nonaktif untuk hostname ini
- SSL/TLS mode: **Full**

Jangan mengarahkan hostname ke `http://localhost:8011`; tunnel berjalan di
container dan tidak bisa menjangkau loopback host.

### 10.1 DNS record wajib ada

Menambah Public Hostname hanya mengubah ingress tunnel. DNS record CNAME-nya
kadang **tidak** dibuat otomatis. Cek dulu:

```bash
dig +short @1.1.1.1 trafic-detection.abdulrosyid.my.id
```

Kalau kosong, tambahkan manual di dashboard zona `abdulrosyid.my.id` →
**DNS → Records → Add record**:

```text
Type    : CNAME
Name    : trafic-detection
Target  : <TUNNEL_ID>.cfargotunnel.com
Proxy   : Proxied (awan oranye)
TTL     : Auto
```

Target CNAME memakai ID tunnel, bukan `traefik:80`. ID didapat dari dashboard
tunnel atau dari ingress:

```bash
docker logs cloudflare-tunnel --tail=20 | grep -o '"hostname":"[^"]*"' | tail -5
```

### 10.2 Cache negatif NXDOMAIN saat uji

Kalau hostname baru ditambahkan, resolver yang sempat ditanya **sebelum** record
ada akan menyimpan NXDOMAIN selama TTL SOA (1800 detik / 30 menit). Gejalanya:
`curl` gagal `Could not resolve host`, padahal DoH sudah mengembalikan IP.

Diagnosis cepat — bandingkan resolver biasa dengan DoH:

```bash
dig +short @1.1.1.1 trafic-detection.abdulrosyid.my.id            # bisa kosong
curl -s -H 'accept: application/dns-json' \
  'https://cloudflare-dns.com/dns-query?name=trafic-detection.abdulrosyid.my.id&type=A'
```

Kalau DoH menjawab tapi `dig` tidak, itu murni cache negatif. Sisa TTL bisa
dilihat di bagian AUTHORITY:

```bash
dig +noall +authority @1.1.1.1 trafic-detection.abdulrosyid.my.id A
```

Sementara menunggu, verifikasi tetap bisa dilakukan tanpa menunggu cache:

```bash
curl --doh-url https://cloudflare-dns.com/dns-query -u operator \
  https://trafic-detection.abdulrosyid.my.id/health

curl --resolve trafic-detection.abdulrosyid.my.id:443:104.21.91.167 \
  -u operator https://trafic-detection.abdulrosyid.my.id/health
```

Solusi permanen di server: aktifkan DNS-over-TLS agar tidak bergantung cache
port 53.

```bash
sudo mkdir -p /etc/systemd/resolved.conf.d
sudo tee /etc/systemd/resolved.conf.d/99-dot.conf >/dev/null <<'EOF'
[Resolve]
DNS=1.1.1.1#cloudflare-dns.com 1.0.0.1#cloudflare-dns.com
DNSOverTLS=opportunistic
EOF
sudo systemctl restart systemd-resolved
```

Pakai `opportunistic`, bukan `yes`, agar DNS tetap jalan bila port 853 diblokir.

## 11. Tuning VRAM

Mulai:

```ini
# Kosong = adaptive (640 / 960 / 1280 sesuai lebar video).
IMGSZ=
FRAME_STRIDE=1
```

Pantau:

```bash
nvidia-smi dmon
docker stats traffic-alpr-app-1
```

Jika OOM atau VRAM penuh:

```ini
IMGSZ=640
FRAME_STRIDE=2
```

Lalu restart app:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  -f docker-compose.homelab.yml \
  up -d app
```

## 12. Backup

Backup dari dalam container (database saja):

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  -f docker-compose.homelab.yml \
  exec app python scripts/backup_db.py
```

Hasil ada di `/data/backups/<timestamp>/`.

Backup penuh volume (database + evidence + video), termasuk checkpoint WAL:

```bash
TS=$(date +%Y%m%d-%H%M%S)
DEST="$HOME/backups/traffic-alpr/${TS}"
mkdir -p "$DEST"

docker exec traffic-alpr-app-1 python -c \
  "import sqlite3;c=sqlite3.connect('/data/violations.db');c.execute('PRAGMA wal_checkpoint(TRUNCATE)');print(c.execute('PRAGMA integrity_check').fetchone()[0]);c.close()"

docker run --rm \
  -v traffic-alpr_app_data:/data:ro \
  -v "$DEST":/backup \
  caddy:2-alpine \
  sh -c 'tar czf /backup/app_data.tar.gz -C /data .'
```

Verifikasi isi arsip sebelum dianggap selesai:

```bash
docker run --rm -v "$DEST":/backup caddy:2-alpine \
  sh -c 'tar tzf /backup/app_data.tar.gz | wc -l'
```

Volume berisi 103 file (±344 MB, terkompresi ±338 MB). Backup pasca-deploy
pertama: `$HOME/backups/traffic-alpr/20261027-212630-post-deploy`.

## 13. Rollback driver

Jika driver 580 membuat desktop/server tidak bisa boot normal:

1. Masuk ke TTY/SSH.
2. Hapus driver:

```bash
sudo apt purge '^nvidia-.*'
sudo apt install xserver-xorg-video-nouveau
sudo reboot
```

3. Jalankan stack CPU:

```bash
docker compose up -d
```

## 14. Troubleshooting

### 14.1 Job gagal `No module named 'lap'`

Gejala di log:

```text
ERROR: Could not install packages due to an OSError:
[Errno 13] Permission denied: '/opt/venv/lib/python3.12/site-packages/lap'
...
ModuleNotFoundError: No module named 'lap'
```

Penyebab: `pipeline.py` memakai `model.track(..., tracker="bytetrack.yaml")`,
bukan `predict()`. ByteTrack membutuhkan paket `lap`. Ultralytics mencoba
memasangnya sendiri saat runtime, tetapi container berjalan sebagai `appuser`
sehingga tidak boleh menulis ke `/opt/venv`.

Perbaikan permanen: `lap==0.5.13` sudah ada di `docker/requirements.txt`.
Untuk image yang sudah terlanjur dibangun tanpa `lap`, tambahkan layer kecil
berikut (tidak perlu rebuild penuh):

```dockerfile
FROM traffic-alpr:gpu
USER root
COPY lap-0.5.13-cp312-cp312-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl /tmp/lap.whl
RUN /opt/venv/bin/pip install --no-cache-dir \
    /tmp/lap-0.5.13-cp312-cp312-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl
USER appuser
```

Nama file wheel **tidak boleh diubah** — pip menolak nama yang tidak sesuai
format `nama-versi-python-abi-platform.whl`.

Verifikasi:

```bash
docker exec traffic-alpr-app-1 python -c "import lap; print(lap.__version__)"
```

Uji tracking pada frame nyata:

```bash
docker exec -i traffic-alpr-app-1 python - <<'PY'
import glob, cv2
from ultralytics import YOLO
cap = cv2.VideoCapture(sorted(glob.glob("/data/uploads/*.mp4"))[0])
m = YOLO("yolo11n.pt")
ids = set()
for _ in range(60):
    ok, f = cap.read()
    if not ok: break
    r = m.track(source=f, persist=True, classes=[2,3,5,7],
                tracker="bytetrack.yaml", device="0", imgsz=960, verbose=False)[0]
    if r.boxes.id is not None:
        ids.update(int(x) for x in r.boxes.id)
print("tracker_ids:", sorted(ids))
PY
```

Harus keluar daftar ID (mis. `[1, 2, 3, 4, 5, 6, 8, 15]`), bukan error.

### 14.2 Job lama berstatus FAILED menumpuk

Job yang gagal karena bug dependency tetap tersimpan di tabel `jobs`. Setelah
perbaikan, bersihkan agar statistik akurat (jalankan setelah memastikan
backup ada):

```bash
docker exec -i traffic-alpr-app-1 python - <<'PY'
import sqlite3
c = sqlite3.connect('/data/violations.db')
for (jid,) in c.execute("select id from jobs where status='FAILED'").fetchall():
    c.execute('delete from violations where job_id=?', (jid,))
    c.execute('delete from jobs where id=?', (jid,))
c.commit()
print('sisa job:', c.execute('select count(*) from jobs').fetchone()[0])
c.close()
PY
```

Hapus juga `/data/uploads/<job_id>.mp4` bila ada.
