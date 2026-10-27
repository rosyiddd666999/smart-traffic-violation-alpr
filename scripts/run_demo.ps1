# Jalankan mode demo tanpa model AI (mock detector + mock OCR).
# Berguna untuk mencoba seluruh alur API dan dashboard dengan video sintetis.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$env:MOCK_PIPELINE = "1"
$env:MOCK_OCR = "1"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Error "Virtual environment belum ada. Jalankan setup terlebih dahulu (lihat README)."
    exit 1
}

$sample = "samples\sample_traffic.mp4"
if (-not (Test-Path $sample)) {
    Write-Host "Membuat video sampel sintetis..." -ForegroundColor Cyan
    & ".venv\Scripts\python.exe" "scripts\make_sample_video.py"
}

Write-Host ""
Write-Host "Mode demo aktif (MOCK_PIPELINE=1, MOCK_OCR=1)." -ForegroundColor Yellow
Write-Host "Buka http://127.0.0.1:8000 lalu unggah $sample" -ForegroundColor Green
Write-Host "Tekan Ctrl+C untuk berhenti." -ForegroundColor DarkGray
Write-Host ""

& ".venv\Scripts\python.exe" -m uvicorn api:app --host 127.0.0.1 --port 8000
