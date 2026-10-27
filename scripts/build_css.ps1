# Build ulang CSS Tailwind lokal (dashboard tidak memakai CDN).
#
# Jalankan dari akar project setiap kali kelas utilitas di index.html berubah:
#   powershell -ExecutionPolicy Bypass -File scripts/build_css.ps1
#
# Butuh Node.js/npx. Hasilnya di-commit bersama kode:
#   static/assets/tailwind.css

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "Membangun Tailwind CSS..." -ForegroundColor Cyan
npx --yes tailwindcss@3.4.17 `
  -c tailwind.config.js `
  -i static/assets/tailwind.src.css `
  -o static/assets/tailwind.css `
  --minify

if ($LASTEXITCODE -ne 0) {
    throw "Build Tailwind gagal (exit code $LASTEXITCODE)."
}

$file = Get-Item "static/assets/tailwind.css"
Write-Host ("Selesai: {0} ({1} byte)" -f $file.FullName, $file.Length) -ForegroundColor Green
