# Security Policy

## Supported versions

Only the `main` branch receives fixes. There are no tagged releases yet.

## Reporting a vulnerability

Please **do not** open a public issue for security problems.

- Use GitHub's private vulnerability reporting:
  `Security` tab -> `Report a vulnerability`.
- If that is unavailable, open a minimal public issue asking for a private
  contact channel, without disclosing details.

Include: affected endpoint or file, reproduction steps, impact, and the version
or commit hash you tested.

You can expect an acknowledgement within 7 days.

## Scope

The project processes video footage and licence plates. Treat any deployment as
handling personal data:

| Data | Where it lives | Comitted to Git? |
|---|---|---|
| Uploaded videos | `storage/uploads/` (or `/data/uploads`) | No |
| Evidence images | `storage/evidence/` | No |
| Rendered videos | `storage/output_videos/` | No |
| SQLite database | `storage/*.db` | No |
| Basic-auth bcrypt hash | `.env` | No |
| Demo dataset (anonymised) | `seed/` | Yes, synthetic only |

The public demo dataset in `seed/` contains **no real licence plates**. Plate
numbers were replaced with synthetic values and the plate pixels in evidence
images were blurred.

## Deployment hardening checklist

Run through this list before exposing the app to the internet.

### Secrets

- [ ] `.env` exists, is **not** tracked by Git, and has mode `600`:
      `git check-ignore -v .env && chmod 600 .env`
- [ ] `BASIC_AUTH_HASH` is a bcrypt hash, never a plaintext password.
      Generate one with:
      `docker run --rm caddy:2-alpine caddy hash-password --plaintext '...'`
- [ ] The hash is wrapped in **single quotes** in `.env`; otherwise Compose
      interpolates the `$` characters and the hash breaks.
- [ ] Rotate credentials if `.env` was ever copied into a chat, a screenshot,
      a log, or a backup that left the machine.
- [ ] No API keys, tokens, or private hostnames are hardcoded in source files.

### Network exposure

- [ ] Only one public entry point (Cloudflare Tunnel). No router port-forward.
- [ ] Caddy binds to `127.0.0.1` only (`APP_BIND=127.0.0.1`).
- [ ] `docker compose ps` shows no `0.0.0.0:` port mappings for `app`/`caddy`.
- [ ] Traefik routes to Caddy, and Caddy enforces basic auth before the app.
- [ ] `/health` answers `401` without credentials from the public hostname.

### TLS and headers

- [ ] Cloudflare SSL/TLS mode is **Full** (or Full Strict).
- [ ] Caddy sends `Cache-Control: private, no-store`, CSP, `X-Content-Type-Options`,
      `X-Frame-Options: DENY`, and hides the `Server` header.
- [ ] Cache Rule **Bypass** applies to the hostname so authenticated responses
      are never stored at the edge.

### Application

- [ ] `MOCK_PIPELINE=0` and `MOCK_OCR=0` in production.
- [ ] `CORS_ORIGINS` lists the exact public origin, not `*`.
- [ ] Upload limit (`MAX_UPLOAD_MB`, `CADDY_MAX_BODY`) is intentional; Caddy's
      limit must be greater than or equal to the app's.
- [ ] Containers run as a non-root user (uid `10001`).
- [ ] No Docker socket is mounted into `app` or `caddy`.

### Operations

- [ ] Backups exist and were restored at least once in a dry run.
- [ ] Logs do not contain plate numbers or credentials at `INFO` level.
- [ ] `nvidia-smi` shows only expected processes when the GPU overlay is used.

## Known limitations

- Basic auth is a single shared credential; there is no per-user audit trail.
  Put Cloudflare Access in front of the hostname if you need that.
- No rate limiting inside the app. Cloudflare provides basic protection, but a
  determined client with valid credentials can still queue many jobs.
- The API has no delete endpoint; removing a job requires direct database work.
- FastAPI's `/docs` and `/redoc` are enabled and protected only by basic auth.
