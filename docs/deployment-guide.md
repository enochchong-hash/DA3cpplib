# Depth Anything 3 Service — Deployment Guide

*Audience: IT administrators deploying and operating the service. No ML or build knowledge required. For build-from-source details see `setup-development-guide.md`; for API usage see `user-guide.md`.*

---

## 1. Service overview

| | |
|---|---|
| Service name | `depth-ui-server` |
| Purpose | Local image → depth-map web service (UI + REST API) |
| Executable | `<repo>/build/depth-ui-server` |
| Start command | `<repo>/scripts/start_all.sh` (or `scripts/start.sh` for options) |
| Default port | **8090** (TCP, HTTP) |
| Bind address | **127.0.0.1 only** (localhost); LAN + HTTPS is opt-in, see §7 |
| Runs as | Any unprivileged user with read access to the repo and write access to `/tmp` and `<repo>/var` |
| External network access | None required at runtime (models are on local disk) |
| Data handling | Uploaded images are written to `/tmp/depth_ui_XXXXXX`, processed, and deleted within the same request. Nothing is stored. |

`<repo>` in this document = wherever this release is checked out. The tree is
**relocatable**: the binary resolves models and web assets relative to its own
location (env vars `DA3_MODEL_DIR`/`DA3_WWW_DIR` override), so the folder can
be moved or copied without rebuilding — as long as the GPU generation matches
(§2).

## 2. Host requirements

- Ubuntu 22.04 (or compatible Linux), x86_64
- NVIDIA GPU with ≥ 2 GB free VRAM (reference deployment: RTX 5060 8 GB); NVIDIA driver installed and working (`nvidia-smi` succeeds)
- ~700 MB disk for the three model files, ~150 MB for binaries
- ~1 GB free RAM

The binary is compiled for the build host's GPU architecture (`--arch native`
by default). Moving the folder to a machine with a **different GPU generation**
requires a rebuild there: `scripts/build.sh` (toolchain prerequisites in
`setup-development-guide.md`).

## 3. Pre-flight checklist

```sh
nvidia-smi                                   # driver OK, GPU visible
ls <repo>/build/depth-ui-server              # binary exists
ls <repo>/resources/nnmodels/DepthAnything-Base-F32/ \
   <repo>/resources/nnmodels/DepthAnything-Base-Q8/ \
   <repo>/resources/nnmodels/DepthAnything-Base-Q4/  # model files exist
ss -ltn | grep 8090                          # port is free (no output = free)
```

Anything missing is installed by `scripts/setup.sh` (submodules, patches,
models, build — idempotent) or piecewise by `scripts/download_models.sh` and
`scripts/build.sh`.

## 4. Start / stop / verify

**Start (background, recommended):**
```sh
<repo>/scripts/start_all.sh
# idempotent: calling it again reports "already running" and exits 0
```

**Start (foreground, e.g. for a terminal session or systemd):**
```sh
<repo>/scripts/start.sh --port 8090
```

**Backend only (REST API, no web UI):**
```sh
<repo>/scripts/start_server.sh --daemon
```

**Verify:**
```sh
curl -s http://127.0.0.1:8090/health
# expected: {"ok":true,"variants":{"f32":{"downloaded":true,...},...}}
<repo>/scripts/start.sh --status
```

**Stop:**
```sh
<repo>/scripts/stop_all.sh                   # all instances (pidfiles + safe pkill sweep)
<repo>/scripts/start.sh --port 8090 --stop   # just this port
```
⚠️ Do **not** use `pkill -f depth-ui-server` — the `-f` full-command-line match
can kill the calling shell itself. The scripts use pidfiles and `pkill -x`.

**Logs:** daemon instances write `<repo>/var/log/da3-<port>.log`; foreground
instances log to stdout/stderr (journal under systemd).

**Expected startup log:** lines showing `ggml_cuda_init: found 1 CUDA devices`, the three model paths, and the listen address. With prewarm (default) all variants load at startup (~1 s) and even the first request is warm; with `--no-prewarm` the first request per variant pays the load cost (up to ~0.4 s).

## 5. Run as a systemd service (recommended)

A template ships at `scripts/da3.service.in`. Instantiate and install:

```sh
sed -e "s|@USER@|$(whoami)|g" -e "s|@REL_ROOT@|<repo>|g" \
    <repo>/scripts/da3.service.in | sudo tee /etc/systemd/system/da3.service
sudo systemctl daemon-reload
sudo systemctl enable --now da3
systemctl status da3
journalctl -u da3 -f              # live logs
```

The template runs `start.sh` in **foreground** mode (`Type=simple`). Do not
add `--daemon` to the unit's ExecStart — systemd would see the fork parent
exit and restart-loop the service.

## 6. Rebuild (after a GPU/driver/CUDA change or code update)

Requires a CUDA 12.8+ toolkit; the scripts auto-detect it at `/usr/local/cuda*`
even when an older `nvcc` shadows it on PATH. `scripts/setup_cuda.sh` installs
the toolkit if missing (and `--remove-old` purges Ubuntu's obsolete CUDA 11.5
apt packages).

```sh
<repo>/scripts/build.sh
```

Then restart the service.

## 7. Exposing beyond localhost (LAN + HTTPS)

The server binds `127.0.0.1` by default and has **no authentication or rate limiting**. For production-style exposure, put it behind a reverse proxy (nginx/caddy) that terminates TLS and enforces access control, proxying to `127.0.0.1:8090`. Allow request bodies up to 64 MB (`client_max_body_size 64m;` in nginx).

For **temporary LAN testing** (e.g. the phone-webcam feature, which requires HTTPS — browsers only grant camera access to secure origins) there is a built-in opt-in:

```sh
<repo>/scripts/start.sh --lan        # = --host 0.0.0.0 --tls
```

This binds all interfaces and serves HTTPS using `resources/certs/{server.crt,server.key}` (a self-signed certificate is auto-generated on first use). This is trusted-LAN-only convenience — anyone on the network can use the service while it runs. Do not leave it running this way unattended, and do not port-forward it.

Getting LAN mode working requires two things beyond the launch command: opening the firewall (§7.1) and a certificate the client browser accepts (§7.2).

### 7.1 Firewall

Some hosts run **both `firewalld` and `ufw`** (the original reference host did — verified active simultaneously), in which case an inbound port must be opened in **both** or LAN clients time out:

```sh
sudo firewall-cmd --add-port=8090/tcp --permanent && sudo firewall-cmd --reload
sudo ufw allow 8090/tcp
```

Verify from another device on the LAN: `curl -sk https://<this-pc-LAN-IP>:8090/health`. Note the firewall only affects other devices — from the host itself everything works even with the port closed, so "works locally, times out from the phone" is the firewall signature.

### 7.2 TLS certificate — two options

Browsers require the certificate's **subjectAltName (SAN)** to match the address being visited (the CN field is ignored), so any certificate must carry the host's LAN IP in its SAN.

**Option A — self-signed (zero setup, one warning per device).**
If `resources/certs/` is empty, `start.sh --tls` auto-generates a self-signed certificate with a SAN covering `localhost`, `127.0.0.1`, and the current LAN IP. Each browsing device shows `ERR_CERT_AUTHORITY_INVALID` once; click **Advanced → Proceed** (Chrome hides the link in some contexts — typing `thisisunsafe` on the warning page also bypasses it). Camera access works after the bypass.

**Option B — locally-trusted CA via mkcert (no warnings, recommended for repeated use).**

```sh
<repo>/scripts/setup_tls.sh
```

The script installs `mkcert` to `~/.local/bin` if needed (no root), creates a local CA on first run (`~/.local/share/mkcert/`), issues `server.crt`/`server.key` for `localhost`, `127.0.0.1`, and the current LAN IP (extra hostnames/IPs can be passed as arguments), and copies the CA's public certificate to `resources/certs/rootCA.pem`. Then import `rootCA.pem` **once per browsing device**:

| Device | Import steps |
|---|---|
| Chrome (Linux/desktop) | `chrome://certificate-manager` → Custom → Import `rootCA.pem` → check "Trust for identifying websites" |
| Firefox (desktop) | Settings → Privacy & Security → Certificates → View Certificates → Authorities → Import |
| Android | Copy the file to the phone → Settings → Security & privacy → More security → Install a certificate → CA certificate |
| iOS | Send the file to the phone, install the profile, then enable it under Settings → General → About → Certificate Trust Settings |

The CA **private key** stays in `~/.local/share/mkcert/` and must never be shared or committed — `rootCA.pem` (public half) is safe to distribute.

**Certificates are loaded at server start only** — restart the service after generating or replacing them. If the host's DHCP address changes, the certificate no longer matches: re-run `setup_tls.sh` (option B) or delete `resources/certs/server.crt`/`server.key` and relaunch (option A), then restart. A DHCP reservation on the router avoids this entirely.

## 8. Capacity & monitoring

- Processes **one image at a time** (requests queue). Warm throughput ≈ 29 fps sequential (~34 ms/request) on the reference GPU at standard resolution — see `perf-baselines.md`.
- GPU memory: ≤ ~1.5 GB with all three model variants loaded; check with `nvidia-smi`.
- Liveness probe: `GET /health` (returns HTTP 200 with JSON). Poll interval 30 s is fine.
- One log line per HTTP request plus one-time model-load lines.

## 9. Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `Binary not found ... build it first` | Not built on this host | Run `<repo>/scripts/build.sh` (or `scripts/setup.sh` for everything) |
| `failed to listen on port 8090` | Port in use | `ss -ltnp \| grep 8090`, stop the other process or use `--port` |
| `/health` shows `"downloaded":false` for a variant | Model file missing | Run `<repo>/scripts/download_models.sh` |
| Requests return 400 "model ... not downloaded" | Same as above | Same as above |
| Very slow responses (~10× slower) | Running on CPU — driver/CUDA problem | Check startup log for `ggml_cuda_init`; run `nvidia-smi`; reboot/reinstall driver if needed |
| Responses ~1.5× slower than the baseline table | Custom CUDA kernels not active | The daemon environment must have `DA_CONV=cuda_fused` (start.sh's default); verify with `scripts/bench.sh` |
| First request after restart slower than the rest | Prewarm disabled (`--no-prewarm`) | Expected; one-time model load |
| 500 "inference failed" on one image | Corrupt/unsupported image file | Confirm the file is a valid JPG/PNG; try another image |
| Service dies immediately under systemd | Wrong `User=`/paths in unit, or `--daemon` in ExecStart | Check `journalctl -u da3`; ExecStart must run foreground |
| Works on the host but LAN devices time out | Port closed in firewalld and/or ufw | Open it in **both** (§7.1) |
| Browser: `ERR_CERT_COMMON_NAME_INVALID` / "not valid for this address" | Certificate SAN doesn't include the address — cert predates the SAN, or the host's DHCP IP changed | Regenerate (§7.2) and restart the service |
| Browser: `ERR_CERT_AUTHORITY_INVALID` | Self-signed certificate (option A) — expected | Advanced → Proceed once per device, or switch to mkcert (§7.2 option B) |
| Warning persists after importing `rootCA.pem` | Server still serving the old self-signed cert, or import skipped the "identify websites" trust checkbox | Restart the service; re-import with the trust option checked |
| Phone camera not offered on the page | Page served over plain HTTP — browsers require a secure origin for getUserMedia | Use `--lan` (or `--tls`) and open via `https://` (§7) |

## 10. Uninstall

```sh
sudo systemctl disable --now da3 && sudo rm /etc/systemd/system/da3.service
# optionally reclaim disk:
rm -rf <repo>/build
rm -rf <repo>/resources/nnmodels/DepthAnything-Base-{F32,Q8,Q4}
```
