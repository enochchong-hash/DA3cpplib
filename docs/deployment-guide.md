# Depth Anything 3 Service — Deployment Guide

*Audience: IT administrators deploying and operating the service. No ML or build knowledge required. For build-from-source details see `setup-development-guide.md`; for API usage see `user-guide.md`.*

---

## 1. Service overview

| | |
|---|---|
| Service name | `depth-ui-server` |
| Purpose | Local image → depth-map web service (UI + REST API) |
| Executable | `<repo>/tools/depth_ui/build/depth-ui-server` |
| Start command | `<repo>/scripts/launch_depth_ui.sh [port]` |
| Default port | **8090** (TCP, HTTP) |
| Bind address | **127.0.0.1 only** (localhost); LAN + HTTPS is opt-in, see §7 |
| Runs as | Any unprivileged user with read access to the repo and write access to `/tmp` |
| External network access | None required at runtime (models are on local disk) |
| Data handling | Uploaded images are written to `/tmp/depth_ui_XXXXXX`, processed, and deleted within the same request. Nothing is stored. |

`<repo>` in this document = `/media/user/01DAEF8E3E5F5300/code/gemma4`.

## 2. Host requirements

- Ubuntu 22.04 (or compatible Linux), x86_64
- NVIDIA GPU with ≥ 2 GB free VRAM (reference deployment: RTX 5060 8 GB); NVIDIA driver installed and working (`nvidia-smi` succeeds)
- ~700 MB disk for the three model files, ~130 MB for binaries
- ~1 GB free RAM

The service is built for the host GPU's architecture. **Moving the built binary to a different machine or moving the repo to a different path is not supported** — paths are compiled in; rebuild on the target instead (one command, see §6).

## 3. Pre-flight checklist

```sh
nvidia-smi                                   # driver OK, GPU visible
ls <repo>/tools/depth_ui/build/depth-ui-server        # binary exists
ls <repo>/models/DepthAnything-Base-F32/ \
   <repo>/models/DepthAnything-Base-Q8/ \
   <repo>/models/DepthAnything-Base-Q4/               # model files exist
ss -ltn | grep 8090                          # port is free (no output = free)
```

Note: ports **8080–8087** on this host are used by the separate LLM server fleet. Do not assign them to this service.

## 4. Start / stop / verify

**Start (foreground):**
```sh
<repo>/scripts/launch_depth_ui.sh 8090
```

**Start (background, simple):**
```sh
nohup <repo>/scripts/launch_depth_ui.sh 8090 > /var/log/depth-ui.log 2>&1 &
```

**Verify:**
```sh
curl -s http://127.0.0.1:8090/health
# expected: {"ok":true,"variants":{"f32":{"downloaded":true,...},...}}
```

**Stop:**
```sh
<repo>/scripts/stop_depth_ui.sh 8090      # preferred (no arg = stop all instances)
pkill -x depth-ui-server                  # fallback
```
⚠️ Do **not** use `pkill -f depth-ui-server` from a script — the `-f` full-command-line match can kill the calling shell itself.

**Expected startup log:** lines showing `ggml_cuda_init: found 1 CUDA devices`, the three model paths, and `C++ server on http://127.0.0.1:8090`. Startup takes under a second; the first request per model variant additionally loads that model (up to ~0.4 s).

## 5. Run as a systemd service (recommended)

Create `/etc/systemd/system/depth-ui.service`:

```ini
[Unit]
Description=Depth Anything 3 depth-map service
After=network.target

[Service]
Type=simple
User=user
ExecStart=/media/user/01DAEF8E3E5F5300/code/gemma4/scripts/launch_depth_ui.sh 8090
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```sh
sudo systemctl daemon-reload
sudo systemctl enable --now depth-ui
systemctl status depth-ui
journalctl -u depth-ui -f        # live logs
```

## 6. Rebuild (after moving the repo, driver/CUDA change, or code update)

Requires the CUDA 12.8 toolkit at `/usr/local/cuda` (already installed on the reference host):

```sh
<repo>/scripts/build_depth.sh
```

Then restart the service.

## 7. Exposing beyond localhost (LAN + HTTPS)

The server binds `127.0.0.1` by default and has **no authentication or rate limiting**. For production-style exposure, put it behind a reverse proxy (nginx/caddy) that terminates TLS and enforces access control, proxying to `127.0.0.1:8090`. Allow request bodies up to 64 MB (`client_max_body_size 64m;` in nginx).

For **temporary LAN testing** (e.g. the phone-webcam feature, which requires HTTPS — browsers only grant camera access to secure origins) there is a built-in opt-in:

```sh
DEPTH_UI_HOST=0.0.0.0 DEPTH_UI_TLS=1 <repo>/scripts/launch_depth_ui.sh
```

This binds all interfaces and serves HTTPS using `tools/depth_ui/certs/{cert.pem,key.pem}`. This is trusted-LAN-only convenience — anyone on the network can use the service while it runs. Do not leave it running this way unattended, and do not port-forward it.

Getting LAN mode working requires two things beyond the launch command: opening the firewall (§7.1) and a certificate the client browser accepts (§7.2).

### 7.1 Firewall

The reference host runs **both `firewalld` and `ufw`** (verified active simultaneously), so an inbound port must be opened in **both** or LAN clients time out:

```sh
sudo firewall-cmd --add-port=8090/tcp --permanent && sudo firewall-cmd --reload
sudo ufw allow 8090/tcp
```

Verify from another device on the LAN: `curl -sk https://<this-pc-LAN-IP>:8090/health`. Note the firewall only affects other devices — from the host itself everything works even with the port closed, so "works locally, times out from the phone" is the firewall signature.

### 7.2 TLS certificate — two options

Browsers require the certificate's **subjectAltName (SAN)** to match the address being visited (the CN field is ignored), so any certificate must carry the host's LAN IP in its SAN.

**Option A — self-signed (zero setup, one warning per device).**
If `tools/depth_ui/certs/` is empty, `launch_depth_ui.sh` auto-generates a self-signed certificate with a SAN covering `localhost`, `127.0.0.1`, and the current LAN IP. Each browsing device shows `ERR_CERT_AUTHORITY_INVALID` once; click **Advanced → Proceed** (Chrome hides the link in some contexts — typing `thisisunsafe` on the warning page also bypasses it). Camera access works after the bypass.

**Option B — locally-trusted CA via mkcert (no warnings, recommended for repeated use).**

```sh
<repo>/scripts/setup_depth_ui_tls.sh
```

The script installs `mkcert` to `~/.local/bin` if needed (no root), creates a local CA on first run (`~/.local/share/mkcert/`), issues `cert.pem`/`key.pem` for `localhost`, `127.0.0.1`, and the current LAN IP (extra hostnames/IPs can be passed as arguments), and copies the CA's public certificate to `tools/depth_ui/certs/rootCA.pem`. Then import `rootCA.pem` **once per browsing device**:

| Device | Import steps |
|---|---|
| Chrome (Linux/desktop) | `chrome://certificate-manager` → Custom → Import `rootCA.pem` → check "Trust for identifying websites" |
| Firefox (desktop) | Settings → Privacy & Security → Certificates → View Certificates → Authorities → Import |
| Android | Copy the file to the phone → Settings → Security & privacy → More security → Install a certificate → CA certificate |
| iOS | Send the file to the phone, install the profile, then enable it under Settings → General → About → Certificate Trust Settings |

The CA **private key** stays in `~/.local/share/mkcert/` and must never be shared or committed — `rootCA.pem` (public half) is safe to distribute.

**Certificates are loaded at server start only** — restart the service after generating or replacing them. If the host's DHCP address changes, the certificate no longer matches: re-run `setup_depth_ui_tls.sh` (option B) or delete `tools/depth_ui/certs/cert.pem`/`key.pem` and relaunch (option A), then restart. A DHCP reservation on the router avoids this entirely.

## 8. Capacity & monitoring

- Processes **one image at a time** (requests queue). Warm throughput ≈ 3 requests/second for ~1 MP images on the reference GPU.
- GPU memory: ≤ ~1.5 GB with all three model variants loaded; check with `nvidia-smi`.
- Liveness probe: `GET /health` (returns HTTP 200 with JSON). Poll interval 30 s is fine.
- Logs go to stdout/stderr (journal if under systemd). One line per HTTP request plus one-time model-load lines.

## 9. Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `Binary not found ... build it first` | Not built on this host | Run `<repo>/scripts/build_depth.sh` |
| `failed to listen on port 8090` | Port in use | `ss -ltnp \| grep 8090`, stop the other process or pick another port |
| `/health` shows `"downloaded":false` for a variant | Model file missing | Re-download (commands in `setup-development-guide.md` §2.3) |
| Requests return 400 "model ... not downloaded" | Same as above | Same as above |
| Very slow responses (~10× slower) | Running on CPU — driver/CUDA problem | Check startup log for `ggml_cuda_init`; run `nvidia-smi`; reboot/reinstall driver if needed |
| First request after restart slower than the rest | Normal (one-time model load) | No action |
| 500 "inference failed" on one image | Corrupt/unsupported image file | Confirm the file is a valid JPG/PNG; try another image |
| Service dies immediately under systemd | Wrong `User=` or repo path in unit file | Check `journalctl -u depth-ui` |
| Works on the host but LAN devices time out | Port 8090 closed in firewalld and/or ufw (both run on this host) | Open it in **both** (§7.1) |
| Browser: `ERR_CERT_COMMON_NAME_INVALID` / "not valid for this address" | Certificate SAN doesn't include the address — cert predates the SAN fix, or the host's DHCP IP changed | Regenerate (§7.2) and restart the service |
| Browser: `ERR_CERT_AUTHORITY_INVALID` | Self-signed certificate (option A) — expected | Advanced → Proceed once per device, or switch to mkcert (§7.2 option B) |
| Warning persists after importing `rootCA.pem` | Server still serving the old self-signed cert, or import skipped the "identify websites" trust checkbox | Restart the service; re-import with the trust option checked |
| Phone camera not offered on the page | Page served over plain HTTP — browsers require a secure origin for getUserMedia | Use `DEPTH_UI_TLS=1` and open via `https://` (§7) |

## 10. Uninstall

```sh
sudo systemctl disable --now depth-ui && sudo rm /etc/systemd/system/depth-ui.service
# optionally reclaim disk:
rm -rf <repo>/tools/depth_ui/build <repo>/tools/depth-anything.cpp/build
rm -rf <repo>/models/DepthAnything-Base-{F32,Q8,Q4}
```
