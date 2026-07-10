# Manual UI Checklist

Use this checklist to verify the web UI functionality before considering a release complete.

## Test Environment

- [ ] Browser: Chrome/Edge/Firefox latest
- [ ] Device: Desktop or mobile on same LAN as server
- [ ] Server running: `./scripts/start_all.sh` (defaults to LAN + HTTPS; firewall opened via `sudo ./scripts/open_firewall.sh`)

## Basic Functionality

### Page Load
- [ ] Page loads without errors in browser console
- [ ] All CSS renders correctly (dark theme, proper layout)
- [ ] No 404s in network tab for resources (index.html, app.js)

### Single Photo Upload
- [x] Drag-drop JPG works
- [ ] Drag-drop PNG works *(untested in UI — no PNG file at hand; PNG input verified at API level by smoke_test.sh test 7)*
- [ ] Click-to-choose file works
- [x] Original image displays in left panel
- [x] Depth map displays in right panel
- [x] Timing table populates with values
- [x] Variant dropdown shows f32/q8/q4 options
- [x] Depth render styles work (switching re-renders instantly)

### Render Styles
Test each style on a still photo:
- [x] Grayscale — depth as grayscale
- [x] Turbo — colorful gradient
- [x] Viridis — colorful gradient
- [x] Inferno — colorful gradient
- [x] Relief 3D — hillshaded 3D look

### High-Res Mode
- [ ] High-res checkbox available
- [ ] Checking it triggers slower inference (expected)
- [ ] Result shows higher density depth map

## Live Camera

### Camera Start
- [x] "Start camera" button visible and enabled
- [x] Camera permission prompt appears
- [x] Camera preview displays in left panel
- [x] Depth preview streams in right panel (real-time)
- [x] HUD shows fps, round-trip, server time, drop count

### Camera Control
- [ ] "Stop camera" button works
- [ ] "Flip camera" button appears when streaming
- [ ] Flip switches front/rear camera (mobile)
- [ ] HUD updates every second

### Streaming Quality
- [ ] FPS stable at ~20-25 on reference hardware
- [ ] Drop count stays low (< 10% of frames)
- [ ] No visual artifacts in depth stream
- [ ] Style changes apply to live stream instantly

## API-Only Mode

Start with: `./scripts/start_server.sh`

- [ ] `/` returns JSON: `{"service":"da3","ui":false,"endpoints":["/depth","/health"]}`
- [ ] `/health` works
- [ ] `/depth` works
- [ ] No UI assets served (404 for `/js/app.js`)

## TLS / LAN Mode

Start with: `./scripts/start_all.sh` (default) or `./scripts/start.sh --lan --daemon`

- [ ] Self-signed certificate generated in `resources/certs/`
- [ ] `https://<lan-ip>:8090` accessible
- [ ] Certificate warning accepted once
- [ ] Camera works on mobile after accepting cert
- [ ] HUD shows fps/drops correctly

## Error Handling

- [ ] Invalid file type (e.g., TXT) shows error message
- [ ] Empty request returns 400
- [ ] Server down shows connection error (not crash)

## Performance

- [ ] First request after start: ~44 ms (prewarmed)
- [ ] Subsequent requests: ~36 ms median
- [ ] Style changes: instant (no server round-trip)
- [ ] High-res mode: ~800 ms (expected slow)

## Notes

Record the following for each test run:
- Browser and version
- Device type (desktop/mobile)
- Server hardware
- Any anomalies or failures

## Test Runs

### 2026-07-10 — first user validation (release build)
- Device: Android phone over LAN, `https://192.168.1.31:8090` (self-signed cert, `start_all.sh` LAN+HTTPS default, firewall opened via `open_firewall.sh`)
- Server: RTX 5060 8 GB, CUDA 12.8, release commit d9623ef
- Verified: live camera video stream, JPG upload, variant dropdown (f32/q8/q4), all five render styles
- Not covered this run: PNG upload in UI (no PNG file available — API path covered by smoke test), click-to-choose, high-res mode, flip camera, error-handling rows
- Anomalies: none reported
