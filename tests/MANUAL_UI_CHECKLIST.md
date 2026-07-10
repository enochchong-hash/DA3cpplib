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
- [ ] Drag-drop JPG works
- [ ] Drag-drop PNG works
- [ ] Click-to-choose file works
- [ ] Original image displays in left panel
- [ ] Depth map displays in right panel
- [ ] Timing table populates with values
- [ ] Variant dropdown shows f32/q8/q4 options
- [ ] Depth render styles work (switching re-renders instantly)

### Render Styles
Test each style on a still photo:
- [ ] Grayscale — depth as grayscale
- [ ] Turbo — colorful gradient
- [ ] Viridis — colorful gradient
- [ ] Inferno — colorful gradient
- [ ] Relief 3D — hillshaded 3D look

### High-Res Mode
- [ ] High-res checkbox available
- [ ] Checking it triggers slower inference (expected)
- [ ] Result shows higher density depth map

## Live Camera

### Camera Start
- [ ] "Start camera" button visible and enabled
- [ ] Camera permission prompt appears
- [ ] Camera preview displays in left panel
- [ ] Depth preview streams in right panel (real-time)
- [ ] HUD shows fps, round-trip, server time, drop count

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
