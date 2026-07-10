# Depth Anything 3 Service — User Guide

*Audience: end users of the web UI and developers integrating the depth REST API into their applications.*

The Depth Anything 3 (DA3) service turns an ordinary photo into a **depth map**: a grayscale image where **bright pixels are close to the camera and dark pixels are far away**. It runs entirely on local hardware — no image ever leaves the machine.

- **Web UI**: `http://127.0.0.1:8090`
- **REST API**: `POST http://127.0.0.1:8090/depth`
- **Health check**: `GET http://127.0.0.1:8090/health`

(The host/port may differ in your installation — ask your administrator. 8090 is the default.)

---

## 1. Using the web UI

### Live camera mode (turn any RGB camera into a 3D camera)

The UI's **Live camera** section streams your webcam through the depth model in real time (~20–25 fps): frames are captured and JPEG-encoded **in the browser**, sent to the server, and the depth stream renders side-by-side with the camera preview. If the model can't keep up, frames are simply **dropped** (never queued), so latency stays bounded at one frame. A HUD shows fps, round-trip and server times, and the drop count.

**Depth render styles** — all applied client-side (switching styles re-renders instantly, no new inference): Grayscale, Turbo, Viridis, Inferno, and **Relief 3D** (hillshaded surface look computed from depth gradients). Styles apply to still photos too.

**Testing from a phone** (or any other device on your LAN):

1. On the host PC: `DEPTH_UI_HOST=0.0.0.0 DEPTH_UI_TLS=1 scripts/launch_depth_ui.sh`
   (both flags are opt-in: this exposes the un-authenticated server to your LAN and generates a self-signed TLS certificate — HTTPS is *mandatory* because phone browsers only allow camera access on secure origins).
2. On the phone, open `https://<pc-lan-ip>:8090` and accept the certificate warning once.
3. Tap **Start camera** and grant camera permission. **Flip camera** switches front/rear.

Notes: the first camera frame pays a one-time ~200 ms graph warm-up if your camera's aspect ratio differs from 3:2; after that it streams steadily. Frame rate is bounded by the model (~28 fps) and your Wi-Fi round-trip.

1. Open `http://127.0.0.1:8090` in a browser.
2. (Optional) Pick a model variant from the dropdown — see [Choosing a variant](#3-choosing-a-model-variant).
3. Drag a **JPG or PNG** photo onto the drop zone, or click it to browse.
4. The original and the computed depth map appear side by side, together with a timing breakdown in milliseconds.

### Reading the timing table

| Row | Meaning |
|---|---|
| save upload | Time to receive and store your image on the server |
| model load (first use) | One-time cost of loading the model into GPU memory. Shows `warm` on every request after the first — the model stays resident |
| inference | The actual depth-estimation computation |
| encode result | Converting the raw depth values into the PNG you see |
| server total | Everything the server did, end to end |
| upload → result (browser) | Full round-trip measured by your browser, including network transfer |

Typical warm-request numbers on the reference hardware (RTX 5060): **~63 ms** server total at standard resolution (504 px, the default); **~800 ms** with the optional high-res mode.

---

## 2. Using the REST API

### `POST /depth?variant=<f32|q8|q4>&res=<std|full>&format=<binary|json>`

Send the **raw image bytes** as the request body (not multipart/form-data).

| | |
|---|---|
| Query param `variant` | `f32` (default), `q8`, or `q4` |
| Query param `res` | `std` (default): production resolution, longest side 504 px, ~52 ms warm. `full`: near-input resolution, ~4× the pixels, ~10× slower |
| Query param `format` | default: **binary JPEG** response (fastest, ~10 KB). `json`: legacy lossless PNG/base64 JSON |
| `Content-Type` header | `image/jpeg` or `image/png` |
| Body | Raw JPG/PNG bytes, max 64 MB |

**Default success response** (`200`, `Content-Type: image/jpeg`): the body is the depth map itself — a grayscale JPEG (quality 90), GPU-encoded, normalized per image so bright = near, dark = far. Metadata rides in response headers:

```
X-Variant: q8
X-Res: std
X-Gpu-Preprocess: true
X-Depth-Width: 504
X-Depth-Height: 336
X-Timings-Ms: {"save_ms":0.1,"model_load_ms":0.0,"preprocess_ms":2.2,"infer_ms":49.9,"encode_ms":0.3,"server_ms":52.5}
```

**`?format=json` response** (`200`, `application/json`) — for callers who need lossless pixel values (JPEG is visually identical but lossy):

```json
{
  "variant": "q8", "res": "std", "gpu_preprocess": true,
  "width": 504, "height": 336,
  "timings_ms": { "save_ms": 0.1, "model_load_ms": 0.0, "preprocess_ms": 2.3,
                  "infer_ms": 49.2, "encode_ms": 4.6, "server_ms": 56.3 },
  "depth_png": "<base64-encoded grayscale PNG>"
}
```

Notes:
- `width`/`height` are the model's processed resolution, which may differ from your input resolution; scale to fit if you need pixel correspondence with the original.
- `model_load_ms` is greater than zero only on the first request per variant since the server started.
- A client should key on the response `Content-Type`: if the GPU encoder is unavailable the server transparently falls back to the JSON/PNG form.

**Error response** (`4xx`/`5xx`):

```json
{ "error": "human-readable description" }
```

| Status | Cause |
|---|---|
| 400 | Unknown variant, model file not installed, empty body, or body too large |
| 500 | Inference or internal failure (details in `error`) |

### `GET /health`

```json
{"ok":true,"variants":{
  "f32":{"downloaded":true,"loaded":true},
  "q8":{"downloaded":true,"loaded":false},
  "q4":{"downloaded":true,"loaded":false}}}
```

`downloaded` = model file present on disk; `loaded` = already resident in GPU memory (a request for a non-loaded variant simply pays the one-time load cost).

### Examples

**curl** — the response body IS the depth map:

```sh
curl -s -X POST --data-binary @photo.jpg -H "Content-Type: image/jpeg" \
  "http://127.0.0.1:8090/depth?variant=q8" -o depth.jpg -D headers.txt
grep X-Timings-Ms headers.txt
```

**JavaScript (browser or Node 18+)**:

```js
const resp = await fetch("http://127.0.0.1:8090/depth?variant=q8", {
  method: "POST",
  headers: { "Content-Type": "image/jpeg" },
  body: imageBlob,               // a File/Blob or Buffer of jpg/png bytes
});
const timings = JSON.parse(resp.headers.get("X-Timings-Ms"));
imgElement.src = URL.createObjectURL(await resp.blob());
console.log("inference took", timings.infer_ms, "ms");
```

**Python**:

```python
import json, requests

with open("photo.jpg", "rb") as f:
    r = requests.post("http://127.0.0.1:8090/depth?variant=q4",
                      data=f.read(), headers={"Content-Type": "image/jpeg"})
r.raise_for_status()
open("depth.jpg", "wb").write(r.content)           # body = grayscale JPEG
print(json.loads(r.headers["X-Timings-Ms"]))
# lossless variant: add params={"format": "json"} and decode depth_png (base64 PNG)
```

---

## 3. Choosing a model variant

All three are the same Depth Anything 3 *base* network at different numeric precision:

| Variant | File size | First-request load | Warm inference | When to use |
|---|---|---|---|---|
| `f32` | 412 MB | ~380 ms | ~315 ms | Reference quality; the default |
| `q8` | 149 MB | ~60 ms | ~300 ms | Near-lossless; faster startup, smaller footprint |
| `q4` | 104 MB | ~45 ms | ~300 ms | Smallest; quality visually indistinguishable in testing |

On GPU, warm inference speed is nearly identical across variants — quantization mainly buys smaller files, faster load, and less GPU memory. There is **no 6-bit variant** of this model upstream.

---

## 4. Good to know

- **Sequential processing**: the server processes one image at a time; concurrent requests queue briefly. For a ~1 MP image budget roughly 3 requests/second sustained.
- **Depth is relative, not metric**: values are normalized per image. "Brighter than" reliably means "closer than" within one image, but pixel value 200 in two different images does not mean the same physical distance.
- **Privacy**: uploads are written to a temporary file for the duration of one request and deleted immediately after processing. Nothing is retained.
- **Formats**: JPG and PNG only.

### CLI alternative

If you are on the host machine, you can skip HTTP entirely:

```sh
scripts/run_depth.sh photo.jpg depth.png f32     # variants: f32 | q8 | q4
```
