'use strict';
const $ = id => document.getElementById(id);
const status_ = $('status');
function setStatus(msg, isErr) { status_.textContent = msg; status_.className = isErr ? 'err' : ''; }

// ===== Client-side depth rendering styles ==================================
// The server returns a grayscale JPEG (bright = near). All styling happens
// here, per-pixel on a canvas -- no server round-trip to change styles.
function lerpLut(stops) {           // stops: [t, r, g, b][] -> Uint8Array[256*3]
  const lut = new Uint8Array(768);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let j = 0;
    while (j < stops.length - 2 && stops[j + 1][0] < t) j++;
    const [t0, r0, g0, b0] = stops[j], [t1, r1, g1, b1] = stops[j + 1];
    const f = Math.min(1, Math.max(0, (t - t0) / (t1 - t0 || 1)));
    lut[i * 3]     = r0 + (r1 - r0) * f;
    lut[i * 3 + 1] = g0 + (g1 - g0) * f;
    lut[i * 3 + 2] = b0 + (b1 - b0) * f;
  }
  return lut;
}
const LUTS = {
  turbo: lerpLut([[0,48,18,59],[.125,70,107,227],[.25,40,170,225],[.375,35,220,160],
                  [.5,120,240,90],[.625,210,220,50],[.75,250,160,30],[.875,230,80,20],[1,122,4,3]]),
  viridis: lerpLut([[0,68,1,84],[.25,59,82,139],[.5,33,145,140],[.75,94,201,98],[1,253,231,37]]),
  inferno: lerpLut([[0,0,0,4],[.25,87,16,110],[.5,188,55,84],[.75,249,142,9],[1,252,255,164]]),
};
const work = document.createElement('canvas');   // offscreen decode target

function renderDepth(bitmap, canvas) {
  const style = $('style').value;
  const w = bitmap.width, h = bitmap.height;
  work.width = w; work.height = h;
  const wctx = work.getContext('2d', { willReadFrequently: true });
  wctx.drawImage(bitmap, 0, 0);
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d');
  if (style === 'gray') { ctx.drawImage(bitmap, 0, 0); return; }
  const img = wctx.getImageData(0, 0, w, h);
  const d = img.data;
  if (style === 'relief') {
    // Hillshade from depth gradients + turbo tint: a lit-3D-surface look.
    const lut = LUTS.turbo;
    const lum = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) lum[i] = d[i * 4];
    // light dir (normalized ~[0.45, -0.45, 0.77])
    const lx = 0.45, ly = -0.45, lz = 0.77, gs = 2.0;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        const xm = lum[y * w + Math.max(0, x - 1)], xp = lum[y * w + Math.min(w - 1, x + 1)];
        const ym = lum[Math.max(0, y - 1) * w + x], yp = lum[Math.min(h - 1, y + 1) * w + x];
        const dx = (xp - xm) * gs / 255, dy = (yp - ym) * gs / 255;
        const inv = 1 / Math.sqrt(dx * dx + dy * dy + 1);
        const shade = Math.max(0, (-dx * lx + -dy * ly + lz) * inv);
        const v = lum[i] | 0, s = 0.35 + 0.65 * shade;
        d[i * 4]     = lut[v * 3] * s;
        d[i * 4 + 1] = lut[v * 3 + 1] * s;
        d[i * 4 + 2] = lut[v * 3 + 2] * s;
      }
    }
  } else {
    const lut = LUTS[style];
    for (let i = 0; i < w * h; i++) {
      const v = d[i * 4];
      d[i * 4] = lut[v * 3]; d[i * 4 + 1] = lut[v * 3 + 1]; d[i * 4 + 2] = lut[v * 3 + 2];
    }
  }
  ctx.putImageData(img, 0, 0);
}

// Re-render the last frame when the style changes (no new inference needed).
let lastStillBitmap = null, lastLiveBitmap = null;
$('style').addEventListener('change', () => {
  if (lastStillBitmap) renderDepth(lastStillBitmap, $('stillDepth'));
  if (lastLiveBitmap)  renderDepth(lastLiveBitmap, $('liveDepth'));
});

// ===== Shared inference call ================================================
async function inferBlob(blob, fullres) {
  const variant = $('variant').value;
  const res = fullres ? '&res=full' : '';
  const resp = await fetch(`/depth?variant=${variant}${res}`, {
    method: 'POST', headers: { 'Content-Type': blob.type || 'image/jpeg' }, body: blob,
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ error: resp.statusText }));
    throw new Error(err.error);
  }
  const ct = resp.headers.get('Content-Type') || '';
  if (ct.startsWith('image/jpeg')) {
    return {
      bitmap: await createImageBitmap(await resp.blob()),
      t: JSON.parse(resp.headers.get('X-Timings-Ms')),
      meta: { variant: resp.headers.get('X-Variant'), res: resp.headers.get('X-Res'),
              gpu: resp.headers.get('X-Gpu-Preprocess') === 'true',
              w: resp.headers.get('X-Depth-Width'), h: resp.headers.get('X-Depth-Height') },
      enc: 'GPU JPEG',
    };
  }
  const data = await resp.json();   // lossless/legacy fallback
  const png = await fetch('data:image/png;base64,' + data.depth_png).then(r => r.blob());
  return { bitmap: await createImageBitmap(png), t: data.timings_ms,
           meta: { variant: data.variant, res: data.res, gpu: data.gpu_preprocess,
                   w: data.width, h: data.height }, enc: 'CPU PNG' };
}

// ===== Single photo =========================================================
const drop = $('drop'), fileInput = $('file');
drop.addEventListener('click', () => fileInput.click());
drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('hover'); });
drop.addEventListener('dragleave', () => drop.classList.remove('hover'));
drop.addEventListener('drop', e => {
  e.preventDefault(); drop.classList.remove('hover');
  if (e.dataTransfer.files.length) runStill(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files.length) runStill(fileInput.files[0]); });

async function runStill(file) {
  if (!/^image\/(jpeg|png)$/.test(file.type)) { setStatus('Please use a jpg or png image.', true); return; }
  drop.classList.add('busy');
  setStatus(`Running inference on ${file.name}…`);
  $('orig').src = URL.createObjectURL(file);
  const t0 = performance.now();
  try {
    const r = await inferBlob(file, $('fullres').checked);
    lastStillBitmap = r.bitmap;
    renderDepth(r.bitmap, $('stillDepth'));
    const total = performance.now() - t0;
    $('t_save').textContent  = r.t.save_ms.toFixed(1);
    $('t_load').textContent  = r.t.model_load_ms > 0 ? r.t.model_load_ms.toFixed(1) : 'warm';
    $('t_pre').textContent   = r.meta.gpu ? r.t.preprocess_ms.toFixed(1) : 'CPU (in inference)';
    $('t_infer').textContent = r.t.infer_ms.toFixed(1);
    $('t_enc').textContent   = `${r.t.encode_ms.toFixed(1)} (${r.enc})`;
    $('t_srv').textContent   = r.t.server_ms.toFixed(1);
    $('t_total').textContent = total.toFixed(1);
    $('variant_used').textContent = `variant: ${r.meta.variant} · ${r.meta.res} res · ${r.meta.w}×${r.meta.h}`;
    $('results').classList.remove('hidden');
    setStatus('Done.');
  } catch (err) {
    setStatus('Error: ' + err.message, true);
  } finally {
    drop.classList.remove('busy');
  }
}

// ===== Live camera ==========================================================
// Frame policy: at most ONE request in flight; capture ticks that land while
// a request is pending are DROPPED (the next send always uses the freshest
// frame). Missed deadlines therefore never queue up -- latency stays bounded
// at one frame regardless of how slow the network/GPU is.
const cam = { on: false, stream: null, inFlight: false, facing: 'environment',
              frames: 0, dropped: 0, lastHud: 0, rtt: 0, srv: 0 };
const video = $('video');
const grab = document.createElement('canvas');

async function startCam() {
  try {
    cam.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: cam.facing, width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
  } catch (e) {
    const hint = location.protocol === 'http:' && location.hostname !== 'localhost' &&
                 location.hostname !== '127.0.0.1'
      ? ' (camera needs HTTPS on non-localhost: relaunch with DEPTH_UI_TLS=1 and use https://)' : '';
    setStatus('Camera error: ' + e.message + hint, true);
    return;
  }
  video.srcObject = cam.stream;
  cam.on = true;
  cam.frames = 0; cam.dropped = 0; cam.lastHud = performance.now();
  $('camBtn').textContent = 'Stop camera';
  $('flipBtn').classList.remove('hidden');
  setStatus('Streaming… frames encode to JPEG in-browser, depth streams back.');
  camTick();
}

function stopCam() {
  cam.on = false;
  if (cam.stream) { cam.stream.getTracks().forEach(t => t.stop()); cam.stream = null; }
  video.srcObject = null;
  $('camBtn').textContent = 'Start camera';
  $('flipBtn').classList.add('hidden');
  $('hud').textContent = '';
  setStatus('Camera stopped.');
}

$('camBtn').addEventListener('click', () => cam.on ? stopCam() : startCam());
$('flipBtn').addEventListener('click', async () => {
  cam.facing = cam.facing === 'environment' ? 'user' : 'environment';
  if (cam.on) { stopCam(); await startCam(); }
});

function camTick() {
  if (!cam.on) return;
  if (!cam.inFlight && video.videoWidth > 0) {
    cam.inFlight = true;
    grab.width = video.videoWidth; grab.height = video.videoHeight;
    grab.getContext('2d').drawImage(video, 0, 0);
    grab.toBlob(async blob => {
      const t0 = performance.now();
      try {
        const r = await inferBlob(blob, false);   // live mode: std res always
        lastLiveBitmap = r.bitmap;
        renderDepth(r.bitmap, $('liveDepth'));
        cam.frames++;
        cam.rtt = performance.now() - t0;
        cam.srv = r.t.server_ms;
      } catch (e) {
        setStatus('Stream error: ' + e.message, true);
      }
      cam.inFlight = false;
    }, 'image/jpeg', 0.85);
  } else if (cam.inFlight) {
    cam.dropped++;                                 // deadline missed -> drop
  }
  const now = performance.now();
  if (now - cam.lastHud >= 1000) {
    const fps = cam.frames / ((now - cam.lastHud) / 1000);
    $('hud').textContent =
      `${fps.toFixed(1)} fps · round-trip ${cam.rtt.toFixed(0)} ms · server ${Number(cam.srv).toFixed(0)} ms · dropped ${cam.dropped}`;
    cam.frames = 0; cam.dropped = 0; cam.lastHud = now;
  }
  setTimeout(camTick, 15);                         // ~66 Hz capture clock
}
