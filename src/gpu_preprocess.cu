// See gpu_preprocess.h. Replicates preprocess.cpp's preprocess_real pipeline on
// the GPU: two-step resize policy (longest side -> target, then round each dim
// to a patch multiple), cv2-faithful INTER_AREA (fractional box filter) and
// INTER_CUBIC (a=-0.75, border replicate), uint8 saturation (round-half-even,
// matching sat_u8/cvRound) BETWEEN steps -- the CPU path materializes a uint8
// image after each resize, so we quantize identically -- then ImageNet
// normalize into CHW f32. Both resizes accumulate in f32 with a single
// saturate at the end (2D direct == the CPU's separable-float version, since
// no quantization happens between the two passes there either).
#include "gpu_preprocess.h"

#include <cuda_runtime.h>
#include <nvjpeg.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

#include <cmath>
#include <cstdio>
#include <cstring>

namespace {

#define CU_OK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "[gpupre] %s failed: %s\n", #x, cudaGetErrorString(e_)); return -1; } } while (0)
#define NVJ_OK(x) do { nvjpegStatus_t s_ = (x); if (s_ != NVJPEG_STATUS_SUCCESS) { \
    fprintf(stderr, "[gpupre] %s failed: %d\n", #x, (int)s_); return -1; } } while (0)

__device__ __forceinline__ uint8_t sat_u8_dev(float v) {
    int r = __float2int_rn(v);              // round-to-nearest-even == cvRound
    return (uint8_t)(r < 0 ? 0 : (r > 255 ? 255 : r));
}

__device__ __forceinline__ float cubic_w_dev(float x) {  // cv2 INTER_CUBIC, a=-0.75
    const float a = -0.75f;
    x = fabsf(x);
    if (x < 1.f) return ((a + 2.f) * x - (a + 3.f)) * x * x + 1.f;
    if (x < 2.f) return (((x - 5.f) * x + 8.f) * x - 4.f) * a;
    return 0.f;
}

// cv2 INTER_AREA tap weights for one destination index (computeResizeAreaTab).
// Writes up to `cap` (si, alpha) pairs; returns the tap count.
__device__ int area_taps(int d, int ssize, double scale, int* si, float* alpha, int cap) {
    double fsx1 = d * scale, fsx2 = fsx1 + scale;
    double cellWidth = fmin(scale, (double)ssize - fsx1);
    int sx1 = (int)ceil(fsx1), sx2 = (int)floor(fsx2);
    if (sx2 > ssize - 1) sx2 = ssize - 1;
    if (sx1 > sx2) sx1 = sx2;
    int n = 0;
    if (sx1 - fsx1 > 1e-3 && n < cap) { si[n] = sx1 - 1; alpha[n] = (float)((sx1 - fsx1) / cellWidth); ++n; }
    for (int sx = sx1; sx < sx2 && n < cap; ++sx) { si[n] = sx; alpha[n] = (float)(1.0 / cellWidth); ++n; }
    if (fsx2 - sx2 > 1e-3 && n < cap) {
        si[n] = sx2;
        alpha[n] = (float)(fmin(fmin(fsx2 - sx2, 1.0), cellWidth) / cellWidth);
        ++n;
    }
    return n;
}

// Max source taps per output pixel per axis. The DA3 policy downscales by
// scale = longest/504; MAXTAP=34 covers inputs up to ~16k px on the long side.
constexpr int MAXTAP = 34;

__global__ void k_resize_area(const uint8_t* src, int sw, int sh,
                              uint8_t* dst, int dw, int dh) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    int xi[MAXTAP], yi[MAXTAP];
    float xa[MAXTAP], ya[MAXTAP];
    const int nx = area_taps(x, sw, (double)sw / dw, xi, xa, MAXTAP);
    const int ny = area_taps(y, sh, (double)sh / dh, yi, ya, MAXTAP);
    float acc[3] = {0.f, 0.f, 0.f};
    for (int j = 0; j < ny; ++j) {
        const uint8_t* row = src + ((size_t)yi[j] * sw) * 3;
        float rx[3] = {0.f, 0.f, 0.f};
        for (int i = 0; i < nx; ++i) {
            const uint8_t* p = row + (size_t)xi[i] * 3;
            rx[0] += xa[i] * p[0]; rx[1] += xa[i] * p[1]; rx[2] += xa[i] * p[2];
        }
        acc[0] += ya[j] * rx[0]; acc[1] += ya[j] * rx[1]; acc[2] += ya[j] * rx[2];
    }
    uint8_t* q = dst + ((size_t)y * dw + x) * 3;
    q[0] = sat_u8_dev(acc[0]); q[1] = sat_u8_dev(acc[1]); q[2] = sat_u8_dev(acc[2]);
}

__global__ void k_resize_cubic(const uint8_t* src, int sw, int sh,
                               uint8_t* dst, int dw, int dh) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    const double sx = (double)sw / dw, sy = (double)sh / dh;
    const double fx = (x + 0.5) * sx - 0.5, fy = (y + 0.5) * sy - 0.5;
    const int ix = (int)floor(fx), iy = (int)floor(fy);
    const float tx = (float)(fx - ix), ty = (float)(fy - iy);
    float wx[4], wy[4]; int xs[4], ys[4];
    for (int k = 0; k < 4; ++k) {
        wx[k] = cubic_w_dev(tx - (k - 1));
        wy[k] = cubic_w_dev(ty - (k - 1));
        int s = ix - 1 + k; xs[k] = s < 0 ? 0 : (s >= sw ? sw - 1 : s);
        s = iy - 1 + k;     ys[k] = s < 0 ? 0 : (s >= sh ? sh - 1 : s);
    }
    float acc[3] = {0.f, 0.f, 0.f};
    for (int j = 0; j < 4; ++j) {
        const uint8_t* row = src + ((size_t)ys[j] * sw) * 3;
        float rx[3] = {0.f, 0.f, 0.f};
        for (int i = 0; i < 4; ++i) {
            const uint8_t* p = row + (size_t)xs[i] * 3;
            rx[0] += wx[i] * p[0]; rx[1] += wx[i] * p[1]; rx[2] += wx[i] * p[2];
        }
        acc[0] += wy[j] * rx[0]; acc[1] += wy[j] * rx[1]; acc[2] += wy[j] * rx[2];
    }
    uint8_t* q = dst + ((size_t)y * dw + x) * 3;
    q[0] = sat_u8_dev(acc[0]); q[1] = sat_u8_dev(acc[1]); q[2] = sat_u8_dev(acc[2]);
}

// uint8 HWC -> normalized f32 CHW: chw[c][y][x] = (u8/255 - mean[c]) / std[c].
__global__ void k_normalize_chw(const uint8_t* rgb, int w, int h,
                                float m0, float m1, float m2,
                                float s0, float s1, float s2, float* chw) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const uint8_t* p = rgb + ((size_t)y * w + x) * 3;
    const size_t hw = (size_t)h * w, at = (size_t)y * w + x;
    chw[at]          = ((float)p[0] / 255.f - m0) / s0;
    chw[hw + at]     = ((float)p[1] / 255.f - m1) / s1;
    chw[2 * hw + at] = ((float)p[2] / 255.f - m2) / s2;
}

// Depth floats -> grayscale u8, flipped so near = bright (the model emits
// distance: larger = farther). lo/hi are the depth min/max for this image.
__global__ void k_depth_to_gray(const float* d, uint8_t* g, int n, float hi, float scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    g[i] = sat_u8_dev((hi - d[i]) * scale);
}

// Host-side dim math, identical to preprocess.cpp.
int py_round(double x) { return (int)std::nearbyint(x); }  // ties to even
int nearest_multiple(int x, int p) {
    int down = (x / p) * p, up = down + p;
    return (std::abs(up - x) <= std::abs(x - down)) ? up : down;
}

struct DevBuf {
    void*  ptr = nullptr;
    size_t cap = 0;
    int ensure(size_t need) {
        if (need <= cap) return 0;
        if (ptr) cudaFree(ptr);
        ptr = nullptr; cap = 0;
        CU_OK(cudaMalloc(&ptr, need));
        cap = need;
        return 0;
    }
    void release() { if (ptr) cudaFree(ptr); ptr = nullptr; cap = 0; }
};

}  // namespace

struct GpuPre {
    nvjpegHandle_t    nvj   = nullptr;
    nvjpegJpegState_t state = nullptr;
    cudaStream_t      stream = nullptr;
    DevBuf rgb_in;    // decoded / uploaded RGB u8 HWC
    DevBuf rgb_a;     // after step-1 resize
    DevBuf rgb_b;     // after step-2 resize
    DevBuf chw;       // final normalized f32 CHW
    size_t chw_bytes = 0;
    // Encoder side (depth map -> grayscale JPEG).
    nvjpegEncoderState_t  enc_state  = nullptr;
    nvjpegEncoderParams_t enc_params = nullptr;
    int enc_quality = 0;  // quality the params are currently set to
    DevBuf depth_f32;     // uploaded depth floats
    DevBuf gray_u8;       // normalized grayscale
};

GpuPre* gpupre_init() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) return nullptr;
    GpuPre* g = new GpuPre();
    if (cudaStreamCreate(&g->stream) != cudaSuccess ||
        nvjpegCreateSimple(&g->nvj) != NVJPEG_STATUS_SUCCESS ||
        nvjpegJpegStateCreate(g->nvj, &g->state) != NVJPEG_STATUS_SUCCESS) {
        gpupre_free(g);
        return nullptr;
    }
    return g;
}

void gpupre_free(GpuPre* g) {
    if (!g) return;
    if (g->enc_params) nvjpegEncoderParamsDestroy(g->enc_params);
    if (g->enc_state)  nvjpegEncoderStateDestroy(g->enc_state);
    if (g->state) nvjpegJpegStateDestroy(g->state);
    if (g->nvj)   nvjpegDestroy(g->nvj);
    g->rgb_in.release(); g->rgb_a.release(); g->rgb_b.release(); g->chw.release();
    g->depth_f32.release(); g->gray_u8.release();
    if (g->stream) cudaStreamDestroy(g->stream);
    delete g;
}

// Shared tail: (device RGB u8 HWC, w, h) -> two-step resize -> normalize.
static int run_pipeline(GpuPre* g, const uint8_t* dev_rgb, int w, int h,
                        int target, int patch, const float mean[3], const float stdv[3],
                        int* outW, int* outH) {
    if (target <= 0) target = 504;
    if (patch  <= 0) patch  = 14;
    const dim3 tb(16, 16);
    const uint8_t* cur = dev_rgb;
    int cw = w, ch = h;

    // Step 1: longest side -> target (upper_bound policy).
    {
        const int bound = cw > ch ? cw : ch;
        if (bound != target) {
            const double scale = (double)target / bound;
            int nw = py_round(cw * scale); if (nw < 1) nw = 1;
            int nh = py_round(ch * scale); if (nh < 1) nh = 1;
            if ((double)cw / nw > (double)MAXTAP - 2 || (double)ch / nh > (double)MAXTAP - 2) return -1;
            if (g->rgb_a.ensure((size_t)nw * nh * 3)) return -1;
            const dim3 gr((nw + 15) / 16, (nh + 15) / 16);
            if (scale > 1.0) k_resize_cubic<<<gr, tb, 0, g->stream>>>(cur, cw, ch, (uint8_t*)g->rgb_a.ptr, nw, nh);
            else             k_resize_area <<<gr, tb, 0, g->stream>>>(cur, cw, ch, (uint8_t*)g->rgb_a.ptr, nw, nh);
            cur = (uint8_t*)g->rgb_a.ptr; cw = nw; ch = nh;
        }
    }
    // Step 2: round each dim to the nearest patch multiple.
    {
        int nw = nearest_multiple(cw, patch); if (nw < 1) nw = patch;
        int nh = nearest_multiple(ch, patch); if (nh < 1) nh = patch;
        if (nw != cw || nh != ch) {
            const bool upscale = (nw > cw) || (nh > ch);
            if (g->rgb_b.ensure((size_t)nw * nh * 3)) return -1;
            const dim3 gr((nw + 15) / 16, (nh + 15) / 16);
            if (upscale) k_resize_cubic<<<gr, tb, 0, g->stream>>>(cur, cw, ch, (uint8_t*)g->rgb_b.ptr, nw, nh);
            else         k_resize_area <<<gr, tb, 0, g->stream>>>(cur, cw, ch, (uint8_t*)g->rgb_b.ptr, nw, nh);
            cur = (uint8_t*)g->rgb_b.ptr; cw = nw; ch = nh;
        }
    }
    // Normalize -> CHW f32.
    g->chw_bytes = (size_t)3 * cw * ch * sizeof(float);
    if (g->chw.ensure(g->chw_bytes)) return -1;
    {
        const dim3 gr((cw + 15) / 16, (ch + 15) / 16);
        k_normalize_chw<<<gr, tb, 0, g->stream>>>(cur, cw, ch,
            mean[0], mean[1], mean[2], stdv[0], stdv[1], stdv[2], (float*)g->chw.ptr);
    }
    CU_OK(cudaGetLastError());
    CU_OK(cudaStreamSynchronize(g->stream));
    *outW = cw; *outH = ch;
    return 0;
}

int gpupre_jpeg(GpuPre* g, const uint8_t* data, size_t n, int target, int patch,
                const float mean[3], const float stdv[3], int* outW, int* outH) {
    if (!g || !data || !n) return -1;
    int comps = 0, widths[NVJPEG_MAX_COMPONENT] = {0}, heights[NVJPEG_MAX_COMPONENT] = {0};
    nvjpegChromaSubsampling_t sub;
    NVJ_OK(nvjpegGetImageInfo(g->nvj, data, n, &comps, &sub, widths, heights));
    const int w = widths[0], h = heights[0];
    if (w <= 0 || h <= 0) return -1;
    if (g->rgb_in.ensure((size_t)w * h * 3)) return -1;
    nvjpegImage_t out{};
    out.channel[0] = (unsigned char*)g->rgb_in.ptr;
    out.pitch[0]   = (unsigned int)(w * 3);
    NVJ_OK(nvjpegDecode(g->nvj, g->state, data, n, NVJPEG_OUTPUT_RGBI, &out, g->stream));
    return run_pipeline(g, (const uint8_t*)g->rgb_in.ptr, w, h, target, patch, mean, stdv, outW, outH);
}

int gpupre_rgb8(GpuPre* g, const uint8_t* rgb, int w, int h, int target, int patch,
                const float mean[3], const float stdv[3], int* outW, int* outH) {
    if (!g || !rgb || w <= 0 || h <= 0) return -1;
    const size_t nb = (size_t)w * h * 3;
    if (g->rgb_in.ensure(nb)) return -1;
    CU_OK(cudaMemcpyAsync(g->rgb_in.ptr, rgb, nb, cudaMemcpyHostToDevice, g->stream));
    return run_pipeline(g, (const uint8_t*)g->rgb_in.ptr, w, h, target, patch, mean, stdv, outW, outH);
}

int gpupre_copy_into(GpuPre* g, void* dst, size_t nbytes, int dst_is_device) {
    if (!g || !dst || nbytes != g->chw_bytes || !g->chw.ptr) return -1;
    CU_OK(cudaMemcpy(dst, g->chw.ptr, nbytes,
                     dst_is_device ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost));
    return 0;
}

int gpuenc_depth_jpeg(GpuPre* g, const float* depth, int w, int h, int quality,
                      std::vector<unsigned char>& jpeg_out) {
    if (!g || !depth || w <= 0 || h <= 0) return -1;
    if (quality < 1) quality = 1;
    if (quality > 100) quality = 100;
    // Lazy encoder init (first call only).
    if (!g->enc_state) {
        NVJ_OK(nvjpegEncoderStateCreate(g->nvj, &g->enc_state, g->stream));
        NVJ_OK(nvjpegEncoderParamsCreate(g->nvj, &g->enc_params, g->stream));
        NVJ_OK(nvjpegEncoderParamsSetSamplingFactors(g->enc_params, NVJPEG_CSS_GRAY, g->stream));
    }
    if (quality != g->enc_quality) {
        NVJ_OK(nvjpegEncoderParamsSetQuality(g->enc_params, quality, g->stream));
        g->enc_quality = quality;
    }
    const size_t n = (size_t)w * h;
    if (g->depth_f32.ensure(n * sizeof(float))) return -1;
    if (g->gray_u8.ensure(n)) return -1;
    CU_OK(cudaMemcpyAsync(g->depth_f32.ptr, depth, n * sizeof(float),
                          cudaMemcpyHostToDevice, g->stream));
    // Min/max on device, then normalize+flip to u8 (near = bright).
    CU_OK(cudaStreamSynchronize(g->stream));  // thrust default stream safety
    thrust::device_ptr<const float> dp((const float*)g->depth_f32.ptr);
    auto mm = thrust::minmax_element(thrust::device, dp, dp + n);
    const float lo = *mm.first, hi = *mm.second;
    const float scale = hi > lo ? 255.0f / (hi - lo) : 0.0f;
    k_depth_to_gray<<<(int)((n + 255) / 256), 256, 0, g->stream>>>(
        (const float*)g->depth_f32.ptr, (uint8_t*)g->gray_u8.ptr, (int)n, hi, scale);
    CU_OK(cudaGetLastError());
    nvjpegImage_t src{};
    src.channel[0] = (unsigned char*)g->gray_u8.ptr;
    src.pitch[0]   = (unsigned int)w;
    NVJ_OK(nvjpegEncodeYUV(g->nvj, g->enc_state, g->enc_params, &src,
                           NVJPEG_CSS_GRAY, w, h, g->stream));
    size_t len = 0;
    NVJ_OK(nvjpegEncodeRetrieveBitstream(g->nvj, g->enc_state, nullptr, &len, g->stream));
    jpeg_out.resize(len);
    CU_OK(cudaStreamSynchronize(g->stream));
    NVJ_OK(nvjpegEncodeRetrieveBitstream(g->nvj, g->enc_state, jpeg_out.data(), &len, g->stream));
    jpeg_out.resize(len);
    return 0;
}
