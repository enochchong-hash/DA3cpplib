// See cuda_head_kernels.hpp. Implicit-GEMM 3x3 conv with fused pre-ReLU /
// bias / residual epilogue, tiled BM=64 x BN=64 x BK=8, 256 threads computing
// 4x4 micro-tiles -- the classic SGEMM shape with the B-matrix load replaced
// by a virtual-im2col gather (so the 9x im2col tensor never exists).
#include "cuda_head_kernels.hpp"

#include <cuda_runtime.h>
#include <mma.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <tuple>

namespace {

// Must match `ggml_cuda_custom_launch` in the patched ggml-cuda.cu exactly.
struct CudaLaunchDesc {
    uint64_t magic;  // 0x3344414355444121ULL
    void (*launch)(ggml_tensor* dst, uintptr_t cuda_stream, void* user);
    void* user;      // flags: bit0 prerelu, bit1 has_bias, bit2 has_residual
};
constexpr uint64_t kMagic = 0x3344414355444121ULL;

constexpr int BM = 64, BN = 64, BK = 8, TT = 256;  // 16x16 threads, 4x4 tiles

// Tile sizes templated: BM_/TM_ so small-OC convs (e.g. out2a, OC=32) use a
// BM=32 tile instead of running half-empty; BN_/TN_ so tiny-spatial convs
// (layer4_rn at 18x12: P=216 -> only 8 blocks at BN=64) use narrow pixel tiles
// that put enough blocks in flight to fill the SMs. 16*TM_=BM_, 16*TN_=BN_.
template <int BM_, int TM_, int BN_, int TN_>
__global__ void k_conv3x3_igemm(const float* __restrict__ X, const float* __restrict__ Wt,
                                const float* __restrict__ B, const float* __restrict__ R,
                                float* __restrict__ Y,
                                int W, int H, int IC, int OC, int flags) {
    const int P = W * H;
    const int K = IC * 9;
    __shared__ float As[BK][BM_];
    __shared__ float Bs[BK][BN_];
    const int m0 = blockIdx.y * BM_;  // output-channel block
    const int n0 = blockIdx.x * BN_;  // pixel block
    const int tid = threadIdx.x;
    const int tm = tid / 16, tn = tid % 16;
    float acc[TM_][TN_] = {};

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Stage A tile: weights are a ready-made row-major [OC][IC*9] matrix.
        for (int i = tid; i < BM_ * BK; i += TT) {
            const int mm = i / BK, kk = i % BK;
            const int m = m0 + mm, k = k0 + kk;
            As[kk][mm] = (m < OC && k < K) ? Wt[(size_t)m * K + k] : 0.f;
        }
        // Stage B tile: virtual im2col gather (zero pad, optional pre-ReLU).
        for (int i = tid; i < BN_ * BK; i += TT) {
            const int nn = i / BK, kk = i % BK;
            const int k = k0 + kk, p = n0 + nn;
            float v = 0.f;
            if (k < K && p < P) {
                const int ic = k / 9, t = k % 9;
                const int iy = p / W + t / 3 - 1;
                const int ix = p % W + t % 3 - 1;
                if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                    v = X[(size_t)ic * P + (size_t)iy * W + ix];
                    if (flags & 1) v = fmaxf(v, 0.f);
                }
            }
            Bs[kk][nn] = v;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a[TM_], b[TN_];
            #pragma unroll
            for (int u = 0; u < TM_; ++u) a[u] = As[kk][tm * TM_ + u];
            #pragma unroll
            for (int v = 0; v < TN_; ++v) b[v] = Bs[kk][tn * TN_ + v];
            #pragma unroll
            for (int u = 0; u < TM_; ++u)
                #pragma unroll
                for (int v = 0; v < TN_; ++v) acc[u][v] += a[u] * b[v];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int u = 0; u < TM_; ++u) {
        const int m = m0 + tm * TM_ + u;
        if (m >= OC) continue;
        #pragma unroll
        for (int v = 0; v < TN_; ++v) {
            const int p = n0 + tn * TN_ + v;
            if (p >= P) continue;
            float c = acc[u][v];
            if (flags & 2) c += B[m];
            if (flags & 4) c += R[(size_t)m * P + p];
            Y[(size_t)m * P + p] = c;
        }
    }
}

// 1x1 stride-1 pad-0 conv = plain GEMM C[OC][P] = W[OC][IC] * X[IC][P], with
// X read in place -- ggml's path materializes an im2col copy of the whole
// input first (e.g. 21MB for out2b at 504x336). B loads are direct and
// coalesced; no halo, no tap math.
template <int BM_, int TM_, int BN_, int TN_>
__global__ void k_conv1x1_igemm(const float* __restrict__ X, const float* __restrict__ Wt,
                                const float* __restrict__ B,
                                float* __restrict__ Y, int P, int IC, int OC, int flags) {
    __shared__ float As[BK][BM_];
    __shared__ float Bs[BK][BN_];
    const int m0 = blockIdx.y * BM_;
    const int n0 = blockIdx.x * BN_;
    const int tid = threadIdx.x;
    const int tm = tid / 16, tn = tid % 16;
    float acc[TM_][TN_] = {};

    for (int k0 = 0; k0 < IC; k0 += BK) {
        for (int i = tid; i < BM_ * BK; i += TT) {
            const int mm = i / BK, kk = i % BK;
            const int m = m0 + mm, k = k0 + kk;
            As[kk][mm] = (m < OC && k < IC) ? Wt[(size_t)m * IC + k] : 0.f;
        }
        for (int i = tid; i < BN_ * BK; i += TT) {
            const int nn = i / BK, kk = i % BK;
            const int k = k0 + kk, p = n0 + nn;
            float v = 0.f;
            if (k < IC && p < P) {
                v = X[(size_t)k * P + p];
                if (flags & 1) v = fmaxf(v, 0.f);
            }
            Bs[kk][nn] = v;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a[TM_], b[TN_];
            #pragma unroll
            for (int u = 0; u < TM_; ++u) a[u] = As[kk][tm * TM_ + u];
            #pragma unroll
            for (int v = 0; v < TN_; ++v) b[v] = Bs[kk][tn * TN_ + v];
            #pragma unroll
            for (int u = 0; u < TM_; ++u)
                #pragma unroll
                for (int v = 0; v < TN_; ++v) acc[u][v] += a[u] * b[v];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int u = 0; u < TM_; ++u) {
        const int m = m0 + tm * TM_ + u;
        if (m >= OC) continue;
        #pragma unroll
        for (int v = 0; v < TN_; ++v) {
            const int p = n0 + tn * TN_ + v;
            if (p >= P) continue;
            float c = acc[u][v];
            if (flags & 2) c += B[m];
            Y[(size_t)m * P + p] = c;
        }
    }
}

// Degenerate-OC 1x1 conv (e.g. out2b: 32 -> 2 at 504x336). A GEMM tile would
// idle 15/16 of its rows; instead each thread owns one pixel and accumulates
// all OC outputs, weights staged once in shared memory.
__global__ void k_conv1x1_tinyoc(const float* __restrict__ X, const float* __restrict__ Wt,
                                 const float* __restrict__ B,
                                 float* __restrict__ Y, int P, int IC, int OC, int flags) {
    extern __shared__ float Wsh[];  // [OC * IC]
    for (int i = threadIdx.x; i < OC * IC; i += blockDim.x) Wsh[i] = Wt[i];
    __syncthreads();
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    float acc[4] = {};  // OC <= 4
    for (int ic = 0; ic < IC; ++ic) {
        float v = X[(size_t)ic * P + p];
        if (flags & 1) v = fmaxf(v, 0.f);
        for (int oc = 0; oc < OC; ++oc) acc[oc] += Wsh[oc * IC + ic] * v;
    }
    for (int oc = 0; oc < OC; ++oc) {
        float c = acc[oc];
        if (flags & 2) c += B[oc];
        Y[(size_t)oc * P + p] = c;
    }
}

// TF32 tensor-core variant of the conv3x3 implicit GEMM (DA_TF32=1). Same
// BM=64/BN=64/BK=8 block tile; 8 warps in a 4(M)x2(N) grid, each computing a
// 16x32 output subtile via two m16n16k8 wmma fragments. TF32 truncates the
// mantissa to 10 bits (accumulation stays F32) -- opt-in because it trades a
// small numeric drift for ~2x GEMM throughput on Ampere+ tensor cores.
__global__ void k_conv3x3_wmma(const float* __restrict__ X, const float* __restrict__ Wt,
                               const float* __restrict__ B, const float* __restrict__ R,
                               float* __restrict__ Y,
                               int W, int H, int IC, int OC, int flags) {
    using namespace nvcuda;
    const int P = W * H;
    const int K = IC * 9;
    __shared__ float AsT[BM][BK];       // row-major M x K tile (ld = BK)
    __shared__ float Bs[BK][BN];        // row-major K x N tile (ld = BN)
    __shared__ float Cs[8][16 * 32];    // per-warp epilogue staging
    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;
    const int tid  = threadIdx.x;
    const int wid  = tid / 32, lane = tid % 32;
    const int warp_m = wid / 2, warp_n = wid % 2;   // 4 x 2 warps

    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> fc[2];
    wmma::fill_fragment(fc[0], 0.f);
    wmma::fill_fragment(fc[1], 0.f);

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int i = tid; i < BM * BK; i += TT) {
            const int mm = i / BK, kk = i % BK;
            const int m = m0 + mm, k = k0 + kk;
            AsT[mm][kk] = (m < OC && k < K) ? Wt[(size_t)m * K + k] : 0.f;
        }
        for (int i = tid; i < BN * BK; i += TT) {
            const int nn = i / BK, kk = i % BK;
            const int k = k0 + kk, p = n0 + nn;
            float v = 0.f;
            if (k < K && p < P) {
                const int ic = k / 9, t = k % 9;
                const int iy = p / W + t / 3 - 1;
                const int ix = p % W + t % 3 - 1;
                if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                    v = X[(size_t)ic * P + (size_t)iy * W + ix];
                    if (flags & 1) v = fmaxf(v, 0.f);
                }
            }
            Bs[kk][nn] = v;
        }
        __syncthreads();
        wmma::load_matrix_sync(fa, &AsT[warp_m * 16][0], BK);
        #pragma unroll
        for (int i = 0; i < fa.num_elements; ++i) fa.x[i] = wmma::__float_to_tf32(fa.x[i]);
        #pragma unroll
        for (int f = 0; f < 2; ++f) {
            wmma::load_matrix_sync(fb, &Bs[0][warp_n * 32 + f * 16], BN);
            #pragma unroll
            for (int i = 0; i < fb.num_elements; ++i) fb.x[i] = wmma::__float_to_tf32(fb.x[i]);
            wmma::mma_sync(fc[f], fa, fb, fc[f]);
        }
        __syncthreads();
    }

    wmma::store_matrix_sync(&Cs[wid][0],  fc[0], 32, wmma::mem_row_major);
    wmma::store_matrix_sync(&Cs[wid][16], fc[1], 32, wmma::mem_row_major);
    __syncwarp();
    for (int e = lane; e < 16 * 32; e += 32) {
        const int row = e / 32, col = e % 32;
        const int m = m0 + warp_m * 16 + row;
        const int p = n0 + warp_n * 32 + col;
        if (m >= OC || p >= P) continue;
        float c = Cs[wid][e];
        if (flags & 2) c += B[m];
        if (flags & 4) c += R[(size_t)m * P + p];
        Y[(size_t)m * P + p] = c;
    }
}

// Fused align-corners-bilinear-upsample -> conv (K2 in the plan): the conv's
// input gather samples the SMALL source map with the same bilerp ggml's
// interpolate(ALIGN_CORNERS) would produce -- the upsampled intermediate never
// exists. KS = kernel size (1: the fusion out-convs; 3: out2a). Optional PE
// tensor (the head's 0.1*UV pos-embed at target resolution) is added after the
// lerp, matching `interp(x) + pe` exactly.
template <int BM_, int TM_, int KS>
__global__ void k_convup_igemm(const float* __restrict__ Xs, const float* __restrict__ Wt,
                               const float* __restrict__ B, const float* __restrict__ PE,
                               float* __restrict__ Y,
                               int W, int H, int Ws, int Hs, float sx, float sy,
                               int IC, int OC, int flags) {
    const int P = W * H;
    const size_t Ps = (size_t)Ws * Hs;
    const int K = IC * KS * KS;
    __shared__ float As[BK][BM_];
    __shared__ float Bs[BK][BN];
    const int m0 = blockIdx.y * BM_;
    const int n0 = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int tm = tid / 16, tn = tid % 16;
    float acc[TM_][4] = {};

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int i = tid; i < BM_ * BK; i += TT) {
            const int mm = i / BK, kk = i % BK;
            const int m = m0 + mm, k = k0 + kk;
            As[kk][mm] = (m < OC && k < K) ? Wt[(size_t)m * K + k] : 0.f;
        }
        for (int i = tid; i < BN * BK; i += TT) {
            const int nn = i / BK, kk = i % BK;
            const int k = k0 + kk, p = n0 + nn;
            float v = 0.f;
            if (k < K && p < P) {
                const int ic = k / (KS * KS), t = k % (KS * KS);
                const int qy = p / W + t / KS - KS / 2;
                const int qx = p % W + t % KS - KS / 2;
                if (qy >= 0 && qy < H && qx >= 0 && qx < W) {
                    const float fy = qy * sy, fx = qx * sx;   // align-corners map
                    int y0 = (int)fy, x0 = (int)fx;
                    if (y0 > Hs - 2) y0 = Hs - 2 < 0 ? 0 : Hs - 2;
                    if (x0 > Ws - 2) x0 = Ws - 2 < 0 ? 0 : Ws - 2;
                    const float dy = fy - y0, dx = fx - x0;
                    const int y1 = y0 + 1 < Hs ? y0 + 1 : Hs - 1;
                    const int x1 = x0 + 1 < Ws ? x0 + 1 : Ws - 1;
                    const float* pl = Xs + ic * Ps;
                    const float v00 = pl[(size_t)y0 * Ws + x0], v01 = pl[(size_t)y0 * Ws + x1];
                    const float v10 = pl[(size_t)y1 * Ws + x0], v11 = pl[(size_t)y1 * Ws + x1];
                    v = (v00 * (1.f - dx) + v01 * dx) * (1.f - dy)
                      + (v10 * (1.f - dx) + v11 * dx) * dy;
                    if (flags & 64) v += PE[(size_t)ic * P + (size_t)qy * W + qx];
                }
            }
            Bs[kk][nn] = v;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a[TM_], b[4];
            #pragma unroll
            for (int u = 0; u < TM_; ++u) a[u] = As[kk][tm * TM_ + u];
            #pragma unroll
            for (int v = 0; v < 4; ++v) b[v] = Bs[kk][tn * 4 + v];
            #pragma unroll
            for (int u = 0; u < TM_; ++u)
                #pragma unroll
                for (int v = 0; v < 4; ++v) acc[u][v] += a[u] * b[v];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int u = 0; u < TM_; ++u) {
        const int m = m0 + tm * TM_ + u;
        if (m >= OC) continue;
        #pragma unroll
        for (int v = 0; v < 4; ++v) {
            const int p = n0 + tn * 4 + v;
            if (p >= P) continue;
            float c = acc[u][v];
            if (flags & 2) c += B[m];
            Y[(size_t)m * P + p] = c;
        }
    }
}

// Transposed conv with stride == kernel size (the DPT reassemble resizes:
// 4x4 s4 and 2x2 s2). Each output pixel then receives EXACTLY one input tap:
// out[oc, iy*s+ky, ix*s+kx] = sum_ic W[kx,ky,oc,ic] * X[ic, iy, ix]  (+bias)
// i.e. a plain GEMM C[M=OC*s*s][N=P_in] = A[M][IC] * B[IC][P_in] with a
// pixel-shuffle epilogue -- no stride/mod divergence, unlike ggml's naive
// conv2d-transpose kernel (one thread per output looping all IC*K*K taps).
// Weight layout [KW,KH,OC,IC]: A[m][ic] = Wdata[ic*(KW*KH*OC) + m] with
// m = (oc*KH + ky)*KW + kx (kx fastest -- matches m enumeration exactly).
__global__ void k_convT_igemm(const float* __restrict__ X, const float* __restrict__ Wt,
                              const float* __restrict__ B,
                              float* __restrict__ Y,
                              int Win, int Hin, int IC, int OC, int S, int flags) {
    const int Pin = Win * Hin;
    const int M = OC * S * S;
    const int Wout = Win * S;
    const size_t Pout = (size_t)Wout * (Hin * S);
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];
    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int tm = tid / 16, tn = tid % 16;
    float acc[4][4] = {};

    for (int k0 = 0; k0 < IC; k0 += BK) {
        for (int i = tid; i < BM * BK; i += TT) {
            const int mm = i / BK, kk = i % BK;
            const int m = m0 + mm, k = k0 + kk;
            As[kk][mm] = (m < M && k < IC) ? Wt[(size_t)k * M + m] : 0.f;
        }
        for (int i = tid; i < BN * BK; i += TT) {
            const int nn = i / BK, kk = i % BK;
            const int k = k0 + kk, p = n0 + nn;
            Bs[kk][nn] = (k < IC && p < Pin) ? X[(size_t)k * Pin + p] : 0.f;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a[4], b[4];
            #pragma unroll
            for (int u = 0; u < 4; ++u) a[u] = As[kk][tm * 4 + u];
            #pragma unroll
            for (int v = 0; v < 4; ++v) b[v] = Bs[kk][tn * 4 + v];
            #pragma unroll
            for (int u = 0; u < 4; ++u)
                #pragma unroll
                for (int v = 0; v < 4; ++v) acc[u][v] += a[u] * b[v];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        const int m = m0 + tm * 4 + u;
        if (m >= M) continue;
        const int kx = m % S, ky = (m / S) % S, oc = m / (S * S);
        #pragma unroll
        for (int v = 0; v < 4; ++v) {
            const int p = n0 + tn * 4 + v;
            if (p >= Pin) continue;
            const int ix = p % Win, iy = p / Win;
            float c = acc[u][v];
            if (flags & 2) c += B[oc];
            Y[(size_t)oc * Pout + (size_t)(iy * S + ky) * Wout + (ix * S + kx)] = c;
        }
    }
}

// DA_CUDA_HEAD_STATS=1: per-shape kernel timing via CUDA events (synchronous;
// diagnostics only), dumped at process exit.
struct ShapeStat { int count = 0; double ms = 0.0; };
std::map<std::tuple<int,int,int,int,int>, ShapeStat> g_stats;
bool stats_enabled() {
    static const bool on = std::getenv("DA_CUDA_HEAD_STATS") != nullptr;
    return on;
}
void dump_stats() {
    double tot = 0.0;
    for (const auto& [k, v] : g_stats) {
        const auto& [W, H, IC, OC, flags] = k;
        fprintf(stderr, "[cuda_head] %4dx%-4d IC=%-4d OC=%-4d flags=%d  n=%-3d total=%7.2fms  avg=%.3fms\n",
                W, H, IC, OC, flags, v.count, v.ms, v.ms / v.count);
        tot += v.ms;
    }
    fprintf(stderr, "[cuda_head] TOTAL fused-conv kernel time: %.2fms\n", tot);
}

void cuda_launch(ggml_tensor* dst, uintptr_t stream, void* user) {
    const int flags = (int)(uintptr_t)user;
    const ggml_tensor* x = dst->src[0];
    const ggml_tensor* w = dst->src[1];
    int si = 2;
    const ggml_tensor* b = (flags & 2) ? dst->src[si++] : nullptr;
    const ggml_tensor* r = (flags & 4) ? dst->src[si++] : nullptr;
    const int W = (int)x->ne[0], H = (int)x->ne[1];
    const int IC = (int)w->ne[2], OC = (int)w->ne[3];
    const int P = W * H;
    const bool small_oc = OC <= 32;
    // Tiny-spatial shapes (layer4_rn at 18x12 = 216 px): BN=64 tiles yield only
    // 8 blocks on a 36-SM GPU; narrow BM=32/BN=16 tiles put 4x more blocks in
    // flight (measured 1.00 -> 0.56ms). Threshold deliberately excludes 36x24
    // (864 px): there the extra per-block A-tile reloads made it SLOWER
    // (0.50 -> 0.56ms) -- the redundant-load trade-off the GEMM-rewrite plan's
    // split-K phase would solve properly.
    const bool small_p = P < 256;
    const int bm = (small_oc || small_p) ? 32 : BM;
    const int bn = small_p ? 16 : BN;
    const dim3 grid((P + bn - 1) / bn, (OC + bm - 1) / bm);

    cudaEvent_t e0 = nullptr, e1 = nullptr;
    if (stats_enabled()) {
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0, (cudaStream_t)stream);
    }
    static const bool tf32 = std::getenv("DA_TF32") != nullptr;
    const float* Xp = (const float*)x->data;
    const float* Wp = (const float*)w->data;
    const float* Bp = b ? (const float*)b->data : nullptr;
    const float* Rp = r ? (const float*)r->data : nullptr;
    float* Yp = (float*)dst->data;
    if (small_p) {
        k_conv3x3_igemm<32, 2, 16, 1><<<grid, TT, 0, (cudaStream_t)stream>>>(
            Xp, Wp, Bp, Rp, Yp, W, H, IC, OC, flags);
    } else if (small_oc) {
        k_conv3x3_igemm<32, 2, BN, 4><<<grid, TT, 0, (cudaStream_t)stream>>>(
            Xp, Wp, Bp, Rp, Yp, W, H, IC, OC, flags);
    } else if (tf32) {
        k_conv3x3_wmma<<<grid, TT, 0, (cudaStream_t)stream>>>(
            Xp, Wp, Bp, Rp, Yp, W, H, IC, OC, flags);
    } else {
        k_conv3x3_igemm<BM, 4, BN, 4><<<grid, TT, 0, (cudaStream_t)stream>>>(
            Xp, Wp, Bp, Rp, Yp, W, H, IC, OC, flags);
    }
    if (stats_enabled()) {
        cudaEventRecord(e1, (cudaStream_t)stream);
        cudaEventSynchronize(e1);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, e0, e1);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        static const bool reg = [] { atexit(dump_stats); return true; }();
        (void)reg;
        auto& st = g_stats[{W, H, IC, OC, flags}];
        st.count += 1; st.ms += ms;
    }
}

// Launch for the 1x1 conv. flags: bit0 prerelu, bit1 bias, bit7 marks 1x1.
void cuda_launch_conv1x1(ggml_tensor* dst, uintptr_t stream, void* user) {
    const int flags = (int)(uintptr_t)user;
    const ggml_tensor* x = dst->src[0];
    const ggml_tensor* w = dst->src[1];
    const ggml_tensor* b = (flags & 2) ? dst->src[2] : nullptr;
    const int W = (int)x->ne[0], H = (int)x->ne[1];
    const int IC = (int)w->ne[2], OC = (int)w->ne[3];
    const int P = W * H;

    cudaEvent_t e0 = nullptr, e1 = nullptr;
    if (stats_enabled()) { cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0, (cudaStream_t)stream); }
    const float* Xp = (const float*)x->data;
    const float* Wp = (const float*)w->data;
    const float* Bp = b ? (const float*)b->data : nullptr;
    float* Yp = (float*)dst->data;
    if (OC <= 4) {
        const int threads = 256;
        k_conv1x1_tinyoc<<<(P + threads - 1) / threads, threads,
                           OC * IC * sizeof(float), (cudaStream_t)stream>>>(
            Xp, Wp, Bp, Yp, P, IC, OC, flags);
    } else if (OC <= 32) {
        const dim3 grid((P + BN - 1) / BN, (OC + 31) / 32);
        k_conv1x1_igemm<32, 2, BN, 4><<<grid, TT, 0, (cudaStream_t)stream>>>(
            Xp, Wp, Bp, Yp, P, IC, OC, flags);
    } else {
        const dim3 grid((P + BN - 1) / BN, (OC + BM - 1) / BM);
        k_conv1x1_igemm<BM, 4, BN, 4><<<grid, TT, 0, (cudaStream_t)stream>>>(
            Xp, Wp, Bp, Yp, P, IC, OC, flags);
    }
    if (stats_enabled()) {
        cudaEventRecord(e1, (cudaStream_t)stream);
        cudaEventSynchronize(e1);
        float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        static const bool reg = [] { atexit(dump_stats); return true; }();
        (void)reg;
        auto& st = g_stats[{W, H, IC, OC, flags}];
        st.count += 1; st.ms += ms;
    }
}

// Launch for the fused upsample->conv. flags: bit1 bias, bit4 KS=3, bit5 KS=1,
// bit6 PE present. Source dims from x; output dims from dst.
void cuda_launch_convup(ggml_tensor* dst, uintptr_t stream, void* user) {
    const int flags = (int)(uintptr_t)user;
    const ggml_tensor* x = dst->src[0];
    const ggml_tensor* w = dst->src[1];
    int si = 2;
    const ggml_tensor* b  = (flags & 2)  ? dst->src[si++] : nullptr;
    const ggml_tensor* pe = (flags & 64) ? dst->src[si++] : nullptr;
    const int W = (int)dst->ne[0], H = (int)dst->ne[1];
    const int Ws = (int)x->ne[0], Hs = (int)x->ne[1];
    const int IC = (int)w->ne[2], OC = (int)w->ne[3];
    const int P = W * H;
    const float sx = W > 1 ? (float)(Ws - 1) / (float)(W - 1) : 0.f;
    const float sy = H > 1 ? (float)(Hs - 1) / (float)(H - 1) : 0.f;
    const bool small_oc = OC <= 32;
    const dim3 grid((P + BN - 1) / BN, (OC + (small_oc ? 32 : BM) - 1) / (small_oc ? 32 : BM));

    cudaEvent_t e0 = nullptr, e1 = nullptr;
    if (stats_enabled()) { cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0, (cudaStream_t)stream); }
    const float* Xp = (const float*)x->data;
    const float* Wp = (const float*)w->data;
    const float* Bp = b ? (const float*)b->data : nullptr;
    const float* Pp = pe ? (const float*)pe->data : nullptr;
    float* Yp = (float*)dst->data;
    if (flags & 16) {          // KS = 3
        if (small_oc) k_convup_igemm<32, 2, 3><<<grid, TT, 0, (cudaStream_t)stream>>>(Xp, Wp, Bp, Pp, Yp, W, H, Ws, Hs, sx, sy, IC, OC, flags);
        else          k_convup_igemm<64, 4, 3><<<grid, TT, 0, (cudaStream_t)stream>>>(Xp, Wp, Bp, Pp, Yp, W, H, Ws, Hs, sx, sy, IC, OC, flags);
    } else {                   // KS = 1
        if (small_oc) k_convup_igemm<32, 2, 1><<<grid, TT, 0, (cudaStream_t)stream>>>(Xp, Wp, Bp, Pp, Yp, W, H, Ws, Hs, sx, sy, IC, OC, flags);
        else          k_convup_igemm<64, 4, 1><<<grid, TT, 0, (cudaStream_t)stream>>>(Xp, Wp, Bp, Pp, Yp, W, H, Ws, Hs, sx, sy, IC, OC, flags);
    }
    if (stats_enabled()) {
        cudaEventRecord(e1, (cudaStream_t)stream);
        cudaEventSynchronize(e1);
        float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        static const bool reg = [] { atexit(dump_stats); return true; }();
        (void)reg;
        auto& st = g_stats[{W, H, IC, OC, flags}];
        st.count += 1; st.ms += ms;
    }
}

// Launch for the transposed conv. flags bit1 = bias; bit3 marks convT (stats
// disambiguation only). Stride recovered from the weight tensor (KW == S).
void cuda_launch_convT(ggml_tensor* dst, uintptr_t stream, void* user) {
    const int flags = (int)(uintptr_t)user;
    const ggml_tensor* x = dst->src[0];
    const ggml_tensor* w = dst->src[1];
    const ggml_tensor* b = (flags & 2) ? dst->src[2] : nullptr;
    const int Win = (int)x->ne[0], Hin = (int)x->ne[1];
    const int S = (int)w->ne[0];
    const int OC = (int)w->ne[2], IC = (int)w->ne[3];
    const int Pin = Win * Hin, M = OC * S * S;
    const dim3 grid((Pin + BN - 1) / BN, (M + BM - 1) / BM);

    cudaEvent_t e0 = nullptr, e1 = nullptr;
    if (stats_enabled()) { cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0, (cudaStream_t)stream); }
    k_convT_igemm<<<grid, TT, 0, (cudaStream_t)stream>>>(
        (const float*)x->data, (const float*)w->data,
        b ? (const float*)b->data : nullptr,
        (float*)dst->data, Win, Hin, IC, OC, S, flags);
    if (stats_enabled()) {
        cudaEventRecord(e1, (cudaStream_t)stream);
        cudaEventSynchronize(e1);
        float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        static const bool reg = [] { atexit(dump_stats); return true; }();
        (void)reg;
        auto& st = g_stats[{Win, Hin, IC, OC, flags}];
        st.count += 1; st.ms += ms;
    }
}

// CPU fallback (only runs if the scheduler ever places the node on the CPU
// backend, which supports_op should prevent). Correct but naive.
void cpu_fallback(ggml_tensor* dst, int ith, int nth, void* userdata) {
    const auto* desc = (const CudaLaunchDesc*)userdata;
    const int flags = (int)(uintptr_t)desc->user;
    const ggml_tensor* x = dst->src[0];
    const ggml_tensor* w = dst->src[1];
    int si = 2;
    const ggml_tensor* b = (flags & 2) ? dst->src[si++] : nullptr;
    const ggml_tensor* r = (flags & 4) ? dst->src[si++] : nullptr;
    const int W = (int)x->ne[0], H = (int)x->ne[1];
    const int IC = (int)w->ne[2], OC = (int)w->ne[3];
    const float* X = (const float*)x->data;
    const float* Wt = (const float*)w->data;
    float* Y = (float*)dst->data;
    const size_t P = (size_t)W * H;
    for (int oc = ith; oc < OC; oc += nth) {
        for (int py = 0; py < H; ++py) for (int px = 0; px < W; ++px) {
            float acc = 0.f;
            for (int ic = 0; ic < IC; ++ic)
                for (int t = 0; t < 9; ++t) {
                    const int iy = py + t / 3 - 1, ix = px + t % 3 - 1;
                    if (iy < 0 || iy >= H || ix < 0 || ix >= W) continue;
                    float v = X[ic * P + (size_t)iy * W + ix];
                    if (flags & 1) v = fmaxf(v, 0.f);
                    acc += Wt[(((size_t)oc * IC + ic) * 3 + t / 3) * 3 + t % 3] * v;
                }
            if (b) acc += ((const float*)b->data)[oc];
            if (r) acc += ((const float*)r->data)[oc * P + (size_t)py * W + px];
            Y[oc * P + (size_t)py * W + px] = acc;
        }
    }
}

// Naive CPU fallback for the transposed conv (same caveat as cpu_fallback).
void cpu_fallback_convT(ggml_tensor* dst, int ith, int nth, void* userdata) {
    const auto* desc = (const CudaLaunchDesc*)userdata;
    const int flags = (int)(uintptr_t)desc->user;
    const ggml_tensor* x = dst->src[0];
    const ggml_tensor* w = dst->src[1];
    const ggml_tensor* b = (flags & 2) ? dst->src[2] : nullptr;
    const int Win = (int)x->ne[0], Hin = (int)x->ne[1];
    const int S = (int)w->ne[0];
    const int OC = (int)w->ne[2], IC = (int)w->ne[3];
    const size_t Pin = (size_t)Win * Hin, Pout = Pin * S * S;
    const int Wout = Win * S;
    const float* X = (const float*)x->data;
    const float* Wt = (const float*)w->data;
    float* Y = (float*)dst->data;
    for (int oc = ith; oc < OC; oc += nth)
        for (int iy = 0; iy < Hin; ++iy) for (int ix = 0; ix < Win; ++ix)
            for (int ky = 0; ky < S; ++ky) for (int kx = 0; kx < S; ++kx) {
                float acc = b ? ((const float*)b->data)[oc] : 0.f;
                for (int ic = 0; ic < IC; ++ic)
                    acc += Wt[(size_t)kx + S*ky + (size_t)S*S*oc + (size_t)S*S*OC*ic] *
                           X[(size_t)ic * Pin + (size_t)iy * Win + ix];
                Y[(size_t)oc * Pout + (size_t)(iy*S+ky) * Wout + (ix*S+kx)] = acc;
            }
}

// Naive CPU fallback for the fused upsample->conv (same caveat as above).
void cpu_fallback_convup(ggml_tensor* dst, int ith, int nth, void* userdata) {
    const auto* desc = (const CudaLaunchDesc*)userdata;
    const int flags = (int)(uintptr_t)desc->user;
    const int KS = (flags & 16) ? 3 : 1;
    const ggml_tensor* x = dst->src[0];
    const ggml_tensor* w = dst->src[1];
    int si = 2;
    const ggml_tensor* b  = (flags & 2)  ? dst->src[si++] : nullptr;
    const ggml_tensor* pe = (flags & 64) ? dst->src[si++] : nullptr;
    const int W = (int)dst->ne[0], H = (int)dst->ne[1];
    const int Ws = (int)x->ne[0], Hs = (int)x->ne[1];
    const int IC = (int)w->ne[2], OC = (int)w->ne[3];
    const size_t P = (size_t)W * H, Ps = (size_t)Ws * Hs;
    const float sx = W > 1 ? (float)(Ws - 1) / (float)(W - 1) : 0.f;
    const float sy = H > 1 ? (float)(Hs - 1) / (float)(H - 1) : 0.f;
    const float* X = (const float*)x->data;
    const float* Wt = (const float*)w->data;
    float* Y = (float*)dst->data;
    auto sample = [&](int ic, int qy, int qx) -> float {
        float fy = qy * sy, fx = qx * sx;
        int y0 = (int)fy, x0 = (int)fx;
        if (y0 > Hs - 2) y0 = Hs - 2 < 0 ? 0 : Hs - 2;
        if (x0 > Ws - 2) x0 = Ws - 2 < 0 ? 0 : Ws - 2;
        float dy = fy - y0, dx = fx - x0;
        int y1 = y0 + 1 < Hs ? y0 + 1 : Hs - 1, x1 = x0 + 1 < Ws ? x0 + 1 : Ws - 1;
        const float* pl = X + ic * Ps;
        float v = (pl[(size_t)y0*Ws+x0]*(1.f-dx) + pl[(size_t)y0*Ws+x1]*dx)*(1.f-dy)
                + (pl[(size_t)y1*Ws+x0]*(1.f-dx) + pl[(size_t)y1*Ws+x1]*dx)*dy;
        if (pe) v += ((const float*)pe->data)[ic*P + (size_t)qy*W + qx];
        return v;
    };
    for (int oc = ith; oc < OC; oc += nth)
        for (int py = 0; py < H; ++py) for (int px = 0; px < W; ++px) {
            float acc = b ? ((const float*)b->data)[oc] : 0.f;
            for (int ic = 0; ic < IC; ++ic)
                for (int t = 0; t < KS*KS; ++t) {
                    int qy = py + t/KS - KS/2, qx = px + t%KS - KS/2;
                    if (qy < 0 || qy >= H || qx < 0 || qx >= W) continue;
                    acc += Wt[((size_t)oc*IC + ic)*KS*KS + t] * sample(ic, qy, qx);
                }
            Y[oc*P + (size_t)py*W + px] = acc;
        }
}

// Naive CPU fallback for the 1x1 conv (same caveat as cpu_fallback).
void cpu_fallback_conv1x1(ggml_tensor* dst, int ith, int nth, void* userdata) {
    const auto* desc = (const CudaLaunchDesc*)userdata;
    const int flags = (int)(uintptr_t)desc->user;
    const ggml_tensor* x = dst->src[0];
    const ggml_tensor* w = dst->src[1];
    const ggml_tensor* b = (flags & 2) ? dst->src[2] : nullptr;
    const int IC = (int)w->ne[2], OC = (int)w->ne[3];
    const size_t P = (size_t)x->ne[0] * x->ne[1];
    const float* X = (const float*)x->data;
    const float* Wt = (const float*)w->data;
    float* Y = (float*)dst->data;
    for (int oc = ith; oc < OC; oc += nth)
        for (size_t p = 0; p < P; ++p) {
            float acc = b ? ((const float*)b->data)[oc] : 0.f;
            for (int ic = 0; ic < IC; ++ic) {
                float v = X[ic * P + p];
                if (flags & 1) v = fmaxf(v, 0.f);
                acc += Wt[(size_t)oc * IC + ic] * v;
            }
            Y[oc * P + p] = acc;
        }
}

// One static descriptor per flag combination; userdata must stay valid for the
// node's lifetime. Launch fn chosen by the mode bits: bit3 convT, bit4/5
// upsample-conv (KS 3/1), bit7 conv1x1, else plain conv3x3.
CudaLaunchDesc* desc_for(int flags) {
    static CudaLaunchDesc pool[256];
    static const bool init = [] {
        for (int i = 0; i < 256; ++i) {
            auto fn = (i & 128) ? cuda_launch_conv1x1
                    : (i & 8) ? cuda_launch_convT
                    : (i & (16 | 32)) ? cuda_launch_convup
                    : cuda_launch;
            pool[i] = {kMagic, fn, (void*)(uintptr_t)i};
        }
        return true;
    }();
    (void)init;
    return &pool[flags];
}

}  // namespace

namespace da {

bool cuda_head_enabled() {
    static const bool on = [] {
        const char* m = std::getenv("DA_CONV");
        return m && std::strcmp(m, "cuda_fused") == 0;
    }();
    return on;
}

ggml_tensor* cuda_fused_conv3x3(ggml_context* ctx, ggml_tensor* w, ggml_tensor* b,
                                ggml_tensor* x, ggml_tensor* residual, bool prerelu) {
    if (!w || !x) return nullptr;
    if (w->ne[0] != 3 || w->ne[1] != 3) return nullptr;
    if (w->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32) return nullptr;
    if (x->ne[3] != 1 || x->ne[2] != w->ne[2]) return nullptr;
    if (!ggml_is_contiguous(w)) return nullptr;
    if (b && (b->type != GGML_TYPE_F32 || !ggml_is_contiguous(b) || b->ne[0] != w->ne[3])) return nullptr;
    if (residual && (residual->type != GGML_TYPE_F32 || !ggml_is_contiguous(residual) ||
                     residual->ne[0] != x->ne[0] || residual->ne[1] != x->ne[1] ||
                     residual->ne[2] != w->ne[3])) return nullptr;
    if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);

    int flags = (prerelu ? 1 : 0) | (b ? 2 : 0) | (residual ? 4 : 0);
    ggml_tensor* args[4];
    int n = 0;
    args[n++] = x;
    args[n++] = w;
    if (b) args[n++] = b;
    if (residual) args[n++] = residual;
    return ggml_custom_4d(ctx, GGML_TYPE_F32, x->ne[0], x->ne[1], w->ne[3], 1,
                          args, n, cpu_fallback, GGML_N_TASKS_MAX, desc_for(flags));
}

ggml_tensor* cuda_fused_convT(ggml_context* ctx, ggml_tensor* w, ggml_tensor* b,
                              ggml_tensor* x, int stride) {
    if (!w || !x) return nullptr;
    // Only the stride==kernel (pixel-shuffle GEMM) case; weight [KW,KH,OC,IC].
    if ((int)w->ne[0] != stride || (int)w->ne[1] != stride) return nullptr;
    if (w->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32) return nullptr;
    if (x->ne[3] != 1 || x->ne[2] != w->ne[3]) return nullptr;
    if (!ggml_is_contiguous(w)) return nullptr;
    if (b && (b->type != GGML_TYPE_F32 || !ggml_is_contiguous(b) || b->ne[0] != w->ne[2])) return nullptr;
    if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);

    const int flags = 8 | (b ? 2 : 0);
    ggml_tensor* args[3];
    int n = 0;
    args[n++] = x;
    args[n++] = w;
    if (b) args[n++] = b;
    return ggml_custom_4d(ctx, GGML_TYPE_F32, x->ne[0] * stride, x->ne[1] * stride,
                          w->ne[2], 1, args, n, cpu_fallback_convT, GGML_N_TASKS_MAX,
                          desc_for(flags));
}

ggml_tensor* cuda_fused_conv1x1(ggml_context* ctx, ggml_tensor* w, ggml_tensor* b,
                                ggml_tensor* x, bool prerelu) {
    if (!w || !x) return nullptr;
    if (w->ne[0] != 1 || w->ne[1] != 1) return nullptr;
    if (w->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32) return nullptr;
    if (x->ne[3] != 1 || x->ne[2] != w->ne[2]) return nullptr;
    if (!ggml_is_contiguous(w)) return nullptr;
    if (b && (b->type != GGML_TYPE_F32 || !ggml_is_contiguous(b) || b->ne[0] != w->ne[3])) return nullptr;
    if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);

    const int flags = 128 | (prerelu ? 1 : 0) | (b ? 2 : 0);
    ggml_tensor* args[3];
    int n = 0;
    args[n++] = x;
    args[n++] = w;
    if (b) args[n++] = b;
    return ggml_custom_4d(ctx, GGML_TYPE_F32, x->ne[0], x->ne[1], w->ne[3], 1,
                          args, n, cpu_fallback_conv1x1, GGML_N_TASKS_MAX,
                          desc_for(flags));
}

ggml_tensor* cuda_fused_conv_up(ggml_context* ctx, ggml_tensor* w, ggml_tensor* b,
                                ggml_tensor* x, ggml_tensor* pe, int Wout, int Hout) {
    if (!w || !x || Wout < 2 || Hout < 2) return nullptr;
    const int KS = (int)w->ne[0];
    if ((KS != 1 && KS != 3) || w->ne[1] != KS) return nullptr;
    if (w->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32) return nullptr;
    if (x->ne[3] != 1 || x->ne[2] != w->ne[2]) return nullptr;   // conv layout: [KW,KH,IC,OC]
    if (Wout < x->ne[0] || Hout < x->ne[1]) return nullptr;   // upsample only
    if (!ggml_is_contiguous(w)) return nullptr;
    if (b && (b->type != GGML_TYPE_F32 || !ggml_is_contiguous(b) || b->ne[0] != w->ne[3])) return nullptr;
    if (pe && (pe->type != GGML_TYPE_F32 || pe->ne[0] != Wout || pe->ne[1] != Hout ||
               pe->ne[2] != x->ne[2])) return nullptr;
    if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);

    const int flags = (KS == 3 ? 16 : 32) | (b ? 2 : 0) | (pe ? 64 : 0);
    ggml_tensor* args[4];
    int n = 0;
    args[n++] = x;
    args[n++] = w;
    if (b) args[n++] = b;
    if (pe) args[n++] = pe;
    return ggml_custom_4d(ctx, GGML_TYPE_F32, Wout, Hout, w->ne[3], 1,
                          args, n, cpu_fallback_convup, GGML_N_TASKS_MAX,
                          desc_for(flags));
}

}  // namespace da
