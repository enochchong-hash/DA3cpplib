// Hand-written fused CUDA kernels for the DPT head's 3x3 convolutions.
//
// The head's 3x3 stride-1 pad-1 F32 convs go through ggml_conv_2d (im2col +
// GEMM) by default, which materializes a 9x-expanded im2col tensor in global
// memory per conv. The fused implicit-GEMM kernel here gathers the im2col
// virtually inside the matmul tiles and additionally fuses the RCU's leading
// ReLU, the bias add, and the residual add -- removing several full-feature-map
// round-trips per FeatureFusionBlock.
//
// Execution: the node is a GGML_OP_CUSTOM whose userdata carries a CUDA launch
// descriptor; the locally patched third_party/ggml ggml-cuda backend detects
// the descriptor (magic), keeps the node on the GPU and calls launch() with
// the backend stream (see "depth-anything.cpp local extension" in ggml-cuda.cu).
//
// Opt-in via DA_CONV=cuda_fused. Only compiled when DA_GGML_CUDA=ON
// (DA_CUDA_HEAD is then defined); dpt_blocks.cpp guards all call sites.
#pragma once
#include "ggml.h"
namespace da {

// True iff DA_CONV=cuda_fused (checked once).
bool cuda_head_enabled();

// Build the fused conv node: y = conv3x3_pad1_stride1(prerelu ? relu(x) : x)
// [+ bias] [+ residual]. All tensors F32; w:[3,3,IC,OC], x:[W,H,IC,1],
// residual (optional) must be [W,H,OC,1] contiguous. Returns nullptr when the
// shapes/types don't qualify -- caller falls back to the ggml path.
ggml_tensor* cuda_fused_conv3x3(ggml_context* ctx, ggml_tensor* w, ggml_tensor* b,
                                ggml_tensor* x, ggml_tensor* residual, bool prerelu);

// 1x1 stride-1 pad-0 conv as a direct GEMM (no im2col copy of the input).
// Covers the head's four 1536-ch projections and out2b (OC=2 gets a dedicated
// per-pixel kernel). nullptr if shapes don't qualify.
ggml_tensor* cuda_fused_conv1x1(ggml_context* ctx, ggml_tensor* w, ggml_tensor* b,
                                ggml_tensor* x, bool prerelu);

// Transposed conv with stride == kernel size (DPT reassemble resize layers):
// reduces to a pixel-shuffle GEMM, replacing ggml's naive conv2d-transpose
// kernel. w:[S,S,OC,IC], x:[W,H,IC,1] -> [W*S,H*S,OC,1]. nullptr if shapes
// don't qualify.
ggml_tensor* cuda_fused_convT(ggml_context* ctx, ggml_tensor* w, ggml_tensor* b,
                              ggml_tensor* x, int stride);

// Fused align-corners-bilinear-upsample -> conv (kernel 1x1 or 3x3, pad KS/2,
// stride 1): replaces `interpolate(ALIGN_CORNERS, Wout x Hout) [+ pe] -> conv`
// without materializing the upsampled map. pe (optional) is the pos-embed
// added AFTER the upsample, at [Wout,Hout,IC]. nullptr if shapes don't qualify.
ggml_tensor* cuda_fused_conv_up(ggml_context* ctx, ggml_tensor* w, ggml_tensor* b,
                                ggml_tensor* x, ggml_tensor* pe, int Wout, int Hout);

}  // namespace da
