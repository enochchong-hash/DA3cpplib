// GPU image preprocessing for the depth server: hardware JPEG decode (nvJPEG)
// plus CUDA resize/normalize kernels replicating depth-anything.cpp's
// preprocess_real (cv2-faithful INTER_AREA/INTER_CUBIC with uint8 quantization
// between steps, then ImageNet normalize to CHW f32). The result stays on the
// GPU; feed it to the engine via Engine::depth_native_prepared, whose upload
// callback calls gpupre_copy_into (device-to-device).
//
// PNG has no hardware decoder on consumer GPUs (it's DEFLATE); decode PNGs on
// the CPU and enter the pipeline via gpupre_rgb8 (one H2D copy of the decoded
// pixels, resize+normalize still on GPU).
#pragma once
#include <cstddef>
#include <cstdint>

struct GpuPre;  // opaque; owns nvJPEG handles, a CUDA stream, device buffers

// nullptr if no CUDA device / nvJPEG init fails (caller falls back to CPU path).
GpuPre* gpupre_init();
void    gpupre_free(GpuPre*);

// Full-GPU path for a JPEG file in memory. On success returns 0 and sets
// *outW/*outH to the DA3 processed dims; the CHW f32 result is device-resident
// inside the context until the next gpupre_* call.
// target/patch/mean/stdv come from the engine's model config.
int gpupre_jpeg(GpuPre*, const uint8_t* data, size_t n, int target, int patch,
                const float mean[3], const float stdv[3], int* outW, int* outH);

// Same pipeline for an already-decoded host RGB8 HWC image (PNG path).
int gpupre_rgb8(GpuPre*, const uint8_t* rgb, int w, int h, int target, int patch,
                const float mean[3], const float stdv[3], int* outW, int* outH);

// Copy the device CHW f32 result (nbytes must equal 3*outH*outW*4) into dst.
// dst_is_device != 0 -> cudaMemcpy device-to-device (ggml CUDA tensor->data);
// 0 -> device-to-host (ggml tensor allocated in a host buffer). Returns 0 ok.
int gpupre_copy_into(GpuPre*, void* dst, size_t nbytes, int dst_is_device);

#include <vector>
// GPU depth-map encode: upload host depth floats (w*h), min-max normalize with
// the near=bright flip (model emits distance) on a CUDA kernel, then nvJPEG
// grayscale encode (quality 1..100). jpeg_out receives the complete JFIF
// bitstream (binary, ready to send as image/jpeg). Returns 0 ok.
int gpuenc_depth_jpeg(GpuPre*, const float* depth, int w, int h, int quality,
                      std::vector<unsigned char>& jpeg_out);
