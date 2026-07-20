#ifndef DA3CPP_DA3_C_H
#define DA3CPP_DA3_C_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct da_ctx da_ctx;

int da_capi_abi_version(void);
da_ctx* da_capi_load(const char* gguf_path, int n_threads);
da_ctx* da_capi_load_nested(const char* anyview_gguf, const char* metric_gguf,
                            int n_threads);
void da_capi_free(da_ctx* ctx);
char* da_capi_info_json(da_ctx* ctx);
void da_capi_free_string(char* value);
const char* da_capi_last_error(da_ctx* ctx);

float* da_capi_depth_path(da_ctx* ctx, const char* image_path,
                          int* out_h, int* out_w);
int da_capi_pose_path(da_ctx* ctx, const char* image_path,
                      float out_ext[12], float out_intr[9]);
float* da_capi_depth_pose_multi(da_ctx* ctx, const char** image_paths,
                                int n_images, int* out_h, int* out_w, int* out_n,
                                float* out_ext, float* out_intr);
int da_capi_depth_dense(da_ctx* ctx, const char* image_path,
                        int* out_h, int* out_w, float** out_depth,
                        float** out_conf, float** out_sky, float out_ext[12],
                        float out_intr[9], int* out_is_metric);
int da_capi_points(da_ctx* ctx, const char* image_path, float conf_thresh,
                   int* out_n, float** out_xyz, unsigned char** out_rgb);
int da_capi_export_glb(da_ctx* ctx, const char* image_path, const char* out_glb);
int da_capi_export_colmap(da_ctx* ctx, const char* image_path,
                          const char* out_dir, int binary);

void da_capi_free_floats(float* value);
void da_capi_free_bytes(unsigned char* value);

#ifdef __cplusplus
}
#endif
#endif
