// depth_ui server: C++ HTTP API + upload UI for depth-anything.cpp.
//
// Pure C++ inference path: links libdepthanything (ggml/CUDA) directly via
// da_capi.h and keeps the GGUF resident between requests -- the model is
// loaded once per variant (lazily, on first use), not per request.
// HTTP layer is cpp-httplib, reused from llama/vendor/cpp-httplib.
//
//   GET  /                    upload page (index.html)
//   GET  /health              which model variants exist / are loaded
//   POST /depth?variant=f32|q8   body = raw jpg/png bytes
//        -> {"depth_png": <base64>, "timings_ms": {...}}
//
// Build via scripts/build_depth.sh; run via scripts/launch_depth_ui.sh.

#include "httplib.h"
#include "engine.hpp"  // C++ engine API: lets us run the production-res
                        // depth-only graph (the C API's dense call also computes
                        // camera pose, ~20ms/request this endpoint would discard)
#include "gpu_preprocess.h"  // nvJPEG decode + CUDA resize/normalize; result
                              // stays on the GPU and is copied device-to-device
                              // into the graph input (no CPU round-trip)

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <unistd.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <memory>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

using clk = std::chrono::steady_clock;

double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

// Directory containing this executable (empty on failure).
std::string exe_dir() {
    char buf[4096];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return {};
    buf[n] = '\0';
    return std::filesystem::path(buf).parent_path().string();
}

// Resource resolution, in order: env var; <exe>/../<rel> (binary in build/,
// resources/ beside it -- keeps the whole tree relocatable and submodule-
// friendly); compile-time default (build-host fallback for unusual layouts).
std::string resource_dir(const char* env_var, const char* rel, const char* compiled) {
    const char* env = getenv(env_var);
    if (env && *env) return env;
    std::string exe = exe_dir();
    if (!exe.empty()) {
        std::filesystem::path p = std::filesystem::path(exe).parent_path() / rel;
        std::error_code ec;
        if (std::filesystem::is_directory(p, ec)) return p.lexically_normal().string();
    }
    return compiled;
}

std::string model_dir() {
#ifdef DA3_DEFAULT_MODEL_DIR
    return resource_dir("DA3_MODEL_DIR", "resources/nnmodels", DA3_DEFAULT_MODEL_DIR);
#else
    return resource_dir("DA3_MODEL_DIR", "resources/nnmodels", "resources/nnmodels");
#endif
}

std::string www_dir() {
#ifdef DA3_DEFAULT_WWW_DIR
    return resource_dir("DA3_WWW_DIR", "resources", DA3_DEFAULT_WWW_DIR);
#else
    return resource_dir("DA3_WWW_DIR", "resources", "resources");
#endif
}

// DA3 base ships f32/f16/q4_k/q8_0 only -- no q6_k exists for this model
// (the q6_k files in the HF repo belong to the older Depth Anything V2).
const std::map<std::string, std::string> MODEL_PATHS = {
    {"f32", model_dir() + "/DepthAnything-Base-F32/depth-anything-base-f32.gguf"},
    {"q8",  model_dir() + "/DepthAnything-Base-Q8/depth-anything-base-q8_0.gguf"},
    {"q4",  model_dir() + "/DepthAnything-Base-Q4/depth-anything-base-q4_k.gguf"},
};

// One resident engine per variant; the engine is not assumed thread-safe, so
// all load + inference calls are serialized behind this mutex.
std::mutex g_mutex;
std::map<std::string, std::unique_ptr<da::Engine>> g_ctxs;
GpuPre* g_gpupre = nullptr;  // GPU preprocess ctx (null -> CPU preprocess path)

bool file_exists(const std::string& p) {
    return access(p.c_str(), R_OK) == 0;
}

std::string b64_encode(const unsigned char* data, size_t len) {
    static const char tbl[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((len + 2) / 3 * 4);
    for (size_t i = 0; i < len; i += 3) {
        unsigned v = data[i] << 16;
        if (i + 1 < len) v |= data[i + 1] << 8;
        if (i + 2 < len) v |= data[i + 2];
        out += tbl[(v >> 18) & 63];
        out += tbl[(v >> 12) & 63];
        out += i + 1 < len ? tbl[(v >> 6) & 63] : '=';
        out += i + 2 < len ? tbl[v & 63] : '=';
    }
    return out;
}

std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if ((unsigned char)c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); out += b; }
                else out += c;
        }
    }
    return out;
}

void send_error(httplib::Response& res, int code, const std::string& msg) {
    res.status = code;
    res.set_content("{\"error\":\"" + json_escape(msg) + "\"}", "application/json");
}

// Normalize the float depth map to 8-bit grayscale and encode as PNG.
// The base DA3 (DualDPT) model emits distance (larger = farther); flip so the
// PNG follows the usual near = bright convention.
std::vector<unsigned char> depth_to_png(const float* depth, int h, int w) {
    float lo = depth[0], hi = depth[0];
    for (int i = 1; i < h * w; ++i) {
        lo = std::min(lo, depth[i]);
        hi = std::max(hi, depth[i]);
    }
    const float scale = hi > lo ? 255.0f / (hi - lo) : 0.0f;
    std::vector<unsigned char> gray(h * w);
    for (int i = 0; i < h * w; ++i) {
        gray[i] = (unsigned char)std::lround((hi - depth[i]) * scale);
    }
    std::vector<unsigned char> png;
    stbi_write_png_to_func(
        [](void* ctx, void* data, int size) {
            auto* v = (std::vector<unsigned char>*)ctx;
            v->insert(v->end(), (unsigned char*)data, (unsigned char*)data + size);
        },
        &png, w, h, 1, gray.data(), w);
    return png;
}

}  // namespace

int main(int argc, char** argv) {
    int port = 8090;
    if (argc > 1) port = atoi(argv[1]);
    const int n_threads = (int)std::thread::hardware_concurrency();

    // Check for API-only mode
    bool api_only = false;
    const char* api_only_env = getenv("DA3_API_ONLY");
    if (api_only_env && strcmp(api_only_env, "1") == 0) {
        api_only = true;
    }

    // Bind address + optional TLS. Defaults stay loopback/plain-HTTP; both are
    // opt-in via env because the server has no auth:
    //   DEPTH_UI_HOST=0.0.0.0            expose on the LAN (e.g. phone testing)
    //   DEPTH_UI_CERT=/..pem DEPTH_UI_KEY=/..pem   serve HTTPS -- required for
    //   phone webcam use: browsers only grant getUserMedia to secure origins.
    // scripts/launch_depth_ui.sh generates a self-signed cert with DEPTH_UI_TLS=1.
    const char* host = getenv("DEPTH_UI_HOST");
    if (!host || !*host) host = "127.0.0.1";
    const char* cert = getenv("DEPTH_UI_CERT");
    const char* key  = getenv("DEPTH_UI_KEY");
    std::unique_ptr<httplib::Server> srv_holder;
    bool tls = false;
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    if (cert && key && file_exists(cert) && file_exists(key)) {
        auto* ssl = new httplib::SSLServer(cert, key);
        srv_holder.reset(ssl);
        if (!ssl->is_valid()) {
            fprintf(stderr, "[depth-ui] failed to load TLS cert/key (%s, %s)\n", cert, key);
            return 1;
        }
        tls = true;
    } else
#endif
    {
        srv_holder = std::make_unique<httplib::Server>();
    }
    httplib::Server& srv = *srv_holder;

    // Root handler: API-only mode returns JSON banner, normal mode serves UI
    srv.Get("/", [api_only](const httplib::Request&, httplib::Response& res) {
        if (api_only) {
            res.set_content(
                "{\"service\":\"da3\",\"ui\":false,\"endpoints\":[\"/depth\",\"/health\"]}",
                "application/json");
            return;
        }
        const std::string www = www_dir();
        FILE* f = fopen((www + "/html/index.html").c_str(), "rb");
        if (!f) { send_error(res, 500, "index.html not found"); return; }
        std::string body;
        char buf[65536];
        size_t n;
        while ((n = fread(buf, 1, sizeof buf, f)) > 0) body.append(buf, n);
        fclose(f);
        res.set_content(body, "text/html; charset=utf-8");
    });

    // Mount /js for app.js
    if (!api_only) {
        const std::string www = www_dir();
        srv.set_mount_point("/js", (www + "/js").c_str());
    }

    srv.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lock(g_mutex);
        std::ostringstream os;
        os << "{\"ok\":true,\"variants\":{";
        bool first = true;
        for (const auto& [name, path] : MODEL_PATHS) {
            os << (first ? "" : ",") << "\"" << name << "\":{\"downloaded\":"
               << (file_exists(path) ? "true" : "false") << ",\"loaded\":"
               << (g_ctxs.count(name) ? "true" : "false") << "}";
            first = false;
        }
        os << "}}";
        res.set_content(os.str(), "application/json");
    });

    srv.Post("/depth", [n_threads](const httplib::Request& req, httplib::Response& res) {
        const auto t0 = clk::now();

        std::string variant = req.has_param("variant") ? req.get_param_value("variant") : "f32";
        auto it = MODEL_PATHS.find(variant);
        if (it == MODEL_PATHS.end()) { send_error(res, 400, "unknown variant '" + variant + "' (use f32|q8|q4)"); return; }
        if (!file_exists(it->second)) {
            send_error(res, 400, "model for '" + variant + "' not downloaded: " + it->second +
                                 " (see scripts/run_depth.sh header for the download command)");
            return;
        }
        if (req.body.empty()) { send_error(res, 400, "empty body; POST raw jpg/png bytes"); return; }
        const bool full_res = req.has_param("res") && req.get_param_value("res") == "full";

        // da_capi takes a file path, so spill the upload to a temp file.
        char tmp_path[] = "/tmp/depth_ui_XXXXXX";
        int fd = mkstemp(tmp_path);
        if (fd < 0) { send_error(res, 500, "mkstemp failed"); return; }
        size_t off = 0;
        while (off < req.body.size()) {
            ssize_t n = write(fd, req.body.data() + off, req.body.size() - off);
            if (n <= 0) break;
            off += (size_t)n;
        }
        close(fd);
        if (off != req.body.size()) { unlink(tmp_path); send_error(res, 500, "temp write failed"); return; }
        const double save_ms = ms_since(t0);

        double load_ms = 0.0, pre_ms = 0.0, infer_ms = 0.0;
        int h = 0, w = 0;
        bool gpu_pre = false;
        std::vector<float> depth, aux;  // aux = conf/sky, computed by the same head
        {
            std::lock_guard<std::mutex> lock(g_mutex);

            da::Engine* eng = nullptr;
            auto cit = g_ctxs.find(variant);
            if (cit != g_ctxs.end()) {
                eng = cit->second.get();
            } else {
                const auto t_load = clk::now();
                auto loaded = da::Engine::load(it->second, n_threads);
                load_ms = ms_since(t_load);
                if (!loaded) { unlink(tmp_path); send_error(res, 500, "model load failed: " + it->second); return; }
                eng = loaded.get();
                g_ctxs[variant] = std::move(loaded);
                fprintf(stderr, "[depth-ui] loaded %s in %.0f ms\n", it->second.c_str(), load_ms);
            }

            bool ok = false;
            // GPU preprocess path (std res, DA3 DualDPT models): nvJPEG decode
            // (or CPU PNG decode + one H2D copy), CUDA resize/normalize, then
            // device-to-device copy into the graph input. Any failure
            // falls through to the CPU preprocess path below.
            if (!full_res && g_gpupre && !eng->is_da2() && !eng->is_mono()) {
                const auto& cfg = eng->config();
                const auto& b = req.body;
                const bool is_jpeg = b.size() > 3 &&
                    (unsigned char)b[0] == 0xFF && (unsigned char)b[1] == 0xD8;
                const bool is_png = b.size() > 8 && memcmp(b.data(), "\x89PNG", 4) == 0;
                if ((is_jpeg || is_png) && cfg.img_mean.size() >= 3 && cfg.img_std.size() >= 3) {
                    const float mean[3] = {cfg.img_mean[0], cfg.img_mean[1], cfg.img_mean[2]};
                    const float stdv[3] = {cfg.img_std[0], cfg.img_std[1], cfg.img_std[2]};
                    const int target = (int)cfg.img_resize_target, patch = (int)cfg.patch_size;
                    int pw = 0, ph = 0, rc = -1;
                    const auto t_pre = clk::now();
                    if (is_jpeg) {
                        rc = gpupre_jpeg(g_gpupre, (const unsigned char*)b.data(), b.size(),
                                         target, patch, mean, stdv, &pw, &ph);
                    } else {
                        da::Image img;  // PNG: CPU decode, GPU resize/normalize
                        if (da::load_image_rgb(tmp_path, img))
                            rc = gpupre_rgb8(g_gpupre, img.rgb.data(), img.w, img.h,
                                             target, patch, mean, stdv, &pw, &ph);
                    }
                    pre_ms = ms_since(t_pre);
                    if (rc == 0) {
                        const auto t_inf = clk::now();
                        ok = eng->depth_native_prepared([&](ggml_tensor* t) {
                            const bool host = t->buffer && ggml_backend_buffer_is_host(t->buffer);
                            gpupre_copy_into(g_gpupre, t->data, ggml_nbytes(t), host ? 0 : 1);
                        }, ph, pw, depth, aux);
                        infer_ms = ms_since(t_inf);
                        if (ok) { h = ph; w = pw; gpu_pre = true; }
                    }
                }
            }
            if (!ok) {
                pre_ms = 0.0;  // CPU path: decode+preprocess counted inside infer_ms
                const auto t_inf = clk::now();
                if (full_res) {
                    // Legacy near-original-resolution path: ~4x the pixels of the
                    // production path, ~10x slower forward. Opt-in via ?res=full.
                    ok = eng->depth(tmp_path, depth, aux, h, w);
                } else if (eng->is_da2()) {
                    ok = eng->depth_relative_path(tmp_path, depth, h, w);
                } else if (eng->is_mono()) {
                    ok = eng->depth_mono_path(tmp_path, depth, aux, h, w);
                } else {
                    // Production DA3 path (longest side resized to 504), depth-only
                    // graph: same as da3-cli, no pose head (this endpoint discards pose).
                    ok = eng->depth_native(tmp_path, depth, aux, h, w);
                }
                infer_ms = ms_since(t_inf);
            }
            if (!ok || depth.empty()) {
                unlink(tmp_path);
                send_error(res, 500, "inference failed (see server log)");
                return;
            }
        }
        unlink(tmp_path);

        // Default response: binary grayscale JPEG encoded on the GPU (nvJPEG),
        // metadata + timings in headers -- no PNG, no base64 (~5ms CPU encode
        // and +33% payload saved). ?format=json keeps the lossless PNG/base64
        // JSON contract (also the fallback if the GPU encoder fails).
        const bool want_json = req.has_param("format") && req.get_param_value("format") == "json";
        const auto t_enc = clk::now();
        if (!want_json && g_gpupre) {
            std::vector<unsigned char> jpg;
            int rc;
            {
                std::lock_guard<std::mutex> lock(g_mutex);  // encoder buffers are shared
                rc = gpuenc_depth_jpeg(g_gpupre, depth.data(), w, h, 90, jpg);
            }
            if (rc == 0) {
                char tb[224];
                snprintf(tb, sizeof tb,
                         "{\"save_ms\":%.1f,\"model_load_ms\":%.1f,\"preprocess_ms\":%.1f,"
                         "\"infer_ms\":%.1f,\"encode_ms\":%.1f,\"server_ms\":%.1f}",
                         save_ms, load_ms, pre_ms, infer_ms, ms_since(t_enc), ms_since(t0));
                res.set_header("X-Variant", variant);
                res.set_header("X-Res", full_res ? "full" : "std");
                res.set_header("X-Gpu-Preprocess", gpu_pre ? "true" : "false");
                res.set_header("X-Depth-Width", std::to_string(w));
                res.set_header("X-Depth-Height", std::to_string(h));
                res.set_header("X-Timings-Ms", tb);
                res.set_content(std::string((const char*)jpg.data(), jpg.size()), "image/jpeg");
                return;
            }
        }

        const std::vector<unsigned char> png = depth_to_png(depth.data(), h, w);
        const std::string png_b64 = b64_encode(png.data(), png.size());
        const double encode_ms = ms_since(t_enc);

        std::ostringstream os;
        os.setf(std::ios::fixed);
        os.precision(1);
        os << "{\"variant\":\"" << variant << "\",\"res\":\"" << (full_res ? "full" : "std")
           << "\",\"gpu_preprocess\":" << (gpu_pre ? "true" : "false")
           << ",\"width\":" << w << ",\"height\":" << h
           << ",\"timings_ms\":{"
           << "\"save_ms\":" << save_ms
           << ",\"model_load_ms\":" << load_ms
           << ",\"preprocess_ms\":" << pre_ms
           << ",\"infer_ms\":" << infer_ms
           << ",\"encode_ms\":" << encode_ms
           << ",\"server_ms\":" << ms_since(t0)
           << "},\"depth_png\":\"" << png_b64 << "\"}";
        res.set_content(os.str(), "application/json");
    });

    g_gpupre = gpupre_init();
    printf("[depth-ui] GPU preprocess (nvJPEG + CUDA resize): %s\n",
           g_gpupre ? "enabled" : "unavailable, using CPU preprocess");

    printf("[depth-ui] C++ server on http%s://%s:%d (threads=%d)\n",
           tls ? "s" : "", host, port, n_threads);
    for (const auto& [name, path] : MODEL_PATHS) {
        printf("[depth-ui]   %-4s %s %s\n", name.c_str(), path.c_str(),
               file_exists(path) ? "" : "(NOT DOWNLOADED)");
    }
    fflush(stdout);

    // Pre-warm (DEPTH_UI_PREWARM=off to skip): load every downloaded variant and
    // run one dummy inference each, so no request ever pays the model load
    // (~45-380ms) or the first-graph CUDA compile (~200ms). The dummy image is
    // 3:2 -- exactly the 504x336 processed dims -- warming the graph for that
    // aspect; other aspect ratios still recompile once on first use.
    const char* pw = getenv("DEPTH_UI_PREWARM");
    if (!pw || strcmp(pw, "off") != 0) {
        char warm_path[] = "/tmp/depth_ui_warm_XXXXXX";
        int wfd = mkstemp(warm_path);
        if (wfd >= 0) {
            close(wfd);
            std::vector<unsigned char> gray((size_t)504 * 336 * 3, 128);
            if (stbi_write_png(warm_path, 504, 336, 3, gray.data(), 504 * 3)) {
                std::lock_guard<std::mutex> lock(g_mutex);
                for (const auto& [name, path] : MODEL_PATHS) {
                    if (!file_exists(path)) continue;
                    const auto t0 = clk::now();
                    auto eng = da::Engine::load(path, n_threads);
                    if (!eng) { fprintf(stderr, "[depth-ui] prewarm: load failed for %s\n", name.c_str()); continue; }
                    std::vector<float> d, a; int h = 0, w = 0;
                    eng->depth_native(warm_path, d, a, h, w);
                    g_ctxs[name] = std::move(eng);
                    printf("[depth-ui] prewarmed %-4s in %.0f ms\n", name.c_str(), ms_since(t0));
                }
                fflush(stdout);
            }
            unlink(warm_path);
        }
    }
    if (!srv.listen(host, port)) {
        fprintf(stderr, "[depth-ui] failed to listen on %s:%d\n", host, port);
        return 1;
    }
    return 0;
}
