// hx.cu - GPU side of hybrid CPU/GPU MoE expert execution (r9 HX), see ggml/include/ggml-hx.h.
//
// POST: copy the layer's activations, routed ids and weights into the layer mailbox (device-mapped pinned
//       host memory) and publish task_seq. Returns immediately; the stream continues with the shared expert.
// WAIT: spin (bounded) until the CPU engine publishes done_seq == task_seq, then copy the routed-expert
//       output into VRAM. Both are plain kernels, so a decode step stays capturable in one HIP graph.

#include "hx.cuh"
#include "uept.cuh"
#include "ggml-hx.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <cstring>

#ifdef GGML_USE_HIP

static const ggml_hx_op * hx_op_of(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_CUSTOM) {
        return nullptr;
    }
    ggml_custom_op_params p;
    memcpy(&p, op->op_params, sizeof(p));
    const ggml_hx_op * u = (const ggml_hx_op *) p.userdata;
    if (u == nullptr || u->magic != GGML_HX_MAGIC || u->mbox_dev == nullptr) {
        return nullptr;
    }
    return u;
}

bool ggml_cuda_hx_is_op(const ggml_tensor * op) {
    const ggml_hx_op * u = hx_op_of(op);
    if (!u) {
        return false;
    }
    if (u->kind == GGML_HX_KIND_POST) {
        const ggml_tensor * x = op->src[0], * ids = op->src[1], * w = op->src[2];
        return x && ids && w &&
               x->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32 && w->type == GGML_TYPE_F32 &&
               x->nb[0] == sizeof(float) && ids->nb[0] == sizeof(int32_t) && w->nb[0] == sizeof(float) &&
               x->ne[0] <= GGML_HX_MAX_EMBD && x->ne[1] <= GGML_HX_MAX_TOKENS && ids->ne[0] <= GGML_HX_MAX_USED &&
               ids->ne[1] == x->ne[1] && w->ne[0] == ids->ne[0] && w->ne[1] == ids->ne[1];
    }
    if (u->kind == GGML_HX_KIND_WAIT) {
        const ggml_tensor * g = op->src[1];
        return op->type == GGML_TYPE_F32 && op->nb[0] == sizeof(float) &&
               op->ne[0] <= GGML_HX_MAX_EMBD && op->ne[1] <= GGML_HX_MAX_TOKENS &&
               (!g || (g->type == GGML_TYPE_F32 && g->nb[0] == sizeof(float) && g->ne[0] == op->ne[0] && g->ne[1] == op->ne[1]));
    }
    if (u->kind == GGML_HX_KIND_REORDER_IDS || u->kind == GGML_HX_KIND_REORDER_W) {
        const ggml_tensor * ids = op->src[0];
        const ggml_tensor * w   = u->kind == GGML_HX_KIND_REORDER_W ? op->src[1] : nullptr;
        const ggml_tensor * exp = u->kind == GGML_HX_KIND_REORDER_W ? op->src[2] : op->src[1];
        return ids && ids->type == GGML_TYPE_I32 && ids->nb[0] == sizeof(int32_t) && ids->ne[0] <= GGML_HX_MAX_USED &&
               op->ne[0] == ids->ne[0] && op->ne[1] == ids->ne[1] && op->nb[0] == ggml_type_size(op->type) &&
               (u->kind == GGML_HX_KIND_REORDER_IDS ? op->type == GGML_TYPE_I32 :
                   (op->type == GGML_TYPE_F32 && w && w->type == GGML_TYPE_F32 && w->nb[0] == sizeof(float) &&
                    w->ne[0] == ids->ne[0] && w->ne[1] == ids->ne[1])) &&
               exp && ggml_cuda_uept_is_tensor(exp);
    }
    return false;
}

static __device__ __forceinline__ uint64_t hx_ld_sys(const uint64_t * p) {
    return __hip_atomic_load(const_cast<uint64_t *>(p), __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

static __global__ void hx_post_kernel(ggml_hx_mailbox * mb,
        const float * x, int64_t x_stride, const int32_t * ids, int64_t ids_stride,
        const float * w, int64_t w_stride, int n_embd, int n_used, int n_tokens) {
    const int tid = threadIdx.x;
    for (int i = tid; i < n_embd * n_tokens; i += blockDim.x) {
        const int t = i / n_embd;
        const int c = i - t * n_embd;
        mb->x[i] = x[t * x_stride + c];
    }
    for (int i = tid; i < n_used * n_tokens; i += blockDim.x) {
        const int t = i / n_used;
        const int k = i - t * n_used;
        mb->ids[i] = ids[t * ids_stride + k];
        mb->w[i]   = w[t * w_stride + k];
    }
    if (tid == 0) {
        mb->n_tokens = n_tokens;
        mb->n_used   = n_used;
        mb->n_embd   = n_embd;
    }
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        const uint64_t seq = __hip_atomic_load(&mb->task_seq, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) + 1;
        __hip_atomic_store(&mb->task_seq, seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

static __global__ void hx_wait_kernel(ggml_hx_mailbox * mb, float * dst, int64_t dst_stride,
        const float * gpu_part, int64_t gpu_stride, int n_embd, int n_tokens, uint64_t timeout_ticks) {
    __shared__ int ok;
    const int tid = threadIdx.x;
    if (tid == 0) {
        const uint64_t want = hx_ld_sys(&mb->task_seq);
        const uint64_t t0 = wall_clock64();
        ok = 1;
        while (hx_ld_sys(&mb->done_seq) < want) {
            __builtin_amdgcn_s_sleep(2);
            if (wall_clock64() - t0 > timeout_ticks) {
                __hip_atomic_store(&mb->gpu_timeout, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
                ok = 0;
                break;
            }
        }
    }
    __syncthreads();
    uint32_t * y = (uint32_t *) mb->y;
    for (int i = tid; i < n_embd * n_tokens; i += blockDim.x) {
        const int t = i / n_embd;
        const int c = i - t * n_embd;
        const uint32_t v = __hip_atomic_load(y + i, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        const float cpu = ok ? __uint_as_float(v) : 0.0f;
        dst[t * dst_stride + c] = gpu_part ? cpu + gpu_part[t * gpu_stride + c] : cpu;
    }
}


// H2/H3: per token, move the routed experts that the GPU computes to the front: experts resident in the VRAM
// cache first, then the highest-weighted misses, up to n_gpu; the CPU gets the rest. Top-k order is by routing
// weight, so "first misses" are the most important ones (and the ones the cache admits).
static __global__ void hx_reorder_kernel(const int32_t * ids, int64_t ids_stride, const float * w, int64_t w_stride,
        const int32_t * entries, int32_t * out_ids, float * out_w, int64_t out_stride, int n_used, int n_tokens, int n_gpu) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tokens) {
        return;
    }
    int perm[GGML_HX_MAX_USED];
    bool taken[GGML_HX_MAX_USED];
    int n = 0;
    for (int k = 0; k < n_used; ++k) {
        taken[k] = false;
    }
    if (entries) {
        for (int k = 0; k < n_used && n < n_gpu; ++k) {
            const int e = ids[t * ids_stride + k];
            if (entries[2 * e] == 2) { // UEPT_VRAM
                perm[n++] = k;
                taken[k] = true;
            }
        }
    }
    for (int k = 0; k < n_used && n < n_gpu; ++k) {
        if (!taken[k]) {
            perm[n++] = k;
            taken[k] = true;
        }
    }
    for (int k = 0; k < n_used; ++k) {
        if (!taken[k]) {
            perm[n++] = k;
        }
    }
    for (int k = 0; k < n_used; ++k) {
        if (out_ids) {
            out_ids[t * out_stride + k] = ids[t * ids_stride + perm[k]];
        }
        if (out_w) {
            out_w[t * out_stride + k] = w[t * w_stride + perm[k]];
        }
    }
}

static uint64_t hx_timeout_ticks(int device) {
    static uint64_t ticks[GGML_CUDA_MAX_DEVICES] = {};
    if (ticks[device] == 0) {
        int khz = 0;
        if (hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, device) != hipSuccess || khz <= 0) {
            khz = 100000; // 100 MHz
        }
        ticks[device] = (uint64_t) khz * 1000ull; // one second, below the 2 s Windows TDR limit
    }
    return ticks[device];
}

bool ggml_cuda_hx_compute(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_hx_op * u = hx_op_of(dst);
    if (!u) {
        return false;
    }
    cudaStream_t stream = ctx.stream();
    if (u->kind == GGML_HX_KIND_POST) {
        static bool logged = false;
        if (!logged) {
            logged = true;
            GGML_LOG_INFO("%s: HX hybrid expert path active on the GPU stream\n", __func__);
        }
        const ggml_tensor * x = dst->src[0], * ids = dst->src[1], * w = dst->src[2];
        hx_post_kernel<<<1, 256, 0, stream>>>(u->mbox_dev,
            (const float *) x->data, x->nb[1] / sizeof(float),
            (const int32_t *) ids->data, ids->nb[1] / sizeof(int32_t),
            (const float *) w->data, w->nb[1] / sizeof(float),
            (int) x->ne[0], (int) ids->ne[0], (int) x->ne[1]);
        CUDA_CHECK(cudaGetLastError());
        return true;
    }
    if (u->kind == GGML_HX_KIND_WAIT) {
        const ggml_tensor * g = dst->src[1];
        hx_wait_kernel<<<1, 256, 0, stream>>>(u->mbox_dev, (float *) dst->data, dst->nb[1] / sizeof(float),
            g ? (const float *) g->data : nullptr, g ? g->nb[1] / sizeof(float) : 0,
            (int) dst->ne[0], (int) dst->ne[1], hx_timeout_ticks(ctx.device));
        CUDA_CHECK(cudaGetLastError());
        return true;
    }
    if (u->kind == GGML_HX_KIND_REORDER_IDS || u->kind == GGML_HX_KIND_REORDER_W) {
        const bool is_w = u->kind == GGML_HX_KIND_REORDER_W;
        const ggml_tensor * ids = dst->src[0];
        const ggml_tensor * w   = is_w ? dst->src[1] : nullptr;
        const ggml_tensor * exp = is_w ? dst->src[2] : dst->src[1];
        const int n_tokens = (int) ids->ne[1];
        hx_reorder_kernel<<<(n_tokens + 63) / 64, 64, 0, stream>>>(
            (const int32_t *) ids->data, ids->nb[1] / sizeof(int32_t),
            w ? (const float *) w->data : nullptr, w ? w->nb[1] / sizeof(float) : 0,
            ggml_cuda_uept_entries(ctx, exp),
            is_w ? nullptr : (int32_t *) dst->data, is_w ? (float *) dst->data : nullptr,
            dst->nb[1] / ggml_type_size(dst->type), (int) ids->ne[0], n_tokens, u->n_gpu);
        CUDA_CHECK(cudaGetLastError());
        return true;
    }
    return false;
}

bool ggml_backend_cuda_hx_alloc(ggml_backend_t backend, size_t size, void ** host, void ** dev) {
    auto & ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
    ggml_cuda_set_device(ctx.device);
    void * h = nullptr;
    hipError_t r = hipHostMalloc(&h, size, hipHostMallocMapped | hipHostMallocPortable | hipHostMallocCoherent);
    if (r != hipSuccess) {
        GGML_LOG_WARN("%s: coherent mapped allocation failed (%s), retrying without the coherent flag\n", __func__, hipGetErrorString(r));
        (void) hipGetLastError();
        r = hipHostMalloc(&h, size, hipHostMallocMapped | hipHostMallocPortable);
    }
    if (r != hipSuccess) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes of mapped pinned memory: %s\n", __func__, size, hipGetErrorString(r));
        (void) hipGetLastError();
        return false;
    }
    memset(h, 0, size);
    void * d = nullptr;
    r = hipHostGetDevicePointer(&d, h, 0);
    if (r != hipSuccess) {
        GGML_LOG_ERROR("%s: hipHostGetDevicePointer failed: %s\n", __func__, hipGetErrorString(r));
        (void) hipHostFree(h);
        return false;
    }
    *host = h;
    *dev = d;
    return true;
}

void ggml_backend_cuda_hx_free(void * host) {
    if (host) {
        (void) hipHostFree(host);
    }
}

#else // !GGML_USE_HIP

bool ggml_cuda_hx_is_op(const ggml_tensor *) { return false; }
bool ggml_cuda_hx_compute(ggml_backend_cuda_context &, ggml_tensor *) { return false; }
bool ggml_backend_cuda_hx_alloc(ggml_backend_t, size_t, void **, void **) { return false; }
void ggml_backend_cuda_hx_free(void *) {}

#endif
