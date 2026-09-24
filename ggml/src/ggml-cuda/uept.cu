#include "uept.cuh"

#ifdef GGML_USE_HIP

#include "ggml-cuda.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "mmvq.cuh"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

struct ggml_cuda_uept_buffer {
    void * host = nullptr;
    void * device = nullptr;
};

// Expose mapped weights as ordinary host tensors to the model loader.
static const char * uept_buffer_name(ggml_backend_buffer_type_t) { return "ROCm_UEPT"; }
static size_t uept_buffer_alignment(ggml_backend_buffer_type_t) { return 256; }
static bool uept_buffer_is_host(ggml_backend_buffer_type_t) { return true; }
// MMQ may load a full quantized tile after the last row; match the native CUDA buffer contract.
static size_t uept_tensor_padding(const ggml_tensor * tensor) {
    const int64_t remainder = tensor->ne[0] % MATRIX_ROW_PADDING;
    if (!ggml_is_quantized(tensor->type) || remainder == 0) {
        return 0;
    }
    GGML_ASSERT(tensor->nb[0] == ggml_element_size(tensor));
    return ggml_row_size(tensor->type, MATRIX_ROW_PADDING - remainder);
}
static size_t uept_buffer_alloc_size(ggml_backend_buffer_type_t, const ggml_tensor * tensor) {
    return ggml_nbytes(tensor) + uept_tensor_padding(tensor);
}
static enum ggml_status uept_buffer_init_tensor(ggml_backend_buffer_t, ggml_tensor * tensor) {
    if (!tensor->view_src) {
        const size_t padding = uept_tensor_padding(tensor);
        if (padding) {
            memset(static_cast<char *>(tensor->data) + ggml_nbytes(tensor), 0, padding);
        }
    }
    return GGML_STATUS_SUCCESS;
}
static void * uept_buffer_base(ggml_backend_buffer_t buffer) {
    return static_cast<ggml_cuda_uept_buffer *>(buffer->context)->host;
}

// Free the original host allocation, never its potentially distinct device alias.
static void uept_buffer_free(ggml_backend_buffer_t buffer) {
    auto * data = static_cast<ggml_cuda_uept_buffer *>(buffer->context);
    CUDA_CHECK(hipHostFree(data->host));
    delete data;
}

// Keep loading and host-side tensor inspection independent of the mapped alias.
static void uept_buffer_set(ggml_backend_buffer_t, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    memcpy(static_cast<char *>(tensor->data) + offset, data, size);
}
static void uept_buffer_get(ggml_backend_buffer_t, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    memcpy(data, static_cast<const char *>(tensor->data) + offset, size);
}
static void uept_buffer_memset(ggml_backend_buffer_t, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset(static_cast<char *>(tensor->data) + offset, value, size);
}
static void uept_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(uept_buffer_base(buffer), value, buffer->size);
}

// Allocation failure is fatal: ordinary host memory is not an execution fallback.
static ggml_backend_buffer_t uept_buffer_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    auto data = std::make_unique<ggml_cuda_uept_buffer>();
    const int device = static_cast<int>(reinterpret_cast<intptr_t>(buft->context));
    ggml_cuda_set_device(device);
    const hipError_t result = hipHostMalloc(&data->host, std::max(size, size_t(1)), hipHostMallocMapped | hipHostMallocPortable);
    if (result != hipSuccess) {
        GGML_ABORT("UEPT: failed to allocate %zu bytes of mapped pinned host memory on device %d: %s",
                   size, device, hipGetErrorString(result));
    }
    CUDA_CHECK(hipHostGetDevicePointer(&data->device, data->host, 0));
    ggml_backend_buffer_i iface = {};
    iface.free_buffer = uept_buffer_free;
    iface.get_base = uept_buffer_base;
    iface.init_tensor = uept_buffer_init_tensor;
    iface.memset_tensor = uept_buffer_memset;
    iface.set_tensor = uept_buffer_set;
    iface.get_tensor = uept_buffer_get;
    iface.clear = uept_buffer_clear;
    return ggml_backend_buffer_init(buft, iface, data.release(), size);
}

// Identify UEPT by its unique buffer interface, not by a user-visible name.
bool ggml_cuda_uept_is_buft(ggml_backend_buffer_type_t buft) {
    return buft && buft->iface.get_name == uept_buffer_name;
}

// Views retain their source buffer; their data offset is resolved at lookup.
bool ggml_cuda_uept_is_tensor(const ggml_tensor * tensor) {
    return tensor && tensor->buffer && ggml_cuda_uept_is_buft(tensor->buffer->buft);
}

// These formats have both the MMVQ and MMQ pointer-table implementations.
bool ggml_cuda_uept_supports_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
            return true;
        default:
            return false;
    }
}

// Return one stable mapped buffer type per backend device.
ggml_backend_buffer_type_t ggml_backend_cuda_uept_buffer_type(int device) {
    static std::mutex mutex;
    static std::map<int, ggml_backend_buffer_type> types;
    std::lock_guard<std::mutex> lock(mutex);
    GGML_ASSERT(device >= 0 && device < ggml_backend_cuda_get_device_count());
    auto found = types.find(device);
    if (found != types.end()) {
        return &found->second;
    }
    ggml_backend_buffer_type type = {};
    type.iface.get_name = uept_buffer_name;
    type.iface.alloc_buffer = uept_buffer_alloc;
    type.iface.get_alignment = uept_buffer_alignment;
    type.iface.get_alloc_size = uept_buffer_alloc_size;
    type.iface.is_host = uept_buffer_is_host;
    type.device = ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), device);
    type.context = reinterpret_cast<void *>(static_cast<intptr_t>(device));
    return &types.emplace(device, type).first->second;
}

enum uept_state : int32_t { UEPT_HOST, UEPT_FILLING, UEPT_VRAM };

struct uept_entry {
    int32_t state;
    int32_t slot;
};

struct uept_counters {
    uint64_t step;
    uint64_t read_bytes;
    uint64_t hit_bytes;
    uint64_t fills;
    uint64_t readonly;
    int32_t hand;
    int32_t inflight;
    int32_t n_fill;   // r9 HX: number of entries in fill_list for the current resolve
    int32_t reserved;
};

// All pointers in this POD refer to device memory or mapped host aliases.
struct uept_layer_device {
    const char ** ptrs[3];
    const char ** host[3];
    char ** fill[3];
    int32_t * writer;
    int32_t * fill_list;   // r9 HX: experts assigned a slot by the current resolve (gather-then-compute fill)
    uept_entry * entries;
    int32_t * owner;
    uint64_t * stamp;
    uept_counters * counters;
    const int32_t * phase;
    char * pool;
    size_t offsets[3];
    size_t role_bytes[3];  // valid bytes of one expert per role (nb[2])
    size_t slot_bytes;
    int32_t n_experts;
    int32_t n_slots;
};

struct uept_layer {
    uept_layer_device device = {};
    std::array<const ggml_tensor *, 3> tensors = {};
    void * allocation = nullptr;
    int32_t mmvq_max_batch = MMVQ_MAX_BATCH_SIZE;
    int32_t number = -1;
};

struct uept_tensor_location { size_t layer; int role; };

struct uept_check_failure {
    int32_t code;
    int32_t index;
    uint64_t step;
};

struct ggml_cuda_uept_context {
    std::vector<uept_layer> layers;
    std::unordered_map<const void *, uept_tensor_location> tensor_map;
    void * pool = nullptr;
    size_t pool_bytes = 0;
    int32_t * phase = nullptr;
    bool allow_fill = true;
    bool prefill_gather = true;
    void * gather_buffer = nullptr;
    const char ** gather_ptrs = nullptr;
    size_t gather_bytes = 0;
    size_t gather_table_bytes = 0;
    bool stats_enabled = false;
    bool checks_enabled = false;
    uept_layer_device * check_layers = nullptr;
    uept_check_failure * check_errors_host = nullptr;
    uept_check_failure * check_errors_device = nullptr;
    uint64_t graph_total = 0;
    uint64_t graph_direct = 0;
    uint64_t graph_first_capture = 0;
    uint64_t graph_recapture = 0;
    uint64_t graph_replay = 0;
    uint64_t report_sample_sync = 0;
    uept_counters * report = nullptr;
    bool report_pending = false;
    uint64_t synchronize_count = 0;
};

// Reestablish HOST state without moving any immutable weight bytes.
static __global__ __launch_bounds__(256) void uept_reset_kernel(uept_layer_device layer,
        bool keep_stats = false) {
    for (int e = threadIdx.x; e < layer.n_experts; e += blockDim.x) {
        layer.entries[e] = { UEPT_HOST, -1 };
        layer.writer[e] = -1;
        for (int r = 0; r < 3; ++r) {
            layer.ptrs[r][e] = layer.host[r][e];
            layer.fill[r][e] = nullptr;
        }
    }
    for (int s = threadIdx.x; s < layer.n_slots; s += blockDim.x) {
        layer.owner[s] = -1;
        layer.stamp[s] = 0;
    }
    if (threadIdx.x == 0) {
        if (keep_stats) {
            layer.counters->hand = 0;
            layer.counters->inflight = 0;
        } else {
            *layer.counters = {};
        }
    }
}

// Explicit release-build validation: one block per layer, after all commits on the execution stream.
static __global__ __launch_bounds__(256) void uept_validate_kernel(const uept_layer_device * layers,
        uept_check_failure * errors) {
    const uept_layer_device layer = layers[blockIdx.x];
    __shared__ unsigned int failure;
    if (threadIdx.x == 0) {
        failure = layer.counters->inflight ? 1 : UINT32_MAX;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < layer.n_experts; e += blockDim.x) {
        const uept_entry entry = layer.entries[e];
        unsigned int code = 0;
        if (entry.state == UEPT_HOST) {
            code = entry.slot != -1 ? 3 : 0;
            for (int r = 0; r < 3; ++r) {
                if (layer.ptrs[r][e] != layer.host[r][e]) {
                    code = 3;
                }
            }
        } else if (entry.state == UEPT_VRAM) {
            if (entry.slot < 0 || entry.slot >= layer.n_slots) {
                code = 4;
            } else if (layer.owner[entry.slot] != e) {
                code = 5;
            } else {
                for (int r = 0; r < 3; ++r) {
                    if (layer.ptrs[r][e] != layer.pool + size_t(entry.slot) * layer.slot_bytes + layer.offsets[r]) {
                        code = 6;
                    }
                }
            }
        } else {
            code = 2; // FILLING must not survive the complete graph's down/commit operations.
        }
        if (code) {
            atomicMin(&failure, (static_cast<unsigned int>(e) << 8) | code);
        }
    }
    for (int s = threadIdx.x; s < layer.n_slots; s += blockDim.x) {
        const int e = layer.owner[s];
        unsigned int code = 0;
        if (e < -1 || e >= layer.n_experts) {
            code = 7;
        } else if (e >= 0 && (layer.entries[e].state != UEPT_VRAM || layer.entries[e].slot != s)) {
            code = 8;
        } else if (layer.stamp[s] > layer.counters->step) {
            code = 9;
        }
        if (code) {
            atomicMin(&failure, (static_cast<unsigned int>(s) << 8) | code);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0 && failure != UINT32_MAX && errors[blockIdx.x].code == 0) {
        // No host traffic on the successful path; a failure is published to this layer's mapped record.
        volatile uept_check_failure & error = errors[blockIdx.x];
        error.index = failure >> 8;
        error.step = layer.counters->step;
        __threadfence_system();
        error.code = failure & 255;
    }
}

// Resolve serially inside one block; at most 80 routed IDs are visited per decode.
static __global__ __launch_bounds__(32) void uept_resolve_kernel(uept_layer_device layer, const int32_t * ids,
        int n_tokens, int n_used, size_t ids_stride) {
    if (threadIdx.x != 0 || (layer.phase && *layer.phase == 0) || layer.counters->inflight) {
        return;
    }
    uept_counters & stats = *layer.counters;
    stats.inflight = 1;
    stats.n_fill = 0;
    const uint64_t step = ++stats.step;
    for (int e = 0; e < layer.n_experts; ++e) {
        layer.writer[e] = -1;
        for (int r = 0; r < 3; ++r) {
            layer.fill[r][e] = nullptr;
        }
    }
    // Invariant 2: pin the entire selected union before evicting any entry.
    for (int t = 0; t < n_tokens; ++t) {
        for (int k = 0; k < n_used; ++k) {
            const int e = ids[t * ids_stride + k];
            assert(e >= 0 && e < layer.n_experts);
            if (layer.entries[e].state == UEPT_VRAM) {
                layer.stamp[layer.entries[e].slot] = step;
                stats.hit_bytes += layer.slot_bytes;
            }
            stats.read_bytes += layer.slot_bytes;
        }
    }
    for (int t = 0; t < n_tokens; ++t) {
        for (int k = 0; k < n_used; ++k) {
            const int e = ids[t * ids_stride + k];
            uept_entry & entry = layer.entries[e];
            if (entry.state != UEPT_HOST || layer.writer[e] == -2) {
                continue;
            }
            int best = -1;
            uint64_t oldest = UINT64_MAX;
            for (int i = 0; i < layer.n_slots; ++i) {
                const int s = (stats.hand + i) % layer.n_slots;
                if (layer.stamp[s] != step && (best < 0 || layer.stamp[s] < oldest)) {
                    best = s;
                    oldest = layer.stamp[s];
                }
            }
            if (best < 0) {
                ++stats.readonly;
                layer.writer[e] = -2; // Avoid recounting duplicate IDs; never a writer.
                continue;
            }
            const int old = layer.owner[best];
            if (old >= 0) {
                layer.entries[old] = { UEPT_HOST, -1 };
                for (int r = 0; r < 3; ++r) {
                    layer.ptrs[r][old] = layer.host[r][old];
                }
            }
            entry = { UEPT_FILLING, best };
            layer.owner[best] = e;
            layer.stamp[best] = step;
            stats.hand = (best + 1) % layer.n_slots;
            // Invariant 3: the first linear token/route index is the unique writer.
            layer.writer[e] = t * n_used + k;
            for (int r = 0; r < 3; ++r) {
                layer.fill[r][e] = layer.pool + best * layer.slot_bytes + layer.offsets[r];
            }
            layer.fill_list[stats.n_fill++] = e;
            ++stats.fills;
        }
    }
}

// Publish cached addresses only after every matrix kernel has completed on this stream.
static __global__ __launch_bounds__(256) void uept_commit_kernel(uept_layer_device layer) {
    if ((layer.phase && *layer.phase == 0) || !layer.counters->inflight) {
        return;
    }
    for (int e = threadIdx.x; e < layer.n_experts; e += blockDim.x) {
        uept_entry & entry = layer.entries[e];
        if (entry.state == UEPT_FILLING) {
            for (int r = 0; r < 3; ++r) {
                layer.ptrs[r][e] = layer.fill[r][e];
            }
            entry.state = UEPT_VRAM;
        }
    }
    __syncthreads();
#ifndef NDEBUG
    for (int s = threadIdx.x; s < layer.n_slots; s += blockDim.x) {
        const int e = layer.owner[s];
        assert(e == -1 || (layer.entries[e].state == UEPT_VRAM && layer.entries[e].slot == s));
    }
#endif
    if (threadIdx.x == 0) {
        layer.counters->inflight = 0;
    }
}

// r9 HX gather-then-compute fill: copy each newly assigned expert's three matrices from pinned host memory into
// its VRAM slot with 16-byte loads, before any matrix kernel reads it. The following commit publishes the slot
// pointers, so MMVQ reads the expert from VRAM and never re-reads host memory to fill (replaces fill-on-read).
static __global__ __launch_bounds__(256) void uept_fill_kernel(uept_layer_device layer) {
    if ((layer.phase && *layer.phase == 0) || !layer.counters->inflight) {
        return;
    }
    const int item = blockIdx.y;
    if (item >= layer.counters->n_fill) {
        return;
    }
    const int e = layer.fill_list[item];
    if (layer.entries[e].state != UEPT_FILLING) {
        return;
    }
    constexpr size_t tile_bytes = 64 * 1024;
    const size_t first = size_t(blockIdx.x) * tile_bytes;
    for (int r = 0; r < 3; ++r) {
        const size_t bytes = layer.role_bytes[r];
        if (first >= bytes) {
            continue;
        }
        const size_t last = first + tile_bytes < bytes ? first + tile_bytes : bytes;
        const char * source = layer.host[r][e];
        char * target = layer.fill[r][e];
        if (((reinterpret_cast<uintptr_t>(source) | reinterpret_cast<uintptr_t>(target)) & 15) == 0) {
            const auto * source_vec = reinterpret_cast<const uint4 *>(source);
            auto * target_vec = reinterpret_cast<uint4 *>(target);
            for (size_t i = first / sizeof(uint4) + threadIdx.x; i < last / sizeof(uint4); i += blockDim.x) {
                target_vec[i] = source_vec[i];
            }
            for (size_t i = last / sizeof(uint4) * sizeof(uint4) + threadIdx.x; i < last; i += blockDim.x) {
                target[i] = source[i];
            }
        } else {
            for (size_t i = first + threadIdx.x; i < last; i += blockDim.x) {
                target[i] = source[i];
            }
        }
    }
}

// Enqueue outside graph capture; captured matrix/directory kernels read this stable address dynamically.
static __global__ __launch_bounds__(1) void uept_set_phase_kernel(int32_t * phase, int32_t allow_fill) {
    *phase = allow_fill;
}

// Stage only selected misses, preserving every persistent directory field (invariant 5).
// Blocks read contiguous host ranges; the following MMQ sees the completed temporary table.
static __global__ __launch_bounds__(256) void uept_gather_kernel(uept_layer_device layer, int role,
        const int32_t * expert_bounds, char * staging, size_t expert_bytes, size_t stage_stride,
        const char ** temporary_ptrs) {
    const int expert = blockIdx.y;
    const char * source = layer.ptrs[role][expert];
    const bool selected = expert_bounds[expert + 1] > expert_bounds[expert];
    const bool miss = layer.entries[expert].state != UEPT_VRAM;
    char * target = staging + size_t(expert) * stage_stride;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        temporary_ptrs[expert] = selected && miss ? target : source;
    }
    if (!selected || !miss) {
        return;
    }
    // Shared staging serves different role sizes over time, so refresh each selected expert's tail.
    if (blockIdx.x == 0) {
        for (size_t i = expert_bytes + threadIdx.x; i < stage_stride; i += blockDim.x) {
            target[i] = 0;
        }
    }
    constexpr size_t tile_bytes = 64 * 1024;
    const size_t first = size_t(blockIdx.x) * tile_bytes;
    const size_t last = first + tile_bytes < expert_bytes ? first + tile_bytes : expert_bytes;
    if ((reinterpret_cast<uintptr_t>(source) & 15) == 0) {
        // Aligned vector transactions amortize the PCIe request cost of quantized byte loads.
        const auto * source_vec = reinterpret_cast<const uint4 *>(source);
        auto * target_vec = reinterpret_cast<uint4 *>(target);
        for (size_t i = first / sizeof(uint4) + threadIdx.x; i < last / sizeof(uint4); i += blockDim.x) {
            target_vec[i] = source_vec[i];
        }
        for (size_t i = last / sizeof(uint4) * sizeof(uint4) + threadIdx.x; i < last; i += blockDim.x) {
            target[i] = source[i];
        }
    } else {
        // Small unaligned test shapes and 17-byte MXFP4 blocks remain byte-exact.
        for (size_t i = first + threadIdx.x; i < last; i += blockDim.x) {
            target[i] = source[i];
        }
    }
}

// Parse only exact routed tensor names; aliases and unrelated tensors are rejected.
static bool uept_tensor_role(const char * name, int & layer, int & role) {
    int count = 0;
    char matrix[16] = {};
    if (sscanf(name, "blk.%d.ffn_%15[^_]_exps.weight%n", &layer, matrix, &count) != 2 || name[count] != '\0') {
        return false;
    }
    role = strcmp(matrix, "gate") == 0 ? 0 : strcmp(matrix, "up") == 0 ? 1 : strcmp(matrix, "down") == 0 ? 2 : -1;
    return layer >= 0 && role >= 0;
}

// Find the device alias even on systems where mapped host and device pointers differ.
static const char * uept_tensor_device_pointer(const ggml_tensor * tensor) {
    const auto * buffer = static_cast<const ggml_cuda_uept_buffer *>(tensor->buffer->context);
    return static_cast<const char *>(buffer->device) +
           (static_cast<const char *>(tensor->data) - static_cast<const char *>(buffer->host));
}

// Carve aligned metadata arrays from a single allocation per layer.
template<typename T> static T * uept_array(char * base, size_t & offset, size_t count) {
    offset = (offset + alignof(T) - 1) & ~(alignof(T) - 1);
    T * result = base ? reinterpret_cast<T *>(base + offset) : nullptr;
    offset += sizeof(T) * count;
    return result;
}

// Describe or initialize the exact same metadata layout using a null or real base.
static size_t uept_layer_layout(uept_layer_device & d, char * base) {
    size_t offset = 0;
    for (int r = 0; r < 3; ++r) {
        d.ptrs[r] = uept_array<const char *>(base, offset, d.n_experts);
        d.host[r] = uept_array<const char *>(base, offset, d.n_experts);
        d.fill[r] = uept_array<char *>(base, offset, d.n_experts);
    }
    d.writer = uept_array<int32_t>(base, offset, d.n_experts);
    d.fill_list = uept_array<int32_t>(base, offset, d.n_experts);
    d.entries = uept_array<uept_entry>(base, offset, d.n_experts);
    d.owner = uept_array<int32_t>(base, offset, d.n_slots);
    d.stamp = uept_array<uint64_t>(base, offset, d.n_slots);
    d.counters = uept_array<uept_counters>(base, offset, 1);
    return offset;
}

// Read sticky mapped errors only after an existing synchronization; reset cannot erase a pending failure.
static void uept_check_errors(ggml_backend_cuda_context & ctx) {
    if (!ctx.uept || !ctx.uept->checks_enabled) {
        return;
    }
    const auto & state = *ctx.uept;
    for (size_t i = 0; i < state.layers.size(); ++i) {
        const volatile uept_check_failure & error = state.check_errors_host[i];
        if (error.code) {
            GGML_ABORT("UEPT directory validation failed: layer=%d index=%d code=%d step=%llu",
                       state.layers[i].number, error.index, error.code, (unsigned long long) error.step);
        }
    }
}

// Release context-owned allocations only after their stream has been synchronized by the caller.
void ggml_cuda_uept_free(ggml_backend_cuda_context & ctx) {
    if (!ctx.uept) {
        return;
    }
    ggml_cuda_set_device(ctx.device);
    uept_check_errors(ctx);
    for (auto & layer : ctx.uept->layers) {
        CUDA_CHECK(cudaFree(layer.allocation));
    }
    if (ctx.uept->pool) {
        CUDA_CHECK(cudaFree(ctx.uept->pool));
    }
    if (ctx.uept->phase) {
        CUDA_CHECK(cudaFree(ctx.uept->phase));
    }
    if (ctx.uept->gather_buffer) {
        CUDA_CHECK(cudaFree(ctx.uept->gather_buffer));
    }
    if (ctx.uept->gather_ptrs) {
        CUDA_CHECK(cudaFree(ctx.uept->gather_ptrs));
    }
    if (ctx.uept->check_layers) {
        CUDA_CHECK(cudaFree(ctx.uept->check_layers));
    }
    if (ctx.uept->check_errors_host) {
        CUDA_CHECK(hipHostFree(ctx.uept->check_errors_host));
    }
    if (ctx.uept->report) {
        CUDA_CHECK(hipHostFree(ctx.uept->report));
    }
    delete ctx.uept;
    ctx.uept = nullptr;
}

// Build per-context directories after KV, compute buffers and optional mmproj are allocated.
bool ggml_backend_cuda_uept_init(ggml_backend_t backend, const ggml_tensor * const * tensors,
        size_t n_tensors, int64_t cache_mib) {
    auto & ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
    GGML_ASSERT(cache_mib >= -1);
    ggml_cuda_set_device(ctx.device);
    // Replanning after server mmproj loading invalidates captured pointer arguments.
    CUDA_CHECK(cudaDeviceSynchronize());
    uept_check_errors(ctx);
    const bool allow_fill = ctx.uept ? ctx.uept->allow_fill : true;
#ifdef USE_CUDA_GRAPH
    ctx.cuda_graphs.clear();
#endif
    ggml_cuda_uept_free(ctx);
    auto state = std::make_unique<ggml_cuda_uept_context>();
    state->allow_fill = allow_fill;
    // Optional observability is context-local; reject typos instead of silently disabling checks.
    const auto env_flag = [](const char * name) {
        const char * value = getenv(name);
        if (!value || strcmp(value, "0") == 0) {
            return false;
        }
        if (strcmp(value, "1") != 0) {
            GGML_ABORT("UEPT: %s must be '0' or '1'", name);
        }
        return true;
    };
    state->stats_enabled = env_flag("GGML_CUDA_UEPT_STATS");
    state->checks_enabled = env_flag("GGML_CUDA_UEPT_CHECKS");
    if (const char * mode = getenv("GGML_CUDA_UEPT_MMQ")) {
        if (strcmp(mode, "direct") == 0) {
            state->prefill_gather = false;
        } else if (strcmp(mode, "gather") != 0) {
            GGML_ABORT("UEPT: GGML_CUDA_UEPT_MMQ must be 'direct' or 'gather', received '%s'", mode);
        }
    }
    std::map<int, std::array<const ggml_tensor *, 3>> layer_tensors;
    size_t uept_bytes = 0;
    for (size_t i = 0; i < n_tensors; ++i) {
        const ggml_tensor * tensor = tensors[i];
        if (!ggml_cuda_uept_is_tensor(tensor) || tensor->buffer->buft->device != backend->device) {
            continue;
        }
        int layer, role;
        if (!uept_tensor_role(tensor->name, layer, role) || !ggml_cuda_uept_supports_type(tensor->type) ||
                !ggml_is_contiguous(tensor) || tensor->ne[2] <= 0 || tensor->ne[2] > INT32_MAX || tensor->ne[3] != 1) {
            GGML_ABORT("UEPT: unsupported expert tensor %s (%s)", tensor->name, ggml_type_name(tensor->type));
        }
        auto & roles = layer_tensors[layer];
        if (roles[role] && roles[role] != tensor) {
            GGML_ABORT("UEPT: duplicate matrix role for %s", tensor->name);
        }
        roles[role] = tensor;
        uept_bytes += ggml_nbytes(tensor);
    }
    if (layer_tensors.empty()) {
        return true;
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state->phase), sizeof(int32_t)));
    uept_set_phase_kernel<<<1, 1, 0, ctx.stream()>>>(state->phase, state->allow_fill ? 1 : 0);
    CUDA_CHECK(cudaGetLastError());
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    size_t bytes_per_slot_all_layers = 0;
    int max_slots = INT32_MAX;
    for (const auto & item : layer_tensors) {
        uept_layer layer;
        layer.number = item.first;
        layer.tensors = item.second;
        layer.device.phase = state->phase;
        for (int r = 0; r < 3; ++r) {
            if (!layer.tensors[r]) {
                GGML_ABORT("UEPT: layer %d does not contain a complete gate/up/down expert triple", layer.number);
            }
            if (r > 0 && layer.tensors[r]->ne[2] != layer.tensors[0]->ne[2]) {
                GGML_ABORT("UEPT: inconsistent expert counts in layer %d", layer.number);
            }
            layer.device.offsets[r] = layer.device.slot_bytes;
            layer.device.role_bytes[r] = layer.tensors[r]->nb[2];
            const size_t padded_expert_bytes = layer.tensors[r]->nb[2] + uept_tensor_padding(layer.tensors[r]);
            layer.device.slot_bytes += (padded_expert_bytes + 255) & ~size_t(255);
            layer.mmvq_max_batch = std::min(layer.mmvq_max_batch, get_mmvq_mmid_max_batch(layer.tensors[r]->type, cc));
            state->tensor_map[layer.tensors[r]->data] = { state->layers.size(), r };
            const size_t stage_stride = (padded_expert_bytes + 15) & ~size_t(15);
            state->gather_bytes = std::max(state->gather_bytes, stage_stride * layer.tensors[r]->ne[2]);
            state->gather_table_bytes = std::max(state->gather_table_bytes,
                                               size_t(layer.tensors[r]->ne[2]) * sizeof(const char *));
        }
        layer.device.n_experts = static_cast<int32_t>(layer.tensors[0]->ne[2]);
        max_slots = std::min(max_slots, layer.device.n_experts);
        bytes_per_slot_all_layers += layer.device.slot_bytes;
        state->layers.push_back(layer);
    }
    // Leave directory storage outside the cache budget as well as the 1 GiB safety reserve.
    size_t metadata_bytes = 0;
    for (auto & layer : state->layers) {
        layer.device.n_slots = max_slots;
        metadata_bytes += uept_layer_layout(layer.device, nullptr);
    }
    if (state->checks_enabled) {
        metadata_bytes += state->layers.size() * sizeof(uept_layer_device);
    }
    size_t free_bytes, total_bytes;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const size_t free_before_staging = free_bytes;
    const size_t reserve = size_t(1) << 30;
    if (state->prefill_gather) {
        const size_t required = state->gather_bytes + state->gather_table_bytes + metadata_bytes + reserve;
        if (free_bytes < required) {
            GGML_ABORT("UEPT: gather requires %zu staging bytes and %zu metadata bytes while preserving a 1 GiB reserve, "
                       "but only %zu device bytes are free; reduce context/ubatch or explicitly select direct MMQ",
                       state->gather_bytes, state->gather_table_bytes + metadata_bytes, free_bytes);
        }
        // Stage storage is mandatory even with cache_mib=0; the cache limit applies only to the cache pool.
        CUDA_CHECK(cudaMalloc(&state->gather_buffer, state->gather_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state->gather_ptrs), state->gather_table_bytes));
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    } else {
        state->gather_bytes = 0;
        state->gather_table_bytes = 0;
    }
    size_t budget = free_bytes > reserve + metadata_bytes ? free_bytes - reserve - metadata_bytes : 0;
    if (cache_mib >= 0) {
        budget = std::min(budget, size_t(cache_mib) * 1024 * 1024);
    }
    const int slots = std::min<size_t>(max_slots, budget / bytes_per_slot_all_layers);
    state->pool_bytes = bytes_per_slot_all_layers * slots;
    if (state->pool_bytes) {
        const cudaError_t result = cudaMalloc(&state->pool, state->pool_bytes);
        if (result != cudaSuccess) {
            GGML_ABORT("UEPT: failed to allocate %zu cache bytes: %s", state->pool_bytes, cudaGetErrorString(result));
        }
        // Role layouts are fixed for this pool's lifetime; MMVQ writers only replace valid weight bytes.
        CUDA_CHECK(cudaMemsetAsync(state->pool, 0, state->pool_bytes, ctx.stream()));
    }
    size_t pool_offset = 0;
    for (auto & layer : state->layers) {
        auto & d = layer.device;
        d.n_slots = slots;
        d.pool = state->pool ? static_cast<char *>(state->pool) + pool_offset : nullptr;
        pool_offset += d.slot_bytes * slots;
        const size_t bytes = uept_layer_layout(d, nullptr);
        CUDA_CHECK(cudaMalloc(&layer.allocation, bytes));
        uept_layer_layout(d, static_cast<char *>(layer.allocation));
        for (int r = 0; r < 3; ++r) {
            std::vector<const char *> pointers(d.n_experts);
            const char * base = uept_tensor_device_pointer(layer.tensors[r]);
            for (int e = 0; e < d.n_experts; ++e) {
                pointers[e] = base + e * layer.tensors[r]->nb[2];
            }
            CUDA_CHECK(cudaMemcpy(d.host[r], pointers.data(), pointers.size() * sizeof(const char *), cudaMemcpyHostToDevice));
        }
        uept_reset_kernel<<<1, 256, 0, ctx.stream()>>>(d);
        CUDA_CHECK(cudaGetLastError());
        GGML_LOG_INFO("UEPT ResolvedPlan: layer=%d experts=%d C=%d slot_bytes=%zu mmvq_max_batch=%d\n",
                      layer.number, d.n_experts, slots, d.slot_bytes, layer.mmvq_max_batch);
    }
    CUDA_CHECK(hipHostMalloc(reinterpret_cast<void **>(&state->report), state->layers.size() * sizeof(uept_counters), hipHostMallocDefault));
    if (state->checks_enabled) {
        std::vector<uept_layer_device> descriptors;
        for (const auto & layer : state->layers) {
            descriptors.push_back(layer.device);
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state->check_layers), descriptors.size() * sizeof(uept_layer_device)));
        CUDA_CHECK(cudaMemcpy(state->check_layers, descriptors.data(), descriptors.size() * sizeof(uept_layer_device), cudaMemcpyHostToDevice));
        CUDA_CHECK(hipHostMalloc(reinterpret_cast<void **>(&state->check_errors_host), descriptors.size() * sizeof(uept_check_failure),
                                hipHostMallocMapped | hipHostMallocPortable));
        memset(state->check_errors_host, 0, descriptors.size() * sizeof(uept_check_failure));
        CUDA_CHECK(hipHostGetDevicePointer(reinterpret_cast<void **>(&state->check_errors_device), state->check_errors_host, 0));
    }
    GGML_LOG_INFO("UEPT ResolvedPlan: expert_exec=uept prefill=%s uept_bytes=%zu staging_bytes=%zu temporary_table_bytes=%zu "
                  "cache_bytes=%zu C=%d reserve_bytes=%zu free_before_staging=%zu free_before_cache=%zu\n",
                  state->prefill_gather ? "gather" : "direct", uept_bytes, state->gather_bytes, state->gather_table_bytes,
                  state->pool_bytes, slots, reserve, free_before_staging, free_bytes);
    GGML_LOG_INFO("UEPT observability: stats=%d checks=%d stats_scope=context stats_period_syncs=100\n",
                  state->stats_enabled, state->checks_enabled);
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    ctx.uept = state.release();
    return true;
}

// The caller classifies the batch explicitly; no token-count or logits heuristic is used here.
void ggml_backend_cuda_uept_set_phase(ggml_backend_t backend, bool allow_fill) {
    auto & ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
    if (!ctx.uept || ctx.uept->allow_fill == allow_fill) {
        return;
    }
    ggml_cuda_set_device(ctx.device);
    hipStreamCaptureStatus capture_status;
    CUDA_CHECK(hipStreamIsCapturing(ctx.stream(), &capture_status));
    if (capture_status != hipStreamCaptureStatusNone) {
        GGML_ABORT("UEPT: phase changes must be queued outside graph capture");
    }
    uept_set_phase_kernel<<<1, 1, 0, ctx.stream()>>>(ctx.uept->phase, allow_fill ? 1 : 0);
    CUDA_CHECK(cudaGetLastError());
    ctx.uept->allow_fill = allow_fill;
}

// Explicit diagnostic only: production inference never calls this synchronized directory snapshot.
size_t ggml_backend_cuda_uept_snapshot(ggml_backend_t backend, void * dst, size_t capacity) {
    auto & ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
    if (!ctx.uept) {
        return 0;
    }
    size_t required = 0;
    for (const auto & layer : ctx.uept->layers) {
        auto layout = layer.device;
        required += uept_layer_layout(layout, nullptr);
    }
    if (!dst || capacity < required) {
        return required;
    }
    ggml_cuda_set_device(ctx.device);
    size_t offset = 0;
    for (const auto & layer : ctx.uept->layers) {
        auto layout = layer.device;
        const size_t bytes = uept_layer_layout(layout, nullptr);
        CUDA_CHECK(cudaMemcpyAsync(static_cast<char *>(dst) + offset, layer.allocation,
                                   bytes, cudaMemcpyDeviceToHost, ctx.stream()));
        offset += bytes;
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    uept_check_errors(ctx);
    return required;
}

// Reset asynchronously on the execution stream; captured graphs retain stable table addresses.
void ggml_backend_cuda_uept_reset(ggml_backend_t backend) {
    auto & ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
    if (!ctx.uept) {
        return;
    }
    ggml_cuda_set_device(ctx.device);
    for (size_t i = 0; i < ctx.uept->layers.size(); ++i) {
        auto & layer = ctx.uept->layers[i];
        uept_reset_kernel<<<1, 256, 0, ctx.stream()>>>(layer.device, ctx.uept->stats_enabled);
        CUDA_CHECK(cudaGetLastError());
    }
}

// Reject unregistered mapped tensors rather than executing with their host address.
static uept_tensor_location uept_find(ggml_backend_cuda_context & ctx, const ggml_tensor * tensor) {
    if (!ctx.uept) {
        GGML_ABORT("UEPT: context directories have not been initialized for %s", tensor->name);
    }
    const auto found = ctx.uept->tensor_map.find(tensor->data);
    if (found == ctx.uept->tensor_map.end()) {
        GGML_ABORT("UEPT: expert tensor is not registered in this context: %s", tensor->name);
    }
    return found->second;
}

// Obtain a read-only directory view; this never changes prefill cache state (invariant 5).
ggml_cuda_uept_view ggml_cuda_uept_get_view(ggml_backend_cuda_context & ctx, const ggml_tensor * tensor) {
    if (!ggml_cuda_uept_is_tensor(tensor)) {
        return {};
    }
    const auto location = uept_find(ctx, tensor);
    const auto & d = ctx.uept->layers[location.layer].device;
    return { d.ptrs[location.role], d.fill[location.role], d.writer, d.phase };
}

// Reuse a context-owned stage after each preceding MMQ completes on the same stream.
const char * const * ggml_cuda_uept_gather(ggml_backend_cuda_context & ctx, const ggml_tensor * tensor,
        const int32_t * expert_bounds) {
    if (!ggml_cuda_uept_is_tensor(tensor)) {
        return nullptr;
    }
    const auto location = uept_find(ctx, tensor);
    const auto & state = *ctx.uept;
    const auto & layer = state.layers[location.layer].device;
    if (!state.prefill_gather) {
        return layer.ptrs[location.role];
    }
    GGML_ASSERT(expert_bounds && state.gather_buffer && state.gather_ptrs);
    const size_t expert_bytes = tensor->nb[2];
    const size_t stage_stride = (expert_bytes + uept_tensor_padding(tensor) + 15) & ~size_t(15);
    GGML_ASSERT(stage_stride * layer.n_experts <= state.gather_bytes);
    constexpr size_t tile_bytes = 64 * 1024;
    const dim3 blocks((expert_bytes + tile_bytes - 1) / tile_bytes, layer.n_experts, 1);
    uept_gather_kernel<<<blocks, 256, 0, ctx.stream()>>>(layer, location.role, expert_bounds,
        static_cast<char *>(state.gather_buffer), expert_bytes, stage_stride, state.gather_ptrs);
    CUDA_CHECK(cudaGetLastError());
    return state.gather_ptrs;
}

// Resolve exactly once per layer invocation, including unfused gate/up execution.
ggml_cuda_uept_view ggml_cuda_uept_prepare(ggml_backend_cuda_context & ctx,
        const ggml_tensor * tensor, const ggml_tensor * ids, int64_t n_tokens, const ggml_tensor * gate) {
    if (!ggml_cuda_uept_is_tensor(tensor)) {
        return {};
    }
    GGML_ASSERT(ids && ids->type == GGML_TYPE_I32 && ids->nb[0] == sizeof(int32_t));
    const auto location = uept_find(ctx, tensor);
    const auto & layer = ctx.uept->layers[location.layer];
    const auto & d = layer.device;
    if (gate) {
        const auto gate_location = uept_find(ctx, gate);
        GGML_ASSERT(gate_location.layer == location.layer);
    }
    if (n_tokens > layer.mmvq_max_batch || d.n_slots == 0) {
        return { d.ptrs[location.role], nullptr, nullptr, d.phase };
    }
    GGML_ASSERT(n_tokens > 0 && ids->ne[0] <= INT32_MAX);
    // r9 HX: resolve once per layer invocation at the first expert matrix (up, or fused gate+up), fill the newly
    // assigned slots with a gather kernel and publish them before any matrix kernel runs; the remaining
    // matrices of this layer (gate when unfused, down) read the committed directory.
    if (location.role == 1) {
        const int n_items = static_cast<int>(std::min<int64_t>(int64_t(n_tokens) * ids->ne[0], d.n_experts));
        uept_resolve_kernel<<<1, 32, 0, ctx.stream()>>>(d, static_cast<const int32_t *>(ids->data),
            static_cast<int>(n_tokens), static_cast<int>(ids->ne[0]), ids->nb[1] / sizeof(int32_t));
        CUDA_CHECK(cudaGetLastError());
        size_t max_bytes = 0;
        for (int r = 0; r < 3; ++r) {
            max_bytes = std::max(max_bytes, d.role_bytes[r]);
        }
        constexpr size_t tile_bytes = 64 * 1024;
        const dim3 blocks((max_bytes + tile_bytes - 1) / tile_bytes, std::max(n_items, 1), 1);
        uept_fill_kernel<<<blocks, 256, 0, ctx.stream()>>>(d);
        CUDA_CHECK(cudaGetLastError());
        uept_commit_kernel<<<1, 256, 0, ctx.stream()>>>(d);
        CUDA_CHECK(cudaGetLastError());
    }
    return { d.ptrs[location.role], nullptr, nullptr, d.phase };
}

// r9 HX: device pointer to the directory entries (int32 pairs {state, slot}; state 2 = VRAM) of a routed tensor.
const int32_t * ggml_cuda_uept_entries(ggml_backend_cuda_context & ctx, const ggml_tensor * tensor) {
    if (!ggml_cuda_uept_is_tensor(tensor) || !ctx.uept) {
        return nullptr;
    }
    const auto location = uept_find(ctx, tensor);
    const auto & d = ctx.uept->layers[location.layer].device;
    return d.n_slots > 0 ? reinterpret_cast<const int32_t *>(d.entries) : nullptr;
}

// The down operation is the stream-ordered commit point for the complete triple.
void ggml_cuda_uept_finish(ggml_backend_cuda_context & ctx, const ggml_tensor * tensor) {
    if (!ggml_cuda_uept_is_tensor(tensor)) {
        return;
    }
    const auto location = uept_find(ctx, tensor);
    if (location.role == 2 && ctx.uept->layers[location.layer].device.n_slots > 0) {
        uept_commit_kernel<<<1, 256, 0, ctx.stream()>>>(ctx.uept->layers[location.layer].device);
        CUDA_CHECK(cudaGetLastError());
    }
}

// Append validation while constructing/directly evaluating a graph; replay executes the captured check kernel.
void ggml_cuda_uept_validate(ggml_backend_cuda_context & ctx) {
    if (ctx.uept && ctx.uept->checks_enabled) {
        uept_validate_kernel<<<ctx.uept->layers.size(), 256, 0, ctx.stream()>>>(ctx.uept->check_layers, ctx.uept->check_errors_device);
        CUDA_CHECK(cudaGetLastError());
    }
}

// Count actual backend execution choices, not ggml topology-reuse hints.
void ggml_cuda_uept_record_graph(ggml_backend_cuda_context & ctx, bool graph, bool capture, bool first_capture) {
    if (!ctx.uept || !ctx.uept->stats_enabled) {
        return;
    }
    auto & state = *ctx.uept;
    ++state.graph_total;
    if (!graph) {
        ++state.graph_direct;
    } else if (!capture) {
        ++state.graph_replay;
    } else if (first_capture) {
        ++state.graph_first_capture;
    } else {
        ++state.graph_recapture;
    }
}

// Inspect mapped failures and infrequent counter copies only after the caller's existing synchronization.
void ggml_cuda_uept_report(ggml_backend_cuda_context & ctx) {
    if (!ctx.uept) {
        return;
    }
    auto & state = *ctx.uept;
    uept_check_errors(ctx);
    if (!state.stats_enabled) {
        return;
    }
    ++state.synchronize_count;
    if (state.report_pending) {
        uint64_t bytes = 0, hits = 0, fills = 0, readonly = 0, step = 0;
        for (size_t i = 0; i < state.layers.size(); ++i) {
            bytes += state.report[i].read_bytes;
            hits += state.report[i].hit_bytes;
            fills += state.report[i].fills;
            readonly += state.report[i].readonly;
            step = std::max(step, state.report[i].step);
        }
        char hit_rate[32] = "unavailable";
        if (bytes) {
            snprintf(hit_rate, sizeof(hit_rate), "%.6f", double(hits) / bytes);
        }
        GGML_LOG_INFO("UEPT stats: scope=context sync=%llu cache_sample_sync=%llu step=%llu read_bytes=%llu hit_bytes=%llu "
                      "miss_bytes=%llu byte_hit_rate=%s fills=%llu readonly=%llu graph_exec=%llu graph_direct=%llu "
                      "graph_first_capture=%llu graph_recapture=%llu graph_replay=%llu checks=%d zero_copy_gbps=unmeasured\n",
                      (unsigned long long) state.synchronize_count, (unsigned long long) state.report_sample_sync,
                      (unsigned long long) step, (unsigned long long) bytes, (unsigned long long) hits,
                      (unsigned long long) (bytes - hits), hit_rate,
                      (unsigned long long) fills, (unsigned long long) readonly,
                      (unsigned long long) state.graph_total, (unsigned long long) state.graph_direct,
                      (unsigned long long) state.graph_first_capture, (unsigned long long) state.graph_recapture,
                      (unsigned long long) state.graph_replay, state.checks_enabled);
        state.report_pending = false;
    }
    if (state.synchronize_count % 100 == 99) {
        for (size_t i = 0; i < state.layers.size(); ++i) {
            CUDA_CHECK(cudaMemcpyAsync(state.report + i, state.layers[i].device.counters,
                       sizeof(uept_counters), cudaMemcpyDeviceToHost, ctx.stream()));
        }
        state.report_sample_sync = state.synchronize_count;
        state.report_pending = true;
    }
}

#include "uept-test.cuh"

#endif // GGML_USE_HIP
