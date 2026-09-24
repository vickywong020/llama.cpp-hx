#pragma once

#include "common.cuh"

// Device tables passed by value to the existing quantized matrix kernels.
struct ggml_cuda_uept_view {
    const char * const * ptrs = nullptr;
    char * const * fill_ptrs = nullptr;
    const int32_t * writer = nullptr;
    const int32_t * phase = nullptr; // Device value: 0=prefill/readonly, 1=decode/fill; null preserves legacy callers.
};

#ifdef GGML_USE_HIP
struct ggml_cuda_uept_context;

bool ggml_cuda_uept_is_buft(ggml_backend_buffer_type_t buft);
bool ggml_cuda_uept_is_tensor(const ggml_tensor * tensor);
bool ggml_cuda_uept_supports_type(ggml_type type);
ggml_backend_buffer_type_t ggml_backend_cuda_uept_buffer_type(int device);
bool ggml_backend_cuda_uept_init(ggml_backend_t backend, const ggml_tensor * const * tensors,
                               size_t n_tensors, int64_t cache_mib);
void ggml_backend_cuda_uept_reset(ggml_backend_t backend);
void ggml_backend_cuda_uept_set_phase(ggml_backend_t backend, bool allow_fill);
size_t ggml_backend_cuda_uept_snapshot(ggml_backend_t backend, void * dst, size_t capacity);
bool ggml_backend_cuda_uept_test(ggml_backend_t backend, uint32_t seed, int repetitions);
void ggml_cuda_uept_free(ggml_backend_cuda_context & ctx);
void ggml_cuda_uept_report(ggml_backend_cuda_context & ctx);
void ggml_cuda_uept_validate(ggml_backend_cuda_context & ctx);
void ggml_cuda_uept_record_graph(ggml_backend_cuda_context & ctx, bool graph, bool capture, bool first_capture);
ggml_cuda_uept_view ggml_cuda_uept_prepare(ggml_backend_cuda_context & ctx,
        const ggml_tensor * tensor, const ggml_tensor * ids, int64_t n_tokens, const ggml_tensor * gate = nullptr);
ggml_cuda_uept_view ggml_cuda_uept_get_view(ggml_backend_cuda_context & ctx, const ggml_tensor * tensor);
const char * const * ggml_cuda_uept_gather(ggml_backend_cuda_context & ctx, const ggml_tensor * tensor,
                                        const int32_t * expert_bounds);
void ggml_cuda_uept_finish(ggml_backend_cuda_context & ctx, const ggml_tensor * tensor);
const int32_t * ggml_cuda_uept_entries(ggml_backend_cuda_context & ctx, const ggml_tensor * tensor);
#endif
