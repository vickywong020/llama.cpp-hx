#pragma once

#include "common.cuh"

// Hybrid CPU/GPU MoE (r9 HX): GPU side of the mailbox protocol in ggml-hx.h.
bool ggml_cuda_hx_is_op(const ggml_tensor * op);
bool ggml_cuda_hx_compute(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_backend_cuda_hx_alloc(ggml_backend_t backend, size_t size, void ** host, void ** dev);
void ggml_backend_cuda_hx_free(void * host);
