#pragma once

// Hybrid CPU/GPU MoE expert execution (r9 HX), llama side. See ggml/include/ggml-hx.h.

#include "ggml-hx.h"

#include <cstdint>
#include <vector>

struct llama_hx_layer_ops {
    ggml_hx_op post = {};
    ggml_hx_op wait = {};
    ggml_hx_op reorder_ids = {};   // H2/H3
    ggml_hx_op reorder_w   = {};
    bool       ok   = false;
};

struct llama_hx_state {
    std::vector<llama_hx_layer_ops> layers;
    int max_tokens = 4;
    int n_gpu      = 0;   // H2/H3: routed experts per token computed on the GPU (LLAMA_HX_GPU)

    void * engine    = nullptr;
    void * mbox_host = nullptr;
    int    n_mbox    = 0;

    void (*engine_stop)(void *)            = nullptr;
    void (*engine_stats)(void *, uint64_t *) = nullptr;
    void (*mbox_free)(void *)              = nullptr;

    // true when the op for layer il can run through the hybrid path for a ubatch of n_tokens
    bool enabled(int il, int64_t n_tokens, int64_t n_used) const {
        return il >= 0 && il < (int) layers.size() && layers[il].ok &&
               n_tokens >= 1 && n_tokens <= max_tokens && n_used >= 1 && n_used <= GGML_HX_MAX_USED;
    }

    // number of layers whose GPU wait kernel reported a timeout (CPU result not delivered)
    int timeouts() const;

    ~llama_hx_state();
};
