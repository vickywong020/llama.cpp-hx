// ggml-hx.h - shared definitions for hybrid CPU/GPU MoE expert execution (r9 HX).
//
// A decode step posts each routed-expert FFN to a per-layer mailbox in pinned, device-mapped host
// memory. A CPU engine (ggml-cpu) polls the mailboxes, computes sum_k w_k * down_k(swiglu(gate_k x, up_k x))
// and publishes the result. The GPU keeps working on the shared expert meanwhile and waits only where the
// routed output is consumed. No host synchronization is involved.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_HX_MAGIC       0x45544D48u /* "HMTE" */
#define GGML_HX_KIND_POST   1u
#define GGML_HX_KIND_WAIT   2u
#define GGML_HX_KIND_REORDER_IDS 3u  /* H2/H3: reorder each token's routed ids GPU-first (VRAM hits, then top misses) */
#define GGML_HX_KIND_REORDER_W   4u  /* same permutation applied to the routing weights */
#define GGML_HX_MAX_TOKENS  8
#define GGML_HX_MAX_USED    16
#define GGML_HX_MAX_EMBD    4096

// One mailbox per layer. Written by the GPU (task) and the CPU engine (result).
// task_seq is written only by the GPU, done_seq only by the CPU.
struct ggml_hx_mailbox {
    uint64_t task_seq;
    uint8_t  pad0[56];
    uint64_t done_seq;
    uint8_t  pad1[56];
    uint32_t gpu_timeout;   // set by the GPU wait kernel if the CPU result did not arrive in time
    int32_t  n_tokens;
    int32_t  n_used;
    int32_t  n_embd;
    uint8_t  pad2[48];
    int32_t  ids[GGML_HX_MAX_TOKENS * GGML_HX_MAX_USED];
    float    w  [GGML_HX_MAX_TOKENS * GGML_HX_MAX_USED];
    float    x  [GGML_HX_MAX_TOKENS * GGML_HX_MAX_EMBD]; // packed [n_tokens][n_embd]
    float    y  [GGML_HX_MAX_TOKENS * GGML_HX_MAX_EMBD]; // packed [n_tokens][n_embd]
};

// CPU fallback for GGML_OP_CUSTOM when the op lands on the CPU backend (synchronous, reference path).
typedef void (*ggml_hx_cpu_compute_t)(void * engine, int layer,
                                      const float * x, int64_t x_stride,
                                      const int32_t * ids, int64_t ids_stride,
                                      const float * w, int64_t w_stride,
                                      int n_tokens, int n_used, float * y, int64_t y_stride);

// userdata of the GGML_OP_CUSTOM nodes built by llama (kind = POST, WAIT, REORDER_IDS or REORDER_W)
// WAIT: src[0] = POST node, optional src[1] = GPU partial sum (added to the CPU result)
// REORDER_IDS: src[0] = ids [n_used, n_tokens] I32, src[1] = a routed expert tensor of the layer (directory lookup)
// REORDER_W:   src[0] = ids, src[1] = weights [n_used, n_tokens] F32, src[2] = routed expert tensor
struct ggml_hx_op {
    uint32_t                  magic;
    uint32_t                  kind;
    int32_t                   layer;
    int32_t                   n_gpu;      // H2/H3: routed experts per token computed on the GPU (reorder kinds)
    struct ggml_hx_mailbox  * mbox_host;
    struct ggml_hx_mailbox  * mbox_dev;
    void                    * cpu_engine;
    ggml_hx_cpu_compute_t     cpu_compute;
};

// expert weights of one layer as seen by the CPU engine (host pointers)
struct ggml_hx_layer_desc {
    struct ggml_hx_mailbox * mbox;      // host pointer
    const void * gate;
    const void * up;
    const void * down;
    int32_t type_gate;
    int32_t type_up;
    int32_t type_down;
    int32_t n_expert;
    int64_t n_embd;                     // gate/up K, down N
    int64_t n_ff;                       // gate/up N, down K
    size_t  gate_nb1, gate_nb2;
    size_t  up_nb1,   up_nb2;
    size_t  down_nb1, down_nb2;
};

// ggml-cpu proc addresses:
//   "ggml_backend_cpu_hx_start"   : void * (*)(struct ggml_hx_layer_desc * layers, int n_layers, int n_threads)
//                                   (clears layers[i].mbox for layers the engine cannot run)
//   "ggml_backend_cpu_hx_stop"    : void   (*)(void * engine)
//   "ggml_backend_cpu_hx_compute" : ggml_hx_cpu_compute_t
//   "ggml_backend_cpu_hx_stats"   : void   (*)(void * engine, uint64_t out[4])  // tasks, busy_ns, wait_ns, idle_sleeps
// ggml-cuda (HIP) proc addresses:
//   "ggml_backend_cuda_hx_alloc"  : bool   (*)(ggml_backend_t backend, size_t size, void ** host, void ** dev)
//   "ggml_backend_cuda_hx_free"   : void   (*)(void * host)

#ifdef __cplusplus
}
#endif
