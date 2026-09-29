// hx-moe.cpp - CPU side of hybrid CPU/GPU MoE execution (r9 HX).
//
// A small dedicated thread pool polls per-layer mailboxes written by GPU kernels (see ggml-hx.h),
// computes the routed-expert FFN for the posted tokens and publishes the weighted sum.
// Arithmetic matches the regular CPU MUL_MAT_ID path: activations are quantized with the vec_dot
// type's from_float and the same vec_dot kernels are used; SwiGLU uses ggml_vec_swiglu_f32.

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-hx.h"
#include "ggml-impl.h"
#include "vec.h"
#include "simd-mappings.h"
#if defined(__x86_64__) || defined(_M_X64)
#   include "arch/x86/hx-mxfp4-avx512.h"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <cmath>
#include <vector>

#if defined(_WIN32)
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#endif
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#   include <immintrin.h>
#   define HX_PAUSE() _mm_pause()
#else
#   define HX_PAUSE() std::this_thread::yield()
#endif

namespace {

using clk = std::chrono::steady_clock;

inline uint64_t hx_load_acq(const uint64_t * p) {
    return reinterpret_cast<const std::atomic<uint64_t> *>(p)->load(std::memory_order_acquire);
}
inline void hx_store_rel(uint64_t * p, uint64_t v) {
    reinterpret_cast<std::atomic<uint64_t> *>(p)->store(v, std::memory_order_release);
}

struct hx_layer {
    ggml_hx_layer_desc d;
    const ggml_type_traits_cpu * tg;   // gate traits
    const ggml_type_traits_cpu * tu;   // up traits
    const ggml_type_traits_cpu * td;   // down traits
    ggml_from_float_t q_gate;          // quantizer of the gate vec_dot type
    ggml_from_float_t q_up;
    ggml_from_float_t q_down;
    ggml_type vt_gate, vt_up, vt_down;
    size_t row_gate, row_up, row_down; // bytes of one quantized activation row
    uint64_t done = 0;                 // last completed task sequence (engine private)
};

struct hx_job {
    int      layer;
    uint64_t seq;
};

struct alignas(64) hx_worker_sample {
    int64_t start_us = 0, end_us = 0;
    int groups = 0;
};

struct hx_engine {
    std::vector<hx_layer> layers;
    int n_threads = 8;
    bool profile = getenv("LLAMA_VERIFY_PROFILE") != nullptr;
    bool read_probe = getenv("LLAMA_HX_READ_PROBE") != nullptr;
    bool expert_parallel = getenv("LLAMA_HX_EXPERT_PARALLEL") != nullptr;
    int schedule = [] {
        const char * value = getenv("LLAMA_HX_SCHEDULE");
        return value && std::strcmp(value, "dynamic") == 0 ? 1 :
               value && std::strcmp(value, "memory") == 0 ? 2 : 0;
    }();
    bool balance_profile = getenv("LLAMA_HX_BALANCE_PROFILE") != nullptr;
    std::vector<hx_worker_sample> worker_samples;
    int64_t balance_span_us = 0, balance_busy_us = 0, balance_wait_us = 0;
    int balance_jobs = 0, balance_min_groups = 0, balance_max_groups = 0;
    double profile_a_us = 0, profile_b_us = 0, profile_read_us = 0;
    uint64_t profile_jobs = 0, profile_unique = 0, profile_pairs = 0, profile_bytes = 0;
    int n_primary_layers = 0;
    int64_t max_ff = 0;
    int64_t max_embd = 0;

    std::vector<std::thread> threads;
    std::atomic<bool> stop{false};

    // job broadcast
    std::atomic<uint64_t> gen{0};
    hx_job job{};
    alignas(64) std::atomic<int> next_expert{0};
    std::atomic<int> bar_count{0};
    std::atomic<uint64_t> bar_gen{0};

    // shared intermediates: h[pair][n_ff]
    std::vector<float> h;
    std::vector<float> expert_output;

    // stats
    std::atomic<uint64_t> n_tasks{0};
    std::atomic<uint64_t> busy_ns{0};
    std::atomic<uint64_t> sleeps{0};

    // r10 V-Cache prefetch: while the GPU runs attention/GDN for layer l+1 the DDR is idle; the engine predicts the
    // routed experts of layer l+1 from layer l (online cross-layer co-occurrence + previous token) and every thread
    // streams its own row slices of the predicted experts into its L2/L3 (96 MB V-Cache), aborting when the task
    // arrives. LLAMA_HX_PREFETCH=<experts per layer> (0 = off).
    int pf_k = 0;
    int n_expert = 0;
    std::vector<uint16_t> cooc;        // [layer l][a][b] counts of (a at l, b at l+1) for the same token
    std::vector<int32_t>  last_ids;    // [layer][GGML_HX_MAX_USED] ids of token 0 at the last task of the layer
    std::vector<float>    last_w;
    std::vector<int>      last_n;
    std::vector<int32_t>  pred_ids;    // [layer][32] last prediction for the layer
    std::vector<int>      pred_n;
    std::atomic<uint64_t> pf_gen{0};
    int     pf_layer = -1;
    int     pf_cnt = 0;
    int32_t pf_list[32] = {};
    std::atomic<uint64_t> pf_predicted{0}, pf_hits{0}, pf_bytes{0}, pf_aborts{0};
    // layer-ahead gating predictor: int8 copy of every layer's router (ffn_gate_inp) applied to layer l's MoE input
    // (LLAMA_HX_PREDICT=router, default when routers were provided) or the co-occurrence table (=cooc)
    int pred_mode = 0;                          // 0 = co-occurrence, 1 = router
    int pf_hint = 0;                            // 0 = demand touch, 1 = prefetcht0, 2 = t1, 3 = t2, 4 = nta
    int pf_skip = 0;                            // with H2 the GPU takes the top-weighted experts: skip that many top predictions
    std::vector<std::vector<int8_t>> rq;        // [layer][n_expert][n_embd]
    std::vector<std::vector<float>>  rs;        // [layer][n_expert] row scale
    std::vector<std::vector<int32_t>> rsum;     // [layer][n_expert] sum of int8 row (for the u8 offset trick)
    std::atomic<uint64_t> pred_router{0};

    void barrier() {
        const uint64_t g = bar_gen.load(std::memory_order_acquire);
        if (bar_count.fetch_add(1, std::memory_order_acq_rel) == n_threads - 1) {
            bar_count.store(0, std::memory_order_relaxed);
            bar_gen.store(g + 1, std::memory_order_release);
        } else {
            while (bar_gen.load(std::memory_order_acquire) == g) {
                HX_PAUSE();
            }
        }
    }
};

// ---------------------------------------------------------------------------------------------
// math

struct hx_scratch;

// r10 P2a: dot products of nrows consecutive weight rows (stride nb1) with one quantized activation row.
// GGML_HX_KERNELS selects the MXFP4 x Q8_0 path (AVX-512 builds):
//   2 (default) v3 W4A8 kernel (4 blocks per zmm, activation pre-laid-out once per call, 4 rows per call);
//               not bit-exact with ggml, validated against a long-double reference (scripts/r10/p2a-test3.c)
//   1           v1 4-row kernel, bit-exact with the ggml AVX2 kernel
//   0           one ggml vec_dot call per row (bit-exact)
static int hx_kernel_level() {
    static const int lvl = [] { const char * e = getenv("GGML_HX_KERNELS"); return e ? atoi(e) : 2; }();
    return lvl;
}

#if defined(GGML_HX_MXFP4_AVX512)
struct hx_prep {
    std::vector<uint8_t> buf;      // 3 arrays of nb/4 x 64 bytes, 64-byte aligned inside buf
    hxk3_act a{};
    const hxk3_act * make(int nb, const void * y) {
        const size_t ng = (size_t) nb / 4;
        if (buf.size() < 3 * 64 * ng + 64) { buf.resize(3 * 64 * ng + 64); }
        uint8_t * p = (uint8_t *) (((uintptr_t) buf.data() + 63) & ~(uintptr_t) 63);
        a.lo = (__m512i *) p;
        a.hi = (__m512i *) (p + 64 * ng);
        a.d  = (__m512  *) (p + 128 * ng);
        hxk3_prepare(nb, y, &a);
        return &a;
    }
};
#endif

// m activation rows (acts[j]) against the same nrows weight rows; outs[j] receives nrows results for activation j.
// Every (row, activation) result is computed with exactly the instruction sequence of the single-activation path.
static inline void hx_dot_rows_multi(const ggml_type_traits_cpu * tt, ggml_type wtype, ggml_type vtype, int n,
                                     float * const * outs, const char * w, size_t nb1, int64_t nrows,
                                     const void * const * acts, int m) {
    int64_t r = 0;
#if defined(GGML_HX_MXFP4_AVX512)
    const int lvl = hx_kernel_level();
    if (wtype == GGML_TYPE_MXFP4 && vtype == GGML_TYPE_Q8_0 && lvl > 0) {
        const int nb = n / QK_MXFP4;
        if (lvl >= 2 && nb % 4 == 0) {
            thread_local hx_prep prep[GGML_HX_MAX_TOKENS * GGML_HX_MAX_USED];
            const hxk3_act * pa[GGML_HX_MAX_TOKENS * GGML_HX_MAX_USED];
            for (int j = 0; j < m; ++j) {
                pa[j] = prep[j].make(nb, acts[j]);
            }
            for (; r + 4 <= nrows; r += 4) {
                for (int j0 = 0; j0 < m; j0 += 4) {
                    const int tn = std::min(4, m - j0);
                    float * o[4];
                    for (int jj = 0; jj < tn; ++jj) o[jj] = outs[j0 + jj] + r;
                    hxk3_dot_mxfp4_r4_multi(tn, o, w + r * nb1, nb1, pa + j0);
                }
            }
            for (; r < nrows; ++r) {
                for (int j = 0; j < m; ++j) hxk3_dot_mxfp4_r1(outs[j] + r, w + r * nb1, pa[j]);
            }
        } else if (nb % 2 == 0) {
            for (; r + 4 <= nrows; r += 4) {
                for (int j = 0; j < m; ++j) {
                    hxk_mxfp4_q8_0_r4(nb, outs[j] + r, w + r * nb1, nb1, (const block_q8_0 *) acts[j]);
                }
            }
        }
    }
#else
    (void) wtype; (void) vtype;
#endif
    for (; r < nrows; ++r) {
        for (int j = 0; j < m; ++j) {
            tt->vec_dot(n, outs[j] + r, 0, w + r * nb1, 0, acts[j], 0, 1);
        }
    }
}

// unique experts of a task in first-appearance order; members[u] = pair indices p = t*n_used + k
struct hx_groups {
    int n = 0;
    int32_t expert[GGML_HX_MAX_TOKENS * GGML_HX_MAX_USED];
    int count[GGML_HX_MAX_TOKENS * GGML_HX_MAX_USED];
    int members[GGML_HX_MAX_TOKENS * GGML_HX_MAX_USED][GGML_HX_MAX_TOKENS];
    void build(const int32_t * ids, int64_t ids_stride, int n_tokens, int n_used, int n_expert) {
        n = 0;
        for (int t = 0; t < n_tokens; ++t) {
            for (int k = 0; k < n_used; ++k) {
                const int32_t e = ids[t * ids_stride + k];
                GGML_ASSERT(e >= 0 && e < n_expert);
                int u = 0;
                while (u < n && expert[u] != e) ++u;
                if (u == n) { expert[n] = e; count[n] = 0; ++n; }
                GGML_ASSERT(count[u] < GGML_HX_MAX_TOKENS);
                members[u][count[u]++] = t * n_used + k;
            }
        }
    }
};

struct hx_scratch {
    std::vector<uint8_t> xq_gate, xq_up, hq;
    std::vector<float> g, u, v, acc;
    hx_groups groups;
    std::vector<int> work;
    int processed_groups = 0;
};

// y[t][r] for r in [r0, r1): sum_k w[t,k] * down_{e}(h_{t,k})[r]
// r10 P1: both stages walk the UNIQUE experts of the task; each expert's rows are read once for all tokens that
// selected it (speculative verification batches), with results bit-identical to the per-token order.
static void hx_compute_stage_a(const hx_layer & L, const float * x, int64_t x_stride,
                               const int32_t * ids, int64_t ids_stride,
                               int n_tokens, int n_used, float * h,
                               int ith, int nth, hx_scratch & s) {
    const int64_t n_embd = L.d.n_embd;
    const int64_t n_ff   = L.d.n_ff;
    s.xq_gate.resize((size_t) n_tokens * L.row_gate);
    s.xq_up.resize((size_t) n_tokens * L.row_up);
    for (int t = 0; t < n_tokens; ++t) {
        L.q_gate(x + t * x_stride, s.xq_gate.data() + t * L.row_gate, n_embd);
        if (L.vt_up != L.vt_gate) {
            L.q_up(x + t * x_stride, s.xq_up.data() + t * L.row_up, n_embd);
        }
    }
    const uint8_t * xq_up_base = L.vt_up != L.vt_gate ? s.xq_up.data() : s.xq_gate.data();

    const int64_t r0 = (n_ff * ith) / nth;
    const int64_t r1 = (n_ff * (ith + 1)) / nth;
    const int64_t nr = r1 - r0;
    hx_groups & G = s.groups;
    G.build(ids, ids_stride, n_tokens, n_used, L.d.n_expert);
    s.g.resize((size_t) nr * GGML_HX_MAX_TOKENS);
    s.u.resize((size_t) nr * GGML_HX_MAX_TOKENS);
    for (int ui = 0; ui < G.n; ++ui) {
        const int32_t e = G.expert[ui];
        const int m = G.count[ui];
        const void * ag[GGML_HX_MAX_TOKENS], * au[GGML_HX_MAX_TOKENS];
        float * og[GGML_HX_MAX_TOKENS], * ou[GGML_HX_MAX_TOKENS];
        for (int j = 0; j < m; ++j) {
            const int t = G.members[ui][j] / n_used;
            ag[j] = s.xq_gate.data() + t * L.row_gate;
            au[j] = xq_up_base + t * L.row_up;
            og[j] = s.g.data() + (size_t) j * nr;
            ou[j] = s.u.data() + (size_t) j * nr;
        }
        const char * wg = (const char *) L.d.gate + e * L.d.gate_nb2;
        const char * wu = (const char *) L.d.up   + e * L.d.up_nb2;
        hx_dot_rows_multi(L.tg, (ggml_type) L.d.type_gate, L.vt_gate, (int) n_embd, og, wg + r0 * L.d.gate_nb1, L.d.gate_nb1, nr, ag, m);
        hx_dot_rows_multi(L.tu, (ggml_type) L.d.type_up,   L.vt_up,   (int) n_embd, ou, wu + r0 * L.d.up_nb1,   L.d.up_nb1,   nr, au, m);
        for (int j = 0; j < m; ++j) {
            float * hp = h + (int64_t) G.members[ui][j] * n_ff + r0;
            ggml_vec_swiglu_f32((int) nr, hp, og[j], ou[j]);
        }
    }
}

static void hx_compute_stage_b(const hx_layer & L, const int32_t * ids, int64_t ids_stride,
                               const float * w, int64_t w_stride,
                               int n_tokens, int n_used, const float * h,
                               float * y, int64_t y_stride,
                               int ith, int nth, hx_scratch & s) {
    const int64_t n_embd = L.d.n_embd;
    const int64_t n_ff   = L.d.n_ff;
    const int n_pairs = n_tokens * n_used;
    s.hq.resize((size_t) n_pairs * L.row_down);
    for (int p = 0; p < n_pairs; ++p) {
        L.q_down(h + (int64_t) p * n_ff, s.hq.data() + p * L.row_down, n_ff);
    }
    const int64_t r0 = (n_embd * ith) / nth;
    const int64_t r1 = (n_embd * (ith + 1)) / nth;
    const int64_t nr = r1 - r0;
    hx_groups & G = s.groups;
    G.build(ids, ids_stride, n_tokens, n_used, L.d.n_expert);
    s.v.resize((size_t) n_pairs * nr);
    for (int ui = 0; ui < G.n; ++ui) {
        const int32_t e = G.expert[ui];
        const int m = G.count[ui];
        const void * ad[GGML_HX_MAX_TOKENS];
        float * od[GGML_HX_MAX_TOKENS];
        for (int j = 0; j < m; ++j) {
            const int p = G.members[ui][j];
            ad[j] = s.hq.data() + p * L.row_down;
            od[j] = s.v.data() + (size_t) p * nr;
        }
        const char * wd = (const char *) L.d.down + e * L.d.down_nb2;
        hx_dot_rows_multi(L.td, (ggml_type) L.d.type_down, L.vt_down, (int) n_ff, od, wd + r0 * L.d.down_nb1, L.d.down_nb1, nr, ad, m);
    }
    s.acc.resize((size_t) nr);
    for (int t = 0; t < n_tokens; ++t) {
        std::fill(s.acc.begin(), s.acc.end(), 0.0f);
        for (int k = 0; k < n_used; ++k) {
            const float   wk = w[t * w_stride + k];
            const float * vp = s.v.data() + (size_t) (t * n_used + k) * nr;
            for (int64_t r = 0; r < nr; ++r) {
                s.acc[r] += vp[r] * wk;   // same order as MUL(weights) followed by the chain of ADDs
            }
        }
        float * yt = y + t * y_stride;
        for (int64_t r = 0; r < nr; ++r) {
            yt[r0 + r] = s.acc[r];
        }
    }
}

static void hx_compute_expert_groups(const hx_layer & L, const ggml_hx_mailbox * mb,
        float * output, int ith, int nth, hx_scratch & s, int schedule, std::atomic<int> & next_expert) {
    const int nt = mb->n_tokens, nk = mb->n_used;
    const int64_t ne = L.d.n_embd, nf = L.d.n_ff;
    hx_groups & G = s.groups;
    G.build(mb->ids, nk, nt, nk, L.d.n_expert);

    int order[GGML_HX_MAX_TOKENS * GGML_HX_MAX_USED];
    int owner[GGML_HX_MAX_TOKENS * GGML_HX_MAX_USED];
    for (int i = 0; i < G.n; ++i) order[i] = i;
    std::sort(order, order + G.n, [&](int a, int b) {
        return G.count[a] != G.count[b] ? G.count[a] > G.count[b] : a < b;
    });
    if (schedule != 1) {
        s.work.assign(nth, 0);
        for (int i = 0; i < G.n; ++i) {
            const int u = order[i];
            const int worker = std::min_element(s.work.begin(), s.work.end()) - s.work.begin();
            owner[u] = worker;
            s.work[worker] += schedule == 2 ? 1 : 2 + G.count[u];
        }
    }

    s.xq_gate.resize(nt * L.row_gate);
    s.xq_up.resize(nt * L.row_up);
    for (int t = 0; t < nt; ++t) {
        L.q_gate(mb->x + t * ne, s.xq_gate.data() + t * L.row_gate, ne);
        if (L.vt_up != L.vt_gate) L.q_up(mb->x + t * ne, s.xq_up.data() + t * L.row_up, ne);
    }
    const uint8_t * xu = L.vt_up != L.vt_gate ? s.xq_up.data() : s.xq_gate.data();
    s.g.resize(nt * nf);
    s.u.resize(nt * nf);
    s.hq.resize(nt * L.row_down);
    s.processed_groups = 0;
    for (int i = 0;; ++i) {
        const int index = schedule == 1 ? next_expert.fetch_add(1, std::memory_order_relaxed) : i;
        if (index >= G.n) break;
        const int u = schedule ? order[index] : index;
        if (schedule != 1 && owner[u] != ith) continue;
        ++s.processed_groups;
        const int e = G.expert[u], m = G.count[u];
        const void * ag[GGML_HX_MAX_TOKENS], * au[GGML_HX_MAX_TOKENS], * ad[GGML_HX_MAX_TOKENS];
        float * og[GGML_HX_MAX_TOKENS], * ou[GGML_HX_MAX_TOKENS], * od[GGML_HX_MAX_TOKENS];
        for (int j = 0; j < m; ++j) {
            const int p = G.members[u][j], t = p / nk;
            ag[j] = s.xq_gate.data() + t * L.row_gate;
            au[j] = xu + t * L.row_up;
            og[j] = s.g.data() + j * nf;
            ou[j] = s.u.data() + j * nf;
            ad[j] = s.hq.data() + j * L.row_down;
            od[j] = output + p * ne;
        }
        hx_dot_rows_multi(L.tg, (ggml_type) L.d.type_gate, L.vt_gate, ne, og,
            (const char *) L.d.gate + e * L.d.gate_nb2, L.d.gate_nb1, nf, ag, m);
        hx_dot_rows_multi(L.tu, (ggml_type) L.d.type_up, L.vt_up, ne, ou,
            (const char *) L.d.up + e * L.d.up_nb2, L.d.up_nb1, nf, au, m);
        for (int j = 0; j < m; ++j) {
            ggml_vec_swiglu_f32(nf, og[j], og[j], ou[j]);
            L.q_down(og[j], s.hq.data() + j * L.row_down, nf);
        }
        hx_dot_rows_multi(L.td, (ggml_type) L.d.type_down, L.vt_down, nf, od,
            (const char *) L.d.down + e * L.d.down_nb2, L.d.down_nb1, ne, ad, m);
    }
}

static void hx_reduce_expert_groups(const hx_layer & L, ggml_hx_mailbox * mb,
        const float * output, int ith, int nth) {
    const int64_t ne = L.d.n_embd;
    const int64_t r0 = ne * ith / nth, r1 = ne * (ith + 1) / nth;
    for (int t = 0; t < mb->n_tokens; ++t) {
        float * y = mb->y + t * ne;
        std::fill(y + r0, y + r1, 0.0f);
        // Keep the original expert reduction order for each token.
        for (int k = 0; k < mb->n_used; ++k) {
            const int p = t * mb->n_used + k;
            const float w = mb->w[p];
            const float * v = output + p * ne;
            for (int64_t r = r0; r < r1; ++r) y[r] += v[r] * w;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// r10 V-Cache prefetch

static void hx_pf_init(hx_engine & E) {
    {
        const char * h = getenv("LLAMA_HX_PF_HINT");
        const std::string v = h ? h : "touch";
        E.pf_hint = v == "t0" ? 1 : v == "t1" ? 2 : v == "t2" ? 3 : v == "nta" ? 4 : 0;
        const char * m = getenv("LLAMA_HX_PREDICT");
        E.pred_mode = (m && std::string(m) == "cooc") ? 0 : 1;
        const char * sk = getenv("LLAMA_HX_PF_SKIP");
        const char * g  = getenv("LLAMA_HX_GPU");
        E.pf_skip = std::max(0, std::min(16, sk ? atoi(sk) : (g ? atoi(g) : 0)));
    }
    const char * e = getenv("LLAMA_HX_PREFETCH");
    E.pf_k = e ? std::max(0, std::min(32, atoi(e))) : 0;
    const int nl = E.n_primary_layers;
    int ne = 0;
    for (const auto & L : E.layers) ne = std::max(ne, (int) L.d.n_expert);
    E.n_expert = ne;
    if (!E.pf_k || nl < 2 || ne <= 0) { E.pf_k = 0; return; }
    E.cooc.assign((size_t) (nl - 1) * ne * ne, 0);
    E.last_ids.assign((size_t) nl * GGML_HX_MAX_USED, -1);
    E.last_w.assign((size_t) nl * GGML_HX_MAX_USED, 0.0f);
    E.last_n.assign(nl, 0);
    E.pred_ids.assign((size_t) nl * 32, -1);
    E.pred_n.assign(nl, 0);
}

// leader, at the start of a task for layer li (token 0 of the task): hit statistics, co-occurrence update, remember ids
static void hx_pf_observe(hx_engine & E, int li, const ggml_hx_mailbox * mb) {
    if (!E.pf_k || li >= E.n_primary_layers) return;
    const int n_used = std::min<int>(mb->n_used, GGML_HX_MAX_USED);
    const int ne = E.n_expert;
    if (E.pred_n[li] > 0) {
        int hit = 0;
        for (int k = 0; k < n_used; ++k)
            for (int j = 0; j < E.pred_n[li]; ++j) hit += E.pred_ids[(size_t) li * 32 + j] == mb->ids[k];
        E.pf_hits.fetch_add(hit, std::memory_order_relaxed);
        E.pf_predicted.fetch_add(E.pred_n[li], std::memory_order_relaxed);
        E.pred_n[li] = 0;
    }
    if (li > 0 && E.last_n[li - 1] > 0) {
        uint16_t * C = E.cooc.data() + (size_t) (li - 1) * ne * ne;
        for (int a = 0; a < E.last_n[li - 1]; ++a) {
            uint16_t * row = C + (size_t) E.last_ids[(size_t) (li - 1) * GGML_HX_MAX_USED + a] * ne;
            for (int k = 0; k < n_used; ++k) {
                uint16_t & c = row[mb->ids[k]];
                if (c == 65535) { for (int b = 0; b < ne; ++b) row[b] >>= 1; }
                ++c;
            }
        }
    }
}

// router logits of layer lj for the MoE input x of the current layer (token 0): int8 x int8 with per-row scales.
// Only the ranking matters (softmax gating is monotonic), so int8 precision is ample for choosing what to prefetch.
static void hx_router_scores(const hx_engine & E, int lj, const float * x, int n_embd, float * score) {
    const int ne = (int) E.rs[lj].size();
    float amax = 0.0f;
    for (int i = 0; i < n_embd; ++i) amax = std::max(amax, std::fabs(x[i]));
    const float sx = amax > 0.0f ? amax / 127.0f : 1.0f;
    alignas(64) uint8_t xu[GGML_HX_MAX_EMBD];
    for (int i = 0; i < n_embd; ++i) xu[i] = (uint8_t) (int) (std::lrintf(x[i] / sx) + 128);   // offset: u8 = q + 128
    const int8_t * W = E.rq[lj].data();
    for (int e = 0; e < ne; ++e) {
        const int8_t * wr = W + (size_t) e * n_embd;
        int32_t acc = 0;
        int i = 0;
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
        __m512i va = _mm512_setzero_si512();
        for (; i + 64 <= n_embd; i += 64) {
            va = _mm512_dpbusd_epi32(va, _mm512_loadu_si512(xu + i), _mm512_loadu_si512(wr + i));
        }
        acc = _mm512_reduce_add_epi32(va);
#endif
        for (; i < n_embd; ++i) acc += (int32_t) xu[i] * wr[i];
        score[e] = (float) (acc - 128 * E.rsum[lj][e]) * E.rs[lj][e] * sx;
    }
}

// leader, after finishing layer li: remember its ids, predict layer li+1 and publish the prefetch
static void hx_pf_predict(hx_engine & E, int li, const ggml_hx_mailbox * mb) {
    if (!E.pf_k || li >= E.n_primary_layers) return;
    const int nl = E.n_primary_layers;
    const int n_used = std::min<int>(mb->n_used, GGML_HX_MAX_USED);
    const int lj = li + 1;
    // previous token's ids at li+1 are still in last_ids[lj] (this token has not reached it yet)
    float score[1024];
    const int ne = std::min(E.n_expert, 1024);
    std::fill(score, score + ne, 0.0f);
    const bool use_router = E.pred_mode == 1 && lj < nl && (size_t) lj < E.rq.size() && !E.rq[lj].empty() && E.layers[lj].d.mbox;
    if (use_router) {
        static const bool batch_prefetch = getenv("LLAMA_HX_PREFETCH_BATCH") != nullptr;
        if (batch_prefetch && mb->n_tokens > 1 && E.pf_skip == 0) {
            float token_score[1024];
            int order[1024];
            const int n_embd = (int) E.layers[li].d.n_embd;
            const int nt = std::min<int>(mb->n_tokens, GGML_HX_MAX_TOKENS);
            const int nk = std::min(n_used, ne);
            for (int t = 0; t < nt; ++t) {
                hx_router_scores(E, lj, mb->x + (size_t) t * n_embd, n_embd, token_score);
                for (int e = 0; e < ne; ++e) order[e] = e;
                std::partial_sort(order, order + nk, order + ne, [&](int a, int b) {
                    return token_score[a] > token_score[b] || (token_score[a] == token_score[b] && a < b);
                });
                // Frequency first, then rank. Shared experts run first in expert-parallel mode.
                for (int rank = 0; rank < nk; ++rank) {
                    score[order[rank]] += 1.0f + float(nk - rank) / float(nk * nt);
                }
            }
        } else {
            hx_router_scores(E, lj, mb->x, (int) E.layers[li].d.n_embd, score);
        }
        E.pred_router.fetch_add(1, std::memory_order_relaxed);
    } else if (lj < nl && E.layers[lj].d.mbox) {
        const uint16_t * C = E.cooc.data() + (size_t) li * E.n_expert * E.n_expert;
        float tot = 0.0f;
        for (int k = 0; k < n_used; ++k) {
            const uint16_t * row = C + (size_t) mb->ids[k] * E.n_expert;
            float rs = 0.0f;
            for (int b = 0; b < ne; ++b) rs += row[b];
            if (rs <= 0.0f) continue;
            const float f = mb->w[k] / rs;
            for (int b = 0; b < ne; ++b) score[b] += f * row[b];
            tot += mb->w[k];
        }
        if (tot > 0.0f) for (int b = 0; b < ne; ++b) score[b] /= tot;
        for (int k = 0; k < E.last_n[lj]; ++k) score[E.last_ids[(size_t) lj * GGML_HX_MAX_USED + k]] += 0.5f * E.last_w[(size_t) lj * GGML_HX_MAX_USED + k];
    }
    // remember this layer's ids (token 0) for the co-occurrence update of li+1 and the temporal term of the next token
    for (int k = 0; k < n_used; ++k) {
        E.last_ids[(size_t) li * GGML_HX_MAX_USED + k] = mb->ids[k];
        E.last_w[(size_t) li * GGML_HX_MAX_USED + k] = mb->w[k];
    }
    E.last_n[li] = n_used;
    if (!(lj < nl && E.layers[lj].d.mbox)) return;
    int idx[1024];
    for (int b = 0; b < ne; ++b) idx[b] = b;
    // the router ranking predicts the weights too: with H2 the top pf_skip experts go to the GPU, so skip them
    const int skip = use_router ? std::min(E.pf_skip, ne) : 0;
    const int k = std::min(E.pf_k + skip, ne);
    std::partial_sort(idx, idx + k, idx + ne, [&](int x, int y) { return score[x] > score[y]; });
    int n = 0;
    for (int j = skip; j < k; ++j) if (use_router || score[idx[j]] > 0.0f) E.pf_list[n++] = idx[j];
    for (int j = 0; j < n; ++j) E.pred_ids[(size_t) lj * 32 + j] = E.pf_list[j];
    E.pred_n[lj] = n;
    E.pf_layer = lj;
    E.pf_cnt = n;
    E.pf_gen.fetch_add(1, std::memory_order_acq_rel);
}

// every thread: stream its own row slices of the predicted experts (the rows it will compute) through the cache.
// Returns early when the task for the layer arrives (leader) or a job starts (workers).
static void hx_pf_run(hx_engine & E, int ith, uint64_t job_gen_at_start) {
    const int li = E.pf_layer;
    if (li < 0) return;
    const hx_layer & L = E.layers[li];
    const int nth = E.n_threads;
    const int64_t a0 = (L.d.n_ff * ith) / nth, a1 = (L.d.n_ff * (ith + 1)) / nth;
    const int64_t b0 = (L.d.n_embd * ith) / nth, b1 = (L.d.n_embd * (ith + 1)) / nth;
    uint64_t sink = 0, bytes = 0;
    auto aborted = [&]() {
        if (ith == 0) return hx_load_acq(&L.d.mbox->task_seq) != L.done;
        return E.gen.load(std::memory_order_acquire) != job_gen_at_start;
    };
    auto touch = [&](const char * p, size_t len) -> bool {
        const size_t chunk = 32 * 1024;
        for (size_t off = 0; off < len; off += chunk) {
            const size_t end = std::min(len, off + chunk);
            switch (E.pf_hint) {
                case 1:  for (size_t o = off; o < end; o += 64) _mm_prefetch(p + o, _MM_HINT_T0);  break;
                case 2:  for (size_t o = off; o < end; o += 64) _mm_prefetch(p + o, _MM_HINT_T1);  break;
                case 3:  for (size_t o = off; o < end; o += 64) _mm_prefetch(p + o, _MM_HINT_T2);  break;
                case 4:  for (size_t o = off; o < end; o += 64) _mm_prefetch(p + o, _MM_HINT_NTA); break;
                default: for (size_t o = off; o < end; o += 64) sink += *(const volatile uint64_t *) (p + o); break;
            }
            bytes += end - off;
            if (aborted()) return false;
        }
        return true;
    };
    const int n = E.pf_cnt;
    int32_t list[32];
    for (int j = 0; j < n; ++j) list[j] = E.pf_list[j];
    bool ok = true;
    for (int j = 0; j < n && ok; ++j) {
        const int32_t e = list[j];
        ok = touch((const char *) L.d.gate + e * L.d.gate_nb2 + a0 * L.d.gate_nb1, (size_t) (a1 - a0) * L.d.gate_nb1)
          && touch((const char *) L.d.up   + e * L.d.up_nb2   + a0 * L.d.up_nb1,   (size_t) (a1 - a0) * L.d.up_nb1)
          && touch((const char *) L.d.down + e * L.d.down_nb2 + b0 * L.d.down_nb1, (size_t) (b1 - b0) * L.d.down_nb1);
    }
    if (!ok) E.pf_aborts.fetch_add(1, std::memory_order_relaxed);
    E.pf_bytes.fetch_add(bytes + (sink == 0x5a5a5a5a5a5a5a5aull), std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------------------------
// thread pool

static void hx_pin_thread(int i) {
#if defined(_WIN32)
    static const bool pin = [] { const char * e = getenv("GGML_HX_AFFINITY"); return !e || atoi(e) != 0; }();
    if (!pin) return;
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    const int n_logical = (int) si.dwNumberOfProcessors;
    const int stride = n_logical >= 2 ? 2 : 1;    // SMT siblings are adjacent on Windows
    const int cpu = (i * stride) % std::max(1, n_logical);
    if (cpu < 64) {
        SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) 1 << cpu);
    }
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
#else
    (void) i;
#endif
}

static void hx_run_job(hx_engine & E, int ith, hx_scratch & s) {
    const hx_layer & L = E.layers[E.job.layer];
    ggml_hx_mailbox * mb = L.d.mbox;
    const int n_tokens = mb->n_tokens;
    const int n_used   = mb->n_used;
    const int64_t t0 = E.profile && ith == 0 ? ggml_time_us() : 0;
    if (E.read_probe) {
        s.groups.build(mb->ids, n_used, n_tokens, n_used, L.d.n_expert);
        uint64_t sink = 0;
        const int64_t a0 = L.d.n_ff * ith / E.n_threads, a1 = L.d.n_ff * (ith + 1) / E.n_threads;
        const int64_t b0 = L.d.n_embd * ith / E.n_threads, b1 = L.d.n_embd * (ith + 1) / E.n_threads;
        for (int i = 0; i < s.groups.n; ++i) {
            const int e = s.groups.expert[i];
            const char * ptrs[] = { (const char *) L.d.gate + e * L.d.gate_nb2 + a0 * L.d.gate_nb1,
                (const char *) L.d.up + e * L.d.up_nb2 + a0 * L.d.up_nb1,
                (const char *) L.d.down + e * L.d.down_nb2 + b0 * L.d.down_nb1 };
            const size_t lens[] = { (size_t) (a1-a0)*L.d.gate_nb1, (size_t) (a1-a0)*L.d.up_nb1, (size_t) (b1-b0)*L.d.down_nb1 };
            for (int j = 0; j < 3; ++j) for (size_t k = 0; k < lens[j]; k += 64) sink += *(const volatile uint64_t *) (ptrs[j] + k);
        }
        if (sink == 0x5a5a5a5a5a5a5a5aull) E.sleeps.fetch_add(1, std::memory_order_relaxed);
        E.barrier();
    }
    const int64_t t1 = E.profile && ith == 0 ? ggml_time_us() : 0;
    const bool by_expert = E.expert_parallel && n_tokens > 1;
    const int64_t work_start = E.balance_profile && by_expert ? ggml_time_us() : 0;
    if (by_expert) {
        if (E.schedule && ith == 0) {
            thread_local bool logged = false;
            if (!logged) {
                GGML_LOG_INFO("HX expert schedule active: %s\n", E.schedule == 1 ? "dynamic" : "memory");
                logged = true;
            }
        }
        hx_compute_expert_groups(L, mb, E.expert_output.data(), ith, E.n_threads, s, E.schedule, E.next_expert);
    } else {
        hx_compute_stage_a(L, mb->x, L.d.n_embd, mb->ids, n_used, n_tokens, n_used, E.h.data(), ith, E.n_threads, s);
    }
    if (E.balance_profile && by_expert) {
        E.worker_samples[ith] = {work_start, ggml_time_us(), s.processed_groups};
    }
    E.barrier();
    const int64_t t2 = E.profile && ith == 0 ? ggml_time_us() : 0;
    if (E.balance_profile && by_expert && ith == 0) {
        int64_t first = E.worker_samples[0].start_us, last = E.worker_samples[0].end_us;
        int low = s.groups.n, high = 0;
        for (const auto & sample : E.worker_samples) {
            first = std::min(first, sample.start_us);
            last = std::max(last, sample.end_us);
            low = std::min(low, sample.groups);
            high = std::max(high, sample.groups);
            E.balance_busy_us += sample.end_us - sample.start_us;
        }
        for (const auto & sample : E.worker_samples) E.balance_wait_us += last - sample.end_us;
        E.balance_span_us += last - first;
        E.balance_min_groups += low;
        E.balance_max_groups += high;
        ++E.balance_jobs;
    }
    if (by_expert) {
        hx_reduce_expert_groups(L, mb, E.expert_output.data(), ith, E.n_threads);
    } else {
        hx_compute_stage_b(L, mb->ids, n_used, mb->w, n_used, n_tokens, n_used, E.h.data(), mb->y, L.d.n_embd, ith, E.n_threads, s);
    }
    E.barrier();
    if (E.profile && ith == 0) {
        E.profile_read_us += t1 - t0;
        E.profile_a_us += t2 - t1;
        E.profile_b_us += ggml_time_us() - t2;
        E.profile_jobs++;
        E.profile_unique += s.groups.n;
        E.profile_pairs += n_tokens * n_used;
        E.profile_bytes += s.groups.n * (L.d.gate_nb2 + L.d.up_nb2 + L.d.down_nb2);
    }
}

static void hx_worker(hx_engine * E, int ith) {
    hx_pin_thread(ith);
    hx_scratch s;
    uint64_t seen = 0, pf_seen = 0;
    auto last = clk::now();
    int spins = 0;
    while (!E->stop.load(std::memory_order_relaxed)) {
        const uint64_t g = E->gen.load(std::memory_order_acquire);
        if (g != seen) {
            seen = g;
            hx_run_job(*E, ith, s);
            last = clk::now();
            spins = 0;
            continue;
        }
        const uint64_t pg = E->pf_gen.load(std::memory_order_acquire);
        if (pg != pf_seen) {
            pf_seen = pg;
            hx_pf_run(*E, ith, g);
            continue;
        }
        HX_PAUSE();
        if (++spins >= 4096) {
            spins = 0;
            if (clk::now() - last > std::chrono::milliseconds(50)) {
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
        }
    }
}

static void hx_leader(hx_engine * E) {
    hx_pin_thread(0);
    hx_scratch s;
    const int n_layers = (int) E->layers.size();
    int next = 0;
    auto last = clk::now();
    int spins = 0;
    while (!E->stop.load(std::memory_order_relaxed)) {
        bool found = false;
        for (int i = 0; i < n_layers; ++i) {
            const int li = (next + i) % n_layers;
            hx_layer & L = E->layers[li];
            if (!L.d.mbox) continue;
            const uint64_t seq = hx_load_acq(&L.d.mbox->task_seq);
            if (seq != L.done) {
                const auto t0 = clk::now();
                hx_pf_observe(*E, li, L.d.mbox);
                E->job = { li, seq };
                // Publish the empty queue with the new job generation.
                if (E->schedule == 1) E->next_expert.store(0, std::memory_order_relaxed);
                E->gen.fetch_add(1, std::memory_order_acq_rel);
                hx_run_job(*E, 0, s);
                L.done = seq;
                hx_store_rel(&L.d.mbox->done_seq, seq);
                const auto t1 = clk::now();
                E->n_tasks.fetch_add(1, std::memory_order_relaxed);
                E->busy_ns.fetch_add((uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(), std::memory_order_relaxed);
                if (E->balance_profile && li == E->n_primary_layers - 1) {
                    GGML_LOG_INFO("HX_BALANCE n=%d jobs=%d threads=%d span_us=%lld busy_us=%lld wait_us=%lld min_groups=%d max_groups=%d\n",
                        L.d.mbox->n_tokens, E->balance_jobs, E->n_threads, (long long) E->balance_span_us,
                        (long long) E->balance_busy_us, (long long) E->balance_wait_us, E->balance_min_groups, E->balance_max_groups);
                    E->balance_jobs = E->balance_min_groups = E->balance_max_groups = 0;
                    E->balance_span_us = E->balance_busy_us = E->balance_wait_us = 0;
                }
                if (E->profile && li == E->n_primary_layers - 1) {
                    GGML_LOG_INFO("VERIFY_HX n=%d jobs=%llu unique=%llu pairs=%llu weight_bytes=%llu read_us=%.0f a_us=%.0f b_us=%.0f pf_pred=%llu pf_hits=%llu pf_bytes=%llu\n",
                        L.d.mbox->n_tokens, (unsigned long long) E->profile_jobs, (unsigned long long) E->profile_unique,
                        (unsigned long long) E->profile_pairs, (unsigned long long) E->profile_bytes,
                        E->profile_read_us, E->profile_a_us, E->profile_b_us,
                        (unsigned long long) E->pf_predicted.load(), (unsigned long long) E->pf_hits.load(), (unsigned long long) E->pf_bytes.load());
                    E->profile_jobs = E->profile_unique = E->profile_pairs = E->profile_bytes = 0;
                    E->profile_read_us = E->profile_a_us = E->profile_b_us = 0;
                }
                next = (li + 1) % n_layers;
                last = t1;
                found = true;
                if (E->pf_k && li < E->n_primary_layers) {
                    hx_pf_predict(*E, li, L.d.mbox);
                    hx_pf_run(*E, 0, E->gen.load(std::memory_order_acquire));
                    // servers are usually killed, not stopped: log the predictor statistics periodically
                    if (E->n_tasks.load(std::memory_order_relaxed) % 12288 == 0) {
                        const double pred = (double) E->pf_predicted.load(), hits = (double) E->pf_hits.load();
                        GGML_LOG_INFO("hx_engine: HX prefetch stats: tasks=%llu predictor=%s hint=%d skip=%d precision=%.3f streamed=%.1f GiB aborts=%llu\n",
                                      (unsigned long long) E->n_tasks.load(), E->pred_mode == 1 && !E->rq.empty() ? "router" : "cooc",
                                      E->pf_hint, E->pf_skip, pred > 0 ? hits / pred : 0.0, E->pf_bytes.load() / 1073741824.0,
                                      (unsigned long long) E->pf_aborts.load());
                    }
                }
                break;
            }
        }
        if (found) {
            spins = 0;
            continue;
        }
        HX_PAUSE();
        if (++spins >= 1024) {
            spins = 0;
            if (clk::now() - last > std::chrono::milliseconds(50)) {
                E->sleeps.fetch_add(1, std::memory_order_relaxed);
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
        }
    }
}

static bool hx_init_layer(hx_layer & L, const ggml_hx_layer_desc & d) {
    L.d = d;
    if (!d.gate || !d.up || !d.down || d.n_embd <= 0 || d.n_ff <= 0 || d.n_embd > GGML_HX_MAX_EMBD) {
        return false;
    }
    L.tg = ggml_get_type_traits_cpu((ggml_type) d.type_gate);
    L.tu = ggml_get_type_traits_cpu((ggml_type) d.type_up);
    L.td = ggml_get_type_traits_cpu((ggml_type) d.type_down);
    if (!L.tg->vec_dot || !L.tu->vec_dot || !L.td->vec_dot) {
        return false;
    }
    L.vt_gate = L.tg->vec_dot_type;
    L.vt_up   = L.tu->vec_dot_type;
    L.vt_down = L.td->vec_dot_type;
    L.q_gate = ggml_get_type_traits_cpu(L.vt_gate)->from_float;
    L.q_up   = ggml_get_type_traits_cpu(L.vt_up)->from_float;
    L.q_down = ggml_get_type_traits_cpu(L.vt_down)->from_float;
    if (!L.q_gate || !L.q_up || !L.q_down) {
        return false;
    }
    if (d.n_embd % ggml_blck_size(L.vt_gate) || d.n_embd % ggml_blck_size(L.vt_up) || d.n_ff % ggml_blck_size(L.vt_down)) {
        return false;
    }
    L.row_gate = ggml_row_size(L.vt_gate, d.n_embd);
    L.row_up   = ggml_row_size(L.vt_up,   d.n_embd);
    L.row_down = ggml_row_size(L.vt_down, d.n_ff);
    L.done = d.mbox ? d.mbox->done_seq : 0;
    return true;
}

// Call only with all GPU producers synchronized and no pending mailbox jobs.
static void hx_pause(hx_engine & E) {
    E.stop.store(true);
    for (auto & t : E.threads) t.join();
    E.threads.clear();
}

static void hx_resume(hx_engine & E) {
    E.gen.store(0);
    E.pf_gen.store(0);
    E.pf_layer = -1;
    E.pf_cnt = 0;
    E.bar_count.store(0);
    E.bar_gen.store(0);
    E.stop.store(false);
    E.threads.emplace_back(hx_leader, &E);
    for (int i = 1; i < E.n_threads; ++i) E.threads.emplace_back(hx_worker, &E, i);
}

} // namespace

extern "C" {

GGML_BACKEND_API void * ggml_backend_cpu_hx_start(ggml_hx_layer_desc * layers, int n_layers, int n_threads);
GGML_BACKEND_API void   ggml_backend_cpu_hx_stop(void * engine);
GGML_BACKEND_API int    ggml_backend_cpu_hx_attach(void * engine, ggml_hx_layer_desc * layers, int n_layers);
GGML_BACKEND_API void   ggml_backend_cpu_hx_detach(void * engine, int first, int n_layers);
GGML_BACKEND_API void   ggml_backend_cpu_hx_compute(void * engine, int layer,
                                                   const float * x, int64_t x_stride,
                                                   const int32_t * ids, int64_t ids_stride,
                                                   const float * w, int64_t w_stride,
                                                   int n_tokens, int n_used, float * y, int64_t y_stride);
GGML_BACKEND_API void   ggml_backend_cpu_hx_stats(void * engine, uint64_t * out);
GGML_BACKEND_API void   ggml_backend_cpu_hx_set_router(void * engine, int layer, const float * w, int64_t n_embd, int64_t n_expert);

void * ggml_backend_cpu_hx_start(ggml_hx_layer_desc * layers, int n_layers, int n_threads) {
    auto * E = new hx_engine();
    E->n_threads = std::max(1, n_threads);
    if (E->balance_profile) E->worker_samples.resize(E->n_threads);
    E->n_primary_layers = n_layers;
    E->layers.resize(n_layers);
    for (int i = 0; i < n_layers; ++i) {
        if (!layers[i].mbox) {
            E->layers[i].d = layers[i];
            continue;
        }
        if (!hx_init_layer(E->layers[i], layers[i])) {
            GGML_LOG_WARN("%s: layer %d is not supported by the HX engine; it keeps the GPU path\n", __func__, i);
            E->layers[i] = hx_layer();
            layers[i].mbox = nullptr;   // tell the caller
            continue;
        }
        E->max_ff   = std::max(E->max_ff,   layers[i].n_ff);
        E->max_embd = std::max(E->max_embd, layers[i].n_embd);
    }
    E->h.resize((size_t) GGML_HX_MAX_TOKENS * GGML_HX_MAX_USED * std::max<int64_t>(E->max_ff, 1));
    E->expert_output.resize((size_t) GGML_HX_MAX_TOKENS * GGML_HX_MAX_USED * std::max<int64_t>(E->max_embd, 1));
    hx_pf_init(*E);
    hx_resume(*E);
    GGML_LOG_INFO("%s: HX CPU expert engine started: %d layers, %d threads, kernels level %d, prefetch %d experts/layer\n",
                  __func__, n_layers, E->n_threads, hx_kernel_level(), E->pf_k);
    return E;
}

void ggml_backend_cpu_hx_stop(void * engine) {
    auto * E = (hx_engine *) engine;
    if (!E) return;
    hx_pause(*E);
    GGML_LOG_INFO("%s: HX CPU expert engine stopped: %llu tasks, %.1f ms busy\n", __func__,
                  (unsigned long long) E->n_tasks.load(), E->busy_ns.load() / 1e6);
    if (E->pf_k) {
        const double pred = (double) E->pf_predicted.load(), hits = (double) E->pf_hits.load();
        GGML_LOG_INFO("%s: HX prefetch: k=%d predictor=%s hint=%d predicted=%.0f hits=%.0f precision=%.3f streamed=%.1f GiB aborts=%llu router_predictions=%llu\n", __func__,
                      E->pf_k, E->pred_mode == 1 ? "router" : "cooc", E->pf_hint, pred, hits, pred > 0 ? hits / pred : 0.0,
                      E->pf_bytes.load() / 1073741824.0, (unsigned long long) E->pf_aborts.load(),
                      (unsigned long long) E->pred_router.load());
    }
    delete E;
}

// Attach/detach are setup operations. The caller must synchronize all contexts using this pool.
int ggml_backend_cpu_hx_attach(void * engine, ggml_hx_layer_desc * layers, int n_layers) {
    auto * E = (hx_engine *) engine;
    if (!E || !layers || n_layers <= 0) return -1;
    std::vector<hx_layer> extra(n_layers);
    for (int i = 0; i < n_layers; ++i) {
        if (!layers[i].mbox || !hx_init_layer(extra[i], layers[i])) return -1;
    }
    hx_pause(*E);
    const int first = (int) E->layers.size();
    E->layers.insert(E->layers.end(), extra.begin(), extra.end());
    for (const auto & L : extra) {
        E->max_ff = std::max(E->max_ff, L.d.n_ff);
        E->max_embd = std::max(E->max_embd, L.d.n_embd);
    }
    E->h.resize((size_t) GGML_HX_MAX_TOKENS * GGML_HX_MAX_USED * E->max_ff);
    E->expert_output.resize((size_t) GGML_HX_MAX_TOKENS * GGML_HX_MAX_USED * E->max_embd);
    hx_resume(*E);
    return first;
}

void ggml_backend_cpu_hx_detach(void * engine, int first, int n_layers) {
    auto * E = (hx_engine *) engine;
    GGML_ASSERT(E && first >= E->n_primary_layers && n_layers > 0 && first + n_layers <= (int) E->layers.size());
    hx_pause(*E);
    for (int i = first; i < first + n_layers; ++i) E->layers[i] = hx_layer();
    while ((int) E->layers.size() > E->n_primary_layers && !E->layers.back().d.mbox) E->layers.pop_back();
    hx_resume(*E);
}

// synchronous single-thread reference path (used when the op runs on the CPU backend)
void ggml_backend_cpu_hx_compute(void * engine, int layer,
                                 const float * x, int64_t x_stride,
                                 const int32_t * ids, int64_t ids_stride,
                                 const float * w, int64_t w_stride,
                                 int n_tokens, int n_used, float * y, int64_t y_stride) {
    auto * E = (hx_engine *) engine;
    GGML_ASSERT(E && layer >= 0 && layer < (int) E->layers.size());
    const hx_layer & L = E->layers[layer];
    GGML_ASSERT(L.d.mbox && n_tokens <= GGML_HX_MAX_TOKENS && n_used <= GGML_HX_MAX_USED);
    std::vector<float> h((size_t) n_tokens * n_used * L.d.n_ff);
    hx_scratch s;
    hx_compute_stage_a(L, x, x_stride, ids, ids_stride, n_tokens, n_used, h.data(), 0, 1, s);
    hx_compute_stage_b(L, ids, ids_stride, w, w_stride, n_tokens, n_used, h.data(), y, y_stride, 0, 1, s);
}

// r10: f32 router weights of one layer ([n_expert][n_embd], row-major as ffn_gate_inp) for the layer-ahead predictor
void ggml_backend_cpu_hx_set_router(void * engine, int layer, const float * w, int64_t n_embd, int64_t n_expert) {
    auto * E = (hx_engine *) engine;
    if (!E || layer < 0 || layer >= (int) E->layers.size() || !w || n_embd <= 0 || n_embd > GGML_HX_MAX_EMBD) return;
    if (E->rq.size() != E->layers.size()) { E->rq.resize(E->layers.size()); E->rs.resize(E->layers.size()); E->rsum.resize(E->layers.size()); }
    auto & q = E->rq[layer]; auto & sc = E->rs[layer]; auto & sm = E->rsum[layer];
    q.resize((size_t) n_expert * n_embd); sc.resize(n_expert); sm.resize(n_expert);
    for (int64_t e = 0; e < n_expert; ++e) {
        const float * r = w + e * n_embd;
        float amax = 0.0f;
        for (int64_t i = 0; i < n_embd; ++i) amax = std::max(amax, std::fabs(r[i]));
        const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
        int32_t sum = 0;
        for (int64_t i = 0; i < n_embd; ++i) {
            const int v = (int) std::lrintf(r[i] / s);
            q[e * n_embd + i] = (int8_t) std::max(-127, std::min(127, v));
            sum += q[e * n_embd + i];
        }
        sc[e] = s; sm[e] = sum;
    }
}

void ggml_backend_cpu_hx_stats(void * engine, uint64_t * out) {
    auto * E = (hx_engine *) engine;
    out[0] = E ? E->n_tasks.load() : 0;
    out[1] = E ? E->busy_ns.load() : 0;
    out[2] = 0;
    out[3] = E ? E->sleeps.load() : 0;
}

} // extern "C"
