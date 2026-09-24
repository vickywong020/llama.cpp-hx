// hx-mxfp4-avx512.h - AVX-512 (F/BW/DQ + VNNI + VBMI) MXFP4 x Q8_0 dot products (r10 P2a).
//  v1 (hxk_*):  bit-exact with the AVX2 ggml kernel; used by ggml_vec_dot_mxfp4_q8_0 (legacy and HX per-row paths).
//  v3 (hxk3_*): W4A8 layout-optimized kernel, ~2.4x the AVX2 kernel single-thread, not bit-exact (validated
//               against a long-double reference); used by the HX engine by default (GGML_HX_KERNELS=2).
//
// Bit-exact with the AVX2 path of ggml_vec_dot_mxfp4_q8_0:
//  * integer part: the AVX2 kernel forms |w| * sign(w)*y with maddubs and sums byte quads with madd(ones), giving int32
//    lane i = bytes 4i..4i+3 of a 32-element block. vpdpbusd(|w|, sign(w)*y) yields the same int32 per lane
//    (|w| <= 12, |y| <= 127: no saturation in either form).
//  * float part: AVX2 keeps accum1 (even blocks) and accum2 (odd blocks) with one FMA per block; here lanes 0-7 of one
//    zmm are accum1 and lanes 8-15 are accum2, updated with the same FMA per block in the same order, and the final
//    reduction is hsum_float_8(accum1 + accum2) with the same instruction sequence.
//  Only the block pairs are handled here; the caller adds the scalar tail for an odd trailing block exactly as before.
//
// Requires: block_mxfp4, block_q8_0, GGML_CPU_FP16_TO_FP32, GGML_CPU_E8M0_TO_FP32_HALF in scope.
#pragma once

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__) && defined(__AVX512VNNI__) && defined(__AVX512VBMI__)
#define GGML_HX_MXFP4_AVX512 1

#include <immintrin.h>
#include <stddef.h>

static inline float hxk_mx_hsum_float_8(const __m256 x) {   // same sequence as hsum_float_8 in arch/x86/quants.c
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

static inline float hxk_mx_finish(const __m512 acc) {
    return hxk_mx_hsum_float_8(_mm256_add_ps(_mm512_castps512_ps256(acc), _mm512_extractf32x8_ps(acc, 1)));
}

// |w| table (codes 8..15 are the negatives of codes 0..7); per 128-bit lane nibble shift: lanes 0/2 take the low
// nibbles (elements 0..15 of a block), lanes 1/3 the high nibbles (elements 16..31).
#define HXK_MX_CONSTS                                                                                           \
    const __m512i hxk_lut  = _mm512_broadcast_i32x4(_mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, 1, 2, 3, 4, 6, 8, 12)); \
    const __m512i hxk_m0f  = _mm512_set1_epi8(0x0f);                                                            \
    const __m512i hxk_m08  = _mm512_set1_epi8(0x08);                                                            \
    const __m512i hxk_perm = _mm512_setr_epi64(0, 1, 0, 1, 2, 3, 2, 3);                                         \
    const __m512i hxk_shv  = _mm512_setr_epi64(0, 0, 4, 4, 0, 0, 4, 4);                                         \
    const __m512i hxk_zero = _mm512_setzero_si512();

// blocks ib, ib+1 of row X against activation vector YV (qs of y[ib], y[ib+1]) and activation scales YD0, YD1
#define HXK_MX_PAIR(X, IB, YV, YD0, YD1, ACC)                                                                   \
    {                                                                                                           \
        const __m256i xx_ = _mm256_inserti128_si256(                                                            \
            _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *) (X)[(IB)].qs)),                            \
            _mm_loadu_si128((const __m128i *) (X)[(IB) + 1].qs), 1);                                            \
        const __m512i q_   = _mm512_permutexvar_epi64(hxk_perm, _mm512_castsi256_si512(xx_));                   \
        const __m512i nib_ = _mm512_and_si512(_mm512_srlv_epi64(q_, hxk_shv), hxk_m0f);                         \
        const __m512i aw_  = _mm512_shuffle_epi8(hxk_lut, nib_);                                                \
        const __m512i ys_  = _mm512_mask_sub_epi8(YV, _mm512_test_epi8_mask(nib_, hxk_m08), hxk_zero, YV);      \
        const __m512i p_   = _mm512_dpbusd_epi32(hxk_zero, aw_, ys_);                                           \
        const __m512  sc_  = _mm512_insertf32x8(_mm512_set1_ps((YD0)*GGML_CPU_E8M0_TO_FP32_HALF((X)[(IB)].e)),   \
                                                _mm256_set1_ps((YD1)*GGML_CPU_E8M0_TO_FP32_HALF((X)[(IB) + 1].e)), 1); \
        ACC = _mm512_fmadd_ps(sc_, _mm512_cvtepi32_ps(p_), ACC);                                                \
    }

#define HXK_MX_LOADY(Y, IB)                                                                                     \
    _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *) (Y)[(IB)].qs)),              \
                       _mm256_loadu_si256((const __m256i *) (Y)[(IB) + 1].qs), 1)

// Sum over the block pairs [0, nb & ~1) of one row. The caller handles an odd trailing block.
static inline float hxk_mxfp4_q8_0_pairs(int nb, const block_mxfp4 * x, const block_q8_0 * y) {
    HXK_MX_CONSTS
    __m512 acc = _mm512_setzero_ps();
    for (int ib = 0; ib + 1 < nb; ib += 2) {
        const __m512i yv = HXK_MX_LOADY(y, ib);
        const float yd0 = GGML_CPU_FP16_TO_FP32(y[ib].d), yd1 = GGML_CPU_FP16_TO_FP32(y[ib + 1].d);
        HXK_MX_PAIR(x, ib, yv, yd0, yd1, acc)
    }
    return hxk_mx_finish(acc);
}

// Four rows (row i at x + i*row_stride bytes) against one activation row; nb must be even. Each result is
// bit-identical to the single-row kernel; the four independent FMA chains hide the FMA latency.
static inline void hxk_mxfp4_q8_0_r4(int nb, float * s, const void * vx, size_t row_stride, const block_q8_0 * y) {
    const block_mxfp4 * x0 = (const block_mxfp4 *) vx;
    const block_mxfp4 * x1 = (const block_mxfp4 *) ((const char *) vx + 1 * row_stride);
    const block_mxfp4 * x2 = (const block_mxfp4 *) ((const char *) vx + 2 * row_stride);
    const block_mxfp4 * x3 = (const block_mxfp4 *) ((const char *) vx + 3 * row_stride);
    HXK_MX_CONSTS
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps(), a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
    for (int ib = 0; ib + 1 < nb; ib += 2) {
        const __m512i yv = HXK_MX_LOADY(y, ib);
        const float yd0 = GGML_CPU_FP16_TO_FP32(y[ib].d), yd1 = GGML_CPU_FP16_TO_FP32(y[ib + 1].d);
        HXK_MX_PAIR(x0, ib, yv, yd0, yd1, a0)
        HXK_MX_PAIR(x1, ib, yv, yd0, yd1, a1)
        HXK_MX_PAIR(x2, ib, yv, yd0, yd1, a2)
        HXK_MX_PAIR(x3, ib, yv, yd0, yd1, a3)
    }
    s[0] = hxk_mx_finish(a0);
    s[1] = hxk_mx_finish(a1);
    s[2] = hxk_mx_finish(a2);
    s[3] = hxk_mx_finish(a3);
}

// ------------------------------------------------------------------------------------------------------------------
// v3 (default in the HX engine): W4A8 MXFP4 x Q8_0 fused dot product for Zen 5 (AVX-512 BW/VBMI/VNNI), not bound to the AVX2 summation order.
//
// Math (per row r, blocks b of 32 weights):
//   y_r = sum_b  s_{r,b} * P_{r,b}                     s_{r,b} = d_b * 2^(e_{r,b}-128)   (E8M0 "half" scale x Q8_0 scale)
//   P_{r,b} = sum_{i<32} v(c_{r,b,i}) * q_{b,i}        v = 2*E2M1 in {0,+-1,+-2,+-3,+-4,+-6,+-8,+-12}, q = int8 activation
// The inner sum is exact int32 (|P| <= 32*12*127). The weight scale is a power of two; the activation scale d_b is
// per 32-element block (Q8_0), so the scale cannot leave the block sum: one fp32 multiply-add per block is the minimum.
//
// Layout choices that make it fast:
//  * 4 blocks per zmm: one 64-byte load + a 4-byte masked load cover 4 blocks (17 B each); one vpermt2b gathers the
//    4 x 16 nibble bytes into the four 128-bit lanes. Low nibbles -> elements 0..15, high nibbles -> 16..31.
//  * E2M1 -> int8 via vpshufb (16-entry |v| table per lane); sign moved onto the activation (masked vpsubb), so
//    vpdpbusd (u8 x s8) is exact. Both halves accumulate into the same int32 lanes: lane b holds 4 partial sums of block b.
//  * Activations are pre-laid-out ONCE per token (shared by every row of every expert): lo/hi 16-byte halves of 4
//    blocks per zmm, and d_b broadcast to its lane as fp32.
//  * E8M0 bytes are pulled from the same load with vpermb; e<<23 is the fp32 2^(e-127) (the prepared activation scale
//    carries the factor 1/2), so the scale costs one shift, then one vmulps, one vcvtdq2ps and one vfmadd per 4 blocks.
//    e = 0 (a 2^-128 block scale, i.e. an all-but-zero block) becomes 0 instead of the denormal 2^-128*d: |error| < 1e-33.
//  * 4 rows per call share the activation registers; independent FMA chains hide latency.
// Numerics: same products and the same exact int32 block sums as ggml; only the fp32 accumulation order differs
// (not bit-exact with the AVX2 kernel; validated against a double-precision reference).

// prepared activation for nb blocks (nb % 4 == 0): per group of 4 blocks g: lo[g], hi[g] (64 B each), d[g] (16 fp32)
typedef struct {
    int nb;
    __m512i * lo;
    __m512i * hi;
    __m512  * d;
} hxk3_act;

static const uint8_t hxk3_qidx[64] = {
     1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67 };   // 64..67 come from the second source
static const uint8_t hxk3_eidx[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
    34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34,
    51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51 };

// y: nb q8_0 blocks -> prepared layout (buffers must hold nb/4 entries each, 64-byte aligned)
static inline void hxk3_prepare(int nb, const void * vy, hxk3_act * a) {
    const block_q8_0 * y = (const block_q8_0 *) vy;
    a->nb = nb;
    for (int g = 0; g < nb / 4; ++g) {
        const block_q8_0 * b = y + 4 * g;
        __m512i lo = _mm512_castsi128_si512(_mm_loadu_si128((const __m128i *) b[0].qs));
        lo = _mm512_inserti32x4(lo, _mm_loadu_si128((const __m128i *) b[1].qs), 1);
        lo = _mm512_inserti32x4(lo, _mm_loadu_si128((const __m128i *) b[2].qs), 2);
        lo = _mm512_inserti32x4(lo, _mm_loadu_si128((const __m128i *) b[3].qs), 3);
        __m512i hi = _mm512_castsi128_si512(_mm_loadu_si128((const __m128i *) (b[0].qs + 16)));
        hi = _mm512_inserti32x4(hi, _mm_loadu_si128((const __m128i *) (b[1].qs + 16)), 1);
        hi = _mm512_inserti32x4(hi, _mm_loadu_si128((const __m128i *) (b[2].qs + 16)), 2);
        hi = _mm512_inserti32x4(hi, _mm_loadu_si128((const __m128i *) (b[3].qs + 16)), 3);
        a->lo[g] = lo;
        a->hi[g] = hi;
        const __m128 d4 = _mm_setr_ps(GGML_CPU_FP16_TO_FP32(b[0].d), GGML_CPU_FP16_TO_FP32(b[1].d),
                                      GGML_CPU_FP16_TO_FP32(b[2].d), GGML_CPU_FP16_TO_FP32(b[3].d));
        a->d[g] = _mm512_mul_ps(_mm512_set1_ps(0.5f),
                  _mm512_permutexvar_ps(_mm512_setr_epi32(0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3), _mm512_castps128_ps512(d4)));
    }
}

#define HXK3_CONSTS                                                                                           \
    const __m512i lut   = _mm512_broadcast_i32x4(_mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, 1, 2, 3, 4, 6, 8, 12)); \
    const __m512i m0f   = _mm512_set1_epi8(0x0f);                                                             \
    const __m512i m08   = _mm512_set1_epi8(0x08);                                                             \
    const __m512i zero  = _mm512_setzero_si512();                                                             \
    const __m512i qidx  = _mm512_loadu_si512(hxk3_qidx);                                                      \
    const __m512i eidx  = _mm512_loadu_si512(hxk3_eidx);                                                      \
    const __mmask64 byte0 = 0x1111111111111111ull;

// 4 blocks (group G) of row X: ACC += s * P
#define HXK3_GROUP(X, G, A, ACC)                                                                              \
    {                                                                                                         \
        const char * p_ = (const char *) ((X) + 4 * (G));                                                     \
        const __m512i r0_ = _mm512_loadu_si512(p_);                                                           \
        const __m512i r1_ = _mm512_maskz_loadu_epi8(0xFull, p_ + 64);                                         \
        const __m512i q_  = _mm512_permutex2var_epi8(r0_, qidx, r1_);                                         \
        const __m512i nl_ = _mm512_and_si512(q_, m0f);                                                        \
        const __m512i nh_ = _mm512_and_si512(_mm512_srli_epi16(q_, 4), m0f);                                  \
        const __m512i yl_ = _mm512_mask_sub_epi8((A)->lo[G], _mm512_test_epi8_mask(nl_, m08), zero, (A)->lo[G]); \
        const __m512i yh_ = _mm512_mask_sub_epi8((A)->hi[G], _mm512_test_epi8_mask(nh_, m08), zero, (A)->hi[G]); \
        __m512i s_ = _mm512_dpbusd_epi32(zero, _mm512_shuffle_epi8(lut, nl_), yl_);                           \
        s_ = _mm512_dpbusd_epi32(s_, _mm512_shuffle_epi8(lut, nh_), yh_);                                     \
        const __m512i ed_ = _mm512_maskz_permutexvar_epi8(byte0, eidx, r0_);                                  \
        const __m512i eb_ = _mm512_slli_epi32(ed_, 23);                                                        \
        ACC = _mm512_fmadd_ps(_mm512_mul_ps((A)->d[G], _mm512_castsi512_ps(eb_)), _mm512_cvtepi32_ps(s_), ACC); \
    }

static inline void hxk3_dot_mxfp4_r1(float * s, const void * vx, const hxk3_act * a) {
    const block_mxfp4 * x = (const block_mxfp4 *) vx;
    HXK3_CONSTS
    __m512 acc = _mm512_setzero_ps();
    for (int g = 0; g < a->nb / 4; ++g) {
        HXK3_GROUP(x, g, a, acc)
    }
    *s = _mm512_reduce_add_ps(acc);
}

static inline void hxk3_dot_mxfp4_r4(float * s, const void * vx, size_t row_stride, const hxk3_act * a) {
    const block_mxfp4 * x0 = (const block_mxfp4 *) vx;
    const block_mxfp4 * x1 = (const block_mxfp4 *) ((const char *) vx + 1 * row_stride);
    const block_mxfp4 * x2 = (const block_mxfp4 *) ((const char *) vx + 2 * row_stride);
    const block_mxfp4 * x3 = (const block_mxfp4 *) ((const char *) vx + 3 * row_stride);
    HXK3_CONSTS
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps(), a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
    for (int g = 0; g < a->nb / 4; ++g) {
        HXK3_GROUP(x0, g, a, a0) HXK3_GROUP(x1, g, a, a1) HXK3_GROUP(x2, g, a, a2) HXK3_GROUP(x3, g, a, a3)
    }
    s[0] = _mm512_reduce_add_ps(a0); s[1] = _mm512_reduce_add_ps(a1);
    s[2] = _mm512_reduce_add_ps(a2); s[3] = _mm512_reduce_add_ps(a3);
}

// ---- multi-token form (speculative verification / small batches): 4 rows x TN tokens per call.
// The weight decode (vpermt2b, nibbles, |v| table, sign masks, E8M0 scale) is done once per 4 blocks and reused by
// every token; per token only the sign transfer, 2 x vpdpbusd, vmulps, vcvtdq2ps and vfmadd remain.
// For each token the instruction sequence per lane is the same as hxk3_dot_mxfp4_r4/_r1, so results are bit-identical
// to calling the single-token kernel once per token.
#define HXK3_DEFINE_MULTI(TN)                                                                                  \
static inline void hxk3_dot_mxfp4_r4_t##TN(float * const * o, const void * vx, size_t row_stride,              \
                                          const hxk3_act * const * pa) {                                       \
    const block_mxfp4 * xr_[4];                                                                                \
    for (int i = 0; i < 4; ++i) xr_[i] = (const block_mxfp4 *) ((const char *) vx + (size_t) i * row_stride);  \
    HXK3_CONSTS                                                                                                \
    __m512 acc[TN][4];                                                                                         \
    for (int t = 0; t < TN; ++t) for (int i = 0; i < 4; ++i) acc[t][i] = _mm512_setzero_ps();                  \
    const int ng = pa[0]->nb / 4;                                                                              \
    for (int g = 0; g < ng; ++g) {                                                                             \
        for (int i = 0; i < 4; ++i) {                                                                          \
            const char * p_ = (const char *) (xr_[i] + 4 * g);                                                 \
            const __m512i r0_ = _mm512_loadu_si512(p_);                                                        \
            const __m512i r1_ = _mm512_maskz_loadu_epi8(0xFull, p_ + 64);                                      \
            const __m512i q_  = _mm512_permutex2var_epi8(r0_, qidx, r1_);                                      \
            const __m512i nl_ = _mm512_and_si512(q_, m0f);                                                     \
            const __m512i nh_ = _mm512_and_si512(_mm512_srli_epi16(q_, 4), m0f);                               \
            const __mmask64 ml_ = _mm512_test_epi8_mask(nl_, m08);                                             \
            const __mmask64 mh_ = _mm512_test_epi8_mask(nh_, m08);                                             \
            const __m512i awl_ = _mm512_shuffle_epi8(lut, nl_);                                                \
            const __m512i awh_ = _mm512_shuffle_epi8(lut, nh_);                                                \
            const __m512i ed_ = _mm512_maskz_permutexvar_epi8(byte0, eidx, r0_);                               \
            const __m512  eb_ = _mm512_castsi512_ps(_mm512_slli_epi32(ed_, 23));                               \
            for (int t = 0; t < TN; ++t) {                                                                     \
                const __m512i yl_ = _mm512_mask_sub_epi8(pa[t]->lo[g], ml_, zero, pa[t]->lo[g]);               \
                const __m512i yh_ = _mm512_mask_sub_epi8(pa[t]->hi[g], mh_, zero, pa[t]->hi[g]);               \
                __m512i s_ = _mm512_dpbusd_epi32(zero, awl_, yl_);                                             \
                s_ = _mm512_dpbusd_epi32(s_, awh_, yh_);                                                       \
                acc[t][i] = _mm512_fmadd_ps(_mm512_mul_ps(pa[t]->d[g], eb_), _mm512_cvtepi32_ps(s_), acc[t][i]); \
            }                                                                                                  \
        }                                                                                                      \
    }                                                                                                          \
    for (int t = 0; t < TN; ++t) for (int i = 0; i < 4; ++i) o[t][i] = _mm512_reduce_add_ps(acc[t][i]);       \
}
HXK3_DEFINE_MULTI(1)
HXK3_DEFINE_MULTI(2)
HXK3_DEFINE_MULTI(3)
HXK3_DEFINE_MULTI(4)

// o[t] points to 4 consecutive outputs (rows r..r+3) of token t; tn in 1..4
static inline void hxk3_dot_mxfp4_r4_multi(int tn, float * const * o, const void * vx, size_t row_stride,
                                           const hxk3_act * const * pa) {
    switch (tn) {
        case 1: hxk3_dot_mxfp4_r4_t1(o, vx, row_stride, pa); break;
        case 2: hxk3_dot_mxfp4_r4_t2(o, vx, row_stride, pa); break;
        case 3: hxk3_dot_mxfp4_r4_t3(o, vx, row_stride, pa); break;
        default: hxk3_dot_mxfp4_r4_t4(o, vx, row_stride, pa); break;
    }
}

#endif
