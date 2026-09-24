#pragma once

#ifdef GGML_USE_HIP
#include "uept.cuh"

// Read at kernel execution time: a captured graph can alternate prefill and decode.
static __device__ __forceinline__ bool ggml_cuda_uept_can_fill(const ggml_cuda_uept_view & view) {
    return !view.phase || *view.phase != 0;
}

// Copy each quant block byte once, after its writer has used the source block.
template <int block_bytes, int lanes_per_block>
static __device__ __forceinline__ void ggml_cuda_uept_copy_quant_block(
        const void * src, char * dst, const int block, const int lane) {
    if (!dst) {
        return;
    }
    const char * src_block = static_cast<const char *>(src) + size_t(block)*block_bytes;
    char * dst_block = dst + size_t(block)*block_bytes;
    // Invariants 1 and 3: preserve raw quant bytes; disjoint byte ownership within the sole writer.
    // Byte loads also support MXFP4's 17-byte blocks without unaligned integer accesses.
#pragma unroll
    for (int i = lane; i < block_bytes; i += lanes_per_block) {
        const char value = src_block[i];
        dst_block[i] = value;
    }
}
#endif
