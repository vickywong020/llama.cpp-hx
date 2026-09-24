# UEPT loader and application API

This local implementation follows the 2026-09-23 handoff, PR-1, PR-4 and PR-5.
It is disabled by default. All runtime binaries must be rebuilt together because
the public parameter structs have new fields.

## Parameters

- `--expert-exec legacy` is the default and keeps existing placement and execution.
- `--expert-exec uept` requires `--fit off` and one selected HIP device with UEPT support.
  Select it explicitly, for example `--device ROCm0`. Layer and none split modes are supported.
- `--expert-cache-mib -1` uses the backend's automatic budget, leaving 1 GiB of free VRAM.
  `0` creates host pointer tables with no expert cache. A positive value caps the cache in MiB.
  The value is per context and is ignored in legacy mode.
- The corresponding environment variables are `LLAMA_ARG_EXPERT_EXEC` and
  `LLAMA_ARG_EXPERT_CACHE_MIB`.

The loader forces separate routed `ffn_gate_exps.weight`, `ffn_up_exps.weight`
and `ffn_down_exps.weight` tensors into the selected `ROCm_UEPT` buffer.
Any tensor placement override matching those weights fails loading, including
an override to CPU. Supported formats are Q4_K, Q5_K, Q5_1, Q6_K, Q8_0 and
MXFP4. Fused gate/up weights and other expert formats fail explicitly.
Expert biases and scale tensors retain their normal placement.

The PLE embedding table stays on CPU, including its existing lazy mmap behavior.
A GPU override matching PLE is rejected. UEPT never substitutes another backend
or an ordinary host buffer when the mapped allocation or required API is missing.

## Public API and lifecycle

`llama_model_params.expert_exec` selects an immutable model execution mode.
`llama_context_params.expert_cache_mib` selects the initial cache budget. The
context invokes backend initialization after reserving KV and compute buffers.

`bool llama_expert_cache_init(llama_context * ctx, int64_t cache_mib)` replans
the directory and cache after auxiliary GPU allocations. It returns false on
invalid input or backend initialization failure. It synchronizes pending work,
clears cached expert residency, and invalidates backend graph captures that
referenced old directory storage. KV state and weights are unchanged.

`void llama_expert_cache_reset(llama_context * ctx)` restores host residency
at a request boundary. It queues work on the backend stream; use
`llama_synchronize()` when a completion wait is required. It does not change
KV state or weights. Both APIs are
legacy no-ops, and callers must hold exclusive access to the context. Do not
call them concurrently with encode, decode, reset, or cache initialization.

`void llama_expert_cache_set_phase(llama_context * ctx, llama_expert_phase phase)`
sets the policy for subsequent decode calls. Use `LLAMA_EXPERT_PHASE_PREFILL`
for every prompt batch, including a one-token tail, and `LLAMA_EXPERT_PHASE_DECODE`
for generation/teacher-forced decode. The default remains DECODE for existing
clients. Token count and logits flags cannot distinguish these phases. A mixed
prompt/generation batch must use PREFILL, conservatively suppressing fills for
the entire batch. The setter requires exclusive context access and records only
host intent; decode applies it before microbatch splitting and graph capture.
Reset and reinitialization preserve phase. Legacy execution is unchanged.

The backend uses one stable device phase word, changed on its execution stream
only when the phase changes. Captured resolve, commit and MMVQ fill operations
read that word dynamically. Do not replace their launch arguments with a host
boolean or toggle fill pointers after graph capture. PREFILL preserves the
directory and cache weights; it may read any existing cached experts.

The server initially uses a zero-size cache so model warmup cannot consume
memory required by its projector or draft context. After those allocations it
replans using the requested budget. At each new request it resets the target
cache only when `--parallel 1`; parallel slots share residency without resets.
Before each server batch view, any `is_prompt` token selects PREFILL; only an
entirely generated batch selects DECODE.

## Backend extension contract

The host library uses `ggml_backend_reg_get_proc_address` so its CPU/Vulkan
build does not acquire a direct HIP link dependency. A HIP backend provides:

```cpp
ggml_backend_buffer_type_t ggml_backend_cuda_uept_buffer_type(int device);
bool ggml_backend_cuda_uept_init(
    ggml_backend_t backend,
    const ggml_tensor * const * tensors,
    size_t n_tensors,
    int64_t cache_mib);
void ggml_backend_cuda_uept_reset(ggml_backend_t backend);
void ggml_backend_cuda_uept_set_phase(ggml_backend_t backend, bool allow_fill);
```

`device` is the selected device's index inside its backend registry. `tensors`
contains only this model's mapped expert weights; each backend accepts only
its own device's tensors. Directory and cache storage belong to the context,
not to a process-wide tensor registry. Initialization supports repeated calls
and must retire all graph captures referencing the old allocation before freeing it.

The existing `test-arg-parser` test covers the default mode, unknown mode,
the fit requirement, argument order, zero and positive budgets, negative budgets,
trailing characters and MiB-to-byte overflow. End-to-end placement, allocation
failure, numeric and cache parity checks require the rebuilt HIP runtime.
