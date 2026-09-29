# llama.cpp-hx

**English** | [简体中文](README.zh-CN.md)

**This project is an HX fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** for experimental hybrid CPU/GPU Mixture-of-Experts (MoE) inference on Windows and AMD Radeon. It is based on llama.cpp v0.5.0 and preserves upstream Git history, the MIT license, and author credits. HX changes are maintained in this repository; they do not imply upstream adoption or support.

See [the original llama.cpp README](README-upstream.md) for general upstream project information.

## Features

HX targets local MoE inference where **expert weights exceed GPU memory capacity, but sufficient system RAM is available**. The CPU actively computes experts while the GPU handles attention and other operations. A dedicated coordination path reduces handoff overhead between them. Qwen3.8-Flash-Next was tested on a machine with **16 GiB of VRAM and 192 GiB of system RAM**. The 16 GiB figure describes the GPU, not the model's total memory requirement.

| Feature | Implementation | Purpose and requirements |
|---|---|---|
| Hybrid CPU/GPU expert execution | A dedicated CPU thread pool computes routed experts. The GPU handles attention, routing, shared experts, and optionally some routed experts. | Keeps large expert weights in system RAM while using both CPU and GPU compute. |
| Fewer per-layer host synchronizations | GPU and CPU exchange tasks and results through mailboxes in pinned, device-mapped host memory. | Reduces handoff overhead in the historical legacy CPU expert offload path. HX can work with HIP Graphs. |
| Dedicated CPU kernels and scheduling | MXFP4 AVX-512 kernels, a separate expert thread pool, expert parallelism, and optional dynamic scheduling. | Improves CPU expert execution efficiency; gains depend on CPU instruction support and memory bandwidth. |
| Expert prefetching and VRAM caching | V-Cache prefetching; UEPT manages experts in host memory, GPU access, and configurable VRAM caching. | Cache3G uses a 3072 MiB expert cache and assigns up to 4 routed experts per token to the GPU. |
| Optional MTP1 | One draft token, a GPU-resident draft model, a separate physical batch size, K/V-only catch-up, and direct execution of small target-model verification batches. | Provides an alternative generation profile. Gains depend on draft acceptance and VRAM pressure. The default published profile is Cache3G without MTP. |
| Existing llama.cpp tools and interfaces | Retains the CLI, `llama-server`, GGUF loading, and chat interfaces. | Local production checks covered text, images, streaming, grammar constraints, and tool arguments. Compatible models and templates are required. |

HX extends expert execution and scheduling. Enabling it requires environment variables and runtime arguments, not just a different model file. See the [runtime guide](docs/hx/USAGE.md) for configuration details. The detailed HX guides linked from this README are currently in Chinese.

## Test environment

| Item | Tested configuration |
|---|---|
| Operating system | Windows x64 |
| CPU | AMD Ryzen 7 9800X3D, 8 cores / 16 threads |
| GPU | AMD Radeon RX 9070 XT, 16 GiB, `gfx1201` |
| System RAM | 192 GiB |
| Primary model | Qwen3.8-Flash-Next, with MXFP4 experts and non-expert FP8 tensors converted to Q8_0 storage |
| Other historical tests | HX / legacy comparisons using a Q5-quantized model |
| Server configuration | One model, `parallel=1`, 262144-token context capacity, vision projector loaded |

Other hardware, models, and operating systems require separate validation. Upstream llama.cpp compatibility does not establish HX compatibility. Historical baseline comparisons and later profile tests are presented separately to avoid combining performance figures from different versions.

## Test results

### Historical HX versus legacy comparison

The r9 / H1 tests on 2026-09-23 compared the legacy path with HX using 8 CPU threads on the same machine: Q8_0 KV cache, physical batch size 1024, 262144-token context capacity, and mmproj loaded. Each 2K-input test ran 3 times with 128 decode steps per run; the table reports medians. The 32K results came from separate long-input measurements. All speeds below are generation tokens per second:

| Model / actual input | Legacy | HX (8 threads) | Relative gain |
|---|---:|---:|---:|
| MXFP4 / 2K | 15.46 | 22.71 | About **47%** |
| Q5 / 2K | 13.97 | 18.78 | About **34%** |
| MXFP4 / 32K | 14.41 | 19.33 | About **34%** |
| Q5 / 32K | 12.45 | 16.11 | About **29%** |

These measurements show early HX gains over **the local legacy CPU expert offload path used at that time**. They are not a comparison with the latest upstream release or a speedup claim for current Cache3G. Outputs across execution paths were not identical token for token. Historical Q5 HX did not meet the original top-1 agreement criterion in a check of 832 fixed positions; see the [historical tests and numerical checks](docs/hx/BENCHMARKS.md). Faster generation does not establish equivalent model quality.

### Current Cache3G / MTP1 profile retest

Tests on 2026-09-25 ran serially on the same machine, using fixed WikiText inputs, 129 output tokens, greedy decoding, seed 42, and no prompt-cache reuse. The 2K-input tests ran 3 times and the 8K tests ran 2 times, following a warm-up with 512 input tokens and 32 output tokens. Values below are medians:

| Profile | Actual input tokens | Prompt processing tokens/s | Generation tokens/s | First text, seconds | Full output, seconds |
|---|---:|---:|---:|---:|---:|
| **Cache3G** | 2048 | 271.29 | **28.36** | 7.57 | 12.09 |
| MTP1 | 2048 | 380.65 | 27.05 | 5.38 | 10.11 |
| Historical P2A baseline | 2048 | 555.81 | 25.63 | 3.70 | 8.72 |
| **Cache3G** | 8192 | 279.24 | **26.58** | 29.34 | 34.16 |
| MTP1 | 8192 | 391.29 | 26.46 | 20.95 | 25.79 |
| Historical P2A baseline | 8192 | 576.39 | 24.87 | 14.23 | 19.38 |

| Profile | Expert cache | GPU routed experts per token | Target KV / physical batch | Draft |
|---|---|---:|---|---|
| Cache3G | 3072 MiB | Up to 4 | Q4_0 / 256 | None |
| MTP1 | 0 | 0 | Q4_0 / 512 | One draft token, GPU-resident model, Q4_0 KV / batch 128 |
| P2A | 0 | 0 | Q8_0 / 1024 | None |

**Profile selection:** The local deployment prioritizes generation speed, so `main` corresponds to Cache3G. In this test, it generated about 4.8% faster than MTP1 after a 2K input, but processed prompts more slowly and took longer to produce the first text. The generation difference after an 8K input was only about 0.5%, which does not establish a consistent lead. MTP1 was faster on some code and tool-format text workloads. P2A produced the first text sooner, but uses a historical source baseline; it is not equivalent to changing a few arguments on current `main`.

Time to first text was measured through the local streaming `/completion` API. It includes prompt processing but excludes model loading. Generation speed comes from server timings. These were fixed-text continuation tests, excluding chat templates, reasoning behavior, client rendering, and multi-user queuing. Cache size, KV type, batch size, and execution path differ between the profiles, and their output text was not identical. The differences cannot all be attributed to a single optimization.

### Long inputs and VRAM usage

These results come from separate production validations for each profile, not one controlled comparison. All profiles reserved 262144 tokens of context capacity and generated 65 output tokens for the long-input tests:

| Profile | Actual input tokens | Prompt processing tokens/s | Generation tokens/s | Process dedicated VRAM, GiB | Total GPU dedicated VRAM, GiB |
|---|---:|---:|---:|---:|---:|
| Cache3G | 32768 | 273.15 | 23.13 | 13.57 | 15.12 |
| MTP1 | 32768 | 412.56 | 22.96 | 13.90 | 15.48 |
| MTP1 | 131072 | 343.96 | 15.86 | 14.02 | 15.60 |

Short-chat recovery checks passed after long-input requests. VRAM figures are Windows snapshots taken after requests, not peak measurements. Total GPU usage includes the desktop and other consumers. Shared memory includes experts intentionally placed in system RAM, so it should not all be classified as VRAM spillover.

### Functional and consistency checks

| Check | Historical production validation result |
|---|---|
| Arithmetic in Chinese, chart reading, streamed output, and completion markers | Passed |
| API authentication, model aliases, grammar boundaries, and nested tool arguments | Passed |
| 13 fixed prompts | Cache3G matched a previous run of the same Cache3G configuration token for token in 13/13 cases; MTP1 matched the earlier optimized MTP1 profile in 13/13 cases. |
| Input boundaries and shared-prefix replay | Passed at 1 / 2 / 127 / 128 / 129 / 2047 / 2048 / 2049 input tokens, including replay after prefix edits. |

The 13/13 results are regression checks within each profile. They do not mean Cache3G, MTP1, and legacy produce identical outputs, and they are not a comprehensive quality evaluation. Authentication and template checks also depend on the local production launcher's configuration.

**Validated scope:** The Cache3G retest reached an actual input length of 32K; separate MTP1 validation reached 128K. A 256K context capacity was successfully configured and started, but full 256K inputs, multi-user concurrency, and all-day stability were not tested. This README update summarizes existing records; no new compilation or benchmark run was performed. See the [benchmark notes](docs/hx/BENCHMARKS.md) for methods, sources, and limitations.

## Branches and versions

| Branch | Purpose | Frozen source snapshot |
|---|---|---|
| `main` | Source for the Cache3G production profile, prioritizing generation speed | `v050-schedule-opt-20260925` |
| `hx-mtp1` | Optional MTP1 profile, adding direct execution of small target-model verification batches | `v050-graph-diagnosis-20260925` |
| `hx-v050-base` | Original HX + UEPT + r10 + MTP port | `3ad2c2ca45a86abd1c535f537ad9c427f4401a46` |
| `master` | Upstream branch retained when the fork was created | Not the HX release branch |

The baseline is llama.cpp **`7fe450e19305b828c199d602c23a8337aaa1f03b` (v0.5.0)**, with MTP work from [PR #28243](https://github.com/ggml-org/llama.cpp/pull/28243) at `6fcaa16f`. See [provenance and licensing](docs/hx/PROVENANCE.md) for source attribution, branch differences, and SHA-256 manifests. This does not imply synchronization with the latest upstream version.

## Getting started

```powershell
git clone --branch main https://github.com/vickywong020/llama.cpp-hx.git
Set-Location llama.cpp-hx
```

1. Follow the [Windows build guide](docs/hx/BUILD-WINDOWS.md) to compile the applications, CPU backend, and HIP backend.
2. Obtain the model and companion files from **[ModelScope](https://modelscope.cn)**. Verify the revision, file sizes, and available SHA-256 checksums.
3. Follow the [runtime guide](docs/hx/USAGE.md) to configure HX environment variables and start the server with local GGUF paths.
4. Review the [test results and limitations](docs/hx/BENCHMARKS.md) to choose a profile for your hardware and input lengths.

This repository does not include inference model weights or prebuilt ROCm runtime libraries. All model weights, quantized shards, MTP files, and vision projectors must come from ModelScope. If the required version is unavailable, stop and report it instead of automatically switching sources or models. Resumable downloads must not combine partial caches from different sources. Source code, build tools, and Python/ROCm dependencies may use their official distribution sources. Preserved upstream documentation may describe other model download methods; the HX policy in this paragraph takes precedence for this project.

## Source map

| Path | Contents |
|---|---|
| `ggml/include/ggml-hx.h` | HX mailbox protocol and interfaces |
| `ggml/src/ggml-cpu/hx-moe.cpp` | CPU expert thread pool, scheduling, and prefetching |
| `ggml/src/ggml-cpu/arch/x86/hx-mxfp4-avx512.h` | MXFP4 AVX-512 kernels |
| `ggml/src/ggml-cuda/hx.cu` | GPU-side mailboxes and coordination |
| `ggml/src/ggml-cuda/uept.cu` | Expert placement, caching, and GPU execution |
| `src/llama-context.cpp`, `src/llama-hx.h` | HX lifecycle and model contexts |
| `src/models/qwen4exp.cpp`, `common/speculative.cpp` | Model graphs and MTP logic |
| `docs/development/uept-api.md` | UEPT interface documentation |

Relevant sources in `ggml-cuda` are also used for HIP builds. The directory name does not mean they are limited to NVIDIA GPUs.

## License and contributions

The main source code is covered by the [MIT License](LICENSE). Third-party components retain their respective licenses; see `licenses/`, `vendor/`, and the relevant component directories. Model licenses are separate from the source-code license. Thanks to the llama.cpp, ggml, and PR #28243 contributors.

Please report HX issues and propose HX improvements in this fork first. Contributions to upstream must follow its [CONTRIBUTING.md](CONTRIBUTING.md). Codex assisted with source verification and documentation for this publication. Inference implementation files retain the contents of the frozen source snapshots.
