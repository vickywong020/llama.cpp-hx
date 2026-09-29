# llama.cpp-hx

**本项目是 [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) 的 HX 衍生分支（fork）**，面向 Windows + AMD Radeon 的 CPU/GPU 混合 MoE 推理。它基于 llama.cpp v0.5.0，保留上游 Git 历史、MIT 许可证和作者声明。HX 的本地改动由本仓库维护，不代表上游已经合并或支持这些功能。

**This is an HX fork of llama.cpp**, based on v0.5.0, for experimental hybrid CPU/GPU MoE inference on Windows and AMD Radeon. Upstream history and licensing are preserved. See [the original llama.cpp README](README-upstream.md) for general project information.

## HX 做什么

- GPU 执行注意力、路由、共享专家及可配置数量的路由专家；CPU 专用线程池执行其余路由专家。
- CPU/GPU 通过固定页内存中的 mailbox 交接任务与结果；UEPT 管理专家在主存中的放置、GPU 访问和可选显存缓存。
- 提供 MXFP4 的 AVX-512 CPU 内核、专家并行与预取，以及可选的 MTP 草稿执行优化。
- 保留 llama.cpp 命令行工具和 `llama-server`。HX 需要显式启用；仅编译源码不会自动应用本文的运行配置。

主要验证环境为 **Windows、Ryzen 7 9800X3D、RX 9070 XT 16 GiB（gfx1201）、192 GiB RAM**，模型为 Qwen3.8-Flash-Next MXFP4，非专家 FP8 部分转为 Q8_0 存储。其他硬件、模型和操作系统需要自行验证；上游的兼容范围不等同于 HX 的实测范围。

## 分支与版本

| 分支 | 用途 | 对应冻结源码 |
|---|---|---|
| `main` | Cache3G 生产方案对应源码，优先考虑生成速度 | `v050-schedule-opt-20260925` |
| `hx-mtp1` | 可选 MTP1；增加小批量主模型直接验证路径 | `v050-graph-diagnosis-20260925` |
| `hx-v050-base` | 原始 HX + UEPT + r10 + MTP 移植提交 | `3ad2c2ca45a86abd1c535f537ad9c427f4401a46` |
| `master` | 创建 Fork 时保留的上游分支 | 不代表 HX 发布版本 |

基线为 llama.cpp **`7fe450e19305b828c199d602c23a8337aaa1f03b`（v0.5.0）**，包含 [PR #28243](https://github.com/ggml-org/llama.cpp/pull/28243) 在 `6fcaa16f` 时的 MTP 工作。来源、分支差异及 SHA-256 清单见 [来源与许可证](docs/hx/PROVENANCE.md)。这不是对上游最新版本的同步承诺。

## 开始使用

```powershell
git clone --branch main https://github.com/vickywong020/llama.cpp-hx.git
Set-Location llama.cpp-hx
```

1. 按 [Windows 构建说明](docs/hx/BUILD-WINDOWS.md) 编译主程序、CPU 和 HIP 后端。
2. 从 **[魔搭社区 ModelScope](https://modelscope.cn)** 准备模型及配套文件，核对 revision、文件大小和可用的 SHA-256。
3. 按 [运行配置](docs/hx/USAGE.md) 设置 HX 环境变量，用本地 GGUF 路径启动服务。
4. 对照 [测试结果与限制](docs/hx/BENCHMARKS.md) 判断配置是否适合自己的输入长度和硬件。

仓库不附带推理模型权重或预编译 ROCm 运行库。所有模型权重、量化分片、MTP 文件和视觉投影器统一从 ModelScope 获取；没有所需版本时停止并说明，不自动切换来源或模型。断点续传不得拼接不同来源的缓存。源码、编译工具及 Python/ROCm 依赖可使用其官方来源。上游保留文档中可能介绍其他模型下载方式；本 HX 项目的使用约定以本段为准。

## 已有测量

2026-09-25 同机串行测试，预留 262144 token 容量，固定 WikiText 输入、129 token 输出，各项取中位数：

| 方案 | 2K 输入后生成 token/s | 8K 输入后生成 token/s | 2K 首字秒 | 8K 首字秒 |
|---|---:|---:|---:|---:|
| Cache3G | 28.36 | 26.58 | 7.57 | 29.34 |
| MTP1 | 27.05 | 26.46 | 5.38 | 20.95 |
| 历史 P2A 对照 | 25.63 | 24.87 | 3.70 | 14.23 |

这是完整配置之间的测量，**不是 HX 单一算子的加速比例**。Cache3G 的生成较快，但长提示首字等待更久；8K 生成与 MTP1 的差距约 0.5%，不足以证明稳定领先。262144 是预留容量，不表示实际填满 256K 后仍有上述速度。本次源码发布没有重新运行性能测试。

## 源码地图

| 路径 | 内容 |
|---|---|
| `ggml/include/ggml-hx.h` | HX mailbox 协议与接口 |
| `ggml/src/ggml-cpu/hx-moe.cpp` | CPU 专家线程池、调度与预取 |
| `ggml/src/ggml-cpu/arch/x86/hx-mxfp4-avx512.h` | MXFP4 AVX-512 内核 |
| `ggml/src/ggml-cuda/hx.cu` | GPU 侧 mailbox 与协作 |
| `ggml/src/ggml-cuda/uept.cu` | 专家放置、缓存与 GPU 执行 |
| `src/llama-context.cpp`、`src/llama-hx.h` | HX 生命周期与模型上下文 |
| `src/models/qwen4exp.cpp`、`common/speculative.cpp` | 模型图与 MTP 逻辑 |
| `docs/development/uept-api.md` | UEPT 接口说明 |

`ggml-cuda` 中的相关源文件也用于 HIP 构建，目录名不表示只用于 NVIDIA。

## 许可与贡献

主体源码遵循 [MIT License](LICENSE)，第三方组件遵循各自许可证，见 `licenses/`、`vendor/` 及相应子目录。模型许可证独立于源码许可证。感谢 llama.cpp、ggml 和 PR #28243 的贡献者。

HX 问题和改进优先提交到本 Fork；向上游贡献时遵循上游 [CONTRIBUTING.md](CONTRIBUTING.md)。本次整理借助 Codex 完成源码核对与说明文档，推理实现保持冻结快照内容。
