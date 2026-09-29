# llama.cpp-hx

**本项目是 [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) 的 HX 衍生分支（fork）**，面向 Windows + AMD Radeon 的 CPU/GPU 混合 MoE 推理。它基于 llama.cpp v0.5.0，保留上游 Git 历史、MIT 许可证和作者声明。HX 的本地改动由本仓库维护，不代表上游已经合并或支持这些功能。

**This is an HX fork of llama.cpp**, based on v0.5.0, for experimental hybrid CPU/GPU MoE inference on Windows and AMD Radeon. Upstream history and licensing are preserved. See [the original llama.cpp README](README-upstream.md) for general project information.

## 项目特点

HX 面向**模型专家权重大于显存容量、但系统内存充足**的本地 MoE 推理场景。它让 CPU 持续参与专家计算，GPU 负责注意力等计算，并通过专用协作路径减少两者交接时的等待。本项目在 **16 GiB 显存 + 192 GiB 系统内存**的机器上测试了 Qwen3.8-Flash-Next；16 GiB 是显卡容量，不代表模型只需要 16 GiB 总内存。

| 特点 | 具体实现 | 作用与适用条件 |
|---|---|---|
| CPU/GPU 混合专家执行 | CPU 专用线程池计算路由专家；GPU 执行注意力、路由、共享专家，并可分担部分路由专家 | 利用系统内存保存大模型专家，同时使用 CPU 与 GPU 算力 |
| 减少逐层主机同步 | GPU 与 CPU 在固定页、设备可映射内存中通过 mailbox 投递任务和回传结果 | 改进历史 legacy CPU 专家卸载路径中的交接开销；HX 路径可与 HIP Graph 配合 |
| 专用 CPU 内核与调度 | MXFP4 AVX-512 内核、独立专家线程池、专家并行和可选动态调度 | 提高 CPU 专家执行效率；实际收益取决于 CPU 指令集和内存带宽 |
| 专家预取与显存缓存 | V-Cache 预取；UEPT 管理主存专家、GPU 访问和可配置显存缓存 | Cache3G 使用 3072 MiB 专家缓存，并让 GPU 每 token 最多分担 4 个路由专家 |
| 可选 MTP1 | 单草稿、草稿 GPU 驻留、独立物理批次、仅 K/V 补齐及小批量主模型直接验证 | 提供另一种生成方案；接受率与显存压力决定收益，默认发布方案为无 MTP 的 Cache3G |
| 沿用 llama.cpp 工具与接口 | 保留 CLI、`llama-server`、GGUF 加载和聊天接口 | 本地生产测试覆盖文字、图片、流式输出、语法约束与工具参数；需匹配模型和模板 |

HX 是对专家执行和调度路径的扩展。开启它需要指定环境变量与运行参数，不能只替换模型文件；详细配置见 [运行说明](docs/hx/USAGE.md)。

## 测试环境

| 项目 | 实测配置 |
|---|---|
| 操作系统 | Windows x64 |
| CPU | AMD Ryzen 7 9800X3D，8 核 / 16 线程 |
| GPU | AMD Radeon RX 9070 XT，16 GiB，`gfx1201` |
| 系统内存 | 192 GiB |
| 主要模型 | Qwen3.8-Flash-Next，专家 MXFP4、非专家 FP8 部分转为 Q8_0 存储 |
| 其他历史测试 | Q5 量化模型的 HX / legacy 对照 |
| 服务配置 | 单模型、`parallel=1`，预留 262144 token 上下文，加载视觉投影器 |

其他硬件、模型和操作系统需要单独验证，上游 llama.cpp 的兼容范围不等同于 HX 的实测范围。下面分别列出历史基线对照和当前方案复测，避免跨版本拼接性能数字。

## 测试结果

### HX 相对 legacy 路径的历史对照

2026-09-23 的 r9 / H1 测试使用同机 legacy 路径与 HX 8 线程对比：Q8_0 KV、物理批次 1024、262144 token 容量、加载 mmproj。2K 输入测试各运行 3 轮、每轮解码 128 步，取中位数；32K 为单独长输入测量。单位为生成 token/s：

| 模型 / 实际输入 | legacy | HX（8 线程） | 相对提升 |
|---|---:|---:|---:|
| MXFP4 / 2K | 15.46 | 22.71 | 约 **47%** |
| Q5 / 2K | 13.97 | 18.78 | 约 **34%** |
| MXFP4 / 32K | 14.41 | 19.33 | 约 **34%** |
| Q5 / 32K | 12.45 | 16.11 | 约 **29%** |

这组对照展示早期 HX 相对**当时本地 legacy CPU 专家卸载路径**的收益，不是与今天上游最新版本的对比，也不是当前 Cache3G 的提速比例。不同执行路径的输出并非逐 token 完全一致；历史 Q5 HX 在 832 个固定位置的检查中未达到当时的 top-1 判据，详见 [历史测试与数值检查](docs/hx/BENCHMARKS.md#历史-hx--legacy-对照)。速度结果不代表已经证明模型质量等价。

### 当前 Cache3G / MTP1 方案复测

2026-09-25 同机串行测量，固定 WikiText 输入、129 token 输出，greedy、seed 42、不复用提示缓存。2K 输入重复 3 次、8K 重复 2 次；先以 512 输入 / 32 输出预热。表中是中位数：

| 方案 | 实际输入 token | 输入处理 token/s | 生成 token/s | 首字秒 | 全部输出秒 |
|---|---:|---:|---:|---:|---:|
| **Cache3G** | 2048 | 271.29 | **28.36** | 7.57 | 12.09 |
| MTP1 | 2048 | 380.65 | 27.05 | 5.38 | 10.11 |
| 历史 P2A 对照 | 2048 | 555.81 | 25.63 | 3.70 | 8.72 |
| **Cache3G** | 8192 | 279.24 | **26.58** | 29.34 | 34.16 |
| MTP1 | 8192 | 391.29 | 26.46 | 20.95 | 25.79 |
| 历史 P2A 对照 | 8192 | 576.39 | 24.87 | 14.23 | 19.38 |

| 方案 | 专家缓存 | 每 token GPU 路由专家 | 主模型 KV / 物理批次 | 草稿 |
|---|---|---:|---|---|
| Cache3G | 3072 MiB | 最多 4 | Q4_0 / 256 | 无 |
| MTP1 | 0 | 0 | Q4_0 / 512 | 单草稿、GPU 驻留、Q4_0 KV / 批次 128 |
| P2A | 0 | 0 | Q8_0 / 1024 | 无 |

**方案选择：**本地部署偏重生成速度，因此 `main` 对应 Cache3G。它在本轮 2K 的生成速度比 MTP1 高约 4.8%，但输入处理较慢、首字等待更久；8K 的生成差距仅约 0.5%，不足以认定稳定领先。MTP1 在部分代码和工具格式文本中更快。P2A 首字更快，但它是历史源码基线，不能视为当前 `main` 只改参数的等价方案。

首字时间来自本机 `/completion` 流式 API，包含输入处理、不包含模型加载；生成速度来自服务端 timings。这是固定文本续写测试，未计聊天模板、思考过程、客户端渲染或多用户排队。三套配置同时改变了缓存、KV、批次和执行路径，输出文本也不完全相同，不能把全部差异归因于单个优化。

### 长输入与显存

以下来自各方案的独立生产验收，不是同一轮横向比较。各方案均预留 262144 token 容量，实际长输入输出 65 token：

| 方案 | 实际输入 token | 输入处理 token/s | 生成 token/s | 进程专用显存 GiB | 整卡专用显存 GiB |
|---|---:|---:|---:|---:|---:|
| Cache3G | 32768 | 273.15 | 23.13 | 13.57 | 15.12 |
| MTP1 | 32768 | 412.56 | 22.96 | 13.90 | 15.48 |
| MTP1 | 131072 | 343.96 | 15.86 | 14.02 | 15.60 |

长输入后短对话恢复检查通过。显存为请求后的 Windows 快照，不是峰值；整卡包含桌面等占用，共享内存包含主动放在 CPU RAM 中的专家，不能全部视为显存溢出。

### 功能与一致性检查

| 检查 | 历史生产验收结果 |
|---|---|
| 中文计算、图表读数、流式输出和结束标志 | 通过 |
| API 鉴权、模型别名、语法边界、嵌套工具参数 | 通过 |
| 13 个固定提示 | Cache3G 与此前相同 Cache3G 配置 token 一致 13/13；MTP1 与此前优化版 MTP1 一致 13/13 |
| 输入边界与共享前缀回放 | 1 / 2 / 127 / 128 / 129 / 2047 / 2048 / 2049 边界及前缀改写回放通过 |

这里的 13/13 是各自配置的回归检查，不表示 Cache3G、MTP1、legacy 三者互相输出一致，也不是完整质量评估。鉴权和模板等验收包含本机生产启动器的配置。

**已测范围：**Cache3G 本轮实际输入到 32K；MTP1 独立验收到 128K。256K 是已启动的上下文容量，没有完成填满 256K、多用户并发或全天稳定性测试。本次 README 更新整理已有记录，没有重新编译或跑分。完整方法、来源与限制见 [测试记录](docs/hx/BENCHMARKS.md)。

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
