# 源码来源与许可证

本项目为 **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) 的衍生分支**，保留上游历史，不将 llama.cpp / ggml 的实现宣称为 HX 原创。

- 上游基线：`7fe450e19305b828c199d602c23a8337aaa1f03b`，llama.cpp v0.5.0。
- MTP 来源：[PR #28243](https://github.com/ggml-org/llama.cpp/pull/28243)，集成头 `6fcaa16f`；合并历史保留在 `02abf6580` 及父提交中，不表示该 PR 已被上游正式发布。
- HX 移植基线：`3ad2c2ca45a86abd1c535f537ad9c427f4401a46`，以 `hx-v050-base` 分支保留。

| 历史提交 | 内容 |
|---|---|
| `0f9cc166d` | 将 UEPT + HX + r10 移植到 v0.5.0 |
| `02abf6580` | 合并 PR #28243 的 MTP 工作 |
| `1f75192cb` | 旧版草稿上下文跳过独立 HX 引擎初始化 |
| `3ad2c2ca4` | 草稿使用 direct MMQ；gather 空间不足时回退 |

## 发布快照

| 分支 | 冻结源码 | 对应本地运行时 |
|---|---|---|
| `main` | `v050-schedule-opt-20260925` | `hx-v050-cache3g-20260925` |
| `hx-mtp1` | `v050-graph-diagnosis-20260925` | `hx-mtp1-20260925` |

后续快照包含 CPU 专家调度、MTP、KV 补齐、服务及模型转换等改动。本次发布在历史基线上记录快照最终状态，没有重新设计或改写推理实现。

两份快照的推理源码差异只有 `ggml/src/ggml-cuda/ggml-cuda.cu`：MTP1 增加小批量主模型直接验证路径与可选图差异诊断。快照以历史 HX 基线的 3630 个跟踪文件为边界，不导出 Python 缓存、推理模型权重、生产日志或编译产物。

[source-manifest.json](source-manifest.json) 记录当前分支相对 v0.5.0 有差异的源码文件大小、SHA-256 与快照名。文档和 `.gitignore` 是发布时整理的内容，单独由 Git 跟踪。`README-upstream.md` 保存快照原有 README，主体 `LICENSE`、`AUTHORS` 和第三方许可证完整保留。

历史运行时部分阶段采用增量编译及复用不变对象文件；源码相同不保证新构建 DLL 哈希相同。本次核对源码内容和已有构建清单，不声称已完成全新工具链下的二进制复现。

## 许可和归属

- llama.cpp / ggml 主体遵循 [MIT License](../../LICENSE)，保留 `Copyright (c) 2023-2026 The ggml authors`。
- 第三方源码和资源遵循各自许可证，见 `licenses/`、`vendor/` 及组件目录。
- 本仓库新增的 HX 说明文档也采用 MIT 许可证。
- 模型、量化权重、投影器和 ROCm 等外部软件的条款独立于本仓库。

本次公开发布经项目所有者授权；Codex 协助快照核对、发布整理和文档编写。没有向 llama.cpp 上游提交 PR，也没有代替上游对 HX 作兼容性或性能承诺。
