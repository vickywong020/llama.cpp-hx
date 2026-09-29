# HX 运行配置

从仓库根目录执行，假设已按 [构建说明](BUILD-WINDOWS.md) 生成 `runtime-hx`。替换示例模型路径；多分片 GGUF 的 `--model` 指向第一片，同组分片放在同一目录。

## 模型和设备

模型权重、量化分片、MTP 草稿、视觉投影器只能从 [ModelScope](https://modelscope.cn) 获取。记录仓库、revision、大小与可用的 SHA-256；按来源隔离断点续传缓存。没有所需文件时停止，不自动回退。本仓库不包含下载脚本或访问密钥。

MXFP4 / Q8_0 表示专家为 MXFP4、部分非专家张量为 Q8_0 存储，不等同于任意同名模型的任意量化。MTP 和 mmproj 必须匹配主模型。转换工具名称可能含 `hf`，不改变下载约定；Python 转换工具使用本机 `config/python-runtime.json` 指定的便携解释器。

```powershell
$server = Join-Path (Get-Location).Path 'runtime-hx\llama-server.exe'
$model = 'C:\models\model-00001-of-00012.gguf'
$mmproj = 'C:\models\mmproj.gguf'
& $server --list-devices
```

按实际枚举选择 GPU。历史机器用 `HIP_VISIBLE_DEVICES=1` 隔离独显，隔离后为 `ROCm0`；编号不适用于所有机器。核对型号后设置，再次运行 `--list-devices`。下文假定 `ROCm0` 是目标独显。

## Cache3G（main）

使用新 PowerShell，避免遗留实验变量。3072 MiB 是专家缓存预算，并非整卡显存预算。

```powershell
$env:LLAMA_HX = '1'
$env:LLAMA_HX_THREADS = '8'
$env:LLAMA_HX_MAX_TOKENS = '8'
$env:LLAMA_HX_GPU = '4'
$env:LLAMA_HX_PREFETCH = '8'
$env:LLAMA_HX_PF_SKIP = '0'
$env:LLAMA_HX_EXPERT_PARALLEL = '1'
$env:GGML_CUDA_UEPT_MMQ = 'gather'
$env:KMP_BLOCKTIME = '0'
$env:ROCBLAS_USE_HIPBLASLT = '0'
$env:QWEN4EXP_QSA_GATHER = '0'

& $server --model $model --mmproj $mmproj --alias local-model `
    --host 127.0.0.1 --port 8080 --ctx-size 262144 --parallel 1 `
    --threads 4 --threads-batch 16 --load-mode none `
    --device ROCm0 --gpu-layers 99 --fit off --flash-attn on --kv-offload `
    --batch-size 8192 --ubatch-size 256 --cache-type-k q4_0 --cache-type-v q4_0 `
    --expert-exec uept --expert-cache-mib 3072 --lazy-mode off --op-offload `
    --spec-type none --jinja --reasoning-format auto --reasoning-preserve
```

不使用图片时可移除 mmproj，但这改变显存占用，不能直接复现包含视觉投影器的测量。历史生产配置在容量不超过 131072 时使用 `--ubatch-size 512`，超过时为 256。KV、模型图、暂存区、mmproj 与桌面也占显存。

示例绑定本机；对外提供服务应配置 `--api-key-file`，密钥保留在本机。API 包括 `/health`、`/v1/models` 和 `/v1/chat/completions`，格式见 [server 文档](../../tools/server/README.md)。

## MTP1（hx-mtp1）

切换 `hx-mtp1` 并用独立目录重新构建。它增加小批量主模型直接验证路径和可选图诊断，诊断默认关闭。不能用旧 `hx-v050-base` 只改草稿数为 1 来替代。

在新 PowerShell 设置前面的公共变量，**把 `LLAMA_HX_GPU` 改成 `0`**，并设置下面的 MTP 变量。重新定义 `$server` 指向 MTP1 构建产物，定义主模型、mmproj 和草稿路径。

```powershell
$env:LLAMA_HX_GPU = '0'
$env:LLAMA_HX_SCHEDULE = 'dynamic'
$env:LLAMA_HX_MTP = '1'
$env:LLAMA_MTP_KV_ONLY = '1'
$env:LLAMA_MTP_UBATCH = '128'
$env:LLAMA_MTP_GPU_RESIDENT = '1'
$env:LLAMA_VERIFY_TARGET_NO_GRAPH = '1'
$mtp = 'C:\models\mtp.gguf'

& $server --model $model --mmproj $mmproj --alias local-model `
    --host 127.0.0.1 --port 8080 --ctx-size 262144 --parallel 1 `
    --threads 4 --threads-batch 16 --load-mode none `
    --device ROCm0 --gpu-layers 99 --fit off --flash-attn on --kv-offload `
    --batch-size 8192 --ubatch-size 512 --cache-type-k q4_0 --cache-type-v q4_0 `
    --expert-exec uept --expert-cache-mib 0 --lazy-mode off --op-offload `
    -md $mtp --spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-p-min 0.5 `
    --spec-draft-device ROCm0 --spec-draft-ngl 99 -ctkd q4_0 -ctvd q4_0 `
    --jinja --reasoning-format auto --reasoning-preserve
```

MTP 不保证更快。接受率、草稿开销、输入长度与显存压力均有影响；分别记录首字时间、输入速度、生成速度和请求总耗时。

## 排查

| 现象 | 检查 |
|---|---|
| 找不到 ROCm 设备或 DLL | SDK、PATH、驱动、GPU 架构、`--list-devices`；勿混用 DLL |
| 未启用 HX | `LLAMA_HX=1`、`--expert-exec uept`、`--op-offload` 及启动日志 |
| gather 退回 direct / 首字慢 | 显存余量和物理批次；mmproj、MTP、批次均影响工作集 |
| CPU 满载但生成慢 | CPU 变体、HX 线程数、内存带宽及并行负载 |
| MTP1 未直接验证 | 分支是否为 `hx-mtp1`，是否设置 `LLAMA_VERIFY_TARGET_NO_GRAPH=1` |
| 多 system 消息模板错误 | 本机生产另有兼容模板，不属于本源码快照。使用匹配模型的 `--chat-template-file`，或由客户端合并开头的 system 消息 |

不要把历史实验开关全部打开，诊断自身可能影响速度。结束测试进程并打开新 PowerShell，是清除旧环境变量、按完整方案重新启动的简单方式。
