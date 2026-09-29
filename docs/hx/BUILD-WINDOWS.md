# Windows 构建 HX

本说明按已验证运行时使用的双构建目录方式整理：主程序和 CPU 后端使用 Visual Studio LLVM，HIP 后端使用 ROCm LLVM，再组合相同源码版本的 DLL。此次发布检查了源码与冻结快照的一致性，没有重新执行完整编译和性能验收。

## 环境

- Windows x64；本地验证 CPU 为 Ryzen 7 9800X3D，GPU 为 RX 9070 XT（`gfx1201`）。
- Visual Studio C++ 工具、Windows SDK、LLVM/Clang、CMake 和 Ninja。历史构建使用 Visual Studio 18 Community。
- ROCm 开发 SDK，包括 HIP、hipBLAS、rocBLAS、LLVM 与 device bitcode；历史构建使用便携 `10.1-nightly-gfx1201`。不要混用其他 SDK 的头文件与 DLL。
- Vulkan SDK 用于与历史主程序一致的附加 Vulkan 后端；HX 本身使用 HIP。历史版本为 `1.4.357.0`。
- 足够的磁盘、RAM 和显存。192 GiB RAM 是实测配置，不是已测得的最低内存要求。

工具和依赖从各自官方来源安装。Python 工具使用本机 `config/python-runtime.json` 的 `Python` 字段指定的公用便携解释器，可由 [示例配置](../../config/python-runtime.example.json) 复制生成；该本机配置已加入忽略规则。

## 设置工具路径

Cache3G 使用 `main`，MTP1 使用 `hx-mtp1`。不同分支应使用独立构建及产物目录；切换后重新构建，避免混用旧 DLL。在新 PowerShell 开发会话中，从仓库根目录执行：

```powershell
git switch main
$sourceRoot = (Get-Location).Path
$vsRoot = 'C:\Program Files\Microsoft Visual Studio\18\Community'
$sdk = 'C:\path\to\portable-rocm\_rocm_sdk_devel' # 改为实际目录
$env:VULKAN_SDK = 'C:\VulkanSDK\1.4.357.0'       # 改为实际目录
& "$vsRoot\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
$cmake = "$vsRoot\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$ninja = "$vsRoot\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
$llvm = "$vsRoot\VC\Tools\Llvm\x64\bin"
$env:ROCM_PATH = $sdk
$env:HIP_PATH = $sdk
$env:HIP_PLATFORM = 'amd'
$env:PATH = "$sdk\bin;$llvm;$env:VULKAN_SDK\Bin;$(Split-Path $ninja);$env:PATH"
$env:LIB = "$(Split-Path $llvm)\lib;$env:LIB"
$baseArgs = @('-G', 'Ninja', "-DCMAKE_MAKE_PROGRAM=$ninja", '-DCMAKE_BUILD_TYPE=Release',
    '-DBUILD_SHARED_LIBS=ON', '-DGGML_BACKEND_DL=ON', '-DGGML_NATIVE=OFF')
```

检查环境没有继承其他 ROCm 版本的 CMake 路径。SDK 路径推荐避免空格；按实际安装位置调整 LLVM 的 `libomp.lib`、`libomp.dll` 路径。

## 主程序和 CPU 后端

```powershell
$hostArgs = @('-S', $sourceRoot, '-B', 'build-hx-host') + $baseArgs + @(
    "-DCMAKE_C_COMPILER=$llvm\clang.exe", "-DCMAKE_CXX_COMPILER=$llvm\clang++.exe",
    '-DGGML_CPU_ALL_VARIANTS=ON', '-DGGML_VULKAN=ON', '-DGGML_OPENMP=ON', '-DGGML_OPENMP_FETCH=OFF',
    '-DOpenMP_C_FLAGS=-fopenmp=libomp', '-DOpenMP_CXX_FLAGS=-fopenmp=libomp',
    '-DOpenMP_C_LIB_NAMES=libomp', '-DOpenMP_CXX_LIB_NAMES=libomp',
    "-DOpenMP_libomp_LIBRARY=$(Split-Path $llvm)\lib\libomp.lib",
    '-DLLAMA_BUILD_BORINGSSL=ON', '-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_SUBPROCESS=ON')
& $cmake @hostArgs
if ($LASTEXITCODE -ne 0) { throw 'Host configuration failed' }
& $cmake --build build-hx-host --parallel 12 --target llama-server llama-cli llama-bench
if ($LASTEXITCODE -ne 0) { throw 'Host build failed' }
```

只需 HIP 时可将 `GGML_VULKAN` 改为 `OFF` 并移除 Vulkan SDK 配置，但这与历史双后端产物配置不同。

## HIP 后端

```powershell
$hipArgs = @('-S', $sourceRoot, '-B', 'build-hx-hip') + $baseArgs + @(
    "-DCMAKE_PREFIX_PATH=$sdk",
    "-DCMAKE_C_COMPILER=$sdk\lib\llvm\bin\clang.exe",
    "-DCMAKE_CXX_COMPILER=$sdk\lib\llvm\bin\clang++.exe",
    "-DCMAKE_C_FLAGS=-w --rocm-path=$sdk --rocm-device-lib-path=$sdk\lib\llvm\amdgcn\bitcode",
    "-DCMAKE_CXX_FLAGS=-w --rocm-path=$sdk --rocm-device-lib-path=$sdk\lib\llvm\amdgcn\bitcode",
    '-DGGML_HIP=ON', '-DGGML_CPU=OFF', '-DGGML_CUDA_FA=ON', '-DGGML_CUDA_FA_ALL_QUANTS=ON',
    '-DGGML_HIP_GRAPHS=ON', '-DGGML_HIP_NO_VMM=ON', '-DGGML_HIP_MMQ_MFMA=ON',
    '-DGGML_HIP_RCCL=OFF', '-DGPU_TARGETS=gfx1201', '-DLLAMA_BUILD_TESTS=OFF')
& $cmake @hipArgs
if ($LASTEXITCODE -ne 0) { throw 'HIP configuration failed' }
& $cmake --build build-hx-hip --parallel 12 --target ggml-hip
if ($LASTEXITCODE -ne 0) { throw 'HIP build failed' }
```

检查 `build-hx-hip/CMakeCache.txt` 中 HIP、hipBLAS、rocBLAS 均来自 `$sdk`。`gfx1201` 是本机验证目标，不能直接用于其他架构。

## 组合并检查

```powershell
$runtimeDir = Join-Path $sourceRoot 'runtime-hx'
New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
Get-ChildItem 'build-hx-host\bin' -File |
    Where-Object Extension -In '.dll', '.exe' |
    Copy-Item -Destination $runtimeDir
Copy-Item 'build-hx-hip\bin\ggml-hip.dll' $runtimeDir
foreach ($name in 'amdhip64_7.dll', 'amd_comgr.dll', 'rocm_kpack.dll') {
    Copy-Item (Join-Path "$sdk\bin" $name) $runtimeDir
}
Copy-Item "$llvm\libomp.dll" $runtimeDir
& "$runtimeDir\llama-server.exe" --version
& "$runtimeDir\llama-server.exe" --list-devices
& "$runtimeDir\llama-server.exe" --help
```

保持 SDK 的 `bin` 在 PATH 中，以加载其他依赖。DLL 文件名可能随 SDK 变化；缺失时核对 SDK，不从无关版本复制。CPU 变体应包含 `ggml-cpu-zen4.dll`，HIP 应枚举到目标 GPU。然后按 [运行配置](USAGE.md) 做短提示、流式和所需功能验证。

源码 SHA-256 见 [清单](source-manifest.json)。编译器、依赖、参数和构建路径影响二进制哈希，源码相同不保证 DLL 字节相同。
