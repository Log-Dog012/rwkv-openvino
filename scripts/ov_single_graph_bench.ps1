# OV 单图常驻推理 benchmark —— 绕开 bash 工具 300s 上限
# 在沙箱外 pwsh 跑：绕开 OV CPU 单图 61 层 JIT 编译要 10+ 分钟的时间限制
# 用法：
#   pwsh -File scripts/ov_single_graph_bench.ps1                 # 默认 CPU, n=16
#   pwsh -File scripts/ov_single_graph_bench.ps1 -Device GPU     # GPU
#   pwsh -File scripts/ov_single_graph_bench.ps1 -Device NPU -N 32
#
# 跑完会出稳态 tg t/s（与 llamacpp llama-bench 同口径：单图常驻、权重一次性加载、推理循环测速）
# 首次 CPU 编译可能 10-20 分钟（OV 插件 JIT 融合 K-quant 子图），耐心等出 "compiled" 行
# 编译缓存启用（out/ov_cache），二次跑同设备编译秒级

param(
    [string]$Device = "CPU",
    [int]$Threads = 8,
    [int]$N = 16,
    [int]$Warmup = 2,
    [string]$Prompt = "The Eiffel Tower is located in the city of",
    [string]$Gguf = "C:\Users\Mcsof\models\gguf\rwkv7-g1i-13.3b-Q4_K_M.gguf",
    [string]$Ir = "out\rwkv7-13.3b-q4k_ov.xml"
)

$ErrorActionPreference = "Stop"

# OV 2026.3 预编译包环境（等效 setupvars.bat）
$OVROOT = "C:\Users\Mcsof\Application\openvino_2026.3.0"
$env:INTEL_OPENVINO_DIR = $OVROOT
$env:OPENVINO_DIR = "$OVROOT\runtime\cmake"
$env:OPENVINO_LIB_PATHS = "$OVROOT\runtime\bin\intel64\Release;$OVROOT\runtime\bin\intel64\Debug"
foreach ($p in @(
    "$OVROOT\runtime\3rdparty\tbb\redist\intel64\vc14",
    "$OVROOT\runtime\3rdparty\tbb\bin\intel64\vc14",
    "$OVROOT\runtime\3rdparty\tbb\bin"
)) {
    if (Test-Path $p) { $env:OPENVINO_LIB_PATHS = "$p;$env:OPENVINO_LIB_PATHS"; break }
}
$env:PATH = "$env:OPENVINO_LIB_PATHS;$env:PATH"
$env:PYTHONPATH = "$OVROOT\python"
$env:PYTHONUNBUFFERED = "1"

Write-Host "=== OV 单图常驻 benchmark ===" -ForegroundColor Cyan
Write-Host "Device=$Device Threads=$Threads N=$N Warmup=$Warmup"
Write-Host "IR=$Ir"
Write-Host "GGUF=$Gguf"
Write-Host ""

Set-Location "C:\Users\Mcsof\a\rwkv\rwkv-openvino"

$py = "C:\Users\Mcsof\miniconda3\envs\AI\python.exe"
& $py -u scripts\ov_single_graph_bench.py $Gguf `
    --ir $Ir `
    --device $Device `
    --threads $Threads `
    --n $N `
    --warmup $Warmup `
    --prompt $Prompt

Write-Host ""
Write-Host "=== done ===" -ForegroundColor Cyan
