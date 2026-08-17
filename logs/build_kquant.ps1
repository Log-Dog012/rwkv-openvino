# 编译 rwkv_kquant CPUExtension dll
Import-Module "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
Enter-VsDevShell 0f7231f3 -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64"
$OV = "C:\Users\Mcsof\Application\openvino_2026.3.0"
$src = "C:\Users\Mcsof\a\rwkv\rwkv-openvino\scripts\kquant_op"
$build = "$src\build"
Write-Host "=== cmake configure ==="
cmake -S $src -B $build -G Ninja -DCMAKE_PREFIX_PATH="$OV\runtime\cmake" -DCMAKE_BUILD_TYPE=Release 2>&1 | Out-String | Write-Host
Write-Host "=== cmake build ==="
cmake --build $build 2>&1 | Out-String | Write-Host
Write-Host "=== result ==="
Get-ChildItem $build -Filter "*.dll" | Select-Object Name, Length
