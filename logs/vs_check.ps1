# VS DevShell 编译环境验证（用户给的初始化命令）
Import-Module "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
Enter-VsDevShell 0f7231f3 -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64"
Write-Host "=== cl ==="
(Get-Command cl).Source
Write-Host "=== cmake ==="
(Get-Command cmake).Source
Write-Host "=== ninja ==="
(Get-Command ninja).Source
Write-Host "=== link ==="
(Get-Command link).Source
# 编译 hello world 测链路
$src = "$env:TEMP\vs_test.cpp"
@"
#include <cstdio>
int main() { printf("hello vs ok\n"); return 0; }
"@ | Set-Content $src -Encoding ascii
cl /nologo /EHsc /O2 $src /Fe:"$env:TEMP\vs_test.exe" 2>&1 | Out-String | Write-Host
& "$env:TEMP\vs_test.exe"
