# 推送 :stable(备份) 和 :latest(新版) 两个标签到 Docker Hub
param([string]$User = "", [string]$Pass = "")
$ErrorActionPreference = 'Stop'
if (-not $User) { $User = Read-Host "Docker Hub 用户名" }
if (-not $Pass) {
    $sec = Read-Host "Docker Hub 密码(或 Access Token)" -AsSecureString
    $Pass = [System.Runtime.InteropServices.Marshal]::PtrToStringUni([System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec))
}
if (-not $User) { Write-Host "未提供用户名，退出。" -ForegroundColor Red; exit 1 }
Write-Host "[1/3] docker login" -ForegroundColor Green
$Pass | docker login --username $User --password-stdin
if ($LASTEXITCODE -ne 0) { Write-Host "登录失败" -ForegroundColor Red; exit 1 }
Write-Host "[2/3] push stable 备份: $User/lung3d-web:stable" -ForegroundColor Green
docker push "$User/lung3d-web:stable"
if ($LASTEXITCODE -ne 0) { Write-Host "stable 推送失败" -ForegroundColor Red; exit 1 }
Write-Host "[3/3] push latest 新版: $User/lung3d-web:latest" -ForegroundColor Green
docker push "$User/lung3d-web:latest"
if ($LASTEXITCODE -ne 0) { Write-Host "latest 推送失败" -ForegroundColor Red; exit 1 }
Write-Host "推送完成！云托管新建版本部署 latest。" -ForegroundColor Green