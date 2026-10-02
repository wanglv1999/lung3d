# 直接推送本地已构建好的 lung3d-web 镜像到 Docker Hub（跳过构建）
param([string]$User = "", [string]$Pass = "")
$ErrorActionPreference = 'Stop'
if (-not $User) { $User = Read-Host "Docker Hub 用户名" }
if (-not $Pass) {
    $sec = Read-Host "Docker Hub 密码(或 Access Token)" -AsSecureString
    $Pass = [System.Runtime.InteropServices.Marshal]::PtrToStringUni([System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec))
}
if (-not $User) { Write-Host "未提供用户名，退出。" -ForegroundColor Red; exit 1 }
$Img = "$User/lung3d-web:latest"
Write-Host "[1/2] docker tag lung3d-web $Img" -ForegroundColor Green
docker tag lung3d-web $Img
Write-Host "[2/2] docker push $Img" -ForegroundColor Green
$Pass | docker login --username $User --password-stdin
if ($LASTEXITCODE -ne 0) { Write-Host "登录失败" -ForegroundColor Red; exit 1 }
docker push $Img
if ($LASTEXITCODE -ne 0) { Write-Host "推送失败" -ForegroundColor Red; exit 1 }
Write-Host "推送成功！云托管新建版本选该镜像即可。" -ForegroundColor Green