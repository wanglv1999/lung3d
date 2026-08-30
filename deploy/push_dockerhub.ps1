# 推镜像到 Docker Hub，供微信云托管"从地址拉取镜像"使用
# 用法（在 lung3d\ 目录）：
#   1. 先注册 Docker Hub 账号 https://hub.docker.com
#   2. 运行本脚本，输入 Docker Hub 用户名/密码
#      或直接传参：.\deploy\push_dockerhub.ps1 -User <dockerhub用户名> -Pass <密码>
#   3. 完成后到云托管选"从地址拉取镜像"，填 docker.io/<用户名>/lung3d-web:latest

param(
    [string]$User = "",
    [string]$Pass = ""
)

$ErrorActionPreference = 'Stop'

if (-not $User) { $User = Read-Host "Docker Hub 用户名" }
if (-not $Pass) {
    $sec = Read-Host "Docker Hub 密码(或 Access Token)" -AsSecureString
    $Pass = [System.Runtime.InteropServices.Marshal]::PtrToStringUni(
        [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec))
}
if (-not $User) { Write-Host "未提供用户名，退出。" -ForegroundColor Red; exit 1 }

$Img = "$User/lung3d-web:latest"

Write-Host ""
Write-Host "[1/3] docker login (Docker Hub)" -ForegroundColor Green
$Pass | docker login --username $User --password-stdin
if ($LASTEXITCODE -ne 0) { Write-Host "登录失败，检查用户名/密码。" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "[2/3] docker build -t $Img" -ForegroundColor Green
docker build -t $Img .
if ($LASTEXITCODE -ne 0) { Write-Host "构建失败。" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "[3/3] docker push $Img" -ForegroundColor Green
docker push $Img
if ($LASTEXITCODE -ne 0) { Write-Host "推送失败。" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "推送成功！" -ForegroundColor Green
Write-Host "云托管 -> 选择方式选【从地址拉取镜像】，地址填："
Write-Host "  docker.io/$Img" -ForegroundColor Yellow
Write-Host "（Docker Hub 仓库需设为 public，首次推送会自动创建公开仓库）"