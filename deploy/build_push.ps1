# Lung3D 微信云托管 构建 + 推送 脚本
# 用法（在 lung3d\ 目录）：
#   1. 云托管控制台 -> 你的服务 -> 【镜像仓库】标签，复制"完整镜像地址"
#      （形如 ccr.ccs.tencentyun.com/<命名空间>/<服务名>:latest）
#   2. 运行本脚本，粘贴镜像地址，输入控制台给的 用户名/密码
#      或直接命令行传参：.\deploy\build_push.ps1 -Image <地址> -User <用户> -Pass <密码>
#   3. 自动完成 login -> build -> tag -> push

param(
    [string]$Image = "",
    [string]$User = "",
    [string]$Pass = ""
)

$ErrorActionPreference = 'Stop'

# ---- 1. 镜像地址 ----
if (-not $Image) {
    Write-Host ""
    Write-Host "请从云托管控制台【镜像仓库】复制完整镜像地址，粘贴后回车：" -ForegroundColor Cyan
    $Image = Read-Host
}
if (-not $Image) { Write-Host "未提供镜像地址，退出。" -ForegroundColor Red; exit 1 }

$slash = $Image.IndexOf('/')
if ($slash -lt 1) {
    Write-Host "镜像地址格式错误，应形如：ccr.ccs.tencentyun.com/命名空间/服务名:latest" -ForegroundColor Red
    exit 1
}
$registry = $Image.Substring(0, $slash)

# ---- 2. 登录凭证 ----
if (-not $User) {
    $User = Read-Host "仓库用户名(云托管控制台【镜像仓库】页面获取)"
}
if (-not $Pass) {
    $sec = Read-Host "仓库密码(同上，或生成的一次性密码)" -AsSecureString
    $Pass = [System.Runtime.InteropServices.Marshal]::PtrToStringUni(
        [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec))
}

# ---- 3. 执行 ----
Write-Host ""
Write-Host "[1/4] docker login $registry" -ForegroundColor Green
$Pass | docker login $registry -u $User --password-stdin
if ($LASTEXITCODE -ne 0) { Write-Host "登录失败，检查用户名/密码或网络。" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "[2/4] docker build -t lung3d-web ." -ForegroundColor Green
docker build -t lung3d-web .
if ($LASTEXITCODE -ne 0) { Write-Host "构建失败。" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "[3/4] docker tag lung3d-web $Image" -ForegroundColor Green
docker tag lung3d-web $Image

Write-Host ""
Write-Host "[4/4] docker push $Image" -ForegroundColor Green
docker push $Image
if ($LASTEXITCODE -ne 0) { Write-Host "推送失败。" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "推送成功！" -ForegroundColor Green
Write-Host "接下来到云托管控制台 -> 新建版本：选该镜像，端口 8000，健康检查 /api/health。"