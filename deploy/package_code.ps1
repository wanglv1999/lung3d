# 打包微信云托管"手动上传代码包"所需文件 -> lung3d-cloudrun.zip
# zip 根目录包含：Dockerfile, requirements-web.txt, lung3d_api.py, web/
# 排除 .venv / web_output / miniprogram / deploy / wx_config.json(密钥) 等
$root = 'C:\Users\wangl\Documents\Default Project\lung3d'
$stage = Join-Path $env:TEMP 'lung3d_deploy_stage'
$out = Join-Path $root 'deploy\lung3d-cloudrun.zip'

if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force -Path $stage, (Join-Path $stage 'web') | Out-Null

# 复制构建所需文件（保持 Dockerfile 的 COPY 路径一致）
Copy-Item (Join-Path $root 'Dockerfile') $stage
Copy-Item (Join-Path $root 'requirements-web.txt') $stage
Copy-Item (Join-Path $root 'lung3d_api.py') $stage
Copy-Item (Join-Path $root 'web\*') (Join-Path $stage 'web') -Recurse

# 打 zip（根目录直接是 Dockerfile）
if (Test-Path $out) { Remove-Item $out -Force }
Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $out

Remove-Item -Recurse -Force $stage
Write-Host ("已生成: " + $out)
Write-Host ("大小: " + [math]::Round((Get-Item $out).Length / 1KB) + " KB")