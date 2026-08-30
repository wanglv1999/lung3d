# Lung3D 微信云托管部署说明

本目录方案：**本地 Docker 构建 → 推送镜像到微信云托管**。
不依赖自己买云服务器，微信云托管自带 HTTPS 域名。

## 准备
- 本机安装 Docker Desktop（Windows 用 PowerShell，或直接在 Docker Desktop 里用 WSL2）
- 微信开发者工具里已开通「云开发」，并创建了「云托管」服务
- 拿到云托管控制台里的 **镜像仓库地址**（形如 `ccr.ccs.tencentyun.com/<命名空间>/<服务名>`）和登录凭证

## 1. 本地构建镜像
在 `lung3d/` 目录执行：
```bash
docker build -t lung3d-web .
```
（镜像只含 Web 后端，不含 torch/TotalSegmentator，体积小构建快）

## 2. 登录云托管镜像仓库并推送

### 先在控制台拿到两个东西
- **完整镜像地址**：云托管控制台 → 你的服务 → 【镜像仓库】标签 → 复制
  （形如 `ccr.ccs.tencentyun.com/<命名空间>/<服务名>:latest`）
- **用户名 / 密码**：同一页面提供的仓库登录凭证（密码可能是一次性的）

### 一键推送（推荐）
在 `lung3d/` 目录下，双击 `deploy\build_push.bat`，或在 PowerShell 里运行：
```powershell
.\deploy\build_push.ps1
```
脚本会：
- 提示你粘贴完整镜像地址
- 提示输入 用户名/密码（输入密码不回显）
- 依次执行 `docker login` → `docker build -t lung3d-web .` → `docker tag` → `docker push`

也可以不带交互直接传参：
```powershell
.\deploy\build_push.ps1 -Image ccr.ccs.tencentyun.com/<命名空间>/<服务名>:latest -User <用户> -Pass <密码>
```

### 手动执行（等价命令）
```bash
docker login ccr.ccs.tencentyun.com --username <账号> --password <密码>
docker build -t lung3d-web .
docker tag lung3d-web ccr.ccs.tencentyun.com/<命名空间>/<服务名>:latest
docker push ccr.ccs.tencentyun.com/<命名空间>/<服务名>:latest
```

## 3. 在云托管控制台部署
- 新建版本，选择刚推送的镜像，端口填 **8000**
- 健康检查路径填 **/api/health**
- 部署后云托管会分配一个 HTTPS 域名，形如 `https://<服务ID>-<环境ID>.tcloudbaseapp.com`

## 4. 小程序指向该域名
- `miniprogram/config.js` 的 `BASE` 改成云托管返回的 HTTPS 域名
- 在微信公众平台「开发管理 → 开发设置 → 服务器域名」把该域名加入
  `request / uploadFile / downloadFile` 合法域名
- 用微信开发者工具重新编译上传发布

## 5. 验证
```bash
curl https://<你的域名>/api/health          # -> {"status":"ok"}
curl -X POST -F "files=@病例.zip" https://<你的域名>/api/process
curl "https://<你的域名>/api/wxacode?case=<病例ID>" -o code.png
```

## 注意
- 当前病例网格存在容器**临时盘** `/app/web_output/cases`，重启/扩缩容会丢失。
  先跑通验证没问题后，建议改成挂载 CFS 持久化盘，或把 GLB 存到云存储(COS)。
- 如果上传大 CT 压缩包，需在服务里把请求体大小限制调大。
- `wx_config.json`（AppID/Secret）在容器里通过环境变量 `WX_APPID` / `WX_SECRET` 注入更安全。

## 本地跑一遍（可选）
不推镜像也能先验证：
```bash
docker run --rm -p 8000:8000 lung3d-web
# 浏览器打开 http://localhost:8000
```