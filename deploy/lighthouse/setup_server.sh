#!/usr/bin/env bash
# Lung3D 轻量应用服务器（境内 2核2G，Ubuntu/Debian）一键初始化
# 用法：sed -i 's/\r$//' setup_server.sh && chmod +x setup_server.sh && sudo bash setup_server.sh
set -euo pipefail

echo "==> [1/6] 安装基础工具"
apt-get update -y
apt-get install -y ca-certificates curl gnupg lsb-release socat cron vim

echo "==> [2/6] 安装 Docker（官方脚本）"
if ! command -v docker >/dev/null 2>&1; then
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -y
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi
systemctl enable --now docker

echo "==> [3/6] 创建数据与证书目录"
mkdir -p /data/lung3d/cases
mkdir -p /var/www/acme
mkdir -p /etc/nginx/ssl
mkdir -p /opt/lung3d

echo "==> [4/6] 配置 2GB swap（2核2G 内存兜底，缓解网格提取 OOM）"
if [ ! -f /swapfile ]; then
  fallocate -l 2G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=2048
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
sysctl -w vm.swappiness=10
echo 'vm.swappiness=10' > /etc/sysctl.d/99-swap.conf

echo "==> [5/6] 安装 nginx 并清掉默认站点"
apt-get install -y nginx
rm -f /etc/nginx/sites-enabled/default
systemctl enable --now nginx

echo "==> [6/6] 安装 acme.sh（证书自动续期）"
# 若 get.acme.sh 拉取慢，可换国内镜像：
#   git clone https://gitee.com/neilpang/acme.sh.git && cd acme.sh && ./acme.sh --install
if [ ! -d "$HOME/.acme.sh" ]; then
  curl -fsSL https://get.acme.sh | sh -s email=admin@tdyylung3d.cn || \
    echo "  [警告] acme.sh 安装失败（可能网络受限），可稍后手动安装或用已有证书"
fi

echo ""
echo "==================== 初始化完成 ===================="
echo "内存/交换检查： free -h"
echo "下一步："
echo "  1) 把 docker-compose.yml 与 .env 放到 /opt/lung3d，填好 .env"
echo "  2) cd /opt/lung3d && docker compose up -d"
echo "  3) 证书：拷贝现有证书到 /etc/nginx/ssl/，或 acme.sh 自动签发"
echo "  4) cp nginx/lung3d.conf /etc/nginx/conf.d/ && nginx -t && systemctl reload nginx"
echo "  5) 【核实备案后】再把 DNS A 记录切到本机公网 IP"
echo "===================================================="
