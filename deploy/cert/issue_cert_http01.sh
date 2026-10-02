#!/usr/bin/env bash
# =============================================================================
# Lung3D 证书签发（路线 B：HTTP-01 / webroot）
# 适用：迁移后的轻量应用服务器（DNS 已指向本机，80 端口已放通）
# 用法：cd /opt/lung3d && bash deploy/cert/issue_cert_http01.sh
#
# 签发成功后 acme.sh 会自动写入 cron：每日检查、剩余不足 60 天自动续签、
# 签完自动重载 nginx —— 之后无需人工干预。
# =============================================================================
set -euo pipefail

DOMAIN="${DOMAIN:-tdyylung3d.cn}"
ALT="${ALT:-www.tdyylung3d.cn}"
WEBROOT="${WEBROOT:-/var/www/acme}"          # 与 nginx 配置里的 acme-challenge root 一致
SSL_DIR="${SSL_DIR:-/etc/nginx/ssl}"
EMAIL="${EMAIL:-admin@tdyylung3d.cn}"
STAGING="${STAGING:-0}"                       # 1 = 用 LE 测试环境（不消耗配额，仅调试）

ACME="$HOME/.acme.sh/acme.sh"

log() { echo -e "\033[1;32m==>\033[0m $*"; }
warn() { echo -e "\033[1;33m[warn]\033[0m $*"; }
die() { echo -e "\033[1;31m[fail]\033[0m $*" >&2; exit 1; }

# ---------- 1. 前置检查 ----------
log "[1/6] 前置检查"
command -v nginx >/dev/null 2>&1 || die "未检测到 nginx，请先按 deploy/lighthouse/setup_server.sh 初始化"
[ -d "$SSL_DIR" ] || { warn "$SSL_DIR 不存在，自动创建"; mkdir -p "$SSL_DIR"; }
mkdir -p "$WEBROOT"

# 域名是否解析到本机
MYIP="$(curl -fsS -m 10 https://api.ipify.org 2>/dev/null || echo '')"
DOMIP="$(getent hosts "$DOMAIN" 2>/dev/null | awk '{print $1; exit}' || echo '')"
if [ -n "$MYIP" ] && [ -n "$DOMIP" ]; then
  if [ "$MYIP" != "$DOMIP" ]; then
    warn "本机公网 IP=$MYIP，但 $DOMAIN 解析到 $DOMIP"
    warn "HTTP-01 校验会失败 —— 请先把 DNS A 记录切到本机，或改用 issue_cert_dnspod.sh"
  else
    log "    域名解析正确：$DOMAIN -> $DOMIP"
  fi
fi

# 本机 80 是否可自校验（acme-challenge 目录可达）
if curl -fsS -m 8 -o /dev/null "http://127.0.0.1/.well-known/acme-challenge/__probe__" 2>/dev/null; then
  log "    本机 80 端口 webroot 可达"
else
  warn "本机 80 端口探测未通过（正常情况下 404 也算通，这里只是提示）。请确认 nginx 已加载 lung3d.conf"
fi

# ---------- 2. 安装 acme.sh ----------
log "[2/6] 安装 / 检测 acme.sh"
if [ ! -x "$ACME" ]; then
  # 若 get.acme.sh 慢，可改用：git clone https://gitee.com/neilpang/acme.sh.git && cd acme.sh && ./acme.sh --install
  curl -fsSL https://get.acme.sh | sh -s email="$EMAIL" || die "acme.sh 安装失败"
fi
[ -x "$ACME" ] || ACME="$(command -v acme.sh)" || die "找不到 acme.sh"
log "    acme.sh: $ACME ($("$ACME" --version 2>/dev/null | head -1))"

# ---------- 3. 切换默认 CA 到 Let's Encrypt ----------
# acme.sh 默认用 ZeroSSL，首次需注册账号、国内网络下常卡住；显式切到 LE 更省事
log "[3/6] 设定默认 CA 为 Let's Encrypt"
"$ACME" --set-default-ca --server letsencrypt >/dev/null 2>&1 || warn "设定默认 CA 失败（可忽略，签发时仍会显式指定）"

# ---------- 4. 签发 ----------
log "[4/6] 签发证书：$DOMAIN + $ALT（HTTP-01 / webroot=$WEBROOT）"
LE_ARGS=(--server letsencrypt)
[ "$STAGING" = "1" ] && LE_ARGS=(--server letsencrypt --staging) && warn "使用 LE 测试环境（签出的证书浏览器不信任，仅用于验证流程）"

"$ACME" --issue \
  -d "$DOMAIN" -d "$ALT" \
  -w "$WEBROOT" \
  "${LE_ARGS[@]}" \
  --keylength ec-256 2>&1 | tail -20

# ---------- 5. 安装到 nginx ----------
log "[5/6] 安装证书到 $SSL_DIR（与 deploy/lighthouse/nginx/lung3d.conf 的路径一致）"
# 安装前备份旧证书，便于回滚
if [ -f "$SSL_DIR/$DOMAIN.crt" ]; then
  BK="$SSL_DIR/backup_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$BK" && cp -f "$SSL_DIR/$DOMAIN.crt" "$SSL_DIR/$DOMAIN.key" "$BK/" 2>/dev/null || true
  log "    旧证书已备份到 $BK"
fi

"$ACME" --install-cert -d "$DOMAIN" --ecc \
  --key-file       "$SSL_DIR/$DOMAIN.key" \
  --fullchain-file "$SSL_DIR/$DOMAIN.crt" \
  --reloadcmd      "systemctl reload nginx" || die "安装证书失败"

chmod 600 "$SSL_DIR/$DOMAIN.key"

# ---------- 6. 收尾自检 ----------
log "[6/6] 自检"
nginx -t || die "nginx 配置有误，请检查"
systemctl reload nginx

echo ""
log "签发完成。本地证书信息："
openssl x509 -in "$SSL_DIR/$DOMAIN.crt" -noout -subject -dates -serial 2>/dev/null || true
echo ""
log "线上（走域名）证书信息："
echo | openssl s_client -connect "$DOMAIN:443" -servername "$DOMAIN" 2>/dev/null \
  | openssl x509 -noout -dates -serial 2>/dev/null || warn "线上握手失败，检查安全组 443 是否放通"
echo ""
log "自动续期 cron 状态："
crontab -l 2>/dev/null | grep -i acme || warn "未找到 acme cron 条目，请执行：$ACME --install-cronjob"
echo ""
echo "=========== 完成 ==========="
echo "之后无需人工干预：acme.sh 每日检查，剩余 <60 天自动续签并 reload nginx"
echo "巡检命令：python deploy/cert/check_cert_expiry.py"
echo "============================"
