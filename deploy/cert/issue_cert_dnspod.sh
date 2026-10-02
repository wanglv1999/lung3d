#!/usr/bin/env bash
# =============================================================================
# Lung3D 证书签发（路线 C：DNS-01 / DNSPod API）
# 适用：迁移前也能签（不依赖 80 端口与 DNS 指向）；结果可用于轻量 nginx 或云托管
# 用法：
#   export DP_Id="你的DNSPod密钥ID"
#   export DP_Key="你的DNSPod密钥Token"
#   bash deploy/cert/issue_cert_dnspod.sh
#
# 取得密钥：DNSPod 控制台 -> 右上角头像 -> 密钥管理 -> 创建密钥
# 若鉴权失败，可改用腾讯云 CAM 密钥并把 DNS_API 设为 dns_tencent（见脚本尾部说明）
# =============================================================================
set -euo pipefail

DOMAIN="${DOMAIN:-tdyylung3d.cn}"
ALT="${ALT:-www.tdyylung3d.cn}"
SSL_DIR="${SSL_DIR:-/etc/nginx/ssl}"
EMAIL="${EMAIL:-admin@tdyylung3d.cn}"
DNS_API="${DNS_API:-dns_dp}"        # dns_dp = DNSPod；dns_tencent = 腾讯云 CAM 密钥
STAGING="${STAGING:-0}"

ACME="$HOME/.acme.sh/acme.sh"

log() { echo -e "\033[1;32m==>\033[0m $*"; }
warn() { echo -e "\033[1;33m[warn]\033[0m $*"; }
die() { echo -e "\033[1;31m[fail]\033[0m $*" >&2; exit 1; }

# ---------- 1. 凭据检查 ----------
log "[1/5] 检查 DNS API 凭据"
if [ "$DNS_API" = "dns_dp" ]; then
  if [ -z "${DP_Id:-}" ] || [ -z "${DP_Key:-}" ]; then
    # 允许从已保存的 account.conf 复用（续期场景）
    if [ -f "$HOME/.acme.sh/account.conf" ] && grep -q "SAVED_DP_Id" "$HOME/.acme.sh/account.conf"; then
      log "    使用 account.conf 中已保存的 DP_Id / DP_Key"
    else
      die "缺少 DP_Id / DP_Key。请先 export DP_Id=... DP_Key=..."
    fi
  else
    log "    已提供 DP_Id / DP_Key（首次签发后会自动存入 ~/.acme.sh/account.conf 供续期复用）"
  fi
else
  [ -n "${TC_SecretId:-}" ] && [ -n "${TC_SecretKey:-}" ] || die "缺少 TC_SecretId / TC_SecretKey"
  log "    使用腾讯云 CAM 密钥（$DNS_API）"
fi

# ---------- 2. 安装 acme.sh ----------
log "[2/5] 安装 / 检测 acme.sh"
if [ ! -x "$ACME" ]; then
  curl -fsSL https://get.acme.sh | sh -s email="$EMAIL" || die "acme.sh 安装失败"
fi
[ -x "$ACME" ] || ACME="$(command -v acme.sh)" || die "找不到 acme.sh"

log "[3/5] 设定默认 CA 为 Let's Encrypt"
"$ACME" --set-default-ca --server letsencrypt >/dev/null 2>&1 || warn "设定默认 CA 失败（可忽略）"

# ---------- 4. 签发（DNS-01） ----------
log "[4/5] 签发证书：$DOMAIN + $ALT（DNS-01 / $DNS_API）"
LE_ARGS=(--server letsencrypt)
[ "$STAGING" = "1" ] && LE_ARGS=(--server letsencrypt --staging) && warn "使用 LE 测试环境"

"$ACME" --issue \
  --dns "$DNS_API" \
  -d "$DOMAIN" -d "$ALT" \
  "${LE_ARGS[@]}" \
  --keylength ec-256 2>&1 | tail -25

# ---------- 5. 安装 ----------
log "[5/5] 安装证书"
if [ -d "$SSL_DIR" ]; then
  if [ -f "$SSL_DIR/$DOMAIN.crt" ]; then
    BK="$SSL_DIR/backup_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BK" && cp -f "$SSL_DIR/$DOMAIN.crt" "$SSL_DIR/$DOMAIN.key" "$BK/" 2>/dev/null || true
    log "    旧证书已备份到 $BK"
  fi

  RELOAD="true"
  # 若本机有 nginx，则装完自动重载
  if command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet nginx 2>/dev/null; then
    RELOAD="systemctl reload nginx"
  fi

  "$ACME" --install-cert -d "$DOMAIN" --ecc \
    --key-file       "$SSL_DIR/$DOMAIN.key" \
    --fullchain-file "$SSL_DIR/$DOMAIN.crt" \
    --reloadcmd      "$RELOAD" || die "安装证书失败"

  chmod 600 "$SSL_DIR/$DOMAIN.key"
  log "    已安装到 $SSL_DIR/$DOMAIN.crt 与 .key"
  log "    已写入自动续期 cron（每日检查，剩余 <60 天自动续签）"
else
  warn "$SSL_DIR 不存在，跳过安装。证书仍在 $HOME/.acme.sh/${DOMAIN}_ecc/"
fi

echo ""
log "证书信息："
CERT_PATH="$HOME/.acme.sh/${DOMAIN}_ecc/fullchain.cer"
[ -f "$CERT_PATH" ] || CERT_PATH="$SSL_DIR/$DOMAIN.crt"
openssl x509 -in "$CERT_PATH" -noout -subject -issuer -dates -serial 2>/dev/null || true
echo ""
echo "=============== 完成 ==============="
echo "【迁移到轻量】证书已在 $SSL_DIR，nginx 直接可用，无需再动"
echo "【仍用云托管】把下面两个文件内容拷出，上传到 云托管 -> 自定义域名 -> 编辑："
echo "   证书(fullchain): $CERT_PATH"
echo "   私钥(key)      : $HOME/.acme.sh/${DOMAIN}_ecc/${DOMAIN}.key"
echo "【巡检】python deploy/cert/check_cert_expiry.py"
echo "===================================="
