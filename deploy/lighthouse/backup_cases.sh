#!/usr/bin/env bash
# Lung3D 病例数据每日备份（轻量服务器本机盘）
# 安装：cp backup_cases.sh /opt/lung3d/ && chmod +x /opt/lung3d/backup_cases.sh
# 定时：echo '20 3 * * * /opt/lung3d/backup_cases.sh >> /var/log/lung3d-backup.log 2>&1' | crontab -
# 说明：打包 /data/lung3d/cases 到 /data/lung3d/backup，保留最近 KEEP 份。
#      备份与数据同盘，只防误删/误改；要防整机故障需再异地拉取（rsync/COS）。
set -euo pipefail

SRC=/data/lung3d/cases
DST=/data/lung3d/backup
KEEP=7
STAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$DST"

if [ ! -d "$SRC" ]; then
  echo "[$(date '+%F %T')] 源目录不存在：$SRC"
  exit 1
fi

# 病例都是小文件目录，tar 时排除上传暂存（正常已删除，防御性排除）
tar -czf "$DST/cases_$STAMP.tar.gz" -C "$(dirname "$SRC")" "$(basename "$SRC")" \
  --exclude='*/upload' 2>/dev/null || true

echo "[$(date '+%F %T')] 已备份 -> $DST/cases_$STAMP.tar.gz ($(du -h "$DST/cases_$STAMP.tar.gz" | cut -f1))"

# 轮转：只留最近 KEEP 份
ls -1t "$DST"/cases_*.tar.gz 2>/dev/null | tail -n +$((KEEP + 1)) | xargs -r rm -f

# 磁盘水位提醒（>90% 打印告警，交由日志/监控抓取）
USE=$(df -P "$DST" | awk 'NR==2 {gsub("%","",$5); print $5}')
if [ "$USE" -ge 90 ]; then
  echo "[$(date '+%F %T')] WARN 磁盘使用率 ${USE}%，请清理 $DST 或调整 CASE_EXPIRE_DAYS"
fi
