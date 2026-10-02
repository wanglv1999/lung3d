#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lung3D 证书到期巡检（跨平台，零第三方依赖）

用法：
    python deploy/cert/check_cert_expiry.py
    python deploy/cert/check_cert_expiry.py --warn-days 45
    python deploy/cert/check_cert_expiry.py --local-only         # 只查项目内证书文件
    python deploy/cert/check_cert_expiry.py --host example.com   # 换域名

退出码：
    0 = 全部正常
    1 = 有证书剩余天数 <= warn-days（需要续期）
    2 = 检查失败（网络/文件/工具问题）
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import socket
import ssl
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCAL_CRT = os.path.join(ROOT, "tdyylung3d.cn_nginx", "tdyylung3d.cn_bundle.crt")

GREEN, YELLOW, RED, DIM, RESET = "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[0m"


def days_left(when: dt.datetime) -> int:
    return (when - dt.datetime.now(dt.timezone.utc)).days


def color_for(days: int, warn: int) -> str:
    if days < 0:
        return RED
    if days <= warn:
        return YELLOW
    return GREEN


def fmt(days: int) -> str:
    return f"已过期 {-days} 天" if days < 0 else f"剩余 {days} 天"


def _openssl_info(path: str) -> dict | None:
    """用 openssl 解析证书文件（Windows / Linux 通用，避免引入 cryptography 依赖）。"""
    try:
        out = subprocess.run(
            ["openssl", "x509", "-in", path, "-noout", "-enddate", "-serial", "-subject", "-issuer", "-ext", "subjectAltName"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None

    info: dict = {"not_after": None, "subject": "-", "issuer": "-", "serial": "-", "san": []}
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.startswith("notAfter="):
            # 形如 "Nov 30 00:59:59 2026 GMT"
            for f in ("%b %d %H:%M:%S %Y %Z", "%b %d %H:%M:%S %Y"):
                try:
                    info["not_after"] = dt.datetime.strptime(line.split("=", 1)[1].strip(), f).replace(
                        tzinfo=dt.timezone.utc
                    )
                    break
                except ValueError:
                    continue
        elif line.startswith("serial="):
            info["serial"] = line.split("=", 1)[1].strip()
        elif line.startswith("subject="):
            info["subject"] = line.split("=", 1)[1].strip()
        elif line.startswith("issuer="):
            info["issuer"] = line.split("=", 1)[1].strip()
        elif line.startswith("DNS:"):
            info["san"] = [v.strip() for v in line.replace("DNS:", "").split(",") if v.strip()]

    return info if info["not_after"] else None


def fetch_live_der(host: str, port: int = 443, timeout: int = 15) -> bytes:
    """抓取线上服务端证书的 DER 内容（不做链校验，过期证书也能取到）。"""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with socket.create_connection((host, port), timeout=timeout) as raw:
        with ctx.wrap_socket(raw, server_hostname=host) as s:
            der = s.getpeercert(binary_form=True)
    if not der:
        raise RuntimeError("未取到服务端证书")
    return der


def check_live(host: str) -> dict | None:
    print(f"  {DIM}正在握手 {host}:443 …{RESET}")
    try:
        der = fetch_live_der(host)
    except Exception as exc:  # noqa: BLE001
        print(f"  {RED}✗{RESET} 线上握手失败：{type(exc).__name__}: {exc}")
        return None

    tmp = None
    try:
        with tempfile.NamedTemporaryFile("wb", suffix=".pem", delete=False) as fh:
            tmp = fh.name
            fh.write(ssl.DER_cert_to_PEM_cert(der).encode("ascii"))
        info = _openssl_info(tmp)
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    if info is None:
        print(f"  {RED}✗{RESET} 取到证书但无法解析（需要 openssl 命令在 PATH 中）")
        return None
    info["source"] = f"https://{host}"
    return info


def check_local(path: str) -> dict | None:
    if not os.path.isfile(path):
        print(f"  {DIM}· 本地证书不存在，跳过：{path}{RESET}")
        return None
    info = _openssl_info(path)
    if info is None:
        print(f"  {RED}✗{RESET} 无法解析本地证书（需要 openssl 命令）：{path}")
        return None
    info["source"] = os.path.relpath(path, ROOT)
    return info


def report(title: str, info: dict, warn: int) -> int:
    not_after = info["not_after"]
    days = days_left(not_after)
    c = color_for(days, warn)
    print(f"\n{title}")
    print(f"  来源    : {info.get('source', '-')}")
    print(f"  主体    : {info.get('subject', '-')}")
    print(f"  签发者  : {info.get('issuer', '-')}")
    print(f"  序列号  : {info.get('serial', '-')}")
    if info.get("san"):
        print(f"  覆盖域名: {', '.join(info['san'])}")
    print(f"  到期    : {not_after.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  状态    : {c}{fmt(days)}{RESET}")
    return days


def main() -> int:
    ap = argparse.ArgumentParser(description="Lung3D HTTPS 证书到期巡检")
    ap.add_argument("--host", default="tdyylung3d.cn", help="要检查的线上域名")
    ap.add_argument("--warn-days", type=int, default=30, help="剩余多少天开始告警（默认 30）")
    ap.add_argument("--local-only", action="store_true", help="只检查项目内证书文件")
    ap.add_argument("--local-crt", default=LOCAL_CRT, help="项目内证书路径")
    args = ap.parse_args()

    print(f"{DIM}巡检时间：{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}")

    worst: int | None = None

    if not args.local_only:
        live = check_live(args.host)
        if live:
            worst = report(f"【线上】https://{args.host}", live, args.warn_days)

    local = check_local(args.local_crt)
    if local:
        d = report("【本地】" + local["source"], local, args.warn_days)
        worst = d if worst is None else min(worst, d)

    print()
    if worst is None:
        print(f"{RED}未取得任何证书信息，检查失败。{RESET}")
        return 2

    if worst < 0:
        print(f"{RED}[严重] 证书已过期！立即续期并替换，否则小程序与 Web 全部不可用。{RESET}")
        print("  处理路径：deploy/cert/证书续期方案_20261003.md")
        return 1
    if worst <= args.warn_days:
        print(f"{YELLOW}[提醒] 证书剩余 {worst} 天，已进入续期窗口，请尽快处理。{RESET}")
        print("  处理路径：deploy/cert/证书续期方案_20261003.md")
        return 1
    print(f"{GREEN}[正常] 证书剩余 {worst} 天，无需操作。{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
