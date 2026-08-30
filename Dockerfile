# Lung3D 微信云托管镜像
# 构建：docker build -t <镜像名> .
# 运行（本地验证）：docker run --rm -p 8000:8000 <镜像名>
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
# 使用腾讯云 PyPI 镜像源，避免国内云构建拉包超时
ENV PIP_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple
ENV PIP_TRUSTED_HOST=mirrors.cloud.tencent.com

# 只装 Web 所需轻量依赖（不含 torch/TotalSegmentator）
COPY requirements-web.txt ./
RUN pip install --no-cache-dir -r requirements-web.txt

# 后端 + 网页前端
COPY lung3d_api.py ./
COPY web ./web

# 病例输出目录（临时盘；如需持久化再挂载 CFS/COS）
RUN mkdir -p /app/web_output/cases

EXPOSE 8000

CMD ["python", "lung3d_api.py", "--host", "0.0.0.0", "--port", "8000", "--web", "/app/web"]