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

# vtk 用于全精度等值面引擎（FlyingEdges3D），装完镜像约 +300MB。
# wheel 139MB，容器内直连 PyPI 镜像会 ReadTimeout（腾讯云镜像还会返回空文件，
# sha256 为空内容的哈希 e3b0c442…），所以改成本机下载后 COPY 进镜像。
# 重新获取：pip download vtk==9.7.1 --platform manylinux2014_x86_64 \
#           --python-version 312 --only-binary=:all: -d vendor
COPY vendor/vtk-9.7.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl /tmp/
RUN pip install --no-cache-dir /tmp/vtk-9.7.1-*.whl && rm -f /tmp/vtk-*.whl

# 后端 + 网页前端
COPY lung3d_api.py ./
# 全精度网格引擎（MESH_ENGINE=vtk 时被 lung3d_api 导入）
COPY lung3d_mesh_vtk.py ./
COPY web ./web

# 病例产物目录：容器内默认落临时盘；生产用 CASE_ROOT 指向挂载盘（CFS / 宿主机 bind mount）
# 微信云托管：控制台挂载 CFS 到 /data，CASE_ROOT=/data/cases
# 轻量服务器：docker-compose 把宿主机 /data/lung3d/cases 挂到 /data/cases
ENV CASE_ROOT=/data/cases
# 上传暂存放容器本地临时盘，处理完即删（大文件不写持久盘）
ENV UPLOAD_STAGING_DIR=/tmp/lung3d_uploads
RUN mkdir -p /data/cases /tmp/lung3d_uploads /app/web_output/cases

EXPOSE 8000

CMD ["python", "lung3d_api.py", "--host", "0.0.0.0", "--port", "8000", "--web", "/app/web"]