# 使用 NVIDIA 官方提供的 PyTorch 镜像 (包含了 CUDA, cuDNN)
# 根据实际情况选择 nvcr.io 上的 tag
FROM nvcr.io/nvidia/pytorch:26.01-py3


# 设置工作目录
WORKDIR /workspace

# 设置时区，防止某些包安装时卡在配置界面
ENV DEBIAN_FRONTEND=noninteractive

# 常用系统工具
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Jupyter Lab 默认端口
EXPOSE 8888

# 默认启动 Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token='dl_start'"]