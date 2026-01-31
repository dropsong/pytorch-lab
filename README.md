这个仓库主要是为一些要求很小的深度学习入门玩具提供一个可复现的 Pytorch 环境。个人用。

使用 tensorflow 的资料，请参见：[dropsong / dl-ipynb-examples](https://github.com/dropsong/dl-ipynb-examples), 本仓库的一些内容会与之重复。

# 环境配置

先过一下配置（何意味？）：

```bash
OS: Debian GNU/Linux forky/sid (forky) x86_64
Kernel: Linux 6.18.5+deb14-amd64
CPU: 13th Gen Intel(R) Core(TM) i7-13650HX (20) @ 4.90 GHz
GPU 1: NVIDIA GeForce RTX 5060 Max-Q / Mobile [Discrete]
GPU 2: Intel Raptor Lake-S UHD Graphics @ 1.55 GHz [Integrated]
Memory: 8.58 GiB / 30.96 GiB (28%)
Swap: 48.00 KiB / 13.21 GiB (0%)
```

```bash
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.126.09             Driver Version: 580.126.09     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5060 ...    Off |   00000000:01:00.0  On |                  N/A |
| N/A   41C    P8              3W /   55W |      52MiB /   8151MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            1534      G   /usr/lib/xorg/Xorg                        2MiB |
|    0   N/A  N/A            1783      G   /usr/bin/kwin_wayland                     2MiB |
+-----------------------------------------------------------------------------------------+
```

在 Docker 容器中调用本地 RTX 5060，必须安装 NVIDIA 的容器工具包。

确保已安装 Docker，然后运行以下命令（若已安装可跳过）：

```bash
# 配置仓库 (使用官方源)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 安装
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 配置 Docker 运行时
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

其他必要文件见同目录下其他文件。

# 运行与测试

启动环境： 在本目录下运行：

```bash
docker compose up -d --build
```

查看日志 (获取 Jupyter 地址)：

```bash
docker compose logs -f
```

执行测试：

```bash
docker exec -it zhiyue-pytorch python test_gpu.py
```

得到输出：

```bash
========================================
OS: Linux 6.18.5+deb14-amd64
Python Version: 3.12.3
PyTorch Version: 2.10.0a0+a36e1d39eb.nv26.01.42222806
CUDA Available: True
CUDA Version: 13.1
cuDNN Version: 91701
GPU Device Name: NVIDIA GeForce RTX 5060 Laptop GPU
GPU Capability: (12, 0)
Total GPU Memory: 7.53 GB
Reserved Memory: 0.00 GB
Allocated Memory: 0.00 GB
========================================

Starting Matrix Multiplication Test...
Matrix Size: 8000x8000
Data allocation time: 0.1096s
Computation time: 0.1193s
Test Complete. GPU is working correctly.
```

