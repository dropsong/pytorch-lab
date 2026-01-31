import torch
import time
import platform

def print_system_info():
    print("="*40)
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python Version: {platform.python_version()}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Capability: {torch.cuda.get_device_capability(0)}")
        
        # 显存信息
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print(f"Total GPU Memory: {t / 1024**3:.2f} GB")
        print(f"Reserved Memory: {r / 1024**3:.2f} GB")
        print(f"Allocated Memory: {a / 1024**3:.2f} GB")
    else:
        print("!! NO GPU DETECTED !!")
    print("="*40)

def test_computation():
    if not torch.cuda.is_available():
        return
    
    print("\nStarting Matrix Multiplication Test...")
    device = torch.device("cuda")
    
    # 创建两个较大的随机矩阵 (适应你的 8GB 显存，不要太大导致 OOM)
    size = 8000 
    print(f"Matrix Size: {size}x{size}")
    
    # 移动到 GPU
    start_time = time.time()
    x = torch.rand(size, size, device=device)
    y = torch.rand(size, size, device=device)
    transfer_time = time.time()
    print(f"Data allocation time: {transfer_time - start_time:.4f}s")
    
    # 矩阵乘法
    torch.cuda.synchronize() # 等待 GPU 准备好
    compute_start = time.time()
    z = torch.matmul(x, y)
    torch.cuda.synchronize() # 等待计算完成
    compute_end = time.time()
    
    print(f"Computation time: {compute_end - compute_start:.4f}s")
    print("Test Complete. GPU is working correctly.")

if __name__ == "__main__":
    print_system_info()
    test_computation()