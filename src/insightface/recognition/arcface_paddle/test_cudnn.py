import ctypes
import os

# Try to load the cuDNN library
try:
    paths = [
        "/usr/lib/x86_64-linux-gnu/libcudnn.so.8",
        "/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn.so.8",
        "/workspace/insightface/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn.so.9"
    ]
    
    for path in paths:
        if os.path.exists(path):
            print(f"Found cuDNN at: {path}")
            ctypes.CDLL(path)
            print(f"Successfully loaded {path}")
        else:
            print(f"Not found: {path}")
except Exception as e:
    print(f"Error loading cuDNN: {e}")

# Try to use paddle
try:
    import paddle
    print(f"Paddle version: {paddle.__version__}")
    print(f"CUDA compiled: {paddle.device.is_compiled_with_cuda()}")
    print(f"GPU count: {paddle.device.cuda.device_count()}")
except Exception as e:
    print(f"Error with paddle: {e}")
