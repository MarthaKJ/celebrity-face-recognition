import os
import sys
import ctypes

# Preload the cuDNN library
try:
    cudnn_lib = ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libcudnn.so.8")
    print("Successfully preloaded cuDNN")
except Exception as e:
    print(f"Failed to preload cuDNN: {e}")
    sys.exit(1)

# Run the original shell script
os.system("sh scripts/train_static.sh")
