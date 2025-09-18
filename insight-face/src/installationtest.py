import insightface
import onnxruntime

# Verify InsightFace Installation
print("✅ InsightFace installed successfully!")

# Check ONNX Runtime
providers = onnxruntime.get_available_providers()
print("ONNX Runtime Providers:", providers)

if "CUDAExecutionProvider" in providers:
    print("✅ GPU is available for ONNX Runtime")
else:
    print("❌ GPU not detected. Check installation.")
