# check_model_pt.py
import torch

model_path = "/workspace/src/insightface/recognition/arcface_torch/work_dirs/customr50/model.pt"
model_obj = torch.load(model_path, map_location="cpu")

print("Loaded object type:", type(model_obj))

if isinstance(model_obj, dict):
    print("🧠 This is a state_dict (parameters only)")
else:
    print("✅ This is a full model (architecture + parameters)")
