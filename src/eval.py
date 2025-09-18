import os
import sys
import torch
import pickle
import numpy as np
import sklearn
from tqdm import tqdm

# Add your arcface_torch module to PYTHONPATH
sys.path.append("/workspace/src/insightface/recognition")

from arcface_torch.backbones import get_model
from verification import load_bin, test  # from the script you pasted earlier

# ==== Config ====
MODEL_PATH = "/workspace/src/insightface/recognition/arcface_torch/work_dirs/new/model.pt"
VAL_BIN_PATH = "/workspace/datasets/manually-annotated/data/val.bin"
IMAGE_SIZE = [112, 112]
BATCH_SIZE = 32
EMBEDDING_SIZE = 512
NETWORK = "r50"
NFOLDS = 10

# ==== Load Model ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(NETWORK, EMBEDDING_SIZE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
model.to(device)
print(f"✅ Loaded model from {MODEL_PATH}")

# ==== Load val.bin ====
if not os.path.isfile(VAL_BIN_PATH):
    raise FileNotFoundError(f"val.bin not found at {VAL_BIN_PATH}")
val_set = load_bin(VAL_BIN_PATH, image_size=IMAGE_SIZE)

# ==== Run Verification ====
print("🚀 Running verification...")
acc1, std1, acc2, std2, xnorm, _ = test(val_set, model, batch_size=BATCH_SIZE, nfolds=NFOLDS)

# ==== Results ====
print("\n📊 Verification Results")
print(f"XNorm: {xnorm:.5f}")
print(f"Accuracy: {acc1:.5f} ± {std1:.5f}")
print(f"Accuracy (flip): {acc2:.5f} ± {std2:.5f}")
