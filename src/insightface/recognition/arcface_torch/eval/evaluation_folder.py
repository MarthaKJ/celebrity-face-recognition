#!/usr/bin/env python
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------------------
# Hardcoded paths and settings
# ---------------------------
MODEL_CHECKPOINT = "/workspace/src/insightface/recognition/arcface_torch/work_dirs/customr50/model.pt"
VAL_IMAGE_FOLDER = "/workspace/datasets/manually-annotated/data/val"
EMBEDDING_SIZE = 512

# ---------------------------
# Load InsightFace ArcFace backbone (custom ResNet)
# ---------------------------
from insightface.recognition.arcface_torch.backbones import get_model  # ✅ THIS avoids unipercept

def load_model(device):
    model = get_model("r50", fp16=False, num_features=EMBEDDING_SIZE).to(device)
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location=device)
    if "state_dict_backbone" in checkpoint:
        model.load_state_dict(checkpoint["state_dict_backbone"])
    else:
        model.load_state_dict(checkpoint)
    return model

# ---------------------------
# Utils
# ---------------------------
def load_image(path):
    image = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return transform(image)

def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

def get_data_from_folder(root_dir):
    data = []
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.lower().endswith(".jpg"):
                    data.append({
                        "image_path": os.path.join(folder_path, file),
                        "label": folder
                    })
    return pd.DataFrame(data)

def create_pairs(df, pos_limit=1000, neg_limit=1000, random_state=42):
    pairs = []
    grouped = df.groupby("label")
    rng = np.random.default_rng(random_state)

    # Positive pairs
    for label, group in grouped:
        images = group["image_path"].tolist()
        if len(images) < 2:
            continue
        pos_pairs = [(images[i], images[j], 1)
                     for i in range(len(images)) for j in range(i+1, len(images))]
        sampled = rng.choice(len(pos_pairs), min(pos_limit, len(pos_pairs)), replace=False)
        for idx in sampled:
            pairs.append(pos_pairs[idx])

    # Negative pairs
    all_imgs = df[["image_path", "label"]].values.tolist()
    neg_count = 0
    attempts = 0
    while neg_count < neg_limit and attempts < 10 * neg_limit:
        p1, p2 = rng.choice(all_imgs, 2, replace=False)
        if p1[1] != p2[1]:
            pairs.append((p1[0], p2[0], 0))
            neg_count += 1
        attempts += 1

    return pairs

def evaluate_pairs(model, pairs, device):
    model.eval()
    similarities, labels = [], []
    with torch.no_grad():
        for path1, path2, label in pairs:
            img1 = load_image(path1).unsqueeze(0).to(device)
            img2 = load_image(path2).unsqueeze(0).to(device)
            emb1 = model(img1).cpu().numpy().flatten()
            emb2 = model(img2).cpu().numpy().flatten()
            sim = cosine_similarity(emb1, emb2)
            similarities.append(sim)
            labels.append(label)
    return np.array(similarities), np.array(labels)

def find_best_threshold(similarities, labels):
    best_acc, best_t, best_p, best_r, best_f1 = 0, 0, 0, 0, 0
    for t in np.linspace(similarities.min(), similarities.max(), 100):
        preds = (similarities > t).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_t = t
            best_p = precision_score(labels, preds, zero_division=0)
            best_r = recall_score(labels, preds, zero_division=0)
            best_f1 = f1_score(labels, preds, zero_division=0)
    return best_t, best_acc, best_p, best_r, best_f1

# ---------------------------
# Main
# ---------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    print("✅ Model loaded.")

    df = get_data_from_folder(VAL_IMAGE_FOLDER)
    print(f"📂 Loaded {len(df)} images from {VAL_IMAGE_FOLDER}, {df['label'].nunique()} identities.")

    pairs = create_pairs(df)
    print(f"🔗 Created {len(pairs)} evaluation pairs.")

    sims, labels = evaluate_pairs(model, pairs, device)
    t, acc, prec, rec, f1 = find_best_threshold(sims, labels)

    print("\n📊 Evaluation Metrics:")
    print(f"🔼 Best Threshold: {t:.4f}")
    print(f"✅ Accuracy:       {acc:.4f}")
    print(f"🎯 Precision:      {prec:.4f}")
    print(f"📥 Recall:         {rec:.4f}")
    print(f"📈 F1 Score:       {f1:.4f}")

if __name__ == "__main__":
    main()
