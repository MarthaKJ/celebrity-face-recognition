import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from arcface_torch.backbones import get_model
from arcface_torch.loss import ArcFace
from easydict import EasyDict as edict

# ========== CONFIG ==========
config = edict()
config.network = "r50"
config.embedding_size = 512
config.margin_list = (1.0, 0.5, 0.0)
config.output = "./checkpoints"
config.num_epoch = 50
config.batch_size = 64
config.lr = 0.1
config.momentum = 0.9
config.weight_decay = 5e-4
config.train_dir = "/workspace/datasets/manually-annotated/data/train"
config.val_dir = "/workspace/datasets/manually-annotated/data/val"
config.fp16 = False
config.save_best = True
os.makedirs(config.output, exist_ok=True)

# ========== DEVICE ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== DATA ==========
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

train_dataset = datasets.ImageFolder(config.train_dir, transform=transform)
val_dataset = datasets.ImageFolder(config.val_dir, transform=transform)
config.num_classes = len(train_dataset.classes)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

# ========== MODEL ==========
backbone = get_model(config.network, config.embedding_size).to(device)
arcface_head = ArcFace(
    in_features=config.embedding_size,
    out_features=config.num_classes,
    margin=config.margin_list[0],
    scale=64
).to(device)

optimizer = torch.optim.SGD(
    list(backbone.parameters()) + list(arcface_head.parameters()),
    lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay
)
criterion = nn.CrossEntropyLoss()

# ========== TRAIN ==========
best_f1 = 0.0

for epoch in range(config.num_epoch):
    backbone.train()
    arcface_head.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in tqdm(train_loader, desc=f"[Epoch {epoch+1}] Training"):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        embeddings = backbone(imgs)
        logits = arcface_head(embeddings, labels)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    print(f"📊 Epoch {epoch+1}: Train Loss: {running_loss:.4f} | Train Accuracy: {train_acc:.4f}")

    # ========== VALIDATION ==========
    backbone.eval()
    arcface_head.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"[Epoch {epoch+1}] Validating"):
            imgs = imgs.to(device)
            embeddings = backbone(imgs)
            logits = arcface_head(embeddings, labels.to(device))
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"✅ Val Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    # ========== SAVE BEST MODEL ==========
    if config.save_best and f1 > best_f1:
        best_f1 = f1
        torch.save({
            'epoch': epoch + 1,
            'backbone': backbone.state_dict(),
            'arcface_head': arcface_head.state_dict(),
            'f1': best_f1
        }, os.path.join(config.output, "best_model.pth"))
        print(f"💾 Saved best model (F1: {best_f1:.4f})")

print("🎉 Training complete!")
