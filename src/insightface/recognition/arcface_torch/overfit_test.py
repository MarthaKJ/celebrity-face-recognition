import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

# === Simple model for small test ===
class SimpleBackbone(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 112 * 112, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_size),
        )

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.normalize(x)
        return x

def main():
    data_path = "/workspace/datasets/manually-annotated/data/val"  # Folder with 2 identities
    batch_size = 4
    num_epochs = 100
    embedding_size = 128
    num_classes = len(os.listdir(data_path))

    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    backbone = SimpleBackbone(embedding_size=embedding_size).cuda()
    classifier = nn.Linear(embedding_size, num_classes).cuda()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(backbone.parameters()) + list(classifier.parameters()), lr=0.001)

    for epoch in range(num_epochs):
        correct, total = 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.cuda(), labels.cuda()

            embeddings = backbone(imgs)
            logits = classifier(embeddings)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f} | Accuracy: {acc:.2f}%")

        if acc == 100.0:
            print("🎯 Model overfit the tiny dataset! Your setup works.")
            break

if __name__ == "__main__":
    main()
