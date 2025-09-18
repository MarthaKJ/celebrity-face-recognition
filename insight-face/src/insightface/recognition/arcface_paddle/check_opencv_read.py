import cv2
import os

with open('/workspace/datasets/manually-annotated/paddlepaddle_data/train_label.txt', 'r') as f:
    lines = f.readlines()

unreadable = []
for i, line in enumerate(lines):
    img_path = line.strip().split(' ')[0]
    img = cv2.imread(img_path)
    if img is None:
        unreadable.append((i, img_path))
    if i % 500 == 0:
        print(f"Checked {i} images...")

print(f"Checked {len(lines)} images, found {len(unreadable)} unreadable")
for idx, path in unreadable[:10]:  # Print first 10 unreadable
    print(f"Line {idx+1}: {path}")
