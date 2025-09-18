import os

with open('/workspace/datasets/manually-annotated/paddlepaddle_data/train_label.txt', 'r') as f:
    lines = f.readlines()

missing = []
for i, line in enumerate(lines):
    img_path = line.strip().split(' ')[0]
    if not os.path.exists(img_path):
        missing.append((i, img_path))

print(f"Checked {len(lines)} images, found {len(missing)} missing")
for idx, path in missing[:10]:  # Print first 10 missing
    print(f"Line {idx+1}: {path}")
