import os
import cv2
import random
import pickle
import numpy as np
from glob import glob

# Set your validation folder here
VAL_DIR = "/workspace/datasets/manually-annotated/data/val"
OUTPUT_BIN = "val.bin"

def get_pairs(image_folders, num_pairs=3000):
    same_pairs = []
    diff_pairs = []

    all_classes = list(image_folders.keys())
    while len(same_pairs) < num_pairs // 2:
        cls = random.choice(all_classes)
        if len(image_folders[cls]) < 2:
            continue
        a, b = random.sample(image_folders[cls], 2)
        same_pairs.append((a, b, True))

    while len(diff_pairs) < num_pairs // 2:
        cls1, cls2 = random.sample(all_classes, 2)
        if cls1 == cls2:
            continue
        a = random.choice(image_folders[cls1])
        b = random.choice(image_folders[cls2])
        diff_pairs.append((a, b, False))

    return same_pairs + diff_pairs

def read_image(path, size=(112, 112)):
    img = cv2.imread(path)
    img = cv2.resize(img, size)
    _, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()

def main():
    folders = [d for d in os.listdir(VAL_DIR) if os.path.isdir(os.path.join(VAL_DIR, d))]
    image_folders = {}
    for cls in folders:
        paths = glob(os.path.join(VAL_DIR, cls, "*.jpg"))
        if len(paths) > 1:
            image_folders[cls] = paths

    pairs = get_pairs(image_folders)
    bins = []
    issame_list = []

    for a, b, same in pairs:
        bins.append(read_image(a))
        bins.append(read_image(b))
        issame_list.append(same)

    with open(OUTPUT_BIN, 'wb') as f:
        pickle.dump((bins, issame_list), f)
    print(f"✅ Saved {len(issame_list)} pairs to {OUTPUT_BIN}")

if __name__ == "__main__":
    main()
