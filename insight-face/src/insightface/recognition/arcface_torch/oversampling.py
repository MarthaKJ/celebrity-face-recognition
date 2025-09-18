import os
import random
import shutil

from tqdm import tqdm

INPUT_DIR = "/workspace/datasets/manually-annotated/torch_data"
OUTPUT_DIR = "/workspace/datasets/manually-annotated/oversampled_torch_data"
TARGET_COUNT = 186

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_images(folder):
    return [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]


for identity in tqdm(sorted(os.listdir(INPUT_DIR)), desc="Oversampling identities"):
    identity_path = os.path.join(INPUT_DIR, identity)
    if not os.path.isdir(identity_path):
        continue

    output_identity_path = os.path.join(OUTPUT_DIR, identity)
    os.makedirs(output_identity_path, exist_ok=True)

    images = get_images(identity_path)
    count = len(images)

    if count == 0:
        continue  # skip empty folders

    # Copy original images
    for img in images:
        src = os.path.join(identity_path, img)
        dst = os.path.join(output_identity_path, img)
        shutil.copy2(src, dst)

    # Oversample if needed
    if count < TARGET_COUNT:
        i = 0
        while len(os.listdir(output_identity_path)) < TARGET_COUNT:
            img_to_copy = random.choice(images)
            new_name = f"dup_{i}_{img_to_copy}"
            shutil.copy2(os.path.join(identity_path, img_to_copy), os.path.join(output_identity_path, new_name))
            i += 1

print(f"\n✅ Done. Oversampled dataset saved at: {OUTPUT_DIR}")
