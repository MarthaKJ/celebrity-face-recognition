# create_lfw_bin.py
import os
import pickle
from io import BytesIO

from PIL import Image
from tqdm import tqdm


def read_pairs(pairs_path):
    pairs = []
    with open(pairs_path) as f:
        for i, line in enumerate(f):
            if i == 0 and line.strip().lower().startswith("name"):
                continue  # skip header
            pair = line.strip().split(",")
            if len(pair) >= 3:
                pairs.append(pair)
    return pairs


def load_image_bytes(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((112, 112))
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


def make_lfw_bin(pairs, data_dir, out_path):
    data = []
    issame = []
    skipped = 0

    for pair in tqdm(pairs, desc="Processing pairs"):
        try:
            if len(pair) == 3:
                name, img1_id, img2_id = pair
                path1 = os.path.join(data_dir, name, f"{name}_{int(img1_id):04d}.jpg")
                path2 = os.path.join(data_dir, name, f"{name}_{int(img2_id):04d}.jpg")
                same = True
            elif len(pair) == 4:
                name1, img1_id, name2, img2_id = pair
                path1 = os.path.join(data_dir, name1, f"{name1}_{int(img1_id):04d}.jpg")
                path2 = os.path.join(data_dir, name2, f"{name2}_{int(img2_id):04d}.jpg")
                same = False
            else:
                skipped += 1
                continue

            if not os.path.exists(path1) or not os.path.exists(path2):
                skipped += 1
                continue

            img1_bytes = load_image_bytes(path1)
            img2_bytes = load_image_bytes(path2)

            data.append(img1_bytes)
            data.append(img2_bytes)
            issame.append(same)

        except Exception as e:
            print(f"❌ Error processing {pair}: {e}")
            skipped += 1

    with open(out_path, "wb") as f:
        pickle.dump((data, issame), f)

    print(f"✅ Saved LFW-compatible .bin to: {out_path}")
    print(f"🖼️  Total pairs: {len(issame)}, Total images: {len(data)}, Skipped pairs: {skipped}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create LFW .bin format")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--pairs-path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    pairs = read_pairs(args.pairs_path)
    make_lfw_bin(pairs, args.data_dir, args.output)
