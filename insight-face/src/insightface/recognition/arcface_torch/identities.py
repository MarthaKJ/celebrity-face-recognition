import os

root_dir = "/workspace/datasets/manually-annotated/oversampled_torch_data"
valid_exts = (".jpg", ".jpeg", ".png", ".bmp")  # You can add more if needed

class_counts = {}

for folder in sorted(os.listdir(root_dir)):
    folder_path = os.path.join(root_dir, folder)
    if not os.path.isdir(folder_path):
        continue

    image_count = sum(
        1 for file in os.listdir(folder_path)
        if file.lower().endswith(valid_exts)
    )
    class_counts[folder] = image_count

# Print results
for class_name, count in class_counts.items():
    print(f"{class_name:<25} {count} images")

print(f"\nTotal classes: {len(class_counts)}")
print(f"Min images in a class: {min(class_counts.values())}")
print(f"Max images in a class: {max(class_counts.values())}")
