import os
import shutil

# Path to your original organized images
original_dir = "/workspace/datasets/manually-annotated/confirmed-cases/Scrapped_images_baidu"
# Path for properly formatted directory
target_dir = "/workspace/datasets/manually-annotated/torch_data"

os.makedirs(target_dir, exist_ok=True)

# Counter for new IDs
person_id = 0

# Process each identity folder
for identity_folder in os.listdir(original_dir):
    folder_path = os.path.join(original_dir, identity_folder)
    if not os.path.isdir(folder_path):
        continue

    # Initialize counter for renaming images
    img_count = 0

    # Process each image in the folder
    for img_file in os.listdir(folder_path):
        if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_count += 1

    # If no images are found, skip the folder
    if img_count == 0:
        print(f"Skipping {identity_folder} (no images found)")
        continue

    # Create new folder with proper naming
    new_folder_name = f"0_0_{person_id:07d}"
    new_folder_path = os.path.join(target_dir, new_folder_name)
    os.makedirs(new_folder_path, exist_ok=True)

    # Reset image count for renaming
    img_count = 0

    # Copy and rename each image
    for img_file in os.listdir(folder_path):
        if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            # Format new image name
            new_img_name = f"0_{img_count}.jpg"

            # Copy the image with new name
            shutil.copy(os.path.join(folder_path, img_file), os.path.join(new_folder_path, new_img_name))

            img_count += 1

    print(f"Processed {identity_folder} → {new_folder_name} ({img_count} images)")
    person_id += 1

print(f"Finished organizing {person_id} identities into {target_dir}")
