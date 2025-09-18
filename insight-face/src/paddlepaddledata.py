import glob
import imghdr
import os

from PIL import Image


def reorganize_dataset(source_dir, output_dir, target_size=(112, 112), output_format="jpg"):
    # Create output directories
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Initialize label file
    label_file = os.path.join(output_dir, "label.txt")

    # Initialize counters
    person_id = -1
    current_folder = ""
    corrupt_count = 0
    processed_count = 0

    # List of supported image extensions
    supported_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"]

    with open(label_file, "w") as f:
        # Find all image files recursively with multiple extensions
        all_images = []
        for ext in supported_extensions:
            pattern = os.path.join(source_dir, "**", f"*{ext}")
            all_images.extend(glob.glob(pattern, recursive=True))
            # Also try uppercase extension
            pattern = os.path.join(source_dir, "**", f"*{ext.upper()}")
            all_images.extend(glob.glob(pattern, recursive=True))

        # Process sorted images
        for image_path in sorted(all_images):
            # Check if file is a valid image
            try:
                # Verify it's actually an image file
                img_type = imghdr.what(image_path)
                if img_type is None:
                    print(f"Skipping non-image file: {image_path}")
                    corrupt_count += 1
                    continue

                # Try to open and resize the image
                img = Image.open(image_path)

                # Handle special cases
                if img.mode == "RGBA" and output_format.lower() == "jpg":
                    # Convert RGBA to RGB for JPEG (which doesn't support alpha)
                    img = img.convert("RGB")
                elif img.mode != "RGB":
                    # Ensure RGB mode for all images
                    img = img.convert("RGB")

                # Extract folder name (person identity)
                folder_name = os.path.basename(os.path.dirname(image_path))

                # If we encounter a new person folder, increment the ID
                if folder_name != current_folder:
                    current_folder = folder_name
                    person_id += 1

                # Create new filename with standardized extension
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                new_filename = f"{str(person_id).zfill(8)}_{base_name}.{output_format}"
                new_path = os.path.join(images_dir, new_filename)

                # Resize image to standard size
                resized_img = img.resize(target_size, Image.LANCZOS)

                # Save the resized image in the specified format
                resized_img.save(new_path, quality=95 if output_format.lower() == "jpg" else None)
                processed_count += 1

                # Write to label file (absolute image path followed by person ID)
                absolute_path = os.path.abspath(new_path)
                f.write(f"{absolute_path}\t{person_id}\n")

            except (OSError, Image.DecompressionBombError, Image.UnidentifiedImageError) as e:
                print(f"Corrupt or problematic image skipped: {image_path}. Error: {e}")
                corrupt_count += 1
                continue

    print(f"Dataset reorganized successfully at {output_dir}")
    print(f"Total number of unique identities: {person_id + 1}")
    print(f"Total images processed: {processed_count}")
    print(f"Total corrupt images skipped: {corrupt_count}")


# Example usage
source_directory = "/workspace/datasets/manually-annotated/confirmed-cases/Scrapped_images_baidu"
output_directory = "/workspace/datasets/manually-annotated/paddlepaddle_data"
# Set to 112x112 to match model requirements
reorganize_dataset(source_directory, output_directory, target_size=(112, 112), output_format="jpg")
