import os
import traceback

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError  # Added PIL for better corruption detection

from insightface.app import FaceAnalysis

# Initialize RetinaFace detector with CPU only since CUDA is not available
face_analyzer = FaceAnalysis(providers=["CPUExecutionProvider"])
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

# Path to your renamed folders
source_dir = "/workspace/datasets/manually-annotated/torch_data"

# Statistics tracking
processed_count = 0
corrupt_count = 0
no_face_count = 0

for identity_folder in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, identity_folder)
    if not os.path.isdir(folder_path):
        continue

    folder_processed = 0
    folder_corrupt = 0
    folder_no_face = 0

    print(f"Processing folder: {identity_folder}")

    for img_file in os.listdir(folder_path):
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(folder_path, img_file)

        try:
            # Try to open with PIL to catch corrupt images
            try:
                with Image.open(img_path) as img:
                    img.verify()  # This raises an error if the image is incomplete/corrupt
            except (OSError, UnidentifiedImageError):
                print(f"Corrupt or premature image detected: {img_path}")
                os.remove(img_path)
                print(f"Removed corrupt file: {img_path}")
                corrupt_count += 1
                folder_corrupt += 1
                continue

            # Try to read image with OpenCV
            image = cv2.imread(img_path)

            # Check if image is corrupt or empty
            if image is None or image.size == 0:
                print(f"Corrupt or empty image detected: {img_path}")
                os.remove(img_path)
                print(f"Removed corrupt file: {img_path}")
                corrupt_count += 1
                folder_corrupt += 1
                continue

            # Additional corruption check - try to access pixel data
            height, width = image.shape[:2]
            if height <= 0 or width <= 0:
                print(f"Invalid dimensions in image: {img_path}")
                os.remove(img_path)
                print(f"Removed corrupt file: {img_path}")
                corrupt_count += 1
                folder_corrupt += 1
                continue

            # Detect faces using RetinaFace
            faces = face_analyzer.get(image)

            if len(faces) == 0:
                print(f"No face detected in {img_path}, removing")
                os.remove(img_path)
                no_face_count += 1
                folder_no_face += 1
                continue

            # Get the face with highest detection score
            faces = sorted(faces, key=lambda x: x.det_score, reverse=True)
            face = faces[0]

            # Get bounding box
            bbox = face.bbox.astype(np.int32)
            x1, y1, x2, y2 = bbox

            # Ensure valid bounds within image dimensions
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            # Check if we have a valid box after clamping
            if x2 <= x1 or y2 <= y1:
                print(f"Invalid face bounds in {img_path}, removing")
                os.remove(img_path)
                corrupt_count += 1
                folder_corrupt += 1
                continue

            # Extract face
            face_image = image[y1:y2, x1:x2]

            # Resize to 112x112 for ArcFace model
            face_image = cv2.resize(face_image, (112, 112))

            # Save the aligned face back to the same location
            cv2.imwrite(img_path, face_image)
            processed_count += 1
            folder_processed += 1

            # Print progress every 10 images
            if folder_processed % 10 == 0:
                print(f"Processed {folder_processed} images in {identity_folder}")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            print(traceback.format_exc())
            os.remove(img_path)
            print(f"Removed problematic file: {img_path}")
            corrupt_count += 1
            folder_corrupt += 1

    print(
        f"Completed folder {identity_folder}: {folder_processed} aligned, {folder_corrupt} corrupt removed, {folder_no_face} no-face removed"
    )

print("\nProcessing complete:")
print(f"Total images processed successfully: {processed_count}")
print(f"Total corrupt images removed: {corrupt_count}")
print(f"Total images with no face removed: {no_face_count}")
