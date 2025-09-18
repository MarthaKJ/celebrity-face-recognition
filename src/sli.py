import os
import pickle
import random

def create_verification_pairs(dataset_dir, output_file, num_pos_pairs_per_subject=5, num_negative_pairs=1000):
    # List subdirectories (each representing a subject)
    subject_dirs = [os.path.join(dataset_dir, d) for d in os.listdir(dataset_dir)
                    if os.path.isdir(os.path.join(dataset_dir, d))]
    
    bins = []
    issame_list = []

    # Generate positive pairs
    for subject in subject_dirs:
        # Get all jpg images in the subject folder
        image_files = [os.path.join(subject, f) for f in os.listdir(subject) if f.lower().endswith('.jpg')]
        if len(image_files) < 2:
            continue  # Skip subjects with fewer than 2 images
        for _ in range(num_pos_pairs_per_subject):
            a, b = random.sample(image_files, 2)
            with open(a, 'rb') as fa, open(b, 'rb') as fb:
                bins.append(fa.read())
                bins.append(fb.read())
            issame_list.append(True)

    # Collect images from all subjects for negative pair generation
    subject_images = []
    for subject in subject_dirs:
        images = [os.path.join(subject, f) for f in os.listdir(subject) if f.lower().endswith('.jpg')]
        if images:
            subject_images.append(images)
    
    # Generate negative pairs
    for _ in range(num_negative_pairs):
        # Choose two different subjects randomly
        subj1, subj2 = random.sample(subject_images, 2)
        img1 = random.choice(subj1)
        img2 = random.choice(subj2)
        with open(img1, 'rb') as f1, open(img2, 'rb') as f2:
            bins.append(f1.read())
            bins.append(f2.read())
        issame_list.append(False)
    
    # Save the pairs into a binary file
    with open(output_file, 'wb') as f:
        pickle.dump((bins, issame_list), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(issame_list)} pairs (positive and negative) to {output_file}")

if __name__ == "__main__":
    dataset_dir = "/workspace/datasets/manually-annotated/data/val"  # Replace with the path to your dataset root directory
    output_file = "dataset.bin"           # This file will be created and used by your evaluation script
    create_verification_pairs(dataset_dir, output_file)
