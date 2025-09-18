import random


def split_dataset(source_dir, train_label_file, val_label_file, val_ratio=0.2):
    """
    Split a dataset into training and validation sets.

    Args:
        source_dir: Directory containing all images
        train_label_file: Output file for training labels
        val_label_file: Output file for validation labels
        val_ratio: Percentage of data to use for validation
    """
    # Get all identity folders
    identities = {}

    # Read your current label file
    with open("/workspace/datasets/manually-annotated/paddlepaddle_data/label.txt") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                img_path = parts[0]
                identity = parts[1]

                if identity not in identities:
                    identities[identity] = []
                identities[identity].append(img_path)

    # Open output files
    train_file = open(train_label_file, "w")
    val_file = open(val_label_file, "w")

    # Split each identity
    for identity, images in identities.items():
        # Shuffle images
        random.shuffle(images)

        # Calculate split point
        split = max(1, int(len(images) * val_ratio))

        # Write validation images
        for img in images[:split]:
            val_file.write(f"{img} {identity}\n")

        # Write training images
        for img in images[split:]:
            train_file.write(f"{img} {identity}\n")

    train_file.close()
    val_file.close()

    print(
        f"Created training file with {sum(len(images) - max(1, int(len(images) * val_ratio)) for images in identities.values())} images"
    )
    print(
        f"Created validation file with {sum(max(1, int(len(images) * val_ratio)) for images in identities.values())} images"
    )


# Execute the split
split_dataset(
    "/workspace/datasets/manually-annotated/paddlepaddle_data/images",
    "/workspace/datasets/manually-annotated/paddlepaddle_data/train_label.txt",
    "/workspace/datasets/manually-annotated/paddlepaddle_data/val_label.txt",
    val_ratio=0.2,
)
