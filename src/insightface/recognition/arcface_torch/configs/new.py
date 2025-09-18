from easydict import EasyDict as edict

config = edict()

# Margin values for ArcFace
config.margin_list = (1.0, 0.5, 0.0)

# Model configuration
config.network = "r50"  # You can change this to another backbone (e.g., "r100") if desired
config.resume = False  # Set to True to resume training from a checkpoint
config.output = None  # Specify output directory for models (optional)
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True  # Use mixed precision training
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 8  # Adjust according to your GPU's memory capacity
config.lr = 0.1  # Set initial learning rate
config.verbose = 10  # Display training status every 2000 iterations
config.dali = False  # Set to True if you use NVIDIA DALI for fast data loading

# Update dataset path and statistics
config.rec = "/workspace/datasets/manually-annotated/data/train"  # Path to your custom dataset (train.rec)
config.num_classes = 312  # Number of unique identities in your dataset
config.num_image = 8992  # Total number of images in your dataset
config.num_epoch = 50
config.warmup_epoch = 0

# Validation targets (you can change these to datasets of your choice)
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]  # These are common datasets used for evaluation

# Save configuration for later reference
