from easydict import EasyDict as edict

config = edict()

# Margin values for ArcFace
config.margin_list = (1.0, 0.0, 0.0)

# Model configuration
config.network = "r50"  # You can change this to another backbone (e.g., "r100") if desired
config.resume = False  # Set to True to resume training from a checkpoint
config.output = None  # Specify output directory for models (optional)
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True  # Use mixed precision training
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 32  # Adjust according to your GPU's memory capacity
config.lr = 0.001  # Set initial learning rate
config.verbose = 10  # Display training status every 2000 iterations
config.dali = False  # Set to True if you use NVIDIA DALI for fast data loading

# Update dataset path and statistics
config.rec = "/workspace/datasets/manually-annotated/oversampled_torch_data"  # Path to your custom dataset (train.rec)
config.num_classes = 313  # Number of unique identities in your dataset
config.num_image = 58218  # Total number of images in your dataset
config.num_epoch = 100
config.warmup_epoch = 5

# Validation targets (you can change these to datasets of your choice)
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]  # These are common datasets used for evaluation

# Save configuration for later reference
