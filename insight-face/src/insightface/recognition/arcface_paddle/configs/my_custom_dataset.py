from easydict import EasyDict as edict

config = edict()
config.is_static = True
config.backbone = "FresResNet50"
config.classifier = "LargeScaleClassifier"
config.embedding_size = 512
config.model_parallel = True
config.sample_ratio = 0.1
config.loss = "ArcFace"
config.dropout = 0.0
config.lr = 0.01  # for global batch size = 512
config.lr_decay = 0.1
config.weight_decay = 5e-4
config.momentum = 0.9
config.train_unit = "epoch"  # 'step' or 'epoch'
config.warmup_num = 0
config.train_num = 25
config.decay_boundaries = [10, 16, 22]
config.use_synthetic_dataset = False
config.dataset = "custom_dataset"  # Match the name in the script
config.data_dir = "/workspace/datasets/manually-annotated/paddlepaddle_data/images"
config.label_file = "/workspace/datasets/manually-annotated/paddlepaddle_data/train_label.txt"
config.is_bin = False
config.num_classes = 312
config.batch_size = 32
config.num_workers = 8
config.do_validation_while_train = False
config.validation_interval_step = 2000
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
config.logdir = "./log"
config.log_interval_step = 100
config.output = "MS1M_v3_arcface_static_0.1"  # Match the output in the script
config.resume = False
config.checkpoint_dir = None
config.max_num_last_checkpoint = 1
