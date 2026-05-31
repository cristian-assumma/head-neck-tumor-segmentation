import os
import argparse
from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.registry import RUNNERS
from mmseg.apis import init_model

# Import custom dataset to register it in MMSegmentation
from dataset import TumorHNDataset
from utils import compute_class_weights

def parse_args():
    parser = argparse.ArgumentParser(description='Train U-Net for Head & Neck Tumor Segmentation')
    parser.add_argument('--data-root', type=str, default='../../data/Preprocessed_dataset', help='Path to dataset')
    parser.add_argument('--base-config', type=str, required=True, help='Path to MMSegmentation base config file')
    parser.add_argument('--work-dir', type=str, default='../../models/unet_training', help='Path to save logs and models')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=80, help='Number of training epochs')
    parser.add_argument('--pretrained', type=str, default=None, help='Path to pre-trained weights (e.g., Cityscapes)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("[*] Loading base configuration...")
    cfg = Config.fromfile(args.base_config)
    
    # Base configuration setup
    cfg.dataset_type = 'TumorHNDataset'
    cfg.data_root = args.data_root
    cfg.crop_size = (256, 256)
    cfg.device = 'cuda'

    # Model Architecture Modifications (Replace SyncBN with BN for local training)
    cfg.model.auxiliary_head.norm_cfg = dict(type='BN')
    cfg.model.backbone.norm_cfg = dict(type='BN')
    cfg.model.decode_head.norm_cfg = dict(type='BN')
    cfg.norm_cfg = dict(type='BN')

    cfg.model.auxiliary_head.num_classes = 3
    cfg.model.decode_head.num_classes = 3

    cfg.data_preprocessor.size = cfg.crop_size
    cfg.model.data_preprocessor.size = cfg.crop_size
    
    # Disable padding label 255 (background is 0)
    cfg.model.data_preprocessor.pad_val = 0
    cfg.model.data_preprocessor.seg_pad_val = 0
    cfg.data_preprocessor.pad_val = 0
    cfg.data_preprocessor.seg_pad_val = 0

    # Composite Loss Configuration
    # We use hardcoded weights computed from the dataset to save time, or compute them dynamically:
    # class_weights = compute_class_weights(args.data_root)
    class_weights = [0.02970294, 1.48514853, 1.48514853]
    
    cfg.model.decode_head.loss_decode = [
        dict(type='CrossEntropyLoss', class_weight=class_weights, loss_weight=0.3),
        dict(type='LovaszLoss', per_image=True, loss_weight=0.3),
        dict(type='TverskyLoss', alpha=0.3, beta=0.7, loss_weight=0.4)
    ]

    # Data Pipelines
    cfg.train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', reduce_zero_label=False),
        dict(type='Resize', scale=(256, 256), keep_ratio=False),
        dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.95),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PackSegInputs'),
    ]

    cfg.test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='Resize', scale=(256, 256), keep_ratio=False),
        dict(type='LoadAnnotations', reduce_zero_label=False),
        dict(type='PackSegInputs')
    ]

    # Dataloader Configuration
    cfg.train_dataloader.batch_size = args.batch_size
    cfg.train_dataloader.dataset.type = cfg.dataset_type
    cfg.train_dataloader.dataset.data_root = args.data_root
    cfg.train_dataloader.dataset.data_prefix.img_path = os.path.join('train', 'images')
    cfg.train_dataloader.dataset.data_prefix.seg_map_path = os.path.join('train', 'masks')
    cfg.train_dataloader.dataset.pipeline = cfg.train_pipeline

    cfg.val_dataloader.batch_size = args.batch_size
    cfg.val_dataloader.dataset.type = cfg.dataset_type
    cfg.val_dataloader.dataset.data_root = args.data_root
    cfg.val_dataloader.dataset.data_prefix.img_path = os.path.join('val', 'images')
    cfg.val_dataloader.dataset.data_prefix.seg_map_path = os.path.join('val', 'masks')
    cfg.val_dataloader.dataset.pipeline = cfg.test_pipeline

    # Training Parameters
    train_length = len(os.listdir(os.path.join(args.data_root, 'train', 'images')))
    iter_for_epoch = round(train_length / args.batch_size)
    max_iters = args.epochs * iter_for_epoch
    interval_val = iter_for_epoch  # Refresh validation metrics every epoch

    cfg.train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=interval_val)
    cfg.val_cfg = dict(type='ValLoop')
    cfg.test_cfg = dict(type='TestLoop')
    cfg.workflow = [('train', 1), ('val', 1)]

    # Hooks and Checkpoints
    cfg.default_hooks.checkpoint = dict(
        type='CheckpointHook',
        save_best='mIoU',
        rule='greater',
        interval=interval_val
    )

    if args.pretrained:
        cfg.load_from = args.pretrained
        print(f"[*] Loading pre-trained weights from: {args.pretrained}")
    else:
        cfg.load_from = None

    # Visualization and Output setup
    cfg.visualizer.save_dir = args.work_dir
    cfg.visualizer.classes = TumorHNDataset.METAINFO['classes']
    cfg.visualizer.palette = TumorHNDataset.METAINFO['palette']

    cfg.update(dict(work_dir=args.work_dir))
    cfg.update(dict(launcher='none'))

    print("[*] Starting training runner...")
    runner = RUNNERS.build(cfg)
    runner.train()

if __name__ == '__main__':
    main()
