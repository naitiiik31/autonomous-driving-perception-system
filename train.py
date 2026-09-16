#!/usr/bin/env python3
"""
🏋️ YOLO Car Detection — Training & Fine-Tuning Script
=====================================================

Fine-tune the pre-trained YOLO v2 model on custom driving datasets.
Supports transfer learning with configurable backbone freezing.

Usage:
    # Create dataset structure
    python train.py --setup_data
    
    # Train with default settings
    python train.py --data data/ --epochs 50
    
    # Train with frozen backbone (feature extraction only)
    python train.py --data data/ --freeze_backbone --epochs 30
    
    # Train with augmentation and plots
    python train.py --data data/ --augment --plot
"""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import print_banner
from src.trainer import YOLOTrainer, TrainingConfig, create_dataset_structure
from src.augmentation import AugmentationPipeline


def main():
    parser = argparse.ArgumentParser(
        description="🏋️ YOLO v2 Fine-Tuning & Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Training Strategies:
  Feature Extraction: --freeze_backbone (freeze DarkNet-19, train head only)
  Fine-Tuning:        Default (unfreeze last layers + head)
  Full Training:      --unfreeze_all (train all layers, needs more data)

Dataset Structure:
  data/
  ├── train/images/     ← Training images (.jpg)
  ├── train/labels/     ← YOLO format annotations (.txt)
  ├── val/images/       ← Validation images
  ├── val/labels/       ← Validation annotations
  └── classes.txt       ← Class names (one per line)

YOLO Annotation Format (per .txt file):
  <class_id> <x_center> <y_center> <width> <height>
  (all normalized to [0, 1])

Examples:
  python train.py --setup_data                     # Create dataset dirs
  python train.py --data data/ --epochs 50         # Train
  python train.py --data data/ --freeze_backbone   # Feature extraction
        """,
    )
    
    # Data
    parser.add_argument(
        "--data", type=str, default="data",
        help="Path to dataset directory (default: data/)",
    )
    parser.add_argument(
        "--setup_data", action="store_true",
        help="Create dataset directory structure and exit",
    )
    
    # Training params
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Training batch size (default: 8)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Initial learning rate (default: 0.0001)",
    )
    parser.add_argument(
        "--num_classes", type=int, default=9,
        help="Number of object classes (default: 9 driving classes)",
    )
    
    # Transfer learning
    parser.add_argument(
        "--freeze_backbone", action="store_true",
        help="Freeze DarkNet-19 backbone (feature extraction mode)",
    )
    parser.add_argument(
        "--unfreeze_all", action="store_true",
        help="Unfreeze all layers (full training mode)",
    )
    parser.add_argument(
        "--pretrained", type=str, default="model_data",
        help="Path to pre-trained model (default: model_data/)",
    )
    
    # Augmentation
    parser.add_argument(
        "--augment", action="store_true", default=True,
        help="Enable data augmentation (default: True)",
    )
    parser.add_argument(
        "--no_augment", action="store_true",
        help="Disable data augmentation",
    )
    
    # Output
    parser.add_argument(
        "--output", type=str, default="training_output",
        help="Output directory (default: training_output/)",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate training curve plots",
    )
    parser.add_argument(
        "--export", type=str, default=None,
        help="Export trained model to this directory",
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    # Setup data mode
    if args.setup_data:
        create_dataset_structure(args.data)
        return
    
    # Read class names
    class_names = []
    classes_file = os.path.join(args.data, "classes.txt")
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            class_names = [line.strip() for line in f if line.strip()]
        args.num_classes = len(class_names)
        print(f"📋 Loaded {len(class_names)} classes from {classes_file}")
    
    # Build training config
    config = TrainingConfig(
        train_images_dir=os.path.join(args.data, "train", "images"),
        train_annotations=os.path.join(args.data, "train", "labels"),
        val_images_dir=os.path.join(args.data, "val", "images"),
        val_annotations=os.path.join(args.data, "val", "labels"),
        num_classes=args.num_classes,
        class_names=class_names,
        pretrained_weights=args.pretrained,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        freeze_backbone=args.freeze_backbone,
        augmentation=not args.no_augment,
    )
    
    # Initialize trainer
    trainer = YOLOTrainer(config)
    
    # Initialize augmentation pipeline if enabled
    if config.augmentation:
        augmenter = AugmentationPipeline.driving_preset()
        print(f"[TRAIN] 🎨 Data augmentation: ENABLED (driving preset)")
    
    # Train
    history = trainer.train()
    
    # Plot training curves
    if args.plot:
        trainer.plot_training_curves()
    
    # Export model
    if args.export:
        trainer.export_model(args.export)
    
    print(f"\n✅ Training complete!")
    print(f"   Output directory : {args.output}")
    print(f"   Best val loss    : {trainer.best_val_loss:.4f}")
    print(f"   Total epochs     : {len(history)}")
    
    if args.export:
        print(f"   Exported to      : {args.export}")


if __name__ == "__main__":
    main()
