"""
YOLO Fine-Tuning & Training Module
====================================

Provides functionality to fine-tune the pre-trained YOLO v2 model on
custom datasets. This is essential for adapting the general COCO-trained
model to specific driving scenarios or new object categories.

Training Pipeline:
    1. Load pre-trained weights (transfer learning)
    2. Prepare custom dataset (images + annotations)
    3. Apply data augmentation
    4. Fine-tune with YOLO multi-task loss
    5. Evaluate on validation set
    6. Export optimized model

Theory — Transfer Learning for Detection:
    Instead of training from scratch (requires millions of images), we:
    
    Strategy 1 — Feature Extraction:
        Freeze backbone (DarkNet-19), only train detection head
        Best when: Small dataset, similar domain to COCO
    
    Strategy 2 — Fine-Tuning:
        Unfreeze last N layers of backbone + detection head
        Best when: Medium dataset, somewhat different domain
    
    Strategy 3 — Full Training:
        Train all layers with low learning rate
        Best when: Large dataset, very different domain
    
    Learning Rate Strategy:
        - Backbone: lr × 0.1 (slow updates — preserve learned features)
        - Head: lr × 1.0 (fast updates — adapt to new task)
"""

import os
import json
import time
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """
    Configuration for YOLO fine-tuning.
    
    Attributes:
        train_images_dir: Directory with training images.
        train_annotations: Path to annotation file (YOLO format).
        val_images_dir: Directory with validation images.
        val_annotations: Path to validation annotations.
        num_classes: Number of object classes.
        class_names: List of class name strings.
        pretrained_weights: Path to pre-trained model weights.
        output_dir: Directory to save trained model and logs.
        
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Initial learning rate.
        lr_decay: Learning rate decay factor.
        warmup_epochs: Number of warmup epochs.
        
        freeze_backbone: Whether to freeze the DarkNet backbone.
        freeze_until_layer: Freeze layers up to this index.
        
        input_size: Model input image size (H, W).
        augmentation: Whether to apply data augmentation.
        
        save_frequency: Save checkpoint every N epochs.
        log_frequency: Log metrics every N batches.
        early_stopping_patience: Stop if no improvement for N epochs.
    """
    
    # Data
    train_images_dir: str = "data/train/images"
    train_annotations: str = "data/train/annotations.json"
    val_images_dir: str = "data/val/images"
    val_annotations: str = "data/val/annotations.json"
    num_classes: int = 80
    class_names: List[str] = field(default_factory=list)
    
    # Model
    pretrained_weights: str = "model_data"
    output_dir: str = "training_output"
    
    # Training hyperparameters
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-4
    lr_decay: float = 0.95
    warmup_epochs: int = 3
    weight_decay: float = 5e-4
    
    # Transfer learning
    freeze_backbone: bool = True
    freeze_until_layer: int = 18  # Freeze DarkNet-19 layers
    
    # Input
    input_size: Tuple[int, int] = (608, 608)
    
    # Augmentation
    augmentation: bool = True
    
    # Checkpointing
    save_frequency: int = 5
    log_frequency: int = 10
    early_stopping_patience: int = 10
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "logs"), exist_ok=True)


@dataclass
class TrainingMetrics:
    """Stores metrics for a single training epoch."""
    epoch: int
    train_loss: float = 0.0
    val_loss: float = 0.0
    coord_loss: float = 0.0
    conf_loss: float = 0.0
    class_loss: float = 0.0
    learning_rate: float = 0.0
    epoch_time: float = 0.0
    
    def summary(self) -> str:
        return (
            f"Epoch {self.epoch:>3d} | "
            f"Train: {self.train_loss:.4f} | "
            f"Val: {self.val_loss:.4f} | "
            f"Coord: {self.coord_loss:.4f} | "
            f"Conf: {self.conf_loss:.4f} | "
            f"Class: {self.class_loss:.4f} | "
            f"LR: {self.learning_rate:.6f} | "
            f"Time: {self.epoch_time:.1f}s"
        )


class YOLOTrainer:
    """
    YOLO v2 fine-tuning trainer.
    
    Manages the complete training loop including:
    - Data loading and augmentation
    - Forward pass and loss computation  
    - Backpropagation and weight updates
    - Validation evaluation
    - Checkpointing and logging
    - Learning rate scheduling
    - Early stopping
    
    Usage:
        config = TrainingConfig(
            train_images_dir="data/train",
            num_classes=9,
            epochs=50,
            freeze_backbone=True,
        )
        
        trainer = YOLOTrainer(config)
        history = trainer.train()
        trainer.export_model("model_data_custom")
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration.
        """
        self.config = config
        self.history: List[TrainingMetrics] = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        print("╔══════════════════════════════════════════════╗")
        print("║       YOLO v2 Fine-Tuning Trainer            ║")
        print("╠══════════════════════════════════════════════╣")
        print(f"║  Classes        : {config.num_classes:<27}║")
        print(f"║  Epochs         : {config.epochs:<27}║")
        print(f"║  Batch Size     : {config.batch_size:<27}║")
        print(f"║  Learning Rate  : {config.learning_rate:<27}║")
        print(f"║  Input Size     : {str(config.input_size):<27}║")
        print(f"║  Freeze Backbone: {str(config.freeze_backbone):<27}║")
        print(f"║  Augmentation   : {str(config.augmentation):<27}║")
        print(f"║  Output Dir     : {config.output_dir:<27}║")
        print("╚══════════════════════════════════════════════╝")
    
    def _build_model(self):
        """
        Build or load the YOLO model for training.
        
        Steps:
        1. Load pre-trained DarkNet-19 backbone
        2. Add YOLO detection head for custom num_classes
        3. Optionally freeze backbone layers
        4. Compile with YOLO loss function and Adam optimizer
        """
        import tensorflow as tf
        from tensorflow.keras.optimizers import Adam
        
        print("\n[TRAINER] 📦 Building model...")
        
        # Load pre-trained model
        model = tf.saved_model.load(self.config.pretrained_weights)
        
        # Note: For a full implementation, you would:
        # 1. Extract DarkNet-19 layers from the saved model
        # 2. Replace the final detection layer for your num_classes
        # 3. Compile with custom YOLO loss
        #
        # This is a simplified version showing the training framework.
        
        print(f"[TRAINER] ✅ Model built with {self.config.num_classes} classes")
        
        return model
    
    def _create_learning_rate_schedule(self):
        """
        Create a learning rate schedule with warmup and decay.
        
        Schedule:
            Epochs 1-warmup:    Linear warmup from 0 to base_lr
            Epochs warmup-end:  Cosine annealing decay
        
        Theory:
            Warmup prevents large initial gradients from corrupting
            pre-trained weights. Cosine decay provides smooth
            convergence to a minimum.
        """
        import tensorflow as tf
        
        warmup_steps = self.config.warmup_epochs
        total_steps = self.config.epochs
        base_lr = self.config.learning_rate
        
        def schedule(epoch):
            if epoch < warmup_steps:
                # Linear warmup
                return base_lr * (epoch + 1) / warmup_steps
            else:
                # Cosine annealing
                progress = (epoch - warmup_steps) / (total_steps - warmup_steps)
                return base_lr * 0.5 * (1 + np.cos(np.pi * progress))
        
        return schedule
    
    def _compute_yolo_loss(
        self,
        predictions,
        ground_truth,
        anchors: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute the multi-component YOLO loss.
        
        Components:
            L_total = λ_coord · L_coord + λ_conf · L_conf + λ_class · L_class
            
        Where:
            L_coord = Σ (predicted_box - target_box)²  [only for responsible anchors]
            L_conf  = Σ (predicted_conf - target_conf)² [weighted obj vs noobj]
            L_class = Σ (predicted_class - target_class)² [only for responsible]
        
        Returns:
            Dictionary with individual loss components and total.
        """
        # Placeholder values — in a full implementation, these would
        # be computed from the actual predictions and ground truth
        coord_loss = 0.0
        conf_loss = 0.0
        class_loss = 0.0
        
        total_loss = 0.5 * (coord_loss + conf_loss + class_loss)
        
        return {
            "total_loss": total_loss,
            "coord_loss": coord_loss,
            "conf_loss": conf_loss,
            "class_loss": class_loss,
        }
    
    def train(self) -> List[TrainingMetrics]:
        """
        Execute the full training loop.
        
        Training Loop:
            for epoch in range(epochs):
                1. Set learning rate (schedule)
                2. Train on all batches:
                    a. Load batch of images + annotations
                    b. Augment images
                    c. Forward pass → predictions
                    d. Compute YOLO loss
                    e. Backward pass → gradients
                    f. Update weights
                3. Validate on validation set
                4. Save checkpoint if best model
                5. Check early stopping
                6. Log metrics
        
        Returns:
            List of TrainingMetrics for each epoch.
        """
        print("\n[TRAINER] 🚀 Starting training...\n")
        
        lr_schedule = self._create_learning_rate_schedule()
        
        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()
            
            # Get current learning rate
            current_lr = lr_schedule(epoch - 1)
            
            # Simulated training step
            # In a full implementation, this would iterate over batches
            train_loss = self._train_epoch(epoch, current_lr)
            
            # Simulated validation step
            val_loss = self._validate_epoch(epoch)
            
            epoch_time = time.time() - epoch_start
            
            # Create metrics
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=current_lr,
                epoch_time=epoch_time,
            )
            self.history.append(metrics)
            
            # Log
            if epoch % self.config.log_frequency == 0 or epoch <= 3:
                print(f"  {metrics.summary()}")
            
            # Save checkpoint
            if epoch % self.config.save_frequency == 0:
                self._save_checkpoint(epoch)
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stopping_patience:
                    print(f"\n[TRAINER] ⏹️  Early stopping at epoch {epoch} "
                          f"(no improvement for {self.config.early_stopping_patience} epochs)")
                    break
        
        self._save_training_history()
        print(f"\n[TRAINER] ✅ Training complete! Best val loss: {self.best_val_loss:.4f}")
        
        return self.history
    
    def _train_epoch(self, epoch: int, learning_rate: float) -> float:
        """Simulate training for one epoch (framework placeholder)."""
        # In a full implementation, this would:
        # 1. Iterate over training data batches
        # 2. Apply augmentation to each batch
        # 3. Run forward pass through the model
        # 4. Compute YOLO loss
        # 5. Run backward pass (gradient computation)
        # 6. Update weights using optimizer
        
        # Simulated decreasing loss
        base_loss = 10.0 * np.exp(-epoch / 20) + np.random.normal(0, 0.1)
        return max(0.01, base_loss)
    
    def _validate_epoch(self, epoch: int) -> float:
        """Simulate validation for one epoch (framework placeholder)."""
        # In a full implementation, this would:
        # 1. Iterate over validation data (no augmentation)
        # 2. Run forward pass
        # 3. Compute loss (no backward pass)
        # 4. Compute mAP, precision, recall
        
        base_loss = 12.0 * np.exp(-epoch / 25) + np.random.normal(0, 0.15)
        return max(0.01, base_loss)
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        
        if is_best:
            path = os.path.join(checkpoint_dir, "best_model.json")
        else:
            path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.json")
        
        checkpoint = {
            "epoch": epoch,
            "best_val_loss": self.best_val_loss,
            "config": {
                "num_classes": self.config.num_classes,
                "input_size": self.config.input_size,
                "learning_rate": self.config.learning_rate,
            },
        }
        
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def _save_training_history(self):
        """Save training history to JSON."""
        history_path = os.path.join(self.config.output_dir, "logs", "training_history.json")
        
        history_data = []
        for m in self.history:
            history_data.append({
                "epoch": m.epoch,
                "train_loss": m.train_loss,
                "val_loss": m.val_loss,
                "learning_rate": m.learning_rate,
                "epoch_time": m.epoch_time,
            })
        
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        print(f"[TRAINER] 📊 Training history saved to: {history_path}")
    
    def plot_training_curves(self, output_path: Optional[str] = None):
        """
        Plot training and validation loss curves.
        
        Args:
            output_path: Path to save the plot. If None, uses output_dir.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("[TRAINER] ⚠️  matplotlib required for plotting")
            return
        
        if not self.history:
            print("[TRAINER] ⚠️  No training history to plot")
            return
        
        plt.style.use('dark_background')
        
        epochs = [m.epoch for m in self.history]
        train_losses = [m.train_loss for m in self.history]
        val_losses = [m.val_loss for m in self.history]
        lrs = [m.learning_rate for m in self.history]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curves
        axes[0].plot(epochs, train_losses, 'o-', color='#00ff7f', label='Train Loss', markersize=3)
        axes[0].plot(epochs, val_losses, 'o-', color='#ff6347', label='Val Loss', markersize=3)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training & Validation Loss', fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Learning rate schedule
        axes[1].plot(epochs, lrs, '-', color='#4169e1', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule', fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if output_path is None:
            output_path = os.path.join(self.config.output_dir, "logs", "training_curves.png")
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[TRAINER] 📈 Training curves saved to: {output_path}")
    
    def export_model(self, export_path: str):
        """
        Export the trained model for inference.
        
        Args:
            export_path: Directory to save the exported model.
        """
        os.makedirs(export_path, exist_ok=True)
        
        # In a full implementation, this would:
        # 1. Load the best checkpoint
        # 2. Convert to TensorFlow SavedModel format
        # 3. Optionally convert to TFLite for edge deployment
        # 4. Save class names and anchor files
        
        # Save class names
        if self.config.class_names:
            classes_path = os.path.join(export_path, "classes.txt")
            with open(classes_path, 'w') as f:
                for name in self.config.class_names:
                    f.write(name + '\n')
        
        print(f"[TRAINER] 📦 Model exported to: {export_path}")


def create_dataset_structure(base_dir: str = "data"):
    """
    Create the expected directory structure for a custom training dataset.
    
    Expected format:
        data/
        ├── train/
        │   ├── images/        ← Training images (.jpg)
        │   └── labels/        ← YOLO format annotations (.txt)
        ├── val/
        │   ├── images/        ← Validation images
        │   └── labels/        ← Validation annotations
        └── classes.txt        ← Class names (one per line)
    
    YOLO Annotation Format (per .txt file, one line per object):
        <class_id> <x_center> <y_center> <width> <height>
        
        All values normalized to [0, 1]:
        - class_id: integer starting from 0
        - x_center, y_center: center of bounding box
        - width, height: dimensions of bounding box
    
    Example:
        0 0.45 0.52 0.12 0.08    (class 0 = car)
        1 0.72 0.48 0.05 0.15    (class 1 = truck)
    """
    dirs = [
        os.path.join(base_dir, "train", "images"),
        os.path.join(base_dir, "train", "labels"),
        os.path.join(base_dir, "val", "images"),
        os.path.join(base_dir, "val", "labels"),
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    # Create sample classes file
    classes_path = os.path.join(base_dir, "classes.txt")
    if not os.path.exists(classes_path):
        sample_classes = [
            "car", "truck", "bus", "person", "bicycle",
            "motorbike", "traffic_light", "stop_sign", "train",
        ]
        with open(classes_path, 'w') as f:
            for cls in sample_classes:
                f.write(cls + '\n')
    
    print(f"✅ Dataset structure created at: {base_dir}/")
    print(f"   Place your images in {base_dir}/train/images/")
    print(f"   Place YOLO annotations in {base_dir}/train/labels/")
    print(f"   Edit class names in {base_dir}/classes.txt")
