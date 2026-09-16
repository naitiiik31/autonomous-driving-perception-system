"""
Configuration Management
========================

Centralized configuration for the YOLO car detection pipeline.
Supports multiple detection profiles and customizable thresholds.
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple


@dataclass
class Config:
    """
    Master configuration class for the YOLO detection pipeline.
    
    This class manages all hyperparameters, file paths, and detection
    settings used throughout the pipeline. It supports serialization
    to/from JSON for reproducibility.
    
    Attributes:
        model_path (str): Path to the TensorFlow SavedModel directory.
        anchors_path (str): Path to the anchor box definitions file.
        classes_path (str): Path to the class names file.
        score_threshold (float): Minimum confidence score to keep a detection.
        iou_threshold (float): IoU threshold for Non-Max Suppression.
        max_boxes (int): Maximum number of detections per image.
        model_image_size (Tuple[int, int]): Expected input size for the model (H, W).
        input_image_dir (str): Directory containing input images.
        output_image_dir (str): Directory to save annotated output images.
        font_path (str): Path to the font file for text annotations.
        driving_classes (List[str]): Classes relevant to autonomous driving.
        enable_driving_filter (bool): If True, only show driving-relevant objects.
        save_detection_crops (bool): If True, save cropped detections.
        save_statistics (bool): If True, generate detection statistics.
        batch_size (int): Number of images to process in a batch.
        verbose (bool): If True, print detailed progress information.
    """
    
    # ── Model Paths ──────────────────────────────────────────────
    model_path: str = "model_data"
    anchors_path: str = "model_data/yolo_anchors.txt"
    classes_path: str = "model_data/coco_classes.txt"
    
    # ── Detection Thresholds ─────────────────────────────────────
    score_threshold: float = 0.5
    iou_threshold: float = 0.5
    max_boxes: int = 50
    max_boxes_per_class: int = 20
    
    # ── Model Input ──────────────────────────────────────────────
    model_image_size: Tuple[int, int] = (608, 608)
    
    # ── File Paths ───────────────────────────────────────────────
    input_image_dir: str = "images"
    output_image_dir: str = "out"
    font_path: str = "font/FiraMono-Medium.otf"
    crops_dir: str = "out/crops"
    stats_dir: str = "out/stats"
    
    # ── Autonomous Driving Filter ────────────────────────────────
    driving_classes: List[str] = field(default_factory=lambda: [
        "car", "truck", "bus", "motorbike", "bicycle",
        "person", "traffic light", "stop sign", "train"
    ])
    enable_driving_filter: bool = False
    target_classes: Optional[List[str]] = None  # None = all classes
    
    # ── Processing Options ───────────────────────────────────────
    save_detection_crops: bool = False
    save_statistics: bool = True
    batch_size: int = 1
    verbose: bool = True
    
    # ── Visualization ────────────────────────────────────────────
    box_thickness: int = 2
    font_scale: float = 0.03
    show_confidence: bool = True
    show_count_overlay: bool = True
    show_processing_time: bool = True
    overlay_alpha: float = 0.7
    
    # ── Color Scheme (RGB) for driving classes ───────────────────
    class_colors: dict = field(default_factory=lambda: {
        "car": (0, 255, 127),        # Spring Green
        "truck": (255, 165, 0),      # Orange
        "bus": (255, 69, 0),         # Red-Orange
        "motorbike": (138, 43, 226), # Blue-Violet
        "bicycle": (0, 191, 255),    # Deep Sky Blue
        "person": (255, 20, 147),    # Deep Pink
        "traffic light": (255, 255, 0),  # Yellow
        "stop sign": (255, 0, 0),    # Red
        "train": (70, 130, 180),     # Steel Blue
    })
    
    # ── Video Processing ─────────────────────────────────────────
    video_fps: int = 25
    video_codec: str = "mp4v"
    video_skip_frames: int = 0  # 0 = process every frame
    
    def __post_init__(self):
        """Create output directories if they don't exist."""
        os.makedirs(self.output_image_dir, exist_ok=True)
        if self.save_detection_crops:
            os.makedirs(self.crops_dir, exist_ok=True)
        if self.save_statistics:
            os.makedirs(self.stats_dir, exist_ok=True)
    
    @classmethod
    def driving_mode(cls) -> "Config":
        """
        Factory method for autonomous-driving-optimized configuration.
        
        Uses stricter thresholds and filters for driving-relevant classes only.
        
        Returns:
            Config: Configuration optimized for driving scenarios.
        """
        return cls(
            score_threshold=0.4,
            iou_threshold=0.5,
            max_boxes=100,
            enable_driving_filter=True,
            save_statistics=True,
            show_count_overlay=True,
            verbose=True,
        )
    
    @classmethod
    def high_precision(cls) -> "Config":
        """
        Factory method for high-precision detection (fewer false positives).
        
        Returns:
            Config: Configuration with high confidence threshold.
        """
        return cls(
            score_threshold=0.7,
            iou_threshold=0.4,
            max_boxes=30,
            verbose=True,
        )
    
    @classmethod
    def high_recall(cls) -> "Config":
        """
        Factory method for high-recall detection (fewer missed objects).
        
        Returns:
            Config: Configuration with low confidence threshold.
        """
        return cls(
            score_threshold=0.3,
            iou_threshold=0.6,
            max_boxes=100,
            verbose=True,
        )
    
    def save(self, path: str):
        """Save configuration to a JSON file for reproducibility."""
        config_dict = asdict(self)
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "Config":
        """Load configuration from a JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        # Convert tuple keys back
        if 'model_image_size' in config_dict:
            config_dict['model_image_size'] = tuple(config_dict['model_image_size'])
        return cls(**config_dict)
    
    def summary(self) -> str:
        """Return a formatted summary string of the current configuration."""
        lines = [
            "╔══════════════════════════════════════════════════════╗",
            "║         YOLO Car Detection — Configuration          ║",
            "╠══════════════════════════════════════════════════════╣",
            f"║  Model Path      : {self.model_path:<34}║",
            f"║  Classes File    : {self.classes_path:<34}║",
            f"║  Anchors File    : {self.anchors_path:<34}║",
            f"║  Input Size      : {str(self.model_image_size):<34}║",
            "╠══════════════════════════════════════════════════════╣",
            f"║  Score Threshold : {self.score_threshold:<34}║",
            f"║  IoU Threshold   : {self.iou_threshold:<34}║",
            f"║  Max Boxes       : {self.max_boxes:<34}║",
            f"║  Driving Filter  : {str(self.enable_driving_filter):<34}║",
            "╠══════════════════════════════════════════════════════╣",
            f"║  Save Crops      : {str(self.save_detection_crops):<34}║",
            f"║  Save Statistics : {str(self.save_statistics):<34}║",
            f"║  Verbose         : {str(self.verbose):<34}║",
            "╚══════════════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)
