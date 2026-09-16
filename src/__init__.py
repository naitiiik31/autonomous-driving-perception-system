"""
Autonomous Driving — Car Detection using YOLO
=============================================

A production-grade deep learning pipeline for real-time vehicle detection
in autonomous driving scenarios using YOLO (You Only Look Once) architecture.

Modules:
    - config: Configuration management and hyperparameters
    - logger: Structured logging (console + JSON file)
    - model: YOLO v2 model loading and architecture (legacy)
    - yolov8_detector: YOLOv8 detector wrapper (recommended)
    - detector: Core YOLO v2 detection engine (legacy)
    - postprocessor: NMS, filtering, box decoding
    - visualizer: Advanced bounding box visualization
    - video_processor: Video processing pipeline
    - metrics: mAP, precision-recall evaluation
    - utils: General utility functions
    - lane_detection: Hough-based lane line detector
    - distance_estimation: Monocular distance + TTC collision warning
    - augmentation: Training data augmentation pipeline
"""

__version__ = "3.0.0"
__author__ = "Deep Learning Project"

from .config import Config
from .logger import get_logger, setup_logging
from .detector import YOLODetector
from .yolov8_detector import YOLOv8Detector, YOLOv8Config
from .visualizer import DetectionVisualizer
from .video_processor import VideoProcessor
from .lane_detection import LaneDetector
from .distance_estimation import DistanceEstimator, CollisionAdvisor
from .metrics import DetectionMetrics
