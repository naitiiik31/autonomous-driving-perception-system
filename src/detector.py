"""
Core Detection Engine
=====================

The YOLODetector class is the main orchestrator that ties together model
loading, image preprocessing, inference, and post-processing into a
single, easy-to-use detection pipeline.

Usage:
    from src import YOLODetector, Config
    
    config = Config.driving_mode()
    detector = YOLODetector(config)
    detections = detector.detect("images/0001.jpg")
    
    for det in detections:
        print(f"{det.class_name}: {det.score:.2f} at {det.box}")
"""

import os
import time
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import defaultdict

from .config import Config
from .model import YOLOModel
from .postprocessor import PostProcessor, Detection


class DetectionResult:
    """
    Container for detection results from a single image.
    
    Stores the detections along with metadata about the processing,
    including timing information and statistics.
    
    Attributes:
        image_path (str): Path to the input image.
        detections (List[Detection]): List of detected objects.
        processing_time (float): Time taken for inference + post-processing (seconds).
        image_shape (Tuple[int, int]): Original image dimensions (height, width).
        stats (dict): Aggregate statistics about the detections.
    """
    
    def __init__(
        self,
        image_path: str,
        detections: List[Detection],
        processing_time: float,
        image_shape: Tuple[int, int],
    ):
        self.image_path = image_path
        self.detections = detections
        self.processing_time = processing_time
        self.image_shape = image_shape
        self.stats = self._compute_stats()
    
    def _compute_stats(self) -> dict:
        """Compute aggregate statistics about the detections."""
        stats = {
            "total_objects": len(self.detections),
            "unique_classes": len(set(d.class_name for d in self.detections)),
            "avg_confidence": 0.0,
            "min_confidence": 0.0,
            "max_confidence": 0.0,
            "class_counts": defaultdict(int),
            "fps": 1.0 / self.processing_time if self.processing_time > 0 else 0,
        }
        
        if self.detections:
            scores = [d.score for d in self.detections]
            stats["avg_confidence"] = np.mean(scores)
            stats["min_confidence"] = np.min(scores)
            stats["max_confidence"] = np.max(scores)
            
            for d in self.detections:
                stats["class_counts"][d.class_name] += 1
        
        return stats
    
    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = [
            f"┌─ Detection Results: {os.path.basename(self.image_path)} ─┐",
            f"│  Objects Detected : {self.stats['total_objects']}",
            f"│  Unique Classes   : {self.stats['unique_classes']}",
            f"│  Avg Confidence   : {self.stats['avg_confidence']:.3f}",
            f"│  Processing Time  : {self.processing_time*1000:.1f} ms",
            f"│  FPS              : {self.stats['fps']:.1f}",
            f"├─ Class Breakdown ─┤",
        ]
        for cls_name, count in sorted(
            self.stats["class_counts"].items(), key=lambda x: -x[1]
        ):
            lines.append(f"│  {cls_name:<18}: {count}")
        lines.append("└" + "─" * 40 + "┘")
        return "\n".join(lines)
    
    def filter_by_class(self, class_names: List[str]) -> List[Detection]:
        """Return detections matching the given class names."""
        return [d for d in self.detections if d.class_name in class_names]
    
    def filter_by_confidence(self, min_score: float) -> List[Detection]:
        """Return detections above the given confidence threshold."""
        return [d for d in self.detections if d.score >= min_score]


class YOLODetector:
    """
    Main YOLO detection pipeline orchestrator.
    
    This class provides a high-level API for object detection, combining
    model loading, preprocessing, inference, and post-processing into
    simple method calls.
    
    The detection pipeline:
    
        Image Path → Preprocess → Model Inference → Post-Process → Detections
                     (resize,       (forward pass)    (decode,
                      normalize)                       filter,
                                                       NMS)
    
    Attributes:
        config (Config): Pipeline configuration.
        model (YOLOModel): Loaded YOLO model.
        postprocessor (PostProcessor): Output post-processing engine.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the YOLO detector.
        
        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or Config()
        
        if self.config.verbose:
            print(self.config.summary())
            print()
        
        # Initialize model
        self.model = YOLOModel(
            model_path=self.config.model_path,
            classes_path=self.config.classes_path,
            anchors_path=self.config.anchors_path,
            input_size=self.config.model_image_size,
        )
        
        # Initialize post-processor
        self.postprocessor = PostProcessor(
            anchors=self.model.anchors,
            num_classes=self.model.num_classes,
            class_names=self.model.class_names,
            score_threshold=self.config.score_threshold,
            iou_threshold=self.config.iou_threshold,
            max_boxes=self.config.max_boxes,
        )
        
        # Warm up the model with a dummy prediction
        self._warmup()
        
        if self.config.verbose:
            print(f"\n[DETECTOR] ✅ YOLO Detector initialized and ready!")
            print(f"[DETECTOR]    Score threshold: {self.config.score_threshold}")
            print(f"[DETECTOR]    IoU threshold: {self.config.iou_threshold}")
            print(f"[DETECTOR]    Max boxes: {self.config.max_boxes}")
            if self.config.enable_driving_filter:
                print(f"[DETECTOR]    🚗 Driving mode ENABLED — filtering for: {', '.join(self.config.driving_classes)}")
            print()
    
    def _warmup(self):
        """Run a dummy prediction to warm up the model (compile graph)."""
        dummy_input = np.zeros((1, *self.config.model_image_size, 3), dtype=np.float32)
        try:
            self.model.predict(dummy_input)
            if self.config.verbose:
                print("[DETECTOR] 🔥 Model warmup complete")
        except Exception as e:
            print(f"[DETECTOR] ⚠️  Warmup failed (non-critical): {e}")
    
    def detect(self, image_path: str) -> DetectionResult:
        """
        Run object detection on a single image.
        
        This is the main detection method. It handles the full pipeline:
        1. Load and preprocess the image
        2. Run model inference
        3. Post-process outputs (decode, filter, NMS)
        4. Package results into a DetectionResult
        
        Args:
            image_path: Path to the input image file.
            
        Returns:
            DetectionResult containing all detections and metadata.
            
        Raises:
            FileNotFoundError: If the image path does not exist.
        """
        start_time = time.time()
        
        # Step 1: Preprocess
        original_image, image_data = self.model.preprocess_image(image_path)
        image_shape = (original_image.size[1], original_image.size[0])  # (H, W)
        
        # Step 2: Model inference
        raw_output = self.model.predict(image_data)
        
        # Step 3: Post-process
        target_classes = None
        if self.config.enable_driving_filter:
            target_classes = self.config.driving_classes
        elif self.config.target_classes:
            target_classes = self.config.target_classes
        
        detections = self.postprocessor.process(
            raw_output, image_shape, target_classes
        )
        
        processing_time = time.time() - start_time
        
        # Step 4: Create result
        result = DetectionResult(
            image_path=image_path,
            detections=detections,
            processing_time=processing_time,
            image_shape=image_shape,
        )
        
        if self.config.verbose:
            print(f"[DETECT] 📸 {os.path.basename(image_path)}: "
                  f"{len(detections)} objects detected in {processing_time*1000:.1f}ms")
        
        return result
    
    def detect_from_array(
        self,
        image_array: np.ndarray,
        image_name: str = "frame",
    ) -> DetectionResult:
        """
        Run object detection on a numpy array (e.g., video frame).
        
        Args:
            image_array: Input image as numpy array (H, W, 3) in RGB, values [0, 255].
            image_name: Name identifier for the image.
            
        Returns:
            DetectionResult containing all detections and metadata.
        """
        from PIL import Image
        
        start_time = time.time()
        
        # Get original shape
        image_shape = (image_array.shape[0], image_array.shape[1])
        
        # Convert RGBA to RGB if needed
        if image_array.ndim == 3 and image_array.shape[2] == 4:
            image_array = image_array[:, :, :3]
        
        # Resize and normalize for model
        pil_image = Image.fromarray(image_array)
        resized = pil_image.resize(
            (self.config.model_image_size[1], self.config.model_image_size[0]),
            Image.BICUBIC,
        )
        image_data = np.array(resized, dtype='float32') / 255.0
        image_data = np.expand_dims(image_data, axis=0)
        
        # Inference
        raw_output = self.model.predict(image_data)
        
        # Post-process
        target_classes = None
        if self.config.enable_driving_filter:
            target_classes = self.config.driving_classes
        elif self.config.target_classes:
            target_classes = self.config.target_classes
        
        detections = self.postprocessor.process(
            raw_output, image_shape, target_classes
        )
        
        processing_time = time.time() - start_time
        
        return DetectionResult(
            image_path=image_name,
            detections=detections,
            processing_time=processing_time,
            image_shape=image_shape,
        )
    
    def detect_batch(self, image_paths: List[str]) -> List[DetectionResult]:
        """
        Run detection on multiple images with progress tracking.
        
        Args:
            image_paths: List of paths to input images.
            
        Returns:
            List of DetectionResult objects, one per image.
        """
        results = []
        total = len(image_paths)
        
        try:
            from tqdm import tqdm
            iterator = tqdm(image_paths, desc="🔍 Detecting", unit="img")
        except ImportError:
            iterator = image_paths
        
        for i, path in enumerate(iterator):
            try:
                result = self.detect(path)
                results.append(result)
            except Exception as e:
                print(f"[ERROR] ❌ Failed to process {path}: {e}")
        
        # Print batch summary
        if self.config.verbose and results:
            total_objects = sum(r.stats["total_objects"] for r in results)
            avg_time = np.mean([r.processing_time for r in results])
            print(f"\n[BATCH] ✅ Processed {len(results)}/{total} images")
            print(f"[BATCH]    Total objects detected: {total_objects}")
            print(f"[BATCH]    Avg processing time: {avg_time*1000:.1f}ms")
            print(f"[BATCH]    Avg FPS: {1.0/avg_time:.1f}")
        
        return results
    
    def get_class_names(self) -> List[str]:
        """Return the list of class names the model can detect."""
        return self.model.class_names
    
    def get_driving_classes(self) -> List[str]:
        """Return the list of driving-relevant class names."""
        return self.config.driving_classes
    
    def update_thresholds(
        self,
        score_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        max_boxes: Optional[int] = None,
    ):
        """
        Dynamically update detection thresholds without reloading the model.
        
        Args:
            score_threshold: New minimum confidence threshold.
            iou_threshold: New IoU threshold for NMS.
            max_boxes: New maximum detection count.
        """
        if score_threshold is not None:
            self.config.score_threshold = score_threshold
            self.postprocessor.score_threshold = score_threshold
        
        if iou_threshold is not None:
            self.config.iou_threshold = iou_threshold
            self.postprocessor.iou_threshold = iou_threshold
        
        if max_boxes is not None:
            self.config.max_boxes = max_boxes
            self.postprocessor.max_boxes = max_boxes
        
        if self.config.verbose:
            print(f"[DETECTOR] 🔄 Thresholds updated: "
                  f"score={self.config.score_threshold}, "
                  f"iou={self.config.iou_threshold}, "
                  f"max_boxes={self.config.max_boxes}")
