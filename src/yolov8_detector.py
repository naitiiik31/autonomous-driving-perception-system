"""
YOLOv8 Detector Wrapper
========================
Drop-in replacement for the legacy YOLO v2 / TF SavedModel detector.
Uses Ultralytics YOLOv8, which provides:
  - State-of-the-art accuracy (53.9 mAP@50 COCO)
  - Native Python API
  - Built-in multi-object tracking (ByteTrack)
  - ONNX / TFLite / CoreML export
  - Active maintenance

Installation:
    pip install ultralytics

Usage:
    from src.yolov8_detector import YOLOv8Detector, YOLOv8Config

    cfg      = YOLOv8Config()
    detector = YOLOv8Detector(cfg)
    results  = detector.detect("images/0001.jpg")

    for det in results.detections:
        print(f"{det.class_name} {det.score:.2%} @ {det.box}")
"""

import os
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from collections import defaultdict

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class YOLOv8Config:
    """
    Configuration for the YOLOv8 detection pipeline.

    Defaults optimised for autonomous driving inference.
    """
    # Model
    weights:          str   = "models/yolov8m.pt"   # moved from root after cleanup
    device:           str   = ""                     # "" = auto (GPU if available, else CPU)
    input_size:       int   = 640                    # inference image size (pixels)

    # Thresholds
    confidence:       float = 0.50
    iou:              float = 0.45
    max_detections:   int   = 100

    # Tracking
    enable_tracking:  bool  = True
    tracker:          str   = "bytetrack.yaml"       # "botsort.yaml" for appearance re-id

    # Driving filter
    enable_driving_filter: bool      = False
    driving_classes:       List[str] = field(default_factory=lambda: [
        "car", "truck", "bus", "motorcycle", "bicycle",
        "person", "traffic light", "stop sign",
    ])
    target_classes: Optional[List[str]] = None

    # Output
    verbose: bool = False
    output_dir: str = "outputs"


@dataclass
class Detection:
    """Single object detection result (compatible with existing code)."""
    box:        np.ndarray   # [y_min, x_min, y_max, x_max] pixels
    score:      float
    class_id:   int
    class_name: str
    track_id:   Optional[int] = None


@dataclass
class DetectionResult:
    """Aggregated result for one image or frame."""
    image_path:      str
    detections:      List[Detection]
    processing_time: float
    image_shape:     Tuple[int, int]   # (H, W)

    @property
    def stats(self) -> Dict:
        scores = [d.score for d in self.detections]
        counts: Dict[str, int] = defaultdict(int)
        for d in self.detections:
            counts[d.class_name] += 1
        return {
            "total_objects":   len(self.detections),
            "unique_classes":  len(counts),
            "avg_confidence":  float(np.mean(scores)) if scores else 0.0,
            "max_confidence":  float(np.max(scores))  if scores else 0.0,
            "fps":             1.0 / self.processing_time if self.processing_time > 0 else 0.0,
            "class_counts":    dict(counts),
        }

    def summary(self) -> str:
        s = self.stats
        lines = [
            f"┌─ {os.path.basename(self.image_path)} ─",
            f"│  Objects   : {s['total_objects']}",
            f"│  Classes   : {s['unique_classes']}",
            f"│  Avg Conf  : {s['avg_confidence']:.3f}",
            f"│  Inference : {self.processing_time*1000:.1f} ms",
            f"│  FPS       : {s['fps']:.1f}",
        ]
        for cls, cnt in sorted(s["class_counts"].items(), key=lambda x: -x[1]):
            lines.append(f"│    {cls:<18}: {cnt}")
        lines.append("└" + "─" * 38)
        return "\n".join(lines)


class YOLOv8Detector:
    """
    Production YOLOv8 detection pipeline.

    Wraps the Ultralytics YOLOv8 API into the same interface as the
    legacy YOLODetector so existing scripts (detect_image.py, etc.)
    need minimal changes.

    The detection pipeline:
        Image → YOLOv8 Inference → NMS (built-in) → Filter → DetectionResult

    Tracking pipeline (enable_tracking=True):
        Frame N → ByteTrack → Persistent Track IDs → DetectionResult
    """

    def __init__(self, config: Optional[YOLOv8Config] = None):
        self.config = config or YOLOv8Config()
        self._model = None
        self._load_model()

    def _load_model(self):
        """Load YOLOv8 model. Downloads weights if not found locally."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "Ultralytics not installed. Run:\n  pip install ultralytics"
            )

        weights = self.config.weights
        logger.info("Loading YOLOv8 model: %s", weights)
        t0 = time.time()
        self._model = YOLO(weights)
        logger.info("YOLOv8 loaded in %.2fs | device=%s", time.time() - t0, self.config.device or "auto")

    # ── Public API ───────────────────────────────────────────────

    def detect(self, image_path: str) -> DetectionResult:
        """
        Run detection on a single image file.

        Args:
            image_path: Path to the input image.

        Returns:
            DetectionResult with all detections.

        Raises:
            FileNotFoundError: If image_path does not exist.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        import cv2
        frame = cv2.imread(image_path)
        h, w  = frame.shape[:2]

        t0 = time.time()
        raw = self._model.predict(
            source     = image_path,
            conf       = self.config.confidence,
            iou        = self.config.iou,
            max_det    = self.config.max_detections,
            imgsz      = self.config.input_size,
            device     = self.config.device,
            verbose    = self.config.verbose,
        )
        elapsed = time.time() - t0

        detections = self._parse_results(raw, track_ids=None)
        detections = self._apply_filter(detections)

        logger.info(
            "Detect | %s | %d objects | %.1fms",
            os.path.basename(image_path), len(detections), elapsed * 1000,
        )

        return DetectionResult(
            image_path      = image_path,
            detections      = detections,
            processing_time = elapsed,
            image_shape     = (h, w),
        )

    def detect_from_array(
        self,
        frame_rgb:  np.ndarray,
        image_name: str = "frame",
    ) -> DetectionResult:
        """
        Run detection on a numpy RGB array (e.g. video frame).

        Args:
            frame_rgb:  RGB numpy array (H, W, 3).
            image_name: Identifier string for logging.

        Returns:
            DetectionResult with all detections.
        """
        h, w = frame_rgb.shape[:2]
        t0   = time.time()

        if self.config.enable_tracking:
            raw = self._model.track(
                source    = frame_rgb,
                conf      = self.config.confidence,
                iou       = self.config.iou,
                max_det   = self.config.max_detections,
                imgsz     = self.config.input_size,
                device    = self.config.device,
                verbose   = self.config.verbose,
                tracker   = self.config.tracker,
                persist   = True,
            )
            track_ids = self._extract_track_ids(raw)
        else:
            raw = self._model.predict(
                source  = frame_rgb,
                conf    = self.config.confidence,
                iou     = self.config.iou,
                max_det = self.config.max_detections,
                imgsz   = self.config.input_size,
                device  = self.config.device,
                verbose = self.config.verbose,
            )
            track_ids = None

        elapsed    = time.time() - t0
        detections = self._parse_results(raw, track_ids=track_ids)
        detections = self._apply_filter(detections)

        return DetectionResult(
            image_path      = image_name,
            detections      = detections,
            processing_time = elapsed,
            image_shape     = (h, w),
        )

    def detect_batch(self, image_paths: List[str]) -> List[DetectionResult]:
        """Run detection on a list of images."""
        results = []
        try:
            from tqdm import tqdm
            it = tqdm(image_paths, desc="YOLOv8 Detecting", unit="img")
        except ImportError:
            it = image_paths

        for path in it:
            try:
                results.append(self.detect(path))
            except Exception as e:
                logger.error("Failed to process %s: %s", path, e)

        return results

    def get_class_names(self) -> List[str]:
        """Return all class names this model was trained on."""
        return list(self._model.names.values())

    # ── Private helpers ──────────────────────────────────────────

    def _parse_results(self, raw_results, track_ids) -> List[Detection]:
        """Convert Ultralytics result objects → List[Detection]."""
        detections: List[Detection] = []
        if not raw_results:
            return detections

        result = raw_results[0]
        boxes  = result.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        xyxy_np   = boxes.xyxy.cpu().numpy()    # (N, 4) x1,y1,x2,y2
        conf_np   = boxes.conf.cpu().numpy()    # (N,)
        cls_np    = boxes.cls.cpu().numpy().astype(int)  # (N,)

        for i in range(len(xyxy_np)):
            x1, y1, x2, y2 = xyxy_np[i]
            # Convert to [y_min, x_min, y_max, x_max] format (consistent with legacy)
            box = np.array([y1, x1, y2, x2], dtype=np.float32)
            tid = int(track_ids[i]) if track_ids is not None and i < len(track_ids) else None
            cls_id = int(cls_np[i])
            cls_name = self._model.names.get(cls_id, f"class_{cls_id}")

            detections.append(Detection(
                box        = box,
                score      = float(conf_np[i]),
                class_id   = cls_id,
                class_name = cls_name,
                track_id   = tid,
            ))

        detections.sort(key=lambda d: d.score, reverse=True)
        return detections

    def _extract_track_ids(self, raw_results) -> Optional[np.ndarray]:
        """Extract track IDs from Ultralytics tracking results."""
        if not raw_results:
            return None
        boxes = raw_results[0].boxes
        if boxes is None or boxes.id is None:
            return None
        return boxes.id.cpu().numpy().astype(int)

    def _apply_filter(self, detections: List[Detection]) -> List[Detection]:
        """Apply driving class filter or custom target_classes filter."""
        cfg = self.config
        if cfg.enable_driving_filter:
            return [d for d in detections if d.class_name in cfg.driving_classes]
        if cfg.target_classes:
            return [d for d in detections if d.class_name in cfg.target_classes]
        return detections
