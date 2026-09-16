"""
Video Processing Pipeline
=========================

Processes video files frame-by-frame through the YOLO detection pipeline,
producing annotated output videos with bounding boxes and detection info.

Features:
- Frame-by-frame object detection with progress tracking
- Configurable frame skipping for faster processing
- Detection statistics aggregation over video duration
- Output video generation with annotations
- Per-frame detection logging
"""

import os
import time
import numpy as np
from typing import Optional, List, Tuple, Dict
from collections import defaultdict

from .config import Config
from .detector import YOLODetector, DetectionResult
from .visualizer import DetectionVisualizer


class VideoProcessor:
    """
    Video processing pipeline for YOLO object detection.
    
    Processes video files frame-by-frame, running detection on each frame
    and producing annotated output videos along with aggregate statistics.
    
    The pipeline:
        Video → Frame Extraction → Detection → Annotation → Output Video
                                                         → Statistics
    
    Attributes:
        detector (YOLODetector): The detection engine.
        visualizer (DetectionVisualizer): The visualization engine.
        config (Config): Pipeline configuration.
    """
    
    def __init__(
        self,
        detector: YOLODetector,
        visualizer: Optional[DetectionVisualizer] = None,
        config: Optional[Config] = None,
    ):
        """
        Initialize the video processor.
        
        Args:
            detector: Initialized YOLO detector instance.
            visualizer: Optional visualizer instance. Created if not provided.
            config: Optional configuration. Uses detector's config if not provided.
        """
        self.detector = detector
        self.config = config or detector.config
        self.visualizer = visualizer or DetectionVisualizer(self.config)
    
    def process_video(
        self,
        input_path: str,
        output_path: str,
        max_frames: Optional[int] = None,
        skip_frames: Optional[int] = None,
    ) -> Dict:
        """
        Process an entire video file with object detection.
        
        Reads the input video frame-by-frame, runs detection on each frame,
        annotates the frame with bounding boxes, and writes to an output video.
        
        Args:
            input_path: Path to the input video file.
            output_path: Path for the output annotated video.
            max_frames: Maximum number of frames to process (None = all).
            skip_frames: Process every Nth frame (None = use config value).
            
        Returns:
            Dictionary with video processing statistics.
            
        Raises:
            FileNotFoundError: If the input video does not exist.
            ImportError: If OpenCV is not installed.
        """
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "OpenCV (cv2) is required for video processing. "
                "Install it with: pip install opencv-python"
            )
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Video not found: {input_path}")
        
        skip = skip_frames if skip_frames is not None else self.config.video_skip_frames
        
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {input_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or self.config.video_fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"\n{'='*60}")
        print(f"🎬 Video Processing Pipeline")
        print(f"{'='*60}")
        print(f"  Input       : {input_path}")
        print(f"  Output      : {output_path}")
        print(f"  Resolution  : {width}×{height}")
        print(f"  FPS         : {fps}")
        print(f"  Total Frames: {total_frames}")
        if skip > 0:
            print(f"  Skip Frames : {skip} (processing every {skip+1}th frame)")
        print(f"{'='*60}\n")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*self.config.video_codec)
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out_writer.isOpened():
            raise RuntimeError(f"Failed to create output video: {output_path}")
        
        # Process frames
        stats = self._process_frames(
            cap, out_writer, total_frames, skip, width, height,
        )
        
        # Cleanup
        cap.release()
        out_writer.release()
        
        # Print summary
        self._print_video_summary(stats, input_path, output_path)
        
        return stats
    
    def _process_frames(
        self,
        cap,
        out_writer,
        total_frames: int,
        skip: int,
        width: int,
        height: int,
    ) -> Dict:
        """Process all frames with detection and annotation."""
        import cv2
        from PIL import Image
        
        stats = {
            "total_frames": 0,
            "processed_frames": 0,
            "total_detections": 0,
            "total_time": 0.0,
            "frame_times": [],
            "class_counts": defaultdict(int),
            "per_frame_counts": [],
            "max_objects_frame": 0,
            "max_objects_count": 0,
        }
        
        frame_idx = 0
        processed = 0
        overall_start = time.time()
        
        try:
            from tqdm import tqdm
            pbar = tqdm(total=total_frames, desc="🔍 Processing video", unit="frame")
        except ImportError:
            pbar = None
        
        last_detections = []
        
        while True:
            ret, frame = cap.read()
            if not ret or (total_frames and frame_idx >= total_frames):
                break
            
            frame_idx += 1
            stats["total_frames"] = frame_idx
            
            # Skip frames if configured
            if skip > 0 and frame_idx % (skip + 1) != 0:
                # Use last detections for annotation on skipped frames
                if last_detections:
                    annotated = self._annotate_frame(
                        frame, last_detections, 0.0, width, height,
                    )
                    out_writer.write(annotated)
                else:
                    out_writer.write(frame)
                
                if pbar:
                    pbar.update(1)
                continue
            
            # Convert BGR to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run detection
            frame_start = time.time()
            result = self.detector.detect_from_array(
                frame_rgb, image_name=f"frame_{frame_idx:06d}",
            )
            frame_time = time.time() - frame_start
            
            last_detections = result.detections
            processed += 1
            
            # Update stats
            stats["processed_frames"] = processed
            stats["total_detections"] += len(result.detections)
            stats["total_time"] += frame_time
            stats["frame_times"].append(frame_time)
            stats["per_frame_counts"].append(len(result.detections))
            
            if len(result.detections) > stats["max_objects_count"]:
                stats["max_objects_count"] = len(result.detections)
                stats["max_objects_frame"] = frame_idx
            
            for det in result.detections:
                stats["class_counts"][det.class_name] += 1
            
            # Annotate frame
            annotated = self._annotate_frame(
                frame, result.detections, frame_time, width, height,
            )
            out_writer.write(annotated)
            
            if pbar:
                pbar.update(1)
                pbar.set_postfix({
                    "objects": len(result.detections),
                    "ms": f"{frame_time*1000:.0f}",
                })
            elif frame_idx % 50 == 0:
                print(f"  Frame {frame_idx}/{total_frames}: "
                      f"{len(result.detections)} objects, "
                      f"{frame_time*1000:.0f}ms")
        
        if pbar:
            pbar.close()
        
        stats["overall_time"] = time.time() - overall_start
        
        return stats
    
    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: list,
        processing_time: float,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Annotate a single video frame with detections."""
        import cv2
        from PIL import Image
        
        # Convert BGR frame to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Draw detections
        annotated_pil = self.visualizer.draw_detections(
            pil_image,
            detections,
            show_confidence=self.config.show_confidence,
            show_overlay=self.config.show_count_overlay,
            processing_time=processing_time,
        )
        
        # Convert back to BGR numpy array for OpenCV
        annotated_np = np.array(annotated_pil)
        annotated_bgr = cv2.cvtColor(annotated_np, cv2.COLOR_RGB2BGR)
        
        return annotated_bgr
    
    def _print_video_summary(self, stats: Dict, input_path: str, output_path: str):
        """Print a formatted summary of video processing results."""
        print(f"\n{'='*60}")
        print(f"📊 Video Processing Summary")
        print(f"{'='*60}")
        print(f"  Total Frames     : {stats['total_frames']}")
        print(f"  Processed Frames : {stats['processed_frames']}")
        print(f"  Total Detections : {stats['total_detections']}")
        
        if stats['processed_frames'] > 0:
            avg_detections = stats['total_detections'] / stats['processed_frames']
            avg_time = stats['total_time'] / stats['processed_frames']
            avg_fps = 1.0 / avg_time if avg_time > 0 else 0
            
            print(f"  Avg Objects/Frame: {avg_detections:.1f}")
            print(f"  Max Objects      : {stats['max_objects_count']} (frame {stats['max_objects_frame']})")
            print(f"  Avg Inference    : {avg_time*1000:.1f}ms")
            print(f"  Avg FPS          : {avg_fps:.1f}")
            print(f"  Total Time       : {stats.get('overall_time', 0):.1f}s")
        
        if stats['class_counts']:
            print(f"\n  Class Breakdown:")
            sorted_classes = sorted(stats['class_counts'].items(), key=lambda x: -x[1])
            for cls_name, count in sorted_classes[:10]:
                bar = "█" * min(30, count // max(1, stats['processed_frames']))
                print(f"    {cls_name:<18}: {count:>6} {bar}")
        
        print(f"\n  Output saved to: {output_path}")
        
        # Save stats to file if configured
        if self.config.save_statistics:
            self._save_video_stats(stats, output_path)
        
        print(f"{'='*60}\n")
    
    def _save_video_stats(self, stats: Dict, video_path: str):
        """Save video processing statistics to a JSON file."""
        import json
        
        stats_path = os.path.join(
            self.config.stats_dir,
            os.path.splitext(os.path.basename(video_path))[0] + "_stats.json",
        )
        
        # Convert defaultdict to regular dict for JSON
        serializable_stats = {
            k: dict(v) if isinstance(v, defaultdict) else v
            for k, v in stats.items()
            if k != 'frame_times'  # Don't save per-frame timings (too large)
        }
        serializable_stats['avg_frame_time'] = float(np.mean(stats['frame_times'])) if stats['frame_times'] else 0
        serializable_stats['std_frame_time'] = float(np.std(stats['frame_times'])) if stats['frame_times'] else 0
        
        os.makedirs(os.path.dirname(stats_path) or '.', exist_ok=True)
        with open(stats_path, 'w') as f:
            json.dump(serializable_stats, f, indent=2)
        
        print(f"  📈 Statistics saved to: {stats_path}")
