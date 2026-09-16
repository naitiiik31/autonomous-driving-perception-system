"""
Utility Functions
=================

General-purpose utility functions used across the YOLO detection pipeline.
Includes file management, logging, and helper functions.
"""

import os
import glob
import time
import json
import numpy as np
from typing import List, Tuple, Optional
from datetime import datetime


def get_image_paths(
    directory: str,
    extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.webp'),
) -> List[str]:
    """
    Get all image file paths from a directory.
    
    Args:
        directory: Path to the image directory.
        extensions: Tuple of valid image file extensions.
        
    Returns:
        Sorted list of absolute image file paths.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(directory, f"*{ext}")))
        image_paths.extend(glob.glob(os.path.join(directory, f"*{ext.upper()}")))
    
    return sorted(set(image_paths))


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist and return the path."""
    os.makedirs(path, exist_ok=True)
    return path


def format_time(seconds: float) -> str:
    """
    Format seconds into a human-readable string.
    
    Args:
        seconds: Duration in seconds.
        
    Returns:
        Formatted string (e.g., "2m 30s" or "150ms").
    """
    if seconds < 0.001:
        return f"{seconds * 1e6:.0f}μs"
    elif seconds < 1.0:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def print_banner():
    """Print the project banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🚗  AUTONOMOUS DRIVING — CAR DETECTION USING YOLO  🚗     ║
║                                                              ║
║   Deep Learning Pipeline for Real-Time Object Detection      ║
║   Architecture: YOLO v2 + DarkNet-19 Backbone               ║
║   Framework: TensorFlow 2.x + Keras                         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def save_detection_results(
    results: list,
    output_path: str,
):
    """
    Save detection results to a JSON file.
    
    Args:
        results: List of DetectionResult objects.
        output_path: Path to save the JSON file.
    """
    serializable = []
    for result in results:
        entry = {
            "image": result.image_path,
            "processing_time_ms": result.processing_time * 1000,
            "image_shape": list(result.image_shape),
            "num_detections": len(result.detections),
            "detections": [
                {
                    "class": det.class_name,
                    "class_id": det.class_id,
                    "confidence": float(det.score),
                    "box": {
                        "y_min": float(det.box[0]),
                        "x_min": float(det.box[1]),
                        "y_max": float(det.box[2]),
                        "x_max": float(det.box[3]),
                    },
                }
                for det in result.detections
            ],
        }
        serializable.append(entry)
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"💾 Detection results saved to: {output_path}")


def create_detection_summary_table(results: list) -> str:
    """
    Create a formatted ASCII table of detection results.
    
    Args:
        results: List of DetectionResult objects.
        
    Returns:
        Formatted table string.
    """
    if not results:
        return "No results to display."
    
    # Header
    lines = [
        "┌────────────────────────┬──────────┬──────────┬──────────────┐",
        "│ Image                  │ Objects  │ Time(ms) │ Top Class    │",
        "├────────────────────────┼──────────┼──────────┼──────────────┤",
    ]
    
    for result in results:
        name = os.path.basename(result.image_path)[:22]
        num_obj = len(result.detections)
        time_ms = result.processing_time * 1000
        
        # Find top class
        if result.detections:
            from collections import Counter
            class_counts = Counter(d.class_name for d in result.detections)
            top_class = class_counts.most_common(1)[0][0]
        else:
            top_class = "—"
        
        lines.append(
            f"│ {name:<22} │ {num_obj:>8} │ {time_ms:>8.1f} │ {top_class:<12} │"
        )
    
    lines.append(
        "└────────────────────────┴──────────┴──────────┴──────────────┘"
    )
    
    # Summary row
    total_obj = sum(len(r.detections) for r in results)
    avg_time = np.mean([r.processing_time for r in results]) * 1000
    lines.append(f"\n  Total: {len(results)} images | {total_obj} objects | Avg: {avg_time:.1f}ms")
    
    return "\n".join(lines)


def get_timestamp() -> str:
    """Get a formatted timestamp string for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def compute_image_stats(image_path: str) -> dict:
    """
    Compute basic statistics about an image.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        Dictionary with image statistics.
    """
    from PIL import Image
    
    img = Image.open(image_path)
    img_array = np.array(img)
    
    return {
        "path": image_path,
        "width": img.size[0],
        "height": img.size[1],
        "channels": len(img.getbands()),
        "mode": img.mode,
        "file_size_kb": os.path.getsize(image_path) / 1024,
        "mean_intensity": float(np.mean(img_array)),
        "std_intensity": float(np.std(img_array)),
    }
