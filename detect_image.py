#!/usr/bin/env python3
"""
🖼️ YOLO Car Detection — Image Detection Script
================================================

Run YOLO object detection on single images or entire directories.
Supports multiple output formats, crop extraction, and filtering.

Usage:
    # Single image
    python detect_image.py --image images/0001.jpg
    
    # Batch processing
    python detect_image.py --input_dir images/ --output_dir out/
    
    # With custom thresholds
    python detect_image.py --image images/test.jpg --confidence 0.4 --iou 0.5
    
    # Driving mode (vehicle classes only)
    python detect_image.py --input_dir images/ --driving_mode
    
    # With crop extraction
    python detect_image.py --image images/0001.jpg --save_crops
    
    # Filter specific classes
    python detect_image.py --image images/test.jpg --classes car truck bus
"""

import os
import sys
import argparse
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.detector import YOLODetector
from src.visualizer import DetectionVisualizer
from src.metrics import DetectionMetrics
from src.utils import (
    print_banner, get_image_paths, create_detection_summary_table,
    save_detection_results, get_timestamp,
)


def detect_single_image(
    detector: YOLODetector,
    visualizer: DetectionVisualizer,
    image_path: str,
    output_path: str,
    config: Config,
):
    """Detect objects in a single image and save the annotated result."""
    from PIL import Image
    
    # Run detection
    result = detector.detect(image_path)
    
    # Load original image
    original = Image.open(image_path)
    
    # Save annotated image
    visualizer.save_annotated_image(
        original, result.detections, output_path,
        processing_time=result.processing_time,
    )
    
    # Save comparison
    annotated = visualizer.draw_detections(
        original, result.detections,
        processing_time=result.processing_time,
    )
    
    # Save crops if enabled
    if config.save_detection_crops:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        visualizer.save_detection_crops(
            original, result.detections, config.crops_dir, image_name,
        )
    
    # Print individual results
    print(result.summary())
    
    return result


def detect_batch(
    detector: YOLODetector,
    visualizer: DetectionVisualizer,
    input_dir: str,
    output_dir: str,
    config: Config,
):
    """Detect objects in all images in a directory."""
    from PIL import Image
    
    image_paths = get_image_paths(input_dir)
    
    if not image_paths:
        print(f"❌ No images found in '{input_dir}'!")
        return []
    
    print(f"\n📂 Found {len(image_paths)} images in '{input_dir}'")
    print(f"   Output directory: '{output_dir}'\n")
    
    # Run batch detection
    results = detector.detect_batch(image_paths)
    
    # Annotate and save each result
    print(f"\n🎨 Saving annotated images...\n")
    for result in results:
        original = Image.open(result.image_path)
        basename = os.path.basename(result.image_path)
        output_path = os.path.join(output_dir, f"detected_{basename}")
        
        visualizer.save_annotated_image(
            original, result.detections, output_path,
            processing_time=result.processing_time,
        )
        
        if config.save_detection_crops:
            image_name = os.path.splitext(basename)[0]
            visualizer.save_detection_crops(
                original, result.detections, config.crops_dir, image_name,
            )
    
    # Print summary table
    print(f"\n{create_detection_summary_table(results)}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="🖼️ YOLO Car Detection — Image Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image", type=str,
        help="Path to a single image file",
    )
    input_group.add_argument(
        "--input_dir", type=str,
        help="Path to directory containing images",
    )
    
    # Output options
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for single image (default: out/detected_<name>)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="out",
        help="Output directory for batch processing (default: out/)",
    )
    
    # Detection parameters
    parser.add_argument(
        "--confidence", type=float, default=0.5,
        help="Minimum confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--iou", type=float, default=0.5,
        help="IoU threshold for NMS (default: 0.5)",
    )
    parser.add_argument(
        "--max_boxes", type=int, default=50,
        help="Maximum number of detections (default: 50)",
    )
    
    # Filtering
    parser.add_argument(
        "--driving_mode", action="store_true",
        help="Enable driving mode (filter for vehicle classes only)",
    )
    parser.add_argument(
        "--classes", nargs="+", type=str, default=None,
        help="Specific classes to detect (e.g., --classes car truck bus)",
    )
    
    # Output options
    parser.add_argument(
        "--save_crops", action="store_true",
        help="Save cropped detections as separate images",
    )
    parser.add_argument(
        "--save_json", action="store_true",
        help="Save detection results as JSON",
    )
    parser.add_argument(
        "--no_overlay", action="store_true",
        help="Disable the info overlay on output images",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Print detailed progress information",
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    # Build configuration
    if args.driving_mode:
        config = Config.driving_mode()
    else:
        config = Config()
    
    config.score_threshold = args.confidence
    config.iou_threshold = args.iou
    config.max_boxes = args.max_boxes
    config.save_detection_crops = args.save_crops
    config.show_count_overlay = not args.no_overlay
    config.verbose = args.verbose
    config.output_image_dir = args.output_dir
    
    if args.classes:
        config.target_classes = args.classes
        config.enable_driving_filter = False
    
    # Initialize pipeline
    detector = YOLODetector(config)
    visualizer = DetectionVisualizer(config)
    metrics = DetectionMetrics()
    
    # Run detection
    if args.image:
        # Single image mode
        output_path = args.output or os.path.join(
            config.output_image_dir,
            f"detected_{os.path.basename(args.image)}",
        )
        result = detect_single_image(
            detector, visualizer, args.image, output_path, config,
        )
        results = [result]
    else:
        # Batch mode
        results = detect_batch(
            detector, visualizer, args.input_dir, args.output_dir, config,
        )
    
    # Compute and display metrics
    if results:
        metrics.add_results(results)
        report = metrics.compute_detection_statistics()
        metrics.print_report(report)
        
        # Save JSON results
        if args.save_json:
            json_path = os.path.join(
                config.stats_dir, f"detections_{get_timestamp()}.json"
            )
            save_detection_results(results, json_path)
            metrics.save_report(
                os.path.join(config.stats_dir, f"report_{get_timestamp()}.json"),
                report,
            )
    
    print(f"\n✅ Detection complete! Results saved to '{config.output_image_dir}'")


if __name__ == "__main__":
    main()
