#!/usr/bin/env python3
"""
📊 YOLO Car Detection — Evaluation & Benchmarking Script
=========================================================

Run comprehensive evaluation of the YOLO detection pipeline,
including detection statistics, speed benchmarks, and visualization.

Usage:
    # Full evaluation on all images
    python evaluate.py --images images/
    
    # Speed benchmark
    python evaluate.py --benchmark --image images/test.jpg
    
    # Generate evaluation report with plots
    python evaluate.py --images images/ --plots --report
    
    # Evaluate with driving mode
    python evaluate.py --images images/ --driving_mode --verbose
"""

import os
import sys
import argparse

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


def run_evaluation(
    detector: YOLODetector,
    image_paths: list,
    config: Config,
    generate_plots: bool = False,
    save_report: bool = False,
):
    """
    Run full evaluation on a set of images.
    
    Args:
        detector: Initialized YOLO detector.
        image_paths: List of image paths to evaluate.
        config: Pipeline configuration.
        generate_plots: Whether to generate visualization plots.
        save_report: Whether to save the report to JSON.
    """
    print(f"\n📊 Running evaluation on {len(image_paths)} images...\n")
    
    # Run detection on all images
    results = detector.detect_batch(image_paths)
    
    # Initialize metrics
    metrics = DetectionMetrics()
    metrics.add_results(results)
    
    # Compute and print report
    report = metrics.compute_detection_statistics()
    metrics.print_report(report)
    
    # Size analysis
    size_analysis = metrics.compute_size_analysis()
    print(f"\n{'─'*28} SIZE ANALYSIS {'─'*28}")
    for size_cat, data in size_analysis.items():
        print(f"  {size_cat.upper():<10}: {data['count']:>5} detections "
              f"({data['percentage']:.1f}%) | "
              f"avg conf: {data['avg_confidence']:.3f}")
    
    # Print summary table
    print(f"\n{create_detection_summary_table(results[:20])}")
    
    if len(results) > 20:
        print(f"  ... (showing first 20 of {len(results)} images)")
    
    # Generate plots
    if generate_plots:
        print(f"\n📈 Generating evaluation plots...\n")
        
        try:
            metrics.plot_confidence_distribution(
                os.path.join(config.stats_dir, "eval_confidence_dist.png")
            )
            metrics.plot_class_distribution(
                os.path.join(config.stats_dir, "eval_class_dist.png")
            )
            
            # Generate heatmap
            visualizer = DetectionVisualizer(config)
            if results:
                all_dets = [r.detections for r in results]
                visualizer.generate_detection_heatmap(
                    results[0].image_shape,
                    all_dets,
                    os.path.join(config.stats_dir, "eval_heatmap.png"),
                )
        except Exception as e:
            print(f"⚠️  Plot generation error: {e}")
    
    # Save report
    if save_report:
        timestamp = get_timestamp()
        report_path = os.path.join(config.stats_dir, f"eval_report_{timestamp}.json")
        metrics.save_report(report_path, report)
        
        results_path = os.path.join(config.stats_dir, f"eval_detections_{timestamp}.json")
        save_detection_results(results, results_path)
    
    return results, report


def run_benchmark(
    detector: YOLODetector,
    image_path: str,
    num_iterations: int = 50,
    config: Config = None,
):
    """
    Run speed benchmark on a single image.
    
    Args:
        detector: Initialized YOLO detector.
        image_path: Path to benchmark image.
        num_iterations: Number of iterations.
        config: Pipeline configuration.
    """
    print(f"\n⏱️  Running speed benchmark on: {image_path}")
    print(f"   Iterations: {num_iterations}\n")
    
    metrics = DetectionMetrics()
    benchmark = metrics.benchmark_speed(detector, image_path, num_iterations)
    
    print(f"\n{'='*50}")
    print(f"  ⏱️  SPEED BENCHMARK RESULTS")
    print(f"{'='*50}")
    print(f"  Mean Latency    : {benchmark['mean_ms']:.1f}ms")
    print(f"  Median Latency  : {benchmark['median_ms']:.1f}ms")
    print(f"  Std Dev         : {benchmark['std_ms']:.1f}ms")
    print(f"  Min / Max       : {benchmark['min_ms']:.1f}ms / {benchmark['max_ms']:.1f}ms")
    print(f"  P95 Latency     : {benchmark['p95_ms']:.1f}ms")
    print(f"  P99 Latency     : {benchmark['p99_ms']:.1f}ms")
    print(f"  Mean FPS        : {benchmark['mean_fps']:.1f}")
    print(f"{'='*50}\n")
    
    # Save benchmark results
    if config and config.save_statistics:
        import json
        benchmark_path = os.path.join(
            config.stats_dir, f"benchmark_{get_timestamp()}.json"
        )
        os.makedirs(os.path.dirname(benchmark_path) or '.', exist_ok=True)
        with open(benchmark_path, 'w') as f:
            json.dump(benchmark, f, indent=2)
        print(f"  💾 Benchmark saved to: {benchmark_path}")
    
    return benchmark


def main():
    parser = argparse.ArgumentParser(
        description="📊 YOLO Car Detection — Evaluation & Benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Evaluation mode
    parser.add_argument(
        "--images", type=str, default=None,
        help="Directory of images to evaluate",
    )
    
    # Benchmark mode
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run speed benchmark",
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Single image for benchmark (default: first image in images/)",
    )
    parser.add_argument(
        "--iterations", type=int, default=50,
        help="Number of benchmark iterations (default: 50)",
    )
    
    # Detection parameters
    parser.add_argument(
        "--confidence", type=float, default=0.5,
        help="Confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--driving_mode", action="store_true",
        help="Enable driving mode filtering",
    )
    
    # Output options
    parser.add_argument(
        "--plots", action="store_true",
        help="Generate evaluation plots",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Save evaluation report to JSON",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Print detailed progress",
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    # Build configuration
    if args.driving_mode:
        config = Config.driving_mode()
    else:
        config = Config()
    
    config.score_threshold = args.confidence
    config.verbose = args.verbose
    config.save_statistics = True
    
    # Initialize detector
    detector = YOLODetector(config)
    
    if args.images:
        # Full evaluation mode
        image_paths = get_image_paths(args.images)
        if not image_paths:
            print(f"❌ No images found in '{args.images}'")
            sys.exit(1)
        
        run_evaluation(
            detector, image_paths, config,
            generate_plots=args.plots,
            save_report=args.report,
        )
    
    if args.benchmark:
        # Benchmark mode
        bench_image = args.image
        if not bench_image:
            # Use first available image
            paths = get_image_paths(config.input_image_dir)
            if paths:
                bench_image = paths[0]
            else:
                print("❌ No image specified and no images found in images/")
                sys.exit(1)
        
        run_benchmark(detector, bench_image, args.iterations, config)
    
    if not args.images and not args.benchmark:
        # Default: run both with default settings
        image_paths = get_image_paths(config.input_image_dir)
        if image_paths:
            run_evaluation(
                detector, image_paths[:10], config,
                generate_plots=True,
                save_report=True,
            )
            run_benchmark(detector, image_paths[0], 20, config)
        else:
            print("❌ No images found. Specify --images or --image flag.")
            sys.exit(1)
    
    print(f"\n✅ Evaluation complete! Check '{config.stats_dir}' for reports.\n")


if __name__ == "__main__":
    main()
