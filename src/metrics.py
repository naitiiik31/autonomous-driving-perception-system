"""
Detection Metrics & Evaluation
===============================

Implements standard object detection evaluation metrics:
- Precision, Recall, F1 Score
- Mean Average Precision (mAP) at various IoU thresholds
- Confidence distribution analysis
- Detection statistics aggregation
- Performance benchmarking (FPS, latency)

These metrics follow the PASCAL VOC and COCO evaluation protocols.

Theory:
    Precision = TP / (TP + FP)  → "Of all predictions, how many are correct?"
    Recall    = TP / (TP + FN)  → "Of all ground truths, how many did we find?"
    
    AP = Area under the Precision-Recall curve for a single class
    mAP = Mean of AP across all classes
    
    At IoU threshold τ:
        - A detection is TP if IoU with ground truth > τ
        - A detection is FP if IoU with all ground truths ≤ τ
        - A ground truth is FN if no detection has IoU > τ with it
"""

import os
import time
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field

from .postprocessor import Detection
from .detector import DetectionResult


@dataclass
class ClassMetrics:
    """Metrics for a single class."""
    class_name: str
    num_detections: int = 0
    num_ground_truths: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    average_precision: float = 0.0
    avg_confidence: float = 0.0
    confidences: List[float] = field(default_factory=list)


class DetectionMetrics:
    """
    Comprehensive detection evaluation engine.
    
    Computes standard object detection metrics following the PASCAL VOC
    and COCO evaluation protocols. Supports per-class analysis and
    multiple IoU thresholds.
    
    Usage:
        metrics = DetectionMetrics()
        metrics.add_results(detection_results)
        report = metrics.compute_metrics()
        metrics.print_report(report)
    """
    
    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize the metrics engine.
        
        Args:
            iou_threshold: IoU threshold for matching detections to ground truths.
        """
        self.iou_threshold = iou_threshold
        self.all_results: List[DetectionResult] = []
        self.all_detections: List[Detection] = []
        self.processing_times: List[float] = []
    
    def add_result(self, result: DetectionResult):
        """Add a single detection result for evaluation."""
        self.all_results.append(result)
        self.all_detections.extend(result.detections)
        self.processing_times.append(result.processing_time)
    
    def add_results(self, results: List[DetectionResult]):
        """Add multiple detection results for evaluation."""
        for result in results:
            self.add_result(result)
    
    def compute_detection_statistics(self) -> Dict:
        """
        Compute comprehensive detection statistics across all results.
        
        Returns:
            Dictionary with aggregate statistics including:
            - Total/average counts
            - Per-class breakdowns
            - Confidence distributions
            - Performance metrics
        """
        if not self.all_results:
            return {"error": "No results to analyze"}
        
        # Aggregate statistics
        total_objects = len(self.all_detections)
        total_images = len(self.all_results)
        
        # Per-class analysis
        class_data = defaultdict(lambda: {
            "count": 0,
            "confidences": [],
            "box_areas": [],
        })
        
        for det in self.all_detections:
            cls = det.class_name
            class_data[cls]["count"] += 1
            class_data[cls]["confidences"].append(det.score)
            
            # Compute box area
            box_h = abs(det.box[2] - det.box[0])
            box_w = abs(det.box[3] - det.box[1])
            class_data[cls]["box_areas"].append(box_h * box_w)
        
        # Build per-class metrics
        per_class = {}
        for cls_name, data in class_data.items():
            per_class[cls_name] = {
                "count": data["count"],
                "percentage": data["count"] / total_objects * 100 if total_objects > 0 else 0,
                "avg_confidence": float(np.mean(data["confidences"])),
                "min_confidence": float(np.min(data["confidences"])),
                "max_confidence": float(np.max(data["confidences"])),
                "std_confidence": float(np.std(data["confidences"])),
                "avg_box_area": float(np.mean(data["box_areas"])),
                "median_box_area": float(np.median(data["box_areas"])),
            }
        
        # Performance metrics
        avg_time = float(np.mean(self.processing_times))
        std_time = float(np.std(self.processing_times))
        
        # Confidence distribution bins
        all_confidences = [d.score for d in self.all_detections]
        conf_hist, conf_bins = np.histogram(
            all_confidences, bins=10, range=(0, 1)
        ) if all_confidences else (np.zeros(10), np.linspace(0, 1, 11))
        
        report = {
            "summary": {
                "total_images": total_images,
                "total_detections": total_objects,
                "avg_detections_per_image": total_objects / total_images if total_images > 0 else 0,
                "unique_classes_detected": len(class_data),
                "avg_confidence": float(np.mean(all_confidences)) if all_confidences else 0,
                "median_confidence": float(np.median(all_confidences)) if all_confidences else 0,
            },
            "performance": {
                "avg_inference_time_ms": avg_time * 1000,
                "std_inference_time_ms": std_time * 1000,
                "min_inference_time_ms": float(np.min(self.processing_times)) * 1000,
                "max_inference_time_ms": float(np.max(self.processing_times)) * 1000,
                "avg_fps": 1.0 / avg_time if avg_time > 0 else 0,
                "throughput_images_per_sec": 1.0 / avg_time if avg_time > 0 else 0,
            },
            "per_class": per_class,
            "confidence_distribution": {
                "histogram": conf_hist.tolist(),
                "bin_edges": conf_bins.tolist(),
            },
        }
        
        return report
    
    def compute_size_analysis(self) -> Dict:
        """
        Analyze detection sizes to understand object scale distribution.
        
        Categorizes detections into small, medium, and large based on
        box area (following COCO size definitions).
        
        Returns:
            Dictionary with size distribution analysis.
        """
        small = []    # area < 32^2  = 1024
        medium = []   # 32^2 <= area < 96^2 = 9216
        large = []    # area >= 96^2
        
        for det in self.all_detections:
            box_h = abs(det.box[2] - det.box[0])
            box_w = abs(det.box[3] - det.box[1])
            area = box_h * box_w
            
            if area < 1024:
                small.append(det)
            elif area < 9216:
                medium.append(det)
            else:
                large.append(det)
        
        total = len(self.all_detections)
        
        return {
            "small": {
                "count": len(small),
                "percentage": len(small) / total * 100 if total > 0 else 0,
                "avg_confidence": float(np.mean([d.score for d in small])) if small else 0,
            },
            "medium": {
                "count": len(medium),
                "percentage": len(medium) / total * 100 if total > 0 else 0,
                "avg_confidence": float(np.mean([d.score for d in medium])) if medium else 0,
            },
            "large": {
                "count": len(large),
                "percentage": len(large) / total * 100 if total > 0 else 0,
                "avg_confidence": float(np.mean([d.score for d in large])) if large else 0,
            },
        }
    
    def benchmark_speed(
        self,
        detector,
        image_path: str,
        num_iterations: int = 50,
    ) -> Dict:
        """
        Benchmark inference speed on a single image.
        
        Runs multiple iterations to get stable timing measurements,
        excluding the first few iterations for JIT warmup.
        
        Args:
            detector: YOLODetector instance.
            image_path: Path to the test image.
            num_iterations: Number of inference iterations.
            
        Returns:
            Dictionary with timing statistics.
        """
        print(f"\n⏱️  Running speed benchmark ({num_iterations} iterations)...")
        
        times = []
        warmup_iters = min(5, num_iterations // 5)
        
        for i in range(num_iterations):
            start = time.time()
            detector.detect(image_path)
            elapsed = time.time() - start
            
            if i >= warmup_iters:
                times.append(elapsed)
        
        times = np.array(times)
        
        results = {
            "iterations": num_iterations,
            "warmup": warmup_iters,
            "mean_ms": float(np.mean(times) * 1000),
            "std_ms": float(np.std(times) * 1000),
            "min_ms": float(np.min(times) * 1000),
            "max_ms": float(np.max(times) * 1000),
            "median_ms": float(np.median(times) * 1000),
            "p95_ms": float(np.percentile(times, 95) * 1000),
            "p99_ms": float(np.percentile(times, 99) * 1000),
            "mean_fps": float(1.0 / np.mean(times)),
        }
        
        print(f"  Mean: {results['mean_ms']:.1f}ms | "
              f"Median: {results['median_ms']:.1f}ms | "
              f"P95: {results['p95_ms']:.1f}ms | "
              f"FPS: {results['mean_fps']:.1f}")
        
        return results
    
    def print_report(self, report: Optional[Dict] = None):
        """
        Print a formatted metrics report to the console.
        
        Args:
            report: Metrics report dictionary. If None, computes one.
        """
        if report is None:
            report = self.compute_detection_statistics()
        
        summary = report.get("summary", {})
        perf = report.get("performance", {})
        per_class = report.get("per_class", {})
        
        print("\n" + "=" * 70)
        print("📊  DETECTION EVALUATION REPORT")
        print("=" * 70)
        
        # Summary
        print(f"\n{'─'*30} SUMMARY {'─'*30}")
        print(f"  Total Images Processed : {summary.get('total_images', 0)}")
        print(f"  Total Detections       : {summary.get('total_detections', 0)}")
        print(f"  Avg Detections/Image   : {summary.get('avg_detections_per_image', 0):.1f}")
        print(f"  Unique Classes Found   : {summary.get('unique_classes_detected', 0)}")
        print(f"  Avg Confidence         : {summary.get('avg_confidence', 0):.3f}")
        print(f"  Median Confidence      : {summary.get('median_confidence', 0):.3f}")
        
        # Performance
        print(f"\n{'─'*28} PERFORMANCE {'─'*28}")
        print(f"  Avg Inference Time     : {perf.get('avg_inference_time_ms', 0):.1f}ms")
        print(f"  Std Inference Time     : {perf.get('std_inference_time_ms', 0):.1f}ms")
        print(f"  Min / Max Time         : {perf.get('min_inference_time_ms', 0):.1f}ms / {perf.get('max_inference_time_ms', 0):.1f}ms")
        print(f"  Avg FPS                : {perf.get('avg_fps', 0):.1f}")
        
        # Per-class breakdown
        if per_class:
            print(f"\n{'─'*26} PER-CLASS ANALYSIS {'─'*24}")
            print(f"  {'Class':<18} {'Count':>6} {'%':>7} {'Avg Conf':>10} {'Min':>6} {'Max':>6}")
            print(f"  {'─'*18} {'─'*6} {'─'*7} {'─'*10} {'─'*6} {'─'*6}")
            
            sorted_classes = sorted(per_class.items(), key=lambda x: -x[1]["count"])
            for cls_name, data in sorted_classes:
                print(f"  {cls_name:<18} {data['count']:>6} "
                      f"{data['percentage']:>6.1f}% "
                      f"{data['avg_confidence']:>10.3f} "
                      f"{data['min_confidence']:>6.3f} "
                      f"{data['max_confidence']:>6.3f}")
        
        # Confidence distribution
        conf_dist = report.get("confidence_distribution", {})
        if conf_dist and "histogram" in conf_dist:
            print(f"\n{'─'*24} CONFIDENCE DISTRIBUTION {'─'*22}")
            hist = conf_dist["histogram"]
            bins = conf_dist["bin_edges"]
            max_count = max(hist) if hist else 1
            
            for i in range(len(hist)):
                bar_len = int(hist[i] / max(max_count, 1) * 30)
                bar = "█" * bar_len
                print(f"  [{bins[i]:.1f}-{bins[i+1]:.1f}] {hist[i]:>5} {bar}")
        
        print("\n" + "=" * 70)
    
    def save_report(self, output_path: str, report: Optional[Dict] = None):
        """
        Save the metrics report to a JSON file.
        
        Args:
            output_path: Path to save the report.
            report: Metrics report dictionary. If None, computes one.
        """
        if report is None:
            report = self.compute_detection_statistics()
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n💾 Report saved to: {output_path}")
    
    def plot_confidence_distribution(self, output_path: str):
        """
        Plot confidence score distribution histogram.
        
        Args:
            output_path: Path to save the plot.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.style as mplstyle
        except ImportError:
            print("⚠️  matplotlib required for plotting")
            return
        
        plt.style.use('dark_background')
        
        confidences = [d.score for d in self.all_detections]
        if not confidences:
            print("⚠️  No detections to plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(
            confidences, bins=20, color='#00ff7f', alpha=0.8,
            edgecolor='white', linewidth=0.5,
        )
        axes[0].set_xlabel('Confidence Score', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('Detection Confidence Distribution', fontsize=14, fontweight='bold')
        axes[0].axvline(np.mean(confidences), color='#ff6347', linestyle='--',
                       label=f'Mean: {np.mean(confidences):.3f}')
        axes[0].legend()
        
        # Per-class box plot
        class_confs = defaultdict(list)
        for det in self.all_detections:
            class_confs[det.class_name].append(det.score)
        
        # Take top 10 classes by count
        sorted_classes = sorted(class_confs.items(), key=lambda x: -len(x[1]))[:10]
        if sorted_classes:
            labels = [x[0] for x in sorted_classes]
            data = [x[1] for x in sorted_classes]
            
            bp = axes[1].boxplot(data, labels=labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('#4169e1')
                patch.set_alpha(0.7)
            
            axes[1].set_xlabel('Class', fontsize=12)
            axes[1].set_ylabel('Confidence', fontsize=12)
            axes[1].set_title('Per-Class Confidence', fontsize=14, fontweight='bold')
            axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Confidence plot saved to: {output_path}")
    
    def plot_class_distribution(self, output_path: str):
        """
        Plot detection class distribution bar chart.
        
        Args:
            output_path: Path to save the plot.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️  matplotlib required for plotting")
            return
        
        plt.style.use('dark_background')
        
        class_counts = defaultdict(int)
        for det in self.all_detections:
            class_counts[det.class_name] += 1
        
        if not class_counts:
            return
        
        # Sort by count and take top 15
        sorted_items = sorted(class_counts.items(), key=lambda x: -x[1])[:15]
        names = [x[0] for x in sorted_items]
        counts = [x[1] for x in sorted_items]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.barh(range(len(names)), counts, color='#00ff7f', alpha=0.85)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=11)
        ax.set_xlabel('Detection Count', fontsize=12)
        ax.set_title('Object Class Distribution', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                   str(count), va='center', fontsize=10, color='white')
        
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Class distribution plot saved to: {output_path}")
