"""
Advanced Detection Visualization
=================================

Provides rich, publication-quality visualization of YOLO detection results.
Features include:
- Color-coded bounding boxes per class
- Confidence score overlays
- Detection count dashboard
- Processing time indicator
- Cropped detection extraction
- Side-by-side comparison views
- Detection heatmap generation
"""

import os
import colorsys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional, Dict
from collections import defaultdict

from .postprocessor import Detection
from .config import Config


class DetectionVisualizer:
    """
    Advanced visualization engine for object detection results.
    
    Draws annotated bounding boxes, confidence scores, class labels,
    and optional overlays on input images. Supports multiple output
    formats and visualization styles.
    
    Attributes:
        config (Config): Visualization configuration.
        colors (dict): Mapping of class names to RGB color tuples.
        font: PIL ImageFont for text rendering.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the visualizer.
        
        Args:
            config: Configuration object. If None, uses defaults.
        """
        self.config = config or Config()
        self.colors = self._generate_colors()
        self.font = self._load_font()
    
    def _load_font(self) -> ImageFont.ImageFont:
        """Load the font for text annotations."""
        try:
            # Try loading the custom font
            font = ImageFont.truetype(
                font=self.config.font_path,
                size=16,
            )
            return font
        except (IOError, OSError):
            try:
                # Fallback to a common system font
                font = ImageFont.truetype("arial.ttf", 16)
                return font
            except (IOError, OSError):
                # Final fallback to default
                return ImageFont.load_default()
    
    def _generate_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """
        Generate a visually distinct color palette for all classes.
        
        Uses HSV color space with evenly spaced hues for maximum
        visual distinction between classes.
        
        Returns:
            Dictionary mapping class names to RGB tuples.
        """
        colors = dict(self.config.class_colors)
        
        # Generate colors for any classes not in the predefined palette
        all_classes = []
        try:
            with open(self.config.classes_path, 'r') as f:
                all_classes = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            pass
        
        num_classes = max(len(all_classes), 80)
        for i, cls in enumerate(all_classes):
            if cls not in colors:
                hue = i / num_classes
                rgb = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
                colors[cls] = tuple(int(c * 255) for c in rgb)
        
        return colors
    
    def get_color(self, class_name: str) -> Tuple[int, int, int]:
        """Get the color for a given class name."""
        if class_name in self.colors:
            return self.colors[class_name]
        # Deterministic fallback based on hash
        hue = hash(class_name) % 360 / 360.0
        rgb = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        color = tuple(int(c * 255) for c in rgb)
        self.colors[class_name] = color
        return color
    
    def draw_detections(
        self,
        image: Image.Image,
        detections: List[Detection],
        show_confidence: bool = True,
        show_overlay: bool = True,
        processing_time: float = 0.0,
    ) -> Image.Image:
        """
        Draw all detections on an image with rich annotations.
        
        Features:
        - Color-coded bounding boxes with class labels
        - Confidence score display
        - Semi-transparent header overlay with detection count
        - Processing time indicator
        
        Args:
            image: Input PIL Image to annotate.
            detections: List of Detection objects to draw.
            show_confidence: Whether to show confidence scores.
            show_overlay: Whether to show the info overlay.
            processing_time: Processing time in seconds (for display).
            
        Returns:
            Annotated PIL Image (copy of original).
        """
        # Work on a copy
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        
        # Calculate dynamic sizing based on image dimensions
        img_w, img_h = annotated.size
        thickness = max(2, int((img_w + img_h) / 400))
        
        # Scale font size based on image
        font_size = max(12, int(self.config.font_scale * img_h))
        try:
            font = ImageFont.truetype(self.config.font_path, font_size)
        except (IOError, OSError):
            font = self.font
        
        # Scale small font for overlay
        small_font_size = max(10, font_size - 4)
        try:
            small_font = ImageFont.truetype(self.config.font_path, small_font_size)
        except (IOError, OSError):
            small_font = self.font
        
        # Draw each detection
        for det in detections:
            self._draw_single_detection(
                draw, det, font, thickness, show_confidence, img_w, img_h,
            )
        
        # Draw info overlay
        if show_overlay and self.config.show_count_overlay:
            self._draw_overlay(
                annotated, draw, detections, small_font, processing_time,
            )
        
        return annotated
    
    def _draw_single_detection(
        self,
        draw: ImageDraw.Draw,
        detection: Detection,
        font: ImageFont.ImageFont,
        thickness: int,
        show_confidence: bool,
        img_w: int,
        img_h: int,
    ):
        """
        Draw a single bounding box with label on the image.
        
        The bounding box is drawn with a colored border and a filled
        label background at the top of the box.
        """
        color = self.get_color(detection.class_name)
        box = detection.box  # [y_min, x_min, y_max, x_max]
        
        # Extract and clamp coordinates
        top = max(0, int(np.floor(box[0] + 0.5)))
        left = max(0, int(np.floor(box[1] + 0.5)))
        bottom = min(img_h, int(np.floor(box[2] + 0.5)))
        right = min(img_w, int(np.floor(box[3] + 0.5)))
        
        # Build label text
        if show_confidence:
            label = f"{detection.class_name} {detection.score:.0%}"
        else:
            label = detection.class_name
        
        # Get label dimensions
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            label_w = bbox[2] - bbox[0]
            label_h = bbox[3] - bbox[1]
        except AttributeError:
            # Fallback for older Pillow versions
            label_w, label_h = draw.textsize(label, font=font)
        
        # Draw bounding box (multiple rectangles for thickness)
        for t in range(thickness):
            draw.rectangle(
                [left + t, top + t, right - t, bottom - t],
                outline=color,
            )
        
        # Draw label background
        if top - label_h - 6 >= 0:
            label_y = top - label_h - 6
        else:
            label_y = top + 2
        
        label_x = left
        
        # Background rectangle for label
        draw.rectangle(
            [label_x, label_y, label_x + label_w + 8, label_y + label_h + 6],
            fill=color,
        )
        
        # Draw label text (use dark text for light backgrounds, white for dark)
        text_color = (0, 0, 0) if sum(color) > 400 else (255, 255, 255)
        draw.text(
            (label_x + 4, label_y + 2),
            label,
            fill=text_color,
            font=font,
        )
    
    def _draw_overlay(
        self,
        image: Image.Image,
        draw: ImageDraw.Draw,
        detections: List[Detection],
        font: ImageFont.ImageFont,
        processing_time: float,
    ):
        """
        Draw an info overlay with detection statistics.
        
        Shows:
        - Total object count
        - Per-class breakdown
        - Processing time / FPS
        """
        img_w, img_h = image.size
        
        # Count objects per class
        class_counts = defaultdict(int)
        for det in detections:
            class_counts[det.class_name] += 1
        
        # Build overlay text lines
        lines = [f"Detected: {len(detections)} objects"]
        
        if self.config.show_processing_time and processing_time > 0:
            fps = 1.0 / processing_time if processing_time > 0 else 0
            lines.append(f"Time: {processing_time*1000:.0f}ms | FPS: {fps:.1f}")
        
        # Add per-class counts (top 5)
        sorted_classes = sorted(class_counts.items(), key=lambda x: -x[1])
        for cls_name, count in sorted_classes[:5]:
            color = self.get_color(cls_name)
            lines.append(f"  {cls_name}: {count}")
        
        # Calculate overlay dimensions
        line_height = 18
        overlay_height = len(lines) * line_height + 16
        overlay_width = 220
        
        # Draw semi-transparent background
        overlay = Image.new('RGBA', (overlay_width, overlay_height), (0, 0, 0, 180))
        
        # Convert image to RGBA if needed, paste overlay
        if image.mode != 'RGBA':
            rgba_image = image.convert('RGBA')
        else:
            rgba_image = image
        
        rgba_image.paste(overlay, (8, 8), overlay)
        
        # Copy pixels back
        image.paste(rgba_image.convert(image.mode))
        
        # Re-create draw object for the modified image
        draw = ImageDraw.Draw(image)
        
        # Draw text
        y = 14
        for i, line in enumerate(lines):
            color = (255, 255, 255) if i < 2 else (200, 200, 200)
            draw.text((14, y), line, fill=color, font=font)
            y += line_height
    
    def save_annotated_image(
        self,
        image: Image.Image,
        detections: List[Detection],
        output_path: str,
        processing_time: float = 0.0,
    ) -> str:
        """
        Draw detections and save the annotated image.
        
        Args:
            image: Original PIL Image.
            detections: List of Detection objects.
            output_path: Path to save the annotated image.
            processing_time: Processing time for overlay display.
            
        Returns:
            Path to the saved image.
        """
        annotated = self.draw_detections(
            image, detections,
            show_confidence=self.config.show_confidence,
            show_overlay=self.config.show_count_overlay,
            processing_time=processing_time,
        )
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        # Save
        annotated.save(output_path, quality=95)
        
        if self.config.verbose:
            print(f"[VIS] 💾 Saved annotated image: {output_path}")
        
        return output_path
    
    def save_detection_crops(
        self,
        image: Image.Image,
        detections: List[Detection],
        output_dir: str,
        image_name: str = "image",
    ) -> List[str]:
        """
        Crop and save individual detected objects.
        
        Useful for building classification datasets or analyzing
        individual detections in detail.
        
        Args:
            image: Original PIL Image.
            detections: List of Detection objects.
            output_dir: Directory to save cropped images.
            image_name: Base name for the crop files.
            
        Returns:
            List of paths to saved crop images.
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []
        
        img_w, img_h = image.size
        
        for i, det in enumerate(detections):
            # Extract box coordinates with clamping
            top = max(0, int(det.box[0]))
            left = max(0, int(det.box[1]))
            bottom = min(img_h, int(det.box[2]))
            right = min(img_w, int(det.box[3]))
            
            if right <= left or bottom <= top:
                continue
            
            # Crop and save
            crop = image.crop((left, top, right, bottom))
            crop_name = f"{image_name}_{det.class_name}_{i}_{det.score:.2f}.jpg"
            crop_path = os.path.join(output_dir, crop_name)
            crop.save(crop_path, quality=95)
            saved_paths.append(crop_path)
        
        if self.config.verbose:
            print(f"[VIS] ✂️  Saved {len(saved_paths)} detection crops to: {output_dir}")
        
        return saved_paths
    
    def create_comparison(
        self,
        original: Image.Image,
        annotated: Image.Image,
        output_path: str,
    ) -> str:
        """
        Create a side-by-side comparison of original and annotated images.
        
        Args:
            original: Original PIL Image.
            annotated: Annotated PIL Image.
            output_path: Path to save the comparison image.
            
        Returns:
            Path to the saved comparison image.
        """
        # Match sizes
        w = max(original.size[0], annotated.size[0])
        h = max(original.size[1], annotated.size[1])
        
        original_resized = original.resize((w, h), Image.BICUBIC)
        annotated_resized = annotated.resize((w, h), Image.BICUBIC)
        
        # Create side-by-side canvas
        comparison = Image.new('RGB', (w * 2 + 4, h + 40), (30, 30, 30))
        comparison.paste(original_resized, (0, 40))
        comparison.paste(annotated_resized, (w + 4, 40))
        
        # Add labels
        draw = ImageDraw.Draw(comparison)
        try:
            title_font = ImageFont.truetype(self.config.font_path, 20)
        except (IOError, OSError):
            title_font = self.font
        
        draw.text((w // 2 - 30, 10), "Original", fill=(200, 200, 200), font=title_font)
        draw.text((w + w // 2 - 30, 10), "Detected", fill=(0, 255, 127), font=title_font)
        
        comparison.save(output_path, quality=95)
        
        if self.config.verbose:
            print(f"[VIS] 🔄 Saved comparison: {output_path}")
        
        return output_path
    
    def generate_detection_heatmap(
        self,
        image_shape: Tuple[int, int],
        all_detections: List[List[Detection]],
        output_path: str,
    ) -> str:
        """
        Generate a spatial heatmap showing detection density.
        
        Useful for analyzing which regions of images tend to have
        more detected objects (e.g., road center for cars).
        
        Args:
            image_shape: (height, width) of the reference image.
            all_detections: List of detection lists (one per image).
            output_path: Path to save the heatmap image.
            
        Returns:
            Path to the saved heatmap.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("[VIS] ⚠️  matplotlib required for heatmap generation")
            return ""
        
        h, w = image_shape
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        for detections in all_detections:
            for det in detections:
                y_min = max(0, int(det.box[0]))
                x_min = max(0, int(det.box[1]))
                y_max = min(h, int(det.box[2]))
                x_max = min(w, int(det.box[3]))
                heatmap[y_min:y_max, x_min:x_max] += 1
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        im = ax.imshow(heatmap, cmap='hot', interpolation='gaussian')
        ax.set_title('Detection Density Heatmap', fontsize=16, fontweight='bold')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=ax, label='Detection Density')
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if self.config.verbose:
            print(f"[VIS] 🗺️  Saved detection heatmap: {output_path}")
        
        return output_path
