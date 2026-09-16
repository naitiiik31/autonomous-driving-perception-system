"""
YOLO Model Loading & Architecture
==================================

Handles loading the pre-trained YOLO v2 model from TensorFlow SavedModel
format, reading anchor boxes, and reading class definitions.

The model architecture follows DarkNet-19 as the backbone with YOLO v2
detection head on top. The model outputs a tensor of shape:
    (batch_size, grid_h, grid_w, num_anchors * (5 + num_classes))

Where 5 = [tx, ty, tw, th, confidence] per anchor box.
"""

import os
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Optional


class YOLOModel:
    """
    YOLO v2 model wrapper for loading and inference.
    
    This class encapsulates the TensorFlow SavedModel and provides
    methods for preprocessing, inference, and output decoding.
    
    Attributes:
        model: Loaded TensorFlow SavedModel.
        class_names: List of object class names.
        anchors: NumPy array of anchor box dimensions (shape: [N, 2]).
        num_classes: Number of object classes.
        num_anchors: Number of anchor boxes per grid cell.
        input_size: Expected model input image size (H, W).
    """
    
    def __init__(
        self,
        model_path: str = "model_data",
        classes_path: str = "model_data/coco_classes.txt",
        anchors_path: str = "model_data/yolo_anchors.txt",
        input_size: Tuple[int, int] = (608, 608),
    ):
        """
        Initialize the YOLO model.
        
        Args:
            model_path: Path to TensorFlow SavedModel directory.
            classes_path: Path to file containing class names (one per line).
            anchors_path: Path to file containing anchor box dimensions.
            input_size: Expected model input image size (height, width).
        """
        self.input_size = input_size
        self.class_names = self._read_classes(classes_path)
        self.anchors = self._read_anchors(anchors_path)
        self.num_classes = len(self.class_names)
        self.num_anchors = len(self.anchors)
        
        # Load the TensorFlow SavedModel
        self.model = self._load_model(model_path)
        
        print(f"[MODEL] ✅ Loaded YOLO v2 model successfully")
        print(f"[MODEL]    Classes: {self.num_classes} | Anchors: {self.num_anchors}")
        print(f"[MODEL]    Input size: {self.input_size}")
    
    def _load_model(self, model_path: str) -> tf.saved_model.load:
        """
        Load a TensorFlow SavedModel from disk.
        
        Args:
            model_path: Path to the SavedModel directory.
            
        Returns:
            Loaded TensorFlow model.
            
        Raises:
            FileNotFoundError: If the model path does not exist.
            RuntimeError: If the model fails to load.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at '{model_path}'. "
                f"Please ensure the SavedModel directory exists."
            )
        
        try:
            model = tf.saved_model.load(model_path)
            print(f"[MODEL] 📦 Loaded SavedModel from: {model_path}")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    @staticmethod
    def _read_classes(classes_path: str) -> List[str]:
        """
        Read class names from a text file.
        
        Args:
            classes_path: Path to file with one class name per line.
            
        Returns:
            List of class name strings.
        """
        if not os.path.exists(classes_path):
            raise FileNotFoundError(f"Classes file not found: {classes_path}")
        
        with open(classes_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"[MODEL] 📋 Loaded {len(class_names)} classes from: {classes_path}")
        return class_names
    
    @staticmethod
    def _read_anchors(anchors_path: str) -> np.ndarray:
        """
        Read anchor box dimensions from a text file.
        
        The anchor file should contain comma-separated values that represent
        width-height pairs: w1, h1, w2, h2, ...
        
        Args:
            anchors_path: Path to anchor definitions file.
            
        Returns:
            NumPy array of shape (N, 2) with [width, height] per anchor.
        """
        if not os.path.exists(anchors_path):
            raise FileNotFoundError(f"Anchors file not found: {anchors_path}")
        
        with open(anchors_path, 'r') as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        
        print(f"[MODEL] ⚓ Loaded {len(anchors)} anchors from: {anchors_path}")
        return anchors
    
    def predict(self, image_data: np.ndarray) -> tf.Tensor:
        """
        Run inference on preprocessed image data.
        
        Args:
            image_data: Preprocessed image array of shape 
                        (1, height, width, 3), normalized to [0, 1].
                        
        Returns:
            Raw model output tensor.
        """
        image_tensor = tf.constant(image_data, dtype=tf.float32)
        
        # Get the serving function from SavedModel
        infer = self.model.signatures.get("serving_default")
        
        if infer is None:
            # Fallback: try calling the model directly
            output = self.model(image_tensor)
        else:
            # Use the serving signature
            result = infer(image_tensor)
            # Get the first output (model may have multiple outputs)
            output_key = list(result.keys())[0]
            output = result[output_key]
        
        return output
    
    def preprocess_image(self, image_path: str) -> Tuple[object, np.ndarray]:
        """
        Load and preprocess an image for model inference.
        
        Steps:
        1. Load the image from disk
        2. Resize to model input size
        3. Normalize pixel values to [0, 1]
        4. Add batch dimension
        
        Args:
            image_path: Path to the input image file.
            
        Returns:
            Tuple of (original PIL Image, preprocessed numpy array).
        """
        from PIL import Image
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load original image
        image = Image.open(image_path)
        
        # Convert to RGB if the image has an alpha channel (e.g., RGBA PNGs)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize for model input
        resized = image.resize(
            (self.input_size[1], self.input_size[0]),  # PIL uses (w, h)
            Image.BICUBIC
        )
        
        # Normalize to [0, 1] and add batch dimension
        image_data = np.array(resized, dtype='float32') / 255.0
        image_data = np.expand_dims(image_data, axis=0)
        
        return image, image_data
    
    def get_class_color(self, class_name: str, class_colors: dict = None) -> Tuple[int, int, int]:
        """
        Get a deterministic color for a class name.
        
        Args:
            class_name: Name of the object class.
            class_colors: Optional dictionary mapping class names to RGB tuples.
            
        Returns:
            RGB color tuple.
        """
        if class_colors and class_name in class_colors:
            return class_colors[class_name]
        
        # Generate deterministic color based on class name hash
        import colorsys
        hue = hash(class_name) % 360 / 360.0
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.95)
        return tuple(int(c * 255) for c in rgb)
    
    def summary(self) -> str:
        """Return a formatted summary of the model configuration."""
        lines = [
            "┌─────────────────────────────────────┐",
            "│       YOLO v2 Model Summary         │",
            "├─────────────────────────────────────┤",
            f"│  Classes      : {self.num_classes:<20}│",
            f"│  Anchors      : {self.num_anchors:<20}│",
            f"│  Input Size   : {str(self.input_size):<20}│",
            f"│  Grid Size    : {str((self.input_size[0]//32, self.input_size[1]//32)):<20}│",
            f"│  Predictions  : {self.num_anchors * (self.input_size[0]//32) * (self.input_size[1]//32):<20}│",
            "└─────────────────────────────────────┘",
        ]
        return "\n".join(lines)
