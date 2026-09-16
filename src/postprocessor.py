import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class Detection:
    """
    Represents a single object detection.
    
    Attributes:
        box: Bounding box coordinates [y_min, x_min, y_max, x_max].
        score: Detection confidence score (0.0 to 1.0).
        class_id: Integer class identifier.
        class_name: Human-readable class name.
    """
    box: np.ndarray      # [y_min, x_min, y_max, x_max]
    score: float
    class_id: int
    class_name: str


class PostProcessor:
    """
    YOLO output post-processing pipeline.
    
    Converts raw neural network outputs into actionable detection results
    through a multi-step pipeline of decoding, filtering, and deduplication.
    
    The pipeline:
        Raw Output → Decode Boxes → Filter by Confidence → NMS → Scale to Image
    """
    
    def __init__(
        self,
        anchors: np.ndarray,
        num_classes: int,
        class_names: List[str],
        score_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        max_boxes: int = 50,
    ):
        """
        Initialize the post-processor.
        
        Args:
            anchors: Anchor box dimensions, shape (num_anchors, 2).
            num_classes: Total number of object classes.
            class_names: List of class name strings.
            score_threshold: Minimum score to keep a detection.
            iou_threshold: IoU threshold for Non-Max Suppression.
            max_boxes: Maximum number of output detections.
        """
        self.anchors = anchors
        self.num_classes = num_classes
        self.class_names = class_names
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.max_boxes = max_boxes
        self.num_anchors = len(anchors)
    
    def yolo_head(self, feats: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Convert raw YOLO output features to bounding box parameters.
        
        This function decodes the raw network output into interpretable
        bounding box coordinates, confidence scores, and class probabilities.
        
        Theory:
            The network outputs a tensor of shape (batch, H, W, B*(5+C)) where:
            - B = number of anchors
            - 5 = [tx, ty, tw, th, to]
            - C = number of classes
            
            Decoding:
                box_xy = sigmoid(tx, ty) + cell_offset  → normalized [0,1]
                box_wh = anchor_size * exp(tw, th)       → normalized [0,1]
                confidence = sigmoid(to)
                class_probs = softmax(tc1, ..., tcC)
        
        Args:
            feats: Raw model output tensor.
            
        Returns:
            Tuple of (box_xy, box_wh, box_confidence, box_class_probs).
        """
        anchors_tensor = K.reshape(
            K.variable(self.anchors), [1, 1, 1, self.num_anchors, 2]
        )
        
        # Dynamic grid dimensions
        conv_dims = K.shape(feats)[1:3]
        
        # Create grid of cell offsets
        conv_height_index = K.arange(0, stop=conv_dims[0])
        conv_width_index = K.arange(0, stop=conv_dims[1])
        conv_height_index = K.tile(conv_height_index, [conv_dims[1]])
        
        conv_width_index = K.tile(
            K.expand_dims(conv_width_index, 0), [conv_dims[0], 1]
        )
        conv_width_index = K.flatten(K.transpose(conv_width_index))
        conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
        conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
        conv_index = K.cast(conv_index, K.dtype(feats))
        
        # Reshape features: (batch, H, W, B*(5+C)) → (batch, H, W, B, 5+C)
        feats = K.reshape(
            feats, [-1, conv_dims[0], conv_dims[1], self.num_anchors, self.num_classes + 5]
        )
        conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))
        
        # Decode predictions
        box_xy = K.sigmoid(feats[..., :2])          # Center coordinates
        box_wh = K.exp(feats[..., 2:4])             # Width and height
        box_confidence = K.sigmoid(feats[..., 4:5]) # Objectness
        box_class_probs = K.softmax(feats[..., 5:]) # Class probabilities
        
        # Adjust to absolute grid coordinates, then normalize to [0, 1]
        box_xy = (box_xy + conv_index) / conv_dims
        box_wh = box_wh * anchors_tensor / conv_dims
        
        return box_xy, box_wh, box_confidence, box_class_probs
    
    def yolo_boxes_to_corners(
        self, box_xy: tf.Tensor, box_wh: tf.Tensor
    ) -> tf.Tensor:
        """
        Convert YOLO box center/size format to corner format.
        
        Conversion:
            (center_x, center_y, width, height) → (y_min, x_min, y_max, x_max)
        
        Args:
            box_xy: Box centers, shape (..., 2).
            box_wh: Box dimensions, shape (..., 2).
            
        Returns:
            Boxes in corner format, shape (..., 4).
        """
        box_mins = box_xy - (box_wh / 2.0)
        box_maxes = box_xy + (box_wh / 2.0)
        
        return K.concatenate([
            box_mins[..., 1:2],   # y_min
            box_mins[..., 0:1],   # x_min
            box_maxes[..., 1:2],  # y_max
            box_maxes[..., 0:1],  # x_max
        ])
    
    def yolo_filter_boxes(
        self,
        boxes: tf.Tensor,
        box_confidence: tf.Tensor,
        box_class_probs: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Filter YOLO boxes by confidence threshold.
        
        For each box, the detection score is computed as:
            score = P(Object) × P(Class_i | Object)
        
        Only boxes with max class score >= threshold are kept.
        
        Args:
            boxes: Box corner coordinates.
            box_confidence: Objectness probability per box.
            box_class_probs: Class probability distribution per box.
            
        Returns:
            Tuple of (filtered_boxes, filtered_scores, filtered_classes).
        """
        # Compute per-class detection scores
        box_scores = box_confidence * box_class_probs  # (H, W, B, C)
        
        # Get best class for each box
        box_classes = K.argmax(box_scores, axis=-1)      # (H, W, B)
        box_class_scores = K.max(box_scores, axis=-1)    # (H, W, B)
        
        # Create boolean mask for scores above threshold
        prediction_mask = box_class_scores >= self.score_threshold
        
        # Apply mask using tf.boolean_mask
        boxes = tf.boolean_mask(boxes, prediction_mask)
        scores = tf.boolean_mask(box_class_scores, prediction_mask)
        classes = tf.boolean_mask(box_classes, prediction_mask)
        
        return boxes, scores, classes
    
    def yolo_non_max_suppression(
        self,
        scores: tf.Tensor,
        boxes: tf.Tensor,
        classes: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Apply Non-Maximum Suppression (NMS) to remove overlapping detections.
        
        NMS Algorithm:
        1. Sort detections by confidence score (descending)
        2. Select the highest-scoring detection
        3. Remove all detections with IoU > threshold with the selected one
        4. Repeat from step 2 until no detections remain
        
        This implementation uses tf.image.non_max_suppression which is
        optimized for GPU execution.
        
        Args:
            scores: Detection scores after filtering.
            boxes: Detection box coordinates after filtering.
            classes: Detection class IDs after filtering.
            
        Returns:
            Tuple of (final_scores, final_boxes, final_classes) after NMS.
        """
        max_boxes_tensor = tf.constant(self.max_boxes, dtype=tf.int32)
        
        # Apply TensorFlow's optimized NMS
        nms_indices = tf.image.non_max_suppression(
            boxes,
            scores,
            max_boxes_tensor,
            iou_threshold=self.iou_threshold,
        )
        
        # Gather the surviving detections
        scores = K.gather(scores, nms_indices)
        boxes = K.gather(boxes, nms_indices)
        classes = K.gather(classes, nms_indices)
        
        return scores, boxes, classes
    
    def process(
        self,
        yolo_outputs: tf.Tensor,
        image_shape: Tuple[int, int],
        target_classes: Optional[List[str]] = None,
    ) -> List[Detection]:
        """
        Full post-processing pipeline: decode → filter → NMS → scale → output.
        
        This is the main entry point for post-processing raw YOLO model 
        outputs into a list of Detection objects.
        
        Args:
            yolo_outputs: Raw model output tensor.
            image_shape: Original image dimensions (height, width).
            target_classes: Optional list of class names to filter for.
            
        Returns:
            List of Detection objects sorted by confidence (descending).
        """
        # Step 1: Decode raw output to box parameters
        box_xy, box_wh, box_confidence, box_class_probs = self.yolo_head(yolo_outputs)
        
        # Step 2: Convert center/size to corner format
        boxes = self.yolo_boxes_to_corners(box_xy, box_wh)
        
        # Step 3: Filter by confidence threshold
        boxes, scores, classes = self.yolo_filter_boxes(
            boxes, box_confidence, box_class_probs
        )
        
        # Step 4: Scale boxes to original image dimensions
        height, width = float(image_shape[0]), float(image_shape[1])
        image_dims = K.stack([height, width, height, width])
        image_dims = K.reshape(image_dims, [1, 4])
        boxes = boxes * image_dims
        
        # Step 5: Apply Non-Max Suppression
        scores, boxes, classes = self.yolo_non_max_suppression(scores, boxes, classes)
        
        # Step 6: Convert to Detection objects
        detections = []
        scores_np = scores.numpy()
        boxes_np = boxes.numpy()
        classes_np = classes.numpy()
        
        for i in range(len(scores_np)):
            class_id = int(classes_np[i])
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            
            # Apply target class filter if specified
            if target_classes and class_name not in target_classes:
                continue
            
            detections.append(Detection(
                box=boxes_np[i],
                score=float(scores_np[i]),
                class_id=class_id,
                class_name=class_name,
            ))
        
        # Sort by confidence score (descending)
        detections.sort(key=lambda d: d.score, reverse=True)
        
        return detections
    
    @staticmethod
    def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Compute Intersection over Union (IoU) between two boxes.
        
        IoU = Area(Intersection) / Area(Union)
        
        Args:
            box1: First box [y_min, x_min, y_max, x_max].
            box2: Second box [y_min, x_min, y_max, x_max].
            
        Returns:
            IoU value between 0.0 and 1.0.
        """
        # Intersection coordinates
        inter_y_min = max(box1[0], box2[0])
        inter_x_min = max(box1[1], box2[1])
        inter_y_max = min(box1[2], box2[2])
        inter_x_max = min(box1[3], box2[3])
        
        # Intersection area (clip to zero for non-overlapping boxes)
        inter_area = max(0, inter_y_max - inter_y_min) * max(0, inter_x_max - inter_x_min)
        
        # Individual box areas
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Union area
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
