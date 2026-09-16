#!/usr/bin/env python3
"""
Generate Demo Assets
====================
Runs the upgraded YOLOv8 + Lane Detection + Distance Estimation pipeline
on a few sample images and saves the annotated results to the `assets/`
directory for GitHub documentation.
"""

import os
import cv2
import numpy as np
from src.yolov8_detector import YOLOv8Detector, YOLOv8Config
from src.lane_detection import LaneDetector
from src.distance_estimation import DistanceEstimator, CollisionAdvisor
from src.logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

RISK_COLORS = {
    "SAFE":    (0, 200, 80),     # green
    "WARNING": (0, 165, 255),    # orange
    "DANGER":  (0, 0, 255),      # red
}

def annotate_image(image_path, detector, lane_det, estimator, advisor):
    # Load image in BGR
    frame_bgr = cv2.imread(image_path)
    if frame_bgr is None:
        logger.error(f"Could not load image: {image_path}")
        return None
        
    h, w = frame_bgr.shape[:2]
    
    # BGR -> RGB for YOLOv8
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # Run detector
    result = detector.detect_from_array(frame_rgb, image_name=os.path.basename(image_path))
    
    # Run lane detection
    if lane_det:
        lane_result = lane_det.detect(frame_bgr)
        frame_bgr = lane_det.draw_lanes(frame_bgr, lane_result)
        
    # Run distance estimation and draw detections
    active_ids = []
    warnings = {}
    for det in result.detections:
        # Since these are single images, we can assign a mock track_id based on index
        mock_track_id = det.class_id + 100 # just some ID
        active_ids.append(mock_track_id)
        
        warn = advisor.assess(det.box, det.class_name, mock_track_id)
        warnings[mock_track_id] = warn
        
        # Draw bounding boxes with distance and risk color
        y1, x1, y2, x2 = (int(v) for v in det.box)
        risk = warn.risk_level
        colour = RISK_COLORS[risk]
        
        # Bounding box
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), colour, 2)
        
        # Label
        dist_str = f"{warn.distance_m:.1f}m"
        label = f"{det.class_name} {det.score:.0%} {dist_str}"
        
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_y = max(y1 - 8, 18)
        cv2.rectangle(frame_bgr, (x1, label_y - lh - 4), (x1 + lw + 6, label_y + 2), colour, -1)
        text_col = (0, 0, 0) if sum(colour) > 400 else (255, 255, 255)
        cv2.putText(frame_bgr, label, (x1 + 3, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_col, 1, cv2.LINE_AA)
                    
        # Danger warning text
        if risk == "DANGER":
            cv2.putText(frame_bgr, "COLLISION ALERT", (x1, y2 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
    advisor.prune_stale_tracks(active_ids)
    
    # Draw simple HUD on top corner
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (280, 80), (0, 0, 0), -1)
    frame_bgr = cv2.addWeighted(overlay, 0.55, frame_bgr, 0.45, 0)
    
    cv2.putText(frame_bgr, f"Model: YOLOv8m (COCO)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(frame_bgr, f"Detections: {len(result.detections)} objects", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(frame_bgr, f"Lane Status: Active", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
                
    return frame_bgr

def main():
    # Setup directories
    os.makedirs("assets", exist_ok=True)
    
    # Load pipeline
    cfg = YOLOv8Config(weights="models/yolov8m.pt", confidence=0.45, enable_tracking=False)
    detector = YOLOv8Detector(cfg)
    lane_det = LaneDetector()
    estimator = DistanceEstimator(focal_length_px=800.0)
    advisor = CollisionAdvisor(estimator)
    
    # Test images list
    test_images = ["images/0001.jpg", "images/0002.jpg", "images/0003.jpg", "images/test.jpg"]
    
    for idx, img_path in enumerate(test_images):
        if not os.path.exists(img_path):
            logger.warning(f"Skipping {img_path} (not found)")
            continue
            
        logger.info(f"Processing {img_path} for demo assets...")
        output_frame = annotate_image(img_path, detector, lane_det, estimator, advisor)
        
        if output_frame is not None:
            output_path = f"assets/demo_detection_{idx+1}.jpg"
            cv2.imwrite(output_path, output_frame)
            logger.info(f"Saved demo asset to {output_path}")

if __name__ == "__main__":
    main()
