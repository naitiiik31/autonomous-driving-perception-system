#!/usr/bin/env python3
"""
🎬 Production Video Pipeline
=============================
Unified real-time video processing integrating:
  • YOLOv8 object detection
  • ByteTrack multi-object tracking
  • Lane detection (Hough Transform)
  • Monocular distance estimation
  • TTC-based collision warning
  • FPS counter + HUD overlay

Usage:
    python detect_video_v2.py --video path/to/video.mp4
    python detect_video_v2.py --video video.mp4 --driving_mode --save_json
    python detect_video_v2.py --webcam 0
"""

import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.logger import setup_logging, get_logger
from src.yolov8_detector import YOLOv8Detector, YOLOv8Config
from src.lane_detection import LaneDetector
from src.distance_estimation import DistanceEstimator, CollisionAdvisor

setup_logging()
logger = get_logger(__name__)

# ── Risk colours (BGR) ────────────────────────────────────────────
RISK_COLORS = {
    "SAFE":    (0,  200,  80),    # green
    "WARNING": (0,  165, 255),    # orange
    "DANGER":  (0,   0,  255),    # red
}


def draw_hud(
    frame: np.ndarray,
    frame_idx: int,
    fps: float,
    num_detections: int,
    num_warnings: int,
    num_danger: int,
) -> np.ndarray:
    """Draw a corner HUD with live stats."""
    import cv2
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent black panel
    cv2.rectangle(overlay, (0, 0), (270, 130), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    lines = [
        f"Frame : {frame_idx:>6}",
        f"FPS   : {fps:>6.1f}",
        f"Objs  : {num_detections:>6}",
        f"WARN  : {num_warnings:>6}",
        f"DANGER: {num_danger:>6}",
    ]
    colors = [(200, 200, 200)] * 3 + [(0, 165, 255), (0, 0, 255)]

    for i, (line, colour) in enumerate(zip(lines, colors)):
        cv2.putText(frame, line, (10, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, colour, 1, cv2.LINE_AA)
    return frame


def draw_detections_v2(
    frame:       np.ndarray,
    detections:  list,
    warnings:    dict,       # track_id → CollisionWarning
    estimator:   DistanceEstimator,
) -> np.ndarray:
    """Draw bounding boxes with distance and risk colour coding."""
    import cv2

    h, w = frame.shape[:2]

    for det in detections:
        y1, x1, y2, x2 = (int(v) for v in det.box)

        # Get collision warning for this track
        warning  = warnings.get(det.track_id)
        risk     = warning.risk_level if warning else "SAFE"
        colour   = RISK_COLORS[risk]

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        # Label
        dist_str = f"{warning.distance_m:.1f}m" if warning else ""
        tid_str  = f"#{det.track_id}" if det.track_id is not None else ""
        label    = f"{det.class_name} {det.score:.0%} {tid_str} {dist_str}".strip()

        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_y = max(y1 - 8, 18)
        cv2.rectangle(frame, (x1, label_y - lh - 4), (x1 + lw + 6, label_y + 2), colour, -1)
        text_col = (0, 0, 0) if sum(colour) > 400 else (255, 255, 255)
        cv2.putText(frame, label, (x1 + 3, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_col, 1, cv2.LINE_AA)

        # TTC badge for DANGER
        if risk == "DANGER" and warning and warning.ttc_s > 0:
            badge = f"TTC {warning.ttc_s:.1f}s"
            cv2.putText(frame, badge, (x1, y2 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return frame


def process_video(args):
    import cv2

    # ── Init components ──────────────────────────────────────────
    cfg = YOLOv8Config(
        weights          = args.weights,
        confidence       = args.confidence,
        iou              = args.iou,
        enable_tracking  = True,
        enable_driving_filter = args.driving_mode,
    )
    detector  = YOLOv8Detector(cfg)
    lane_det  = LaneDetector() if args.lane_detection else None
    estimator = DistanceEstimator(focal_length_px=args.focal_length)
    advisor   = CollisionAdvisor(estimator)

    # ── Open video source ────────────────────────────────────────
    source = int(args.webcam) if args.webcam is not None else args.video
    cap    = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Cannot open video source: %s", source)
        return

    fps_in   = cap.get(cv2.CAP_PROP_FPS) or 30
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info("Source: %s | %dx%d @ %.0f FPS | %d frames",
                source, width, height, fps_in, total_fr)

    # ── Output writer ────────────────────────────────────────────
    out_writer = None
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(args.output, fourcc, fps_in, (width, height))
        logger.info("Writing output to: %s", args.output)

    # ── Processing loop ──────────────────────────────────────────
    frame_idx     = 0
    fps_timer     = time.time()
    fps_display   = 0.0
    frame_count_fps = 0

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_idx       += 1
            frame_count_fps += 1

            # BGR → RGB for detector
            frame_rgb = frame_bgr[:, :, ::-1]

            # ── Detection + tracking ─────────────────────────────
            result = detector.detect_from_array(frame_rgb, image_name=f"frame_{frame_idx:06d}")

            # ── Lane detection ───────────────────────────────────
            if lane_det:
                lane_result = lane_det.detect(frame_bgr)
                frame_bgr   = lane_det.draw_lanes(frame_bgr, lane_result)

            # ── Distance + collision warning ─────────────────────
            warnings   = {}
            active_ids = []
            for det in result.detections:
                if det.track_id is not None:
                    active_ids.append(det.track_id)
                    warn = advisor.assess(det.box, det.class_name, det.track_id)
                    warnings[det.track_id] = warn

            advisor.prune_stale_tracks(active_ids)

            # ── Annotate frame ───────────────────────────────────
            frame_bgr = draw_detections_v2(frame_bgr, result.detections, warnings, estimator)

            # ── FPS ──────────────────────────────────────────────
            elapsed = time.time() - fps_timer
            if elapsed >= 0.5:
                fps_display     = frame_count_fps / elapsed
                fps_timer       = time.time()
                frame_count_fps = 0

            num_warn   = sum(1 for w in warnings.values() if w.risk_level == "WARNING")
            num_danger = sum(1 for w in warnings.values() if w.risk_level == "DANGER")

            frame_bgr = draw_hud(frame_bgr, frame_idx, fps_display,
                                 len(result.detections), num_warn, num_danger)

            if out_writer:
                out_writer.write(frame_bgr)

            if args.show:
                cv2.imshow("Autonomous Driving Perception", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_idx % 100 == 0:
                logger.info("Frame %d | FPS %.1f | Objects %d",
                            frame_idx, fps_display, len(result.detections))

    finally:
        cap.release()
        if out_writer:
            out_writer.release()
        cv2.destroyAllWindows()
        logger.info("Done. Processed %d frames.", frame_idx)


def main():
    ap = argparse.ArgumentParser(
        description="🎬 Autonomous Driving Perception — Video Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video",  type=str, help="Path to input video file")
    src.add_argument("--webcam", type=int, metavar="ID",
                     help="Webcam device ID (e.g. 0)")

    ap.add_argument("--output",         type=str,  default=None,
                    help="Output video path (e.g. outputs/result.mp4)")
    ap.add_argument("--weights",        type=str,  default="yolov8m.pt",
                    help="YOLOv8 model weights (default: yolov8m.pt)")
    ap.add_argument("--confidence",     type=float,default=0.5)
    ap.add_argument("--iou",            type=float,default=0.45)
    ap.add_argument("--driving_mode",   action="store_true",
                    help="Filter for driving-relevant classes only")
    ap.add_argument("--lane_detection", action="store_true",
                    help="Enable lane detection overlay")
    ap.add_argument("--focal_length",   type=float,default=800.0,
                    help="Camera focal length in pixels (for distance estimation)")
    ap.add_argument("--show",           action="store_true",
                    help="Display live video window")
    ap.add_argument("--save_json",      action="store_true",
                    help="Save per-frame detection JSON (future)")

    args = ap.parse_args()
    process_video(args)


if __name__ == "__main__":
    main()
