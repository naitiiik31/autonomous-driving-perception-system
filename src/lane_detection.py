"""
Lane Detection Module
=====================
Classical computer-vision lane detection pipeline for autonomous driving.

Pipeline:
    Frame → Grayscale → Gaussian Blur → Canny → ROI Mask
          → Hough Lines → Segment Merge → Overlay

Detects left and right lane lines using:
  1. Canny edge detection
  2. Trapezoidal Region-of-Interest (ROI) masking
  3. Probabilistic Hough Transform
  4. Line segment averaging & extension to full lane lines

Usage:
    detector = LaneDetector()
    result   = detector.detect(frame_bgr)          # numpy BGR array
    overlay  = detector.draw_lanes(frame_bgr, result)
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class LaneResult:
    """
    Output of a single-frame lane detection run.

    Attributes:
        left_line:       (x1, y1, x2, y2) pixel coords of the left lane line.
        right_line:      (x1, y1, x2, y2) pixel coords of the right lane line.
        lane_width_px:   Estimated lane width in pixels at the bottom of ROI.
        departure_score: Float in [-1, 1]. Negative = drifting left, positive = right.
        valid:           Whether at least one lane line was detected.
    """
    left_line:       Optional[Tuple[int, int, int, int]] = None
    right_line:      Optional[Tuple[int, int, int, int]] = None
    lane_width_px:   float = 0.0
    departure_score: float = 0.0
    valid:           bool  = False


class LaneDetector:
    """
    Classical lane line detector for forward-facing camera video.

    Designed for highway / structured road environments. Works without
    any deep learning model — uses only OpenCV primitives.

    Attributes:
        roi_vertices: Trapezoidal ROI vertices as fraction of (H, W).
            Default covers the lower 45% of the frame in a trapezoid
            that mirrors a typical dashboard camera field of view.
    """

    def __init__(
        self,
        # Canny thresholds
        canny_low:    int   = 50,
        canny_high:   int   = 150,
        # Gaussian blur kernel
        blur_ksize:   int   = 5,
        # Hough transform parameters
        hough_rho:    float = 1.0,
        hough_theta:  float = np.pi / 180,
        hough_thresh: int   = 30,
        hough_min_len:int   = 40,
        hough_max_gap:int   = 150,
        # ROI as fractions of (height, width)
        roi_top_left_frac:     Tuple[float, float] = (0.60, 0.42),
        roi_top_right_frac:    Tuple[float, float] = (0.60, 0.58),
        roi_bottom_left_frac:  Tuple[float, float] = (0.95, 0.10),
        roi_bottom_right_frac: Tuple[float, float] = (0.95, 0.90),
    ):
        self.canny_low    = canny_low
        self.canny_high   = canny_high
        self.blur_ksize   = blur_ksize | 1  # must be odd
        self.hough_rho    = hough_rho
        self.hough_theta  = hough_theta
        self.hough_thresh = hough_thresh
        self.hough_min_len = hough_min_len
        self.hough_max_gap = hough_max_gap
        self.roi_fracs = (
            roi_top_left_frac,
            roi_top_right_frac,
            roi_bottom_left_frac,
            roi_bottom_right_frac,
        )

    # ── Public API ───────────────────────────────────────────────

    def detect(self, frame_bgr: np.ndarray) -> LaneResult:
        """
        Detect lane lines in a single BGR video frame.

        Args:
            frame_bgr: OpenCV BGR numpy array (H, W, 3).

        Returns:
            LaneResult with detected lane lines and metadata.
        """
        h, w = frame_bgr.shape[:2]

        # Step 1 — Preprocess
        gray    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)
        edges   = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # Step 2 — ROI mask
        roi_mask = self._build_roi_mask(h, w)
        masked   = cv2.bitwise_and(edges, edges, mask=roi_mask)

        # Step 3 — Hough lines
        raw_lines = cv2.HoughLinesP(
            masked,
            rho=self.hough_rho,
            theta=self.hough_theta,
            threshold=self.hough_thresh,
            minLineLength=self.hough_min_len,
            maxLineGap=self.hough_max_gap,
        )

        if raw_lines is None:
            return LaneResult(valid=False)

        # Step 4 — Separate + average left / right lines
        y_bottom = int(h * self.roi_fracs[2][0])   # bottom of ROI
        y_top    = int(h * self.roi_fracs[0][0])   # top of ROI

        left_line, right_line = self._average_lines(raw_lines, h, y_bottom, y_top)

        # Step 5 — Compute departure score
        departure = self._compute_departure(left_line, right_line, w)

        # Step 6 — Estimate lane width in pixels
        lane_width_px = 0.0
        if left_line and right_line:
            lane_width_px = float(right_line[0] - left_line[0])

        result = LaneResult(
            left_line       = left_line,
            right_line      = right_line,
            lane_width_px   = lane_width_px,
            departure_score = departure,
            valid           = (left_line is not None or right_line is not None),
        )

        logger.debug(
            "Lane detection | left=%s right=%s width=%.0fpx departure=%.2f",
            left_line, right_line, lane_width_px, departure,
        )

        return result

    def draw_lanes(
        self,
        frame_bgr:  np.ndarray,
        result:     LaneResult,
        alpha:      float = 0.3,
    ) -> np.ndarray:
        """
        Draw detected lane lines and filled lane region on a BGR frame.

        Args:
            frame_bgr: Original OpenCV BGR frame.
            result:    Output of detect().
            alpha:     Transparency of the filled lane polygon.

        Returns:
            Annotated BGR numpy array (copy of input).
        """
        overlay = frame_bgr.copy()
        h, w    = frame_bgr.shape[:2]

        # Draw filled lane polygon (green)
        if result.left_line and result.right_line:
            ll, rl = result.left_line, result.right_line
            pts = np.array([
                [ll[0], ll[1]], [ll[2], ll[3]],
                [rl[2], rl[3]], [rl[0], rl[1]],
            ], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 200, 80))

        blended = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)

        # Draw individual lane lines
        if result.left_line:
            x1, y1, x2, y2 = result.left_line
            cv2.line(blended, (x1, y1), (x2, y2), (0, 255, 0), 4)

        if result.right_line:
            x1, y1, x2, y2 = result.right_line
            cv2.line(blended, (x1, y1), (x2, y2), (0, 255, 0), 4)

        # Departure warning badge
        if abs(result.departure_score) > 0.15:
            direction = "LEFT" if result.departure_score < 0 else "RIGHT"
            text      = f"⚠ LANE DEPARTURE — {direction}"
            cv2.putText(
                blended, text, (w // 2 - 200, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
            )

        return blended

    # ── Private helpers ──────────────────────────────────────────

    def _build_roi_mask(self, h: int, w: int) -> np.ndarray:
        """Build a binary mask covering the trapezoidal road ROI."""
        tl, tr, bl, br = self.roi_fracs
        vertices = np.array([[
            (int(bl[1] * w), int(bl[0] * h)),   # bottom-left
            (int(tl[1] * w), int(tl[0] * h)),   # top-left
            (int(tr[1] * w), int(tr[0] * h)),   # top-right
            (int(br[1] * w), int(br[0] * h)),   # bottom-right
        ]], dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, vertices, 255)
        return mask

    def _average_lines(
        self,
        lines:    np.ndarray,
        h:        int,
        y_bottom: int,
        y_top:    int,
    ) -> Tuple[Optional[Tuple], Optional[Tuple]]:
        """
        Separate Hough segments into left / right by slope sign,
        then fit a single averaged line per side.
        """
        left_pts:  List[Tuple[float, float]] = []
        right_pts: List[Tuple[float, float]] = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            # Filter near-horizontal lines
            if abs(slope) < 0.4:
                continue
            if slope < 0:  # left lane (negative slope in image coords)
                left_pts.extend([(x1, y1), (x2, y2)])
            else:          # right lane
                right_pts.extend([(x1, y1), (x2, y2)])

        left_line  = self._fit_line(left_pts,  y_bottom, y_top)
        right_line = self._fit_line(right_pts, y_bottom, y_top)

        return left_line, right_line

    @staticmethod
    def _fit_line(
        points:   List[Tuple[float, float]],
        y_bottom: int,
        y_top:    int,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Fit a polynomial to a set of (x, y) points and extrapolate."""
        if len(points) < 2:
            return None
        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])
        try:
            coeffs = np.polyfit(ys, xs, 1)  # x = m*y + b
        except np.linalg.LinAlgError:
            return None
        x_bottom = int(np.polyval(coeffs, y_bottom))
        x_top    = int(np.polyval(coeffs, y_top))
        return (x_bottom, y_bottom, x_top, y_top)

    @staticmethod
    def _compute_departure(
        left_line:  Optional[Tuple],
        right_line: Optional[Tuple],
        frame_width: int,
    ) -> float:
        """
        Estimate lane departure score in [-1, 1].
        0.0 = centred, negative = left drift, positive = right drift.
        """
        cx = frame_width / 2.0
        if left_line and right_line:
            lane_cx = (left_line[0] + right_line[0]) / 2.0
        elif left_line:
            lane_cx = left_line[0] + frame_width * 0.25
        elif right_line:
            lane_cx = right_line[0] - frame_width * 0.25
        else:
            return 0.0
        return float(np.clip((lane_cx - cx) / (frame_width / 2.0), -1.0, 1.0))
