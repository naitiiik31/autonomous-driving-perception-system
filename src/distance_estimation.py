"""
Distance Estimation & Collision Warning
========================================
Estimates object distance from a monocular camera using bounding-box
geometry and triggers collision warnings via Time-to-Collision (TTC).

Theory — Monocular Distance Estimation
---------------------------------------
For a calibrated camera:

    distance = (real_height × focal_length) / pixel_height 

Where:
    real_height   = known physical height of the object class (metres)
    focal_length  = camera focal length in pixels
    pixel_height  = bounding-box height in pixels

Focal length estimation from known FOV:
    focal_length = (image_height / 2) / tan(vertical_fov / 2)

Time-to-Collision (TTC)
-----------------------
    TTC = distance / approach_speed

Risk zones:
    DANGER:  distance < 5 m  OR  TTC < 2 s
    WARNING: distance < 15 m OR  TTC < 5 s
    SAFE:    otherwise

Usage:
    estimator = DistanceEstimator(focal_length_px=800)
    dist = estimator.estimate(box_pixel_height=120, class_name="car")

    advisor = CollisionAdvisor(estimator)
    warning = advisor.assess(box, class_name, track_id, frame_time)
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from collections import deque

from .logger import get_logger

logger = get_logger(__name__)


# ── Known object heights (metres) ────────────────────────────────
KNOWN_HEIGHTS: Dict[str, float] = {
    "car":           1.5,
    "truck":         3.5,
    "bus":           3.2,
    "motorbike":     1.1,
    "bicycle":       1.0,
    "person":        1.7,
    "traffic light": 0.6,
    "stop sign":     0.75,
    "train":         4.0,
}

DEFAULT_HEIGHT = 1.5   # fallback for unknown classes


@dataclass
class DistanceResult:
    """Per-object distance measurement."""
    class_name:    str
    distance_m:    float          # metres
    pixel_height:  float          # bounding-box height in pixels
    confidence:    float = 1.0
    risk_level:    str   = "SAFE" # "SAFE" | "WARNING" | "DANGER"


@dataclass
class CollisionWarning:
    """Collision risk assessment for a single tracked object."""
    track_id:      int
    class_name:    str
    distance_m:    float
    ttc_s:         float          # Time-to-Collision in seconds (-1 = no estimate)
    risk_level:    str            # "SAFE" | "WARNING" | "DANGER"
    approach_speed_mps: float     # metres per second (negative = approaching)


class DistanceEstimator:
    """
    Monocular distance estimator using bounding-box apparent size.

    Works without depth sensors — uses known object dimensions and the
    pinhole camera model.

    Args:
        focal_length_px: Camera focal length in pixels. Estimate with:
            focal_length = (image_height / 2) / tan(vertical_fov_rad / 2)
            e.g. 1080p, 60° vFOV → focal_length ≈ 935 px
        known_heights:   Override or extend the built-in object height table.
    """

    def __init__(
        self,
        focal_length_px: float = 800.0,
        known_heights:   Optional[Dict[str, float]] = None,
    ):
        self.focal_length = focal_length_px
        self.heights = {**KNOWN_HEIGHTS, **(known_heights or {})}

    def estimate(
        self,
        box:        np.ndarray,   # [y_min, x_min, y_max, x_max] in pixels
        class_name: str,
    ) -> DistanceResult:
        """
        Estimate distance to a detected object.

        Args:
            box:        Bounding box [y_min, x_min, y_max, x_max].
            class_name: Object class name (used for height lookup).

        Returns:
            DistanceResult with distance_m and risk_level.
        """
        pixel_height = abs(float(box[2]) - float(box[0]))
        if pixel_height < 1:
            return DistanceResult(class_name=class_name, distance_m=9999.0,
                                  pixel_height=0, risk_level="SAFE")

        real_height = self.heights.get(class_name, DEFAULT_HEIGHT)
        distance_m  = (real_height * self.focal_length) / pixel_height

        risk = self._classify_risk(distance_m)

        logger.debug(
            "Distance | class=%s h_px=%.0f d=%.1fm risk=%s",
            class_name, pixel_height, distance_m, risk,
        )

        return DistanceResult(
            class_name   = class_name,
            distance_m   = round(distance_m, 2),
            pixel_height = pixel_height,
            risk_level   = risk,
        )

    @staticmethod
    def _classify_risk(distance_m: float) -> str:
        if distance_m < 5.0:
            return "DANGER"
        if distance_m < 15.0:
            return "WARNING"
        return "SAFE"

    def estimate_focal_length(
        self,
        known_object_distance_m: float,
        known_object_height_m:   float,
        pixel_height:            float,
    ) -> float:
        """
        Calibrate focal length from a reference measurement.

        Place an object of known height at a known distance and
        measure its bounding-box pixel height. Returns focal length.

        Args:
            known_object_distance_m: Actual distance in metres.
            known_object_height_m:   Actual height in metres.
            pixel_height:            Observed bounding-box height.

        Returns:
            Estimated focal length in pixels.
        """
        fl = (pixel_height * known_object_distance_m) / known_object_height_m
        logger.info("Calibrated focal length: %.1f px", fl)
        return fl


class CollisionAdvisor:
    """
    Time-to-Collision collision warning system.

    Maintains a short history of distances per tracked object ID
    to estimate approach speed and compute TTC.

    Args:
        estimator:      DistanceEstimator instance.
        history_frames: Number of past frames to use for speed estimation.
    """

    def __init__(
        self,
        estimator:      DistanceEstimator,
        history_frames: int = 10,
    ):
        self.estimator      = estimator
        self.history_frames = history_frames
        # track_id → deque of (timestamp, distance_m)
        self._history: Dict[int, deque] = {}

    def assess(
        self,
        box:        np.ndarray,
        class_name: str,
        track_id:   int,
        timestamp:  Optional[float] = None,
    ) -> CollisionWarning:
        """
        Estimate collision risk for one tracked object.

        Args:
            box:        Detection bounding box [y_min, x_min, y_max, x_max].
            class_name: Object class string.
            track_id:   Unique object tracking ID (from ByteTrack / SORT).
            timestamp:  Current time (seconds). Defaults to time.time().

        Returns:
            CollisionWarning with TTC, risk level and approach speed.
        """
        ts = timestamp or time.time()

        dist_result = self.estimator.estimate(box, class_name)
        distance    = dist_result.distance_m

        # Maintain per-track history
        if track_id not in self._history:
            self._history[track_id] = deque(maxlen=self.history_frames)
        self._history[track_id].append((ts, distance))

        # Estimate approach speed from history
        approach_speed, ttc = self._compute_ttc(track_id, distance)

        # Final risk: combine distance-based and TTC-based risk
        risk = self._combined_risk(distance, ttc)

        if risk in ("DANGER", "WARNING"):
            logger.warning(
                "Collision risk | track=%d class=%s dist=%.1fm ttc=%.1fs risk=%s",
                track_id, class_name, distance, ttc, risk,
            )

        return CollisionWarning(
            track_id          = track_id,
            class_name        = class_name,
            distance_m        = distance,
            ttc_s             = ttc,
            risk_level        = risk,
            approach_speed_mps= approach_speed,
        )

    def _compute_ttc(self, track_id: int, current_dist: float) -> Tuple[float, float]:
        """Estimate approach speed (m/s) and TTC (s) from history."""
        hist = list(self._history[track_id])
        if len(hist) < 2:
            return 0.0, -1.0

        # Linear regression over recent distances
        times = np.array([h[0] for h in hist])
        dists = np.array([h[1] for h in hist])
        times = times - times[0]  # normalise to start=0

        if times[-1] < 1e-6:
            return 0.0, -1.0

        coeffs        = np.polyfit(times, dists, 1)
        speed_mps     = coeffs[0]          # negative → approaching
        approach_speed = float(-speed_mps) # positive = getting closer

        if approach_speed > 0.1:
            ttc = float(current_dist / approach_speed)
        else:
            ttc = float("inf")

        return approach_speed, ttc

    @staticmethod
    def _combined_risk(distance_m: float, ttc_s: float) -> str:
        """Combine distance and TTC into a single risk level."""
        if distance_m < 5.0 or (0 < ttc_s < 2.0):
            return "DANGER"
        if distance_m < 15.0 or (0 < ttc_s < 5.0):
            return "WARNING"
        return "SAFE"

    def prune_stale_tracks(self, active_ids: List[int]) -> None:
        """Remove history for track IDs that are no longer active."""
        stale = [tid for tid in self._history if tid not in active_ids]
        for tid in stale:
            del self._history[tid]
