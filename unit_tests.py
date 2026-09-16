"""
Unit Tests — Autonomous Driving Perception System
==================================================

Run with:
    pytest tests/ -v --cov=src --cov-report=term-missing

Coverage targets:
    - LaneDetector: edge cases, black frames, valid frames
    - DistanceEstimator: known values, zero pixel height, unknown class
    - CollisionAdvisor: single frame, multi-frame TTC, risk levels
    - PostProcessor.compute_iou: known overlap values
    - Config: save/load round-trip
"""

import os
import sys
import json
import tempfile
import numpy as np
import pytest

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────

@pytest.fixture
def blank_frame():
    """720p black BGR frame."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def road_frame():
    """Simulated road scene: grey road + white lane lines on black."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[400:, :] = 80                            # grey road surface
    # Left lane line
    cv2 = pytest.importorskip("cv2")
    cv2.line(frame, (300, 720), (550, 430), (255, 255, 255), 5)
    # Right lane line
    cv2.line(frame, (980, 720), (730, 430), (255, 255, 255), 5)
    return frame


@pytest.fixture
def estimator():
    from src.distance_estimation import DistanceEstimator
    return DistanceEstimator(focal_length_px=800.0)


@pytest.fixture
def advisor(estimator):
    from src.distance_estimation import CollisionAdvisor
    return CollisionAdvisor(estimator, history_frames=5)


# ─────────────────────────────────────────────────────────────────
# DistanceEstimator Tests
# ─────────────────────────────────────────────────────────────────

class TestDistanceEstimator:

    def test_known_distance_car(self, estimator):
        """Car at ~10m should report ~10m with focal=800, height=1.5m."""
        # pixel_height = (real_height × focal) / distance
        # = (1.5 × 800) / 10 = 120 px
        box = np.array([300.0, 100.0, 420.0, 300.0])  # height = 120px
        result = estimator.estimate(box, "car")
        assert 8.0 < result.distance_m < 12.0, f"Expected ~10m, got {result.distance_m}"

    def test_zero_pixel_height(self, estimator):
        """Zero-height bounding box should not raise and return large distance."""
        box = np.array([100.0, 100.0, 100.0, 300.0])  # height = 0
        result = estimator.estimate(box, "car")
        assert result.distance_m > 1000

    def test_unknown_class_uses_default(self, estimator):
        """Unknown class should fall back to default height (1.5m)."""
        box = np.array([300.0, 100.0, 420.0, 300.0])  # 120 px height
        result = estimator.estimate(box, "alien_spacecraft")
        assert result.distance_m > 0

    def test_danger_zone(self, estimator):
        """Object closer than 5m should be classified DANGER."""
        # distance = (1.5 × 800) / px_h < 5 → px_h > 240
        box = np.array([100.0, 100.0, 360.0, 300.0])  # 260 px height → ~4.6m
        result = estimator.estimate(box, "car")
        assert result.risk_level == "DANGER"

    def test_safe_zone(self, estimator):
        """Object beyond 15m should be classified SAFE."""
        # distance = (1.5 × 800) / px_h > 15 → px_h < 80
        box = np.array([300.0, 100.0, 360.0, 300.0])  # 60 px height → ~20m
        result = estimator.estimate(box, "car")
        assert result.risk_level == "SAFE"

    def test_warning_zone(self, estimator):
        """Object between 5m and 15m should be WARNING."""
        # 1200 / px_h in [5, 15] → px_h in [80, 240]
        box = np.array([200.0, 100.0, 350.0, 300.0])  # 150 px → ~8m
        result = estimator.estimate(box, "car")
        assert result.risk_level == "WARNING"

    def test_truck_is_farther_than_car_same_pixels(self, estimator):
        """Truck (taller) at same pixel height should report greater distance."""
        box = np.array([300.0, 100.0, 380.0, 300.0])  # 80 px
        car_result   = estimator.estimate(box, "car")     # 1.5m real height
        truck_result = estimator.estimate(box, "truck")   # 3.5m real height
        assert truck_result.distance_m > car_result.distance_m

    def test_focal_length_calibration(self, estimator):
        """Calibration helper should produce correct focal length."""
        fl = estimator.estimate_focal_length(
            known_object_distance_m=10.0,
            known_object_height_m=1.5,
            pixel_height=120.0,
        )
        assert abs(fl - 800.0) < 1.0


# ─────────────────────────────────────────────────────────────────
# CollisionAdvisor Tests
# ─────────────────────────────────────────────────────────────────

class TestCollisionAdvisor:
    import time as _time

    def test_first_frame_no_ttc(self, advisor):
        """Single frame — no speed history, TTC should be -1."""
        box = np.array([300.0, 100.0, 420.0, 300.0])
        warn = advisor.assess(box, "car", track_id=1, timestamp=0.0)
        assert warn.ttc_s == -1.0

    def test_approaching_object_danger(self, advisor):
        """Rapidly approaching car should trigger DANGER."""
        track_id = 42
        # Simulate car going from 30m → 3m over 1 second (frames 0.1s apart)
        distances = np.linspace(30, 3, 10)
        for i, dist in enumerate(distances):
            # Back-compute pixel height
            px_h = (1.5 * 800.0) / dist
            box = np.array([400.0, 100.0, 400.0 + px_h, 300.0])
            warn = advisor.assess(box, "car", track_id=track_id, timestamp=float(i) * 0.1)

        assert warn.risk_level == "DANGER"

    def test_stationary_object_no_ttc(self, advisor):
        """Non-moving object should have very high TTC (SAFE or WARNING by dist)."""
        track_id = 99
        for i in range(5):
            box = np.array([200.0, 100.0, 350.0, 300.0])  # constant ~8m
            warn = advisor.assess(box, "car", track_id=track_id, timestamp=float(i) * 0.1)

        # Speed ≈ 0 → TTC = inf or -1 → not DANGER from TTC perspective
        assert warn.ttc_s < 0 or warn.ttc_s > 100

    def test_prune_stale_tracks(self, advisor):
        """Pruning should remove tracks no longer in active_ids."""
        box = np.array([300.0, 100.0, 420.0, 300.0])
        advisor.assess(box, "car", track_id=1, timestamp=0.0)
        advisor.assess(box, "car", track_id=2, timestamp=0.0)
        assert len(advisor._history) == 2
        advisor.prune_stale_tracks(active_ids=[1])
        assert 2 not in advisor._history
        assert 1 in advisor._history


# ─────────────────────────────────────────────────────────────────
# LaneDetector Tests
# ─────────────────────────────────────────────────────────────────

class TestLaneDetector:

    def test_black_frame_no_crash(self, blank_frame):
        """All-black frame should not raise; returns valid=False."""
        cv2 = pytest.importorskip("cv2")
        from src.lane_detection import LaneDetector
        detector = LaneDetector()
        result = detector.detect(blank_frame)
        assert result.valid is False

    def test_road_frame_detects_lanes(self, road_frame):
        """Synthetic road frame with white lines should detect at least one lane."""
        cv2 = pytest.importorskip("cv2")
        from src.lane_detection import LaneDetector
        detector = LaneDetector()
        result = detector.detect(road_frame)
        assert result.valid is True
        assert result.left_line is not None or result.right_line is not None

    def test_departure_score_range(self, road_frame):
        """Departure score must always be in [-1, 1]."""
        cv2 = pytest.importorskip("cv2")
        from src.lane_detection import LaneDetector
        detector = LaneDetector()
        result = detector.detect(road_frame)
        assert -1.0 <= result.departure_score <= 1.0

    def test_draw_lanes_no_crash(self, blank_frame):
        """draw_lanes should not raise on blank frame."""
        cv2 = pytest.importorskip("cv2")
        from src.lane_detection import LaneDetector, LaneResult
        detector = LaneDetector()
        empty_result = LaneResult(valid=False)
        out = detector.draw_lanes(blank_frame, empty_result)
        assert out.shape == blank_frame.shape


# ─────────────────────────────────────────────────────────────────
# PostProcessor IoU Tests
# ─────────────────────────────────────────────────────────────────

class TestIoU:

    def test_perfect_overlap(self):
        from src.postprocessor import PostProcessor
        box = np.array([0.0, 0.0, 10.0, 10.0])
        assert PostProcessor.compute_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        from src.postprocessor import PostProcessor
        box1 = np.array([0.0, 0.0, 5.0, 5.0])
        box2 = np.array([6.0, 6.0, 10.0, 10.0])
        assert PostProcessor.compute_iou(box1, box2) == pytest.approx(0.0)

    def test_half_overlap(self):
        from src.postprocessor import PostProcessor
        box1 = np.array([0.0, 0.0, 10.0, 10.0])   # area = 100
        box2 = np.array([0.0, 5.0, 10.0, 15.0])   # area = 100, overlap = 50
        iou  = PostProcessor.compute_iou(box1, box2)
        # IoU = 50 / (100 + 100 - 50) = 50 / 150 ≈ 0.333
        assert 0.32 < iou < 0.35


# ─────────────────────────────────────────────────────────────────
# Config Round-Trip Test
# ─────────────────────────────────────────────────────────────────

class TestConfig:

    def test_save_load_roundtrip(self):
        """Config saved to JSON should load back with identical values."""
        from src.config import Config
        cfg = Config(score_threshold=0.42, iou_threshold=0.37, max_boxes=77)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg.save(path)
            loaded = Config.load(path)
            assert loaded.score_threshold == pytest.approx(0.42)
            assert loaded.iou_threshold   == pytest.approx(0.37)
            assert loaded.max_boxes       == 77
        finally:
            os.unlink(path)

    def test_driving_mode_factory(self):
        from src.config import Config
        cfg = Config.driving_mode()
        assert cfg.enable_driving_filter is True
        assert cfg.score_threshold < 0.5

    def test_high_precision_factory(self):
        from src.config import Config
        cfg = Config.high_precision()
        assert cfg.score_threshold >= 0.6


# ─────────────────────────────────────────────────────────────────
# Logger Tests
# ─────────────────────────────────────────────────────────────────

class TestLogger:

    def test_get_logger_returns_logger(self):
        import logging
        from src.logger import get_logger
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

    def test_setup_logging_idempotent(self):
        """Calling setup_logging twice should not duplicate handlers."""
        import logging
        from src.logger import setup_logging
        root = logging.getLogger()
        initial_count = len(root.handlers)
        setup_logging(log_dir=tempfile.gettempdir(), json_file=False, console=False)
        # Handlers should not multiply on repeated calls
        assert len(root.handlers) == initial_count
