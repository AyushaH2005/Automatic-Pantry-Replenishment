"""
Pantry item detector.
Uses YOLO-style inference (swappable backbone) + visual fill estimation.
Designed to run on edge hardware (Raspberry Pi 5 / Jetson Nano).
"""
from __future__ import annotations
import time, logging
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class BBox:
    x1: float; y1: float; x2: float; y2: float

    @property
    def area(self) -> float:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def iou(self, other: "BBox") -> float:
        ix1 = max(self.x1, other.x1); iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2); iy2 = min(self.y2, other.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def centre(self) -> tuple:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass
class Detection:
    bbox:         BBox
    confidence:   float
    class_id:     int
    class_name:   str
    track_id:     Optional[int]      = None
    fill_ratio:   float              = 1.0
    pkg_type:     str                = "auto"
    embedding:    Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class InventoryItem:
    track_id:        int
    name:            str
    sku:             Optional[str] = None
    fill_ratio:      float         = 1.0
    fill_history:    list          = field(default_factory=list)
    last_seen_ts:    float         = field(default_factory=time.time)
    reorder_pending: bool          = False
    REORDER_THRESHOLD: float       = 0.25

    def update(self, det: Detection, ts: float) -> None:
        self.fill_ratio   = det.fill_ratio
        self.last_seen_ts = ts
        self.fill_history.append((ts, det.fill_ratio))
        if len(self.fill_history) > 500:
            self.fill_history = self.fill_history[-500:]

    @property
    def is_low_stock(self) -> bool:
        return self.fill_ratio < self.REORDER_THRESHOLD

    def consumption_rate(self) -> Optional[float]:
        """Returns fill units lost per second, or None if insufficient data."""
        if len(self.fill_history) < 5:
            return None
        ts0, f0 = self.fill_history[0]
        ts1, f1 = self.fill_history[-1]
        dt = ts1 - ts0
        if dt < 60:
            return None
        return max(0.0, f0 - f1) / dt

    def estimated_time_to_empty(self) -> Optional[float]:
        rate = self.consumption_rate()
        if rate is None or rate == 0:
            return None
        return self.fill_ratio / rate


# ── Fill ratio estimation ─────────────────────────────────────────────────────

class FillEstimator:
    """
    Visual fill estimation per packaging type.

    Transparent (glass/plastic): scan vertical brightness gradient.
    Box/carton: measure top-edge compression via Canny edges.
    Packet/pouch: saturation-area proxy.
    """

    def estimate(self, crop: np.ndarray, pkg_type: str = "auto") -> float:
        if crop is None or crop.size == 0:
            return 1.0
        if pkg_type == "transparent" or (
                pkg_type == "auto" and self._is_transparent(crop)):
            return self._gradient_fill(crop)
        if pkg_type in ("box", "carton"):
            return self._edge_compression_fill(crop)
        return self._saturation_fill(crop)

    @staticmethod
    def _is_transparent(crop: np.ndarray) -> bool:
        try:
            import cv2
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            return float(hsv[:, :, 2].std()) > 40
        except Exception:
            return False

    @staticmethod
    def _gradient_fill(crop: np.ndarray) -> float:
        """Transparent containers: liquid fill line detection."""
        try:
            import cv2
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            col_mean = gray.mean(axis=1).astype(float)
            k = max(3, len(col_mean) // 10)
            smoothed = np.convolve(col_mean, np.ones(k) / k, mode='same')
            diff = np.diff(smoothed)
            fill_idx = int(np.argmax(np.abs(diff)))
            return float(np.clip(1.0 - fill_idx / len(col_mean), 0.0, 1.0))
        except Exception:
            return 0.5

    @staticmethod
    def _edge_compression_fill(crop: np.ndarray) -> float:
        """Boxes/cartons: top-edge density reveals collapse."""
        try:
            import cv2
            gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            h     = edges.shape[0]
            top_density  = edges[:h // 4].mean()
            full_density = edges.mean()
            if full_density < 1:
                return 1.0
            ratio = 1.0 - min(1.0, top_density / full_density - 0.5)
            return float(np.clip(ratio, 0.0, 1.0))
        except Exception:
            return 0.5

    @staticmethod
    def _saturation_fill(crop: np.ndarray) -> float:
        """Opaque packs: saturated label area as proxy."""
        try:
            import cv2
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            ratio = (hsv[:, :, 1] > 50).mean()
            return float(np.clip(ratio * 1.2, 0.0, 1.0))
        except Exception:
            return 0.5


# ── Burst deduplication ───────────────────────────────────────────────────────

class BurstAggregator:
    """
    Merges 5 burst frames into one reliable detection set.
    Keeps detections that appear in >= vote_thresh frames.
    """

    def __init__(self, iou_thresh: float = 0.45, vote_thresh: int = 2):
        self.iou_thresh  = iou_thresh
        self.vote_thresh = vote_thresh

    def aggregate(self,
                  burst: list[list[Detection]]) -> list[Detection]:
        candidates: list[list] = []   # [Detection, vote_count]

        for frame_dets in burst:
            for det in frame_dets:
                matched = False
                for entry in candidates:
                    cand, votes = entry[0], entry[1]
                    if (cand.class_id == det.class_id
                            and cand.bbox.iou(det.bbox) > self.iou_thresh):
                        b, db = cand.bbox, det.bbox
                        cand.bbox = BBox(
                            (b.x1 + db.x1) / 2, (b.y1 + db.y1) / 2,
                            (b.x2 + db.x2) / 2, (b.y2 + db.y2) / 2,
                        )
                        cand.fill_ratio = (cand.fill_ratio + det.fill_ratio) / 2
                        cand.confidence = max(cand.confidence, det.confidence)
                        entry[1]    = votes + 1
                        matched     = True
                        break
                if not matched:
                    candidates.append([det, 1])

        return [entry[0] for entry in candidates
                if entry[1] >= self.vote_thresh]


# ── Main detector ─────────────────────────────────────────────────────────────

class PantryDetector:
    """
    Inference wrapper with pluggable backends.

    Backends:
      'yolov8n'  – YOLOv8 nano, ~50 ms on RPi 5 (fastest edge)
      'yolov8s'  – YOLOv8 small, ~130 ms (better accuracy)
      'grounded' – GroundedDINO (open-vocab, Jetson/cloud)
      'mock'     – synthetic data for offline testing
    """

    def __init__(self, backend: str = "mock", config: dict = None):
        self.backend        = backend
        self.config         = config or {}
        self.fill_estimator = FillEstimator()
        self.burst_agg      = BurstAggregator()
        self._model         = None
        self._load_model()

    def _load_model(self) -> None:
        if self.backend == "mock":
            logger.info("PantryDetector: mock backend")
            return
        try:
            if self.backend.startswith("yolov8"):
                from ultralytics import YOLO
                self._model = YOLO(f"{self.backend}.pt")
                logger.info(f"Loaded {self.backend}")
            elif self.backend == "grounded":
                from groundingdino.util.inference import load_model
                self._model = load_model(
                    self.config["gdino_config"],
                    self.config["gdino_checkpoint"],
                )
                logger.info("Loaded GroundedDINO")
        except ImportError as exc:
            logger.warning(f"Backend unavailable ({exc}), using mock")
            self.backend = "mock"

    def detect_frame(self, frame: np.ndarray,
                     ts: float = None) -> list[Detection]:
        ts = ts or time.time()
        if self.backend == "mock":
            return self._mock_detections(ts)
        if self.backend.startswith("yolov8"):
            return self._yolo_detect(frame, ts)
        return []

    def detect_burst(self, frames: list[np.ndarray],
                     ts: float = None) -> list[Detection]:
        ts = ts or time.time()
        per_frame = [self.detect_frame(f, ts) for f in frames]
        return self.burst_agg.aggregate(per_frame)

    # ── private ──────────────────────────────────────────────────────────────

    def _yolo_detect(self, frame: np.ndarray, ts: float) -> list[Detection]:
        results = self._model.predict(
            frame,
            conf=self.config.get("conf_threshold", 0.35),
            iou=self.config.get("nms_iou", 0.45),
            verbose=False,
        )
        dets = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cid  = int(box.cls[0])
                crop = frame[int(y1):int(y2), int(x1):int(x2)]
                fill = self.fill_estimator.estimate(crop)
                dets.append(Detection(
                    bbox=BBox(x1, y1, x2, y2),
                    confidence=float(box.conf[0]),
                    class_id=cid,
                    class_name=self._model.names[cid],
                    fill_ratio=fill,
                ))
        return dets

    @staticmethod
    def _mock_detections(ts: float) -> list[Detection]:
        import random
        rng = random.Random(int(ts) % 1000)
        items = [
            ("olive_oil",    0, BBox(50,  80,  180, 380)),
            ("coffee_jar",   1, BBox(200, 100, 330, 360)),
            ("cereal_box",   2, BBox(350, 50,  510, 400)),
            ("pasta_packet", 3, BBox(530, 120, 650, 350)),
        ]
        dets = []
        for name, cid, bbox in items:
            if rng.random() > 0.05:
                base = max(0.05, 1.0 - (ts % 3600) / 3600)
                fill = float(np.clip(base + rng.gauss(0, 0.03), 0, 1))
                dets.append(Detection(
                    bbox=bbox,
                    confidence=rng.uniform(0.65, 0.95),
                    class_id=cid,
                    class_name=name,
                    fill_ratio=fill,
                ))
        return dets