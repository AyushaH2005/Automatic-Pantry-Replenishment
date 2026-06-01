"""
Lightweight multi-object tracker for pantry items.

Uses centroid IoU + Hungarian assignment.
No heavy deps (scipy only for linear_sum_assignment).
On resource-constrained hardware, falls back to greedy matching.
"""
from __future__ import annotations
import time, logging
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from .detector import BBox, Detection, InventoryItem

logger = logging.getLogger(__name__)


@dataclass
class Track:
    track_id:     int
    det:          Detection
    age:          int   = 0
    hits:         int   = 1
    time_since_update: int = 0
    item:         Optional[InventoryItem] = None

    def predict(self) -> None:
        """Age the track (simple constant-position model)."""
        self.time_since_update += 1
        self.age               += 1

    def update(self, det: Detection, ts: float) -> None:
        self.det               = det
        self.hits             += 1
        self.time_since_update = 0
        if self.item is not None:
            self.item.update(det, ts)


class PantryTracker:
    """
    Track pantry items across frames with stable IDs.

    Parameters
    ----------
    max_age       : frames a track survives without a matching detection
    min_hits      : frames before a track is considered confirmed
    iou_threshold : minimum IoU to accept a match
    """

    def __init__(self,
                 max_age:       int   = 30,
                 min_hits:      int   = 3,
                 iou_threshold: float = 0.35):
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self._tracks:  list[Track] = []
        self._next_id: int         = 0

    # ── public API ────────────────────────────────────────────────────────────

    def update(self, detections: list[Detection],
               ts: float = None) -> list[Track]:
        """
        Match detections to existing tracks.
        Returns list of *confirmed* tracks (hits >= min_hits).
        """
        ts = ts or time.time()

        # Age all tracks
        for t in self._tracks:
            t.predict()

        if not detections:
            self._remove_dead()
            return self.confirmed_tracks

        if not self._tracks:
            for det in detections:
                self._spawn(det, ts)
            return self.confirmed_tracks

        # Build IoU matrix: rows=tracks, cols=detections
        iou_mat = self._iou_matrix(self._tracks, detections)

        # Match
        matched, unmatched_trks, unmatched_dets = \
            self._match(iou_mat, len(self._tracks), len(detections))

        for trk_idx, det_idx in matched:
            det = detections[det_idx]
            det.track_id = self._tracks[trk_idx].track_id
            self._tracks[trk_idx].update(det, ts)

        for det_idx in unmatched_dets:
            self._spawn(detections[det_idx], ts)

        self._remove_dead()
        return self.confirmed_tracks

    @property
    def confirmed_tracks(self) -> list[Track]:
        return [t for t in self._tracks
                if t.hits >= self.min_hits
                and t.time_since_update == 0]

    @property
    def all_items(self) -> dict[int, InventoryItem]:
        return {t.track_id: t.item
                for t in self._tracks
                if t.item is not None}

    # ── private ───────────────────────────────────────────────────────────────

    def _spawn(self, det: Detection, ts: float) -> None:
        tid  = self._next_id
        self._next_id += 1
        det.track_id  = tid
        item = InventoryItem(
            track_id = tid,
            name     = det.class_name,
        )
        item.update(det, ts)
        self._tracks.append(Track(track_id=tid, det=det, item=item))

    def _remove_dead(self) -> None:
        self._tracks = [t for t in self._tracks
                        if t.time_since_update <= self.max_age]

    @staticmethod
    def _iou_matrix(tracks: list[Track],
                    dets:   list[Detection]) -> np.ndarray:
        mat = np.zeros((len(tracks), len(dets)), dtype=np.float32)
        for i, trk in enumerate(tracks):
            for j, det in enumerate(dets):
                if trk.det.class_id == det.class_id:
                    mat[i, j] = trk.det.bbox.iou(det.bbox)
        return mat

    def _match(self,
               iou_mat:    np.ndarray,
               n_tracks:   int,
               n_dets:     int
               ) -> tuple[list[tuple], list[int], list[int]]:
        """Hungarian assignment with fallback to greedy."""
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(-iou_mat)
            pairs = [(r, c) for r, c in zip(row_ind, col_ind)
                     if iou_mat[r, c] >= self.iou_threshold]
        except ImportError:
            # Greedy fallback for constrained devices
            pairs = []
            used_cols = set()
            for r in range(n_tracks):
                best_c  = -1
                best_iou = self.iou_threshold
                for c in range(n_dets):
                    if c not in used_cols and iou_mat[r, c] > best_iou:
                        best_iou = iou_mat[r, c]
                        best_c   = c
                if best_c >= 0:
                    pairs.append((r, best_c))
                    used_cols.add(best_c)

        matched_trks = {r for r, _ in pairs}
        matched_dets = {c for _, c in pairs}
        unmatched_trks = [r for r in range(n_tracks) if r not in matched_trks]
        unmatched_dets = [c for c in range(n_dets)   if c not in matched_dets]
        return pairs, unmatched_trks, unmatched_dets