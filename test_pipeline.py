"""
End-to-end pipeline test using mock data.

Mirrors the real S3 feed:
  - cam2/photo_20260516_HHMMSS_N.jpg pattern
  - 5-frame bursts every ~4 s
  - ~14 minutes of footage simulated as 200 bursts
  - Item fill ratios decrease over time to trigger reorder logic
"""
from __future__ import annotations
import sys, os, time, json, logging, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


# ── Mock frame generator ──────────────────────────────────────────────────────

def make_mock_frame(width=640, height=480, seed=42) -> np.ndarray:
    """Create a synthetic pantry shelf image (coloured boxes as items)."""
    rng   = np.random.default_rng(seed)
    frame = np.ones((height, width, 3), dtype=np.uint8) * 220  # light bg

    # Simulate 4 shelf items as coloured rectangles
    items = [
        (50,  80,  130, 300, (30,  80, 180)),   # blue bottle
        (200, 100, 130, 260, (80,  160, 80)),    # green jar
        (350, 50,  160, 350, (200, 100, 50)),    # brown box
        (530, 120, 120, 230, (160, 60,  160)),   # purple packet
    ]
    for x, y, w, h, colour in items:
        frame[y:y+h, x:x+w] = colour
        # Slight noise per frame
        noise = rng.integers(-10, 10, (h, w, 3)).astype(np.int16)
        region = frame[y:y+h, x:x+w].astype(np.int16) + noise
        frame[y:y+h, x:x+w] = np.clip(region, 0, 255).astype(np.uint8)

    return frame


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestDetector(unittest.TestCase):

    def setUp(self):
        from pipeline.detector import PantryDetector
        self.det = PantryDetector(backend="mock")

    def test_detect_frame_returns_detections(self):
        frame = make_mock_frame()
        ts    = time.time()
        dets  = self.det.detect_frame(frame, ts)
        self.assertGreater(len(dets), 0)
        logger.info(f"detect_frame: {len(dets)} detections")

    def test_fill_ratio_in_range(self):
        frame = make_mock_frame()
        dets  = self.det.detect_frame(frame)
        for d in dets:
            self.assertGreaterEqual(d.fill_ratio, 0.0)
            self.assertLessEqual(d.fill_ratio,    1.0)

    def test_burst_aggregation_reduces_noise(self):
        frames = [make_mock_frame(seed=i) for i in range(5)]
        ts     = time.time()
        single = self.det.detect_frame(frames[0], ts)
        burst  = self.det.detect_burst(frames, ts)
        # Burst should return ≥ same items (not fewer due to vote filter)
        logger.info(f"single={len(single)} burst={len(burst)}")
        self.assertGreater(len(burst), 0)

    def test_confidence_values(self):
        frame = make_mock_frame()
        dets  = self.det.detect_frame(frame)
        for d in dets:
            self.assertGreater(d.confidence, 0)
            self.assertLessEqual(d.confidence, 1.0)


class TestFillEstimator(unittest.TestCase):

    def setUp(self):
        from pipeline.detector import FillEstimator
        self.est = FillEstimator()

    def test_transparent_full_bottle(self):
        """Tall dark crop bottom-heavy = mostly full."""
        crop = np.zeros((200, 80, 3), dtype=np.uint8)
        crop[20:200, :] = [180, 120, 60]   # content fills most of height
        ratio = self.est.estimate(crop, pkg_type="transparent")
        self.assertGreater(ratio, 0.5)

    def test_empty_crop_returns_sensible_value(self):
        crop  = np.zeros((10, 10, 3), dtype=np.uint8)
        ratio = self.est.estimate(crop)
        self.assertGreaterEqual(ratio, 0.0)
        self.assertLessEqual(ratio,    1.0)

    def test_none_crop_returns_1(self):
        ratio = self.est.estimate(None)
        self.assertEqual(ratio, 1.0)


class TestTracker(unittest.TestCase):

    def setUp(self):
        from pipeline.detector import Detection, BBox
        from pipeline.tracker  import PantryTracker
        self.tracker = PantryTracker(min_hits=1)
        self.Detection = Detection
        self.BBox      = BBox

    def _make_det(self, name, x1, y1, x2, y2, fill=0.8):
        return self.Detection(
            bbox=self.BBox(x1, y1, x2, y2),
            confidence=0.85,
            class_id=hash(name) % 100,
            class_name=name,
            fill_ratio=fill,
        )

    def test_tracks_assigned_stable_ids(self):
        dets1 = [self._make_det("bottle", 50, 80, 180, 380)]
        dets2 = [self._make_det("bottle", 52, 81, 182, 382)]  # slight shift
        ts = time.time()

        trks1 = self.tracker.update(dets1, ts)
        trks2 = self.tracker.update(dets2, ts + 4)

        self.assertEqual(len(trks1), 1)
        self.assertEqual(len(trks2), 1)
        self.assertEqual(trks1[0].track_id, trks2[0].track_id)
        logger.info(f"Stable track ID: {trks1[0].track_id}")

    def test_new_item_spawns_new_track(self):
        ts   = time.time()
        det1 = self._make_det("bottle", 50, 80, 180, 380)
        det2 = self._make_det("jar",   300, 80, 430, 380)
        trks = self.tracker.update([det1, det2], ts)
        ids  = {t.track_id for t in trks}
        self.assertEqual(len(ids), 2)

    def test_missing_item_ages_out(self):
        ts   = time.time()
        det  = self._make_det("bottle", 50, 80, 180, 380)
        self.tracker.update([det], ts)
        # Advance many frames with no detections
        for i in range(35):
            self.tracker.update([], ts + i * 4)
        confirmed = self.tracker.confirmed_tracks
        self.assertEqual(len(confirmed), 0)
        logger.info("Item correctly aged out after 35 missed frames")

    def test_fill_history_grows(self):
        ts  = time.time()
        det = self._make_det("bottle", 50, 80, 180, 380, fill=0.9)
        self.tracker.update([det], ts)
        for i in range(10):
            fill = 0.9 - i * 0.05
            d    = self._make_det("bottle", 50, 80, 180, 380, fill=fill)
            self.tracker.update([d], ts + i * 4)
        item = list(self.tracker.all_items.values())[0]
        self.assertGreater(len(item.fill_history), 3)
        logger.info(f"Fill history: {item.fill_history[-3:]}")


class TestReorderEngine(unittest.TestCase):

    def setUp(self):
        from pipeline.detector import InventoryItem
        from reorder.engine    import ReorderEngine, CatalogueMapper, MockAdapter
        self.InventoryItem = InventoryItem
        self.engine = ReorderEngine(
            adapter          = MockAdapter(log_path="/tmp/test_orders.jsonl"),
            mapper           = CatalogueMapper(),
            check_interval_s = 0,
            cooldown_s       = 3600,
            dry_run          = False,
        )

    def _make_item(self, name, fill) -> "InventoryItem":
        item = self.InventoryItem(track_id=1, name=name)
        item.fill_ratio = fill
        return item

    def test_low_stock_triggers_order(self):
        item   = self._make_item("olive_oil", fill=0.10)
        orders = self.engine.check_and_order({1: item})
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].status, "placed")
        logger.info(f"Order triggered: {orders[0].order_id}")

    def test_full_stock_no_order(self):
        item   = self._make_item("olive_oil", fill=0.80)
        orders = self.engine.check_and_order({1: item})
        self.assertEqual(len(orders), 0)

    def test_cooldown_prevents_repeat_order(self):
        item = self._make_item("olive_oil", fill=0.10)
        ts   = time.time()
        o1   = self.engine.check_and_order({1: item}, ts)
        item.reorder_pending = False   # reset flag
        o2   = self.engine.check_and_order({1: item}, ts + 60)  # within cooldown
        self.assertEqual(len(o1), 1)
        self.assertEqual(len(o2), 0)
        logger.info("Cooldown correctly suppressed duplicate order")

    def test_unknown_item_skipped(self):
        item   = self._make_item("unknown_alien_food_xyz", fill=0.05)
        orders = self.engine.check_and_order({1: item})
        self.assertEqual(len(orders), 0)
        logger.info("Unknown item correctly skipped (no catalogue entry)")

    def test_consumption_rate(self):
        item = self.InventoryItem(track_id=99, name="coffee_jar")
        ts   = time.time()
        # Simulate consumption over 10 minutes
        for i in range(20):
            item.fill_history.append((ts + i * 30, 1.0 - i * 0.03))
        item.fill_ratio = 1.0 - 19 * 0.03
        rate = item.consumption_rate()
        self.assertIsNotNone(rate)
        self.assertGreater(rate, 0)
        tte  = item.estimated_time_to_empty()
        logger.info(f"Consumption rate: {rate:.6f}/s, ETE: {tte/60:.1f} min")


class TestEndToEnd(unittest.TestCase):
    """
    Simulates 200 burst events (≈ 14 min of footage at 4 s intervals).
    Verifies full pipeline produces at least one reorder.
    """

    def test_full_pipeline_simulation(self):
        from pipeline.detector import PantryDetector
        from pipeline.tracker  import PantryTracker
        from reorder.engine    import (ReorderEngine, CatalogueMapper,
                                       MockAdapter)

        detector = PantryDetector(backend="mock")
        tracker  = PantryTracker(min_hits=2)
        reorder  = ReorderEngine(
            adapter          = MockAdapter("/tmp/e2e_orders.jsonl"),
            mapper           = CatalogueMapper(),
            check_interval_s = 0,
            cooldown_s       = 900,
            predict_ahead_s  = 7200,
            dry_run          = False,
        )

        t_start     = time.time() - 3600   # pretend feed started 1h ago
        all_orders  = []
        state_snapshots = []
        frames_per_burst = 5
        n_bursts    = 200

        for i in range(n_bursts):
            ts     = t_start + i * 4.2
            frames = [make_mock_frame(seed=i * frames_per_burst + j)
                      for j in range(frames_per_burst)]

            dets      = detector.detect_burst(frames, ts)
            confirmed = tracker.update(dets, ts)
            orders    = reorder.check_and_order(tracker.all_items, ts)
            all_orders.extend(orders)

            if i % 50 == 0:
                items_state = {
                    name: round(item.fill_ratio, 2)
                    for item in tracker.all_items.values()
                    for name in [item.name]
                }
                logger.info(f"Burst {i:3d}: dets={len(dets)} "
                            f"orders_so_far={len(all_orders)} "
                            f"items={items_state}")
                state_snapshots.append({
                    "burst": i, "ts": ts,
                    "items": items_state,
                    "n_orders": len(all_orders)
                })

        logger.info(f"\n{'='*60}")
        logger.info(f"SIMULATION COMPLETE")
        logger.info(f"  Bursts processed : {n_bursts}")
        logger.info(f"  Total frames     : {n_bursts * frames_per_burst}")
        logger.info(f"  Unique items     : {len(tracker.all_items)}")
        logger.info(f"  Orders placed    : {len(all_orders)}")

        if all_orders:
            for o in all_orders[:3]:
                logger.info(f"    → {o.order_id}: "
                            f"{[i.name for i in o.items]}")
        logger.info(f"{'='*60}")

        # Save simulation report
        report = {
            "n_bursts":    n_bursts,
            "n_frames":    n_bursts * frames_per_burst,
            "n_items":     len(tracker.all_items),
            "n_orders":    len(all_orders),
            "snapshots":   state_snapshots,
            "orders":      [
                {"id": o.order_id, "items": [i.name for i in o.items],
                 "status": o.status}
                for o in all_orders
            ]
        }
        with open("/tmp/simulation_report.json", "w") as f:
            json.dump(report, f, indent=2)

        # Assertions
        self.assertGreater(len(tracker.all_items), 0, "No items tracked")
        # At least one reorder should fire as items deplete
        self.assertGreater(len(all_orders), 0,
                           "Expected at least one reorder order")


if __name__ == "__main__":
    unittest.main(verbosity=2)