"""
Edge pipeline.

Pulls frames from the S3 bucket (or local buffer),
runs detection → tracking → fill estimation → reorder check.

Designed for:
  Raspberry Pi 5 (4 GB) + Pi Camera Module 3
  or
  Jetson Nano (4 GB) with USB camera

S3 feed mode: polls the bucket listing for new files at `poll_interval_s`.
Live mode:    captures directly from /dev/video0 or PiCamera.
"""
from __future__ import annotations
import time, json, io, logging, threading
from pathlib import Path
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports – only load heavy libs when needed
def _import_cv2():
    import cv2
    return cv2

def _import_boto3():
    import boto3
    return boto3


# ── S3 frame fetcher ─────────────────────────────────────────────────────────

class S3FrameFetcher:
    """
    Continuously fetches burst frames from the S3 bucket.

    Naming convention observed in the feed:
        cam2/photo_YYYYMMDD_HHMMSS_N.jpg   (N = 1..5)

    Strategy:
      1. List bucket with prefix=cam2/ and sort by LastModified.
      2. Group consecutive keys sharing the same HHMMSS timestamp.
      3. Download the group and yield as a burst (list of np.ndarray).
    """

    def __init__(self,
                 bucket:          str   = "smart-inventory-management-device",
                 prefix:          str   = "cam2/",
                 region:          str   = "ap-south-1",
                 poll_interval_s: float = 5.0,
                 burst_size:      int   = 5,
                 max_lag_s:       float = 30.0):
        self.bucket          = bucket
        self.prefix          = prefix
        self.region          = region
        self.poll_interval_s = poll_interval_s
        self.burst_size      = burst_size
        self.max_lag_s       = max_lag_s

        self._s3        = None
        self._last_key  = None
        self._lock      = threading.Lock()

    def _get_s3(self):
        if self._s3 is None:
            boto3 = _import_boto3()
            self._s3 = boto3.client("s3", region_name=self.region)
        return self._s3

    def list_new_bursts(self) -> list[dict]:
        """
        Returns a list of burst metadata dicts (not yet downloaded).
        Each dict: {"timestamp": "HHMMSS", "keys": [...5 keys...]}
        """
        s3 = self._get_s3()
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self.bucket, Prefix=self.prefix)

        all_keys = []
        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                # Only process new keys
                if self._last_key and key <= self._last_key:
                    continue
                all_keys.append({"key": key, "ts": obj["LastModified"]})

        if not all_keys:
            return []

        # Group by YYYYMMDD_HHMMSS (characters 9-22 in filename)
        from collections import defaultdict
        groups: dict[str, list[str]] = defaultdict(list)
        for item in all_keys:
            fname = Path(item["key"]).stem          # photo_20260516_144431_2
            parts = fname.split("_")
            if len(parts) >= 3:
                ts_key = f"{parts[1]}_{parts[2]}"  # 20260516_144431
                groups[ts_key].append(item["key"])

        bursts = []
        for ts_key in sorted(groups.keys()):
            keys = sorted(groups[ts_key])
            if len(keys) >= 2:              # need at least 2 frames to aggregate
                bursts.append({"timestamp": ts_key, "keys": keys})

        if bursts:
            all_sorted_keys = [k for b in bursts for k in b["keys"]]
            with self._lock:
                self._last_key = max(all_sorted_keys)

        return bursts

    def download_burst(self, burst_meta: dict) -> Optional[list[np.ndarray]]:
        """Download a burst's frames as numpy arrays."""
        cv2 = _import_cv2()
        s3  = self._get_s3()
        frames = []
        for key in burst_meta["keys"][:self.burst_size]:
            try:
                resp = s3.get_object(Bucket=self.bucket, Key=key)
                buf  = np.frombuffer(resp["Body"].read(), np.uint8)
                img  = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img is not None:
                    frames.append(img)
            except Exception as e:
                logger.warning(f"Failed to download {key}: {e}")
        return frames if len(frames) >= 2 else None

    def stream_bursts(self):
        """Generator: yields (timestamp_str, [frames]) tuples indefinitely."""
        while True:
            t0 = time.time()
            try:
                bursts = self.list_new_bursts()
                for burst_meta in bursts:
                    frames = self.download_burst(burst_meta)
                    if frames:
                        yield burst_meta["timestamp"], frames
            except Exception as e:
                logger.error(f"S3 fetch error: {e}")
            # Pace to poll_interval
            elapsed = time.time() - t0
            time.sleep(max(0.0, self.poll_interval_s - elapsed))


# ── Live camera fetcher ───────────────────────────────────────────────────────

class LiveCameraFetcher:
    """
    Captures burst frames from a locally connected camera.
    Supports V4L2 (USB cam) and Raspberry Pi Camera Module 3.
    """

    def __init__(self,
                 device:         int   = 0,
                 burst_size:     int   = 5,
                 burst_interval: float = 4.0,
                 width:          int   = 1920,
                 height:         int   = 1080):
        self.device         = device
        self.burst_size     = burst_size
        self.burst_interval = burst_interval
        self.width          = width
        self.height         = height
        self._cap           = None

    def _open(self):
        cv2 = _import_cv2()
        self._cap = cv2.VideoCapture(self.device)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def capture_burst(self) -> Optional[list[np.ndarray]]:
        if self._cap is None:
            self._open()
        frames = []
        for _ in range(self.burst_size):
            ret, frame = self._cap.read()
            if ret:
                frames.append(frame)
            time.sleep(0.02)    # ~50 ms between burst frames
        return frames if len(frames) >= 2 else None

    def stream_bursts(self):
        """Generator: yields (timestamp_str, [frames]) tuples."""
        while True:
            t0     = time.time()
            ts_str = time.strftime("%Y%m%d_%H%M%S")
            frames = self.capture_burst()
            if frames:
                yield ts_str, frames
            elapsed = time.time() - t0
            time.sleep(max(0.0, self.burst_interval - elapsed))

    def __del__(self):
        if self._cap is not None:
            self._cap.release()


# ── Main edge pipeline ────────────────────────────────────────────────────────

class EdgePipeline:
    """
    Orchestrates the full pipeline:
      fetch → detect → track → fill_estimate → reorder

    Runs on a single thread (edge device). All heavy ops are synchronous.
    For Jetson with CUDA: set backend='yolov8n' and half_precision=True.
    """

    def __init__(self, config: dict):
        self.config   = config
        self._setup()

    def _setup(self):
        from pipeline.detector import PantryDetector
        from pipeline.tracker  import PantryTracker
        from reorder.engine    import (ReorderEngine, CatalogueMapper,
                                       MockAdapter, BlinkitAdapter)

        det_cfg = self.config.get("detector", {})
        self.detector = PantryDetector(
            backend=det_cfg.get("backend", "mock"),
            config=det_cfg,
        )
        trk_cfg = self.config.get("tracker", {})
        self.tracker = PantryTracker(
            max_age       = trk_cfg.get("max_age", 30),
            min_hits      = trk_cfg.get("min_hits", 3),
            iou_threshold = trk_cfg.get("iou_threshold", 0.35),
        )
        mapper = CatalogueMapper(
            self.config.get("catalogue_path")
        )
        platform = self.config.get("platform", "mock")
        if platform == "blinkit":
            adapter = BlinkitAdapter(
                api_key     = self.config["blinkit_api_key"],
                location_id = self.config["blinkit_location_id"],
            )
        else:
            adapter = MockAdapter()

        reorder_cfg = self.config.get("reorder", {})
        self.reorder = ReorderEngine(
            adapter          = adapter,
            mapper           = mapper,
            check_interval_s = reorder_cfg.get("check_interval_s",  60),
            cooldown_s       = reorder_cfg.get("cooldown_s",       3600),
            predict_ahead_s  = reorder_cfg.get("predict_ahead_s",  7200),
            dry_run          = reorder_cfg.get("dry_run",           True),
        )
        # Frame source
        src_cfg = self.config.get("source", {})
        if src_cfg.get("type") == "s3":
            self.fetcher = S3FrameFetcher(
                bucket          = src_cfg.get("bucket", "smart-inventory-management-device"),
                prefix          = src_cfg.get("prefix", "cam2/"),
                region          = src_cfg.get("region", "ap-south-1"),
                poll_interval_s = src_cfg.get("poll_interval_s", 5.0),
            )
        else:
            self.fetcher = LiveCameraFetcher(
                device      = src_cfg.get("device", 0),
                burst_size  = src_cfg.get("burst_size", 5),
            )

        self._frame_count    = 0
        self._last_reorder_check = 0.0
        self._state_path     = self.config.get("state_path", "/tmp/pantry_state.json")

    def run(self, max_bursts: Optional[int] = None):
        """Main loop. Set max_bursts for finite runs (testing)."""
        logger.info("EdgePipeline starting…")
        for i, (ts_str, frames) in enumerate(self.fetcher.stream_bursts()):
            if max_bursts and i >= max_bursts:
                break
            self._process_burst(ts_str, frames)

    def _process_burst(self, ts_str: str, frames: list[np.ndarray]):
        t_start = time.time()
        ts      = time.time()

        # 1. Detect + aggregate burst
        detections = self.detector.detect_burst(frames, ts)

        # 2. Track
        confirmed = self.tracker.update(detections, ts)

        # 3. Reorder check (throttled)
        orders = []
        if ts - self._last_reorder_check > self.reorder.check_interval_s:
            orders = self.reorder.check_and_order(self.tracker.all_items, ts)
            self._last_reorder_check = ts

        # 4. Log / persist state
        self._frame_count += len(frames)
        latency_ms = (time.time() - t_start) * 1000

        state = {
            "ts":            ts_str,
            "n_detections":  len(detections),
            "n_tracked":     len(confirmed),
            "items":         {
                str(t.track_id): {
                    "name":       t.item.name,
                    "fill_ratio": round(t.item.fill_ratio, 3),
                    "is_low":     t.item.is_low_stock,
                    "reorder":    t.item.reorder_pending,
                }
                for t in confirmed if t.item
            },
            "orders_placed": len(orders),
            "latency_ms":    round(latency_ms, 1),
        }

        logger.info(
            f"[{ts_str}] dets={len(detections)} tracks={len(confirmed)} "
            f"orders={len(orders)} latency={latency_ms:.0f}ms"
        )

        # Persist lightweight JSON state (readable by dashboard)
        try:
            with open(self._state_path, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"State write failed: {e}")

        return state


# ── CLI entrypoint ────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pantry edge pipeline")
    parser.add_argument("--config", default="config/edge_config.json")
    parser.add_argument("--bursts", type=int, default=None,
                        help="Stop after N bursts (omit = run forever)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    with open(args.config) as f:
        config = json.load(f)

    pipeline = EdgePipeline(config)
    pipeline.run(max_bursts=args.bursts)


if __name__ == "__main__":
    main()