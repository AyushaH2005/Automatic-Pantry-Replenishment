"""
Reorder engine.

Monitors tracked InventoryItems, applies thresholds,
and triggers purchase orders via grocery platform APIs.

Supported platforms (pluggable adapter pattern):
  - Blinkit   (Zomato) – ap-south-1 relevant
  - BigBasket
  - Amazon Fresh
  - Instacart (for US deployments)
  - Mock       (for testing)
"""
from __future__ import annotations
import time, json, logging, hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Optional
import urllib.request, urllib.error

logger = logging.getLogger(__name__)


# ── Order structures ──────────────────────────────────────────────────────────

@dataclass
class OrderItem:
    sku:         str
    name:        str
    quantity:    int
    unit:        str          = "unit"
    platform_id: Optional[str] = None

@dataclass
class Order:
    order_id:    str
    items:       list[OrderItem]
    platform:    str
    status:      str  = "pending"
    created_at:  float = field(default_factory=time.time)
    placed_at:   Optional[float] = None
    total_est:   Optional[float] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# ── Catalogue mapper ──────────────────────────────────────────────────────────

class CatalogueMapper:
    """
    Maps detected item names → grocery platform SKUs.

    In production: calls platform search API or uses a cached
    name→SKU embedding lookup (CLIP-based similarity).
    """

    # Hard-coded fallback catalogue (extendable via JSON config)
    _DEFAULT_CATALOGUE: dict[str, dict] = {
        "olive_oil":    {"sku": "OIL-OLV-500",  "name": "Olive Oil 500ml",    "unit": "bottle"},
        "coffee_jar":   {"sku": "BEV-COF-200",  "name": "Instant Coffee 200g", "unit": "jar"},
        "cereal_box":   {"sku": "BRK-CRL-400",  "name": "Oats Cereal 400g",   "unit": "box"},
        "pasta_packet": {"sku": "DRY-PAS-500",  "name": "Spaghetti 500g",     "unit": "packet"},
        "milk_carton":  {"sku": "DAI-MLK-1L",   "name": "Full Fat Milk 1L",   "unit": "carton"},
        "rice_bag":     {"sku": "DRY-RIC-1K",   "name": "Basmati Rice 1kg",   "unit": "bag"},
        "tea_box":      {"sku": "BEV-TEA-100",  "name": "Chai Tea 100 bags",  "unit": "box"},
        "biscuit_packet":{"sku":"SNK-BSC-200",  "name": "Digestive Biscuits", "unit": "packet"},
        "sauce_bottle": {"sku": "CON-SAU-300",  "name": "Tomato Sauce 300g",  "unit": "bottle"},
        "juice_bottle": {"sku": "BEV-JUI-1L",   "name": "Orange Juice 1L",    "unit": "bottle"},
        # Generic fallbacks by detection class
        "bottle":       {"sku": "GEN-BOT-001",  "name": "Bottle",             "unit": "bottle"},
        "can":          {"sku": "GEN-CAN-001",  "name": "Can",                "unit": "can"},
        "jar":          {"sku": "GEN-JAR-001",  "name": "Jar",                "unit": "jar"},
        "box":          {"sku": "GEN-BOX-001",  "name": "Box",                "unit": "box"},
    }

    def __init__(self, catalogue_path: Optional[str] = None):
        self._cat = dict(self._DEFAULT_CATALOGUE)
        if catalogue_path:
            try:
                with open(catalogue_path) as f:
                    self._cat.update(json.load(f))
            except Exception as e:
                logger.warning(f"Catalogue load failed: {e}")

    def lookup(self, item_name: str) -> Optional[dict]:
        name = item_name.lower().replace(" ", "_")
        # Exact match first
        if name in self._cat:
            return self._cat[name]
        # Prefix match
        for key, val in self._cat.items():
            if key in name or name in key:
                return val
        return None


# ── Platform adapters ─────────────────────────────────────────────────────────

class GroceryAdapter(ABC):
    """Abstract adapter – one implementation per grocery platform."""

    @abstractmethod
    def place_order(self, order: Order) -> bool:
        ...

    @abstractmethod
    def check_availability(self, sku: str) -> Optional[dict]:
        ...


class MockAdapter(GroceryAdapter):
    """Logs orders to console / file. Used in testing and demo mode."""

    def __init__(self, log_path: str = "/tmp/mock_orders.jsonl"):
        self.log_path = log_path

    def place_order(self, order: Order) -> bool:
        order.status   = "placed"
        order.placed_at = time.time()
        logger.info(f"[MOCK ORDER] {order.order_id}: "
                    f"{[i.name for i in order.items]}")
        with open(self.log_path, "a") as f:
            f.write(order.to_json() + "\n")
        return True

    def check_availability(self, sku: str) -> Optional[dict]:
        return {"sku": sku, "available": True, "price": 99.0, "currency": "INR"}


class BlinkitAdapter(GroceryAdapter):
    """
    Blinkit (Zomato) integration via unofficial REST API.

    In production: replace with official partner API credentials.
    Auth: bearer token (refresh via OAuth2 PKCE).
    """

    BASE_URL = "https://api.blinkit.com/v1"

    def __init__(self, api_key: str, location_id: str):
        self.api_key     = api_key
        self.location_id = location_id

    def _post(self, endpoint: str, payload: dict) -> Optional[dict]:
        url  = f"{self.BASE_URL}{endpoint}"
        data = json.dumps(payload).encode()
        req  = urllib.request.Request(
            url, data=data,
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read())
        except urllib.error.URLError as e:
            logger.error(f"Blinkit API error: {e}")
            return None

    def place_order(self, order: Order) -> bool:
        payload = {
            "location_id": self.location_id,
            "items": [
                {"product_id": item.platform_id or item.sku,
                 "quantity":   item.quantity}
                for item in order.items
            ]
        }
        result = self._post("/orders", payload)
        if result and result.get("order_id"):
            order.status   = "placed"
            order.placed_at = time.time()
            return True
        return False

    def check_availability(self, sku: str) -> Optional[dict]:
        try:
            url = (f"{self.BASE_URL}/products/search?q={sku}"
                   f"&location_id={self.location_id}")
            req = urllib.request.Request(
                url,
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                products = data.get("products", [])
                return products[0] if products else None
        except Exception as e:
            logger.error(f"Blinkit availability check failed: {e}")
            return None


# ── Reorder engine ────────────────────────────────────────────────────────────

class ReorderEngine:
    """
    Watches inventory state and autonomously places reorders.

    Logic:
      1. Every `check_interval_s` seconds, scan all tracked items.
      2. If fill_ratio < REORDER_THRESHOLD and no pending order → trigger.
      3. Cooldown prevents repeat orders within `cooldown_s`.
      4. Optional consumption-rate prediction: order before empty.
    """

    def __init__(self,
                 adapter:           GroceryAdapter,
                 mapper:            CatalogueMapper,
                 check_interval_s:  float = 60.0,
                 cooldown_s:        float = 3600.0,
                 predict_ahead_s:   float = 7200.0,
                 dry_run:           bool  = False):
        self.adapter          = adapter
        self.mapper           = mapper
        self.check_interval_s = check_interval_s
        self.cooldown_s       = cooldown_s
        self.predict_ahead_s  = predict_ahead_s
        self.dry_run          = dry_run

        self._last_order_ts:  dict[str, float] = {}   # name → last order time
        self._pending_orders: list[Order]       = []

    def check_and_order(self,
                        items: dict[int, "InventoryItem"],
                        ts: float = None) -> list[Order]:
        """
        Evaluate all tracked items. Returns list of newly placed orders.
        Call this once per detection cycle (or on a timer).
        """
        ts = ts or time.time()
        to_order: list[tuple[str, str]] = []   # (item_name, sku)

        for item in items.values():
            if self._should_order(item, ts):
                cat = self.mapper.lookup(item.name)
                if cat:
                    to_order.append((item.name, cat["sku"], cat))
                    item.reorder_pending = True
                else:
                    logger.warning(f"No catalogue entry for '{item.name}'")

        if not to_order:
            return []

        order = Order(
            order_id = self._gen_order_id(to_order, ts),
            items    = [
                OrderItem(sku=sku, name=cat["name"],
                          quantity=1, unit=cat["unit"])
                for _, sku, cat in to_order
            ],
            platform = type(self.adapter).__name__,
        )

        if self.dry_run:
            logger.info(f"[DRY RUN] Would place order: {order.order_id}")
            order.status = "dry_run"
            return [order]

        success = self.adapter.place_order(order)
        if success:
            for name, _, _ in to_order:
                self._last_order_ts[name] = ts
            self._pending_orders.append(order)
            logger.info(f"Order placed: {order.order_id} "
                        f"({len(to_order)} items)")
            return [order]

        logger.error(f"Failed to place order {order.order_id}")
        return []

    def _should_order(self, item: "InventoryItem", ts: float) -> bool:
        if item.reorder_pending:
            return False
        # Hard threshold
        if item.is_low_stock:
            return not self._in_cooldown(item.name, ts)
        # Predictive: will run out within predict_ahead_s?
        tte = item.estimated_time_to_empty()
        if tte is not None and tte < self.predict_ahead_s:
            return not self._in_cooldown(item.name, ts)
        return False

    def _in_cooldown(self, name: str, ts: float) -> bool:
        last = self._last_order_ts.get(name)
        return last is not None and (ts - last) < self.cooldown_s

    @staticmethod
    def _gen_order_id(items: list, ts: float) -> str:
        key = f"{ts}:{[n for n,_,_ in items]}"
        return "ORD-" + hashlib.md5(key.encode()).hexdigest()[:8].upper()