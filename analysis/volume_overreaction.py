"""
Volume Acceleration & Overreaction Detector

Two tightly coupled signals from XGBoost research:

1. VOLUME ACCELERATION (Feature #2):
   Sudden 5x+ volume spikes indicate insider knowledge or breaking news.
   Detect before price fully adjusts → front-run the move.

2. SERIAL CORRELATION / OVERREACTION (Feature #3):
   58% of political markets show negative serial correlation.
   Big price move → market overreacts → price reverts.
   Strategy: fade the move, profit from mean reversion.
"""

import time
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

from config.settings import Settings
from data.gamma_client import Market, Event


@dataclass
class VolumeSpike:
    """Detected volume acceleration event."""
    condition_id: str
    question: str
    category: str
    current_volume: float
    avg_volume: float
    acceleration: float  # current / avg ratio
    direction: str  # "unknown", "yes_buying", "no_buying"
    detected_at: float
    confidence: float  # how reliable is this spike

    @property
    def is_extreme(self) -> bool:
        return self.acceleration >= 5.0

    def __str__(self):
        emoji = "🚀" if self.acceleration >= 5 else "📊"
        return (
            f"{emoji} Volume {self.acceleration:.1f}x | "
            f"{self.question[:50]} | {self.direction}"
        )


@dataclass
class OverreactionSignal:
    """Detected overreaction / mean-reversion opportunity."""
    condition_id: str
    question: str
    category: str
    current_price: float
    pre_move_price: float
    move_size: float  # absolute price change
    move_pct: float  # percentage move
    serial_correlation: float  # lag-1 autocorrelation
    reversion_probability: float  # estimated probability of reverting
    suggested_side: str  # "yes" or "no" — which to buy for reversion
    suggested_entry: float  # suggested entry price
    suggested_target: float  # expected reversion target
    detected_at: float

    @property
    def edge_pct(self) -> float:
        return abs(self.suggested_target - self.suggested_entry) * 100

    def __str__(self):
        arrow = "↩️" if self.serial_correlation < 0 else "→"
        return (
            f"{arrow} Overreaction {self.move_pct:+.1f}% | "
            f"Revert prob: {self.reversion_probability:.0%} | "
            f"{self.question[:50]}"
        )


class VolumeAccelerationDetector:
    """
    Tracks volume across scan cycles, detects abnormal spikes.

    Algorithm:
    1. Maintain rolling 24h volume observations per market
    2. Each scan: compute acceleration = current_vol / rolling_avg
    3. If acceleration > threshold (3x default) → signal
    4. Direction inference: if price moved up with spike → YES buying
    """

    def __init__(self, spike_threshold: float = 3.0):
        self.spike_threshold = spike_threshold
        # condition_id -> deque of (timestamp, volume)
        self._history: Dict[str, deque] = {}
        self._max_window = 86400 * 3  # 3 days of data
        # Last known price for direction inference
        self._last_price: Dict[str, float] = {}

    def record(self, condition_id: str, volume: float, price: float = 0):
        """Record a volume observation."""
        now = time.time()
        if condition_id not in self._history:
            self._history[condition_id] = deque(maxlen=500)
        self._history[condition_id].append((now, volume))
        if price > 0:
            self._last_price[condition_id] = price

    def detect(self, market: Market, current_volume: float,
               current_price: float = 0) -> Optional[VolumeSpike]:
        """Check if current volume is abnormally high."""
        cid = market.condition_id
        self.record(cid, current_volume, current_price)

        hist = self._history.get(cid)
        if not hist or len(hist) < 3:
            return None

        now = time.time()

        # Calculate rolling average (exclude latest observation)
        observations = [(t, v) for t, v in hist
                        if now - t < self._max_window]
        if len(observations) < 3:
            return None

        avg_vol = sum(v for _, v in observations[:-1]) / max(len(observations) - 1, 1)
        if avg_vol <= 0:
            return None

        acceleration = current_volume / avg_vol

        if acceleration < self.spike_threshold:
            return None

        # Infer direction from price movement
        direction = "unknown"
        prev_price = self._last_price.get(cid, 0)
        if prev_price > 0 and current_price > 0:
            if current_price > prev_price + 0.02:
                direction = "yes_buying"
            elif current_price < prev_price - 0.02:
                direction = "no_buying"

        # Confidence based on how many data points we have
        data_quality = min(1.0, len(observations) / 20)
        spike_clarity = min(1.0, acceleration / 10)
        confidence = (data_quality * 0.4 + spike_clarity * 0.6)

        return VolumeSpike(
            condition_id=cid,
            question=market.question,
            category=market.category or "",
            current_volume=current_volume,
            avg_volume=avg_vol,
            acceleration=acceleration,
            direction=direction,
            detected_at=now,
            confidence=confidence,
        )

    def scan_markets(self, markets: List[Tuple[Market, float, float]]
                      ) -> List[VolumeSpike]:
        """
        Scan multiple markets for volume spikes.
        Args: List of (market, current_volume, current_price)
        """
        spikes = []
        for market, vol, price in markets:
            spike = self.detect(market, vol, price)
            if spike:
                spikes.append(spike)

        spikes.sort(key=lambda s: -s.acceleration)
        return spikes


class OverreactionDetector:
    """
    Detects mean-reversion opportunities from market overreactions.

    Based on research: 58% of political markets have negative serial correlation.
    After a large move (>5%), there's a statistical tendency to revert.

    Algorithm:
    1. Track price changes per market
    2. Compute lag-1 serial correlation
    3. After large moves, if serial correlation is negative → reversion signal
    4. Size the trade based on historical reversion rate
    """

    def __init__(self, move_threshold: float = 0.05,
                 min_serial_corr: float = -0.15):
        self.move_threshold = move_threshold  # 5 cents minimum move
        self.min_serial_corr = min_serial_corr
        # condition_id -> list of (timestamp, price)
        self._price_history: Dict[str, List[Tuple[float, float]]] = {}
        # Track detected signals (cooldown)
        self._recent_signals: Dict[str, float] = {}
        self._signal_cooldown = 3600  # 1 hour between signals for same market

    def record_price(self, condition_id: str, price: float):
        """Record a price observation."""
        now = time.time()
        hist = self._price_history.setdefault(condition_id, [])
        hist.append((now, price))
        # Keep last 200 observations
        if len(hist) > 200:
            hist[:] = hist[-200:]

    def detect(self, market: Market, current_price: float
               ) -> Optional[OverreactionSignal]:
        """Check if market just overreacted and is due for reversion."""
        cid = market.condition_id
        now = time.time()
        self.record_price(cid, current_price)

        # Cooldown check
        last_signal = self._recent_signals.get(cid, 0)
        if now - last_signal < self._signal_cooldown:
            return None

        hist = self._price_history.get(cid, [])
        if len(hist) < 10:
            return None

        prices = [p for _, p in hist]

        # Calculate recent move (last observation vs 1h ago)
        recent_prices = [p for t, p in hist if now - t < 7200]  # 2h window
        if len(recent_prices) < 3:
            return None

        pre_move = recent_prices[0]
        move = current_price - pre_move
        move_abs = abs(move)

        if move_abs < self.move_threshold:
            return None  # move not large enough

        # Calculate serial correlation
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        if len(changes) < 5:
            return None

        serial_corr = self._serial_correlation(changes)

        # Only signal if negative serial correlation (mean-reverting)
        if serial_corr > self.min_serial_corr:
            return None

        # Estimate reversion probability
        reversal_count = sum(
            1 for i in range(1, len(changes))
            if abs(changes[i-1]) > self.move_threshold / 2
            and changes[i] * changes[i-1] < 0
        )
        large_moves = sum(
            1 for c in changes[:-1] if abs(c) > self.move_threshold / 2
        )
        reversion_prob = reversal_count / max(large_moves, 1)

        # Need meaningful reversion probability
        if reversion_prob < 0.4:
            return None

        # Determine trade direction: fade the move
        if move > 0:
            # Price moved up → buy NO (expect reversion down)
            suggested_side = "no"
            suggested_entry = 1 - current_price
            suggested_target = 1 - (current_price - move_abs * 0.5)
        else:
            # Price moved down → buy YES (expect reversion up)
            suggested_side = "yes"
            suggested_entry = current_price
            suggested_target = current_price + move_abs * 0.5

        # Category bonus: politics gets higher confidence
        cat = (market.category or "").lower()
        cat_bonus = 0.1 if cat in ("politics", "elections") else 0

        self._recent_signals[cid] = now

        return OverreactionSignal(
            condition_id=cid,
            question=market.question,
            category=cat,
            current_price=current_price,
            pre_move_price=pre_move,
            move_size=move_abs,
            move_pct=move / max(pre_move, 0.01) * 100,
            serial_correlation=serial_corr,
            reversion_probability=min(0.95, reversion_prob + cat_bonus),
            suggested_side=suggested_side,
            suggested_entry=suggested_entry,
            suggested_target=suggested_target,
            detected_at=now,
        )

    def scan_markets(self, markets: List[Tuple[Market, float]]
                      ) -> List[OverreactionSignal]:
        """
        Scan multiple markets for overreaction signals.
        Args: List of (market, current_price)
        """
        signals = []
        for market, price in markets:
            sig = self.detect(market, price)
            if sig:
                signals.append(sig)

        signals.sort(key=lambda s: -s.reversion_probability)
        return signals

    @staticmethod
    def _serial_correlation(changes: List[float]) -> float:
        """Compute lag-1 autocorrelation of price changes."""
        n = len(changes)
        if n < 4:
            return 0
        mean_c = sum(changes) / n
        var_c = sum((c - mean_c) ** 2 for c in changes) / n
        if var_c < 1e-10:
            return 0
        cov = sum(
            (changes[i] - mean_c) * (changes[i-1] - mean_c)
            for i in range(1, n)
        ) / (n - 1)
        return cov / var_c
