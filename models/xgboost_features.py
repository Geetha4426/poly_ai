"""
XGBoost Feature Engine — 47-feature market prediction model.

Based on TemsYanik research: 12,847 Polymarket markets, 79% accuracy.
Top features by importance:
  1. Price-Probability Divergence (84% accuracy when >15pt)
  2. Volume Acceleration
  3. Serial Correlation (58% political markets overreact)

Category accuracy: Economics 83%, Politics 82%, Crypto 77%, Sports 74%.
XGBoost >> Random Forest >> Logistic Regression >> LSTM for this domain.
"""

import math
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from data.gamma_client import Market, Event


@dataclass
class MarketFeatures:
    """47-feature vector for a single market."""
    condition_id: str
    question: str
    category: str
    features: Dict[str, float] = field(default_factory=dict)
    computed_at: float = 0

    @property
    def price_prob_divergence(self) -> float:
        return self.features.get("price_prob_divergence", 0)

    @property
    def volume_acceleration(self) -> float:
        return self.features.get("volume_acceleration", 0)

    @property
    def serial_correlation(self) -> float:
        return self.features.get("serial_correlation", 0)

    def as_vector(self) -> List[float]:
        """Return ordered feature vector for model input."""
        return [self.features.get(k, 0) for k in FEATURE_NAMES]


# ═══════════════════════════════════════════════════════════════
# FEATURE DEFINITIONS (47 features)
# ═══════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    # Price features (1-8)
    "yes_price", "no_price", "mid_price", "spread",
    "price_distance_from_50", "price_extremity",
    "price_prob_divergence", "price_momentum_1h",

    # Volume features (9-16)
    "volume_24h", "volume_7d", "volume_total",
    "volume_acceleration", "volume_ratio_24h_7d",
    "volume_per_liquidity", "volume_trend", "volume_spike",

    # Liquidity features (17-22)
    "liquidity", "liquidity_depth_ratio",
    "bid_ask_spread", "book_imbalance",
    "slippage_50usd", "slippage_200usd",

    # Time features (23-28)
    "hours_to_expiry", "days_to_expiry",
    "time_decay_factor", "urgency_score",
    "pct_time_elapsed", "is_closing_soon",

    # Volatility features (29-34)
    "volatility_24h", "volatility_48h", "volatility_7d",
    "volatility_ratio_short_long", "price_range_24h",
    "max_drawdown_7d",

    # Market structure (35-40)
    "num_outcomes", "is_neg_risk", "has_related_markets",
    "category_encoded", "subcategory_encoded", "tag_count",

    # Serial correlation / momentum (41-44)
    "serial_correlation", "mean_reversion_score",
    "trend_strength", "overreaction_score",

    # Composite signals (45-47)
    "edge_composite", "risk_reward_ratio", "kelly_fraction",
]

# Category encoding (based on accuracy tiers)
CATEGORY_ENCODING = {
    "economics": 0.83, "finance": 0.83,
    "politics": 0.82, "elections": 0.82,
    "crypto": 0.77,
    "sports": 0.74,
    "culture": 0.70, "entertainment": 0.70,
    "science": 0.75, "geopolitics": 0.78,
    "other": 0.65,
}


class XGBoostFeatureEngine:
    """
    Computes 47 features per market for ML-based edge detection.

    This is a FEATURE EXTRACTOR, not the model itself.
    Features are designed to match the TemsYanik research findings.
    Can be used to:
    1. Score markets for edge probability
    2. Rank markets by predicted profitability
    3. Feed into a trained XGBoost model
    """

    def __init__(self):
        # Price history cache: condition_id -> [(timestamp, price)]
        self._price_history: Dict[str, List[Tuple[float, float]]] = {}
        # Volume history cache: condition_id -> [(timestamp, volume)]
        self._volume_history: Dict[str, List[Tuple[float, float]]] = {}
        self._max_history = 500  # max data points per market

    def record_price(self, condition_id: str, price: float, ts: float = 0):
        """Record a price observation for history tracking."""
        if not ts:
            ts = time.time()
        hist = self._price_history.setdefault(condition_id, [])
        hist.append((ts, price))
        if len(hist) > self._max_history:
            hist[:] = hist[-self._max_history:]

    def record_volume(self, condition_id: str, volume: float, ts: float = 0):
        """Record a volume observation."""
        if not ts:
            ts = time.time()
        hist = self._volume_history.setdefault(condition_id, [])
        hist.append((ts, volume))
        if len(hist) > self._max_history:
            hist[:] = hist[-self._max_history:]

    def compute_features(self, market: Market, event: Event,
                          orderbook: Optional[Dict] = None,
                          external_prob: Optional[float] = None) -> MarketFeatures:
        """
        Compute all 47 features for a market.

        Args:
            market: Market data from Gamma API
            event: Parent event
            orderbook: Optional CLOB orderbook {bids: [(p,s)], asks: [(p,s)]}
            external_prob: Optional external probability estimate for divergence
        """
        f = {}
        now = time.time()
        yes_price = market.best_yes_price
        no_price = market.best_no_price
        mid = (yes_price + no_price) / 2

        # ── Price features (1-8) ──
        f["yes_price"] = yes_price
        f["no_price"] = no_price
        f["mid_price"] = mid
        f["spread"] = abs(yes_price + no_price - 1.0)
        f["price_distance_from_50"] = abs(mid - 0.5)
        f["price_extremity"] = max(yes_price, no_price)

        # Price-Probability Divergence (#1 feature, 84% accuracy at >15pt)
        if external_prob is not None:
            f["price_prob_divergence"] = abs(external_prob - yes_price) * 100
        else:
            # Without external, use distance from 0.5 as proxy
            f["price_prob_divergence"] = abs(yes_price - 0.5) * 100 * 0.3

        # Price momentum (from history)
        price_hist = self._price_history.get(market.condition_id, [])
        f["price_momentum_1h"] = self._calc_momentum(price_hist, 3600)

        # ── Volume features (9-16) ──
        vol_24h = getattr(market, 'volume', 0) or 0
        f["volume_24h"] = vol_24h
        f["volume_7d"] = vol_24h * 5  # estimate if no 7d data
        f["volume_total"] = event.volume if event else vol_24h
        f["volume_acceleration"] = self._calc_volume_acceleration(
            market.condition_id, vol_24h
        )
        f["volume_ratio_24h_7d"] = (
            vol_24h / (vol_24h * 5) if vol_24h > 0 else 0.2
        )
        liq = getattr(market, 'liquidity', 0) or 1
        f["volume_per_liquidity"] = vol_24h / max(liq, 1)
        f["volume_trend"] = self._calc_volume_trend(market.condition_id)
        f["volume_spike"] = 1.0 if f["volume_acceleration"] > 3.0 else 0.0

        # ── Liquidity features (17-22) ──
        f["liquidity"] = liq
        f["liquidity_depth_ratio"] = min(liq / max(vol_24h, 1), 10)

        if orderbook:
            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])
            if bids and asks:
                f["bid_ask_spread"] = asks[0][0] - bids[0][0] if asks and bids else 0.02
                bid_vol = sum(s for _, s in bids[:5])
                ask_vol = sum(s for _, s in asks[:5])
                total_vol = bid_vol + ask_vol
                f["book_imbalance"] = (bid_vol - ask_vol) / total_vol if total_vol > 0 else 0
                f["slippage_50usd"] = self._calc_slippage(asks, 50)
                f["slippage_200usd"] = self._calc_slippage(asks, 200)
            else:
                f["bid_ask_spread"] = 0.02
                f["book_imbalance"] = 0
                f["slippage_50usd"] = 0.01
                f["slippage_200usd"] = 0.03
        else:
            f["bid_ask_spread"] = f["spread"]
            f["book_imbalance"] = 0
            f["slippage_50usd"] = 0.01
            f["slippage_200usd"] = 0.03

        # ── Time features (23-28) ──
        hours_left, pct_elapsed = self._calc_time_features(market, event)
        f["hours_to_expiry"] = hours_left
        f["days_to_expiry"] = hours_left / 24
        f["time_decay_factor"] = math.exp(-0.01 * max(0, 720 - hours_left))
        f["urgency_score"] = 1.0 / max(hours_left, 1)
        f["pct_time_elapsed"] = pct_elapsed
        f["is_closing_soon"] = 1.0 if hours_left < 48 else 0.0

        # ── Volatility features (29-34) ──
        f["volatility_24h"] = self._calc_volatility(price_hist, 86400)
        f["volatility_48h"] = self._calc_volatility(price_hist, 172800)
        f["volatility_7d"] = self._calc_volatility(price_hist, 604800)
        vol_short = f["volatility_24h"]
        vol_long = f["volatility_7d"]
        f["volatility_ratio_short_long"] = (
            vol_short / max(vol_long, 0.001)
        )
        f["price_range_24h"] = self._calc_price_range(price_hist, 86400)
        f["max_drawdown_7d"] = self._calc_max_drawdown(price_hist, 604800)

        # ── Market structure (35-40) ──
        f["num_outcomes"] = len(event.markets) if event else 2
        f["is_neg_risk"] = 1.0 if market.neg_risk else 0.0
        f["has_related_markets"] = 1.0 if event and len(event.markets) > 1 else 0.0
        cat = market.category.lower() if market.category else "other"
        f["category_encoded"] = CATEGORY_ENCODING.get(cat, 0.65)
        f["subcategory_encoded"] = 0.5  # placeholder, can be refined
        f["tag_count"] = len(market.tags) if market.tags else 0

        # ── Serial correlation / momentum (41-44) ──
        f["serial_correlation"] = self._calc_serial_correlation(price_hist)
        f["mean_reversion_score"] = -f["serial_correlation"]  # negative serial corr = mean reverting
        f["trend_strength"] = abs(f["price_momentum_1h"]) * abs(f["serial_correlation"])
        f["overreaction_score"] = self._calc_overreaction(price_hist)

        # ── Composite signals (45-47) ──
        # Edge composite: weighted combination of top features
        f["edge_composite"] = (
            f["price_prob_divergence"] * 0.4 +
            f["volume_acceleration"] * 10 * 0.25 +
            abs(f["serial_correlation"]) * 100 * 0.2 +
            f["overreaction_score"] * 0.15
        )

        # Risk/reward
        if yes_price > 0 and yes_price < 1:
            f["risk_reward_ratio"] = (1 - yes_price) / yes_price
        else:
            f["risk_reward_ratio"] = 0

        # Kelly fraction (based on edge composite as proxy for win prob)
        edge_prob = min(0.95, max(0.05, yes_price + f["edge_composite"] / 200))
        if yes_price > 0:
            odds = (1 - yes_price) / yes_price
            q = 1 - edge_prob
            kelly = (edge_prob * odds - q) / odds if odds > 0 else 0
            f["kelly_fraction"] = max(0, min(0.5, kelly * 0.25))
        else:
            f["kelly_fraction"] = 0

        return MarketFeatures(
            condition_id=market.condition_id,
            question=market.question,
            category=cat,
            features=f,
            computed_at=now,
        )

    def score_market(self, features: MarketFeatures) -> float:
        """
        Score a market 0-100 based on feature vector.

        Uses the research-validated feature importance weights.
        Not a trained model — rule-based scoring aligned with XGBoost findings.
        """
        f = features.features
        score = 0

        # Feature 1: Price-Probability Divergence (weight: 35%)
        div = f.get("price_prob_divergence", 0)
        if div > 15:
            score += 35  # 84% accuracy threshold
        elif div > 10:
            score += 25
        elif div > 5:
            score += 15
        else:
            score += div * 2

        # Feature 2: Volume Acceleration (weight: 25%)
        va = f.get("volume_acceleration", 0)
        if va > 5:
            score += 25  # massive spike
        elif va > 3:
            score += 20
        elif va > 1.5:
            score += 12
        else:
            score += va * 5

        # Feature 3: Serial Correlation (weight: 15%)
        sc = f.get("serial_correlation", 0)
        # Negative serial corr in politics = overreaction = opportunity
        if sc < -0.3:
            score += 15
        elif sc < -0.15:
            score += 10
        elif abs(sc) > 0.3:
            score += 8

        # Category accuracy bonus (weight: 10%)
        cat_accuracy = f.get("category_encoded", 0.65)
        score += (cat_accuracy - 0.5) * 33  # maps 0.5-0.85 to 0-11.5

        # Time decay (weight: 8%)
        closing = f.get("is_closing_soon", 0)
        if closing:
            score += 8  # near-expiry markets have clearer signals

        # Liquidity quality (weight: 7%)
        liq = f.get("liquidity", 0)
        if liq > 50000:
            score += 7
        elif liq > 10000:
            score += 5
        elif liq > 5000:
            score += 3

        return min(100, max(0, score))

    def rank_markets(self, features_list: List[MarketFeatures]) -> List[Tuple[MarketFeatures, float]]:
        """Score and rank all markets. Returns sorted list of (features, score)."""
        scored = [(f, self.score_market(f)) for f in features_list]
        scored.sort(key=lambda x: -x[1])
        return scored

    # ─── Private helpers ─────────────────────────────────────────

    def _calc_momentum(self, history: List[Tuple[float, float]],
                        window_secs: float) -> float:
        """Price momentum over a time window."""
        if len(history) < 2:
            return 0
        now = time.time()
        cutoff = now - window_secs
        past_prices = [p for t, p in history if t >= cutoff]
        if len(past_prices) < 2:
            return 0
        return (past_prices[-1] / past_prices[0]) - 1 if past_prices[0] > 0 else 0

    def _calc_volume_acceleration(self, condition_id: str,
                                    current_vol: float) -> float:
        """Ratio of recent volume to historical average."""
        hist = self._volume_history.get(condition_id, [])
        if len(hist) < 2:
            return 1.0
        avg = sum(v for _, v in hist[:-1]) / len(hist[:-1]) if len(hist) > 1 else 1
        return current_vol / max(avg, 1)

    def _calc_volume_trend(self, condition_id: str) -> float:
        """Volume trend: positive = increasing, negative = decreasing."""
        hist = self._volume_history.get(condition_id, [])
        if len(hist) < 3:
            return 0
        recent = [v for _, v in hist[-3:]]
        if recent[0] <= 0:
            return 0
        return (recent[-1] / recent[0]) - 1

    def _calc_slippage(self, asks: List[Tuple[float, float]],
                        amount_usd: float) -> float:
        """Estimate slippage for a given order size."""
        if not asks:
            return 0.05
        remaining = amount_usd
        weighted = 0.0
        filled = 0.0
        for price, size in asks:
            if remaining <= 0:
                break
            fill = min(remaining, price * size)
            weighted += price * fill
            filled += fill
            remaining -= fill
        if filled <= 0:
            return 0.05
        avg_price = weighted / filled
        return abs(avg_price - asks[0][0]) / max(asks[0][0], 0.01)

    def _calc_time_features(self, market: Market,
                             event: Event) -> Tuple[float, float]:
        """Returns (hours_to_expiry, pct_time_elapsed)."""
        now = time.time()
        end_str = market.end_date or (event.end_date if event else None)
        start_str = event.start_date if event else None

        hours_left = 720  # default 30 days
        pct_elapsed = 0.5

        if end_str:
            try:
                from datetime import datetime
                end_ts = datetime.fromisoformat(
                    end_str.replace("Z", "+00:00")
                ).timestamp()
                hours_left = max(0, (end_ts - now) / 3600)
            except (ValueError, TypeError):
                pass

        if start_str and end_str:
            try:
                start_ts = datetime.fromisoformat(
                    start_str.replace("Z", "+00:00")
                ).timestamp()
                end_ts = datetime.fromisoformat(
                    end_str.replace("Z", "+00:00")
                ).timestamp()
                total = end_ts - start_ts
                elapsed = now - start_ts
                pct_elapsed = max(0, min(1, elapsed / total)) if total > 0 else 0.5
            except (ValueError, TypeError):
                pass

        return hours_left, pct_elapsed

    def _calc_volatility(self, history: List[Tuple[float, float]],
                          window_secs: float) -> float:
        """Standard deviation of returns over a window."""
        if len(history) < 3:
            return 0
        now = time.time()
        cutoff = now - window_secs
        prices = [p for t, p in history if t >= cutoff]
        if len(prices) < 3:
            return 0
        returns = [
            (prices[i] - prices[i-1]) / max(prices[i-1], 0.01)
            for i in range(1, len(prices))
        ]
        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
        return math.sqrt(variance)

    def _calc_price_range(self, history: List[Tuple[float, float]],
                           window_secs: float) -> float:
        """High-low range in a time window."""
        if not history:
            return 0
        now = time.time()
        cutoff = now - window_secs
        prices = [p for t, p in history if t >= cutoff]
        if not prices:
            return 0
        return max(prices) - min(prices)

    def _calc_max_drawdown(self, history: List[Tuple[float, float]],
                            window_secs: float) -> float:
        """Maximum drawdown from peak in a window."""
        if not history:
            return 0
        now = time.time()
        cutoff = now - window_secs
        prices = [p for t, p in history if t >= cutoff]
        if len(prices) < 2:
            return 0
        peak = prices[0]
        max_dd = 0
        for p in prices[1:]:
            if p > peak:
                peak = p
            dd = (peak - p) / max(peak, 0.01)
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def _calc_serial_correlation(self, history: List[Tuple[float, float]]) -> float:
        """
        Serial correlation of price changes.
        
        Negative = mean-reverting (overreaction detected).
        58% of political markets show negative serial correlation.
        """
        if len(history) < 10:
            return 0
        prices = [p for _, p in history[-50:]]
        if len(prices) < 5:
            return 0
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        if len(changes) < 4:
            return 0
        n = len(changes)
        mean_c = sum(changes) / n
        var_c = sum((c - mean_c) ** 2 for c in changes) / n
        if var_c < 1e-10:
            return 0
        # Lag-1 autocorrelation
        cov = sum(
            (changes[i] - mean_c) * (changes[i-1] - mean_c)
            for i in range(1, n)
        ) / (n - 1)
        return cov / var_c

    def _calc_overreaction(self, history: List[Tuple[float, float]]) -> float:
        """
        Overreaction score: large moves that tend to reverse.
        
        High score = market likely overreacted, mean reversion opportunity.
        """
        if len(history) < 5:
            return 0
        prices = [p for _, p in history[-20:]]
        if len(prices) < 5:
            return 0
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        if len(changes) < 3:
            return 0
        # Count reversals after large moves
        large_threshold = 0.03  # 3 cent move
        reversals = 0
        large_moves = 0
        for i in range(1, len(changes)):
            if abs(changes[i-1]) > large_threshold:
                large_moves += 1
                # Did it reverse?
                if changes[i] * changes[i-1] < 0:
                    reversals += 1
        if large_moves == 0:
            return 0
        return (reversals / large_moves) * abs(changes[-1]) * 100
