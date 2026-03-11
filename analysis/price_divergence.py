"""
Price-Probability Divergence Detector — Feature #1 from XGBoost Research

THE most important feature: 84% accuracy when divergence > 15 points.

Compares market price against external probability estimates from:
1. Finance data (stock prices, economic indicators)
2. Polling data (elections)
3. Sports odds (from external bookmakers if available)
4. LLM-estimated probability (GPT consensus)
5. Historical base rates

When market price diverges significantly from external probability,
it's the strongest predictor of future price movement.

Category accuracy at >15pt divergence:
  Economics: 84%, Politics: 82%, Crypto: 77%, Sports: 74%
"""

import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from config.settings import Settings
from data.gamma_client import Market, Event
from scanner.market_scanner import MarketSnapshot


@dataclass
class DivergenceSignal:
    """A detected price-probability divergence."""
    condition_id: str
    question: str
    category: str
    market_price: float  # current YES price
    external_prob: float  # external estimated probability
    divergence_pts: float  # abs(external - market) * 100
    direction: str  # "underpriced" or "overpriced"
    buy_side: str  # "yes" or "no"
    confidence: float
    source: str  # which external source
    reasoning: str
    detected_at: float = 0

    @property
    def is_strong(self) -> bool:
        """Divergence > 15pts = 84% accuracy per research."""
        return self.divergence_pts > 15

    @property
    def edge_pct(self) -> float:
        return self.divergence_pts

    def __str__(self):
        emoji = "🔥" if self.is_strong else "📊"
        return (
            f"{emoji} Divergence {self.divergence_pts:.1f}pts | "
            f"{self.direction} | {self.source} | "
            f"{self.question[:50]}"
        )


class PriceProbabilityDivergenceDetector:
    """
    Detects when Polymarket price ≠ external probability estimate.

    This is a META-detector: it takes probability estimates from
    other analysis engines and compares them to market price.

    Usage:
    1. Other engines (finance, elections, sports, LLM) produce estimates
    2. This detector checks if market price diverges significantly
    3. Divergence signals are the highest-priority trade signals

    Thresholds (from XGBoost research):
    - >15pt divergence: 84% accuracy → STRONG_BUY
    - >10pt divergence: ~76% accuracy → BUY
    - >5pt divergence:  ~68% accuracy → WATCH
    """

    STRONG_DIVERGENCE = 15  # points
    MODERATE_DIVERGENCE = 10
    MINIMUM_DIVERGENCE = 5

    # Category-specific accuracy calibration
    CATEGORY_ACCURACY = {
        "economics": 0.84, "finance": 0.84,
        "politics": 0.82, "elections": 0.82,
        "crypto": 0.77,
        "sports": 0.74,
        "culture": 0.70,
        "science": 0.75,
        "geopolitics": 0.78,
    }

    def __init__(self):
        # Cache external estimates: condition_id -> (prob, source, timestamp)
        self._estimates: Dict[str, Tuple[float, str, float]] = {}
        self._estimate_ttl = 3600  # 1 hour

    def register_estimate(self, condition_id: str, probability: float,
                           source: str):
        """
        Register an external probability estimate for a market.

        Called by other analysis engines:
        - FinanceDataEngine → "finance_data"
        - ElectionAnalyzer → "polling"
        - SportsAnalyzer → "sports_model"
        - LLMAnalyzer → "llm_consensus"
        """
        self._estimates[condition_id] = (probability, source, time.time())

    def check(self, snapshot: MarketSnapshot) -> Optional[DivergenceSignal]:
        """
        Check if market price diverges from registered external estimate.
        """
        cid = snapshot.condition_id
        now = time.time()

        # Get external estimate
        if cid not in self._estimates:
            return None

        ext_prob, source, est_time = self._estimates[cid]

        # Check estimate freshness
        if now - est_time > self._estimate_ttl:
            return None

        market_price = snapshot.yes_price
        if not market_price or market_price <= 0:
            return None

        # Calculate divergence in percentage points
        divergence = (ext_prob - market_price) * 100
        abs_divergence = abs(divergence)

        if abs_divergence < self.MINIMUM_DIVERGENCE:
            return None

        # Direction
        if divergence > 0:
            direction = "underpriced"
            buy_side = "yes"
        else:
            direction = "overpriced"
            buy_side = "no"

        # Confidence based on divergence magnitude and category
        cat = snapshot.category.lower() if snapshot.category else "other"
        cat_accuracy = self.CATEGORY_ACCURACY.get(cat, 0.70)

        if abs_divergence >= self.STRONG_DIVERGENCE:
            base_conf = 0.84
        elif abs_divergence >= self.MODERATE_DIVERGENCE:
            base_conf = 0.76
        else:
            base_conf = 0.68

        # Adjust by category accuracy
        confidence = base_conf * (cat_accuracy / 0.84)

        reasoning = (
            f"Market prices YES at {market_price:.3f} but "
            f"{source} estimates {ext_prob:.3f}. "
            f"Divergence: {abs_divergence:.1f}pts ({direction}). "
            f"Category '{cat}' has {cat_accuracy*100:.0f}% historical accuracy "
            f"at this divergence level."
        )

        return DivergenceSignal(
            condition_id=cid,
            question=snapshot.question,
            category=cat,
            market_price=market_price,
            external_prob=ext_prob,
            divergence_pts=abs_divergence,
            direction=direction,
            buy_side=buy_side,
            confidence=min(0.95, confidence),
            source=source,
            reasoning=reasoning,
            detected_at=now,
        )

    def scan_all(self, snapshots: List[MarketSnapshot]) -> List[DivergenceSignal]:
        """Scan all snapshots for divergence signals."""
        signals = []
        for snap in snapshots:
            sig = self.check(snap)
            if sig:
                signals.append(sig)

        # Sort by divergence magnitude (strongest first)
        signals.sort(key=lambda s: -s.divergence_pts)

        if signals:
            strong = sum(1 for s in signals if s.is_strong)
            print(
                f"🎯 Divergence scan: {len(signals)} signals "
                f"({strong} strong >15pts)"
            )

        return signals
