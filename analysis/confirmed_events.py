"""
Confirmed Event Detector — "Free Money" Finder

Detects markets where the outcome is near-certain but the price hasn't
reached 99¢ yet. These are the easiest, lowest-risk trades.

Examples:
- Election completed, winner announced → market at 93¢ (should be 99¢)
- Official data released confirming outcome → market at 88¢
- Court ruling done, sentence confirmed → market still at 91¢
- Dead person can't win an election → "No" at 7¢

Strategy: Buy the confirmed outcome, hold until settlement at $1.00.
Profit = (1.00 - buy_price) per share.
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass

from config.settings import Settings
from analysis.news_llm import LLMAnalyzer, get_llm_analyzer
from scanner.market_scanner import MarketSnapshot


@dataclass
class ConfirmedSignal:
    """A market where the outcome appears confirmed."""
    snapshot: MarketSnapshot
    confirmed_outcome: str  # "yes" or "no"
    confirmed_token_id: str
    current_price: float
    target_price: float  # 1.0 (settlement)
    edge_percent: float  # (1.0 - current_price) * 100
    confidence: float  # 0-1 how certain is confirmation
    reason: str
    evidence: List[str]
    signal_time: float = 0

    @property
    def expected_profit_pct(self) -> float:
        """Expected profit per $ invested."""
        if self.current_price <= 0:
            return 0
        return (1.0 - self.current_price) / self.current_price * 100


class ConfirmedEventDetector:
    """
    Finds markets where outcomes are effectively decided but not yet settled.
    
    Detection methods:
    1. Price-based: One outcome at 90%+ but not 98%+ (gap = opportunity)
    2. Time-based: Event end date passed, prices resolving
    3. News-based: LLM confirms outcome from news
    4. Multi-market consistency: Cross-referencing related markets
    """

    def __init__(self, llm: Optional[LLMAnalyzer] = None):
        self.llm = llm or get_llm_analyzer()
        self._processed: Dict[str, float] = {}  # condition_id -> last_checked

    async def scan(self, snapshots: List[MarketSnapshot]) -> List[ConfirmedSignal]:
        """
        Scan all snapshots for confirmed-outcome opportunities.
        """
        signals = []

        for snap in snapshots:
            # Skip recently processed
            last = self._processed.get(snap.key, 0)
            if time.time() - last < 600:  # check each market every 10 min
                continue

            signal = await self._check_market(snap)
            if signal:
                signals.append(signal)
            self._processed[snap.key] = time.time()

        # Sort by edge (highest first)
        return sorted(signals, key=lambda s: s.edge_percent, reverse=True)

    async def _check_market(self, snap: MarketSnapshot) -> Optional[ConfirmedSignal]:
        """Check if a market has a confirmed outcome."""
        market = snap.market
        yes_price = market.best_yes_price
        no_price = market.best_no_price

        # ── Method 1: High-probability but not fully priced ──
        # If YES is 90-97%, there might be free money
        threshold = Settings.CONFIRMED_EVENT_THRESHOLD

        if yes_price >= threshold and yes_price < 0.98:
            # Potential confirmed YES — verify with LLM
            signal = await self._verify_with_llm(snap, "yes", yes_price)
            if signal:
                return signal

        # If NO side is cheap (YES < 0.10), might be a confirmed NO
        if yes_price <= (1 - threshold) and yes_price > 0.02:
            signal = await self._verify_with_llm(snap, "no", 1 - yes_price)
            if signal:
                return signal

        # ── Method 2: Time-expired market still trading ──
        if snap.hours_until_close is not None and snap.hours_until_close < 0:
            # Market past end date — outcome likely decided
            if yes_price >= 0.80 and yes_price < 0.98:
                return self._build_signal(snap, "yes", yes_price, 0.85,
                    "Market past end date, YES at high probability",
                    ["End date passed", f"YES price: {yes_price:.2f}"])
            if yes_price <= 0.20 and yes_price > 0.02:
                return self._build_signal(snap, "no", 1 - yes_price, 0.85,
                    "Market past end date, NO at high probability",
                    ["End date passed", f"NO price: {1-yes_price:.2f}"])

        return None

    async def _verify_with_llm(self, snap: MarketSnapshot,
                                outcome: str, current_price: float) -> Optional[ConfirmedSignal]:
        """Ask LLM if the outcome is confirmed."""
        if not Settings.has_llm():
            # Without LLM, use conservative price threshold only
            if current_price >= 0.95:
                return self._build_signal(snap, outcome, current_price, 0.7,
                    f"Very high probability ({current_price:.0%}) — likely confirmed",
                    [f"Price: {current_price:.2f}", "No LLM verification available"])
            return None

        analysis = await self.llm.analyze_market(
            snap.market.question,
            additional_context=(
                f"Current market price: YES={snap.market.best_yes_price:.2f}, "
                f"NO={snap.market.best_no_price:.2f}. "
                f"Category: {snap.category}. "
                f"Event: {snap.event.title}. "
                f"QUESTION: Has this event's outcome been officially confirmed/decided?"
            )
        )

        if not analysis:
            return None

        # LLM confirms with high confidence
        if analysis.confidence >= 0.7:
            llm_agrees_yes = (outcome == "yes" and analysis.probability_estimate >= 0.85)
            llm_agrees_no = (outcome == "no" and analysis.probability_estimate <= 0.15)

            if llm_agrees_yes or llm_agrees_no:
                return self._build_signal(
                    snap, outcome, current_price,
                    analysis.confidence,
                    analysis.reasoning,
                    analysis.key_facts
                )

        return None

    def _build_signal(self, snap: MarketSnapshot, outcome: str,
                      current_price: float, confidence: float,
                      reason: str, evidence: List[str]) -> ConfirmedSignal:
        """Build a ConfirmedSignal."""
        token = snap.market.yes_token if outcome == "yes" else snap.market.no_token
        token_id = token.token_id if token else ""

        return ConfirmedSignal(
            snapshot=snap,
            confirmed_outcome=outcome,
            confirmed_token_id=token_id,
            current_price=current_price,
            target_price=1.0,
            edge_percent=(1.0 - current_price) * 100,
            confidence=confidence,
            reason=reason,
            evidence=evidence,
            signal_time=time.time(),
        )
