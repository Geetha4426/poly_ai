"""
Narrative Shift Detector — Detects when news changes the thesis

Monitors positions we hold and detects when:
1. Breaking news contradicts our position
2. Sentiment flips (bullish → bearish or vice versa)
3. Key assumption invalidated (e.g., candidate drops out)
4. Market structure change (volume spike, price crash/spike)

Triggers: HOLD, SELL, or BUY MORE recommendations.
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass

from config.settings import Settings
from analysis.news_llm import LLMAnalyzer, get_llm_analyzer
from scanner.market_scanner import MarketSnapshot


@dataclass
class PositionState:
    """Tracked state of a position for narrative monitoring."""
    condition_id: str
    token_id: str
    question: str
    outcome: str  # "yes" or "no"
    entry_price: float
    current_price: float
    size: float
    last_sentiment: str  # "bullish", "bearish", "neutral"
    last_probability_estimate: float
    last_checked: float
    entry_reasoning: str  # why we entered the trade


@dataclass
class NarrativeShift:
    """A detected narrative shift affecting a position."""
    position: PositionState
    shift_type: str  # "breaking_news", "sentiment_flip", "assumption_invalid", "price_crash", "consolidation"
    severity: str  # "critical", "major", "minor"
    direction: str  # "positive" (helps our position) or "negative" (hurts)
    trigger: str  # what caused the shift
    recommendation: str  # "SELL", "HOLD", "BUY_MORE", "REDUCE"
    new_probability: float
    reasoning: str
    detected_at: float = 0

    @property
    def is_urgent(self) -> bool:
        return self.severity == "critical" or (
            self.severity == "major" and self.direction == "negative"
        )


class NarrativeShiftDetector:
    """
    Monitors held positions for narrative shifts.
    
    Runs periodically for each open position:
    1. Check price movement (fast — no API needed)
    2. Check news if price moved significantly or on schedule
    3. LLM analysis if news detected
    4. Generate recommendation
    """

    def __init__(self, llm: Optional[LLMAnalyzer] = None):
        self.llm = llm or get_llm_analyzer()
        self.tracked: Dict[str, PositionState] = {}  # condition_id -> state
        self._check_interval = 600  # 10 min between checks per position

    def track_position(self, condition_id: str, token_id: str,
                       question: str, outcome: str,
                       entry_price: float, size: float,
                       reasoning: str = ""):
        """Start tracking a position for narrative shifts."""
        self.tracked[condition_id] = PositionState(
            condition_id=condition_id,
            token_id=token_id,
            question=question,
            outcome=outcome,
            entry_price=entry_price,
            current_price=entry_price,
            size=size,
            last_sentiment="neutral",
            last_probability_estimate=entry_price,
            last_checked=time.time(),
            entry_reasoning=reasoning,
        )

    def untrack_position(self, condition_id: str):
        """Stop tracking a position."""
        self.tracked.pop(condition_id, None)

    def update_price(self, condition_id: str, new_price: float):
        """Update current price of a tracked position."""
        if condition_id in self.tracked:
            self.tracked[condition_id].current_price = new_price

    async def check_all(self, snapshots: Optional[Dict[str, MarketSnapshot]] = None
                        ) -> List[NarrativeShift]:
        """
        Check all tracked positions for narrative shifts.
        Returns list of detected shifts (empty = all good).
        """
        shifts = []
        now = time.time()

        for cid, pos in list(self.tracked.items()):
            # Skip if checked recently
            if now - pos.last_checked < self._check_interval:
                continue

            # ── Check 1: Price movement ──
            price_shift = self._check_price_movement(pos)
            if price_shift:
                shifts.append(price_shift)
                if price_shift.severity == "critical":
                    continue  # Skip further checks if critical price move

            # ── Check 2: News-based narrative shift ──
            if Settings.has_llm():
                news_shift = await self._check_news_shift(pos)
                if news_shift:
                    shifts.append(news_shift)

            pos.last_checked = now

        return shifts

    def _check_price_movement(self, pos: PositionState) -> Optional[NarrativeShift]:
        """Check for significant price movements."""
        price_change = pos.current_price - pos.entry_price
        pct_change = abs(price_change / pos.entry_price) * 100 if pos.entry_price > 0 else 0

        # Our position direction
        is_long_yes = pos.outcome == "yes"

        # Price moved against us
        if is_long_yes and price_change < -0.10:  # YES dropped 10%+
            severity = "critical" if price_change < -0.20 else "major"
            return NarrativeShift(
                position=pos,
                shift_type="price_crash",
                severity=severity,
                direction="negative",
                trigger=f"YES price dropped {abs(price_change):.0%} from entry",
                recommendation="SELL" if severity == "critical" else "REDUCE",
                new_probability=pos.current_price,
                reasoning=f"Price dropped from {pos.entry_price:.2f} to {pos.current_price:.2f}. Consider cutting losses.",
                detected_at=time.time(),
            )
        elif not is_long_yes and price_change > 0.10:  # We're short YES but it went up
            severity = "critical" if price_change > 0.20 else "major"
            return NarrativeShift(
                position=pos,
                shift_type="price_crash",
                severity=severity,
                direction="negative",
                trigger=f"YES price rose {price_change:.0%} against our NO position",
                recommendation="SELL" if severity == "critical" else "REDUCE",
                new_probability=pos.current_price,
                reasoning=f"Price moved against NO position. From {pos.entry_price:.2f} to {pos.current_price:.2f}.",
                detected_at=time.time(),
            )

        # Price moved strongly in our favor — possible take profit
        if is_long_yes and price_change > 0.15:
            return NarrativeShift(
                position=pos,
                shift_type="consolidation",
                severity="minor",
                direction="positive",
                trigger=f"YES price up {price_change:.0%} from entry",
                recommendation="HOLD" if pos.current_price < 0.90 else "SELL",
                new_probability=pos.current_price,
                reasoning=f"In profit. Price at {pos.current_price:.2f}. Consider taking partial profits.",
                detected_at=time.time(),
            )

        return None

    async def _check_news_shift(self, pos: PositionState) -> Optional[NarrativeShift]:
        """Check if news has shifted the narrative."""
        result = await self.llm.detect_narrative_shift(
            pos.question,
            pos.last_sentiment,
            pos.last_probability_estimate
        )

        if not result or not result.get("shifted"):
            return None

        direction_raw = result.get("direction", "none")
        magnitude = float(result.get("magnitude", 0))
        new_prob = float(result.get("new_probability", pos.current_price))

        # Determine if shift helps or hurts our position
        is_long_yes = pos.outcome == "yes"
        if is_long_yes:
            is_positive = direction_raw == "up"
        else:
            is_positive = direction_raw == "down"

        severity = "critical" if magnitude > 0.15 else "major" if magnitude > 0.08 else "minor"

        if is_positive:
            recommendation = "BUY_MORE" if severity == "major" else "HOLD"
        else:
            recommendation = "SELL" if severity == "critical" else "REDUCE" if severity == "major" else "HOLD"

        # Update tracked state
        pos.last_sentiment = "bullish" if direction_raw == "up" else "bearish" if direction_raw == "down" else "neutral"
        pos.last_probability_estimate = new_prob

        return NarrativeShift(
            position=pos,
            shift_type="breaking_news" if magnitude > 0.10 else "sentiment_flip",
            severity=severity,
            direction="positive" if is_positive else "negative",
            trigger=result.get("trigger", "News event"),
            recommendation=recommendation,
            new_probability=new_prob,
            reasoning=result.get("trigger", "Narrative shift detected via news analysis"),
            detected_at=time.time(),
        )
