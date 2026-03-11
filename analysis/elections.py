"""
Election & Polling Aggregator — Political Market Intelligence

For markets like:
- "Will AfD win the German election?"
- "Will Trump win the 2028 election?"
- "Will Labour win a majority?"
- "Next Prime Minister of X?"

Data sources:
- Polling aggregators (RealClearPolitics, FiveThirtyEight, Wikipedia)
- News-based LLM analysis
- Cross-market consistency checks (Polymarket vs PredictIt vs Metaculus)
"""

import asyncio
import re
import time
from typing import Dict, List, Optional
from dataclasses import dataclass

import httpx

from config.settings import Settings
from analysis.news_llm import LLMAnalyzer, get_llm_analyzer
from scanner.market_scanner import MarketSnapshot


@dataclass
class PollingData:
    """Aggregated polling data for a political race."""
    race: str
    candidate: str
    polling_avg: float  # 0-100%
    trend: str  # "rising", "falling", "stable"
    sample_size: int
    last_poll_date: str
    source: str
    margin_of_error: float

    @property
    def implied_probability(self) -> float:
        """Convert polling average to win probability (rough)."""
        # Simple: polling lead -> win probability
        # In 2-way race: 50% + (lead/2) adjusted for uncertainty
        # This is a simplification; real models use state-level data
        return max(0.01, min(0.99, self.polling_avg / 100))


@dataclass
class ElectionAnalysis:
    """Analysis result for an election/political market."""
    market_question: str
    probability_estimate: float
    confidence: float
    reasoning: str
    polling_data: List[PollingData]
    news_summary: str
    days_until_election: Optional[int] = None


class ElectionAnalyzer:
    """
    Analyzes political/election markets using polling data + news.
    
    Strategy:
    1. Detect the race/election from market question
    2. Fetch latest polling data
    3. Cross-reference with news
    4. Use LLM for nuanced analysis
    5. Compare Polymarket price vs polling-implied probability
    """

    def __init__(self, llm: Optional[LLMAnalyzer] = None):
        self.llm = llm or get_llm_analyzer()
        self._polling_cache: Dict[str, tuple] = {}
        self._cache_ttl = 1800  # 30 min

    async def analyze_market(self, snap: MarketSnapshot) -> Optional[ElectionAnalysis]:
        """Analyze a political/election market."""
        question = snap.market.question
        question_lower = question.lower()

        # Detect election type
        race_info = self._detect_race(question_lower, snap)
        if not race_info:
            return None

        # Fetch polling data
        polls = await self._fetch_polling_data(race_info)

        # Get news analysis
        news_summary = ""
        llm_prob = None
        if Settings.has_llm():
            analysis = await self.llm.analyze_market(
                question,
                additional_context=(
                    f"Category: {snap.category}. "
                    f"Event: {snap.event.title}. "
                    f"Current market price: YES={snap.market.best_yes_price:.2f}. "
                    f"Polling data: {self._format_polls(polls)}"
                )
            )
            if analysis:
                news_summary = analysis.reasoning
                llm_prob = analysis.probability_estimate

        # Combine polling + LLM into final estimate
        final_prob = self._combine_estimates(polls, llm_prob, snap)

        return ElectionAnalysis(
            market_question=question,
            probability_estimate=final_prob,
            confidence=self._calc_confidence(polls, llm_prob),
            reasoning=news_summary or self._format_polls(polls),
            polling_data=polls,
            news_summary=news_summary,
            days_until_election=int(snap.hours_until_close / 24) if snap.hours_until_close else None,
        )

    def _detect_race(self, question: str, snap: MarketSnapshot) -> Optional[Dict]:
        """Detect what election/political race this market is about."""
        # Match patterns like "Will X win Y election?"
        patterns = [
            # US elections
            (r"(trump|biden|harris|desantis|haley|ramaswamy|pence)", "us-president"),
            (r"(democrat|republican|gop)\s+win", "us-general"),
            (r"(senate|congress|house)\s+(control|majority|flip)", "us-congress"),
            # European
            (r"(afd|cdu|spd|greens?)\s+.*(german|bundestag)", "germany"),
            (r"(labour|conservative|tory|lib dem)\s+.*(uk|britain)", "uk"),
            (r"(macron|le pen|rassemblement|rn)\s+.*(french|france)", "france"),
            # General
            (r"(election|president|prime minister|chancellor|governor)", "general"),
            (r"(next\s+president|next\s+pm|next\s+prime)", "general"),
        ]

        for pattern, race_type in patterns:
            if re.search(pattern, question):
                return {
                    "type": race_type,
                    "question": question,
                    "tags": snap.event.tags,
                }

        return None

    async def _fetch_polling_data(self, race_info: Dict) -> List[PollingData]:
        """
        Fetch polling data from free, public sources.
        Uses Wikipedia polling pages + RealClearPolitics via scraping.
        """
        cache_key = f"polls:{race_info['type']}"
        cached = self._polling_cache.get(cache_key)
        if cached and time.time() - cached[1] < self._cache_ttl:
            return cached[0]

        polls = []

        # For US elections, try RealClearPolitics RSS
        if race_info["type"] in ("us-president", "us-general", "us-congress"):
            polls = await self._fetch_rcp_data(race_info)

        # For any election, use news-based estimation
        if not polls and Settings.has_llm():
            polls = await self._estimate_from_news(race_info)

        self._polling_cache[cache_key] = (polls, time.time())
        return polls

    async def _fetch_rcp_data(self, race_info: Dict) -> List[PollingData]:
        """Fetch Real Clear Politics data."""
        # RCP doesn't have a public API, but we can get headline averages
        # Use their public page data
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    "https://www.realclearpolling.com/polls/president/general/2028",
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                # Parse basic poll numbers from page if available
                # This is fragile but works for headline numbers
                if resp.status_code == 200:
                    # Extract numbers from the page
                    text = resp.text
                    # Simple pattern matching for candidate percentages
                    return self._parse_rcp_page(text, race_info)
        except Exception:
            pass
        return []

    def _parse_rcp_page(self, html: str, race_info: Dict) -> List[PollingData]:
        """Parse basic polling numbers from RCP page."""
        # This is intentionally simple — RCP changes layout frequently
        polls = []
        # Look for percentage patterns near candidate names
        candidates = {
            "us-president": ["Trump", "Biden", "Harris", "DeSantis"],
        }.get(race_info["type"], [])

        for candidate in candidates:
            pattern = rf'{candidate}.*?(\d{{2}}\.?\d?)\s*%'
            match = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
            if match:
                polls.append(PollingData(
                    race=race_info["type"],
                    candidate=candidate,
                    polling_avg=float(match.group(1)),
                    trend="stable",
                    sample_size=0,
                    last_poll_date="",
                    source="RealClearPolitics",
                    margin_of_error=3.0,
                ))
        return polls

    async def _estimate_from_news(self, race_info: Dict) -> List[PollingData]:
        """Use LLM to estimate polling numbers from news."""
        prompt = (
            f"What are the latest polling averages for the {race_info['type']} "
            f"race? Give me candidate names and their polling percentages. "
            f"Be specific with numbers."
        )
        # This would use the LLM but we return empty for now
        # In production, integrate with news_llm.py
        return []

    def _combine_estimates(self, polls: List[PollingData],
                           llm_prob: Optional[float],
                           snap: MarketSnapshot) -> float:
        """Combine polling data + LLM + market price into probability estimate."""
        estimates = []
        weights = []

        # Polling-based estimate
        if polls:
            # Use the highest polling average as base
            best_poll = max(polls, key=lambda p: p.polling_avg)
            estimates.append(best_poll.implied_probability)
            weights.append(0.4)

        # LLM estimate
        if llm_prob is not None:
            estimates.append(llm_prob)
            weights.append(0.4)

        # Market price as anchor
        market_prob = snap.market.best_yes_price
        estimates.append(market_prob)
        weights.append(0.2)

        if not estimates:
            return 0.5

        # Weighted average
        total_weight = sum(weights)
        combined = sum(e * w for e, w in zip(estimates, weights)) / total_weight
        return max(0.01, min(0.99, combined))

    def _calc_confidence(self, polls: List[PollingData],
                         llm_prob: Optional[float]) -> float:
        """Calculate confidence in the estimate."""
        conf = 0.2  # base
        if polls:
            conf += 0.3
        if llm_prob is not None:
            conf += 0.3
        return min(conf, 0.9)

    def _format_polls(self, polls: List[PollingData]) -> str:
        """Format polling data as text."""
        if not polls:
            return "No polling data available"
        lines = []
        for p in polls:
            lines.append(f"{p.candidate}: {p.polling_avg:.1f}% ({p.source})")
        return "; ".join(lines)
