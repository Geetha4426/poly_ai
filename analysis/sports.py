"""
Sports & Esports Analysis Engine

For markets like:
- "Who wins NBA Finals 2025?"
- "Will Team X win [Valorant/CS2/LoL] tournament?"
- "Premier League winner?"
- "Cricket: India vs Australia winner?"

Uses:
- ESPN/Sports data APIs (free)
- Elo/Glicko rating systems
- Head-to-head records
- Recent form analysis
- LLM for contextual factors (injuries, roster changes)
"""

import asyncio
import time
from typing import Dict, List, Optional
from dataclasses import dataclass

import httpx

from config.settings import Settings
from analysis.news_llm import LLMAnalyzer, get_llm_analyzer
from scanner.market_scanner import MarketSnapshot


@dataclass
class TeamStats:
    """Team/player statistics for probability estimation."""
    name: str
    sport: str
    wins: int
    losses: int
    draws: int = 0
    elo_rating: float = 1500
    recent_form: str = ""  # "WWLWW"
    key_players: List[str] = None

    def __post_init__(self):
        if self.key_players is None:
            self.key_players = []

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses + self.draws
        return self.wins / total if total > 0 else 0.5

    @property
    def recent_win_rate(self) -> float:
        """Win rate from recent form string."""
        if not self.recent_form:
            return self.win_rate
        wins = self.recent_form.upper().count("W")
        total = len(self.recent_form)
        return wins / total if total > 0 else 0.5


@dataclass
class SportsAnalysis:
    """Analysis result for a sports market."""
    market_question: str
    probability_estimate: float
    confidence: float
    reasoning: str
    team_stats: Optional[TeamStats] = None
    opponent_stats: Optional[TeamStats] = None
    data_source: str = ""


# Elo-based expected win probability
def elo_expected(rating_a: float, rating_b: float) -> float:
    """Calculate expected win probability using Elo formula."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


class SportsAnalyzer:
    """
    Analyzes sports and esports markets using public data + LLM.
    
    For sports we don't have direct API data for, we use LLM as the 
    primary analyst, cross-referenced with the market price for calibration.
    """

    def __init__(self, llm: Optional[LLMAnalyzer] = None):
        self.llm = llm or get_llm_analyzer()
        self._cache: Dict[str, tuple] = {}
        self._cache_ttl = 600  # 10 min

    async def analyze_market(self, snap: MarketSnapshot) -> Optional[SportsAnalysis]:
        """Analyze a sports/esports market."""
        question = snap.market.question
        category = snap.subcategory or snap.category

        # Route to specific handler
        if category == "esports":
            return await self._analyze_esports(snap)
        else:
            return await self._analyze_general_sports(snap)

    async def _analyze_esports(self, snap: MarketSnapshot) -> Optional[SportsAnalysis]:
        """Analyze esports markets (Valorant, CS2, LoL, etc.)."""
        question = snap.market.question

        # Esports is hard to get structured data for — use LLM primarily
        if Settings.has_llm():
            analysis = await self.llm.analyze_market(
                question,
                additional_context=(
                    f"This is an esports prediction market. "
                    f"Current price: YES={snap.market.best_yes_price:.2f}. "
                    f"Event: {snap.event.title}. "
                    f"Consider: team rankings, recent tournament results, player roster, "
                    f"head-to-head record, map pool, form."
                ),
                news_query=self._build_esports_query(question),
            )
            if analysis:
                return SportsAnalysis(
                    market_question=question,
                    probability_estimate=analysis.probability_estimate,
                    confidence=min(analysis.confidence, 0.5),  # cap esports confidence
                    reasoning=analysis.reasoning,
                    data_source="LLM + News",
                )

        return None

    async def _analyze_general_sports(self, snap: MarketSnapshot) -> Optional[SportsAnalysis]:
        """Analyze traditional sports using stats APIs + LLM."""
        question = snap.market.question.lower()

        # Try to get structured data first
        stats = await self._fetch_sports_data(snap)

        # Use LLM with stats context
        if Settings.has_llm():
            stats_context = ""
            if stats:
                stats_context = (
                    f"Team stats: {stats.name} - {stats.wins}W/{stats.losses}L, "
                    f"Elo: {stats.elo_rating:.0f}, Recent: {stats.recent_form}"
                )

            analysis = await self.llm.analyze_market(
                snap.market.question,
                additional_context=(
                    f"Sports market. Price: YES={snap.market.best_yes_price:.2f}. "
                    f"Event: {snap.event.title}. {stats_context}"
                ),
            )
            if analysis:
                # Blend LLM with stats if available
                prob = analysis.probability_estimate
                if stats:
                    stats_prob = stats.recent_win_rate
                    prob = 0.6 * prob + 0.4 * stats_prob

                return SportsAnalysis(
                    market_question=snap.market.question,
                    probability_estimate=max(0.01, min(0.99, prob)),
                    confidence=analysis.confidence * 0.7,  # discount for sports uncertainty
                    reasoning=analysis.reasoning,
                    team_stats=stats,
                    data_source="LLM + Stats" if stats else "LLM",
                )

        # Fallback: market price +/- small correction
        return SportsAnalysis(
            market_question=snap.market.question,
            probability_estimate=snap.market.best_yes_price,
            confidence=0.2,
            reasoning="Insufficient data — using market price as estimate",
            data_source="Market Price",
        )

    async def _fetch_sports_data(self, snap: MarketSnapshot) -> Optional[TeamStats]:
        """
        Fetch sports data from free APIs.
        ESPN, TheSportsDB (free), or API-SPORTS (free tier).
        """
        question_lower = snap.market.question.lower()

        # Try TheSportsDB (free, no API key needed)
        teams = self._extract_team_names(question_lower)
        if not teams:
            return None

        team_name = teams[0]
        cache_key = f"sports:{team_name}"
        cached = self._cache.get(cache_key)
        if cached and time.time() - cached[1] < self._cache_ttl:
            return cached[0]

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    "https://www.thesportsdb.com/api/v1/json/3/searchteams.php",
                    params={"t": team_name}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    teams_data = data.get("teams", [])
                    if teams_data:
                        t = teams_data[0]
                        stats = TeamStats(
                            name=t.get("strTeam", team_name),
                            sport=t.get("strSport", ""),
                            wins=0,  # Would need event/season data
                            losses=0,
                            elo_rating=1500,
                        )
                        self._cache[cache_key] = (stats, time.time())
                        return stats
        except Exception:
            pass

        return None

    @staticmethod
    def _extract_team_names(question: str) -> List[str]:
        """Extract team/player names from a question."""
        import re
        # Common patterns: "X vs Y", "Will X win", "X to beat Y"
        vs_match = re.search(r'(\w[\w\s]+?)\s+(?:vs?\.?|versus|against)\s+(\w[\w\s]+?)(?:\?|$)', question)
        if vs_match:
            return [vs_match.group(1).strip(), vs_match.group(2).strip()]

        will_win = re.search(r'will\s+(.+?)\s+win', question)
        if will_win:
            return [will_win.group(1).strip()]

        return []

    @staticmethod
    def _build_esports_query(question: str) -> str:
        """Build search query for esports news."""
        games = {
            "valorant": "Valorant esports",
            "cs2": "Counter-Strike 2 esports",
            "counter-strike": "Counter-Strike esports",
            "league of legends": "LoL esports",
            "lol": "League of Legends esports",
            "dota": "Dota 2 esports",
        }
        question_lower = question.lower()
        for game, query_prefix in games.items():
            if game in question_lower:
                return f"{query_prefix} {question}"
        return f"esports {question}"
