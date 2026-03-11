"""
Market Scanner — Continuously discovers & categorizes ALL Polymarket events.

Runs on a timer, builds a live catalogue of tradeable markets grouped by:
- Category (politics, finance, crypto, sports, geopolitics, culture, science)
- Edge type (confirmed event, news-driven, data-arb, mispricing, etc.)
- Urgency (ending soon, just opened, high volume spike)

Feeds results to analysis engines for edge detection.
"""

import asyncio
import time
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone

from config.settings import Settings
from data.gamma_client import GammaClient, Event, Market, get_gamma_client


@dataclass
class MarketSnapshot:
    """A market with scanner metadata."""
    market: Market
    event: Event
    category: str
    subcategory: str  # e.g. "election", "fed-rate", "mrbeast", "esports"
    hours_until_close: Optional[float] = None
    volume_24h: float = 0
    price_momentum: float = 0  # change in yes_price over last scan
    scan_time: float = 0

    @property
    def key(self) -> str:
        return self.market.condition_id


# ═══════════════════════════════════════════════════════════════
# CATEGORY DETECTION
# ═══════════════════════════════════════════════════════════════

CATEGORY_RULES = {
    "politics": {
        "tags": ["politics", "elections", "election", "us-politics", "trump", "biden",
                 "democrat", "republican", "congress", "senate", "governor", "presidency",
                 "european-politics", "uk-politics", "german-election", "french-election"],
        "keywords": ["election", "president", "prime minister", "governor", "senator",
                      "congress", "parliament", "vote", "ballot", "polling",
                      "trump", "biden", "afd", "labour", "tory", "macron", "starmer"]
    },
    "finance": {
        "tags": ["finance", "fed", "federal-reserve", "interest-rates", "inflation",
                 "commodities", "oil", "gold", "silver", "stocks", "market-cap",
                 "treasury", "bonds", "gdp", "economic"],
        "keywords": ["fed rate", "interest rate", "fomc", "cpi", "inflation",
                      "crude oil", "gold price", "silver", "s&p 500", "nasdaq",
                      "market cap", "largest company", "treasury", "gdp", "recession",
                      "apple", "nvidia", "microsoft", "tesla stock", "ipo"]
    },
    "crypto": {
        "tags": ["crypto", "bitcoin", "ethereum", "solana", "btc", "eth",
                 "cryptocurrency", "defi", "nft", "web3", "altcoin"],
        "keywords": ["bitcoin", "ethereum", "solana", "btc", "eth", "crypto",
                      "token", "defi", "blockchain", "halving", "etf approval"]
    },
    "geopolitics": {
        "tags": ["geopolitics", "war", "conflict", "iran", "russia", "ukraine",
                 "china", "taiwan", "north-korea", "venezuela", "middle-east",
                 "military", "sanctions", "nato"],
        "keywords": ["war", "invasion", "ceasefire", "sanctions", "military",
                      "nuclear", "missile", "troops", "nato", "un security",
                      "iran", "russia", "ukraine", "china", "taiwan", "israel"]
    },
    "sports": {
        "tags": ["sports", "nba", "nfl", "mlb", "nhl", "cricket", "football",
                 "soccer", "tennis", "ufc", "mma", "f1", "golf", "esports",
                 "valorant", "cs2", "lol", "league-of-legends"],
        "keywords": ["championship", "playoffs", "finals", "world cup", "super bowl",
                      "wimbledon", "world series", "stanley cup", "grand prix",
                      "valorant", "counter-strike", "league of legends", "esports",
                      "nba", "nfl", "mlb", "premier league", "champions league"]
    },
    "culture": {
        "tags": ["culture", "entertainment", "celebrity", "movies", "music",
                 "oscar", "grammy", "tech", "ai", "youtube", "social-media",
                 "elon-musk", "nobel", "space", "spacex"],
        "keywords": ["mrbeast", "youtube", "views", "subscribers", "twitter",
                      "elon musk", "oscar", "grammy", "emmy", "box office",
                      "nobel prize", "spacex", "space", "nasa", "ai model",
                      "chatgpt", "openai", "google", "apple", "meta",
                      "viral", "tiktok", "instagram"]
    },
    "science": {
        "tags": ["science", "weather", "climate", "health", "pandemic",
                 "temperature", "hurricane", "earthquake"],
        "keywords": ["temperature", "hurricane", "earthquake", "pandemic",
                      "vaccine", "outbreak", "climate", "global warming",
                      "meteor", "asteroid", "eclipse"]
    },
}

# Subcategory detection for finer-grained routing
SUBCATEGORY_RULES = {
    "election": ["election", "president", "governor", "senator", "prime minister",
                 "parliament", "vote", "ballot", "afd", "labour", "tory"],
    "fed-rate": ["fed rate", "fomc", "interest rate", "federal reserve"],
    "inflation": ["cpi", "inflation", "pce"],
    "commodities": ["crude oil", "gold price", "silver", "oil price", "wti", "brent"],
    "market-cap": ["largest company", "market cap", "most valuable"],
    "ipo": ["ipo", "goes public", "direct listing"],
    "crypto-price": ["bitcoin price", "btc price", "eth price", "crypto etf"],
    "mrbeast": ["mrbeast", "mr beast", "beast video"],
    "youtube": ["youtube", "video views", "subscriber"],
    "esports": ["valorant", "counter-strike", "cs2", "league of legends", "lol esports",
                "dota", "overwatch"],
    "elon": ["elon musk", "musk tweet", "x.com", "doge"],
    "ai-tech": ["openai", "chatgpt", "gpt-5", "claude", "gemini", "ai model"],
    "space": ["spacex", "nasa", "launch", "starship", "moon", "mars"],
    "nobel": ["nobel prize", "nobel peace"],
    "war": ["war", "invasion", "ceasefire", "military strike"],
}


def detect_category(event: Event, market: Market) -> str:
    """Detect primary category from event/market text and tags."""
    all_tags = set(t.lower() for t in (event.tags + market.tags))
    text = f"{event.title} {event.description} {market.question}".lower()

    # Check tags first (most reliable)
    for cat, rules in CATEGORY_RULES.items():
        if any(t in all_tags for t in rules["tags"]):
            return cat

    # Fallback to keyword matching
    for cat, rules in CATEGORY_RULES.items():
        if any(kw in text for kw in rules["keywords"]):
            return cat

    return "other"


def detect_subcategory(event: Event, market: Market) -> str:
    """Detect subcategory for fine-grained analysis routing."""
    text = f"{event.title} {market.question}".lower()
    for subcat, keywords in SUBCATEGORY_RULES.items():
        if any(kw in text for kw in keywords):
            return subcat
    return ""


def hours_until(date_str: Optional[str]) -> Optional[float]:
    """How many hours until a date."""
    if not date_str:
        return None
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta = dt - now
        return delta.total_seconds() / 3600
    except (ValueError, TypeError):
        return None


class MarketScanner:
    """
    Continuous market scanner.
    
    Discovers all active Polymarket events, categorizes them,
    and maintains a live catalogue for analysis engines.
    """

    def __init__(self, gamma: Optional[GammaClient] = None):
        self.gamma = gamma or get_gamma_client()
        self.catalogue: Dict[str, MarketSnapshot] = {}  # condition_id -> snapshot
        self._previous_prices: Dict[str, float] = {}  # condition_id -> last yes_price
        self._last_scan: float = 0
        self._scan_count = 0

    @property
    def snapshots(self) -> Dict[str, MarketSnapshot]:
        """Alias for catalogue — used by AutoTrader."""
        return self.catalogue

    async def scan(self) -> List[MarketSnapshot]:
        """
        Run one scan cycle. Fetches all active events, categorizes markets.
        Returns list of new/updated snapshots.
        """
        t0 = time.time()
        events = await self.gamma.get_all_active_events(
            min_liquidity=Settings.MIN_LIQUIDITY_USD,
            min_volume=Settings.MIN_VOLUME_24H,
        )

        snapshots = []
        seen_ids: Set[str] = set()

        for event in events:
            for market in event.markets:
                if not market.condition_id or market.closed:
                    continue
                if not market.accepting_orders:
                    continue
                if market.condition_id in seen_ids:
                    continue
                seen_ids.add(market.condition_id)

                # Skip markets with no valid tokens
                if not market.tokens or not any(t.token_id for t in market.tokens):
                    continue

                cat = detect_category(event, market)
                subcat = detect_subcategory(event, market)
                hrs = hours_until(market.end_date or event.end_date)

                # Price momentum
                prev_price = self._previous_prices.get(market.condition_id, 0)
                cur_price = market.best_yes_price
                momentum = cur_price - prev_price if prev_price > 0 else 0
                self._previous_prices[market.condition_id] = cur_price

                snap = MarketSnapshot(
                    market=market,
                    event=event,
                    category=cat,
                    subcategory=subcat,
                    hours_until_close=hrs,
                    volume_24h=market.volume,
                    price_momentum=momentum,
                    scan_time=time.time(),
                )
                snapshots.append(snap)
                self.catalogue[market.condition_id] = snap

        # Clean stale entries
        stale_keys = [k for k in self.catalogue if k not in seen_ids]
        for k in stale_keys:
            del self.catalogue[k]

        elapsed = time.time() - t0
        self._last_scan = time.time()
        self._scan_count += 1
        print(f"📡 Scan #{self._scan_count}: {len(snapshots)} markets in {len(events)} events ({elapsed:.1f}s)")

        return snapshots

    def get_by_category(self, category: str) -> List[MarketSnapshot]:
        """Get all snapshots in a category."""
        return [s for s in self.catalogue.values() if s.category == category]

    def get_by_subcategory(self, subcat: str) -> List[MarketSnapshot]:
        """Get all snapshots by subcategory."""
        return [s for s in self.catalogue.values() if s.subcategory == subcat]

    def get_closing_soon(self, hours: float = 48) -> List[MarketSnapshot]:
        """Get markets closing within N hours."""
        results = []
        for s in self.catalogue.values():
            if s.hours_until_close is not None and 0 < s.hours_until_close <= hours:
                results.append(s)
        return sorted(results, key=lambda s: s.hours_until_close or 999)

    def get_high_volume(self, min_volume: float = 50000) -> List[MarketSnapshot]:
        """Get markets with volume above threshold."""
        return sorted(
            [s for s in self.catalogue.values() if s.volume_24h >= min_volume],
            key=lambda s: s.volume_24h, reverse=True
        )

    def get_movers(self, min_move: float = 0.03) -> List[MarketSnapshot]:
        """Get markets with biggest price moves since last scan."""
        return sorted(
            [s for s in self.catalogue.values() if abs(s.price_momentum) >= min_move],
            key=lambda s: abs(s.price_momentum), reverse=True
        )

    def summary(self) -> Dict[str, int]:
        """Category counts."""
        counts: Dict[str, int] = {}
        for s in self.catalogue.values():
            counts[s.category] = counts.get(s.category, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    async def run_loop(self, callback=None):
        """
        Run continuous scan loop.
        callback(snapshots) is called after each scan if provided.
        """
        print(f"🔄 Scanner starting | interval={Settings.SCAN_INTERVAL_SECONDS}s")
        while True:
            try:
                snapshots = await self.scan()
                if callback:
                    await callback(snapshots)
            except Exception as e:
                print(f"⚠️ Scan error: {e}")
            await asyncio.sleep(Settings.SCAN_INTERVAL_SECONDS)
