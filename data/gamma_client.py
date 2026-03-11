"""
Polymarket Gamma API Client — Universal Market Discovery

Fetches events, markets, tokens across ALL categories.
No hardcoded sports/crypto filters — fully generic.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

import httpx

from config.settings import Settings


@dataclass
class Token:
    """A tradeable outcome token."""
    token_id: str
    outcome: str  # "Yes", "No", "India", "Trump", etc.
    price: float  # 0.0 - 1.0


@dataclass
class Market:
    """A single market (binary or multi-outcome)."""
    condition_id: str
    question: str
    description: str
    tokens: List[Token]
    volume: float
    liquidity: float
    end_date: Optional[str] = None
    market_slug: str = ""
    neg_risk: bool = False
    active: bool = True
    closed: bool = False
    accepting_orders: bool = True
    # Populated from parent event
    event_slug: str = ""
    event_title: str = ""
    category: str = ""
    tags: List[str] = field(default_factory=list)

    @property
    def yes_token(self) -> Optional[Token]:
        for t in self.tokens:
            if t.outcome.lower() == "yes":
                return t
        return self.tokens[0] if self.tokens else None

    @property
    def no_token(self) -> Optional[Token]:
        for t in self.tokens:
            if t.outcome.lower() == "no":
                return t
        return self.tokens[1] if len(self.tokens) > 1 else None

    @property
    def best_yes_price(self) -> float:
        t = self.yes_token
        return t.price if t else 0.5

    @property
    def best_no_price(self) -> float:
        t = self.no_token
        return t.price if t else 0.5


@dataclass
class Event:
    """A Polymarket event containing one or more markets."""
    event_id: str
    title: str
    slug: str
    description: str
    category: str
    tags: List[str]
    markets: List[Market]
    volume: float
    liquidity: float
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    neg_risk: bool = False
    image: str = ""


class GammaClient:
    """Async client for the Gamma API — discovers all Polymarket events/markets."""

    def __init__(self):
        self.base_url = Settings.GAMMA_API_URL
        self._cache: Dict[str, Any] = {}
        self._cache_ts: Dict[str, float] = {}
        self._cache_ttl = 60  # 60s cache

    def _cache_get(self, key: str) -> Optional[Any]:
        ts = self._cache_ts.get(key, 0)
        if time.time() - ts < self._cache_ttl:
            return self._cache.get(key)
        return None

    def _cache_set(self, key: str, value: Any):
        self._cache[key] = value
        self._cache_ts[key] = time.time()

    async def _fetch(self, endpoint: str, params: Optional[Dict] = None,
                     timeout: int = 30) -> Optional[Any]:
        """Fetch JSON from Gamma API with retry."""
        url = f"{self.base_url}{endpoint}"
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.get(url, params=params)
                    if resp.status_code == 200:
                        return resp.json()
                    if resp.status_code in (400, 404):
                        return None
                    if resp.status_code in (429, 500, 502, 503):
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return None
            except (httpx.TimeoutException, httpx.ConnectError):
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                print(f"⚠️ Gamma fetch error: {e}")
                return None
        return None

    def _parse_tokens(self, raw_market: Dict) -> List[Token]:
        """Extract tokens from a raw market response."""
        tokens = []

        # Method 1: from 'tokens' array
        raw_tokens = raw_market.get("tokens", [])
        if raw_tokens:
            for t in raw_tokens:
                tokens.append(Token(
                    token_id=t.get("token_id", ""),
                    outcome=t.get("outcome", ""),
                    price=float(t.get("price", 0) or 0)
                ))
            return tokens

        # Method 2: from clobTokenIds + outcomes + outcomePrices
        try:
            clob_ids = json.loads(raw_market.get("clobTokenIds", "[]"))
            outcomes = json.loads(raw_market.get("outcomes", "[]"))
            prices_raw = raw_market.get("outcomePrices", "[]")
            prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
        except (json.JSONDecodeError, TypeError):
            clob_ids, outcomes, prices = [], [], []

        for i in range(len(clob_ids)):
            tokens.append(Token(
                token_id=clob_ids[i] if i < len(clob_ids) else "",
                outcome=outcomes[i] if i < len(outcomes) else f"Outcome {i}",
                price=float(prices[i]) if i < len(prices) else 0.0
            ))

        return tokens

    def _parse_market(self, raw: Dict, event_data: Optional[Dict] = None) -> Market:
        """Parse a raw market dict into a Market dataclass."""
        tokens = self._parse_tokens(raw)
        tags_raw = raw.get("tags", [])
        tags = []
        if isinstance(tags_raw, list):
            for t in tags_raw:
                if isinstance(t, dict):
                    tags.append(t.get("slug", t.get("label", "")))
                elif isinstance(t, str):
                    tags.append(t)

        event_title = ""
        event_slug = ""
        category = ""
        event_tags = []
        if event_data:
            event_title = event_data.get("title", "")
            event_slug = event_data.get("slug", "")
            category = event_data.get("category", "")
            raw_etags = event_data.get("tags", [])
            if isinstance(raw_etags, list):
                for t in raw_etags:
                    if isinstance(t, dict):
                        event_tags.append(t.get("slug", t.get("label", "")))
                    elif isinstance(t, str):
                        event_tags.append(t)

        return Market(
            condition_id=raw.get("conditionId", raw.get("condition_id", "")),
            question=raw.get("question", ""),
            description=raw.get("description", ""),
            tokens=tokens,
            volume=float(raw.get("volume", 0) or 0),
            liquidity=float(raw.get("liquidity", 0) or 0),
            end_date=raw.get("endDate", raw.get("end_date_iso", "")),
            market_slug=raw.get("market_slug", raw.get("slug", "")),
            neg_risk=bool(raw.get("negRisk", False)),
            active=bool(raw.get("active", True)),
            closed=bool(raw.get("closed", False)),
            accepting_orders=bool(raw.get("acceptingOrders",
                                          raw.get("accepting_orders", True))),
            event_slug=event_slug,
            event_title=event_title,
            category=category,
            tags=tags or event_tags,
        )

    def _parse_event(self, raw: Dict) -> Event:
        """Parse a raw event dict into an Event dataclass."""
        raw_markets = raw.get("markets", [])
        markets = [self._parse_market(m, event_data=raw) for m in raw_markets]

        tags_raw = raw.get("tags", [])
        tags = []
        if isinstance(tags_raw, list):
            for t in tags_raw:
                if isinstance(t, dict):
                    tags.append(t.get("slug", t.get("label", "")))
                elif isinstance(t, str):
                    tags.append(t)

        return Event(
            event_id=raw.get("id", ""),
            title=raw.get("title", ""),
            slug=raw.get("slug", ""),
            description=raw.get("description", ""),
            category=raw.get("category", ""),
            tags=tags,
            markets=markets,
            volume=float(raw.get("volume", 0) or 0),
            liquidity=float(raw.get("liquidity", 0) or 0),
            start_date=raw.get("startDate", raw.get("startDatetime", "")),
            end_date=raw.get("endDate", raw.get("endDatetime", "")),
            neg_risk=bool(raw.get("negRisk", False)),
            image=raw.get("image", ""),
        )

    # ═══════════════════════════════════════════════════════════════
    # PUBLIC API — Event Discovery
    # ═══════════════════════════════════════════════════════════════

    async def get_events(self, tag: str = "", limit: int = 100,
                         active: bool = True, closed: bool = False,
                         offset: int = 0) -> List[Event]:
        """
        Fetch events from Gamma API.

        Args:
            tag: Filter by tag slug (e.g. 'politics', 'finance', 'crypto')
            limit: Max results per page
            active: Only active events
            closed: Include closed events
            offset: Pagination offset
        """
        cache_key = f"events:{tag}:{limit}:{offset}:{active}:{closed}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        params = {"limit": limit, "offset": offset}
        if active:
            params["active"] = "true"
        if closed:
            params["closed"] = "true"
        if tag:
            params["tag_slug"] = tag

        data = await self._fetch("/events", params)
        if not data or not isinstance(data, list):
            return []

        events = [self._parse_event(e) for e in data]
        self._cache_set(cache_key, events)
        return events

    async def get_all_active_events(self, tags: Optional[List[str]] = None,
                                     min_liquidity: float = 0,
                                     min_volume: float = 0) -> List[Event]:
        """
        Fetch ALL active events across multiple tags.
        Deduplicates by event_id.
        """
        if tags is None:
            tags = Settings.SCAN_TAGS

        all_events: Dict[str, Event] = {}

        # Fetch all tags in parallel
        async def fetch_tag(tag: str):
            events = []
            offset = 0
            while True:
                batch = await self.get_events(tag=tag, limit=100, offset=offset)
                if not batch:
                    break
                events.extend(batch)
                if len(batch) < 100:
                    break
                offset += 100
            return events

        tasks = [fetch_tag(t) for t in tags]
        # Also fetch without tag to get uncategorized
        tasks.append(fetch_tag(""))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                print(f"⚠️ Tag fetch error: {result}")
                continue
            for event in result:
                if event.event_id not in all_events:
                    all_events[event.event_id] = event

        # Filter by liquidity/volume
        filtered = []
        for event in all_events.values():
            if event.liquidity >= min_liquidity and event.volume >= min_volume:
                filtered.append(event)

        return sorted(filtered, key=lambda e: e.volume, reverse=True)

    async def get_event_by_slug(self, slug: str) -> Optional[Event]:
        """Fetch single event by slug."""
        data = await self._fetch("/events", {"slug": slug, "limit": 1})
        if data and isinstance(data, list) and len(data) > 0:
            return self._parse_event(data[0])
        return None

    async def get_event_by_id(self, event_id: str) -> Optional[Event]:
        """Fetch single event by ID."""
        data = await self._fetch("/events", {"id": event_id, "limit": 1})
        if data and isinstance(data, list) and len(data) > 0:
            return self._parse_event(data[0])
        return None

    async def get_market(self, condition_id: str) -> Optional[Market]:
        """Fetch single market by condition ID."""
        cache_key = f"market:{condition_id}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        data = await self._fetch("/markets", {"condition_id": condition_id})
        if not data:
            # Try direct endpoint
            data = await self._fetch(f"/markets/{condition_id}")
        if data:
            if isinstance(data, list) and data:
                m = self._parse_market(data[0])
            elif isinstance(data, dict):
                m = self._parse_market(data)
            else:
                return None
            self._cache_set(cache_key, m)
            return m
        return None

    async def search_events(self, query: str, limit: int = 20) -> List[Event]:
        """
        Search events by text query.
        Gamma API supports _q parameter for text search on some endpoints.
        Falls back to client-side filtering.
        """
        # Try server-side search first
        data = await self._fetch("/events", {
            "_q": query, "limit": limit, "active": "true"
        })
        if data and isinstance(data, list) and len(data) > 0:
            return [self._parse_event(e) for e in data]

        # Fallback: fetch all and filter
        all_events = await self.get_all_active_events()
        query_lower = query.lower()
        results = []
        for event in all_events:
            text = f"{event.title} {event.description} {' '.join(event.tags)}".lower()
            if query_lower in text:
                results.append(event)
        return results[:limit]

    async def get_tags(self) -> List[Dict]:
        """Fetch all available tags from Gamma API."""
        data = await self._fetch("/tags")
        if data and isinstance(data, list):
            return data
        return []

    async def get_sports(self) -> List[Dict]:
        """Fetch sports series from Gamma API."""
        data = await self._fetch("/sports")
        if data and isinstance(data, list):
            return data
        return []


# Singleton
_gamma_client: Optional[GammaClient] = None


def get_gamma_client() -> GammaClient:
    global _gamma_client
    if _gamma_client is None:
        _gamma_client = GammaClient()
    return _gamma_client
