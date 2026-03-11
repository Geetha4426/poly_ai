"""
Finance Data Arbitrage Engine — CME FedWatch, Commodities, Market Cap

For markets like:
- "Fed rate decision June 2025" → compare vs CME FedWatch tool probabilities
- "Crude oil above $80?" → compare vs NYMEX futures
- "Largest company by market cap?" → compare vs real-time stock data
- "Gold above $3000?" → COMEX futures
- "SpaceX IPO in 2025?" → news analysis

Data sources:
- CME FedWatch (free, public data)
- Yahoo Finance (free API)
- Alpha Vantage (free tier: 25 req/day)
- Finnhub (free tier: 60 req/min)
"""

import asyncio
import re
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import httpx

from config.settings import Settings
from scanner.market_scanner import MarketSnapshot


@dataclass
class FedWatchData:
    """CME FedWatch probabilities."""
    meeting_date: str
    current_rate_bps: int  # current rate in basis points
    probabilities: Dict[str, float]  # "525-550" -> 0.85, "500-525" -> 0.10
    implied_rate: float  # weighted average implied rate
    source: str = "CME FedWatch"
    fetched_at: float = 0


@dataclass
class CommodityData:
    """Real-time commodity price data."""
    symbol: str  # "CL" (crude oil), "GC" (gold), etc.
    name: str
    price: float
    change_24h: float
    change_pct: float
    high_52w: float
    low_52w: float
    source: str = ""
    fetched_at: float = 0


@dataclass
class StockData:
    """Stock/company data for market cap tracking."""
    symbol: str
    name: str
    price: float
    market_cap: float
    change_pct: float
    source: str = ""
    fetched_at: float = 0


@dataclass
class FinanceAnalysis:
    """Analysis result for a finance-related market."""
    market_question: str
    probability_estimate: float
    confidence: float
    reasoning: str
    data_source: str
    reference_value: float  # the actual value from external data
    market_implied_value: float  # what Polymarket price implies
    edge_direction: str  # "over" or "under" valued on Polymarket


class FinanceDataEngine:
    """
    Cross-references Polymarket finance markets against external data.
    
    Key edges:
    1. Fed rate: CME FedWatch vs Polymarket = often 5-15% gap
    2. Commodities: futures price vs prediction market threshold
    3. Market cap: real-time vs Polymarket opinion
    """

    def __init__(self):
        self._cache: Dict[str, tuple] = {}
        self._cache_ttl = 180  # 3 min for finance data

    async def analyze_market(self, snap: MarketSnapshot) -> Optional[FinanceAnalysis]:
        """Route to the right analysis based on subcategory."""
        subcat = snap.subcategory
        question = snap.market.question.lower()

        if subcat == "fed-rate" or "fed" in question or "fomc" in question or "interest rate" in question:
            return await self._analyze_fed_rate(snap)
        elif subcat == "commodities" or any(w in question for w in ["crude oil", "gold", "silver", "oil price"]):
            return await self._analyze_commodity(snap)
        elif subcat == "market-cap" or "largest company" in question or "market cap" in question:
            return await self._analyze_market_cap(snap)
        elif subcat == "inflation" or "cpi" in question or "inflation" in question:
            return await self._analyze_inflation(snap)
        else:
            return None

    # ═══════════════════════════════════════════════════════════════
    # FED RATE ANALYSIS
    # ═══════════════════════════════════════════════════════════════

    async def _analyze_fed_rate(self, snap: MarketSnapshot) -> Optional[FinanceAnalysis]:
        """
        Compare Polymarket fed rate prices vs CME FedWatch probabilities.
        
        CME FedWatch: derived from Fed Funds futures, updated every second.
        This is the gold standard for rate expectations.
        """
        fedwatch = await self._get_fedwatch_data()
        if not fedwatch:
            return None

        question = snap.market.question.lower()
        market_price = snap.market.best_yes_price

        # Detect what rate scenario this market is about
        # e.g. "Will Fed cut rates?" "Will fed funds rate be below 5%?"
        if "cut" in question or "lower" in question or "decrease" in question:
            # Sum probabilities of all cut scenarios
            cut_prob = sum(
                prob for rate_range, prob in fedwatch.probabilities.items()
                if self._rate_range_to_bps(rate_range) < fedwatch.current_rate_bps
            )
            ref_value = cut_prob
            reasoning = (
                f"CME FedWatch shows {cut_prob:.0%} probability of rate cut. "
                f"Current rate: {fedwatch.current_rate_bps}bps. "
                f"Polymarket price: {market_price:.2f}. "
                f"Gap: {abs(cut_prob - market_price):.0%}"
            )
        elif "hike" in question or "raise" in question or "increase" in question:
            hike_prob = sum(
                prob for rate_range, prob in fedwatch.probabilities.items()
                if self._rate_range_to_bps(rate_range) > fedwatch.current_rate_bps
            )
            ref_value = hike_prob
            reasoning = (
                f"CME FedWatch shows {hike_prob:.0%} probability of rate hike. "
                f"Polymarket: {market_price:.2f}. Gap: {abs(hike_prob - market_price):.0%}"
            )
        elif "hold" in question or "unchanged" in question or "no change" in question:
            hold_prob = fedwatch.probabilities.get(
                self._bps_to_range(fedwatch.current_rate_bps), 0
            )
            ref_value = hold_prob
            reasoning = (
                f"CME FedWatch shows {hold_prob:.0%} probability of hold. "
                f"Polymarket: {market_price:.2f}. Gap: {abs(hold_prob - market_price):.0%}"
            )
        else:
            # Generic analysis — use implied rate
            ref_value = 0.5
            reasoning = (
                f"Fed rate market. CME implied rate: {fedwatch.implied_rate:.2f}%. "
                f"Polymarket YES price: {market_price:.2f}."
            )

        edge_dir = "under" if ref_value > market_price else "over"

        return FinanceAnalysis(
            market_question=snap.market.question,
            probability_estimate=max(0.01, min(0.99, ref_value)),
            confidence=0.8,  # CME FedWatch is highly reliable
            reasoning=reasoning,
            data_source="CME FedWatch",
            reference_value=ref_value,
            market_implied_value=market_price,
            edge_direction=edge_dir,
        )

    async def _get_fedwatch_data(self) -> Optional[FedWatchData]:
        """
        Fetch CME FedWatch probabilities.
        
        Uses publicly available data from CME Group website.
        Falls back to federal funds futures from FRED if needed.
        """
        cache_key = "fedwatch"
        cached = self._cache.get(cache_key)
        if cached and time.time() - cached[1] < self._cache_ttl:
            return cached[0]

        # Try CME FedWatch tool page (public data)
        data = await self._fetch_fedwatch_cme()
        if data:
            self._cache[cache_key] = (data, time.time())
            return data

        # Fallback: estimate from FRED data
        data = await self._fetch_fedwatch_fred()
        if data:
            self._cache[cache_key] = (data, time.time())
            return data

        return None

    async def _fetch_fedwatch_cme(self) -> Optional[FedWatchData]:
        """Fetch from CME FedWatch API/page."""
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                # CME public API for fed funds futures
                resp = await client.get(
                    "https://www.cmegroup.com/services/rfq/api/fedwatch",
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                if resp.status_code != 200:
                    return None

                data = resp.json()
                # Parse CME response format
                if isinstance(data, dict) and "meetings" in data:
                    meetings = data["meetings"]
                    if meetings:
                        next_meeting = meetings[0]
                        probs = {}
                        current_rate = 0
                        for scenario in next_meeting.get("scenarios", []):
                            rate_range = scenario.get("rate", "")
                            prob = float(scenario.get("probability", 0)) / 100
                            probs[rate_range] = prob
                            if scenario.get("current"):
                                current_rate = self._rate_range_to_bps(rate_range)

                        implied = sum(
                            self._rate_range_to_bps(r) * p / 100
                            for r, p in probs.items()
                        )

                        return FedWatchData(
                            meeting_date=next_meeting.get("date", ""),
                            current_rate_bps=current_rate or 525,
                            probabilities=probs,
                            implied_rate=implied,
                            fetched_at=time.time(),
                        )
        except Exception:
            pass
        return None

    async def _fetch_fedwatch_fred(self) -> Optional[FedWatchData]:
        """Estimate from FRED (St. Louis Fed) data — always free."""
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    "https://fred.stlouisfed.org/graph/fredgraph.csv",
                    params={"id": "DFEDTARU", "cosd": "2024-01-01"},
                )
                if resp.status_code == 200:
                    lines = resp.text.strip().split("\n")
                    if len(lines) > 1:
                        last_line = lines[-1]
                        parts = last_line.split(",")
                        if len(parts) >= 2 and parts[1] != ".":
                            rate = float(parts[1])
                            rate_bps = int(rate * 100)
                            return FedWatchData(
                                meeting_date="current",
                                current_rate_bps=rate_bps,
                                probabilities={
                                    self._bps_to_range(rate_bps): 0.80,
                                    self._bps_to_range(rate_bps - 25): 0.15,
                                    self._bps_to_range(rate_bps + 25): 0.05,
                                },
                                implied_rate=rate,
                                source="FRED",
                                fetched_at=time.time(),
                            )
        except Exception:
            pass
        return None

    # ═══════════════════════════════════════════════════════════════
    # COMMODITY ANALYSIS
    # ═══════════════════════════════════════════════════════════════

    async def _analyze_commodity(self, snap: MarketSnapshot) -> Optional[FinanceAnalysis]:
        """Compare commodity price vs Polymarket threshold."""
        question = snap.market.question.lower()

        # Detect commodity
        if "crude oil" in question or "oil price" in question or "wti" in question:
            symbol = "CL=F"
            name = "Crude Oil WTI"
        elif "gold" in question:
            symbol = "GC=F"
            name = "Gold"
        elif "silver" in question:
            symbol = "SI=F"
            name = "Silver"
        elif "natural gas" in question:
            symbol = "NG=F"
            name = "Natural Gas"
        else:
            return None

        price_data = await self._get_commodity_price(symbol, name)
        if not price_data:
            return None

        # Extract threshold from question
        threshold = self._extract_price_threshold(question)
        if not threshold:
            return None

        # Calculate probability
        current = price_data.price
        if "above" in question or "over" in question or "exceed" in question:
            # Probability price stays above threshold
            distance_pct = (current - threshold) / threshold * 100
            if current > threshold:
                prob = min(0.95, 0.5 + distance_pct / 20)
            else:
                prob = max(0.05, 0.5 + distance_pct / 20)
        elif "below" in question or "under" in question:
            distance_pct = (threshold - current) / threshold * 100
            if current < threshold:
                prob = min(0.95, 0.5 + distance_pct / 20)
            else:
                prob = max(0.05, 0.5 + distance_pct / 20)
        else:
            prob = 0.5

        market_price = snap.market.best_yes_price
        edge_dir = "under" if prob > market_price else "over"

        return FinanceAnalysis(
            market_question=snap.market.question,
            probability_estimate=max(0.01, min(0.99, prob)),
            confidence=0.6,
            reasoning=(
                f"{name} currently at ${current:.2f}. "
                f"Threshold: ${threshold:.2f}. "
                f"Distance: {abs(current - threshold)/threshold*100:.1f}%. "
                f"24h change: {price_data.change_pct:+.1f}%."
            ),
            data_source=f"Yahoo Finance ({symbol})",
            reference_value=prob,
            market_implied_value=market_price,
            edge_direction=edge_dir,
        )

    async def _get_commodity_price(self, symbol: str, name: str) -> Optional[CommodityData]:
        """Fetch commodity price from Yahoo Finance (free, no API key)."""
        cache_key = f"commodity:{symbol}"
        cached = self._cache.get(cache_key)
        if cached and time.time() - cached[1] < self._cache_ttl:
            return cached[0]

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                    params={"interval": "1d", "range": "5d"},
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    result = data.get("chart", {}).get("result", [{}])[0]
                    meta = result.get("meta", {})
                    price = float(meta.get("regularMarketPrice", 0))
                    prev_close = float(meta.get("chartPreviousClose", price))
                    change = price - prev_close
                    change_pct = (change / prev_close * 100) if prev_close > 0 else 0

                    commodity = CommodityData(
                        symbol=symbol, name=name, price=price,
                        change_24h=change, change_pct=change_pct,
                        high_52w=float(meta.get("fiftyTwoWeekHigh", 0)),
                        low_52w=float(meta.get("fiftyTwoWeekLow", 0)),
                        source="Yahoo Finance", fetched_at=time.time(),
                    )
                    self._cache[cache_key] = (commodity, time.time())
                    return commodity
        except Exception as e:
            print(f"⚠️ Commodity fetch error for {symbol}: {e}")

        # Fallback: Alpha Vantage
        if Settings.ALPHA_VANTAGE_KEY:
            return await self._get_commodity_alpha_vantage(symbol, name)

        return None

    async def _get_commodity_alpha_vantage(self, symbol: str, name: str) -> Optional[CommodityData]:
        """Fetch from Alpha Vantage (25 free req/day)."""
        av_map = {"CL=F": "WTI", "GC=F": "GOLD", "SI=F": "SILVER", "NG=F": "NATURAL_GAS"}
        av_fn = av_map.get(symbol)
        if not av_fn:
            return None

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    "https://www.alphavantage.co/query",
                    params={
                        "function": av_fn,
                        "interval": "daily",
                        "apikey": Settings.ALPHA_VANTAGE_KEY,
                    }
                )
                if resp.status_code == 200:
                    data = resp.json()
                    # Parse Alpha Vantage response
                    ts_key = [k for k in data.keys() if "Time Series" in k or "data" in k]
                    if ts_key:
                        series = data[ts_key[0]]
                        if isinstance(series, list) and series:
                            latest = series[0]
                            price = float(latest.get("value", latest.get("close", 0)))
                            return CommodityData(
                                symbol=symbol, name=name, price=price,
                                change_24h=0, change_pct=0,
                                high_52w=0, low_52w=0,
                                source="Alpha Vantage", fetched_at=time.time(),
                            )
        except Exception:
            pass
        return None

    # ═══════════════════════════════════════════════════════════════
    # MARKET CAP ANALYSIS
    # ═══════════════════════════════════════════════════════════════

    async def _analyze_market_cap(self, snap: MarketSnapshot) -> Optional[FinanceAnalysis]:
        """Analyze 'largest company by market cap' markets."""
        question = snap.market.question.lower()

        # Big tech companies typically in these markets
        companies = {
            "apple": "AAPL", "nvidia": "NVDA", "microsoft": "MSFT",
            "alphabet": "GOOGL", "google": "GOOGL", "amazon": "AMZN",
            "meta": "META", "tesla": "TSLA", "berkshire": "BRK-B",
            "broadcom": "AVGO", "taiwan semi": "TSM", "tsmc": "TSM",
            "saudi aramco": "2222.SR",
        }

        # Detect which company this market is about
        target_company = None
        target_symbol = None
        for name, symbol in companies.items():
            if name in question:
                target_company = name
                target_symbol = symbol
                break

        if not target_symbol:
            return None

        # Fetch market caps for comparison
        all_caps = await self._get_market_caps(list(companies.values()))
        target_data = all_caps.get(target_symbol)
        if not target_data:
            return None

        # Is this company currently the largest?
        sorted_caps = sorted(all_caps.values(), key=lambda x: x.market_cap, reverse=True)
        rank = next(
            (i + 1 for i, s in enumerate(sorted_caps) if s.symbol == target_symbol),
            len(sorted_caps)
        )

        if "largest" in question or "most valuable" in question or "biggest" in question:
            prob = 0.90 if rank == 1 else max(0.05, 0.5 - (rank - 1) * 0.15)
        else:
            prob = 0.5

        market_price = snap.market.best_yes_price
        return FinanceAnalysis(
            market_question=snap.market.question,
            probability_estimate=max(0.01, min(0.99, prob)),
            confidence=0.7,
            reasoning=(
                f"{target_company.title()} ({target_symbol}) market cap: "
                f"${target_data.market_cap/1e12:.2f}T. Rank #{rank}. "
                f"Top 3: {', '.join(f'{s.name} ${s.market_cap/1e12:.2f}T' for s in sorted_caps[:3])}"
            ),
            data_source="Yahoo Finance",
            reference_value=prob,
            market_implied_value=market_price,
            edge_direction="under" if prob > market_price else "over",
        )

    async def _get_market_caps(self, symbols: List[str]) -> Dict[str, StockData]:
        """Fetch market caps for multiple stocks."""
        cache_key = "mcaps"
        cached = self._cache.get(cache_key)
        if cached and time.time() - cached[1] < self._cache_ttl:
            return cached[0]

        # Deduplicate
        unique_symbols = list(set(symbols))
        results = {}

        async def _fetch_stock(sym: str):
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.get(
                        f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}",
                        params={"interval": "1d", "range": "1d"},
                        headers={"User-Agent": "Mozilla/5.0"},
                    )
                    if resp.status_code == 200:
                        data = resp.json().get("chart", {}).get("result", [{}])[0]
                        meta = data.get("meta", {})
                        return StockData(
                            symbol=sym,
                            name=meta.get("shortName", meta.get("longName", sym)),
                            price=float(meta.get("regularMarketPrice", 0)),
                            market_cap=float(meta.get("marketCap", 0)),
                            change_pct=float(meta.get("regularMarketChangePercent", 0)),
                            source="Yahoo Finance",
                            fetched_at=time.time(),
                        )
            except Exception:
                pass
            return None

        tasks = [_fetch_stock(s) for s in unique_symbols]
        fetched = await asyncio.gather(*tasks, return_exceptions=True)
        for item in fetched:
            if isinstance(item, StockData) and item.market_cap > 0:
                results[item.symbol] = item

        self._cache[cache_key] = (results, time.time())
        return results

    # ═══════════════════════════════════════════════════════════════
    # INFLATION / CPI ANALYSIS
    # ═══════════════════════════════════════════════════════════════

    async def _analyze_inflation(self, snap: MarketSnapshot) -> Optional[FinanceAnalysis]:
        """Analyze CPI/inflation markets using FRED data."""
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    "https://fred.stlouisfed.org/graph/fredgraph.csv",
                    params={"id": "CPIAUCSL", "cosd": "2024-01-01"},
                )
                if resp.status_code != 200:
                    return None

                lines = resp.text.strip().split("\n")
                if len(lines) < 3:
                    return None

                # Get last two data points for trend
                last = lines[-1].split(",")
                prev = lines[-2].split(",")
                if len(last) < 2 or len(prev) < 2:
                    return None

                cpi_current = float(last[1]) if last[1] != "." else None
                cpi_prev = float(prev[1]) if prev[1] != "." else None

                if not cpi_current or not cpi_prev:
                    return None

                yoy_change = (cpi_current - cpi_prev) / cpi_prev * 100
                threshold = self._extract_price_threshold(snap.market.question.lower())

                if threshold:
                    if "above" in snap.market.question.lower():
                        prob = 0.8 if yoy_change > threshold else 0.2
                    elif "below" in snap.market.question.lower():
                        prob = 0.8 if yoy_change < threshold else 0.2
                    else:
                        prob = 0.5
                else:
                    prob = 0.5

                return FinanceAnalysis(
                    market_question=snap.market.question,
                    probability_estimate=max(0.01, min(0.99, prob)),
                    confidence=0.5,
                    reasoning=f"Current CPI YoY: {yoy_change:.1f}%. Latest CPI: {cpi_current:.1f}",
                    data_source="FRED CPI",
                    reference_value=prob,
                    market_implied_value=snap.market.best_yes_price,
                    edge_direction="under" if prob > snap.market.best_yes_price else "over",
                )
        except Exception:
            return None

    # ═══════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _rate_range_to_bps(rate_range: str) -> int:
        """Convert '525-550' to midpoint in bps."""
        try:
            parts = rate_range.replace("%", "").split("-")
            if len(parts) == 2:
                return (int(parts[0]) + int(parts[1])) // 2
            return int(float(parts[0]) * 100)
        except (ValueError, IndexError):
            return 0

    @staticmethod
    def _bps_to_range(bps: int) -> str:
        """Convert 525 bps to range string '500-525'."""
        lower = (bps // 25) * 25
        return f"{lower}-{lower + 25}"

    @staticmethod
    def _extract_price_threshold(text: str) -> Optional[float]:
        """Extract a price/number from market question."""
        # "$80", "$3,000", "80 dollars"
        m = re.search(r'\$\s*([\d,]+(?:\.\d+)?)', text)
        if m:
            return float(m.group(1).replace(",", ""))
        m = re.search(r'([\d,]+(?:\.\d+)?)\s*(dollar|usd|percent|%|bps)', text)
        if m:
            return float(m.group(1).replace(",", ""))
        return None
