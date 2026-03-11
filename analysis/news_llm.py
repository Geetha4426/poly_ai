"""
News & LLM Analysis Engine

Fetches real-time news, runs LLM analysis to:
1. Summarize relevant news for a market
2. Estimate probability impact of breaking news
3. Detect narrative shifts
4. Generate trade signals from news events

Supports: NewsAPI, NewsData.io, GNews.io (free tiers)
LLM: OpenAI GPT-4o-mini for fast, cheap analysis
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

import httpx

from config.settings import Settings


@dataclass
class NewsArticle:
    """A news article."""
    title: str
    description: str
    source: str
    url: str
    published_at: Optional[str] = None
    content: str = ""

    @property
    def age_hours(self) -> float:
        if not self.published_at:
            return 999
        try:
            dt = datetime.fromisoformat(self.published_at.replace("Z", "+00:00"))
            delta = datetime.now(timezone.utc) - dt
            return delta.total_seconds() / 3600
        except (ValueError, TypeError):
            return 999

    @property
    def is_fresh(self) -> bool:
        return self.age_hours <= Settings.STALE_NEWS_HOURS


@dataclass
class NewsAnalysis:
    """LLM analysis of news impact on a market."""
    market_question: str
    probability_estimate: float  # 0.0-1.0 estimated probability of "Yes"
    confidence: float  # 0.0-1.0 how confident the LLM is
    reasoning: str
    key_facts: List[str]
    sentiment: str  # bullish, bearish, neutral
    breaking: bool  # is this breaking/major news?
    articles_analyzed: int
    analysis_time: float = 0


class NewsFetcher:
    """Fetches news from multiple free APIs."""

    def __init__(self):
        self._cache: Dict[str, Tuple[List[NewsArticle], float]] = {}
        self._cache_ttl = 300  # 5 min cache per query

    async def search(self, query: str, max_results: int = 10) -> List[NewsArticle]:
        """Search news across all configured providers."""
        cache_key = f"{query}:{max_results}"
        cached = self._cache.get(cache_key)
        if cached and time.time() - cached[1] < self._cache_ttl:
            return cached[0]

        articles = []

        # Fetch from all configured sources in parallel
        tasks = []
        if Settings.NEWS_API_KEY:
            tasks.append(self._newsapi(query, max_results))
        if Settings.NEWSDATA_API_KEY:
            tasks.append(self._newsdata(query, max_results))
        if Settings.GNEWS_API_KEY:
            tasks.append(self._gnews(query, max_results))

        if not tasks:
            # No API keys — use free RSS/public sources
            tasks.append(self._free_news(query, max_results))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, list):
                articles.extend(result)

        # Deduplicate by title similarity
        seen_titles: set = set()
        unique = []
        for a in articles:
            key = a.title.lower()[:50]
            if key not in seen_titles:
                seen_titles.add(key)
                unique.append(a)

        # Sort by freshness
        unique.sort(key=lambda a: a.age_hours)
        result = unique[:max_results]
        self._cache[cache_key] = (result, time.time())
        return result

    async def _newsapi(self, query: str, limit: int) -> List[NewsArticle]:
        """Fetch from newsapi.org (free: 100 req/day)."""
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    "https://newsapi.org/v2/everything",
                    params={
                        "q": query,
                        "sortBy": "publishedAt",
                        "pageSize": limit,
                        "language": "en",
                        "apiKey": Settings.NEWS_API_KEY,
                    }
                )
                if resp.status_code != 200:
                    return []
                data = resp.json()
                return [
                    NewsArticle(
                        title=a.get("title", ""),
                        description=a.get("description", ""),
                        source=a.get("source", {}).get("name", ""),
                        url=a.get("url", ""),
                        published_at=a.get("publishedAt", ""),
                        content=a.get("content", "")[:500],
                    )
                    for a in data.get("articles", [])
                ]
        except Exception:
            return []

    async def _newsdata(self, query: str, limit: int) -> List[NewsArticle]:
        """Fetch from newsdata.io (free: 200 req/day)."""
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    "https://newsdata.io/api/1/latest",
                    params={
                        "q": query,
                        "language": "en",
                        "size": min(limit, 10),
                        "apikey": Settings.NEWSDATA_API_KEY,
                    }
                )
                if resp.status_code != 200:
                    return []
                data = resp.json()
                return [
                    NewsArticle(
                        title=a.get("title", ""),
                        description=a.get("description", ""),
                        source=a.get("source_name", a.get("source_id", "")),
                        url=a.get("link", ""),
                        published_at=a.get("pubDate", ""),
                        content=a.get("content", "")[:500] if a.get("content") else "",
                    )
                    for a in data.get("results", [])
                ]
        except Exception:
            return []

    async def _gnews(self, query: str, limit: int) -> List[NewsArticle]:
        """Fetch from gnews.io (free: 100 req/day)."""
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    "https://gnews.io/api/v4/search",
                    params={
                        "q": query,
                        "lang": "en",
                        "max": min(limit, 10),
                        "token": Settings.GNEWS_API_KEY,
                    }
                )
                if resp.status_code != 200:
                    return []
                data = resp.json()
                return [
                    NewsArticle(
                        title=a.get("title", ""),
                        description=a.get("description", ""),
                        source=a.get("source", {}).get("name", ""),
                        url=a.get("url", ""),
                        published_at=a.get("publishedAt", ""),
                        content=a.get("content", "")[:500] if a.get("content") else "",
                    )
                    for a in data.get("articles", [])
                ]
        except Exception:
            return []

    async def _free_news(self, query: str, limit: int) -> List[NewsArticle]:
        """
        Free news without API key — scrape Google News RSS.
        Limited but works without any account.
        """
        import urllib.parse
        encoded = urllib.parse.quote_plus(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(rss_url)
                if resp.status_code != 200:
                    return []

                # Simple XML parsing (no lxml dependency needed)
                text = resp.text
                articles = []
                items = text.split("<item>")[1:]  # skip channel header

                for item_xml in items[:limit]:
                    title = _extract_xml_tag(item_xml, "title")
                    link = _extract_xml_tag(item_xml, "link")
                    pub_date = _extract_xml_tag(item_xml, "pubDate")
                    source = _extract_xml_tag(item_xml, "source")

                    if title:
                        articles.append(NewsArticle(
                            title=title,
                            description="",
                            source=source or "Google News",
                            url=link or "",
                            published_at=pub_date or "",
                        ))

                return articles
        except Exception:
            return []


def _extract_xml_tag(xml_text: str, tag: str) -> str:
    """Extract text from an XML tag (simple, no external deps)."""
    start = xml_text.find(f"<{tag}")
    if start == -1:
        return ""
    # Handle CDATA
    content_start = xml_text.find(">", start) + 1
    end = xml_text.find(f"</{tag}>", content_start)
    if end == -1:
        return ""
    content = xml_text[content_start:end].strip()
    # Strip CDATA wrapper
    if content.startswith("<![CDATA["):
        content = content[9:]
    if content.endswith("]]>"):
        content = content[:-3]
    return content.strip()


class LLMAnalyzer:
    """
    Uses GPT-4o-mini to analyze news and estimate probabilities.
    
    Key capabilities:
    - Probability estimation: "Given these facts, what's the probability of X?"
    - Narrative shift detection: "Has sentiment changed vs previous analysis?"
    - Multi-market reasoning: "Which of these 5 markets does this news affect?"
    """

    def __init__(self):
        self._news = NewsFetcher()

    async def analyze_market(self, market_question: str,
                              additional_context: str = "",
                              news_query: Optional[str] = None) -> Optional[NewsAnalysis]:
        """
        Full analysis pipeline:
        1. Fetch recent news about this market topic
        2. Feed to LLM with the market question
        3. Get probability estimate + reasoning
        """
        if not Settings.has_llm():
            return None

        t0 = time.time()

        # Build news query from market question if not provided
        if not news_query:
            news_query = self._extract_search_query(market_question)

        articles = await self._news.search(news_query)
        fresh_articles = [a for a in articles if a.is_fresh]

        if not fresh_articles:
            fresh_articles = articles[:3]  # Use whatever we have

        # Build prompt
        news_text = "\n".join([
            f"- [{a.source}] {a.title}: {a.description or ''}" 
            for a in fresh_articles[:8]
        ])

        prompt = f"""You are a prediction market analyst. Analyze the following market question and recent news to estimate the probability of "Yes".

MARKET QUESTION: {market_question}

{f"ADDITIONAL CONTEXT: {additional_context}" if additional_context else ""}

RECENT NEWS:
{news_text if news_text else "No recent news found."}

Respond in EXACTLY this JSON format:
{{
    "probability": 0.XX,
    "confidence": 0.XX,
    "reasoning": "2-3 sentence explanation",
    "key_facts": ["fact1", "fact2", "fact3"],
    "sentiment": "bullish|bearish|neutral",
    "breaking": true|false
}}

Rules:
- probability: Your best estimate of Yes outcome (0.01 to 0.99)
- confidence: How sure you are about your estimate (0.0 to 1.0)
- If no clear information, set confidence low (<0.3)
- breaking: true only if there's major news in the last 6 hours that changes things
- Be calibrated: 0.50 means truly uncertain"""

        try:
            result = await self._call_llm(prompt)
            if not result:
                return None

            # Parse JSON response
            parsed = json.loads(result)

            return NewsAnalysis(
                market_question=market_question,
                probability_estimate=float(parsed.get("probability", 0.5)),
                confidence=float(parsed.get("confidence", 0.3)),
                reasoning=parsed.get("reasoning", ""),
                key_facts=parsed.get("key_facts", []),
                sentiment=parsed.get("sentiment", "neutral"),
                breaking=bool(parsed.get("breaking", False)),
                articles_analyzed=len(fresh_articles),
                analysis_time=time.time() - t0,
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"⚠️ LLM parse error: {e}")
            return None

    async def batch_analyze(self, markets: List[Dict],
                             max_parallel: int = 3) -> List[Optional[NewsAnalysis]]:
        """
        Analyze multiple markets efficiently.
        markets: [{"question": str, "context": str, "news_query": str}, ...]
        """
        semaphore = asyncio.Semaphore(max_parallel)

        async def _analyze(m):
            async with semaphore:
                return await self.analyze_market(
                    m["question"],
                    m.get("context", ""),
                    m.get("news_query"),
                )

        return await asyncio.gather(*[_analyze(m) for m in markets], return_exceptions=True)

    async def detect_narrative_shift(self, market_question: str,
                                      previous_sentiment: str,
                                      previous_probability: float) -> Optional[Dict]:
        """
        Detect if news has shifted the narrative for a market.
        Returns: {"shifted": bool, "direction": "up"|"down", "magnitude": float, "trigger": str}
        """
        if not Settings.has_llm():
            return None

        news_query = self._extract_search_query(market_question)
        articles = await self._news.search(news_query, max_results=5)

        if not articles:
            return None

        news_text = "\n".join([
            f"- [{a.source}] {a.title}" for a in articles[:5]
        ])

        prompt = f"""You are detecting narrative shifts in prediction markets.

MARKET: {market_question}
PREVIOUS sentiment: {previous_sentiment}
PREVIOUS probability estimate: {previous_probability:.2f}

LATEST NEWS:
{news_text}

Has the narrative shifted? Respond in JSON:
{{
    "shifted": true|false,
    "direction": "up|down|none",
    "magnitude": 0.XX,
    "trigger": "brief description of what changed",
    "new_probability": 0.XX
}}

Only set shifted=true if news meaningfully changes the outlook."""

        try:
            result = await self._call_llm(prompt)
            if result:
                return json.loads(result)
        except (json.JSONDecodeError, TypeError):
            pass
        return None

    async def _call_llm(self, prompt: str) -> Optional[str]:
        """Call OpenAI API."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {Settings.OPENAI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": Settings.OPENAI_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": Settings.LLM_TEMPERATURE,
                        "max_tokens": Settings.LLM_MAX_TOKENS,
                        "response_format": {"type": "json_object"},
                    }
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    print(f"⚠️ LLM API error {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            print(f"⚠️ LLM call failed: {e}")
        return None

    def _extract_search_query(self, question: str) -> str:
        """Extract the most relevant search terms from a market question."""
        # Remove common prediction market filler words
        stopwords = {
            "will", "the", "be", "in", "of", "to", "a", "an", "by", "on",
            "before", "after", "between", "this", "that", "with", "for",
            "at", "from", "or", "and", "is", "are", "was", "were",
            "has", "have", "had", "do", "does", "did", "not", "no", "yes",
            "more", "than", "over", "under", "above", "below",
        }
        words = question.replace("?", "").replace("!", "").split()
        keywords = [w for w in words if w.lower() not in stopwords and len(w) > 2]
        return " ".join(keywords[:6])


# Singletons
_news_fetcher: Optional[NewsFetcher] = None
_llm_analyzer: Optional[LLMAnalyzer] = None


def get_news_fetcher() -> NewsFetcher:
    global _news_fetcher
    if _news_fetcher is None:
        _news_fetcher = NewsFetcher()
    return _news_fetcher


def get_llm_analyzer() -> LLMAnalyzer:
    global _llm_analyzer
    if _llm_analyzer is None:
        _llm_analyzer = LLMAnalyzer()
    return _llm_analyzer
