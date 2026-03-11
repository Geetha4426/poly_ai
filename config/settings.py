"""
Polymarket AI Trading Bot — Configuration

Central config loaded from environment variables.
Covers: Polymarket API, Telegram, data feeds, trading params, AI/LLM.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """All configuration. Override via environment or .env file."""

    # ═══════════════════════════════════════════════════════════════
    # POLYMARKET
    # ═══════════════════════════════════════════════════════════════
    GAMMA_API_URL = os.getenv("GAMMA_API_URL", "https://gamma-api.polymarket.com")
    CLOB_API_URL = os.getenv("CLOB_API_URL", "https://clob.polymarket.com")
    CLOB_RELAY_URL = os.getenv("CLOB_RELAY_URL", "")  # geo-block bypass
    CLOB_RELAY_AUTH = os.getenv("CLOB_RELAY_AUTH_TOKEN", "")
    DATA_API_URL = os.getenv("DATA_API_URL", "https://data-api.polymarket.com")
    WS_URL = os.getenv("POLYMARKET_WS_URL", "wss://ws-subscriptions-clob.polymarket.com/ws/market")
    CHAIN_ID = int(os.getenv("POLYGON_CHAIN_ID", "137"))

    # Wallet
    PRIVATE_KEY = os.getenv("POLYGON_PRIVATE_KEY", "")
    FUNDER_ADDRESS = os.getenv("FUNDER_ADDRESS", "")
    SIGNATURE_TYPE = int(os.getenv("SIGNATURE_TYPE", "2"))

    # ═══════════════════════════════════════════════════════════════
    # TELEGRAM
    # ═══════════════════════════════════════════════════════════════
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")  # restrict to one user

    # ═══════════════════════════════════════════════════════════════
    # TRADING MODE & LIMITS
    # ═══════════════════════════════════════════════════════════════
    TRADING_MODE = os.getenv("TRADING_MODE", "paper")
    MIN_TRADE_USD = float(os.getenv("MIN_TRADE_USD", "5"))
    MAX_TRADE_USD = float(os.getenv("MAX_TRADE_USD", "50"))
    MAX_POSITION_USD = float(os.getenv("MAX_POSITION_USD", "200"))
    MAX_DAILY_TRADES = int(os.getenv("MAX_DAILY_TRADES", "50"))
    DEFAULT_SLIPPAGE = float(os.getenv("DEFAULT_SLIPPAGE", "2.0"))
    MAX_PORTFOLIO_RISK = float(os.getenv("MAX_PORTFOLIO_RISK", "0.5"))  # max 50% of balance at risk
    PAPER_BALANCE = float(os.getenv("PAPER_BALANCE", "1000"))

    # ═══════════════════════════════════════════════════════════════
    # AI / LLM
    # ═══════════════════════════════════════════════════════════════
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1000"))

    # ═══════════════════════════════════════════════════════════════
    # DATA FEEDS — external reference sources
    # ═══════════════════════════════════════════════════════════════
    # News
    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")  # newsapi.org
    NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY", "")  # newsdata.io
    GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "")  # gnews.io

    # YouTube
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")

    # Finance
    ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")
    FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

    # Polling
    FIVETHIRTYEIGHT_URL = os.getenv("FIVETHIRTYEIGHT_URL", "https://projects.fivethirtyeight.com")
    POLYMARKET_POLLING_URL = os.getenv("POLYMARKET_POLLING_URL", "https://polymarket.com/elections")

    # ═══════════════════════════════════════════════════════════════
    # EDGE DETECTION THRESHOLDS
    # ═══════════════════════════════════════════════════════════════
    MIN_EDGE_PERCENT = float(os.getenv("MIN_EDGE_PERCENT", "5"))  # min 5% edge to trade
    HIGH_EDGE_PERCENT = float(os.getenv("HIGH_EDGE_PERCENT", "15"))  # aggressive sizing
    CONFIRMED_EVENT_THRESHOLD = float(os.getenv("CONFIRMED_EVENT_THRESHOLD", "0.90"))  # 90%+ certainty
    MIN_LIQUIDITY_USD = float(os.getenv("MIN_LIQUIDITY_USD", "5000"))
    MIN_VOLUME_24H = float(os.getenv("MIN_VOLUME_24H", "1000"))
    STALE_NEWS_HOURS = int(os.getenv("STALE_NEWS_HOURS", "24"))

    # ═══════════════════════════════════════════════════════════════
    # SCANNER SETTINGS
    # ═══════════════════════════════════════════════════════════════
    SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", "120"))
    CATEGORIES = [c.strip() for c in os.getenv("CATEGORIES", "all").split(",")]
    # tags to scan from gamma API
    SCAN_TAGS = [t.strip() for t in os.getenv("SCAN_TAGS",
        "politics,elections,finance,crypto,geopolitics,sports,science,entertainment,culture"
    ).split(",")]

    # ═══════════════════════════════════════════════════════════════
    # DATABASE
    # ═══════════════════════════════════════════════════════════════
    DB_PATH = os.getenv("DB_PATH", "data/polymarket_ai.db")

    @classmethod
    def is_paper(cls) -> bool:
        return cls.TRADING_MODE.lower() == "paper"

    @classmethod
    def get_clob_url(cls) -> str:
        return cls.CLOB_RELAY_URL.rstrip("/") if cls.CLOB_RELAY_URL else cls.CLOB_API_URL

    @classmethod
    def has_llm(cls) -> bool:
        return bool(cls.OPENAI_API_KEY)

    @classmethod
    def has_news(cls) -> bool:
        return bool(cls.NEWS_API_KEY or cls.NEWSDATA_API_KEY or cls.GNEWS_API_KEY)

    @classmethod
    def has_youtube(cls) -> bool:
        return bool(cls.YOUTUBE_API_KEY)

    @classmethod
    def has_finance(cls) -> bool:
        return bool(cls.ALPHA_VANTAGE_KEY or cls.FINNHUB_API_KEY)

    @classmethod
    def print_status(cls):
        mode = "PAPER 📝" if cls.is_paper() else "LIVE 🔴"
        print(f"\n{'='*50}")
        print(f"🧠 POLYMARKET AI TRADER")
        print(f"{'='*50}")
        print(f"📊 Mode: {mode}")
        print(f"💰 Limits: ${cls.MIN_TRADE_USD}-${cls.MAX_TRADE_USD}/trade, ${cls.MAX_POSITION_USD} max pos")
        print(f"🎯 Edge: min {cls.MIN_EDGE_PERCENT}%, aggressive at {cls.HIGH_EDGE_PERCENT}%")
        print(f"📱 Telegram: {'✅' if cls.TELEGRAM_BOT_TOKEN else '❌'}")
        print(f"🔐 Wallet: {'✅' if cls.PRIVATE_KEY else '❌'}")
        print(f"🤖 LLM: {'✅ ' + cls.OPENAI_MODEL if cls.has_llm() else '❌'}")
        print(f"📰 News: {'✅' if cls.has_news() else '❌'}")
        print(f"📺 YouTube: {'✅' if cls.has_youtube() else '❌'}")
        print(f"📈 Finance: {'✅' if cls.has_finance() else '❌'}")
        print(f"🔍 Scan: every {cls.SCAN_INTERVAL_SECONDS}s, tags={cls.SCAN_TAGS}")
        print(f"{'='*50}\n")
