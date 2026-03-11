"""
Polymarket AI Trading Bot — Entry Point

Wires together:
  Scanner → EdgeDetector → AutoTrader → PositionManager → TelegramBot

Usage:
  python main.py           # Start bot (paper mode by default)
  python main.py --scan    # Single scan + signal generation, then exit
  python main.py --live    # Override to live trading mode
"""

import asyncio
import argparse
import os
import sys

from config.settings import Settings
from data.gamma_client import GammaClient
from data.clob_client import ClobClient
from analysis.news_llm import get_llm_analyzer
from analysis.narrative import NarrativeShiftDetector
from scanner.market_scanner import MarketScanner
from strategy.edge_detector import EdgeDetector
from execution.position_manager import PositionManager
from execution.auto_trader import AutoTrader
from bot.telegram_bot import TelegramBot


async def run_single_scan():
    """Run one scan cycle, print signals, and exit."""
    print("\n🔍 Running single scan...\n")

    gamma = GammaClient()
    clob = ClobClient()
    scanner = MarketScanner(gamma, clob)
    detector = EdgeDetector(scanner)

    await scanner.scan()
    summary = scanner.summary()

    print(f"📊 Found {summary['total']} markets")
    for cat, cnt in summary.get("by_category", {}).items():
        print(f"  • {cat}: {cnt}")

    snapshots = list(scanner.snapshots.values())
    if not snapshots:
        print("📭 No markets to analyze")
        return

    signals = await detector.analyze_all(snapshots)

    if not signals:
        print("\n📭 No trade signals found")
        return

    print(f"\n📊 Generated {len(signals)} signals:\n")
    for i, sig in enumerate(signals, 1):
        print(f"  {i}. {sig}")
        print(f"     Confidence: {sig.confidence:.2f} | Source: {sig.source}")
        print(f"     Suggestion: ${sig.suggested_amount_usd:.2f}")
        if sig.reasoning:
            print(f"     Reasoning: {sig.reasoning[:120]}")
        print()


async def run_bot():
    """Start the full trading bot with auto-trading and Telegram."""
    Settings.print_status()

    # Ensure data directory exists for SQLite
    os.makedirs(os.path.dirname(Settings.DB_PATH) or "data", exist_ok=True)

    # Initialize components
    gamma = GammaClient()
    clob = ClobClient()

    # Initialize trading if live mode
    if not Settings.is_paper() and Settings.PRIVATE_KEY:
        clob.init_trading()

    llm = get_llm_analyzer()
    narrative = NarrativeShiftDetector(llm)
    scanner = MarketScanner(gamma, clob)
    detector = EdgeDetector(scanner)
    pm = PositionManager(clob, narrative)
    trader = AutoTrader(scanner, detector, pm)
    telegram = TelegramBot(scanner, detector, pm, trader)

    # Initialize database
    await pm.init_db()
    await pm.load_positions()

    # Start Telegram bot
    await telegram.start()

    # Start auto-trading loop
    print("🚀 Starting auto-trading loop...")
    try:
        await trader.run_loop(callback=telegram._on_cycle_complete)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    finally:
        trader.stop()
        await telegram.stop()

    # Final report
    print("\n" + pm.format_portfolio())


def main():
    parser = argparse.ArgumentParser(description="Polymarket AI Trading Bot")
    parser.add_argument("--scan", action="store_true", help="Single scan + signals, then exit")
    parser.add_argument("--live", action="store_true", help="Override to live trading mode")
    args = parser.parse_args()

    if args.live:
        Settings.TRADING_MODE = "live"

    if args.scan:
        asyncio.run(run_single_scan())
    else:
        asyncio.run(run_bot())


if __name__ == "__main__":
    main()
