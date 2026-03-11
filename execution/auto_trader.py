"""
Auto-Trader — Autonomous execution loop.

Ties together: Scanner → EdgeDetector → PositionManager.
Runs continuously, making trades when strong signals found.
Also handles position monitoring and auto-exits.
"""

import asyncio
import time
from typing import List, Optional

from config.settings import Settings
from scanner.market_scanner import MarketScanner
from strategy.edge_detector import EdgeDetector, Signal, SignalType
from execution.position_manager import PositionManager


class AutoTrader:
    """
    Main autonomous trading loop.
    
    Cycle:
    1. Scanner fetches all markets
    2. EdgeDetector analyzes & generates signals
    3. Execute top signals (respecting risk limits)
    4. Update positions, check narratives
    5. Auto-exit positions that hit stop/take-profit or narrative shift
    6. Sleep & repeat
    """

    def __init__(self, scanner: MarketScanner, detector: EdgeDetector,
                 pm: PositionManager):
        self.scanner = scanner
        self.detector = detector
        self.pm = pm
        self.running = False
        self._cycle_count = 0
        self._last_cycle_time = 0

        # Limits per cycle
        self.max_signals_per_cycle = 5
        self.max_buys_per_cycle = 3

    async def run_loop(self, callback=None):
        """
        Main trading loop. Runs until stopped.
        callback(signals, positions) is called each cycle for Telegram notifications.
        """
        self.running = True
        print("🤖 Auto-trader started")

        while self.running:
            try:
                signals, actions = await self.run_cycle()

                if callback:
                    await callback(signals, actions)

                self._cycle_count += 1
                self._last_cycle_time = time.time()

            except Exception as e:
                print(f"❌ Cycle error: {e}")
                import traceback
                traceback.print_exc()

            await asyncio.sleep(Settings.SCAN_INTERVAL_SECONDS)

    async def run_cycle(self) -> tuple:
        """
        Execute one full cycle:
        1. Scan markets
        2. Generate signals
        3. Execute trades
        4. Monitor positions
        
        Returns: (signals, actions_taken)
        """
        t0 = time.time()
        actions = []

        # ── Step 1: Scan ──
        await self.scanner.scan()
        snapshots = self.scanner.snapshots

        if not snapshots:
            print("📭 No markets found this cycle")
            return [], []

        # ── Step 2: Generate signals ──
        signals = await self.detector.analyze_all(list(snapshots.values()))

        # ── Step 3: Update position prices ──
        await self.pm.update_prices()

        # ── Step 4: Handle sell/reduce signals from narrative shifts ──
        for sig in signals:
            if sig.signal_type == SignalType.SELL:
                pos_id = f"{sig.condition_id}:{sig.outcome}"
                if pos_id in self.pm.positions:
                    result = await self.pm.close_position(
                        pos_id, reason=f"narrative_shift: {sig.reasoning[:100]}"
                    )
                    if result:
                        actions.append(("SELL", result))

            elif sig.signal_type == SignalType.REDUCE:
                # For now, treat REDUCE as a full exit (simplification)
                pos_id = f"{sig.condition_id}:{sig.outcome}"
                if pos_id in self.pm.positions:
                    result = await self.pm.close_position(
                        pos_id, reason=f"reduce: {sig.reasoning[:100]}"
                    )
                    if result:
                        actions.append(("REDUCE", result))

        # ── Step 5: Execute buy signals (priority order) ──
        buy_signals = [
            s for s in signals
            if s.signal_type in (SignalType.CONFIRMED_BUY, SignalType.STRONG_BUY, SignalType.BUY)
        ]

        buys_this_cycle = 0
        for sig in buy_signals[:self.max_signals_per_cycle]:
            if buys_this_cycle >= self.max_buys_per_cycle:
                break

            pos = await self.pm.open_position(sig)
            if pos:
                actions.append(("BUY", pos))
                buys_this_cycle += 1

        # ── Step 6: Auto-exit rules ──
        await self._check_auto_exits(actions)

        elapsed = time.time() - t0
        cycle_num = self._cycle_count + 1
        print(
            f"\n🔄 Cycle #{cycle_num} | {len(signals)} signals, {len(actions)} actions "
            f"| {len(self.pm.positions)} open positions | {elapsed:.1f}s\n"
        )

        return signals, actions

    async def _check_auto_exits(self, actions: List):
        """Check auto-exit rules for open positions."""
        for pos_id, pos in list(self.pm.positions.items()):
            reason = None

            # Take profit: +50% or higher
            if pos.unrealized_pnl_pct >= 50:
                reason = f"take_profit: +{pos.unrealized_pnl_pct:.1f}%"

            # Stop loss: -25% or worse
            elif pos.unrealized_pnl_pct <= -25:
                reason = f"stop_loss: {pos.unrealized_pnl_pct:.1f}%"

            # Near-resolution: price >0.95 (probably resolved, take profit)
            elif pos.current_price >= 0.95 and pos.outcome == "yes":
                reason = f"near_resolution: price={pos.current_price:.3f}"

            # Stale position: held >7 days with no significant edge
            elif pos.hold_duration_hours > 168 and abs(pos.unrealized_pnl_pct) < 3:
                reason = f"stale: held {pos.hold_duration_hours:.0f}h, no movement"

            if reason:
                result = await self.pm.close_position(pos_id, reason=reason)
                if result:
                    actions.append(("AUTO_EXIT", result))

    def stop(self):
        """Stop the trading loop."""
        self.running = False
        print("🛑 Auto-trader stopped")

    def get_status(self) -> dict:
        """Current status of the auto-trader."""
        return {
            "running": self.running,
            "cycles": self._cycle_count,
            "last_cycle": self._last_cycle_time,
            "portfolio": self.pm.get_portfolio_summary(),
            "scanner_markets": len(self.scanner.snapshots),
            "signals_generated": self.detector.signals_generated,
            "markets_analyzed": self.detector.markets_analyzed,
        }
