"""
Position Manager — Tracks all open & closed positions, P&L, risk.

Responsibilities:
1. Record trade entries / exits with full context
2. Track real-time P&L per position and portfolio
3. Enforce risk limits (max portfolio exposure, max position size)
4. Feed positions to NarrativeShiftDetector for monitoring
5. Persist state to SQLite for crash recovery
"""

import asyncio
import time
import aiosqlite
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

from config.settings import Settings
from data.clob_client import ClobClient
from analysis.narrative import NarrativeShiftDetector


class PositionStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    PENDING = "pending"


@dataclass
class Position:
    """A tracked position."""
    id: str  # condition_id:outcome
    condition_id: str
    token_id: str
    question: str
    outcome: str  # "yes" or "no"
    category: str
    subcategory: str
    source: str  # which engine generated the signal

    # Trade details
    entry_price: float
    entry_amount_usd: float
    shares: float
    entry_time: float

    # Current state
    current_price: float = 0
    status: str = "open"

    # Exit details (filled on close)
    exit_price: float = 0
    exit_time: float = 0
    exit_reason: str = ""

    # Signal context
    edge_at_entry: float = 0
    confidence_at_entry: float = 0
    reasoning: str = ""

    @property
    def unrealized_pnl(self) -> float:
        """Current P&L in USD."""
        if self.status == "closed":
            return self.realized_pnl
        return self.shares * (self.current_price - self.entry_price)

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.entry_amount_usd <= 0:
            return 0
        return (self.unrealized_pnl / self.entry_amount_usd) * 100

    @property
    def realized_pnl(self) -> float:
        if self.status != "closed":
            return 0
        return self.shares * (self.exit_price - self.entry_price)

    @property
    def current_value(self) -> float:
        return self.shares * self.current_price

    @property
    def is_winner(self) -> bool:
        return self.unrealized_pnl > 0

    @property
    def hold_duration_hours(self) -> float:
        end = self.exit_time if self.status == "closed" else time.time()
        return (end - self.entry_time) / 3600


class PaperTrader:
    """Simulates trading in paper mode — no real money at risk."""

    def __init__(self):
        self.balance = Settings.PAPER_BALANCE
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.trade_log: List[Dict] = []

    def buy(self, token_id: str, amount_usd: float, price: float, **kwargs) -> Dict:
        if amount_usd > self.balance:
            return {"success": False, "error": f"Insufficient balance: ${self.balance:.2f}"}
        self.balance -= amount_usd
        shares = amount_usd / price
        self.trade_log.append({
            "action": "buy", "token_id": token_id, "amount": amount_usd,
            "price": price, "shares": shares, "time": time.time()
        })
        return {"success": True, "shares": shares, "price": price, "paper": True}

    def sell(self, token_id: str, shares: float, price: float, **kwargs) -> Dict:
        proceeds = shares * price
        self.balance += proceeds
        self.trade_log.append({
            "action": "sell", "token_id": token_id, "shares": shares,
            "price": price, "proceeds": proceeds, "time": time.time()
        })
        return {"success": True, "proceeds": proceeds, "price": price, "paper": True}


class PositionManager:
    """
    Manages all positions — entry, exit, tracking, risk, persistence.
    Works in both paper and live mode.
    """

    def __init__(self, clob: ClobClient, narrative: NarrativeShiftDetector):
        self.clob = clob
        self.narrative = narrative
        self.paper = PaperTrader() if Settings.is_paper() else None
        self.positions: Dict[str, Position] = {}  # id -> Position
        self.closed_positions: List[Position] = []
        self._db_path = Settings.DB_PATH
        self._daily_trade_count = 0
        self._daily_reset_time = 0

    # ═══════════════════════════════════════════════════════════════
    # TRADE ENTRY
    # ═══════════════════════════════════════════════════════════════

    async def open_position(self, signal) -> Optional[Position]:
        """
        Open a new position from a trade signal.
        Returns the Position if successful, None if rejected.
        """
        from strategy.edge_detector import Signal

        # Pre-flight checks
        rejection = self._check_risk_limits(signal)
        if rejection:
            print(f"⛔ Trade rejected: {rejection}")
            return None

        amount_usd = signal.suggested_amount_usd
        if amount_usd < Settings.MIN_TRADE_USD:
            amount_usd = Settings.MIN_TRADE_USD

        # Execute trade
        if self.paper:
            result = self.paper.buy(signal.token_id, amount_usd, signal.current_price)
        else:
            result = await self.clob.buy(
                signal.token_id, amount_usd,
                price=signal.current_price
            )

        if not result.get("success"):
            print(f"❌ Trade failed: {result.get('error', 'unknown')}")
            return None

        shares = result.get("shares", amount_usd / signal.current_price)
        pos_id = f"{signal.condition_id}:{signal.outcome}"

        pos = Position(
            id=pos_id,
            condition_id=signal.condition_id,
            token_id=signal.token_id,
            question=signal.market_question,
            outcome=signal.outcome,
            category=signal.category,
            subcategory=signal.subcategory,
            source=signal.source,
            entry_price=signal.current_price,
            entry_amount_usd=amount_usd,
            shares=shares,
            entry_time=time.time(),
            current_price=signal.current_price,
            edge_at_entry=signal.edge_percent,
            confidence_at_entry=signal.confidence,
            reasoning=signal.reasoning[:500],
        )

        self.positions[pos_id] = pos
        self._daily_trade_count += 1

        # Register with narrative detector
        self.narrative.track_position(
            signal.condition_id, signal.token_id,
            signal.market_question, signal.outcome,
            signal.current_price, shares,
            signal.reasoning[:200]
        )

        # Persist
        await self._save_position(pos)

        mode = "📝 PAPER" if self.paper else "🔴 LIVE"
        print(
            f"✅ {mode} BUY | {signal.outcome.upper()} @ {signal.current_price:.3f} "
            f"| ${amount_usd:.2f} ({shares:.1f} shares) "
            f"| Edge: {signal.edge_percent:+.1f}% | {signal.market_question[:40]}..."
        )

        return pos

    async def close_position(self, pos_id: str, reason: str = "manual",
                              price: Optional[float] = None) -> Optional[Position]:
        """Close a position."""
        pos = self.positions.get(pos_id)
        if not pos:
            return None

        sell_price = price or pos.current_price

        if self.paper:
            result = self.paper.sell(pos.token_id, pos.shares, sell_price)
        else:
            result = await self.clob.sell(pos.token_id, pos.shares, price=sell_price)

        if not result.get("success"):
            print(f"❌ Sell failed for {pos_id}: {result.get('error')}")
            return None

        pos.exit_price = sell_price
        pos.exit_time = time.time()
        pos.exit_reason = reason
        pos.status = "closed"

        del self.positions[pos_id]
        self.closed_positions.append(pos)
        self.narrative.untrack_position(pos.condition_id)
        await self._save_position(pos)

        pnl = pos.realized_pnl
        emoji = "🟢" if pnl > 0 else "🔴"
        mode = "📝" if self.paper else "🔴"
        print(
            f"{emoji} {mode} SELL | {pos.outcome.upper()} exit @ {sell_price:.3f} "
            f"| P&L: ${pnl:+.2f} ({pos.unrealized_pnl_pct:+.1f}%) "
            f"| Held {pos.hold_duration_hours:.1f}h | Reason: {reason}"
        )

        return pos

    # ═══════════════════════════════════════════════════════════════
    # RISK MANAGEMENT
    # ═══════════════════════════════════════════════════════════════

    def _check_risk_limits(self, signal) -> Optional[str]:
        """Check if a trade would violate risk limits. Returns rejection reason or None."""
        # Daily trade limit
        self._maybe_reset_daily()
        if self._daily_trade_count >= Settings.MAX_DAILY_TRADES:
            return f"Daily trade limit reached ({Settings.MAX_DAILY_TRADES})"

        # Position size limit
        if signal.suggested_amount_usd > Settings.MAX_TRADE_USD:
            return f"Amount ${signal.suggested_amount_usd:.2f} > max ${Settings.MAX_TRADE_USD}"

        # Duplicate position check
        pos_id = f"{signal.condition_id}:{signal.outcome}"
        if pos_id in self.positions:
            return f"Already have position in {signal.outcome} for this market"

        # Max portfolio exposure
        total_exposure = sum(p.current_value for p in self.positions.values())
        available = self._get_available_balance()
        max_exposure = available * Settings.MAX_PORTFOLIO_RISK

        if total_exposure + signal.suggested_amount_usd > max_exposure:
            return (
                f"Portfolio exposure ${total_exposure:.2f} + ${signal.suggested_amount_usd:.2f} "
                f"> max ${max_exposure:.2f} ({Settings.MAX_PORTFOLIO_RISK:.0%} of ${available:.2f})"
            )

        return None

    def _get_available_balance(self) -> float:
        if self.paper:
            return self.paper.balance
        return 0  # Will be updated async

    def _maybe_reset_daily(self):
        """Reset daily trade count at midnight."""
        now = time.time()
        if now - self._daily_reset_time > 86400:
            self._daily_trade_count = 0
            self._daily_reset_time = now

    # ═══════════════════════════════════════════════════════════════
    # PRICE UPDATES & MONITORING
    # ═══════════════════════════════════════════════════════════════

    async def update_prices(self):
        """Update current prices for all open positions."""
        if not self.positions:
            return

        token_ids = [p.token_id for p in self.positions.values()]
        prices = await self.clob.get_prices(token_ids)

        for pos in self.positions.values():
            if pos.token_id in prices:
                pos.current_price = prices[pos.token_id]
                self.narrative.update_price(pos.condition_id, pos.current_price)

    # ═══════════════════════════════════════════════════════════════
    # PORTFOLIO STATS
    # ═══════════════════════════════════════════════════════════════

    def get_portfolio_summary(self) -> Dict:
        """Get portfolio overview."""
        open_positions = list(self.positions.values())
        total_invested = sum(p.entry_amount_usd for p in open_positions)
        total_current = sum(p.current_value for p in open_positions)
        total_unrealized = sum(p.unrealized_pnl for p in open_positions)

        all_closed = self.closed_positions
        total_realized = sum(p.realized_pnl for p in all_closed)
        winners = sum(1 for p in all_closed if p.realized_pnl > 0)
        losers = sum(1 for p in all_closed if p.realized_pnl <= 0)
        win_rate = winners / len(all_closed) * 100 if all_closed else 0

        return {
            "open_count": len(open_positions),
            "total_invested": round(total_invested, 2),
            "total_current_value": round(total_current, 2),
            "unrealized_pnl": round(total_unrealized, 2),
            "closed_count": len(all_closed),
            "realized_pnl": round(total_realized, 2),
            "total_pnl": round(total_realized + total_unrealized, 2),
            "win_rate": round(win_rate, 1),
            "winners": winners,
            "losers": losers,
            "daily_trades": self._daily_trade_count,
            "balance": self.paper.balance if self.paper else 0,
        }

    def format_portfolio(self) -> str:
        """Human-readable portfolio summary."""
        s = self.get_portfolio_summary()
        mode = "📝 PAPER" if self.paper else "🔴 LIVE"

        lines = [
            f"{'='*40}",
            f"📊 Portfolio | {mode}",
            f"{'='*40}",
            f"💰 Balance: ${s['balance']:.2f}",
            f"📈 Open: {s['open_count']} positions (${s['total_invested']:.2f} invested)",
            f"💵 Current Value: ${s['total_current_value']:.2f}",
            f"📊 Unrealized P&L: ${s['unrealized_pnl']:+.2f}",
            f"{'─'*40}",
            f"📋 Closed: {s['closed_count']} trades",
            f"🎯 Win Rate: {s['win_rate']:.1f}% ({s['winners']}W / {s['losers']}L)",
            f"💰 Realized P&L: ${s['realized_pnl']:+.2f}",
            f"📊 Total P&L: ${s['total_pnl']:+.2f}",
            f"{'='*40}",
        ]

        # Open position details
        for pos in self.positions.values():
            pnl_emoji = "🟢" if pos.unrealized_pnl > 0 else "🔴"
            lines.append(
                f"  {pnl_emoji} {pos.outcome.upper()} @ {pos.entry_price:.3f} → {pos.current_price:.3f} "
                f"| ${pos.unrealized_pnl:+.2f} ({pos.unrealized_pnl_pct:+.1f}%) "
                f"| {pos.question[:35]}..."
            )

        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════
    # PERSISTENCE (SQLite)
    # ═══════════════════════════════════════════════════════════════

    async def init_db(self):
        """Create tables if they don't exist."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id TEXT PRIMARY KEY,
                    condition_id TEXT,
                    token_id TEXT,
                    question TEXT,
                    outcome TEXT,
                    category TEXT,
                    subcategory TEXT,
                    source TEXT,
                    entry_price REAL,
                    entry_amount_usd REAL,
                    shares REAL,
                    entry_time REAL,
                    current_price REAL,
                    status TEXT,
                    exit_price REAL,
                    exit_time REAL,
                    exit_reason TEXT,
                    edge_at_entry REAL,
                    confidence_at_entry REAL,
                    reasoning TEXT
                )
            """)
            await db.commit()

    async def _save_position(self, pos: Position):
        """Upsert a position to the database."""
        try:
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO positions
                    (id, condition_id, token_id, question, outcome, category,
                     subcategory, source, entry_price, entry_amount_usd, shares,
                     entry_time, current_price, status, exit_price, exit_time,
                     exit_reason, edge_at_entry, confidence_at_entry, reasoning)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pos.id, pos.condition_id, pos.token_id, pos.question,
                    pos.outcome, pos.category, pos.subcategory, pos.source,
                    pos.entry_price, pos.entry_amount_usd, pos.shares,
                    pos.entry_time, pos.current_price, pos.status,
                    pos.exit_price, pos.exit_time, pos.exit_reason,
                    pos.edge_at_entry, pos.confidence_at_entry, pos.reasoning
                ))
                await db.commit()
        except Exception as e:
            print(f"⚠️ DB save error: {e}")

    async def load_positions(self):
        """Load open positions from database on startup."""
        try:
            async with aiosqlite.connect(self._db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT * FROM positions WHERE status = 'open'"
                ) as cursor:
                    rows = await cursor.fetchall()
                    for row in rows:
                        pos = Position(
                            id=row["id"],
                            condition_id=row["condition_id"],
                            token_id=row["token_id"],
                            question=row["question"],
                            outcome=row["outcome"],
                            category=row["category"],
                            subcategory=row["subcategory"],
                            source=row["source"],
                            entry_price=row["entry_price"],
                            entry_amount_usd=row["entry_amount_usd"],
                            shares=row["shares"],
                            entry_time=row["entry_time"],
                            current_price=row["current_price"],
                            status=row["status"],
                            exit_price=row["exit_price"],
                            exit_time=row["exit_time"],
                            exit_reason=row["exit_reason"],
                            edge_at_entry=row["edge_at_entry"],
                            confidence_at_entry=row["confidence_at_entry"],
                            reasoning=row["reasoning"],
                        )
                        self.positions[pos.id] = pos
                        # Re-register with narrative detector
                        self.narrative.track_position(
                            pos.condition_id, pos.token_id,
                            pos.question, pos.outcome,
                            pos.entry_price, pos.shares,
                            pos.reasoning[:200]
                        )

                # Also load closed for stats
                async with db.execute(
                    "SELECT * FROM positions WHERE status = 'closed' ORDER BY exit_time DESC LIMIT 100"
                ) as cursor:
                    rows = await cursor.fetchall()
                    for row in rows:
                        pos = Position(
                            id=row["id"],
                            condition_id=row["condition_id"],
                            token_id=row["token_id"],
                            question=row["question"],
                            outcome=row["outcome"],
                            category=row["category"],
                            subcategory=row["subcategory"],
                            source=row["source"],
                            entry_price=row["entry_price"],
                            entry_amount_usd=row["entry_amount_usd"],
                            shares=row["shares"],
                            entry_time=row["entry_time"],
                            current_price=row["current_price"],
                            status=row["status"],
                            exit_price=row["exit_price"],
                            exit_time=row["exit_time"],
                            exit_reason=row["exit_reason"],
                            edge_at_entry=row["edge_at_entry"],
                            confidence_at_entry=row["confidence_at_entry"],
                            reasoning=row["reasoning"],
                        )
                        self.closed_positions.append(pos)

            print(f"📂 Loaded {len(self.positions)} open + {len(self.closed_positions)} closed positions")
        except Exception as e:
            print(f"⚠️ DB load error (first run?): {e}")
