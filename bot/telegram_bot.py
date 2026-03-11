"""
Telegram Bot Interface — Monitoring & Control

Commands:
- /start — Dashboard overview
- /scan — Trigger a market scan
- /signals — View latest signals
- /positions — Open positions & P&L
- /portfolio — Portfolio summary & stats
- /trade — Manual trade (admin)
- /close — Close a position
- /status — Bot status & health
- /auto — Toggle auto-trading on/off
- /settings — View current settings
- /help — Command list
"""

import asyncio
import logging
import time
from typing import Optional

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    MessageHandler, filters
)

from config.settings import Settings

logger = logging.getLogger(__name__)


class TelegramBot:
    """
    Telegram bot for monitoring and controlling the AI trader.
    """

    def __init__(self, scanner, detector, pm, auto_trader):
        self.scanner = scanner
        self.detector = detector
        self.pm = pm
        self.auto_trader = auto_trader
        self.app: Optional[Application] = None
        self._last_signals = []

    async def start(self):
        """Build and start the Telegram bot."""
        if not Settings.TELEGRAM_BOT_TOKEN:
            print("⚠️ No Telegram bot token — bot disabled")
            return

        self.app = (
            Application.builder()
            .token(Settings.TELEGRAM_BOT_TOKEN)
            .post_init(self._post_init)
            .build()
        )

        # Register handlers
        self.app.add_handler(CommandHandler("start", self._cmd_start))
        self.app.add_handler(CommandHandler("scan", self._cmd_scan))
        self.app.add_handler(CommandHandler("signals", self._cmd_signals))
        self.app.add_handler(CommandHandler("positions", self._cmd_positions))
        self.app.add_handler(CommandHandler("portfolio", self._cmd_portfolio))
        self.app.add_handler(CommandHandler("status", self._cmd_status))
        self.app.add_handler(CommandHandler("auto", self._cmd_auto))
        self.app.add_handler(CommandHandler("settings", self._cmd_settings))
        self.app.add_handler(CommandHandler("close", self._cmd_close))
        self.app.add_handler(CommandHandler("help", self._cmd_help))

        # Callback queries for inline buttons
        self.app.add_handler(CallbackQueryHandler(self._cb_close, pattern="^close:"))
        self.app.add_handler(CallbackQueryHandler(self._cb_refresh, pattern="^refresh:"))

        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(drop_pending_updates=True)
        print("📱 Telegram bot started")

    async def _post_init(self, application):
        """Set bot commands after initialization."""
        await application.bot.set_my_commands([
            BotCommand("start", "Dashboard"),
            BotCommand("scan", "Scan markets now"),
            BotCommand("signals", "View trade signals"),
            BotCommand("positions", "Open positions"),
            BotCommand("portfolio", "Portfolio stats"),
            BotCommand("status", "Bot status"),
            BotCommand("auto", "Toggle auto-trading"),
            BotCommand("close", "Close a position"),
            BotCommand("settings", "View settings"),
            BotCommand("help", "Help"),
        ])

    def _check_user(self, update: Update) -> bool:
        """Restrict to configured chat ID."""
        if not Settings.TELEGRAM_CHAT_ID:
            return True
        return str(update.effective_chat.id) == str(Settings.TELEGRAM_CHAT_ID)

    # ═══════════════════════════════════════════════════════════════
    # COMMANDS
    # ═══════════════════════════════════════════════════════════════

    async def _cmd_start(self, update: Update, context):
        if not self._check_user(update):
            return

        status = self.auto_trader.get_status()
        summary = self.pm.get_portfolio_summary()
        mode = "📝 PAPER" if Settings.is_paper() else "🔴 LIVE"
        auto = "✅ ON" if status["running"] else "❌ OFF"

        text = (
            f"🧠 <b>Polymarket AI Trader</b>\n\n"
            f"📊 Mode: {mode}\n"
            f"🤖 Auto-trade: {auto}\n"
            f"💰 Balance: ${summary['balance']:.2f}\n"
            f"📈 Open: {summary['open_count']} positions\n"
            f"💵 P&L: ${summary['total_pnl']:+.2f}\n"
            f"🎯 Win Rate: {summary['win_rate']:.1f}%\n"
            f"🔄 Cycles: {status['cycles']}\n"
            f"📡 Markets: {status['scanner_markets']}\n\n"
            f"Use /help for commands"
        )

        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("📊 Signals", callback_data="refresh:signals"),
                InlineKeyboardButton("📈 Positions", callback_data="refresh:positions"),
            ],
            [
                InlineKeyboardButton("🔄 Scan Now", callback_data="refresh:scan"),
                InlineKeyboardButton("📋 Portfolio", callback_data="refresh:portfolio"),
            ],
        ])

        await update.message.reply_text(text, parse_mode="HTML", reply_markup=keyboard)

    async def _cmd_scan(self, update: Update, context):
        if not self._check_user(update):
            return

        msg = await update.message.reply_text("🔍 Scanning markets...")
        await self.scanner.scan()
        summary = self.scanner.summary()

        text = (
            f"🔍 <b>Scan Complete</b>\n\n"
            f"📊 Total markets: {summary['total']}\n"
        )
        for cat, count in summary.get("by_category", {}).items():
            text += f"  • {cat}: {count}\n"

        text += f"\n⏰ Closing soon: {summary.get('closing_24h', 0)}\n"
        text += f"📈 High volume: {summary.get('high_volume', 0)}\n"

        await msg.edit_text(text, parse_mode="HTML")

    async def _cmd_signals(self, update: Update, context):
        if not self._check_user(update):
            return

        if not self._last_signals:
            await update.message.reply_text("📭 No signals yet. Run /scan first.")
            return

        text = "📊 <b>Latest Signals</b>\n\n"
        for i, sig in enumerate(self._last_signals[:10], 1):
            emoji = {
                "CONFIRMED_BUY": "💎", "STRONG_BUY": "🔥",
                "BUY": "📈", "SELL": "📉",
                "BUY_MORE": "⬆️", "REDUCE": "⬇️",
            }.get(sig.signal_type.value, "❓")

            text += (
                f"{i}. {emoji} <b>{sig.signal_type.value}</b>\n"
                f"   {sig.market_question[:60]}\n"
                f"   {sig.outcome.upper()} @ {sig.current_price:.3f} | "
                f"Edge: {sig.edge_percent:+.1f}% | {sig.source}\n\n"
            )

        await update.message.reply_text(text, parse_mode="HTML")

    async def _cmd_positions(self, update: Update, context):
        if not self._check_user(update):
            return

        if not self.pm.positions:
            await update.message.reply_text("📭 No open positions.")
            return

        text = "📈 <b>Open Positions</b>\n\n"
        for pos in self.pm.positions.values():
            pnl_emoji = "🟢" if pos.unrealized_pnl > 0 else "🔴"
            text += (
                f"{pnl_emoji} <b>{pos.outcome.upper()}</b> @ {pos.entry_price:.3f} → {pos.current_price:.3f}\n"
                f"   ${pos.unrealized_pnl:+.2f} ({pos.unrealized_pnl_pct:+.1f}%) | "
                f"{pos.hold_duration_hours:.1f}h\n"
                f"   {pos.question[:50]}\n"
            )
            text += f"   /close_{pos.id.replace(':', '_')}\n\n"

        await update.message.reply_text(text, parse_mode="HTML")

    async def _cmd_portfolio(self, update: Update, context):
        if not self._check_user(update):
            return

        text = f"<pre>{self.pm.format_portfolio()}</pre>"
        await update.message.reply_text(text, parse_mode="HTML")

    async def _cmd_status(self, update: Update, context):
        if not self._check_user(update):
            return

        status = self.auto_trader.get_status()
        mode = "📝 PAPER" if Settings.is_paper() else "🔴 LIVE"
        auto = "✅ Running" if status["running"] else "❌ Stopped"

        text = (
            f"🤖 <b>Bot Status</b>\n\n"
            f"📊 Mode: {mode}\n"
            f"🤖 Auto-trade: {auto}\n"
            f"🔄 Cycles completed: {status['cycles']}\n"
            f"📡 Markets tracked: {status['scanner_markets']}\n"
            f"🧠 Signals generated: {status['signals_generated']}\n"
            f"📊 Markets analyzed: {status['markets_analyzed']}\n"
            f"🤖 LLM: {'✅' if Settings.has_llm() else '❌'}\n"
            f"📰 News: {'✅' if Settings.has_news() else '❌'}\n"
        )

        if status["last_cycle"]:
            ago = int(time.time() - status["last_cycle"])
            text += f"⏱️ Last cycle: {ago}s ago\n"

        await update.message.reply_text(text, parse_mode="HTML")

    async def _cmd_auto(self, update: Update, context):
        if not self._check_user(update):
            return

        if self.auto_trader.running:
            self.auto_trader.stop()
            await update.message.reply_text("🛑 Auto-trading stopped.")
        else:
            asyncio.create_task(self.auto_trader.run_loop(self._on_cycle_complete))
            await update.message.reply_text("✅ Auto-trading started!")

    async def _cmd_close(self, update: Update, context):
        if not self._check_user(update):
            return

        if not self.pm.positions:
            await update.message.reply_text("📭 No open positions to close.")
            return

        buttons = []
        for pos in self.pm.positions.values():
            pnl_emoji = "🟢" if pos.unrealized_pnl > 0 else "🔴"
            label = f"{pnl_emoji} {pos.outcome.upper()} ${pos.unrealized_pnl:+.2f} | {pos.question[:30]}"
            buttons.append([InlineKeyboardButton(label, callback_data=f"close:{pos.id}")])

        keyboard = InlineKeyboardMarkup(buttons)
        await update.message.reply_text(
            "Select position to close:", reply_markup=keyboard
        )

    async def _cmd_settings(self, update: Update, context):
        if not self._check_user(update):
            return

        text = (
            f"⚙️ <b>Settings</b>\n\n"
            f"📊 Mode: {Settings.TRADING_MODE}\n"
            f"💰 Trade: ${Settings.MIN_TRADE_USD}-${Settings.MAX_TRADE_USD}\n"
            f"📈 Max position: ${Settings.MAX_POSITION_USD}\n"
            f"🎯 Min edge: {Settings.MIN_EDGE_PERCENT}%\n"
            f"🔥 High edge: {Settings.HIGH_EDGE_PERCENT}%\n"
            f"📡 Scan interval: {Settings.SCAN_INTERVAL_SECONDS}s\n"
            f"🏷️ Tags: {', '.join(Settings.SCAN_TAGS)}\n"
            f"📊 Max daily: {Settings.MAX_DAILY_TRADES}\n"
            f"🛡️ Max portfolio risk: {Settings.MAX_PORTFOLIO_RISK:.0%}\n"
        )

        await update.message.reply_text(text, parse_mode="HTML")

    async def _cmd_help(self, update: Update, context):
        if not self._check_user(update):
            return

        text = (
            "🧠 <b>Polymarket AI Trader</b>\n\n"
            "<b>Commands:</b>\n"
            "/start — Dashboard\n"
            "/scan — Scan markets now\n"
            "/signals — View trade signals\n"
            "/positions — Open positions\n"
            "/portfolio — Portfolio stats\n"
            "/status — Bot health\n"
            "/auto — Toggle auto-trading\n"
            "/close — Close a position\n"
            "/settings — Current settings\n"
            "/help — This message\n"
        )

        await update.message.reply_text(text, parse_mode="HTML")

    # ═══════════════════════════════════════════════════════════════
    # CALLBACKS
    # ═══════════════════════════════════════════════════════════════

    async def _cb_close(self, update: Update, context):
        query = update.callback_query
        await query.answer()

        pos_id = query.data.replace("close:", "")
        result = await self.pm.close_position(pos_id, reason="manual_telegram")
        if result:
            pnl = result.realized_pnl
            emoji = "🟢" if pnl > 0 else "🔴"
            await query.edit_message_text(
                f"{emoji} Closed: {result.outcome.upper()} | P&L: ${pnl:+.2f}"
            )
        else:
            await query.edit_message_text("❌ Position not found or already closed.")

    async def _cb_refresh(self, update: Update, context):
        query = update.callback_query
        await query.answer()
        action = query.data.replace("refresh:", "")

        if action == "signals":
            if self._last_signals:
                text = "📊 <b>Signals (refreshed)</b>\n\n"
                for sig in self._last_signals[:5]:
                    text += f"• {sig}\n"
                await query.edit_message_text(text, parse_mode="HTML")
            else:
                await query.edit_message_text("📭 No signals yet.")

        elif action == "positions":
            if self.pm.positions:
                text = "📈 <b>Positions</b>\n\n"
                for pos in self.pm.positions.values():
                    e = "🟢" if pos.unrealized_pnl > 0 else "🔴"
                    text += f"{e} {pos.outcome.upper()} ${pos.unrealized_pnl:+.2f}\n"
                await query.edit_message_text(text, parse_mode="HTML")
            else:
                await query.edit_message_text("📭 No positions.")

        elif action == "scan":
            await query.edit_message_text("🔍 Scanning...")
            await self.scanner.scan()
            s = self.scanner.summary()
            await query.edit_message_text(f"✅ Scan done: {s['total']} markets")

        elif action == "portfolio":
            text = f"<pre>{self.pm.format_portfolio()}</pre>"
            await query.edit_message_text(text, parse_mode="HTML")

    # ═══════════════════════════════════════════════════════════════
    # NOTIFICATIONS
    # ═══════════════════════════════════════════════════════════════

    async def _on_cycle_complete(self, signals, actions):
        """Called after each auto-trader cycle — sends notifications."""
        self._last_signals = signals

        if not Settings.TELEGRAM_CHAT_ID or not self.app:
            return

        chat_id = Settings.TELEGRAM_CHAT_ID

        # Notify on trades executed
        for action_type, item in actions:
            if action_type == "BUY":
                text = (
                    f"✅ <b>New Trade</b>\n\n"
                    f"📈 {item.outcome.upper()} @ {item.entry_price:.3f}\n"
                    f"💰 ${item.entry_amount_usd:.2f}\n"
                    f"🎯 Edge: {item.edge_at_entry:+.1f}%\n"
                    f"📝 {item.question[:60]}\n"
                    f"🔧 Source: {item.source}"
                )
                await self.app.bot.send_message(chat_id, text, parse_mode="HTML")

            elif action_type in ("SELL", "AUTO_EXIT", "REDUCE"):
                pnl = item.realized_pnl
                emoji = "🟢" if pnl > 0 else "🔴"
                text = (
                    f"{emoji} <b>Position Closed</b>\n\n"
                    f"📉 {item.outcome.upper()} exit @ {item.exit_price:.3f}\n"
                    f"💰 P&L: ${pnl:+.2f}\n"
                    f"📝 Reason: {item.exit_reason}\n"
                    f"📝 {item.question[:60]}"
                )
                await self.app.bot.send_message(chat_id, text, parse_mode="HTML")

        # Notify on strong signals (even if not traded)
        for sig in signals[:3]:
            if sig.signal_type.value in ("CONFIRMED_BUY", "STRONG_BUY"):
                text = (
                    f"🔥 <b>Strong Signal</b>\n\n"
                    f"📊 {sig.signal_type.value}\n"
                    f"📝 {sig.market_question[:60]}\n"
                    f"💰 {sig.outcome.upper()} @ {sig.current_price:.3f}\n"
                    f"🎯 Edge: {sig.edge_percent:+.1f}%\n"
                    f"🔧 Source: {sig.source}\n"
                    f"📝 {sig.reasoning[:100]}"
                )
                await self.app.bot.send_message(chat_id, text, parse_mode="HTML")

    async def stop(self):
        """Stop the Telegram bot."""
        if self.app:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
            print("📱 Telegram bot stopped")
