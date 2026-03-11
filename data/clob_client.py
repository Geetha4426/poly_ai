"""
CLOB Client — Orderbook, Pricing & Order Execution

Wraps py-clob-client for order placement + raw REST for orderbooks.
Handles both negRisk and standard markets.
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple

import httpx

from config.settings import Settings

try:
    from py_clob_client.client import ClobClient as _PyClobClient
    from py_clob_client.clob_types import (
        OrderArgs, MarketOrderArgs, OrderType,
        BookParams, BalanceAllowanceParams, AssetType
    )
    from py_clob_client.order_builder.constants import BUY, SELL
    CLOB_AVAILABLE = True
except ImportError:
    CLOB_AVAILABLE = False


class ClobClient:
    """
    Orderbook data + order execution for Polymarket CLOB.
    
    Two modes:
    - Read-only: fetches prices/orderbooks via REST (always available)
    - Trading: places orders via py-clob-client (requires wallet key)
    """

    def __init__(self):
        self.base_url = Settings.get_clob_url()
        self._py_clob: Optional[_PyClobClient] = None
        self._funder_address = ""
        self._session_headers = {
            "User-Agent": "polymarket-ai/1.0",
            "Accept": "application/json",
        }
        self._price_cache: Dict[str, Tuple[float, float]] = {}  # token_id -> (price, timestamp)
        self._cache_ttl = 5

    def init_trading(self):
        """Initialize the py-clob-client for live trading."""
        if not CLOB_AVAILABLE:
            print("⚠️ py-clob-client not installed — read-only mode")
            return False
        if not Settings.PRIVATE_KEY:
            print("⚠️ No private key — read-only mode")
            return False

        try:
            funder = Settings.FUNDER_ADDRESS
            if not funder or funder.startswith("your_"):
                funder = None

            self._py_clob = _PyClobClient(
                self.base_url,
                key=Settings.PRIVATE_KEY,
                chain_id=Settings.CHAIN_ID,
                signature_type=Settings.SIGNATURE_TYPE,
                funder=funder
            )

            if Settings.CLOB_RELAY_URL and Settings.CLOB_RELAY_AUTH:
                self._inject_relay_auth()

            self._py_clob.set_api_creds(
                self._py_clob.create_or_derive_api_creds()
            )

            self._funder_address = funder or self._py_clob.get_address()
            print(f"✅ CLOB trading initialized | funder={self._funder_address[:8]}...")
            return True
        except Exception as e:
            print(f"❌ CLOB init failed: {e}")
            self._py_clob = None
            return False

    def _inject_relay_auth(self):
        """Inject bearer auth into py-clob-client session for relay."""
        try:
            session = getattr(self._py_clob, "session", None)
            if session is None:
                http = getattr(self._py_clob, "http", None)
                if http:
                    session = getattr(http, "session", None)
            if session and hasattr(session, "headers"):
                session.headers["Authorization"] = f"Bearer {Settings.CLOB_RELAY_AUTH}"
        except Exception:
            pass

    # ═══════════════════════════════════════════════════════════════
    # PRICES & ORDERBOOK (REST — no auth needed)
    # ═══════════════════════════════════════════════════════════════

    async def get_price(self, token_id: str) -> Optional[float]:
        """Get mid-price for a token."""
        # Check cache
        cached = self._price_cache.get(token_id)
        if cached and time.time() - cached[1] < self._cache_ttl:
            return cached[0]

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{self.base_url}/price",
                    params={"token_id": token_id}
                )
                if resp.status_code == 200:
                    price = float(resp.json().get("price", 0))
                    self._price_cache[token_id] = (price, time.time())
                    return price
        except Exception:
            pass
        return None

    async def get_prices(self, token_ids: List[str]) -> Dict[str, float]:
        """Get prices for multiple tokens in parallel."""
        async def _get(tid):
            p = await self.get_price(tid)
            return tid, p

        tasks = [_get(tid) for tid in token_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        prices = {}
        for r in results:
            if isinstance(r, tuple) and r[1] is not None:
                prices[r[0]] = r[1]
        return prices

    async def get_orderbook(self, token_id: str) -> Optional[Dict]:
        """
        Fetch orderbook for a token.
        Returns: {bids, asks, best_bid, best_ask, spread, mid_price, ...}
        """
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{self.base_url}/book",
                    params={"token_id": token_id}
                )
                if resp.status_code != 200:
                    return None

                data = resp.json()
                bids = sorted(
                    [(float(b["price"]), float(b["size"])) for b in data.get("bids", [])],
                    key=lambda x: x[0], reverse=True
                )
                asks = sorted(
                    [(float(a["price"]), float(a["size"])) for a in data.get("asks", [])],
                    key=lambda x: x[0]
                )

                if not bids and not asks:
                    return None

                best_bid = bids[0][0] if bids else 0.0
                best_ask = asks[0][0] if asks else 1.0
                spread = best_ask - best_bid
                mid = (best_bid + best_ask) / 2 if (best_bid + best_ask) > 0 else 0.5

                bid_depth = sum(p * s for p, s in bids[:10])
                ask_depth = sum(p * s for p, s in asks[:10])
                total_depth = bid_depth + ask_depth
                imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0

                return {
                    "token_id": token_id,
                    "bids": bids,
                    "asks": asks,
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "spread": spread,
                    "spread_pct": (spread / best_ask * 100) if best_ask > 0 else 0,
                    "mid_price": mid,
                    "bid_depth": bid_depth,
                    "ask_depth": ask_depth,
                    "imbalance": imbalance,
                }
        except Exception:
            pass
        return None

    def calculate_slippage(self, orderbook: Dict, amount_usd: float, side: str) -> float:
        """Calculate expected slippage for a given order size."""
        levels = orderbook.get("asks" if side == "buy" else "bids", [])
        if not levels:
            return float("inf")

        remaining = amount_usd
        weighted_price = 0.0
        total_filled = 0.0

        for price, size in levels:
            if remaining <= 0:
                break
            level_value = price * size
            fill = min(remaining, level_value)
            weighted_price += price * fill
            total_filled += fill
            remaining -= fill

        if total_filled == 0:
            return float("inf")

        avg_price = weighted_price / total_filled
        ref_price = levels[0][0]
        return abs(avg_price - ref_price) / ref_price * 100 if ref_price > 0 else 0

    # ═══════════════════════════════════════════════════════════════
    # BALANCE & POSITIONS
    # ═══════════════════════════════════════════════════════════════

    async def get_balance(self) -> float:
        """Get USDC balance (in human-readable units)."""
        if not self._py_clob:
            return 0.0
        try:
            builder = getattr(self._py_clob, "builder", None)
            sig_type = getattr(builder, "sig_type", Settings.SIGNATURE_TYPE) if builder else Settings.SIGNATURE_TYPE
            params = BalanceAllowanceParams(
                asset_type=AssetType.COLLATERAL,
                signature_type=sig_type
            )
            bal = await asyncio.to_thread(
                self._py_clob.get_balance_allowance, params
            )
            raw = float(bal.get("balance", 0)) if isinstance(bal, dict) else float(bal or 0)
            return raw / 1e6 if raw >= 1000 else raw
        except Exception as e:
            print(f"⚠️ Balance error: {e}")
            return 0.0

    async def get_positions(self) -> List[Dict]:
        """
        Fetch open positions from Polymarket Data API.
        Returns raw position dicts with token_id, size, avgPrice, curPrice, etc.
        """
        if not self._funder_address:
            return []
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    f"{Settings.DATA_API_URL}/positions",
                    params={"user": self._funder_address.lower()}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list):
                        return [p for p in data if float(p.get("size", 0)) > 0.001]
        except Exception as e:
            print(f"⚠️ Positions fetch error: {e}")
        return []

    # ═══════════════════════════════════════════════════════════════
    # ORDER EXECUTION
    # ═══════════════════════════════════════════════════════════════

    async def buy(self, token_id: str, amount_usd: float,
                  price: Optional[float] = None,
                  order_type: str = "FOK") -> Dict:
        """
        Buy tokens.
        
        Args:
            token_id: The CLOB token ID to buy
            amount_usd: Amount in USDC to spend
            price: Limit price (if None, uses best ask + slippage)
            order_type: FOK (fill-or-kill), GTC (good-til-cancel), FAK (fill-and-kill)
        
        Returns: {"success": bool, "order_id": str, "error": str}
        """
        if not self._py_clob:
            return {"success": False, "error": "Trading not initialized"}

        try:
            # Get price if not provided
            if price is None:
                book = await self.get_orderbook(token_id)
                if book and book["asks"]:
                    price = book["best_ask"]
                else:
                    p = await self.get_price(token_id)
                    if p:
                        price = min(0.99, p + 0.01)
                    else:
                        return {"success": False, "error": "No price available"}

            # Clamp price
            price = max(0.01, min(0.99, price))

            # Calculate size: shares = amount / price
            size = amount_usd / price

            otype = {
                "FOK": OrderType.FOK,
                "GTC": OrderType.GTC,
                "FAK": OrderType.FOK,  # FAK maps to FOK in py-clob-client
            }.get(order_type, OrderType.FOK)

            order_args = OrderArgs(
                price=round(price, 2),
                size=round(size, 2),
                side=BUY,
                token_id=token_id,
            )

            resp = await asyncio.to_thread(
                self._py_clob.create_and_post_order, order_args, otype
            )

            if resp and isinstance(resp, dict):
                order_id = resp.get("orderID", resp.get("id", ""))
                return {"success": True, "order_id": order_id, "price": price, "size": size}
            return {"success": True, "order_id": str(resp), "price": price, "size": size}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def sell(self, token_id: str, size: float,
                   price: Optional[float] = None,
                   order_type: str = "FOK") -> Dict:
        """
        Sell tokens.
        
        Args:
            token_id: The CLOB token ID to sell
            size: Number of shares to sell
            price: Limit price (if None, uses best bid - slippage)
            order_type: FOK, GTC, or FAK
        """
        if not self._py_clob:
            return {"success": False, "error": "Trading not initialized"}

        try:
            if price is None:
                book = await self.get_orderbook(token_id)
                if book and book["bids"]:
                    price = book["best_bid"]
                else:
                    p = await self.get_price(token_id)
                    if p:
                        price = max(0.01, p - 0.01)
                    else:
                        return {"success": False, "error": "No price available"}

            price = max(0.01, min(0.99, price))

            otype = {
                "FOK": OrderType.FOK,
                "GTC": OrderType.GTC,
                "FAK": OrderType.FOK,
            }.get(order_type, OrderType.FOK)

            order_args = OrderArgs(
                price=round(price, 2),
                size=round(size, 2),
                side=SELL,
                token_id=token_id,
            )

            resp = await asyncio.to_thread(
                self._py_clob.create_and_post_order, order_args, otype
            )

            if resp and isinstance(resp, dict):
                order_id = resp.get("orderID", resp.get("id", ""))
                return {"success": True, "order_id": order_id, "price": price, "size": size}
            return {"success": True, "order_id": str(resp), "price": price, "size": size}

        except Exception as e:
            return {"success": False, "error": str(e)}


# Singleton
_clob_client: Optional[ClobClient] = None


def get_clob_client() -> ClobClient:
    global _clob_client
    if _clob_client is None:
        _clob_client = ClobClient()
    return _clob_client
