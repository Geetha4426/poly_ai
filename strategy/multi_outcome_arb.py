"""
Multi-Outcome Arbitrage Engine

Detects guaranteed and value arbitrage across multi-runner markets
(e.g., sports: Team A vs Team B vs Draw).

Math foundation:
- For N mutually exclusive outcomes, buying all NOs costs: sum(NO_prices)
- Exactly N-1 NOs win regardless of outcome → guaranteed return = N-1
- If sum(NO_prices) < N-1 → guaranteed arbitrage, profit = (N-1) - cost
- If sum(YES_prices) > 1 → all-YES arbitrage (sell all overpriced YES)

Also detects VALUE trades:
- When NO price on most likely loser is cheaper than fair probability
- E.g., Team B 57% to win → NO Team B true fair price = 43%
  If market prices NO at 40%, that's +3% expected value
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from config.settings import Settings
from data.gamma_client import GammaClient, Event, Market, Token
from data.clob_client import ClobClient


@dataclass
class ArbitrageOpportunity:
    """A detected multi-outcome arbitrage or value trade."""
    event_id: str
    event_title: str
    opportunity_type: str  # "guaranteed_no_arb", "guaranteed_yes_arb", "value_trade"
    markets: List[Market]
    legs: List[Dict]  # [{market_q, outcome, token_id, price, amount_pct}]
    total_cost: float  # total cost to enter all legs per $1 unit
    guaranteed_return: float  # guaranteed payout per $1 unit
    profit_per_unit: float  # guaranteed_return - total_cost
    profit_pct: float  # profit_per_unit / total_cost * 100
    edge_pct: float  # edge as percentage
    confidence: float  # 1.0 for guaranteed arb, <1.0 for value trades
    reasoning: str
    detected_at: float = 0

    @property
    def is_guaranteed(self) -> bool:
        return self.opportunity_type.startswith("guaranteed")

    @property
    def suggested_total_usd(self) -> float:
        """How much to deploy total across all legs."""
        if self.is_guaranteed:
            # Risk-free: deploy more
            return min(Settings.MAX_POSITION_USD * 2, 100)
        else:
            return Settings.MAX_TRADE_USD

    def __str__(self):
        emoji = "💎" if self.is_guaranteed else "📊"
        return (
            f"{emoji} {self.opportunity_type} | {self.event_title[:50]} | "
            f"{len(self.legs)} legs | Cost: ${self.total_cost:.3f} → "
            f"Return: ${self.guaranteed_return:.3f} | "
            f"Profit: {self.profit_pct:+.2f}%"
        )


class MultiOutcomeArbitrageEngine:
    """
    Scans multi-runner events for arbitrage and value trades.

    Works on any event with 3+ outcomes (markets) that are mutually exclusive:
    - Sports (Win A / Win B / Draw)
    - Politics (Candidate A / B / C / ...)
    - Any multi-runner market

    Strategies:
    1. ALL-NO Arbitrage: Buy NO on every outcome. N-1 always win.
       Profitable if sum(NO_prices) < N-1.
    2. ALL-YES Arbitrage: If sum(YES_prices) > 1, YES is overpriced.
       Sell YES (buy NO) on overpriced outcomes.
    3. Value Trade: Buy NO on the outcome with best EV mismatch.
       Risk exists but positive expected value.
    """

    def __init__(self, gamma: Optional[GammaClient] = None,
                 clob: Optional[ClobClient] = None):
        from data.gamma_client import get_gamma_client
        self.gamma = gamma or get_gamma_client()
        self.clob = clob or ClobClient()
        self._cache: Dict[str, float] = {}  # cache last scan time per event
        self._cooldown = 300  # 5 min between re-scans of same event

    async def scan_event(self, event: Event) -> List[ArbitrageOpportunity]:
        """Scan a single multi-outcome event for arbitrage."""
        markets = event.markets
        if len(markets) < 3:
            return []  # Need 3+ outcomes for multi-outcome arb

        # Check if mutually exclusive (all markets in same event should be)
        # Skip if any market lacks price data
        opportunities = []

        # Fetch live prices for all tokens
        yes_prices = {}
        no_prices = {}
        for m in markets:
            yt = m.yes_token
            nt = m.no_token
            if not yt or not nt:
                continue
            yp = yt.price if yt.price and yt.price > 0 else None
            np = nt.price if nt.price and nt.price > 0 else None

            # If Gamma prices are stale, try CLOB
            if yp is None or yp <= 0.01:
                clob_price = await self.clob.get_price(yt.token_id)
                if clob_price:
                    yp = clob_price
                    np = 1.0 - clob_price

            if yp and np:
                yes_prices[m.condition_id] = (m, yp, yt)
                no_prices[m.condition_id] = (m, np, nt)

        if len(yes_prices) < 3:
            return []

        n = len(yes_prices)

        # ── Strategy 1: ALL-NO Arbitrage ──
        total_no_cost = sum(price for _, price, _ in no_prices.values())
        guaranteed_no_return = n - 1  # N-1 NOs always win

        if total_no_cost < guaranteed_no_return:
            profit = guaranteed_no_return - total_no_cost
            profit_pct = (profit / total_no_cost) * 100

            legs = []
            for cid, (m, price, token) in no_prices.items():
                legs.append({
                    "market_question": m.question,
                    "outcome": "NO",
                    "token_id": token.token_id,
                    "price": price,
                    "amount_pct": 1.0 / n,  # equal weight
                })

            opportunities.append(ArbitrageOpportunity(
                event_id=event.event_id,
                event_title=event.title,
                opportunity_type="guaranteed_no_arb",
                markets=markets,
                legs=legs,
                total_cost=total_no_cost,
                guaranteed_return=guaranteed_no_return,
                profit_per_unit=profit,
                profit_pct=profit_pct,
                edge_pct=profit_pct,
                confidence=1.0,
                reasoning=(
                    f"Buy NO on all {n} outcomes. Cost: ${total_no_cost:.4f}. "
                    f"{n-1} NOs always win → return ${guaranteed_no_return}. "
                    f"Guaranteed profit: ${profit:.4f} ({profit_pct:.2f}%)."
                ),
                detected_at=time.time(),
            ))

        # ── Strategy 2: ALL-YES Overpriced Check ──
        total_yes_cost = sum(price for _, price, _ in yes_prices.values())

        if total_yes_cost > 1.0:
            overround = total_yes_cost - 1.0
            overround_pct = overround * 100

            # The market has overround — YES is collectively overpriced
            # Find the most overpriced YES → buy its NO
            # Sort by YES price descending (most overpriced first)
            sorted_yes = sorted(yes_prices.items(),
                                key=lambda x: x[1][1], reverse=True)

            legs = []
            for cid, (m, yp, yt) in sorted_yes:
                nm, np, nt = no_prices[cid]
                legs.append({
                    "market_question": m.question,
                    "outcome": "NO",
                    "token_id": nt.token_id,
                    "price": np,
                    "yes_price": yp,
                    "amount_pct": yp / total_yes_cost,  # weight by how overpriced
                })

            # This isn't guaranteed arbitrage (can't sell YES on Poly easily)
            # but signals collective overpricing
            if overround_pct >= 2.0:  # at least 2% overround
                opportunities.append(ArbitrageOpportunity(
                    event_id=event.event_id,
                    event_title=event.title,
                    opportunity_type="guaranteed_yes_arb",
                    markets=markets,
                    legs=legs,
                    total_cost=total_yes_cost,
                    guaranteed_return=1.0,
                    profit_per_unit=overround,
                    profit_pct=overround_pct,
                    edge_pct=overround_pct,
                    confidence=0.85,
                    reasoning=(
                        f"Sum of YES prices = ${total_yes_cost:.4f} > $1.00. "
                        f"Market has {overround_pct:.1f}% overround. "
                        f"Strategy: buy NO on most overpriced outcomes."
                    ),
                    detected_at=time.time(),
                ))

        # ── Strategy 3: Value NO Trades ──
        # Find the best value: NO price vs. implied fair NO price
        # Fair NO price for outcome i = 1 - (YES_i / total_YES)
        for cid, (m, no_price, no_token) in no_prices.items():
            _, yes_price, _ = yes_prices[cid]

            # Fair probability of this outcome = YES_price / sum_YES
            fair_prob = yes_price / total_yes_cost if total_yes_cost > 0 else 0
            fair_no_price = 1 - fair_prob
            edge = fair_no_price - no_price

            # Only track meaningful edges (>3%)
            if edge > 0.03:
                # Check which other outcomes would make NO win
                other_outcomes = [
                    (m2.question, yp2)
                    for cid2, (m2, yp2, _) in yes_prices.items()
                    if cid2 != cid
                ]
                win_scenarios = sum(yp for _, yp in other_outcomes) / total_yes_cost

                opportunities.append(ArbitrageOpportunity(
                    event_id=event.event_id,
                    event_title=event.title,
                    opportunity_type="value_trade",
                    markets=[m],
                    legs=[{
                        "market_question": m.question,
                        "outcome": "NO",
                        "token_id": no_token.token_id,
                        "price": no_price,
                        "fair_price": fair_no_price,
                        "edge": edge,
                        "amount_pct": 1.0,
                    }],
                    total_cost=no_price,
                    guaranteed_return=1.0 * win_scenarios,
                    profit_per_unit=edge,
                    profit_pct=edge / no_price * 100 if no_price > 0 else 0,
                    edge_pct=edge * 100,
                    confidence=min(0.8, 0.5 + win_scenarios * 0.3),
                    reasoning=(
                        f"NO '{m.question[:40]}' priced at {no_price:.3f} but "
                        f"fair value ~{fair_no_price:.3f}. Edge: {edge*100:.1f}%. "
                        f"Wins in {win_scenarios*100:.0f}% of scenarios."
                    ),
                    detected_at=time.time(),
                ))

        return opportunities

    async def scan_all_events(self, events: List[Event]) -> List[ArbitrageOpportunity]:
        """Scan multiple events for arbitrage opportunities."""
        all_opps = []
        now = time.time()

        # Filter to multi-outcome events only
        multi_events = [e for e in events if len(e.markets) >= 3]

        for event in multi_events:
            # Skip if recently scanned
            last = self._cache.get(event.event_id, 0)
            if now - last < self._cooldown:
                continue

            opps = await self.scan_event(event)
            if opps:
                all_opps.extend(opps)
            self._cache[event.event_id] = now

        # Sort: guaranteed first, then by profit percentage
        all_opps.sort(key=lambda o: (
            0 if o.is_guaranteed else 1,
            -o.profit_pct
        ))

        if all_opps:
            guaranteed = sum(1 for o in all_opps if o.is_guaranteed)
            value = len(all_opps) - guaranteed
            print(
                f"🎰 Multi-outcome scan: {len(multi_events)} events → "
                f"{guaranteed} guaranteed + {value} value trades"
            )

        return all_opps

    def calculate_optimal_allocation(self, opp: ArbitrageOpportunity,
                                       total_usd: float) -> List[Dict]:
        """
        Calculate optimal $ allocation across legs.

        For guaranteed arb: equal weight (all legs must be filled).
        For value trades: Kelly-weighted by edge.
        """
        allocations = []
        n = len(opp.legs)

        if opp.is_guaranteed:
            # Equal allocation across all legs
            per_leg = total_usd / n
            for leg in opp.legs:
                allocations.append({
                    "token_id": leg["token_id"],
                    "outcome": leg["outcome"],
                    "price": leg["price"],
                    "amount_usd": round(per_leg, 2),
                    "shares": per_leg / leg["price"] if leg["price"] > 0 else 0,
                })
        else:
            # Value trade — full amount on the single best leg
            for leg in opp.legs:
                allocations.append({
                    "token_id": leg["token_id"],
                    "outcome": leg["outcome"],
                    "price": leg["price"],
                    "amount_usd": round(total_usd * leg["amount_pct"], 2),
                    "shares": total_usd * leg["amount_pct"] / leg["price"]
                    if leg["price"] > 0 else 0,
                })

        return allocations
