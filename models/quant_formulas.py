"""
Quant Formulas — Microstructure & technical indicators.

Ported from 5min_trade with additions for prediction markets.
These are stateless functions — no API calls, pure math.
"""

import math
from typing import List, Tuple, Optional


# ═══════════════════════════════════════════════════════════════
# ORDERBOOK MICROSTRUCTURE
# ═══════════════════════════════════════════════════════════════

def microprice(best_bid: float, best_ask: float,
               bid_size: float, ask_size: float) -> float:
    """
    Volume-weighted mid price.
    Better fair value estimate than simple mid when book is imbalanced.
    microprice = (bid * ask_size + ask * bid_size) / (bid_size + ask_size)
    """
    total = bid_size + ask_size
    if total <= 0:
        return (best_bid + best_ask) / 2
    return (best_bid * ask_size + best_ask * bid_size) / total


def effective_spread(best_bid: float, best_ask: float) -> float:
    """Effective spread as percentage of mid price."""
    mid = (best_bid + best_ask) / 2
    if mid <= 0:
        return 0
    return (best_ask - best_bid) / mid * 100


def book_imbalance(bids: List[Tuple[float, float]],
                   asks: List[Tuple[float, float]],
                   levels: int = 5) -> float:
    """
    Order book imbalance [-1, 1].
    +1 = all buy pressure, -1 = all sell pressure.
    """
    bid_vol = sum(s for _, s in bids[:levels])
    ask_vol = sum(s for _, s in asks[:levels])
    total = bid_vol + ask_vol
    if total <= 0:
        return 0
    return (bid_vol - ask_vol) / total


def vwap(levels: List[Tuple[float, float]]) -> float:
    """Volume-weighted average price across orderbook levels."""
    total_vol = sum(s for _, s in levels)
    if total_vol <= 0:
        return 0
    return sum(p * s for p, s in levels) / total_vol


def slippage_cost(levels: List[Tuple[float, float]],
                  amount_usd: float) -> Tuple[float, float]:
    """
    Calculate average execution price and slippage for a given order.
    Returns: (avg_price, slippage_pct)
    """
    remaining = amount_usd
    weighted_price = 0.0
    filled = 0.0

    for price, size in levels:
        if remaining <= 0:
            break
        level_val = price * size
        fill = min(remaining, level_val)
        weighted_price += price * fill
        filled += fill
        remaining -= fill

    if filled <= 0:
        return 0, float("inf")

    avg = weighted_price / filled
    ref = levels[0][0] if levels else avg
    slip = abs(avg - ref) / ref * 100 if ref > 0 else 0
    return avg, slip


# ═══════════════════════════════════════════════════════════════
# PRICE INDICATORS
# ═══════════════════════════════════════════════════════════════

def rsi(prices: List[float], period: int = 14) -> float:
    """
    Relative Strength Index [0, 100].
    >70 = overbought, <30 = oversold.
    """
    if len(prices) < period + 1:
        return 50  # neutral

    gains = []
    losses = []
    for i in range(1, len(prices)):
        change = prices[i] - prices[i - 1]
        gains.append(max(0, change))
        losses.append(max(0, -change))

    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period

    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def bollinger_bands(prices: List[float], period: int = 20,
                     num_std: float = 2.0) -> Tuple[float, float, float]:
    """
    Bollinger Bands: (upper, middle, lower).
    Price near upper = potentially overpriced, near lower = underpriced.
    """
    if len(prices) < period:
        mid = prices[-1] if prices else 0.5
        return mid + 0.1, mid, mid - 0.1

    window = prices[-period:]
    mid = sum(window) / len(window)
    variance = sum((p - mid) ** 2 for p in window) / len(window)
    std = math.sqrt(variance)

    return mid + num_std * std, mid, mid - num_std * std


def ema(prices: List[float], period: int = 12) -> float:
    """Exponential Moving Average."""
    if not prices:
        return 0
    if len(prices) == 1:
        return prices[0]

    k = 2 / (period + 1)
    result = prices[0]
    for p in prices[1:]:
        result = p * k + result * (1 - k)
    return result


def momentum(prices: List[float], period: int = 10) -> float:
    """Price momentum: current / past price - 1."""
    if len(prices) < period + 1:
        return 0
    return (prices[-1] / prices[-period - 1]) - 1 if prices[-period - 1] != 0 else 0


# ═══════════════════════════════════════════════════════════════
# PREDICTION MARKET SPECIFIC
# ═══════════════════════════════════════════════════════════════

def kelly_criterion(win_prob: float, odds: float,
                     fraction: float = 0.25) -> float:
    """
    Fractional Kelly position sizing.
    
    Args:
        win_prob: estimated probability of winning
        odds: payout ratio (e.g., price 0.60 → odds = 0.40/0.60 = 0.667)
        fraction: fraction of Kelly to use (0.25 = quarter Kelly)
    
    Returns: fraction of bankroll to bet
    """
    q = 1 - win_prob
    if odds <= 0:
        return 0
    kelly = (win_prob * odds - q) / odds
    return max(0, min(1, kelly * fraction))


def edge_from_prices(estimated_prob: float, market_price: float) -> float:
    """Calculate edge percentage."""
    return (estimated_prob - market_price) * 100


def expected_value(win_prob: float, price: float) -> float:
    """
    Expected value of buying YES at given price.
    EV = win_prob * (1 - price) - (1 - win_prob) * price
    """
    return win_prob * (1 - price) - (1 - win_prob) * price


def sharpe_like_ratio(edge_pct: float, confidence: float,
                       price_volatility: float = 0.1) -> float:
    """
    Sharpe-like signal quality measure.
    Higher = better risk-adjusted edge.
    """
    if price_volatility <= 0:
        return edge_pct * confidence
    return (edge_pct * confidence) / (price_volatility * 100)


# ═══════════════════════════════════════════════════════════════
# KAPPA (κ) ORDERBOOK STEEPNESS
# Based on MarikWeb3 research:
#   P(fill|δ) = e^(-κδ)  where δ = distance from mid
#   Optimal limit order placement at 1/κ from mid
#   High κ = thin/steep book, low κ = deep/flat book
# ═══════════════════════════════════════════════════════════════

def estimate_kappa(levels: List[Tuple[float, float]],
                    mid_price: float) -> float:
    """
    Estimate κ (kappa) — orderbook steepness parameter.

    Fits exponential decay to cumulative depth vs distance from mid.
    Higher κ = steeper book (less liquidity away from mid).

    Args:
        levels: orderbook levels [(price, size), ...]
        mid_price: midpoint price

    Returns: κ value (typically 5-200 for prediction markets)
    """
    if not levels or mid_price <= 0:
        return 50.0  # default moderate steepness

    # Calculate distance and cumulative size at each level
    points = []
    for price, size in levels:
        delta = abs(price - mid_price)
        if delta > 0 and size > 0:
            points.append((delta, size))

    if len(points) < 2:
        return 50.0

    # Sort by distance
    points.sort(key=lambda x: x[0])

    # Fit: log(size) ≈ -κ * delta + constant
    # Use simple linear regression in log space
    total_size = sum(s for _, s in points)
    if total_size <= 0:
        return 50.0

    # Normalize sizes to fractions
    fracs = [(d, s / total_size) for d, s in points]

    # Cumulative survival: P(depth > delta)
    cum_sum = 0
    survival = []
    for d, f in fracs:
        cum_sum += f
        remaining = max(1e-10, 1 - cum_sum)
        survival.append((d, remaining))

    # Linear regression on log(survival) vs delta
    n = len(survival)
    if n < 2:
        return 50.0

    sum_x = sum(d for d, _ in survival)
    sum_y = sum(math.log(s) for _, s in survival)
    sum_xy = sum(d * math.log(s) for d, s in survival)
    sum_x2 = sum(d * d for d, _ in survival)

    denom = n * sum_x2 - sum_x * sum_x
    if abs(denom) < 1e-10:
        return 50.0

    slope = (n * sum_xy - sum_x * sum_y) / denom
    kappa = max(1.0, min(500.0, -slope))  # κ = -slope, clamped

    return kappa


def optimal_limit_offset(kappa: float) -> float:
    """
    Optimal limit order distance from mid price.

    At 1/κ from mid: maximizes fill probability × price improvement.
    E.g., κ=50 → place limit 0.02 (2 cents) from mid.

    Returns: optimal distance in price units (e.g., 0.02 = 2 cents)
    """
    if kappa <= 0:
        return 0.01
    return 1.0 / kappa


def fill_probability(offset: float, kappa: float) -> float:
    """
    Probability of a limit order getting filled at given offset from mid.

    P(fill|δ) = e^(-κδ)
    At offset = 1/κ: P(fill) ≈ 36.8% (optimal tradeoff)
    At offset = 0: P(fill) = 100% (market order)
    """
    return math.exp(-kappa * max(0, offset))


def kappa_adjusted_price(mid_price: float, kappa: float,
                          side: str = "buy") -> float:
    """
    Compute kappa-optimal limit order price.

    For buying: mid - 1/κ  (place bid below mid)
    For selling: mid + 1/κ  (place ask above mid)
    """
    offset = optimal_limit_offset(kappa)
    if side.lower() == "buy":
        return max(0.01, mid_price - offset)
    else:
        return min(0.99, mid_price + offset)


def kappa_inventory_risk(kappa: float, position_size: float,
                          mid_price: float) -> float:
    """
    Inventory risk score based on kappa.

    Thin books (high κ) amplify inventory risk.
    Score 0-1: 0 = safe, 1 = dangerous.
    """
    if kappa <= 0 or mid_price <= 0:
        return 0.5
    # Time to unwind = position_value / (liquidity_rate)
    # Liquidity rate inversely proportional to kappa
    position_value = position_size * mid_price
    liquidity_rate = 1000 / kappa  # rough estimate: thinner book → slower fills
    time_to_unwind = position_value / max(liquidity_rate, 1)
    # Normalize to 0-1
    return min(1.0, time_to_unwind / 100)


def time_decay_factor(hours_to_close: float,
                       half_life_hours: float = 48) -> float:
    """
    Time decay multiplier for edge sizing.
    Edge is worth more when market closes soon (less time for price to correct).
    Returns value in [0.5, 2.0].
    """
    if hours_to_close <= 0:
        return 2.0
    ratio = half_life_hours / hours_to_close
    return max(0.5, min(2.0, ratio))


def combine_probabilities(probs: List[Tuple[float, float]]) -> float:
    """
    Combine multiple probability estimates with confidence weights.
    
    Args:
        probs: list of (probability, confidence_weight) tuples
    
    Returns: weighted average probability
    """
    total_weight = sum(w for _, w in probs)
    if total_weight <= 0:
        return 0.5
    return sum(p * w for p, w in probs) / total_weight


def log_odds(prob: float) -> float:
    """Convert probability to log-odds (logit)."""
    prob = max(0.001, min(0.999, prob))
    return math.log(prob / (1 - prob))


def from_log_odds(logit: float) -> float:
    """Convert log-odds back to probability."""
    return 1 / (1 + math.exp(-logit))
