"""Synthetic derivatives engine for binary prediction markets.

Pure math, no network, no I/O. Given a market's current YES price (a probability
in 0..1) and an optional order book, model synthetic options (calls/puts) on the
market probability: payoff curves, break-even, max profit/loss, a documented fair
price, implied probability, and Greeks (delta, gamma, theta, vega).

See MODEL.md for the modeling assumptions. This is analytics, not a trading claim.
"""

from .engine import (
    Side,
    OptionType,
    OrderBook,
    OrderLevel,
    payoff_curve,
    intrinsic_payoff,
    break_even,
    max_profit_loss,
    fair_price,
    held_to_resolution_price,
    pricing_basis,
    implied_probability,
    mid_from_book,
    half_spread,
    prob_of_profit,
    greeks,
    greek_curve,
    analyze,
)

__all__ = [
    "Side",
    "OptionType",
    "OrderBook",
    "OrderLevel",
    "payoff_curve",
    "intrinsic_payoff",
    "break_even",
    "max_profit_loss",
    "fair_price",
    "held_to_resolution_price",
    "pricing_basis",
    "implied_probability",
    "mid_from_book",
    "half_spread",
    "prob_of_profit",
    "greeks",
    "greek_curve",
    "analyze",
]
