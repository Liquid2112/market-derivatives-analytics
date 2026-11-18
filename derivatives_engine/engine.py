"""Synthetic derivatives engine for binary prediction markets.

The underlying is a prediction-market probability ``p`` in [0, 1] (the YES price).
At resolution the event settles to 1 (YES) or 0 (NO). We model synthetic European
options whose strike ``K`` is a probability level in [0, 1] and chart everything
across the continuous 0..1 outcome range so a UI can plot it.

Two things are kept strictly separate:

* **Intrinsic payoff** at expiry -- model-free. A synthetic call pays
  ``max(x - K, 0)`` and a put pays ``max(K - x, 0)`` for a settlement value ``x``.
* **Fair price and Greeks before expiry** -- these need a model. We use the
  Bachelier (normal) model with the probability itself as the underlying: the
  settlement value is treated as ``x ~ Normal(p, sigma)``, where ``sigma`` is a
  dispersion ("vol" analogue) that shrinks toward resolution. This gives smooth,
  bounded, closed-form Greeks. See MODEL.md for why and for the honest caveats.

No network, no file I/O, no global state. Every public function is pure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

_SQRT_2PI = math.sqrt(2.0 * math.pi)
# shortcut: Bachelier model needs sigma > 0; clamp to this floor near expiry so
# Greeks stay finite. Revisit if callers need an exact-expiry (sigma=0) branch.
_SIGMA_FLOOR = 1e-9


class OptionType(str, Enum):
    """Synthetic option kind on the market probability."""

    CALL = "call"
    PUT = "put"


class Side(str, Enum):
    """Whether the analytics are from the long (buyer) or short (seller) view."""

    LONG = "long"
    SHORT = "short"


@dataclass(frozen=True)
class OrderLevel:
    """A single resting order-book level: a price in [0, 1] and a size >= 0."""

    price: float
    size: float


@dataclass(frozen=True)
class OrderBook:
    """Bid/ask ladders for a market. Highest bid and lowest ask are the top.

    Both lists are optional; an empty book is allowed. Levels are not required
    to be pre-sorted -- the engine finds the best bid/ask itself.
    """

    bids: list[OrderLevel] = field(default_factory=list)
    asks: list[OrderLevel] = field(default_factory=list)

    def best_bid(self) -> Optional[float]:
        prices = [lv.price for lv in self.bids if lv.size > 0.0]
        return max(prices) if prices else None

    def best_ask(self) -> Optional[float]:
        prices = [lv.price for lv in self.asks if lv.size > 0.0]
        return min(prices) if prices else None


# ---------------------------------------------------------------------------
# Standard normal helpers (stdlib only)
# ---------------------------------------------------------------------------

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _check_prob(name: str, value: float) -> float:
    if not isinstance(value, (int, float)) or math.isnan(value):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return float(value)


def _coerce_option_type(option_type) -> OptionType:
    if isinstance(option_type, OptionType):
        return option_type
    return OptionType(str(option_type).lower())


def _coerce_side(side) -> Side:
    if isinstance(side, Side):
        return side
    return Side(str(side).lower())


# ---------------------------------------------------------------------------
# Intrinsic payoff (model-free)
# ---------------------------------------------------------------------------

def intrinsic_payoff(settlement: float, strike: float, option_type) -> float:
    """Payoff of one synthetic option at a given settlement value (no premium).

    ``settlement`` is the outcome value in [0, 1] (1 if YES resolves, 0 if NO;
    intermediate values are only for plotting the curve). Returns the gross
    payoff, always >= 0 for a long holder.
    """
    strike = _check_prob("strike", strike)
    settlement = _check_prob("settlement", settlement)
    ot = _coerce_option_type(option_type)
    if ot is OptionType.CALL:
        return max(settlement - strike, 0.0)
    return max(strike - settlement, 0.0)


def payoff_curve(
    strike: float,
    option_type,
    side="long",
    premium: Optional[float] = None,
    num_points: int = 101,
):
    """Payoff across the 0..1 outcome range, as plottable arrays.

    Returns a dict with ``x`` (outcome grid, 0..1) and ``payoff`` (net payoff per
    contract). If ``premium`` is given, payoff is net of premium and flips sign
    for a short side; if omitted, the gross intrinsic payoff is returned.
    """
    strike = _check_prob("strike", strike)
    ot = _coerce_option_type(option_type)
    sd = _coerce_side(side)
    if num_points < 2:
        raise ValueError("num_points must be >= 2")

    xs = [i / (num_points - 1) for i in range(num_points)]
    payoffs: list[float] = []
    for x in xs:
        gross = intrinsic_payoff(x, strike, ot)
        if premium is None:
            net = gross
        else:
            net = gross - premium
            if sd is Side.SHORT:
                net = -net
        payoffs.append(net)
    return {"x": xs, "payoff": payoffs}


def break_even(strike: float, option_type, premium: float):
    """Break-even outcome value(s) for a long position paying ``premium``.

    Returns a list of outcome values in [0, 1] where net payoff crosses zero.
    A premium of 0 degenerates to the strike. A premium that can never be
    recovered within [0, 1] yields an empty list.
    """
    strike = _check_prob("strike", strike)
    ot = _coerce_option_type(option_type)
    if premium < 0.0:
        raise ValueError("premium must be >= 0")

    if ot is OptionType.CALL:
        # payoff = max(x-K,0) - premium = 0  ->  x = K + premium
        be = strike + premium
    else:
        # payoff = max(K-x,0) - premium = 0  ->  x = K - premium
        be = strike - premium
    if be < 0.0 or be > 1.0:
        return []
    return [be]


def max_profit_loss(strike: float, option_type, side, premium: float):
    """Max profit and max loss over the [0, 1] outcome range for one contract.

    Signs are from the holder of ``side``. Loss is reported as a negative number.
    """
    strike = _check_prob("strike", strike)
    ot = _coerce_option_type(option_type)
    sd = _coerce_side(side)
    if premium < 0.0:
        raise ValueError("premium must be >= 0")

    # Evaluate net payoff at the two outcome extremes; payoff is piecewise linear
    # and monotonic in x on [0,1], so the extremes bound profit and loss.
    nets = []
    for x in (0.0, 1.0):
        gross = intrinsic_payoff(x, strike, ot)
        net = gross - premium
        if sd is Side.SHORT:
            net = -net
        nets.append(net)
    return {"max_profit": max(nets), "max_loss": min(nets)}


def prob_of_profit(
    p: float,
    strike: float,
    option_type,
    side,
    premium: float,
    sigma: float = 0.15,
    tau: float = 1.0,
    book: Optional[OrderBook] = None,
) -> float:
    """Model-based probability the position ends in profit, in [0, 1].

    Under the pricing model the settlement value is
    ``x ~ Normal(p, sigma_eff)`` with ``sigma_eff = effective_sigma * sqrt(tau)``
    (the same dispersion the Greeks use). The position is profitable on exactly
    one side of its break-even (payoff is piecewise linear and monotonic on each
    side of the strike), so ``PoP`` is the normal probability mass on that side.

    This is a *model-based* figure, not a frequency: it inherits the Bachelier
    assumptions in MODEL.md, including a little mass outside [0, 1].
    """
    p = _check_prob("p", p)
    strike = _check_prob("strike", strike)
    ot = _coerce_option_type(option_type)
    sd = _coerce_side(side)
    if premium < 0.0:
        raise ValueError("premium must be >= 0")
    if tau <= 0.0 or tau > 1.0:
        raise ValueError("tau must be in (0, 1]")

    sig = max(_effective_sigma(sigma, p, book) * math.sqrt(tau), _SIGMA_FLOOR)

    be = break_even(strike, ot, premium)
    if not be:
        # Break-even lies outside [0, 1]: a long position can never recover the
        # premium within the settlement range, so it is never in profit; the
        # short counterparty always keeps the premium.
        return 0.0 if sd is Side.LONG else 1.0

    threshold = be[0]
    # P(x <= threshold) under Normal(p, sig).
    cdf_below = _norm_cdf((threshold - p) / sig)

    # Long call / short put profit when x is ABOVE break-even; long put /
    # short call profit when x is BELOW it. (Short flips the long region.)
    long_profits_above = ot is OptionType.CALL
    profits_above = long_profits_above if sd is Side.LONG else not long_profits_above

    return (1.0 - cdf_below) if profits_above else cdf_below


# ---------------------------------------------------------------------------
# Order-book derived quantities
# ---------------------------------------------------------------------------

def mid_from_book(book: OrderBook) -> Optional[float]:
    """Mid price from the top of book, or None if one side is empty."""
    bb = book.best_bid()
    ba = book.best_ask()
    if bb is None or ba is None:
        return None
    return 0.5 * (bb + ba)


def _effective_sigma(sigma: float, p: float, book: Optional[OrderBook]) -> float:
    """Dispersion used for pricing.

    Starts from the caller's ``sigma`` and, if an order book is supplied, widens
    it by the half-spread -- a wider book signals more disagreement about the
    settlement, i.e. more "vol". Kept simple and documented in MODEL.md.
    """
    sig = sigma
    if book is not None:
        bb = book.best_bid()
        ba = book.best_ask()
        if bb is not None and ba is not None:
            sig += 0.5 * max(ba - bb, 0.0)
    return max(sig, _SIGMA_FLOOR)


# ---------------------------------------------------------------------------
# Pricing and Greeks (Bachelier / normal model on the probability)
# ---------------------------------------------------------------------------

def fair_price(
    p: float,
    strike: float,
    option_type,
    sigma: float = 0.15,
    book: Optional[OrderBook] = None,
) -> float:
    """Model fair value of one synthetic option before resolution.

    Bachelier (normal) model with the probability ``p`` as the underlying and
    ``sigma`` the dispersion of the settlement value. See MODEL.md.
    """
    p = _check_prob("p", p)
    strike = _check_prob("strike", strike)
    ot = _coerce_option_type(option_type)
    if sigma < 0.0:
        raise ValueError("sigma must be >= 0")

    sig = _effective_sigma(sigma, p, book)
    d = (p - strike) / sig
    call = (p - strike) * _norm_cdf(d) + sig * _norm_pdf(d)
    if ot is OptionType.CALL:
        return call
    # put-call parity for Bachelier: P = C - (p - K)
    return call - (p - strike)


def implied_probability(p: float, book: Optional[OrderBook] = None) -> float:
    """The market's own implied probability of YES.

    For a binary prediction market the YES price *is* the implied probability.
    When an order book is given we prefer its mid as a cleaner estimate, falling
    back to ``p``.
    """
    p = _check_prob("p", p)
    if book is not None:
        mid = mid_from_book(book)
        if mid is not None:
            return _check_prob("book mid", mid)
    return p


def greeks(
    p: float,
    strike: float,
    option_type,
    sigma: float = 0.15,
    tau: float = 1.0,
    book: Optional[OrderBook] = None,
) -> dict:
    """Analytic Greeks for one synthetic option under the Bachelier model.

    * ``delta``  = d(price)/dp               -- bounded in [0, 1] (call) / [-1, 0] (put)
    * ``gamma``  = d2(price)/dp2             -- phi(d)/sigma, peaks at p == strike
    * ``vega``   = d(price)/dsigma           -- phi(d) (>= 0), per unit of dispersion
    * ``theta``  = d(price)/d(time passing)  -- negative time-decay (see MODEL.md)

    ``tau`` in (0, 1] is the fraction of time remaining to resolution; sigma is
    the dispersion at ``tau == 1`` and scales as ``sigma * sqrt(tau)``. Theta is
    reported per unit of ``tau`` elapsed (so it is <= 0 for a long option).
    """
    p = _check_prob("p", p)
    strike = _check_prob("strike", strike)
    ot = _coerce_option_type(option_type)
    if sigma < 0.0:
        raise ValueError("sigma must be >= 0")
    if tau <= 0.0 or tau > 1.0:
        raise ValueError("tau must be in (0, 1]")

    base_sig = _effective_sigma(sigma, p, book)
    sig = max(base_sig * math.sqrt(tau), _SIGMA_FLOOR)
    d = (p - strike) / sig
    cdf_d = _norm_cdf(d)
    pdf_d = _norm_pdf(d)

    if ot is OptionType.CALL:
        delta = cdf_d
    else:
        delta = cdf_d - 1.0

    gamma = pdf_d / sig
    vega = pdf_d  # d(price)/d(sigma), identical for call and put

    # price = (p-K) * Phi(d) + sig*phi(d); d(sig)/d(tau) = base_sig/(2*sqrt(tau)).
    # d(price)/d(sig) = phi(d) = vega. Time *passing* reduces tau, so decay is
    # the negative of d(price)/d(tau).
    dsig_dtau = base_sig / (2.0 * math.sqrt(tau))
    theta = -(vega * dsig_dtau)

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "sigma_effective": sig,
        "d": d,
    }


def greek_curve(
    strike: float,
    option_type,
    sigma: float = 0.15,
    tau: float = 1.0,
    num_points: int = 101,
    book: Optional[OrderBook] = None,
) -> dict:
    """Each Greek as a function of the underlying probability, for charting.

    Returns ``p`` (probability grid, 0..1) and parallel arrays ``delta``,
    ``gamma``, ``vega``, ``theta``, plus ``price``.
    """
    strike = _check_prob("strike", strike)
    ot = _coerce_option_type(option_type)
    if num_points < 2:
        raise ValueError("num_points must be >= 2")

    ps = [i / (num_points - 1) for i in range(num_points)]
    delta, gamma, vega, theta, price = [], [], [], [], []
    for p in ps:
        g = greeks(p, strike, ot, sigma=sigma, tau=tau, book=book)
        delta.append(g["delta"])
        gamma.append(g["gamma"])
        vega.append(g["vega"])
        theta.append(g["theta"])
        price.append(fair_price(p, strike, ot, sigma=sigma, book=book))
    return {
        "p": ps,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "price": price,
    }


# ---------------------------------------------------------------------------
# Bundle for the API / UI
# ---------------------------------------------------------------------------

def analyze(
    p: float,
    strike: float,
    option_type,
    side="long",
    sigma: float = 0.15,
    tau: float = 1.0,
    book: Optional[OrderBook] = None,
    num_points: int = 101,
) -> dict:
    """Everything the UI needs for one synthetic option, as plain JSON data.

    Bundles the fair price (used as the premium for break-even / max P&L), the
    implied probability, payoff and Greek curves, break-even point(s), max
    profit/loss, and point Greeks at the current probability. All values are
    JSON-serializable (floats, lists, strings).
    """
    p = _check_prob("p", p)
    strike = _check_prob("strike", strike)
    ot = _coerce_option_type(option_type)
    sd = _coerce_side(side)

    price = fair_price(p, strike, ot, sigma=sigma, book=book)
    implied = implied_probability(p, book=book)
    pcurve = payoff_curve(strike, ot, side=sd, premium=price, num_points=num_points)
    gcurve = greek_curve(strike, ot, sigma=sigma, tau=tau, num_points=num_points, book=book)
    be = break_even(strike, ot, price)
    mpl = max_profit_loss(strike, ot, sd, price)
    point_greeks = greeks(p, strike, ot, sigma=sigma, tau=tau, book=book)
    pop = prob_of_profit(
        p, strike, ot, sd, price, sigma=sigma, tau=tau, book=book
    )

    return {
        "inputs": {
            "p": p,
            "strike": strike,
            "option_type": ot.value,
            "side": sd.value,
            "sigma": sigma,
            "tau": tau,
            "has_book": book is not None,
        },
        "implied_probability": implied,
        "fair_price": price,
        "break_even": be,
        "max_profit": mpl["max_profit"],
        "max_loss": mpl["max_loss"],
        "prob_of_profit": pop,
        "payoff_curve": pcurve,
        "greek_curve": gcurve,
        "greeks": point_greeks,
    }
