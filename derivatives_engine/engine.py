"""Synthetic derivatives engine for binary prediction markets.

The underlying is a prediction-market probability ``p`` in [0, 1] (the YES price).
At resolution the event settles to 1 (YES) or 0 (NO). We model synthetic options
whose strike ``K`` is a probability level in [0, 1] and chart everything across
the continuous 0..1 outcome range so a UI can plot it.

Two layers are kept strictly separate:

* **Intrinsic payoff** at expiry -- model-free. A synthetic call pays
  ``max(x - K, 0)`` and a put pays ``max(K - x, 0)`` for a settlement value ``x``.
* **Fair price, chance of profit and Greeks before expiry** -- these need a model
  of how the YES probability moves between now and the option's expiry.

Pricing model (see MODEL.md for the full derivation and caveats).  An option held
to resolution is a *scaled binary*: at resolution ``x`` is 1 (w.p. ``p``) or 0
(w.p. ``1 - p``), so its exact, model-free value is ``call = p*(1-K)``,
``put = (1-p)*K``.  For an expiry *before* resolution we model the YES
probability's path with a process bounded in [0, 1] -- a logit-normal
martingale ``P_t = logistic(logit(p) + m + s*Z)``, ``Z ~ Normal(0, 1)`` -- with
the drift ``m`` chosen so ``E[P_t] = p`` (a martingale), and the dispersion ``s``
growing with the time left.  The option price is ``E[intrinsic(P_t)]``.  As ``s``
grows the terminal law tends to ``Bernoulli(p)`` and the price tends to the exact
held-to-resolution value, so the exact formulas are the *limit* of the model, not
a separate special case.  The expectations are evaluated with Gauss-Hermite
quadrature; the held-to-resolution setting is the closed-form limit.

No network, no file I/O, no global state. Every public function is pure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

_SQRT_2PI = math.sqrt(2.0 * math.pi)
# shortcut: dispersion floor so model math stays finite when tau -> 0 (expiry is
# now); below this the price is effectively the intrinsic value. Revisit only if
# a caller needs an exact tau=0 branch distinct from "very small s".
_S_FLOOR = 1e-9


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
    A premium of 0 degenerates to the strike. A break-even that falls outside
    [0, 1] yields an empty list -- it is reported as absent, never clamped to a
    boundary (a clamped value would be a break-even that does not exist).
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


def half_spread(book: Optional[OrderBook]) -> float:
    """Top-of-book half-spread, or 0.0 when the book is missing/one-sided."""
    if book is None:
        return 0.0
    bb = book.best_bid()
    ba = book.best_ask()
    if bb is None or ba is None:
        return 0.0
    return 0.5 * max(ba - bb, 0.0)


# ---------------------------------------------------------------------------
# The pricing model: a logit-normal martingale for the YES probability path
# ---------------------------------------------------------------------------
# An option held to resolution is a scaled binary: the exact, model-free value is
# ``call = p*(1-K)``, ``put = (1-p)*K``. For an expiry *before* resolution we model
# the YES probability's path with a process bounded in [0, 1]: a logit-normal
# martingale ``P_t = logistic(logit(p) + m + s*Z)``, ``Z ~ Normal(0, 1)``, with the
# drift ``m`` solved so ``E[P_t] = p`` (the martingale constraint, which makes
# put-call parity exact). The option price is ``E[intrinsic(P_t)]``.
#
# As ``s -> infinity`` the terminal law tends to ``Bernoulli(p)``: ``P_t`` is
# bounded in [0, 1] and the identity is a bounded continuous function, so
# ``E[P_t] = p`` forces every weak limit to have mean ``p``; the interior
# transition mass vanishes like O(1/s), leaving a two-point law on {0, 1} with
# mean ``p``. So the exact held-to-resolution value is the ``s -> infinity`` limit
# of this model, not a separate case. Convergence is slow -- the interior mass and
# the gap to the exact price shrink like ~1/s -- which MODEL.md states and the
# continuity tests reflect. We return the exact closed form at ``tau == 1`` (the
# limit) and quadrature for ``tau < 1``.
#
# Quadrature (stable at any ``s``): we integrate in ``u = m + s*Z`` space, so
# ``U ~ Normal(m, s^2)`` and ``E[f(logistic(U))]`` is a fixed-grid composite
# Simpson integral of ``f(logistic(u)) * N(u; m, s^2)`` over ``u in [-U_HI, U_HI]``
# plus the two boundary tails ``P(U > U_HI)*f(1)`` and ``P(U < -U_HI)*f(0)`` (exact
# via erfc). The grid resolution does not depend on ``s``. The drift ``m`` scales
# like ``s * Phi^{-1}(p)``, so it is bracketed over ``+/- (20*s + 40)``.

# shortcut: base dispersion of the logit-normal path at the half-way slider point
# (tau=0.5). A wider order-book spread widens it (more disagreement about the
# outcome). Documented default in MODEL.md; revisit if a market needs a calibrated
# per-market dispersion rather than this book-widened constant.
_S_BASE = 2.0
# Order-book half-spread is a probability in [0, 1]; scale it into logit space so
# a wide book meaningfully widens the path dispersion.
_SPREAD_TO_S = 6.0
# Above this effective dispersion the logit-normal terminal law is within ~1/s of
# Bernoulli(p); we use the exact held-to-resolution formulas at/above it to stay
# closed-form and exact at the top of the tau slider.
_S_EXACT = 1.0e4
# Fixed logit-space (u) integration half-width and node count for E[.]; resolution
# is independent of s because we integrate in u = m + s*Z, not in Z.
_U_HI = 40.0
_U_NODES = 401  # odd, for composite Simpson

_SQRT_PI = math.sqrt(math.pi)
_SQRT_2 = math.sqrt(2.0)


def _norm_sf(x: float) -> float:
    """Upper tail ``P(Z > x)`` for a standard normal, via erfc."""
    return 0.5 * math.erfc(x / _SQRT_2)


# Fixed composite-Simpson grid on u in [-_U_HI, _U_HI]; built once at import.
_U_H = (2.0 * _U_HI) / (_U_NODES - 1)
_U_GRID = tuple(-_U_HI + i * _U_H for i in range(_U_NODES))
# Simpson weights: 1,4,2,4,...,4,1 (times h/3), folded into the grid.
_U_SIMPSON = tuple(
    (1.0 if i in (0, _U_NODES - 1) else (4.0 if i % 2 == 1 else 2.0))
    for i in range(_U_NODES)
)


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _logistic(x: float) -> float:
    # Numerically stable logistic.
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _s_of_tau(tau: float, book: Optional[OrderBook]) -> float:
    """Dispersion ``s`` of the logit-normal path as a function of time left.

    ``s`` grows with the time remaining: ``s -> 0`` as ``tau -> 0`` (expiry is
    now, so the price is the intrinsic value) and ``s -> infinity`` as
    ``tau -> 1`` (held to resolution, so the terminal law tends to
    ``Bernoulli(p)`` and the price to the exact value). The
    ``sqrt(tau/(1-tau))`` shape sends ``s`` to infinity at the top of the slider
    while staying well-scaled through the middle. An order-book spread widens it.
    """
    if tau <= 0.0 or tau > 1.0:
        raise ValueError("tau must be in (0, 1]")
    base = _S_BASE + _SPREAD_TO_S * half_spread(book)
    if tau >= 1.0:
        return math.inf
    return base * math.sqrt(tau / (1.0 - tau))


def _solve_drift(p: float, s: float) -> float:
    """Drift ``m`` so that ``E[logistic(m + s*Z)] = p`` for ``Z ~ Normal(0, 1)``.

    ``m`` scales like ``s * Phi^{-1}(p)``, so a good initial guess is
    ``s * Phi^{-1}(p)``; Newton then converges in a few steps. The mean is
    strictly increasing in ``m`` with derivative ``E[logistic'(U)]``, computed in
    the same pass. Falls back to bisection over a wide (``s``-scaled) bracket if
    Newton steps out of range.
    """
    # Initial guess: m ~ s * Phi^{-1}(p). _inv_norm_cdf is a good enough start.
    m = s * _inv_norm_cdf(p)
    for _ in range(60):
        mean, dmean = _u_mean_and_deriv(m, s)
        err = mean - p
        if abs(err) < 1e-12:
            return m
        if dmean <= 1e-300:
            break
        step = err / dmean
        # Damp over-long Newton steps (keeps it inside the increasing region).
        if step > 10.0 * s + 10.0:
            step = 10.0 * s + 10.0
        elif step < -(10.0 * s + 10.0):
            step = -(10.0 * s + 10.0)
        m -= step
    # Bisection fallback over an s-scaled bracket.
    span = 20.0 * s + 40.0
    lo, hi = -span, span
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _u_mean(mid, s) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _inv_norm_cdf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation)."""
    if p <= 0.0:
        return -40.0
    if p >= 1.0:
        return 40.0
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def _u_mean_and_deriv(m: float, s: float):
    """``(E[logistic(U)], d/dm E[logistic(U)])`` for ``U ~ Normal(m, s^2)``.

    The derivative in ``m`` is ``E[logistic'(U)] = E[q*(1-q)]`` with ``q =
    logistic(U)`` (shift-invariance of the Gaussian in ``m``). Both come from one
    fixed-grid pass plus the (derivative-free) boundary tails.
    """
    inv_s = 1.0 / s
    norm = inv_s / _SQRT_2PI
    mean = 0.0
    deriv = 0.0
    for u, sw in zip(_U_GRID, _U_SIMPSON):
        z = (u - m) * inv_s
        dens = norm * math.exp(-0.5 * z * z)
        q = _logistic(u)
        w = sw * dens
        mean += w * q
        deriv += w * q * (1.0 - q)
    scale = _U_H / 3.0
    mean *= scale
    deriv *= scale
    # Boundary tails: logistic -> 1 (contributes to mean) or 0; derivative ~0.
    mean += _norm_sf((_U_HI - m) * inv_s) * 1.0
    return mean, deriv


def _u_integral(fn, m: float, s: float) -> float:
    """``E[fn(logistic(U))]`` for ``U ~ Normal(m, s^2)`` on a fixed u-grid.

    Composite Simpson over ``u in [-_U_HI, _U_HI]`` of
    ``fn(logistic(u)) * N(u; m, s^2)``, plus the two boundary tails
    ``P(U > _U_HI)*fn(1)`` and ``P(U < -_U_HI)*fn(0)`` (exact via the normal
    survival function). Grid resolution is independent of ``s`` because we
    integrate in ``u``, not in ``Z``. ``fn`` takes a probability in [0, 1].
    """
    inv_s = 1.0 / s
    norm = inv_s / _SQRT_2PI
    total = 0.0
    for u, sw in zip(_U_GRID, _U_SIMPSON):
        z = (u - m) * inv_s
        dens = norm * math.exp(-0.5 * z * z)
        total += sw * fn(_logistic(u)) * dens
    total *= _U_H / 3.0
    # Boundary tails: all mass beyond +/-_U_HI maps to logistic -> 1 or 0.
    total += _norm_sf((_U_HI - m) * inv_s) * fn(1.0)
    total += _norm_sf((_U_HI + m) * inv_s) * fn(0.0)
    return total


def _u_mean(m: float, s: float) -> float:
    """``E[logistic(U)]`` for ``U ~ Normal(m, s^2)`` (the martingale mean)."""
    return _u_integral(lambda q: q, m, s)


def _ln_price(p: float, strike: float, ot: OptionType, s: float) -> float:
    """``E[intrinsic(P_t)]`` for the logit-normal law at dispersion ``s``.

    The drift is solved so ``E[P_t] = p``. Uses the exact Bernoulli limit when
    ``s`` is effectively infinite.
    """
    if s >= _S_EXACT or math.isinf(s):
        return _price_exact(p, strike, ot)
    if s <= _S_FLOOR:
        return intrinsic_payoff(p, strike, ot)
    m = _solve_drift(p, s)
    if ot is OptionType.CALL:
        return _u_integral(lambda q: max(q - strike, 0.0), m, s)
    return _u_integral(lambda q: max(strike - q, 0.0), m, s)


def _ln_mass_above(p: float, s: float, threshold: float) -> float:
    """``P(P_t > threshold)`` for the logit-normal law with ``E[P_t] = p``.

    ``P_t > threshold`` iff ``U > logit(threshold)`` (logistic is monotone), so
    this is an exact normal tail -- no quadrature needed. At the exact
    (held-to-resolution) limit the law is ``Bernoulli(p)`` and this is ``p`` for
    any interior threshold.
    """
    if threshold <= 0.0:
        return 1.0
    if threshold >= 1.0:
        return 0.0
    if s >= _S_EXACT or math.isinf(s):
        return p  # Bernoulli(p): mass p at 1, (1-p) at 0
    if s <= _S_FLOOR:
        return 1.0 if p > threshold else 0.0
    m = _solve_drift(p, s)
    return _norm_sf((_logit(threshold) - m) / s)


def _price_model(
    p: float, strike: float, ot: OptionType, tau: float, book: Optional[OrderBook]
) -> float:
    """Model price ``E[intrinsic(P_t)]`` of a long option at time-left ``tau``."""
    s = _s_of_tau(tau, book)
    return _ln_price(p, strike, ot, s)


def _price_exact(p: float, strike: float, ot: OptionType) -> float:
    """Exact, model-free held-to-resolution value of a long option.

    Settlement is ``Bernoulli(p)``: ``call = p*(1-K)``, ``put = (1-p)*K``.
    """
    if ot is OptionType.CALL:
        return p * (1.0 - strike)
    return (1.0 - p) * strike


# ---------------------------------------------------------------------------
# Pricing (public)
# ---------------------------------------------------------------------------

def held_to_resolution_price(p: float, strike: float, option_type) -> float:
    """Exact, model-free price of an option held to resolution (no model).

    ``call = p*(1-K)``, ``put = (1-p)*K``. This is the ``tau -> held to
    resolution`` limit of :func:`fair_price` and the basis the UI should label
    as model-free.
    """
    p = _check_prob("p", p)
    strike = _check_prob("strike", strike)
    ot = _coerce_option_type(option_type)
    return _price_exact(p, strike, ot)


def fair_price(
    p: float,
    strike: float,
    option_type,
    tau: float = 1.0,
    book: Optional[OrderBook] = None,
    sigma: Optional[float] = None,
) -> float:
    """Model fair value of one synthetic option, as a function of time left.

    Priced as ``E[intrinsic(P_t)]`` under the logit-normal martingale (MODEL.md).
    At ``tau == 1`` (held to resolution) this is the exact model-free value
    ``p*(1-K)`` / ``(1-p)*K``; at smaller ``tau`` the expiry is before resolution
    and the price falls toward the intrinsic value ``max(p-K,0)`` / ``max(K-p,0)``.
    The value is non-decreasing in ``tau`` and satisfies the no-arbitrage bounds
    and put-call parity for every ``tau`` (see MODEL.md / the tests).

    ``sigma`` is accepted and ignored for backward compatibility; the dispersion
    is now driven by ``tau`` and the order-book spread, not a free ``sigma``.
    """
    p = _check_prob("p", p)
    strike = _check_prob("strike", strike)
    ot = _coerce_option_type(option_type)
    return _price_model(p, strike, ot, tau, book)


def pricing_basis(tau: float = 1.0) -> str:
    """Whether the premium at ``tau`` is the exact model-free value or modeled.

    ``"held_to_resolution"`` (exact, model-free) when ``tau`` is at the top of
    the slider, else ``"model"`` (logit-normal). Lets the UI word its note.
    """
    if tau <= 0.0 or tau > 1.0:
        raise ValueError("tau must be in (0, 1]")
    return "held_to_resolution" if tau >= 1.0 else "model"


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


# ---------------------------------------------------------------------------
# Chance of profit (same premium, same tau as the displayed price)
# ---------------------------------------------------------------------------

def prob_of_profit(
    p: float,
    strike: float,
    option_type,
    side,
    premium: float,
    tau: float = 1.0,
    book: Optional[OrderBook] = None,
    sigma: Optional[float] = None,
) -> float:
    """Probability the position ends in profit, in [0, 1].

    Uses the *same* logit-normal terminal law (same ``tau`` and order book) and
    the *same* ``premium`` as the displayed price. The net payoff is positive on
    exactly one side of the break-even, so this is the probability mass of the
    terminal YES probability on that side. As ``tau -> held to resolution`` the
    terminal law tends to ``Bernoulli(p)`` and this tends to the exact value: a
    long call is profitable iff YES (PoP ``= p``), a long put iff NO
    (PoP ``= 1 - p``), the short sides the complements.

    ``sigma`` is accepted and ignored (kept for backward compatibility).
    """
    p = _check_prob("p", p)
    strike = _check_prob("strike", strike)
    ot = _coerce_option_type(option_type)
    sd = _coerce_side(side)
    if premium < 0.0:
        raise ValueError("premium must be >= 0")

    be = break_even(strike, ot, premium)
    s = _s_of_tau(tau, book)

    # Long call / short put profit when the terminal prob is ABOVE break-even;
    # long put / short call profit when it is BELOW. (Short flips the long side.)
    long_profits_above = ot is OptionType.CALL
    profits_above = long_profits_above if sd is Side.LONG else not long_profits_above

    if not be:
        # Break-even lies outside [0, 1]: a long position can never recover the
        # premium within the settlement range, so it never profits; the short
        # counterparty always keeps the premium.
        return 0.0 if sd is Side.LONG else 1.0

    threshold = be[0]
    mass_above = _ln_mass_above(p, s, threshold)
    return mass_above if profits_above else (1.0 - mass_above)


# ---------------------------------------------------------------------------
# Greeks (derivatives of the model price)
# ---------------------------------------------------------------------------

def greeks(
    p: float,
    strike: float,
    option_type,
    tau: float = 1.0,
    book: Optional[OrderBook] = None,
    sigma: Optional[float] = None,
) -> dict:
    """Greeks for one synthetic option under the logit-normal model.

    * ``delta`` = d(price)/dp           -- how value moves with the YES probability
    * ``gamma`` = d2(price)/dp2         -- convexity of value in the probability
    * ``vega``  = d(price)/ds           -- sensitivity to the path dispersion ``s``
    * ``theta`` = d(price)/d(tau)       -- rate of change of the *displayed* price
                                           with the time-left slider (``>= 0`` for a
                                           long option, since more time left is
                                           worth more). This is the finite-
                                           difference rate of the same premium the
                                           UI shows, by construction.

    At the held-to-resolution limit (``tau == 1``) these are the Greeks of the
    exact formula: for a call ``delta = 1 - K``, ``gamma = 0``, ``theta = 0``;
    for a put ``delta = -K``, ``gamma = 0``, ``theta = 0``.

    ``sigma`` is accepted and ignored (kept for backward compatibility).
    """
    p = _check_prob("p", p)
    strike = _check_prob("strike", strike)
    ot = _coerce_option_type(option_type)
    if tau <= 0.0 or tau > 1.0:
        raise ValueError("tau must be in (0, 1]")

    s = _s_of_tau(tau, book)

    if s >= _S_EXACT or math.isinf(s):
        # Exact held-to-resolution Greeks (price is linear in p, flat in s/tau).
        delta = (1.0 - strike) if ot is OptionType.CALL else -strike
        return {
            "delta": delta,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "s_effective": s if not math.isinf(s) else float("inf"),
        }

    # delta and gamma: central finite difference of the model price in p. Keep a
    # symmetric step that stays strictly inside (0, 1) -- near the boundary we
    # shrink the step but never let it collapse (which would blow up the FD).
    hp = min(1e-3, p - 1e-6, (1.0 - 1e-6) - p)
    if hp < 1e-9:
        hp = 1e-9
    lo = p - hp
    hi = p + hp
    f_lo = _ln_price(lo, strike, ot, s)
    f_hi = _ln_price(hi, strike, ot, s)
    f_mid = _ln_price(p, strike, ot, s)
    delta = (f_hi - f_lo) / (2.0 * hp)
    gamma = (f_hi - 2.0 * f_mid + f_lo) / (hp * hp)

    # vega: d(price)/ds by central difference in the dispersion.
    hs = max(1e-4, 1e-3 * s)
    s_lo = max(s - hs, _S_FLOOR)
    s_hi = s + hs
    vega = (_ln_price(p, strike, ot, s_hi) - _ln_price(p, strike, ot, s_lo)) / (s_hi - s_lo)

    # theta: d(price)/d(tau) -- the slope of the SAME displayed premium against
    # the time-left slider. Positive for a long option (more time left is worth
    # more). Central difference in tau, clamped inside (0, 1].
    ht = 1e-4
    t_lo = max(tau - ht, 1e-9)
    t_hi = min(tau + ht, 1.0)
    p_lo = fair_price(p, strike, ot, tau=t_lo, book=book)
    p_hi = fair_price(p, strike, ot, tau=t_hi, book=book)
    theta = (p_hi - p_lo) / (t_hi - t_lo) if t_hi > t_lo else 0.0

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "s_effective": s,
    }


def greek_curve(
    strike: float,
    option_type,
    tau: float = 1.0,
    num_points: int = 101,
    book: Optional[OrderBook] = None,
    sigma: Optional[float] = None,
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
        # Evaluate the Greeks on a safe interior margin so the central FD step
        # always fits inside (0, 1); the endpoints p=0/p=1 are degenerate for the
        # sensitivity curve, so we clamp the evaluation point (the chart x-axis
        # still spans 0..1). The price curve is evaluated at the true p.
        pp = min(max(p, 0.01), 0.99)
        g = greeks(pp, strike, ot, tau=tau, book=book)
        delta.append(g["delta"])
        gamma.append(g["gamma"])
        vega.append(g["vega"])
        theta.append(g["theta"])
        price.append(fair_price(min(max(p, 1e-9), 1.0 - 1e-9), strike, ot, tau=tau, book=book))
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
    tau: float = 1.0,
    book: Optional[OrderBook] = None,
    num_points: int = 101,
    sigma: Optional[float] = None,
) -> dict:
    """Everything the UI needs for one synthetic option, as plain JSON data.

    Bundles the fair price (used as the premium for break-even / max P&L), the
    implied probability, payoff and Greek curves, break-even point(s), max
    profit/loss, chance of profit, the pricing basis, and point Greeks at the
    current probability. All values are JSON-serializable.

    ``sigma`` is accepted and ignored (kept for backward compatibility); the
    dispersion is driven by ``tau`` and the order book.
    """
    p = _check_prob("p", p)
    strike = _check_prob("strike", strike)
    ot = _coerce_option_type(option_type)
    sd = _coerce_side(side)

    price = fair_price(p, strike, ot, tau=tau, book=book)
    implied = implied_probability(p, book=book)
    pcurve = payoff_curve(strike, ot, side=sd, premium=price, num_points=num_points)
    gcurve = greek_curve(strike, ot, tau=tau, num_points=num_points, book=book)
    be = break_even(strike, ot, price)
    mpl = max_profit_loss(strike, ot, sd, price)
    point_greeks = greeks(p, strike, ot, tau=tau, book=book)
    pop = prob_of_profit(p, strike, ot, sd, price, tau=tau, book=book)

    return {
        "inputs": {
            "p": p,
            "strike": strike,
            "option_type": ot.value,
            "side": sd.value,
            "tau": tau,
            "has_book": book is not None,
        },
        "implied_probability": implied,
        "fair_price": price,
        "pricing_basis": pricing_basis(tau),
        "break_even": be,
        "max_profit": mpl["max_profit"],
        "max_loss": mpl["max_loss"],
        "prob_of_profit": pop,
        "payoff_curve": pcurve,
        "greek_curve": gcurve,
        "greeks": point_greeks,
    }
