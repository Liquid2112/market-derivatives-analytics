"""The single seam between the UI/API layer and the two sibling layers.

Two sibling workers own code in this same repository:

  * the **data layer** — fetches and normalizes prediction markets and order
    books from Polymarket and Kalshi into a schema (``schema.py`` / ``SCHEMA.md``);
  * the **derivatives engine** — pure Python that computes payoff, break-even,
    implied probability and the four Greeks (``MODEL.md``).

Neither has landed on ``main`` yet. So every call the API makes into those
layers goes through this one module. Today it is backed by deterministic
mocks (``_MockDataLayer`` / ``_MockEngine``) that satisfy the documented
contracts, so the UI runs and the tests pass standalone.

FINAL WIRING (one place, this file): set the environment variable
``MDA_USE_REAL_BACKENDS=1`` and fill in ``_real_data_layer`` /
``_real_engine`` below to import the siblings' modules. The contract the
adapter exposes to the rest of the app (``get_data_layer`` / ``get_engine``
and the two Protocols) does not change, so nothing downstream needs editing.

The expected sibling contracts are the ``DataLayer`` and ``Engine``
Protocols defined here. If a sibling's real signature differs, adjust the
thin wrapper in ``_real_*`` — not the callers.
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Any, Protocol


# --------------------------------------------------------------------------
# Contracts expected from the sibling layers.
# --------------------------------------------------------------------------
class DataLayer(Protocol):
    """Expected contract of the data layer (``schema.py`` / ``SCHEMA.md``)."""

    def search_markets(
        self, query: str, provider: str | None = None, limit: int = 25
    ) -> list[dict[str, Any]]:
        """Return a list of market summaries matching ``query``.

        ``provider`` is ``"polymarket"``, ``"kalshi"`` or ``None`` (both).
        Each summary: ``{id, provider, title, yes_price, no_price, volume,
        close_time, category}``.
        """
        ...

    def get_market(self, market_id: str) -> dict[str, Any]:
        """Return one market plus its order book.

        Shape: ``{id, provider, title, yes_price, no_price, volume,
        close_time, category, order_book: {bids: [[price, size], ...],
        asks: [[price, size], ...]}}``. Prices are probabilities in ``[0, 1]``.
        """
        ...


class Engine(Protocol):
    """Expected contract of the derivatives engine (``MODEL.md``)."""

    def bundle(
        self,
        yes_price: float,
        strike: float,
        option_type: str,
        side: str,
        premium: float | None = None,
    ) -> dict[str, Any]:
        """Return a JSON-serializable analytics bundle for a synthetic option.

        ``yes_price`` / ``strike`` are probabilities in ``[0, 1]``.
        ``option_type`` is ``"call"`` or ``"put"``; ``side`` is ``"long"`` or
        ``"short"``. Returns: ``{premium, break_even, implied_prob,
        payoff_curve: [{p, payoff}, ...], greeks: {delta, gamma, theta, vega},
        greek_curves: {delta: [{p, value}], gamma: [...], theta: [...],
        vega: [...]}}``.
        """
        ...


# --------------------------------------------------------------------------
# Mock data layer — deterministic, no network, satisfies the contract.
# --------------------------------------------------------------------------
_MOCK_MARKETS: list[dict[str, Any]] = [
    {
        "id": "poly-us-recession-2026",
        "provider": "polymarket",
        "title": "US recession declared in 2026?",
        "yes_price": 0.37,
        "volume": 4_820_000.0,
        "close_time": "2026-12-31T23:59:00Z",
        "category": "Economics",
    },
    {
        "id": "poly-btc-100k-eoy",
        "provider": "polymarket",
        "title": "Bitcoin above $100k at year end?",
        "yes_price": 0.62,
        "volume": 12_440_000.0,
        "close_time": "2026-12-31T23:59:00Z",
        "category": "Crypto",
    },
    {
        "id": "kalshi-fed-cut-dec",
        "provider": "kalshi",
        "title": "Fed cuts rates at the December meeting?",
        "yes_price": 0.78,
        "volume": 2_110_000.0,
        "close_time": "2026-12-18T19:00:00Z",
        "category": "Economics",
    },
    {
        "id": "kalshi-cpi-above-3",
        "provider": "kalshi",
        "title": "CPI year-over-year above 3.0% next print?",
        "yes_price": 0.44,
        "volume": 980_000.0,
        "close_time": "2026-12-11T13:30:00Z",
        "category": "Economics",
    },
    {
        "id": "poly-ai-model-release",
        "provider": "polymarket",
        "title": "A major new frontier AI model ships this quarter?",
        "yes_price": 0.71,
        "volume": 3_300_000.0,
        "close_time": "2026-12-31T23:59:00Z",
        "category": "Technology",
    },
    {
        "id": "kalshi-snow-nyc-dec",
        "provider": "kalshi",
        "title": "Measurable snow in NYC before December 15?",
        "yes_price": 0.29,
        "volume": 410_000.0,
        "close_time": "2026-12-15T05:00:00Z",
        "category": "Weather",
    },
]


def _seeded_depth(market_id: str, yes_price: float) -> dict[str, list[list[float]]]:
    """Build a deterministic order book around ``yes_price``.

    Depth is derived from a hash of the market id so each market has a
    stable but distinct book without any network call.
    """
    seed = int(hashlib.sha256(market_id.encode()).hexdigest()[:8], 16)
    rng = (seed % 1000) / 1000.0  # 0..1, deterministic per market
    bids: list[list[float]] = []
    asks: list[list[float]] = []
    for level in range(1, 6):
        spread = 0.01 * level
        bid_px = round(max(0.01, yes_price - spread), 3)
        ask_px = round(min(0.99, yes_price + spread), 3)
        base_size = 500 + ((seed >> level) % 4000)
        size = round(base_size * (1.0 + rng) / level, 1)
        bids.append([bid_px, size])
        asks.append([ask_px, round(size * (0.8 + rng * 0.4), 1)])
    return {"bids": bids, "asks": asks}


class _MockDataLayer:
    def search_markets(
        self, query: str, provider: str | None = None, limit: int = 25
    ) -> list[dict[str, Any]]:
        q = (query or "").strip().lower()
        out = []
        for m in _MOCK_MARKETS:
            if provider and m["provider"] != provider:
                continue
            if q and q not in m["title"].lower() and q not in m["category"].lower():
                continue
            summary = dict(m)
            summary["no_price"] = round(1.0 - m["yes_price"], 3)
            out.append(summary)
        return out[:limit]

    def get_market(self, market_id: str) -> dict[str, Any]:
        for m in _MOCK_MARKETS:
            if m["id"] == market_id:
                full = dict(m)
                full["no_price"] = round(1.0 - m["yes_price"], 3)
                full["order_book"] = _seeded_depth(market_id, m["yes_price"])
                return full
        raise KeyError(market_id)


# --------------------------------------------------------------------------
# Mock engine — a self-contained, documented options model over probabilities.
# --------------------------------------------------------------------------
# shortcut: this mock prices a synthetic option whose underlying is the YES
# probability p in [0,1], treating p like a tradeable price with intrinsic
# payoff max(p-K,0)/max(K-p,0). It uses a Bachelier (normal) model for a
# smooth premium/Greeks surface, which is a reasonable choice for a bounded
# [0,1] underlying. Swap for the real engine (MODEL.md) at final wiring.
_SIGMA = 0.18   # volatility of the probability underlying
_TAU = 0.25     # time to expiry (years) used for the smooth model


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class _MockEngine:
    def _premium_call(self, p: float, k: float) -> float:
        vol = _SIGMA * math.sqrt(_TAU)
        if vol <= 0:
            return max(p - k, 0.0)
        d = (p - k) / vol
        return (p - k) * _norm_cdf(d) + vol * _norm_pdf(d)

    def _premium(self, p: float, k: float, option_type: str) -> float:
        call = self._premium_call(p, k)
        if option_type == "put":
            # put-call parity for the Bachelier model: P = C - (p - k)
            return call - (p - k)
        return call

    def bundle(
        self,
        yes_price: float,
        strike: float,
        option_type: str,
        side: str,
        premium: float | None = None,
    ) -> dict[str, Any]:
        option_type = option_type.lower()
        side = side.lower()
        if option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")
        if side not in ("long", "short"):
            raise ValueError("side must be 'long' or 'short'")
        p = _clamp01(yes_price)
        k = _clamp01(strike)
        prem = premium if premium is not None else self._premium(p, k, option_type)
        sign = 1.0 if side == "long" else -1.0

        # payoff at expiry across the probability grid
        grid = [i / 100.0 for i in range(0, 101)]
        payoff_curve = []
        for gp in grid:
            if option_type == "call":
                intrinsic = max(gp - k, 0.0)
            else:
                intrinsic = max(k - gp, 0.0)
            pnl = sign * (intrinsic - prem)
            payoff_curve.append({"p": round(gp, 3), "payoff": round(pnl, 5)})

        # break-even: probability where payoff crosses zero
        if option_type == "call":
            break_even = _clamp01(k + prem)
        else:
            break_even = _clamp01(k - prem)

        # Greeks via the smooth Bachelier model, evaluated at current p
        vol = _SIGMA * math.sqrt(_TAU)
        d = (p - k) / vol if vol > 0 else 0.0
        delta_call = _norm_cdf(d)
        delta = delta_call if option_type == "call" else delta_call - 1.0
        gamma = _norm_pdf(d) / vol if vol > 0 else 0.0
        vega = math.sqrt(_TAU) * _norm_pdf(d)
        theta = -(_SIGMA * _norm_pdf(d)) / (2.0 * math.sqrt(_TAU)) if _TAU > 0 else 0.0

        def _greek_at(gp: float, which: str) -> float:
            dd = (gp - k) / vol if vol > 0 else 0.0
            if which == "delta":
                dc = _norm_cdf(dd)
                val = dc if option_type == "call" else dc - 1.0
            elif which == "gamma":
                val = _norm_pdf(dd) / vol if vol > 0 else 0.0
            elif which == "vega":
                val = math.sqrt(_TAU) * _norm_pdf(dd)
            else:  # theta
                val = -(_SIGMA * _norm_pdf(dd)) / (2.0 * math.sqrt(_TAU)) if _TAU > 0 else 0.0
            return sign * val

        greek_curves = {
            name: [{"p": round(gp, 3), "value": round(_greek_at(gp, name), 6)} for gp in grid]
            for name in ("delta", "gamma", "theta", "vega")
        }

        return {
            "underlying_prob": round(p, 4),
            "strike": round(k, 4),
            "option_type": option_type,
            "side": side,
            "premium": round(prem, 5),
            "break_even": round(break_even, 4),
            "implied_prob": round(k, 4),
            "payoff_curve": payoff_curve,
            "greeks": {
                "delta": round(sign * delta, 5),
                "gamma": round(sign * gamma, 5),
                "theta": round(sign * theta, 5),
                "vega": round(sign * vega, 5),
            },
            "greek_curves": greek_curves,
        }


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


# --------------------------------------------------------------------------
# Selection: mock by default, real siblings when MDA_USE_REAL_BACKENDS=1.
# --------------------------------------------------------------------------
def _real_data_layer() -> DataLayer:
    # FINAL WIRING: import the data-layer sibling and return an object
    # satisfying the DataLayer Protocol, e.g.:
    #   from schema import DataStore   # noqa
    #   return DataStore()
    raise NotImplementedError(
        "Real data layer not wired yet. Unset MDA_USE_REAL_BACKENDS or "
        "implement _real_data_layer() in app/adapter.py."
    )


def _real_engine() -> Engine:
    # FINAL WIRING: import the derivatives-engine sibling (MODEL.md) and
    # return an object satisfying the Engine Protocol.
    raise NotImplementedError(
        "Real engine not wired yet. Unset MDA_USE_REAL_BACKENDS or "
        "implement _real_engine() in app/adapter.py."
    )


_USE_REAL = os.getenv("MDA_USE_REAL_BACKENDS") == "1"
_data_layer: DataLayer = _real_data_layer() if _USE_REAL else _MockDataLayer()
_engine: Engine = _real_engine() if _USE_REAL else _MockEngine()


def get_data_layer() -> DataLayer:
    return _data_layer


def get_engine() -> Engine:
    return _engine


def using_mock_backends() -> bool:
    return not _USE_REAL
