"""The single seam between the UI/API layer and the two real backends.

This module is the one place the UI/API layer reaches its two backends:

  * the **data layer** (``mdlab``) — fetches and normalizes prediction markets
    and order books from Polymarket and Kalshi into one schema
    (``mdlab/schema.py`` / ``SCHEMA.md``);
  * the **derivatives engine** (``derivatives_engine``) — pure Python that
    computes payoff, break-even, implied probability and the four Greeks
    (``MODEL.md``).

Both now live in this repository, so the real path is the default:
``MDA_USE_REAL_BACKENDS`` is ``1`` unless it is explicitly set to ``0``. The
deterministic mocks (``_MockDataLayer`` / ``_MockEngine``) are kept only so the
service and the shape tests can run offline with no network; set
``MDA_USE_REAL_BACKENDS=0`` to force them.

The rest of the app only sees the ``DataLayer`` and ``Engine`` Protocols and
the ``get_data_layer`` / ``get_engine`` accessors. The real backends are
adapted to those Protocols by ``_RealDataLayer`` / ``_RealEngine`` below, which
reconcile the small shape differences between ``mdlab``'s dataclasses, the
engine's ``analyze`` bundle, and the ``{price, size}`` / ``{p, value}`` shapes
the UI charts consume. All adapting happens here — never in the callers.
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


def _popularity_key(summary: dict[str, Any]) -> float:
    """Sort key that ranks a market by traded volume (most popular first).

    ``volume`` may be ``None`` (a provider that did not report it); those sort
    last. Used only to order the default landing view so recognizable, liquid
    markets surface first; it never changes which markets are returned.
    """
    vol = summary.get("volume")
    return float(vol) if vol is not None else -1.0


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
        # Default landing view shows the most liquid (recognizable) markets
        # first; an explicit search keeps the provider's relevance order.
        out.sort(key=_popularity_key, reverse=True)
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
# Real data layer — wraps mdlab's Polymarket + Kalshi clients and maps their
# schema (mdlab.Market / OrderBook) onto the dict shapes the API/UI expect.
# --------------------------------------------------------------------------
# A market id round-trips through the UI, so it must carry the provider. mdlab
# ids are provider-native (a Polymarket market id, a Kalshi ticker); we prefix
# them with "<provider>:" here and strip the prefix before calling mdlab.
_ID_SEP = ":"


def _compose_id(provider: str, native_id: str) -> str:
    return f"{provider}{_ID_SEP}{native_id}"


def _split_id(market_id: str) -> tuple[str, str]:
    provider, _, native = market_id.partition(_ID_SEP)
    if not native:
        raise KeyError(market_id)
    return provider, native


def _book_to_levels(order_book) -> dict[str, list[list[float]]]:
    """mdlab OrderBook -> {bids: [[price, size]], asks: [[price, size]]}."""
    if order_book is None:
        return {"bids": [], "asks": []}
    return {
        "bids": [[lv.price, lv.size] for lv in order_book.bids],
        "asks": [[lv.price, lv.size] for lv in order_book.asks],
    }


def _market_summary(market) -> dict[str, Any]:
    """mdlab Market -> the summary dict the UI list expects."""
    provider = market.provider.value
    yes = market.yes_price
    no = market.no_price
    if no is None and yes is not None:
        no = round(1.0 - yes, 4)
    return {
        "id": _compose_id(provider, market.id),
        "provider": provider,
        "title": market.title,
        "yes_price": yes,
        "no_price": no,
        "volume": market.volume,
        "close_time": market.close_time,
        # mdlab's schema has no category; surface the lifecycle status instead
        # so the UI's category slot is populated with something real.
        "category": (market.status or "").title() or None,
    }


class _RealDataLayer:
    """DataLayer over mdlab's Polymarket + Kalshi read-only clients.

    The clients are injected so tests can pass fixture-backed ``httpx`` clients
    (no network). The default constructor builds live read-only clients.
    """

    def __init__(self, polymarket_client=None, kalshi_client=None) -> None:
        from mdlab.kalshi import KalshiClient
        from mdlab.polymarket import PolymarketClient

        self._poly = polymarket_client or PolymarketClient()
        self._kalshi = kalshi_client or KalshiClient()

    def search_markets(
        self, query: str, provider: str | None = None, limit: int = 25
    ) -> list[dict[str, Any]]:
        q = (query or "").strip() or None
        out: list[dict[str, Any]] = []
        want = {provider} if provider else {"polymarket", "kalshi"}
        if "polymarket" in want:
            for m in self._poly.list_markets(limit=limit, search=q):
                if m.yes_price is None:
                    continue
                out.append(_market_summary(m))
        if "kalshi" in want:
            for m in self._kalshi.list_markets(limit=limit, search=q):
                if m.yes_price is None:
                    continue
                out.append(_market_summary(m))
        # Rank the default landing view by traded volume so popular,
        # recognizable markets lead; a text search keeps relevance order.
        if q is None:
            out.sort(key=_popularity_key, reverse=True)
        return out[:limit]

    def get_market(self, market_id: str) -> dict[str, Any]:
        provider, native = _split_id(market_id)
        if provider == "polymarket":
            client = self._poly
        elif provider == "kalshi":
            client = self._kalshi
        else:
            raise KeyError(market_id)
        try:
            market = client.get_market(native)
            book = client.get_order_book(native)
        except Exception as exc:  # network / lookup failure -> not found
            raise KeyError(market_id) from exc
        summary = _market_summary(market)
        summary["order_book"] = _book_to_levels(book)
        return summary


# --------------------------------------------------------------------------
# Real engine — wraps derivatives_engine.analyze and reshapes its bundle into
# the {p, payoff} / {p, value} shapes the UI charts consume.
# --------------------------------------------------------------------------
class _RealEngine:
    """Engine over ``derivatives_engine.analyze``.

    The engine computes long-side point Greeks; a short position is the sign
    flip of the long one, so we apply the side sign here (matching the UI's
    expectation and the mock engine's behaviour). ``break_even`` from the
    engine is a list (possibly empty) over [0,1]; the UI wants one scalar, so
    we take the first crossing and clamp as a fallback.
    """

    def bundle(
        self,
        yes_price: float,
        strike: float,
        option_type: str,
        side: str,
        premium: float | None = None,
    ) -> dict[str, Any]:
        from derivatives_engine import analyze

        option_type = option_type.lower()
        side = side.lower()
        if option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")
        if side not in ("long", "short"):
            raise ValueError("side must be 'long' or 'short'")

        p = _clamp01(yes_price)
        k = _clamp01(strike)
        result = analyze(
            p=p,
            strike=k,
            option_type=option_type,
            side=side,
            num_points=101,
        )
        sign = 1.0 if side == "long" else -1.0

        # payoff_curve: engine returns parallel {x: [...], payoff: [...]} where
        # payoff already carries the side sign (analyze passes side through).
        pc = result["payoff_curve"]
        payoff_curve = [
            {"p": round(xp, 3), "payoff": round(py, 5)}
            for xp, py in zip(pc["x"], pc["payoff"])
        ]

        # greek_curve: engine returns long-side parallel arrays; apply side sign.
        gc = result["greek_curve"]
        ps = gc["p"]
        greek_curves = {
            name: [
                {"p": round(ps[i], 3), "value": round(sign * gc[name][i], 6)}
                for i in range(len(ps))
            ]
            for name in ("delta", "gamma", "theta", "vega")
        }

        g = result["greeks"]  # long-side point Greeks
        be_list = result["break_even"]
        break_even = be_list[0] if be_list else _clamp01(
            k + result["fair_price"] if option_type == "call"
            else k - result["fair_price"]
        )

        return {
            "underlying_prob": round(p, 4),
            "strike": round(k, 4),
            "option_type": option_type,
            "side": side,
            "premium": round(result["fair_price"], 5),
            "break_even": round(break_even, 4),
            "implied_prob": round(result["implied_probability"], 4),
            "payoff_curve": payoff_curve,
            "greeks": {
                "delta": round(sign * g["delta"], 5),
                "gamma": round(sign * g["gamma"], 5),
                "theta": round(sign * g["theta"], 5),
                "vega": round(sign * g["vega"], 5),
            },
            "greek_curves": greek_curves,
        }


# --------------------------------------------------------------------------
# Selection: real backends (mdlab + engine) by default; mocks only when
# MDA_USE_REAL_BACKENDS=0, so the service can run fully offline.
#
# The choice is read per call from the environment (and the instances cached
# per mode) so tests can pin the mode with the env var regardless of import
# order, and so an injected layer in a test overrides both.
# --------------------------------------------------------------------------
def _real_data_layer() -> DataLayer:
    return _RealDataLayer()


def _real_engine() -> Engine:
    return _RealEngine()


def _use_real() -> bool:
    return os.getenv("MDA_USE_REAL_BACKENDS", "1") != "0"


_CACHE: dict[str, Any] = {}


def get_data_layer() -> DataLayer:
    if "data_override" in _CACHE:
        return _CACHE["data_override"]
    key = "data_real" if _use_real() else "data_mock"
    if key not in _CACHE:
        _CACHE[key] = _real_data_layer() if _use_real() else _MockDataLayer()
    return _CACHE[key]


def get_engine() -> Engine:
    if "engine_override" in _CACHE:
        return _CACHE["engine_override"]
    key = "engine_real" if _use_real() else "engine_mock"
    if key not in _CACHE:
        _CACHE[key] = _real_engine() if _use_real() else _MockEngine()
    return _CACHE[key]


def set_data_layer(layer: DataLayer | None) -> None:
    """Override the data layer (for tests). Pass ``None`` to clear."""
    if layer is None:
        _CACHE.pop("data_override", None)
    else:
        _CACHE["data_override"] = layer


def set_engine(engine: Engine | None) -> None:
    """Override the engine (for tests). Pass ``None`` to clear."""
    if engine is None:
        _CACHE.pop("engine_override", None)
    else:
        _CACHE["engine_override"] = engine


def using_mock_backends() -> bool:
    if "data_override" in _CACHE:
        return False
    return not _use_real()
