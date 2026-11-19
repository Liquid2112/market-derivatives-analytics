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
import os

import httpx
from typing import Any, Protocol, TypedDict


class SearchResult(TypedDict):
    """Result of a market search that also carries provider-level failures.

    ``markets`` is the summaries collected (possibly empty); ``warnings`` is
    one ``{provider, message}`` per provider that failed; ``all_failed`` is
    true only when every requested provider raised.
    """

    markets: list[dict[str, Any]]
    warnings: list[dict[str, str]]
    all_failed: bool


class MarketNotFoundError(KeyError):
    """The requested market genuinely does not exist (maps to HTTP 404)."""


class ProviderError(RuntimeError):
    """An upstream/network failure reaching a provider (maps to HTTP 502)."""


class MarketHasNoPriceError(ValueError):
    """The market exists but has no usable YES price yet (maps to HTTP 422)."""


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

    def search_markets_result(
        self, query: str, provider: str | None = None, limit: int = 25
    ) -> "SearchResult":
        """Like :meth:`search_markets`, but also reports per-provider failures.

        Returns ``{"markets": [...], "warnings": [{provider, message}, ...],
        "all_failed": bool}``. ``all_failed`` is true only when every provider
        that was asked raised (so the API can distinguish an empty-but-healthy
        result from a total outage).
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
        tau: float = 1.0,
        order_book: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a JSON-serializable analytics bundle for a synthetic option.

        ``yes_price`` / ``strike`` are probabilities in ``[0, 1]``.
        ``option_type`` is ``"call"`` or ``"put"``; ``side`` is ``"long"`` or
        ``"short"``. ``tau`` in ``(0, 1]`` is the modeled time-to-resolution: at
        ``tau == 1`` the option is held to resolution and ``premium`` is the exact
        model-free value (``call = p*(1-K)``, ``put = (1-p)*K``); below 1 the
        expiry is before resolution and the premium is the logit-normal model
        value (MODEL.md). ``order_book`` (``{bids: [[price, size], ...], asks:
        [...]}``), when present, sets the implied probability from the top-of-book
        mid and widens the model dispersion by the half-spread. Returns:
        ``{premium, pricing_basis, break_even, implied_prob, prob_of_profit,
        max_gain, max_loss, payoff_curve: [{p, payoff}, ...], greeks: {delta,
        gamma, theta, vega}, greek_curves: {delta: [{p, value}], gamma: [...],
        theta: [...], vega: [...]}}``. ``break_even`` is a YES probability in
        ``[0, 1]`` or ``null`` when no break-even exists in range (never clamped).
        ``pricing_basis`` is ``"held_to_resolution"`` (exact) or ``"model"``.
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

    def search_markets_result(
        self, query: str, provider: str | None = None, limit: int = 25
    ) -> SearchResult:
        # The mock never reaches a network, so it never warns or fails.
        return {
            "markets": self.search_markets(query, provider=provider, limit=limit),
            "warnings": [],
            "all_failed": False,
        }

    def get_market(self, market_id: str) -> dict[str, Any]:
        for m in _MOCK_MARKETS:
            if m["id"] == market_id:
                full = dict(m)
                full["no_price"] = round(1.0 - m["yes_price"], 3)
                full["order_book"] = _seeded_depth(market_id, m["yes_price"])
                return full
        raise KeyError(market_id)


# --------------------------------------------------------------------------
# Engine adapter — one source of truth.
# --------------------------------------------------------------------------
# Both the real and the mock engine reshape the SAME ``derivatives_engine.analyze``
# bundle into the ``{p, payoff}`` / ``{p, value}`` shapes the UI charts consume.
# There is no second pricing model: the mock path differs only in where the
# market data comes from (deterministic fixtures vs. live providers), never in the
# math, so the two can never diverge. See MODEL.md for the model itself.


def _book_from_levels(order_book: dict[str, Any] | None):
    """Dict ``{bids, asks}`` of ``[price, size]`` -> engine ``OrderBook`` or None.

    Tolerates a missing, empty, or one-sided book (the engine falls back to the
    bare YES price when a mid/spread cannot be formed).
    """
    from derivatives_engine import OrderBook, OrderLevel

    if not order_book:
        return None
    bids = order_book.get("bids") or []
    asks = order_book.get("asks") or []
    if not bids and not asks:
        return None
    return OrderBook(
        bids=[OrderLevel(float(px), float(sz)) for px, sz in bids],
        asks=[OrderLevel(float(px), float(sz)) for px, sz in asks],
    )


def _build_bundle(
    yes_price: float,
    strike: float,
    option_type: str,
    side: str,
    tau: float = 1.0,
    order_book: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reshape ``derivatives_engine.analyze`` into the UI's bundle dict.

    The single place the API's engine bundle is built, shared by the real and the
    mock engine so their output cannot diverge. The engine already side-signs the
    payoff curve (it receives ``side``); the point and curve Greeks are returned
    long-side, so we apply the side sign to them here (matching the UI).
    """
    from derivatives_engine import analyze

    option_type = option_type.lower()
    side = side.lower()
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    if side not in ("long", "short"):
        raise ValueError("side must be 'long' or 'short'")

    p = _clamp01(yes_price)
    k = _clamp01(strike)
    book = _book_from_levels(order_book)
    result = analyze(
        p=p,
        strike=k,
        option_type=option_type,
        side=side,
        tau=tau,
        book=book,
        num_points=101,
    )
    sign = 1.0 if side == "long" else -1.0

    pc = result["payoff_curve"]
    payoff_curve = [
        {"p": round(xp, 3), "payoff": round(py, 5)}
        for xp, py in zip(pc["x"], pc["payoff"])
    ]

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
    # A break-even that does not exist in [0, 1] is reported as null, never
    # clamped to a boundary (a clamped value would be a break-even that is not
    # real). The engine returns an empty list in that case.
    break_even = round(be_list[0], 4) if be_list else None

    return {
        "underlying_prob": round(p, 4),
        "strike": round(k, 4),
        "option_type": option_type,
        "side": side,
        "premium": round(result["fair_price"], 5),
        # Whether the premium is the exact model-free value (held to resolution)
        # or the logit-normal model value, so the UI can word its note correctly.
        "pricing_basis": result["pricing_basis"],
        "break_even": break_even,
        "implied_prob": round(result["implied_probability"], 4),
        "prob_of_profit": round(result["prob_of_profit"], 5),
        "max_gain": round(result["max_profit"], 5),
        "max_loss": round(result["max_loss"], 5),
        "payoff_curve": payoff_curve,
        "greeks": {
            "delta": round(sign * g["delta"], 5),
            "gamma": round(sign * g["gamma"], 5),
            "theta": round(sign * g["theta"], 5),
            "vega": round(sign * g["vega"], 5),
        },
        "greek_curves": greek_curves,
    }


class _MockEngine:
    """Engine over the real ``derivatives_engine`` -- identical math to the real
    engine, kept as a separate class only so the offline/mock mode has an engine
    instance to select. It carries no pricing model of its own.
    """

    def bundle(
        self,
        yes_price: float,
        strike: float,
        option_type: str,
        side: str,
        premium: float | None = None,
        tau: float = 1.0,
        order_book: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not (0.0 < tau <= 1.0):
            raise ValueError("tau must be in (0, 1]")
        return _build_bundle(yes_price, strike, option_type, side, tau, order_book)


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


# A bounded per-request timeout (seconds) for live provider calls on the
# search path. Kept short so a slow or unresponsive provider fails fast and
# the view degrades to the other provider instead of hanging for ~20s.
_SEARCH_TIMEOUT = 8.0

# Default time-to-live (seconds) for the per-market fetch cache. A slider
# scrub fires many /api/derivative calls for the same market; caching the
# market + its book for a few seconds collapses those into one live fetch.
# Overridable via MDA_MARKET_CACHE_TTL so it can be tuned/disabled per deploy.
_DEFAULT_MARKET_TTL = 10.0


def _market_cache_ttl() -> float:
    raw = os.getenv("MDA_MARKET_CACHE_TTL")
    if raw is None:
        return _DEFAULT_MARKET_TTL
    try:
        return max(0.0, float(raw))
    except (ValueError, TypeError):
        return _DEFAULT_MARKET_TTL


def _is_not_found(exc: Exception) -> bool:
    """True when an exception means the market genuinely does not exist.

    A 404 from a provider is a real not-found; any other HTTP status, a
    transport error (timeout, connection reset), or a malformed-payload
    ``ValueError``/``KeyError`` is an upstream/network failure, not absence.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 404
    return False


class _RealDataLayer:
    """DataLayer over mdlab's Polymarket + Kalshi read-only clients.
    The clients are injected so tests can pass fixture-backed ``httpx`` clients
    (no network). The default constructor builds live read-only clients.

    ``get_market`` memoizes the (market, book) fetch per composed id for a
    short TTL so a burst of derivative requests for the same market makes one
    live fetch, not one per request. The cache is clearable and uses an
    injectable clock so tests never depend on wall-clock timing or order.
    """

    def __init__(
        self,
        polymarket_client=None,
        kalshi_client=None,
        *,
        cache_ttl: float | None = None,
        time_fn=None,
    ) -> None:
        from mdlab.kalshi import KalshiClient
        from mdlab.polymarket import PolymarketClient
        import time as _time

        # A bounded per-request timeout so one slow provider degrades
        # gracefully (we return what the other gave) instead of the whole
        # search hanging for tens of seconds.
        self._poly = polymarket_client or PolymarketClient(
            timeout=_SEARCH_TIMEOUT
        )
        self._kalshi = kalshi_client or KalshiClient(timeout=_SEARCH_TIMEOUT)
        self._ttl = cache_ttl if cache_ttl is not None else _market_cache_ttl()
        self._now = time_fn or _time.monotonic
        # market_id -> (expires_at, summary_dict)
        self._market_cache: dict[str, tuple[float, dict[str, Any]]] = {}

    def clear_cache(self) -> None:
        """Drop every cached market fetch (tests and manual refresh)."""
        self._market_cache.clear()

    def _client_for(self, provider: str):
        if provider == "polymarket":
            return self._poly
        if provider == "kalshi":
            return self._kalshi
        raise MarketNotFoundError(provider)

    def search_markets(
        self, query: str, provider: str | None = None, limit: int = 25
    ) -> list[dict[str, Any]]:
        return self.search_markets_result(
            query, provider=provider, limit=limit
        )["markets"]

    def search_markets_result(
        self, query: str, provider: str | None = None, limit: int = 25
    ) -> SearchResult:
        q = (query or "").strip() or None
        out: list[dict[str, Any]] = []
        warnings: list[dict[str, str]] = []
        want = (
            [provider] if provider else ["polymarket", "kalshi"]
        )
        failed = 0
        # Each provider is queried independently: a timeout or error from one
        # must not blank out results the other returned, so we capture the
        # failure per-provider as a warning and keep what the others gave.
        for name in want:
            rows, warning = self._safe_list(self._client_for(name), name, q, limit)
            out.extend(rows)
            if warning is not None:
                warnings.append(warning)
                failed += 1
        # Rank the default landing view by traded volume so popular,
        # recognizable markets lead; a text search keeps relevance order.
        if q is None:
            out.sort(key=_popularity_key, reverse=True)
        return {
            "markets": out[:limit],
            "warnings": warnings,
            # Total failure only when every requested provider raised.
            "all_failed": failed == len(want) and failed > 0,
        }

    @staticmethod
    def _safe_list(
        client, provider: str, q: str | None, limit: int
    ) -> tuple[list[dict[str, Any]], dict[str, str] | None]:
        """Fetch + summarize one provider.

        Returns ``(summaries, None)`` on success or ``([], {provider, message})``
        when the provider errored, so the caller can surface the outage instead
        of silently dropping it.
        """
        try:
            markets = client.list_markets(limit=limit, search=q)
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            return [], {
                "provider": provider,
                "message": f"{provider} is temporarily unavailable ({type(exc).__name__}).",
            }
        summaries = [
            _market_summary(m) for m in markets if m.yes_price is not None
        ]
        return summaries, None

    def get_market(self, market_id: str) -> dict[str, Any]:
        """Fetch one market + its book, memoized per id for ``self._ttl`` sec.

        Raises :class:`MarketNotFoundError` only when the market genuinely does
        not exist (a provider 404), and :class:`ProviderError` for an upstream
        or network failure, so the API can map them to 404 vs 502 instead of
        collapsing every failure into a 404.
        """
        now = self._now()
        hit = self._market_cache.get(market_id)
        if hit is not None and hit[0] > now:
            return hit[1]

        provider, native = _split_id(market_id)
        client = self._client_for(provider)
        try:
            market = client.get_market(native)
            # Polymarket's order-book fetch would otherwise re-fetch the market
            # metadata just to read the YES token id; reuse the payload we just
            # fetched so one derivative request = one market fetch + one book
            # fetch, not two market fetches. Kalshi's book endpoint is direct.
            if provider == "polymarket":
                book = client.get_order_book(native, raw=market.raw)
            else:
                book = client.get_order_book(native)
        except Exception as exc:
            if _is_not_found(exc):
                raise MarketNotFoundError(market_id) from exc
            raise ProviderError(
                f"could not reach {provider} for market '{native}'"
            ) from exc
        summary = _market_summary(market)
        summary["order_book"] = _book_to_levels(book)
        if self._ttl > 0:
            self._market_cache[market_id] = (now + self._ttl, summary)
        return summary


# --------------------------------------------------------------------------
# Real engine — wraps derivatives_engine.analyze and reshapes its bundle into
# the {p, payoff} / {p, value} shapes the UI charts consume.
# --------------------------------------------------------------------------
class _RealEngine:
    """Engine over ``derivatives_engine.analyze`` via the shared bundle builder.

    Carries no pricing math of its own: it delegates to :func:`_build_bundle`,
    the one place the bundle is shaped, so the real and mock engines cannot
    diverge. The order book is passed through to the engine, which sets the
    implied probability from the top-of-book mid and widens the model dispersion
    by the half-spread (MODEL.md).
    """

    def bundle(
        self,
        yes_price: float,
        strike: float,
        option_type: str,
        side: str,
        premium: float | None = None,
        tau: float = 1.0,
        order_book: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return _build_bundle(yes_price, strike, option_type, side, tau, order_book)


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
