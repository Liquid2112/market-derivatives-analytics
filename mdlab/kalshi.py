"""Kalshi adapter: fetch + normalize into the shared schema.

Read-only. Uses Kalshi's public trade-api v2 market-data endpoints
(``https://api.elections.kalshi.com/trade-api/v2``):

* ``GET /markets`` for discovery/search and ``GET /markets/{ticker}`` for a
  single market's metadata.
* ``GET /markets/{ticker}/orderbook`` for the live book.

Prices arrive as decimal-dollar strings in ``[0, 1]`` (e.g. ``"0.2100"`` for
21c). The order book is two-sided but both sides are expressed as *bids*:
``yes_dollars`` are resting YES bids, ``no_dollars`` are resting NO bids. A NO
bid at price ``p`` is economically a YES ask at ``1 - p``, so we fold the NO
side into YES asks to produce a standard single-outcome book.

Market-data reads here are public and need no credentials. An optional API key
can be supplied; when present it is sent as the documented header. The default
path works unauthenticated. The normalizers are pure functions over parsed
JSON so tests run against fixtures with no network.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from .schema import Market, OrderBook, PriceLevel, Provider, clamp_probability

API_BASE = "https://api.elections.kalshi.com/trade-api/v2"

_DEFAULT_TIMEOUT = 20.0


def _to_price(value: Any) -> Optional[float]:
    """Parse a Kalshi dollar-string price into a 0..1 float, or ``None``."""
    if value is None or value == "":
        return None
    try:
        return clamp_probability(float(value))
    except (ValueError, TypeError):
        return None


def _parse_ts(value: Any) -> Optional[int]:
    """Parse an ISO-8601 time string into epoch milliseconds, or ``None``."""
    if not value or not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except (ValueError, TypeError):
        return None


def normalize_order_book(raw: dict) -> OrderBook:
    """Normalize a Kalshi ``/orderbook`` payload into an :class:`OrderBook`.

    Accepts the ``orderbook_fp`` envelope seen on the elections API
    (``yes_dollars`` / ``no_dollars`` as ``[price, size]`` pairs) and the
    legacy ``orderbook`` envelope (``yes`` / ``no`` as ``[cents, count]``
    integer pairs). YES levels become bids; NO levels are mirrored to YES
    asks at ``1 - price``. Both sides are returned best-price-first.
    """
    book = raw.get("orderbook_fp") or raw.get("orderbook") or {}
    legacy = "orderbook_fp" not in raw and "orderbook" in raw

    yes_levels = book.get("yes_dollars")
    no_levels = book.get("no_dollars")
    if yes_levels is None and no_levels is None:
        yes_levels = book.get("yes")
        no_levels = book.get("no")
        legacy = True

    def _price(v: Any) -> float:
        return float(v) / 100.0 if legacy else float(v)

    bids = [
        PriceLevel(price=_price(p), size=float(s))
        for p, s in (yes_levels or [])
    ]
    # NO bid at p == YES ask at 1 - p.
    asks = [
        PriceLevel(price=clamp_probability(1.0 - _price(p)), size=float(s))
        for p, s in (no_levels or [])
    ]
    bids.sort(key=lambda lvl: lvl.price, reverse=True)
    asks.sort(key=lambda lvl: lvl.price)
    return OrderBook(bids=bids, asks=asks)


def _quote_price(value: Any) -> Optional[float]:
    """Parse a Kalshi quote/last field, treating a 0 as 'no quote'.

    Kalshi prices live in cents (minimum $0.01), so ``"0.0000"`` is not a
    one-cent quote — it is the sentinel for *no resting order* (an empty bid or
    ask side) or *never traded* (last price). We map it to ``None`` so a market
    with no real quote surfaces as 'no price' instead of a misleading $0.00.
    """
    p = _to_price(value)
    if p is None or p <= 0.0:
        return None
    return p


def _resolve_price(
    bid: Optional[float], ask: Optional[float], last: Optional[float]
) -> Optional[float]:
    """Pick the best usable price from a quote.

    Prefers the bid/ask midpoint when both sides exist (a live, two-sided
    quote), falls back to whichever single side is present, then to the last
    trade. Returns ``None`` when nothing usable exists — a market that has
    never traded and has no resting quote has *no price*, which is distinct
    from a real ``0.0``. The last trade is used only as a last resort because
    it can be stale, or ``0`` for a market that never traded.
    """
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    if bid is not None:
        return bid
    if ask is not None:
        return ask
    return last


def normalize_market(
    raw: dict, *, order_book: Optional[OrderBook] = None
) -> Market:
    """Normalize a Kalshi market object into a :class:`Market`.

    ``yes_price`` prefers the bid/ask midpoint when both sides exist, then a
    single resting side, then the last trade, and is ``None`` when the market
    has no usable price (never traded, no resting quote) rather than a
    misleading ``0``. ``no_price`` is derived the same way from the NO quote,
    falling back to ``1 - yes_price``.
    """
    yes_bid = _quote_price(raw.get("yes_bid_dollars"))
    yes_ask = _quote_price(raw.get("yes_ask_dollars"))
    no_bid = _quote_price(raw.get("no_bid_dollars"))
    no_ask = _quote_price(raw.get("no_ask_dollars"))
    last = _quote_price(raw.get("last_price_dollars"))

    yes_price = _resolve_price(yes_bid, yes_ask, last)
    no_price = _resolve_price(no_bid, no_ask, None)
    if no_price is None and yes_price is not None:
        no_price = clamp_probability(1.0 - yes_price)

    status = raw.get("status")
    if isinstance(status, str):
        status = status.lower()

    volume = None
    if raw.get("volume_fp") is not None:
        try:
            volume = float(raw["volume_fp"])
        except (ValueError, TypeError):
            volume = None

    ticker = str(raw.get("ticker") or "")
    event = raw.get("event_ticker")
    url = (
        f"https://kalshi.com/markets/{str(event).lower()}"
        if event
        else None
    )

    if order_book is not None and order_book.timestamp is None:
        ts = _parse_ts(raw.get("updated_time"))
        if ts is not None:
            order_book = OrderBook(
                bids=order_book.bids, asks=order_book.asks, timestamp=ts
            )

    return Market(
        provider=Provider.KALSHI,
        id=ticker,
        title=str(raw.get("title") or raw.get("yes_sub_title") or ""),
        yes_price=yes_price,
        no_price=no_price,
        order_book=order_book,
        status=status,
        close_time=raw.get("close_time"),
        volume=volume,
        url=url,
        raw=raw,
    )


class KalshiClient:
    """Read-only HTTP client for Kalshi public market data.

    Market-data reads are public; ``api_key`` is optional and only added as a
    header when provided. No credentials are needed for the default path.

    Example:
        >>> client = KalshiClient()
        >>> markets = client.list_markets(limit=5)   # doctest: +SKIP
        >>> book = client.get_order_book(markets[0].id)   # doctest: +SKIP
    """

    def __init__(
        self,
        api_base: str = API_BASE,
        timeout: float = _DEFAULT_TIMEOUT,
        api_key: Optional[str] = None,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        headers = {"Accept": "application/json"}
        if api_key:
            # Optional authenticated path. Kalshi's documented key header;
            # unauthenticated public reads work without it.
            headers["KALSHI-ACCESS-KEY"] = api_key
        self._client = client or httpx.Client(timeout=timeout, headers=headers)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "KalshiClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # When searching, Kalshi's public API has no text param, so we page
    # through the open book and filter client-side. Bounded so one slow page
    # cannot stall the request; Kalshi caps page size at 1000.
    _SEARCH_PAGE_SIZE = 1000
    _SEARCH_MAX_PAGES = 8

    def list_markets(
        self,
        *,
        limit: int = 20,
        search: Optional[str] = None,
        status: Optional[str] = "open",
        series_ticker: Optional[str] = None,
    ) -> list[Market]:
        """List or search markets (metadata only, no order book).

        ``series_ticker`` narrows to one Kalshi series. The public API has no
        free-text search param, so ``search`` is applied client-side against
        the title/subtitle.

        Multivariate/combo "parlay" markets (``KXMVE*`` tickers) are dropped
        via the documented ``mve_filter=exclude`` param: the open book is
        otherwise dominated by tens of thousands of $0-volume parlay shards
        that bury every real market. Because results come back newest-first
        (not by liquidity), the default (no-search) view pages until it has
        collected ``limit`` markets that have actually traded (non-zero
        volume), so the landing view shows real, liquid markets instead of a
        page of freshly created $0 ones.
        """
        if not search:
            return self._list_liquid(
                limit=limit, status=status, series_ticker=series_ticker
            )

        needle = search.lower()
        matches: list[Market] = []
        cursor: Optional[str] = None
        for _ in range(self._SEARCH_MAX_PAGES):
            params: dict[str, Any] = {
                "limit": self._SEARCH_PAGE_SIZE,
                "mve_filter": "exclude",
            }
            if status:
                params["status"] = status
            if series_ticker:
                params["series_ticker"] = series_ticker
            if cursor:
                params["cursor"] = cursor
            resp = self._client.get(f"{self.api_base}/markets", params=params)
            resp.raise_for_status()
            body = resp.json()
            rows = body.get("markets", [])
            for row in rows:
                m = normalize_market(row)
                if needle in m.title.lower():
                    matches.append(m)
                    if len(matches) >= limit:
                        return matches
            cursor = body.get("cursor")
            if not cursor or not rows:
                break
        # shortcut: Kalshi has no server-side text search, so a term whose
        # markets lie past _SEARCH_MAX_PAGES of the open book is missed; raise
        # the page bound or add a Kalshi search endpoint if one ships.
        return matches

    def _list_liquid(
        self,
        *,
        limit: int,
        status: Optional[str],
        series_ticker: Optional[str],
    ) -> list[Market]:
        """Default view: page the open book for traded (non-parlay) markets.

        Excludes ``KXMVE*`` parlays and keeps only markets with non-zero
        volume, paging newest-first until ``limit`` are collected or the page
        bound is hit. If the bound is reached without filling ``limit`` (a
        quiet book), any non-parlay markets seen are returned as a fallback so
        the view is never empty.
        """
        liquid: list[Market] = []
        seen: list[Market] = []
        cursor: Optional[str] = None
        for _ in range(self._SEARCH_MAX_PAGES):
            params: dict[str, Any] = {
                "limit": self._SEARCH_PAGE_SIZE,
                "mve_filter": "exclude",
            }
            if status:
                params["status"] = status
            if series_ticker:
                params["series_ticker"] = series_ticker
            if cursor:
                params["cursor"] = cursor
            resp = self._client.get(f"{self.api_base}/markets", params=params)
            resp.raise_for_status()
            body = resp.json()
            rows = body.get("markets", [])
            for row in rows:
                m = normalize_market(row)
                seen.append(m)
                if m.volume is not None and m.volume > 0.0:
                    liquid.append(m)
                    if len(liquid) >= limit:
                        return liquid
            cursor = body.get("cursor")
            if not cursor or not rows:
                break
        # Quiet book: nothing with volume within the page bound. Fall back to
        # the non-parlay markets we did see so the view is never blank.
        return (liquid or seen)[:limit]

    def get_market(self, ticker: str) -> Market:
        """Fetch a single market's metadata by ticker (no order book)."""
        resp = self._client.get(f"{self.api_base}/markets/{ticker}")
        resp.raise_for_status()
        return normalize_market(resp.json().get("market", {}))

    def get_order_book(self, ticker: str, *, depth: int = 10) -> OrderBook:
        """Fetch the live order book for a market ticker."""
        resp = self._client.get(
            f"{self.api_base}/markets/{ticker}/orderbook",
            params={"depth": depth},
        )
        resp.raise_for_status()
        return normalize_order_book(resp.json())
