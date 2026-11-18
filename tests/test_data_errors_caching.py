"""Tests for audit findings 13-16: Kalshi parlay filtering, error surfacing,
and per-market fetch caching. All offline — the network is replayed from
fixtures or a counting stub, never hit.

Findings covered:
  13. KalshiClient drops KXMVE parlay markets (mve_filter=exclude) and the
      default list surfaces markets that have actually traded.
  15. GET /api/markets surfaces per-provider failures in an additive
      ``warnings`` field and 502s on a total outage; get_market returns 404
      only for a genuine not-found and 502 for an upstream/network failure;
      /api/derivative returns 422 for a market with no YES price.
  16. A derivative request fetches a market once, and a repeat within the
      cache TTL makes zero extra live calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app import adapter
from app.main import app
from mdlab.kalshi import KalshiClient
from mdlab.polymarket import PolymarketClient

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _load(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8-sig"))


# --------------------------------------------------------------------------
# Finding 13: parlay filtering + traded-market default view (KalshiClient).
# --------------------------------------------------------------------------
def test_kalshi_default_list_sends_mve_filter_exclude_and_drops_parlays():
    """The default (no-search) list must request mve_filter=exclude so the
    KXMVE parlay shards never reach the user, and must surface markets that
    have actually traded (non-zero volume)."""
    seen_params: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_params.append(dict(request.url.params))
        # The real API, unfiltered, returns parlays first; filtered it returns
        # real markets. Model that: honour the filter the client sends.
        if request.url.params.get("mve_filter") == "exclude":
            return httpx.Response(200, json=_load("kalshi_markets_mixed.json"))
        return httpx.Response(
            200,
            json={"markets": [{"ticker": "KXMVECROSS-1", "title": "parlay"}]},
        )

    client = KalshiClient(
        client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    markets = client.list_markets(limit=10)
    # the filter was actually requested
    assert all(p.get("mve_filter") == "exclude" for p in seen_params)
    tickers = [m.id for m in markets]
    assert tickers, "the default list must not be empty"
    assert not any(t.startswith("KXMVE") for t in tickers)
    # every surfaced market has actually traded
    assert all(m.volume and m.volume > 0 for m in markets)


def test_kalshi_search_sends_mve_filter_exclude():
    """The search path must also exclude parlays so a 'bitcoin' query is not
    drowned by parlay shards."""
    seen_params: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_params.append(dict(request.url.params))
        return httpx.Response(200, json=_load("kalshi_markets_mixed.json"))

    client = KalshiClient(
        client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    results = client.list_markets(limit=10, search="bitcoin")
    assert all(p.get("mve_filter") == "exclude" for p in seen_params)
    titles = [m.title.lower() for m in results]
    assert any("bitcoin" in t for t in titles)
    # the unrelated Fed market is filtered out by the title match
    assert all("fed" not in t for t in titles)


def test_kalshi_quiet_book_falls_back_to_seen_markets():
    """If no market in the page bound has traded, the client still returns the
    non-parlay markets it saw rather than an empty list."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "markets": [
                    {
                        "ticker": "KXQUIET-1",
                        "title": "quiet",
                        "yes_bid_dollars": "0.40",
                        "yes_ask_dollars": "0.44",
                        "volume_fp": "0.00",
                    }
                ]
            },
        )

    client = KalshiClient(
        client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    markets = client.list_markets(limit=5)
    assert len(markets) == 1
    assert markets[0].id == "KXQUIET-1"


# --------------------------------------------------------------------------
# Finding 15a: /api/markets surfaces provider failures; total outage -> 502.
# --------------------------------------------------------------------------
def _kalshi_ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=_load("kalshi_markets_mixed.json"))


def _poly_ok(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("/public-search"):
        return httpx.Response(200, json=_load("polymarket_search.json"))
    if path.endswith("/book"):
        return httpx.Response(200, json=_load("polymarket_book.json"))
    return httpx.Response(200, json=_load("polymarket_markets.json"))


def _boom(request: httpx.Request) -> httpx.Response:
    return httpx.Response(500, json={})


def _make_client(poly_handler, kalshi_handler, **kwargs) -> TestClient:
    poly = PolymarketClient(
        client=httpx.Client(transport=httpx.MockTransport(poly_handler))
    )
    kalshi = KalshiClient(
        client=httpx.Client(transport=httpx.MockTransport(kalshi_handler))
    )
    adapter.set_data_layer(
        adapter._RealDataLayer(
            polymarket_client=poly, kalshi_client=kalshi, **kwargs
        )
    )
    adapter.set_engine(adapter._RealEngine())
    return TestClient(app)


@pytest.fixture(autouse=True)
def _reset_overrides():
    yield
    adapter.set_data_layer(None)
    adapter.set_engine(None)


def test_one_provider_failing_adds_a_warning_but_still_200():
    """A provider outage is surfaced as an additive ``warnings`` entry; the
    other provider's results still come back with a 200 (markets/count keys
    unchanged)."""
    client = _make_client(_boom, _kalshi_ok)
    r = client.get("/api/markets", params={"q": "bitcoin"})
    assert r.status_code == 200
    body = r.json()
    assert "markets" in body and "count" in body
    assert body["markets"], "the healthy provider's results must survive"
    assert {m["provider"] for m in body["markets"]} == {"kalshi"}
    warnings = body["warnings"]
    assert any(w["provider"] == "polymarket" for w in warnings)


def test_total_outage_returns_502_not_empty_200():
    """When every requested provider fails and nothing is found, the response
    is a 502, not a misleading empty 200."""
    client = _make_client(_boom, _boom)
    r = client.get("/api/markets", params={"q": "bitcoin"})
    assert r.status_code == 502
    detail = r.json()["detail"]
    assert "unavailable" in detail["message"].lower()
    assert len(detail["warnings"]) == 2


def test_healthy_search_has_no_warnings_key():
    """A clean search adds no warnings field, keeping the response minimal."""
    client = _make_client(_poly_ok, _kalshi_ok)
    r = client.get("/api/markets", params={"q": "nfl"})
    assert r.status_code == 200
    assert "warnings" not in r.json()


def test_empty_result_with_both_healthy_is_200_not_502():
    """No matches but both providers healthy is a normal empty 200, not 502."""

    def _poly_empty(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/public-search"):
            return httpx.Response(200, json={"events": []})
        return httpx.Response(200, json=[])

    def _kalshi_empty(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"markets": []})

    client = _make_client(_poly_empty, _kalshi_empty)
    r = client.get("/api/markets", params={"q": "zzzznomatch"})
    assert r.status_code == 200
    body = r.json()
    assert body["markets"] == []
    assert "warnings" not in body


# --------------------------------------------------------------------------
# Finding 15b: get_market distinguishes 404 (absent) from 502 (upstream).
# --------------------------------------------------------------------------
def test_get_market_network_failure_is_502_not_404():
    client = _make_client(_poly_ok, _boom)
    # ask Kalshi for a ticker; the upstream 500 must map to 502, not 404
    r = client.get("/api/market/kalshi:KXANY-1")
    assert r.status_code == 502


def test_get_market_genuine_not_found_is_404():
    def _poly_404(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={})

    client = _make_client(_poly_404, _kalshi_ok)
    r = client.get("/api/market/polymarket:does-not-exist")
    assert r.status_code == 404


# --------------------------------------------------------------------------
# Finding 15c: a market with no YES price -> 422 from /api/derivative.
# --------------------------------------------------------------------------
def test_derivative_on_no_price_market_returns_422():
    def _kalshi_no_price(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/orderbook"):
            return httpx.Response(
                200, json={"orderbook_fp": {"yes_dollars": [], "no_dollars": []}}
            )
        return httpx.Response(200, json=_load("kalshi_market_no_price.json"))

    client = _make_client(_poly_ok, _kalshi_no_price)
    r = client.get(
        "/api/market/kalshi:KXQUIET-1"
    )
    assert r.status_code == 200
    assert r.json()["yes_price"] is None
    d = client.get(
        "/api/derivative/kalshi:KXQUIET-1",
        params={"option_type": "call", "side": "long", "strike": 0.5},
    )
    assert d.status_code == 422
    assert "no yes price" in d.json()["detail"].lower()


# --------------------------------------------------------------------------
# Finding 16: per-market fetch caching. Count live calls on a stub client.
# --------------------------------------------------------------------------
class _CountingClient:
    """Minimal stub standing in for an mdlab client; counts fetch calls."""

    def __init__(self, provider: str):
        from mdlab.schema import Market, OrderBook, PriceLevel, Provider

        self._Market = Market
        self._OrderBook = OrderBook
        self._PriceLevel = PriceLevel
        self._prov = Provider(provider)
        self.market_calls = 0
        self.book_calls = 0

    def list_markets(self, **kwargs):
        return []

    def get_market(self, native):
        self.market_calls += 1
        return self._Market(
            provider=self._prov,
            id=native,
            title="stub market",
            yes_price=0.42,
            no_price=0.58,
            raw={},
        )

    def get_order_book(self, native, **kwargs):
        self.book_calls += 1
        return self._OrderBook(
            bids=[self._PriceLevel(0.41, 100.0)],
            asks=[self._PriceLevel(0.43, 100.0)],
        )


def _layer_with_counting(ttl, time_fn):
    poly = _CountingClient("polymarket")
    kalshi = _CountingClient("kalshi")
    layer = adapter._RealDataLayer(
        polymarket_client=poly,
        kalshi_client=kalshi,
        cache_ttl=ttl,
        time_fn=time_fn,
    )
    return layer, poly


def test_one_market_fetch_is_one_market_call_plus_one_book_call():
    """A single get_market is exactly one market fetch + one book fetch (no
    duplicate market fetch to read the order-book token)."""
    layer, poly = _layer_with_counting(ttl=10.0, time_fn=lambda: 0.0)
    layer.get_market("polymarket:7111962")
    assert poly.market_calls == 1
    assert poly.book_calls == 1


def test_repeat_within_ttl_makes_zero_extra_live_calls():
    """A repeated get_market for the same id within the TTL is served from the
    cache and makes no extra live calls — the slider-scrub case."""
    clock = {"t": 100.0}
    layer, poly = _layer_with_counting(ttl=10.0, time_fn=lambda: clock["t"])
    layer.get_market("polymarket:7111962")
    clock["t"] = 105.0  # still inside the 10s TTL
    layer.get_market("polymarket:7111962")
    layer.get_market("polymarket:7111962")
    assert poly.market_calls == 1
    assert poly.book_calls == 1


def test_cache_expires_after_ttl():
    clock = {"t": 0.0}
    layer, poly = _layer_with_counting(ttl=10.0, time_fn=lambda: clock["t"])
    layer.get_market("polymarket:7111962")
    clock["t"] = 20.0  # past the TTL
    layer.get_market("polymarket:7111962")
    assert poly.market_calls == 2


def test_clear_cache_forces_a_refetch():
    layer, poly = _layer_with_counting(ttl=10.0, time_fn=lambda: 0.0)
    layer.get_market("polymarket:7111962")
    layer.clear_cache()
    layer.get_market("polymarket:7111962")
    assert poly.market_calls == 2


def test_cache_does_not_leak_across_market_ids():
    """Different markets are cached independently, so one does not serve
    another (and tests do not depend on each other's order)."""
    layer, poly = _layer_with_counting(ttl=10.0, time_fn=lambda: 0.0)
    layer.get_market("polymarket:aaa")
    layer.get_market("polymarket:bbb")
    assert poly.market_calls == 2
