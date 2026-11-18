"""Integration tests for the REAL adapter path (mdlab -> engine -> bundle).

These exercise the real wiring that production uses: the ``mdlab`` Polymarket
and Kalshi clients, the ``derivatives_engine`` analytics, and the adapter's
reshaping into the API/UI bundle shape. Only the NETWORK is mocked — each
provider client is given an ``httpx`` client backed by an in-memory transport
that replays the saved fixtures in ``fixtures/``. No mock data layer and no
mock engine are involved.

The point of these tests is to catch shape mismatches between the three real
layers that the mock-backed ``test_api.py`` cannot see, and to prove the four
Greeks come out of real computed data.
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
# In-memory transports that replay fixtures based on the request path.
# --------------------------------------------------------------------------
def _polymarket_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("/book"):
        return httpx.Response(200, json=_load("polymarket_book.json"))
    markets = _load("polymarket_markets.json")
    # GET /markets/{id}
    parts = [p for p in path.split("/") if p]
    if len(parts) >= 2 and parts[-2] == "markets":
        mid = parts[-1]
        for m in markets:
            if str(m.get("id")) == mid:
                return httpx.Response(200, json=m)
        return httpx.Response(404, json={})
    # GET /markets  (list / search)
    return httpx.Response(200, json=markets)


def _kalshi_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("/orderbook"):
        return httpx.Response(200, json=_load("kalshi_book.json"))
    market = _load("kalshi_market.json")  # {"market": {...}}
    parts = [p for p in path.split("/") if p]
    if parts and parts[-1] == "markets":
        # list
        return httpx.Response(200, json={"markets": [market["market"]]})
    if len(parts) >= 2 and parts[-2] == "markets":
        # single
        return httpx.Response(200, json=market)
    return httpx.Response(404, json={})


@pytest.fixture
def real_client():
    """FastAPI client whose adapter uses the real data layer + real engine,
    with both provider clients backed by fixture transports (no network)."""
    poly = PolymarketClient(
        client=httpx.Client(transport=httpx.MockTransport(_polymarket_handler))
    )
    kalshi = KalshiClient(
        client=httpx.Client(transport=httpx.MockTransport(_kalshi_handler))
    )
    real_data = adapter._RealDataLayer(polymarket_client=poly, kalshi_client=kalshi)
    adapter.set_data_layer(real_data)
    adapter.set_engine(adapter._RealEngine())
    try:
        yield TestClient(app)
    finally:
        adapter.set_data_layer(None)
        adapter.set_engine(None)


# --------------------------------------------------------------------------
# Data layer: mdlab Market/OrderBook mapped into the API shapes.
# --------------------------------------------------------------------------
def test_real_search_returns_both_providers(real_client):
    body = real_client.get("/api/markets").json()
    assert body["count"] >= 2
    providers = {m["provider"] for m in body["markets"]}
    assert providers == {"polymarket", "kalshi"}
    for m in body["markets"]:
        assert 0.0 <= m["yes_price"] <= 1.0
        # ids are provider-prefixed so they round-trip through the UI
        assert m["id"].startswith(m["provider"] + ":")


def test_real_provider_filter(real_client):
    body = real_client.get("/api/markets", params={"provider": "kalshi"}).json()
    assert body["count"] >= 1
    assert all(m["provider"] == "kalshi" for m in body["markets"])


def test_real_default_view_ranked_by_volume_desc(real_client):
    """The default landing view (no query) is ordered by traded volume across
    both providers, so popular markets lead regardless of provider fetch order.
    The fixtures interleave providers whose volumes are out of order, so a pass
    proves the adapter re-ranked them rather than echoing the fetch order."""
    markets = real_client.get("/api/markets").json()["markets"]
    vols = [m["volume"] for m in markets if m["volume"] is not None]
    assert vols == sorted(vols, reverse=True)
    # any market missing a volume sorts after every market that reported one
    reported = [m["volume"] is not None for m in markets]
    assert reported == sorted(reported, reverse=True)


def test_real_get_market_has_live_order_book(real_client):
    mid = real_client.get(
        "/api/markets", params={"provider": "polymarket"}
    ).json()["markets"][0]["id"]
    m = real_client.get(f"/api/market/{mid}").json()
    assert m["id"] == mid
    ob = m["order_book"]
    assert ob["bids"] and ob["asks"]
    for px, size in ob["bids"] + ob["asks"]:
        assert 0.0 <= px <= 1.0 and size >= 0.0


def test_real_kalshi_order_book_no_side_folds_to_yes_asks(real_client):
    mid = real_client.get(
        "/api/markets", params={"provider": "kalshi"}
    ).json()["markets"][0]["id"]
    ob = real_client.get(f"/api/market/{mid}").json()["order_book"]
    # the real Kalshi normalizer folds NO bids into YES asks; book is uncrossed
    best_bid = max(px for px, _ in ob["bids"])
    best_ask = min(px for px, _ in ob["asks"])
    assert best_bid <= best_ask


# --------------------------------------------------------------------------
# Engine: real analytics reshaped into the bundle, the four Greeks headline.
# --------------------------------------------------------------------------
def _first_market_id(client, provider):
    return client.get("/api/markets", params={"provider": provider}).json()[
        "markets"
    ][0]["id"]


@pytest.mark.parametrize("provider", ["polymarket", "kalshi"])
def test_real_derivative_bundle_shape_and_greeks(real_client, provider):
    mid = _first_market_id(real_client, provider)
    b = real_client.get(
        f"/api/derivative/{mid}",
        params={"option_type": "call", "side": "long", "strike": 0.5},
    ).json()

    # the four Greeks are present, real numbers (the headline feature)
    for k in ("delta", "gamma", "theta", "vega"):
        assert k in b["greeks"]
        assert isinstance(b["greeks"][k], (int, float))
    # at least one Greek is non-zero -> computed from real data, not a placeholder
    assert any(b["greeks"][k] != 0.0 for k in ("delta", "gamma", "theta", "vega"))

    # payoff and every Greek curve span the 0..1 grid in 101 points
    assert len(b["payoff_curve"]) == 101
    assert b["payoff_curve"][0]["p"] == 0.0 and b["payoff_curve"][-1]["p"] == 1.0
    for name in ("delta", "gamma", "theta", "vega"):
        assert len(b["greek_curves"][name]) == 101
        assert all("value" in pt for pt in b["greek_curves"][name])
    assert 0.0 <= b["break_even"] <= 1.0
    assert b["market"]["id"] == mid


def test_real_long_call_delta_matches_engine(real_client):
    """The bundle's delta equals derivatives_engine.greeks() on the market's
    real YES price — proves the number is the engine's, not reshaped noise."""
    from derivatives_engine import greeks

    mid = _first_market_id(real_client, "polymarket")
    market = real_client.get(f"/api/market/{mid}").json()
    p = market["yes_price"]
    b = real_client.get(
        f"/api/derivative/{mid}",
        params={"option_type": "call", "side": "long", "strike": 0.5},
    ).json()
    expected = greeks(p, 0.5, "call")["delta"]
    assert b["greeks"]["delta"] == pytest.approx(round(expected, 5), abs=1e-5)


def test_real_short_is_sign_flip_of_long(real_client):
    mid = _first_market_id(real_client, "polymarket")
    lng = real_client.get(
        f"/api/derivative/{mid}",
        params={"option_type": "put", "side": "long", "strike": 0.6},
    ).json()
    sht = real_client.get(
        f"/api/derivative/{mid}",
        params={"option_type": "put", "side": "short", "strike": 0.6},
    ).json()
    for k in ("delta", "gamma", "theta", "vega"):
        assert lng["greeks"][k] == pytest.approx(-sht["greeks"][k], abs=1e-6)


def test_real_call_payoff_monotonic_nondecreasing(real_client):
    mid = _first_market_id(real_client, "kalshi")
    b = real_client.get(
        f"/api/derivative/{mid}",
        params={"option_type": "call", "side": "long", "strike": 0.4},
    ).json()
    vals = [pt["payoff"] for pt in b["payoff_curve"]]
    assert all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1))


def test_real_health_reports_live_when_overridden(real_client):
    # with a real data layer injected, health must not claim mock backends
    h = real_client.get("/api/health").json()
    assert h["status"] == "ok"
    assert h["using_mock_backends"] is False
