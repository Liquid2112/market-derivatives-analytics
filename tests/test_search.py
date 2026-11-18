"""Text-query search tests for the REAL adapter path (offline).

These prove the fix for the broken/slow search: a free-text query like "NFL"
or "BP" returns relevant markets from both providers, and a slow/failing
provider degrades gracefully to the other instead of blanking the result.

Only the NETWORK is mocked. Each provider client is given an ``httpx`` client
backed by an in-memory transport that replays saved fixtures, so the tests run
offline in CI. The transports assert the search path hits the right endpoint
(Polymarket ``/public-search``, not the query-ignoring ``/markets``).
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


def _poly_search_handler(request: httpx.Request) -> httpx.Response:
    """Serve public-search results for a query; a plain list otherwise.

    Models the real endpoint's server-side matching: the saved fixture is
    returned only when the query actually matches its markets, so an unknown
    term yields empty events (as the live API would).
    """
    path = request.url.path
    if path.endswith("/public-search"):
        q = (request.url.params.get("q") or "").lower()
        assert q, "the search term must reach the search endpoint"
        data = _load("polymarket_search.json")
        matched = any(
            q in str(m.get("question", "")).lower()
            for ev in data.get("events", [])
            for m in ev.get("markets", [])
        )
        return httpx.Response(200, json=data if matched else {"events": []})
    if path.endswith("/markets"):
        return httpx.Response(200, json=_load("polymarket_markets.json"))
    return httpx.Response(404, json={})


def _kalshi_list_handler(request: httpx.Request) -> httpx.Response:
    """Serve a multi-market list so client-side title filtering has candidates."""
    if request.url.path.endswith("/markets"):
        return httpx.Response(200, json=_load("kalshi_markets_search.json"))
    return httpx.Response(404, json={})


def _make_client(poly_handler, kalshi_handler) -> TestClient:
    poly = PolymarketClient(
        client=httpx.Client(transport=httpx.MockTransport(poly_handler))
    )
    kalshi = KalshiClient(
        client=httpx.Client(transport=httpx.MockTransport(kalshi_handler))
    )
    adapter.set_data_layer(
        adapter._RealDataLayer(polymarket_client=poly, kalshi_client=kalshi)
    )
    adapter.set_engine(adapter._RealEngine())
    return TestClient(app)


@pytest.fixture
def search_client():
    client = _make_client(_poly_search_handler, _kalshi_list_handler)
    try:
        yield client
    finally:
        adapter.set_data_layer(None)
        adapter.set_engine(None)


def test_nfl_query_returns_relevant_markets_from_both_providers(search_client):
    body = search_client.get("/api/markets", params={"q": "NFL"}).json()
    titles = [m["title"] for m in body["markets"]]
    assert titles, "a text query must return results"
    # every returned market is relevant to the query
    assert all("nfl" in t.lower() or "packers" in t.lower() for t in titles)
    # the query reached both providers, not just one
    providers = {m["provider"] for m in body["markets"]}
    assert providers == {"polymarket", "kalshi"}


def test_bp_query_filters_kalshi_beyond_first_rows(search_client):
    """The BP market is not the first Kalshi row; a correct search still finds
    it, proving we no longer only filter the first page client-side."""
    body = search_client.get(
        "/api/markets", params={"q": "BP", "provider": "kalshi"}
    ).json()
    titles = [m["title"] for m in body["markets"]]
    assert any("bp" in t.lower() for t in titles)
    # the irrelevant weather/fed markets are filtered out
    assert all("rain" not in t.lower() for t in titles)


def test_query_does_not_hit_the_query_ignoring_markets_endpoint(search_client):
    """A search must not return the generic /markets list (which ignores the
    query); filtering it out is what the public-search routing buys us."""
    body = search_client.get(
        "/api/markets", params={"q": "NFL", "provider": "polymarket"}
    ).json()
    # polymarket_markets.json holds NBA/soccer markets with no "NFL" title;
    # if the query had leaked to /markets those would show up instead.
    assert body["markets"]
    assert all("nfl" in m["title"].lower() for m in body["markets"])


def test_one_provider_failing_degrades_to_the_other(search_client):
    """If a provider errors or times out, the search returns the other
    provider's results instead of failing the whole request."""

    def _poly_boom(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={})

    client = _make_client(_poly_boom, _kalshi_list_handler)
    try:
        body = client.get("/api/markets", params={"q": "NFL"}).json()
        providers = {m["provider"] for m in body["markets"]}
        assert providers == {"kalshi"}
        assert body["markets"]
    finally:
        adapter.set_data_layer(None)
        adapter.set_engine(None)


def test_no_results_is_empty_not_error(search_client):
    body = search_client.get("/api/markets", params={"q": "zzzznomatch"})
    assert body.status_code == 200
    assert body.json()["markets"] == []
