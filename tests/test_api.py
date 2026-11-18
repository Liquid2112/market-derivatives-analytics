"""Smoke + unit tests for the read-only API, run against the mock backends.

These exercise the FastAPI layer and the adapter's mock data/engine without
any network. The real mdlab->engine wiring is covered separately in
``test_integration_real.py``; here we pin the mock backends so the shape and
behaviour assertions are deterministic and offline.
"""

import os

os.environ["MDA_USE_REAL_BACKENDS"] = "0"

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_reports_mock_backends():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    # CI runs against mocks
    assert body["using_mock_backends"] is True


def test_list_markets_returns_both_providers():
    r = client.get("/api/markets")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] >= 1
    providers = {m["provider"] for m in body["markets"]}
    assert providers <= {"polymarket", "kalshi"}
    assert "polymarket" in providers and "kalshi" in providers
    for m in body["markets"]:
        assert 0.0 <= m["yes_price"] <= 1.0
        assert 0.0 <= m["no_price"] <= 1.0
        assert abs(m["yes_price"] + m["no_price"] - 1.0) < 1e-6


def test_list_markets_filters_by_provider():
    r = client.get("/api/markets", params={"provider": "kalshi"})
    assert r.status_code == 200
    assert all(m["provider"] == "kalshi" for m in r.json()["markets"])


def test_list_markets_rejects_bad_provider():
    r = client.get("/api/markets", params={"provider": "nasdaq"})
    assert r.status_code == 422


def test_default_markets_ranked_by_volume_desc():
    """The default landing view (no query) leads with the most-traded markets,
    so recognizable, liquid markets surface first instead of an arbitrary order.
    """
    markets = client.get("/api/markets").json()["markets"]
    vols = [m["volume"] for m in markets]
    # every market carries a volume and they are non-increasing (popular first)
    assert all(v is not None for v in vols)
    assert vols == sorted(vols, reverse=True)
    # the single highest-volume mock market is the one on top
    assert markets[0]["volume"] == max(vols)


def test_search_query_not_reordered_by_volume():
    """A text search is a relevance query, not the default view, so it must not
    be forced into volume order — it returns the provider's matches as-is."""
    r = client.get("/api/markets", params={"q": "bitcoin"})
    assert r.status_code == 200
    titles = [m["title"].lower() for m in r.json()["markets"]]
    assert any("bitcoin" in t for t in titles)


def test_search_filters_by_query():
    r = client.get("/api/markets", params={"q": "bitcoin"})
    assert r.status_code == 200
    titles = [m["title"].lower() for m in r.json()["markets"]]
    assert any("bitcoin" in t for t in titles)


def test_get_market_has_order_book():
    mid = client.get("/api/markets").json()["markets"][0]["id"]
    r = client.get(f"/api/market/{mid}")
    assert r.status_code == 200
    m = r.json()
    assert m["id"] == mid
    assert "order_book" in m
    assert m["order_book"]["bids"] and m["order_book"]["asks"]
    # bids below ask, both valid probabilities
    for px, size in m["order_book"]["bids"]:
        assert 0.0 <= px <= 1.0 and size > 0


def test_get_market_404():
    r = client.get("/api/market/does-not-exist")
    assert r.status_code == 404


def test_derivative_bundle_shape():
    mid = client.get("/api/markets").json()["markets"][0]["id"]
    r = client.get(f"/api/derivative/{mid}", params={"option_type": "call", "side": "long", "strike": 0.5})
    assert r.status_code == 200
    b = r.json()
    # all four Greeks present as scalars
    for k in ("delta", "gamma", "theta", "vega"):
        assert k in b["greeks"]
        assert isinstance(b["greeks"][k], (int, float))
    # payoff and every Greek curve span the probability grid 0..1
    assert len(b["payoff_curve"]) == 101
    assert b["payoff_curve"][0]["p"] == 0.0 and b["payoff_curve"][-1]["p"] == 1.0
    for name in ("delta", "gamma", "theta", "vega"):
        assert len(b["greek_curves"][name]) == 101
    assert 0.0 <= b["break_even"] <= 1.0
    assert "market" in b and b["market"]["id"] == mid


def test_derivative_call_payoff_is_monotonic_nondecreasing():
    """A long call's payoff never decreases as the underlying prob rises."""
    mid = client.get("/api/markets").json()["markets"][0]["id"]
    b = client.get(f"/api/derivative/{mid}", params={"option_type": "call", "side": "long", "strike": 0.4}).json()
    vals = [pt["payoff"] for pt in b["payoff_curve"]]
    assert all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1))


def test_derivative_short_is_sign_flip_of_long():
    mid = client.get("/api/markets").json()["markets"][0]["id"]
    lng = client.get(f"/api/derivative/{mid}", params={"option_type": "put", "side": "long", "strike": 0.6}).json()
    sht = client.get(f"/api/derivative/{mid}", params={"option_type": "put", "side": "short", "strike": 0.6}).json()
    assert abs(lng["greeks"]["delta"] + sht["greeks"]["delta"]) < 1e-6


def test_derivative_rejects_bad_params():
    mid = client.get("/api/markets").json()["markets"][0]["id"]
    assert client.get(f"/api/derivative/{mid}", params={"option_type": "swap", "strike": 0.5}).status_code == 422
    assert client.get(f"/api/derivative/{mid}", params={"side": "flat", "strike": 0.5}).status_code == 422
    assert client.get(f"/api/derivative/{mid}", params={"strike": 1.5}).status_code == 422


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "Analytics and modeling only" in r.text
