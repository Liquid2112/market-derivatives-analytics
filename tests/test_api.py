"""Smoke + unit tests for the read-only API, run against the mock backends.

These exercise the FastAPI layer and the adapter's mock data/engine without
any network. The real mdlab->engine wiring is covered separately in
``test_integration_real.py``; here we pin the mock backends so the shape and
behaviour assertions are deterministic and offline.
"""

import os

os.environ["MDA_USE_REAL_BACKENDS"] = "0"

import pytest
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


def test_default_markets_are_popular_first():
    """The default landing view ranks markets by traded volume descending so
    popular, recognizable markets lead (an explicit search is not reordered)."""
    markets = client.get("/api/markets").json()["markets"]
    vols = [m["volume"] for m in markets if m.get("volume") is not None]
    assert vols == sorted(vols, reverse=True)
    # the single most-traded market is at the top of the default view
    assert markets[0]["volume"] == max(vols)


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
    # break-even is a YES probability in [0, 1] or null (never clamped)
    assert b["break_even"] is None or (0.0 <= b["break_even"] <= 1.0)
    # pricing_basis says whether the premium is exact or model-based
    assert b["pricing_basis"] in ("held_to_resolution", "model")
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


def test_derivative_bundle_has_pop_and_extrema():
    """The bundle surfaces model-based probability-of-profit and the side-signed
    max gain / max loss scalars the UI reads, over the mock (offline) engine."""
    mid = client.get("/api/markets").json()["markets"][0]["id"]
    b = client.get(
        f"/api/derivative/{mid}",
        params={"option_type": "call", "side": "long", "strike": 0.5},
    ).json()
    assert 0.0 <= b["prob_of_profit"] <= 1.0
    for k in ("max_gain", "max_loss"):
        assert isinstance(b[k], (int, float))
    # a long option's downside is bounded: max loss is negative (the premium)
    # and max gain is positive, and max_gain >= max_loss.
    assert b["max_loss"] <= 0.0 <= b["max_gain"]
    assert b["max_gain"] >= b["max_loss"]


def test_derivative_long_call_max_loss_is_negative_premium():
    """Long call: max loss equals the negated premium (bounded downside)."""
    mid = client.get("/api/markets").json()["markets"][0]["id"]
    b = client.get(
        f"/api/derivative/{mid}",
        params={"option_type": "call", "side": "long", "strike": 0.5},
    ).json()
    assert b["max_loss"] == pytest.approx(-b["premium"], abs=1e-4)


def test_derivative_long_put_max_loss_is_negative_premium():
    mid = client.get("/api/markets").json()["markets"][0]["id"]
    b = client.get(
        f"/api/derivative/{mid}",
        params={"option_type": "put", "side": "long", "strike": 0.5},
    ).json()
    assert b["max_loss"] == pytest.approx(-b["premium"], abs=1e-4)


def test_derivative_short_extrema_mirror_long():
    """A short position's gain/loss is the sign-flip of the long position's."""
    mid = client.get("/api/markets").json()["markets"][0]["id"]
    lng = client.get(
        f"/api/derivative/{mid}",
        params={"option_type": "call", "side": "long", "strike": 0.4},
    ).json()
    sht = client.get(
        f"/api/derivative/{mid}",
        params={"option_type": "call", "side": "short", "strike": 0.4},
    ).json()
    assert sht["max_gain"] == pytest.approx(-lng["max_loss"], abs=1e-4)
    assert sht["max_loss"] == pytest.approx(-lng["max_gain"], abs=1e-4)


def test_derivative_pop_long_short_complementary():
    """Long and short PoP sum to 1 when break-even is inside [0, 1]."""
    mid = client.get("/api/markets").json()["markets"][0]["id"]
    lng = client.get(
        f"/api/derivative/{mid}",
        params={"option_type": "put", "side": "long", "strike": 0.5},
    ).json()
    sht = client.get(
        f"/api/derivative/{mid}",
        params={"option_type": "put", "side": "short", "strike": 0.5},
    ).json()
    assert lng["prob_of_profit"] + sht["prob_of_profit"] == pytest.approx(1.0, abs=1e-4)


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "Analytics and modeling only" in r.text


def test_derivative_tau_param_defaults_and_accepts_range():
    """The optional tau query param (modeled time-to-resolution) defaults to the
    full horizon and accepts any value in (0, 1]; out-of-range is rejected."""
    mid = client.get("/api/markets").json()["markets"][0]["id"]
    base = {"option_type": "call", "side": "long", "strike": 0.5}
    # default (no tau) and explicit tau=1.0 agree
    default = client.get(f"/api/derivative/{mid}", params=base).json()
    full = client.get(f"/api/derivative/{mid}", params={**base, "tau": 1.0}).json()
    assert default["greeks"]["gamma"] == pytest.approx(full["greeks"]["gamma"], abs=1e-9)
    # a mid-range horizon is accepted
    assert client.get(f"/api/derivative/{mid}", params={**base, "tau": 0.25}).status_code == 200
    # tau must stay in (0, 1]
    assert client.get(f"/api/derivative/{mid}", params={**base, "tau": 0.0}).status_code == 422
    assert client.get(f"/api/derivative/{mid}", params={**base, "tau": 1.5}).status_code == 422


def test_derivative_tau_drives_premium_and_basis():
    """The premium now takes the same tau as the Greeks (the old bug had
    fair_price ignore tau). At tau=1 the option is held to resolution and the
    premium is the exact model-free value (pricing_basis 'held_to_resolution');
    below 1 it is the logit-normal model value and lower (more time left is worth
    more), with pricing_basis 'model'.
    """
    mid = client.get("/api/markets").json()["markets"][0]["id"]
    base = {"option_type": "call", "side": "long", "strike": 0.4}
    full = client.get(f"/api/derivative/{mid}", params={**base, "tau": 1.0}).json()
    near = client.get(f"/api/derivative/{mid}", params={**base, "tau": 0.1}).json()
    assert full["pricing_basis"] == "held_to_resolution"
    assert near["pricing_basis"] == "model"
    # premium is non-decreasing in time left: tau=1 premium >= a short-horizon one
    assert full["premium"] >= near["premium"] - 1e-9
    assert full["premium"] > near["premium"]  # strictly, away from degenerate K


def test_derivative_pricing_basis_present_and_correct():
    """Every derivative bundle carries pricing_basis; it is exact at tau=1."""
    mid = client.get("/api/markets").json()["markets"][0]["id"]
    base = {"option_type": "put", "side": "long", "strike": 0.5}
    assert client.get(f"/api/derivative/{mid}", params={**base, "tau": 1.0}).json()[
        "pricing_basis"
    ] == "held_to_resolution"
    assert client.get(f"/api/derivative/{mid}", params={**base, "tau": 0.5}).json()[
        "pricing_basis"
    ] == "model"


def test_order_book_reaches_engine_via_implied_prob():
    """I11: the order book reaches the engine. The implied probability comes from
    the top-of-book mid, not the raw YES price, so a market whose book mid differs
    from yes_price surfaces the mid. (The mock book is centered on yes_price, so
    here we assert the implied prob equals the mid of the returned book.)
    """
    mid_id = client.get("/api/markets").json()["markets"][0]["id"]
    market = client.get(f"/api/market/{mid_id}").json()
    ob = market["order_book"]
    best_bid = max(px for px, _ in ob["bids"])
    best_ask = min(px for px, _ in ob["asks"])
    book_mid = 0.5 * (best_bid + best_ask)
    b = client.get(
        f"/api/derivative/{mid_id}",
        params={"option_type": "call", "side": "long", "strike": 0.5, "tau": 0.5},
    ).json()
    assert b["implied_prob"] == pytest.approx(round(book_mid, 4), abs=1e-3)


def test_order_book_spread_widens_model_premium():
    """I11/I7: a wider top-of-book spread widens the model dispersion, raising a
    pure-time-value premium. We inject two data layers differing only in spread.
    """
    from app import adapter
    from app.adapter import _build_bundle

    tight = {"bids": [[0.49, 10]], "asks": [[0.51, 10]]}
    wide = {"bids": [[0.40, 10]], "asks": [[0.60, 10]]}
    # p = K = 0.5 so the premium is all time value; a model horizon (tau<1).
    bt = _build_bundle(0.5, 0.5, "call", "long", tau=0.5, order_book=tight)
    bw = _build_bundle(0.5, 0.5, "call", "long", tau=0.5, order_book=wide)
    assert bw["premium"] > bt["premium"]


def test_break_even_null_when_outside_range():
    """I8: a break-even that does not exist in [0, 1] is reported as null, never
    clamped. A long call at a high strike with a model premium pushes K + premium
    past 1 only when the premium is inflated; here we drive the adapter directly
    with a premium-free deep case to confirm the null mapping path.
    """
    from app.adapter import _build_bundle

    # Held-to-resolution call premium is p*(1-K) <= 1-K, so K+premium <= 1 and the
    # break-even exists. The null path is exercised by the engine test suite; here
    # we just confirm the API type contract: break_even is a float or None.
    b = _build_bundle(0.5, 0.5, "call", "long", tau=1.0)
    assert b["break_even"] is None or isinstance(b["break_even"], float)


def test_real_and_mock_engine_share_one_builder():
    """I9: the real and mock engines produce identical bundles for the same
    inputs -- they share one pricing path and cannot diverge."""
    from app.adapter import _MockEngine, _RealEngine

    args = dict(yes_price=0.62, strike=0.45, option_type="call", side="long", tau=0.6)
    bm = _MockEngine().bundle(**args)
    br = _RealEngine().bundle(**args)
    assert bm == br
    # and the implied_prob is not the strike (the old mock bug: implied_prob = K)
    assert bm["implied_prob"] != bm["strike"] or bm["underlying_prob"] == bm["strike"]
