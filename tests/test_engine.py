"""Unit tests for the synthetic derivatives engine.

Deterministic inputs only. Covers payoff signs, break-even correctness, max
profit/loss, pricing bounds, implied probability, Greek relationships (delta
bounds, gamma peaks at the strike, put-call parity) and a finite-difference
cross-check of the analytic Greeks against the pricing model.
"""

import math

import pytest

from derivatives_engine import (
    OptionType,
    OrderBook,
    OrderLevel,
    Side,
    analyze,
    break_even,
    fair_price,
    greek_curve,
    greeks,
    implied_probability,
    intrinsic_payoff,
    max_profit_loss,
    mid_from_book,
    payoff_curve,
)


# --------------------------------------------------------------------------
# Intrinsic payoff
# --------------------------------------------------------------------------

def test_call_intrinsic_payoff_signs():
    # YES resolves (x=1): call with strike 0.4 pays 0.6; NO resolves (x=0): pays 0
    assert intrinsic_payoff(1.0, 0.4, OptionType.CALL) == pytest.approx(0.6)
    assert intrinsic_payoff(0.0, 0.4, OptionType.CALL) == 0.0
    # never negative
    assert intrinsic_payoff(0.3, 0.4, OptionType.CALL) == 0.0


def test_put_intrinsic_payoff_signs():
    assert intrinsic_payoff(0.0, 0.4, OptionType.PUT) == pytest.approx(0.4)
    assert intrinsic_payoff(1.0, 0.4, OptionType.PUT) == 0.0
    assert intrinsic_payoff(0.5, 0.4, OptionType.PUT) == 0.0


def test_payoff_is_never_negative_for_gross_long():
    curve = payoff_curve(0.5, OptionType.CALL)
    assert all(v >= 0.0 for v in curve["payoff"])
    assert curve["x"][0] == 0.0 and curve["x"][-1] == 1.0
    assert len(curve["x"]) == len(curve["payoff"]) == 101


def test_payoff_net_of_premium_long_vs_short_mirror():
    long_c = payoff_curve(0.5, OptionType.CALL, side="long", premium=0.1)
    short_c = payoff_curve(0.5, OptionType.CALL, side="short", premium=0.1)
    for a, b in zip(long_c["payoff"], short_c["payoff"]):
        assert a == pytest.approx(-b)


# --------------------------------------------------------------------------
# Break-even
# --------------------------------------------------------------------------

def test_break_even_call():
    # call strike 0.4, premium 0.2 -> break-even at 0.6
    assert break_even(0.4, OptionType.CALL, 0.2) == [pytest.approx(0.6)]


def test_break_even_put():
    # put strike 0.6, premium 0.2 -> break-even at 0.4
    assert break_even(0.6, OptionType.PUT, 0.2) == [pytest.approx(0.4)]


def test_break_even_zero_premium_is_strike():
    assert break_even(0.5, OptionType.CALL, 0.0) == [pytest.approx(0.5)]


def test_break_even_unreachable_is_empty():
    # call strike 0.9 + premium 0.5 = 1.4 > 1 -> unreachable
    assert break_even(0.9, OptionType.CALL, 0.5) == []


def test_break_even_crosses_payoff_zero():
    premium = 0.15
    be = break_even(0.4, OptionType.CALL, premium)[0]
    net_at_be = intrinsic_payoff(be, 0.4, OptionType.CALL) - premium
    assert net_at_be == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# Max profit / loss
# --------------------------------------------------------------------------

def test_max_profit_loss_long_call():
    mpl = max_profit_loss(0.4, OptionType.CALL, Side.LONG, 0.2)
    # best at x=1: (1-0.4)-0.2 = 0.4 ; worst at x=0: -0.2
    assert mpl["max_profit"] == pytest.approx(0.4)
    assert mpl["max_loss"] == pytest.approx(-0.2)


def test_max_profit_loss_short_call_is_mirror():
    long_mpl = max_profit_loss(0.4, OptionType.CALL, Side.LONG, 0.2)
    short_mpl = max_profit_loss(0.4, OptionType.CALL, Side.SHORT, 0.2)
    assert short_mpl["max_profit"] == pytest.approx(-long_mpl["max_loss"])
    assert short_mpl["max_loss"] == pytest.approx(-long_mpl["max_profit"])


# --------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------

def test_fair_price_within_payoff_bounds():
    # option value is bounded by [0, 1] for strikes/probabilities in [0,1]
    for p in (0.1, 0.5, 0.9):
        for k in (0.2, 0.5, 0.8):
            c = fair_price(p, k, OptionType.CALL)
            put = fair_price(p, k, OptionType.PUT)
            assert 0.0 <= c <= 1.0
            assert 0.0 <= put <= 1.0


def test_put_call_parity():
    # Bachelier parity: C - P = p - K
    for p in (0.2, 0.5, 0.75):
        for k in (0.3, 0.5, 0.7):
            c = fair_price(p, k, OptionType.CALL, sigma=0.2)
            put = fair_price(p, k, OptionType.PUT, sigma=0.2)
            assert (c - put) == pytest.approx(p - k, abs=1e-12)


def test_price_monotonic_in_probability_for_call():
    prev = -1.0
    for p in [i / 20 for i in range(21)]:
        c = fair_price(p, 0.5, OptionType.CALL, sigma=0.2)
        assert c >= prev - 1e-12
        prev = c


def test_lower_sigma_closer_to_intrinsic():
    p, k = 0.7, 0.4
    intrinsic = max(p - k, 0.0)
    near = fair_price(p, k, OptionType.CALL, sigma=0.01)
    far = fair_price(p, k, OptionType.CALL, sigma=0.3)
    assert abs(near - intrinsic) < abs(far - intrinsic)


# --------------------------------------------------------------------------
# Implied probability & order book
# --------------------------------------------------------------------------

def test_implied_probability_defaults_to_yes_price():
    assert implied_probability(0.63) == pytest.approx(0.63)


def test_implied_probability_prefers_book_mid():
    book = OrderBook(
        bids=[OrderLevel(0.60, 100), OrderLevel(0.58, 50)],
        asks=[OrderLevel(0.64, 80), OrderLevel(0.66, 40)],
    )
    assert mid_from_book(book) == pytest.approx(0.62)
    assert implied_probability(0.55, book=book) == pytest.approx(0.62)


def test_empty_book_mid_is_none():
    assert mid_from_book(OrderBook()) is None
    assert mid_from_book(OrderBook(bids=[OrderLevel(0.5, 10)])) is None


def test_zero_size_levels_ignored():
    book = OrderBook(
        bids=[OrderLevel(0.70, 0.0), OrderLevel(0.60, 5.0)],
        asks=[OrderLevel(0.65, 0.0), OrderLevel(0.68, 5.0)],
    )
    assert book.best_bid() == pytest.approx(0.60)
    assert book.best_ask() == pytest.approx(0.68)


def test_wider_book_increases_price_via_sigma():
    p, k = 0.5, 0.5
    tight = OrderBook(bids=[OrderLevel(0.49, 10)], asks=[OrderLevel(0.51, 10)])
    wide = OrderBook(bids=[OrderLevel(0.40, 10)], asks=[OrderLevel(0.60, 10)])
    # at p=K the option is all time value, so a wider book -> higher price
    assert fair_price(p, k, OptionType.CALL, sigma=0.1, book=wide) > \
        fair_price(p, k, OptionType.CALL, sigma=0.1, book=tight)


# --------------------------------------------------------------------------
# Greeks
# --------------------------------------------------------------------------

def test_delta_bounds_call_and_put():
    for p in [i / 20 for i in range(21)]:
        gc = greeks(p, 0.5, OptionType.CALL, sigma=0.2)
        gp = greeks(p, 0.5, OptionType.PUT, sigma=0.2)
        assert 0.0 <= gc["delta"] <= 1.0
        assert -1.0 <= gp["delta"] <= 0.0


def test_call_put_delta_differ_by_one():
    for p in (0.2, 0.5, 0.8):
        gc = greeks(p, 0.5, OptionType.CALL, sigma=0.2)
        gp = greeks(p, 0.5, OptionType.PUT, sigma=0.2)
        assert (gc["delta"] - gp["delta"]) == pytest.approx(1.0, abs=1e-12)


def test_gamma_peaks_at_strike():
    strike = 0.5
    curve = greek_curve(strike, OptionType.CALL, sigma=0.2, num_points=101)
    peak_idx = max(range(len(curve["gamma"])), key=lambda i: curve["gamma"][i])
    # grid point nearest the strike should carry the max gamma
    assert curve["p"][peak_idx] == pytest.approx(strike, abs=0.01)


def test_gamma_vega_nonnegative():
    for p in [i / 10 for i in range(11)]:
        g = greeks(p, 0.5, OptionType.CALL, sigma=0.2)
        assert g["gamma"] >= 0.0
        assert g["vega"] >= 0.0


def test_gamma_same_for_call_and_put():
    for p in (0.3, 0.5, 0.7):
        gc = greeks(p, 0.5, OptionType.CALL, sigma=0.2)
        gp = greeks(p, 0.5, OptionType.PUT, sigma=0.2)
        assert gc["gamma"] == pytest.approx(gp["gamma"], abs=1e-12)
        assert gc["vega"] == pytest.approx(gp["vega"], abs=1e-12)


def test_theta_nonpositive_for_long_option():
    for p in (0.2, 0.5, 0.8):
        g = greeks(p, 0.5, OptionType.CALL, sigma=0.2, tau=0.5)
        assert g["theta"] <= 0.0


# --------------------------------------------------------------------------
# Finite-difference cross-checks of analytic Greeks
# --------------------------------------------------------------------------

def test_delta_matches_finite_difference():
    p, k, sigma = 0.55, 0.5, 0.2
    h = 1e-5
    for ot in (OptionType.CALL, OptionType.PUT):
        up = fair_price(p + h, k, ot, sigma=sigma)
        dn = fair_price(p - h, k, ot, sigma=sigma)
        fd = (up - dn) / (2 * h)
        analytic = greeks(p, k, ot, sigma=sigma)["delta"]
        assert analytic == pytest.approx(fd, abs=1e-5)


def test_gamma_matches_finite_difference():
    p, k, sigma = 0.52, 0.5, 0.2
    h = 1e-4
    up = fair_price(p + h, k, OptionType.CALL, sigma=sigma)
    mid = fair_price(p, k, OptionType.CALL, sigma=sigma)
    dn = fair_price(p - h, k, OptionType.CALL, sigma=sigma)
    fd = (up - 2 * mid + dn) / (h * h)
    analytic = greeks(p, k, OptionType.CALL, sigma=sigma)["gamma"]
    assert analytic == pytest.approx(fd, rel=1e-3, abs=1e-3)


def test_vega_matches_finite_difference():
    p, k, sigma = 0.55, 0.5, 0.2
    h = 1e-5
    up = fair_price(p, k, OptionType.CALL, sigma=sigma + h)
    dn = fair_price(p, k, OptionType.CALL, sigma=sigma - h)
    fd = (up - dn) / (2 * h)
    analytic = greeks(p, k, OptionType.CALL, sigma=sigma)["vega"]
    assert analytic == pytest.approx(fd, abs=1e-5)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan")])
def test_probability_out_of_range_rejected(bad):
    with pytest.raises(ValueError):
        fair_price(bad, 0.5, OptionType.CALL)


def test_negative_sigma_rejected():
    with pytest.raises(ValueError):
        fair_price(0.5, 0.5, OptionType.CALL, sigma=-0.1)


def test_tau_out_of_range_rejected():
    with pytest.raises(ValueError):
        greeks(0.5, 0.5, OptionType.CALL, tau=0.0)
    with pytest.raises(ValueError):
        greeks(0.5, 0.5, OptionType.CALL, tau=1.5)


def test_num_points_too_small_rejected():
    with pytest.raises(ValueError):
        payoff_curve(0.5, OptionType.CALL, num_points=1)


# --------------------------------------------------------------------------
# Bundle / JSON-serializability
# --------------------------------------------------------------------------

def test_analyze_bundle_shape_and_json():
    import json

    book = OrderBook(bids=[OrderLevel(0.60, 100)], asks=[OrderLevel(0.64, 80)])
    bundle = analyze(0.6, 0.5, "call", side="long", sigma=0.2, tau=0.8, book=book)

    # required keys
    for key in (
        "inputs", "implied_probability", "fair_price", "break_even",
        "max_profit", "max_loss", "payoff_curve", "greek_curve", "greeks",
    ):
        assert key in bundle

    assert bundle["inputs"]["option_type"] == "call"
    assert len(bundle["payoff_curve"]["x"]) == 101
    assert set(bundle["greek_curve"]) == {"p", "delta", "gamma", "vega", "theta", "price"}

    # must round-trip through JSON unchanged in structure
    s = json.dumps(bundle)
    assert json.loads(s)["fair_price"] == pytest.approx(bundle["fair_price"])


def test_analyze_accepts_string_enums():
    b1 = analyze(0.6, 0.5, "put", side="short")
    b2 = analyze(0.6, 0.5, OptionType.PUT, side=Side.SHORT)
    assert b1["fair_price"] == pytest.approx(b2["fair_price"])
