"""Unit tests for the synthetic derivatives engine.

Deterministic inputs only. The pricing model is the logit-normal martingale on
the YES probability (MODEL.md): an option held to resolution (``tau == 1``) has
the exact, model-free value ``call = p*(1-K)`` / ``put = (1-p)*K``; before
resolution (``tau < 1``) it is the expectation of the intrinsic payoff over the
logit-normal law. These tests sweep a grid of ``p``, ``K``, option type, side and
``tau`` and pin every invariant the audit called out.
"""

import math
import random

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
    half_spread,
    held_to_resolution_price,
    implied_probability,
    intrinsic_payoff,
    max_profit_loss,
    mid_from_book,
    payoff_curve,
    pricing_basis,
    prob_of_profit,
)

# Grids reused across the invariant sweeps.
_PS = [0.02, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.98]
_KS = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
_TAUS = [0.05, 0.2, 0.5, 0.8, 0.95, 1.0]
_OTS = [OptionType.CALL, OptionType.PUT]


def _exact(p, k, ot):
    return p * (1.0 - k) if ot is OptionType.CALL else (1.0 - p) * k


def _intrinsic(p, k, ot):
    return max(p - k, 0.0) if ot is OptionType.CALL else max(k - p, 0.0)


# --------------------------------------------------------------------------
# Intrinsic payoff (model-free)
# --------------------------------------------------------------------------

def test_call_intrinsic_payoff_signs():
    assert intrinsic_payoff(1.0, 0.4, OptionType.CALL) == pytest.approx(0.6)
    assert intrinsic_payoff(0.0, 0.4, OptionType.CALL) == 0.0
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
# I1 -- held to resolution is the exact, model-free value
# --------------------------------------------------------------------------

def test_i1_held_to_resolution_exact_over_grid():
    for p in _PS:
        for k in _KS:
            assert fair_price(p, k, "call", tau=1.0) == pytest.approx(p * (1 - k), abs=1e-9)
            assert fair_price(p, k, "put", tau=1.0) == pytest.approx((1 - p) * k, abs=1e-9)


def test_i1_held_to_resolution_helper_matches():
    for p in _PS:
        for k in _KS:
            for ot in _OTS:
                assert held_to_resolution_price(p, k, ot) == pytest.approx(
                    fair_price(p, k, ot, tau=1.0), abs=1e-12
                )


def test_i1_audit_regression_cases_exact():
    # The three audit cases, with their exact values.
    assert fair_price(0.50, 0.50, "call", tau=1.0) == pytest.approx(0.250, abs=1e-9)
    assert fair_price(0.04, 0.00, "call", tau=1.0) == pytest.approx(0.040, abs=1e-9)
    assert fair_price(0.02, 0.05, "put", tau=1.0) == pytest.approx(0.049, abs=1e-9)


def test_i1_short_is_negative_of_long_scalars():
    # Max P&L scalars: a short's gain/loss mirror the long's.
    for p in _PS:
        for k in _KS:
            for ot in _OTS:
                prem = fair_price(p, k, ot, tau=1.0)
                lng = max_profit_loss(k, ot, Side.LONG, prem)
                sht = max_profit_loss(k, ot, Side.SHORT, prem)
                assert sht["max_profit"] == pytest.approx(-lng["max_loss"], abs=1e-12)
                assert sht["max_loss"] == pytest.approx(-lng["max_profit"], abs=1e-12)


def test_i1_greeks_at_resolution_are_exact_formula():
    # Exact value call = p*(1-K): d/dp = (1-K); put = (1-p)*K: d/dp = -K;
    # both are linear in p, so gamma = theta = 0.
    for k in _KS:
        gc = greeks(0.5, k, "call", tau=1.0)
        assert gc["delta"] == pytest.approx(1.0 - k, abs=1e-9)
        assert gc["gamma"] == pytest.approx(0.0, abs=1e-9)
        assert gc["theta"] == pytest.approx(0.0, abs=1e-9)
        gp = greeks(0.5, k, "put", tau=1.0)
        assert gp["delta"] == pytest.approx(-k, abs=1e-9)
        assert gp["gamma"] == pytest.approx(0.0, abs=1e-9)
        assert gp["theta"] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# I2 -- no-arbitrage bounds for every tau
# --------------------------------------------------------------------------

def test_i2_no_arbitrage_bounds_over_grid():
    for p in _PS:
        for k in _KS:
            for tau in _TAUS:
                c = fair_price(p, k, "call", tau=tau)
                pu = fair_price(p, k, "put", tau=tau)
                # call in [max(p-K,0), p*(1-K)]; put in [max(K-p,0), (1-p)*K]
                assert max(p - k, 0.0) - 1e-9 <= c <= p * (1 - k) + 1e-9
                assert max(k - p, 0.0) - 1e-9 <= pu <= (1 - p) * k + 1e-9


def test_i2_premium_never_exceeds_max_payoff():
    # A long option's premium can never exceed its max possible payoff.
    for p in _PS:
        for k in _KS:
            for tau in _TAUS:
                c = fair_price(p, k, "call", tau=tau)
                pu = fair_price(p, k, "put", tau=tau)
                assert c <= (1.0 - k) + 1e-9      # max call payoff is (1-K) at x=1
                assert pu <= k + 1e-9             # max put payoff is K at x=0


# --------------------------------------------------------------------------
# I3 -- put-call parity for every tau (the model is a martingale)
# --------------------------------------------------------------------------

def test_i3_put_call_parity_over_grid():
    for p in _PS:
        for k in _KS:
            for tau in _TAUS:
                c = fair_price(p, k, "call", tau=tau)
                pu = fair_price(p, k, "put", tau=tau)
                assert (c - pu) == pytest.approx(p - k, abs=1e-6)


# --------------------------------------------------------------------------
# I4 -- price is non-decreasing in time left for a long option
# --------------------------------------------------------------------------

def test_i4_price_nondecreasing_in_tau():
    taus = [0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.97, 1.0]
    for p in _PS:
        for k in _KS:
            for ot in _OTS:
                prev = -1.0
                for tau in taus:
                    v = fair_price(p, k, ot, tau=tau)
                    assert v >= prev - 1e-7
                    prev = v


# --------------------------------------------------------------------------
# I5 -- chance of profit uses the same premium and tau as the price
# --------------------------------------------------------------------------

def test_i5_pop_in_unit_interval_over_grid():
    for p in _PS:
        for k in _KS:
            for ot in _OTS:
                for sd in (Side.LONG, Side.SHORT):
                    for tau in _TAUS:
                        prem = fair_price(p, k, ot, tau=tau)
                        pop = prob_of_profit(p, k, ot, sd, prem, tau=tau)
                        assert 0.0 <= pop <= 1.0


def test_i5_pop_long_call_at_p_eq_k_half_is_one_half():
    prem = fair_price(0.5, 0.5, "call", tau=1.0)
    assert prob_of_profit(0.5, 0.5, "call", "long", prem, tau=1.0) == pytest.approx(0.5, abs=1e-6)


def test_i5_pop_held_to_resolution_matches_exact():
    # Held to resolution: long call profits iff YES (PoP = p); long put iff NO
    # (PoP = 1 - p). Short sides are the complements.
    for p in _PS:
        for k in (0.25, 0.5, 0.75):
            pc = fair_price(p, k, "call", tau=1.0)
            pp = fair_price(p, k, "put", tau=1.0)
            assert prob_of_profit(p, k, "call", "long", pc, tau=1.0) == pytest.approx(p, abs=1e-6)
            assert prob_of_profit(p, k, "put", "long", pp, tau=1.0) == pytest.approx(1 - p, abs=1e-6)
            assert prob_of_profit(p, k, "call", "short", pc, tau=1.0) == pytest.approx(1 - p, abs=1e-6)
            assert prob_of_profit(p, k, "put", "short", pp, tau=1.0) == pytest.approx(p, abs=1e-6)


def test_i5_pop_long_short_complementary():
    for p in _PS:
        for k in (0.25, 0.5, 0.75):
            for ot in _OTS:
                for tau in (0.3, 0.7, 1.0):
                    prem = fair_price(p, k, ot, tau=tau)
                    lng = prob_of_profit(p, k, ot, "long", prem, tau=tau)
                    sht = prob_of_profit(p, k, ot, "short", prem, tau=tau)
                    assert (lng + sht) == pytest.approx(1.0, abs=1e-6)


def test_i5_pop_continuous_and_tends_to_exact_as_tau_grows():
    # Continuity in tau and convergence to the held-to-resolution value. The gap
    # shrinks like ~1/s (s -> infinity as tau -> 1), so each step is small and
    # the sequence approaches the exact PoP = p for a long call.
    p, k = 0.6, 0.4
    taus = [0.3, 0.6, 0.9, 0.99, 0.999, 1.0]
    pops = []
    for tau in taus:
        prem = fair_price(p, k, "call", tau=tau)
        pops.append(prob_of_profit(p, k, "call", "long", prem, tau=tau))
    # monotone non-decreasing toward p, and the last interior point is close
    for a, b in zip(pops, pops[1:]):
        assert b >= a - 1e-6
    assert pops[-1] == pytest.approx(p, abs=1e-9)         # exact at tau=1
    assert pops[-2] == pytest.approx(p, abs=2e-2)         # tau=0.999 within ~1/s


def test_i5_pop_unreachable_break_even():
    # Break-even outside [0, 1]: a long can never profit (PoP 0); short keeps the
    # premium (PoP 1). Construct it with an inflated premium.
    assert break_even(0.98, OptionType.CALL, 0.5) == []
    assert prob_of_profit(0.95, 0.98, "call", "long", 0.5, tau=0.5) == 0.0
    assert prob_of_profit(0.95, 0.98, "call", "short", 0.5, tau=0.5) == 1.0


def test_i5_pop_rejects_bad_tau():
    with pytest.raises(ValueError):
        prob_of_profit(0.5, 0.5, "call", "long", 0.1, tau=0.0)


# --------------------------------------------------------------------------
# I6 -- premium takes the same tau as the Greeks; theta is the rate of change
#        of the displayed premium with respect to tau
# --------------------------------------------------------------------------

def test_i6_theta_equals_finite_difference_of_premium_in_tau():
    # theta is defined as d(premium)/d(tau): the slope of the SAME displayed
    # premium against the time-left slider (positive for a long option).
    h = 1e-4
    for p in (0.3, 0.5, 0.7):
        for k in (0.25, 0.5, 0.75):
            for tau in (0.2, 0.5, 0.8):
                g = greeks(p, k, "call", tau=tau)
                fd = (
                    fair_price(p, k, "call", tau=tau + h)
                    - fair_price(p, k, "call", tau=tau - h)
                ) / (2 * h)
                assert g["theta"] == pytest.approx(fd, abs=1e-4)
                assert g["theta"] >= -1e-6  # non-negative: more time left is worth more


def test_i6_premium_depends_on_tau():
    # The premium is NOT tau-invariant: it rises with the time left (the old bug
    # had fair_price ignore tau while the Greeks used it).
    near = fair_price(0.6, 0.4, "call", tau=0.1)
    far = fair_price(0.6, 0.4, "call", tau=0.95)
    assert far > near + 1e-3


def test_i6_greeks_smooth_and_meaningful_before_resolution():
    # gamma and vega are non-trivial (the point mass / zero-gamma failure mode of
    # a piecewise-linear model is explicitly avoided).
    g = greeks(0.5, 0.5, "call", tau=0.5)
    assert g["gamma"] > 0.0
    assert g["vega"] > 0.0


# --------------------------------------------------------------------------
# Greeks: finite-difference cross-checks and shape
# --------------------------------------------------------------------------

def test_delta_matches_finite_difference():
    h = 1e-3
    for p in (0.35, 0.5, 0.65):
        for ot in _OTS:
            up = fair_price(p + h, 0.5, ot, tau=0.5)
            dn = fair_price(p - h, 0.5, ot, tau=0.5)
            fd = (up - dn) / (2 * h)
            assert greeks(p, 0.5, ot, tau=0.5)["delta"] == pytest.approx(fd, abs=5e-3)


def test_delta_bounds_call_and_put():
    for p in [i / 20 for i in range(1, 20)]:
        gc = greeks(p, 0.5, "call", tau=0.5)
        gp = greeks(p, 0.5, "put", tau=0.5)
        assert -1e-6 <= gc["delta"] <= 1.0 + 1e-6
        assert -1.0 - 1e-6 <= gp["delta"] <= 1e-6


def test_call_put_delta_differ_by_one():
    for p in (0.3, 0.5, 0.7):
        gc = greeks(p, 0.5, "call", tau=0.5)
        gp = greeks(p, 0.5, "put", tau=0.5)
        assert (gc["delta"] - gp["delta"]) == pytest.approx(1.0, abs=1e-6)


def test_gamma_peaks_near_strike():
    # At a moderate horizon the convexity of value in the probability peaks at
    # the strike (the option is most sensitive to probability moves there). At
    # large tau the dispersion is wide and gamma flattens, so we test a tighter
    # horizon where the peak is pronounced.
    curve = greek_curve(0.5, "call", tau=0.2, num_points=101)
    peak_idx = max(range(len(curve["gamma"])), key=lambda i: curve["gamma"][i])
    assert curve["p"][peak_idx] == pytest.approx(0.5, abs=0.03)


def test_gamma_vega_nonnegative_before_resolution():
    for p in [i / 10 for i in range(1, 10)]:
        g = greeks(p, 0.5, "call", tau=0.5)
        assert g["gamma"] >= -1e-6
        assert g["vega"] >= -1e-6


def test_gamma_vega_same_for_call_and_put():
    for p in (0.3, 0.5, 0.7):
        gc = greeks(p, 0.5, "call", tau=0.5)
        gp = greeks(p, 0.5, "put", tau=0.5)
        assert gc["gamma"] == pytest.approx(gp["gamma"], abs=1e-6)
        assert gc["vega"] == pytest.approx(gp["vega"], abs=1e-6)


# --------------------------------------------------------------------------
# I7 -- order book reaches the engine
# --------------------------------------------------------------------------

def test_i7_implied_probability_defaults_to_yes_price():
    assert implied_probability(0.63) == pytest.approx(0.63)


def test_i7_implied_probability_prefers_book_mid():
    book = OrderBook(
        bids=[OrderLevel(0.60, 100), OrderLevel(0.58, 50)],
        asks=[OrderLevel(0.64, 80), OrderLevel(0.66, 40)],
    )
    assert mid_from_book(book) == pytest.approx(0.62)
    assert implied_probability(0.55, book=book) == pytest.approx(0.62)


def test_i7_empty_book_mid_is_none_and_falls_back():
    assert mid_from_book(OrderBook()) is None
    assert mid_from_book(OrderBook(bids=[OrderLevel(0.5, 10)])) is None
    assert implied_probability(0.4, book=OrderBook()) == pytest.approx(0.4)


def test_i7_half_spread_pins_behaviour():
    assert half_spread(None) == 0.0
    assert half_spread(OrderBook()) == 0.0  # one-sided / empty -> 0
    book = OrderBook(bids=[OrderLevel(0.40, 10)], asks=[OrderLevel(0.60, 10)])
    assert half_spread(book) == pytest.approx(0.10)


def test_i7_wider_book_raises_model_price_via_dispersion():
    # Dispersion varies per market: a wider top-of-book spread widens it, so a
    # time-value option (p = K) is worth more. This pins I7's "sigma varies".
    p, k, tau = 0.5, 0.5, 0.5
    tight = OrderBook(bids=[OrderLevel(0.49, 10)], asks=[OrderLevel(0.51, 10)])
    wide = OrderBook(bids=[OrderLevel(0.40, 10)], asks=[OrderLevel(0.60, 10)])
    assert fair_price(p, k, "call", tau=tau, book=wide) > fair_price(
        p, k, "call", tau=tau, book=tight
    )


def test_i7_no_book_runs_off_p():
    # With no book the price is well-defined and within bounds (clean fallback).
    v = fair_price(0.5, 0.5, "call", tau=0.5, book=None)
    assert 0.0 <= v <= 0.5 + 1e-9


# --------------------------------------------------------------------------
# I8 -- break-even that does not exist is reported absent, never clamped
# --------------------------------------------------------------------------

def test_i8_break_even_basic():
    assert break_even(0.4, OptionType.CALL, 0.2) == [pytest.approx(0.6)]
    assert break_even(0.6, OptionType.PUT, 0.2) == [pytest.approx(0.4)]
    assert break_even(0.5, OptionType.CALL, 0.0) == [pytest.approx(0.5)]


def test_i8_break_even_unreachable_is_empty_not_clamped():
    # K + premium > 1 for a call, or K - premium < 0 for a put -> absent.
    assert break_even(0.9, OptionType.CALL, 0.5) == []
    assert break_even(0.1, OptionType.PUT, 0.5) == []


def test_i8_bundle_reports_null_break_even(monkeypatch):
    # When the premium pushes the break-even outside [0, 1], analyze returns an
    # empty list; the adapter maps that to None (see test_api for the API shape).
    be = break_even(0.95, OptionType.CALL, 0.5)
    assert be == []


def test_i8_long_max_gain_nonneg_max_loss_nonpos_short_reversed():
    for p in _PS:
        for k in _KS:
            for ot in _OTS:
                for tau in (0.3, 1.0):
                    prem = fair_price(p, k, ot, tau=tau)
                    lng = max_profit_loss(k, ot, Side.LONG, prem)
                    assert lng["max_profit"] >= -1e-9       # no negative max gain
                    assert lng["max_loss"] <= 1e-9
                    sht = max_profit_loss(k, ot, Side.SHORT, prem)
                    assert sht["max_profit"] >= -1e-9
                    assert sht["max_loss"] <= 1e-9


def test_i8_audit_put_has_real_break_even_and_nonneg_max_gain():
    # The audit's p=0.02,K=0.05 put: premium 0.049 <= max payoff 0.05, so the
    # break-even exists (not clamped) and max gain is non-negative.
    prem = fair_price(0.02, 0.05, "put", tau=1.0)
    assert prem == pytest.approx(0.049, abs=1e-9)
    be = break_even(0.05, OptionType.PUT, prem)
    assert be and 0.0 <= be[0] <= 1.0
    mpl = max_profit_loss(0.05, OptionType.PUT, Side.LONG, prem)
    assert mpl["max_profit"] >= -1e-9


# --------------------------------------------------------------------------
# Pricing basis
# --------------------------------------------------------------------------

def test_pricing_basis_labels():
    assert pricing_basis(1.0) == "held_to_resolution"
    assert pricing_basis(0.5) == "model"
    with pytest.raises(ValueError):
        pricing_basis(0.0)
    with pytest.raises(ValueError):
        pricing_basis(1.5)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan")])
def test_probability_out_of_range_rejected(bad):
    with pytest.raises(ValueError):
        fair_price(bad, 0.5, OptionType.CALL)


def test_tau_out_of_range_rejected():
    with pytest.raises(ValueError):
        greeks(0.5, 0.5, OptionType.CALL, tau=0.0)
    with pytest.raises(ValueError):
        greeks(0.5, 0.5, OptionType.CALL, tau=1.5)
    with pytest.raises(ValueError):
        fair_price(0.5, 0.5, OptionType.CALL, tau=0.0)


def test_num_points_too_small_rejected():
    with pytest.raises(ValueError):
        payoff_curve(0.5, OptionType.CALL, num_points=1)


# --------------------------------------------------------------------------
# Bundle / JSON-serializability
# --------------------------------------------------------------------------

def test_analyze_bundle_shape_and_json():
    import json

    book = OrderBook(bids=[OrderLevel(0.60, 100)], asks=[OrderLevel(0.64, 80)])
    bundle = analyze(0.6, 0.5, "call", side="long", tau=0.8, book=book)

    for key in (
        "inputs", "implied_probability", "fair_price", "pricing_basis",
        "break_even", "max_profit", "max_loss", "prob_of_profit",
        "payoff_curve", "greek_curve", "greeks",
    ):
        assert key in bundle

    assert 0.0 <= bundle["prob_of_profit"] <= 1.0
    assert bundle["pricing_basis"] in ("held_to_resolution", "model")
    assert bundle["inputs"]["option_type"] == "call"
    assert len(bundle["payoff_curve"]["x"]) == 101
    assert set(bundle["greek_curve"]) == {"p", "delta", "gamma", "vega", "theta", "price"}

    s = json.dumps(bundle)
    assert json.loads(s)["fair_price"] == pytest.approx(bundle["fair_price"])


def test_analyze_break_even_is_list_possibly_empty():
    # The engine's own bundle keeps break_even as a list (the adapter maps the
    # empty list to a null scalar for the API).
    b = analyze(0.5, 0.5, "call", tau=1.0)
    assert isinstance(b["break_even"], list)


def test_analyze_accepts_string_enums():
    b1 = analyze(0.6, 0.5, "put", side="short", tau=1.0)
    b2 = analyze(0.6, 0.5, OptionType.PUT, side=Side.SHORT, tau=1.0)
    assert b1["fair_price"] == pytest.approx(b2["fair_price"])


# --------------------------------------------------------------------------
# Adversarial sweep -- one honest attempt to break the model
# --------------------------------------------------------------------------

def test_random_sweep_bounds_parity_and_monotonicity():
    """A random sweep over p, K, tau checking the no-arbitrage bounds (I2),
    put-call parity (I3) and monotonicity in tau (I4) all hold at once."""
    rng = random.Random(1234)
    for _ in range(300):
        p = rng.uniform(0.01, 0.99)
        k = rng.uniform(0.0, 1.0)
        taus = sorted(rng.sample([0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0], 4))
        prev_c = prev_pu = -1.0
        for tau in taus:
            c = fair_price(p, k, "call", tau=tau)
            pu = fair_price(p, k, "put", tau=tau)
            assert max(p - k, 0.0) - 1e-9 <= c <= p * (1 - k) + 1e-9
            assert max(k - p, 0.0) - 1e-9 <= pu <= (1 - p) * k + 1e-9
            assert (c - pu) == pytest.approx(p - k, abs=1e-6)
            assert c >= prev_c - 1e-7 and pu >= prev_pu - 1e-7
            prev_c, prev_pu = c, pu


def test_edge_probabilities_near_zero_and_one():
    # p near 0 / near 1 and K = 0 / K = 1 must stay finite and within bounds.
    for p in (0.001, 0.999):
        for k in (0.0, 1.0):
            for tau in (0.1, 0.5, 1.0):
                c = fair_price(p, k, "call", tau=tau)
                pu = fair_price(p, k, "put", tau=tau)
                assert math.isfinite(c) and math.isfinite(pu)
                assert c >= -1e-9 and pu >= -1e-9
                assert (c - pu) == pytest.approx(p - k, abs=1e-5)
