# Derivatives engine

Pure Python module that models **synthetic** options on prediction-market
probabilities. No network, no file I/O, no global state. Given a market's YES
price `p` (a probability in `[0, 1]`) and an optional order book, it returns
payoff curves, break-even points, max profit/loss, a fair price, a chance of
profit, an implied probability, and option Greeks (delta, gamma, theta, vega) as
plain, JSON-serializable data ready for an API or UI.

This is analytics, not a trading claim. See [MODEL.md](MODEL.md) for the model
behind the pricing, the Greeks, and the honest limits.

## The model in one paragraph

An option **held to resolution** is a scaled binary with an exact, model-free
price (`call = p*(1-K)`, `put = (1-p)*K`). For an expiry **before** resolution the
YES probability path is a logit-normal martingale (bounded in `[0, 1]`, mean `p`),
and the option price is the expectation of its intrinsic payoff. The held-to-
resolution value is the `tau -> 1` limit of that model, so the two are one
consistent model, not two rules. Prices obey the no-arbitrage bounds and put-call
parity for every `tau`.

## Install / run

Stdlib only at runtime (Python 3.10+). Tests use `pytest`.

```bash
pip install -e ".[test]"
pytest -q
```

## Usage

```python
from derivatives_engine import analyze, OrderBook, OrderLevel

book = OrderBook(
    bids=[OrderLevel(0.60, 100), OrderLevel(0.58, 50)],
    asks=[OrderLevel(0.64, 80)],
)

bundle = analyze(
    p=0.62,            # current YES price / implied probability
    strike=0.50,       # strike as a probability level
    option_type="call",
    side="long",
    tau=1.0,           # time left in (0, 1]; 1.0 = held to resolution (exact)
    book=book,         # optional
)
# bundle is plain JSON-serializable data: fair_price, pricing_basis,
# implied_probability, break_even (list, possibly empty), max_profit, max_loss,
# prob_of_profit, payoff_curve {x, payoff}, greek_curve {p, delta, gamma, vega,
# theta, price}, and point greeks.
```

### `tau` — the time-left slider

`tau in (0, 1]` is the fraction of time remaining to resolution:

* `tau = 1` — the option is **held to resolution**; the premium is the exact
  model-free value and `pricing_basis` is `"held_to_resolution"`.
* `tau < 1` — the expiry is before resolution; the premium is the logit-normal
  model value (lower, since less time means less chance of a favourable move) and
  `pricing_basis` is `"model"`.

The premium, the Greeks and the chance of profit all take the same `tau`, so
`theta` is the actual rate of change of the displayed premium with the slider.

## Public functions the UI should call

| Function | Returns |
|----------|---------|
| `analyze(p, strike, option_type, side, tau, book, num_points)` | the full bundle, as one dict |
| `fair_price(p, strike, option_type, tau, book)` | model fair value (exact at `tau = 1`) |
| `held_to_resolution_price(p, strike, option_type)` | the exact model-free value `p*(1-K)` / `(1-p)*K` |
| `pricing_basis(tau)` | `"held_to_resolution"` or `"model"` |
| `prob_of_profit(p, strike, option_type, side, premium, tau, book)` | chance of profit, same `tau`/premium as the price |
| `payoff_curve(strike, option_type, side, premium, num_points)` | `{x, payoff}` arrays over the 0..1 outcome range |
| `break_even(strike, option_type, premium)` | list of break-even value(s), **empty if none exist in `[0, 1]`** |
| `max_profit_loss(strike, option_type, side, premium)` | `{max_profit, max_loss}` |
| `implied_probability(p, book)` | market implied probability of YES (book mid when available) |
| `greeks(p, strike, option_type, tau, book)` | `{delta, gamma, vega, theta, ...}` at one probability |
| `greek_curve(strike, option_type, tau, num_points, book)` | `{p, delta, gamma, vega, theta, price}` arrays for charting |

> A `sigma` keyword is still accepted by the pricing functions for backward
> compatibility and is ignored — the dispersion is now driven by `tau` and the
> order-book spread, not a free `sigma`.

### Greeks for the UI

`greek_curve(...)` is the function to chart: it returns the probability grid `p`
and parallel arrays for `delta`, `gamma`, `vega`, `theta`, and `price`. `greeks(...)`
gives the point values at the current probability. The meaning of each Greek in
this binary-market context is documented in
[MODEL.md](MODEL.md#the-greeks-in-this-context).

## Decoupling from the data layer

The engine takes simple typed inputs (a `p` in `[0, 1]` and an `OrderBook` of
`(price, size)` levels). The `app/adapter.py` seam maps the data layer's
normalized market and order book onto these inputs; no engine change is needed.
