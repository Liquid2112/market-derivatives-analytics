# Derivatives engine

Pure Python module that models **synthetic** options on prediction-market
probabilities. No network, no file I/O, no global state. Given a market's YES
price `p` (a probability in `[0, 1]`) and an optional order book, it returns
payoff curves, break-even points, max profit/loss, a fair price, an implied
probability, and option Greeks (delta, gamma, theta, vega) as plain,
JSON-serializable data ready for an API or UI.

This is analytics, not a trading claim. See [MODEL.md](MODEL.md) for the
assumptions behind the pricing and the Greeks.

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
    sigma=0.15,        # dispersion ("vol" analogue); see MODEL.md
    tau=0.8,           # fraction of time to resolution
    book=book,         # optional
)
# bundle is plain JSON-serializable data: fair_price, implied_probability,
# break_even, max_profit, max_loss, payoff_curve {x, payoff},
# greek_curve {p, delta, gamma, vega, theta, price}, and point greeks.
```

## Public functions the UI should call

| Function | Returns |
|----------|---------|
| `analyze(p, strike, option_type, side, sigma, tau, book, num_points)` | the full bundle below, as one dict |
| `payoff_curve(strike, option_type, side, premium, num_points)` | `{x, payoff}` arrays over the 0..1 outcome range |
| `break_even(strike, option_type, premium)` | list of break-even outcome value(s) |
| `max_profit_loss(strike, option_type, side, premium)` | `{max_profit, max_loss}` |
| `fair_price(p, strike, option_type, sigma, book)` | model fair value of the option |
| `implied_probability(p, book)` | market implied probability of YES |
| `greeks(p, strike, option_type, sigma, tau, book)` | `{delta, gamma, vega, theta, ...}` at one probability |
| `greek_curve(strike, option_type, sigma, tau, num_points, book)` | `{p, delta, gamma, vega, theta, price}` arrays for charting |

### Greeks for the UI

`greek_curve(...)` is the function to chart: it returns the probability grid `p`
and parallel arrays for `delta`, `gamma`, `vega`, `theta`, and `price`.
`greeks(...)` gives the point values at the current probability. The meaning of
each Greek in this binary-market context is documented in
[MODEL.md](MODEL.md#the-greeks-in-this-context).

## Decoupling from the data layer

The engine takes simple typed inputs (a `p` in `[0, 1]` and an `OrderBook` of
`(price, size)` levels). When the data layer's normalized `Market` schema lands,
map `Market.yes_price -> p` and `Market.order_book -> OrderBook` in a thin
adapter; no engine change is needed.
