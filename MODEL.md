# Modeling assumptions

This engine computes **synthetic** derivatives on prediction-market probabilities.
Nothing here trades and nothing here is advice. It is analytics: given a market's
current YES price and (optionally) its order book, it produces payoff curves,
break-even points, a fair price, an implied probability, and option Greeks, as
plottable data.

## The underlying

A binary prediction market (Polymarket, Kalshi) trades a single contract whose
price `p` lives in `[0, 1]`. That price **is** the market's implied probability
that the event resolves YES. At resolution the contract settles to `1` (YES) or
`0` (NO).

We treat that probability `p` as the underlying of a synthetic option with a
strike `K`, where `K` is itself a probability level in `[0, 1]`.

* A **synthetic call** pays `max(x - K, 0)` at settlement value `x`.
* A **synthetic put** pays `max(K - x, 0)`.

The payoff curves are charted across the continuous `x in [0, 1]` range so a UI
can draw a line, even though a real binary market settles only at `x = 0` or
`x = 1`.

## Intrinsic payoff vs. model price

Two layers are kept separate on purpose:

1. **Intrinsic payoff, break-even, max profit/loss** are model-free. They follow
   directly from the payoff definition above and the premium paid. These are
   exact, not assumptions.
2. **Fair price and Greeks before resolution** require a model of how uncertain
   the settlement is today. Those depend on the choice below.

## Pricing model: Bachelier (normal) on the probability

We price the option as the expected payoff when the settlement value is
`x ~ Normal(p, sigma)`:

```
d      = (p - K) / sigma
call   = (p - K) * Phi(d) + sigma * phi(d)
put    = call - (p - K)          # Bachelier put-call parity
```

where `Phi` and `phi` are the standard-normal CDF and PDF.

**Why Bachelier rather than Black-Scholes.** Black-Scholes assumes a
log-normal, strictly positive, unbounded underlying and a multiplicative
`sigma * sqrt(T)` diffusion. A probability is additive, bounded in `[0, 1]`, and
can sit at either end, so a normal (arithmetic) model fits the quantity far more
honestly and gives clean, closed-form, bounded Greeks. It is the standard choice
when the underlying is a rate or a spread that can be near zero.

**Honest caveat.** The normal distribution has tails outside `[0, 1]`, so the
model assigns a little probability mass to impossible settlement values. We do
not truncate, because for the dispersions we use the effect is small and
truncation would cost the closed-form Greeks. Prices are additionally clamped in
spirit by the tests (they stay within `[0, 1]`). Treat the fair price as an
analytic reference value, not a tradeable quote.

## What `sigma` and `tau` mean here (the "vol" and "time" analogues)

Classic option Greeks assume a volatility and a time-to-expiry. In a binary
market neither exists literally, so we map them to honest analogues and say so:

* **`sigma` ("vol" analogue)** is the dispersion of the settlement value today:
  how unresolved the market still is. A market pinned near 0 or 1 that everyone
  agrees on has low `sigma`; a contested 50/50 market has high `sigma`. When an
  order book is supplied, `sigma` is widened by the **half-spread** of the top of
  book, because a wider book signals more disagreement about the outcome. This is
  a deliberately simple, documented rule, not a calibrated vol surface.
* **`tau` ("time" analogue)** is the fraction of time remaining to resolution, in
  `(0, 1]`. The effective dispersion scales as `sigma * sqrt(tau)`, so as
  resolution approaches (`tau -> 0`) the price collapses to the intrinsic payoff.

## The Greeks in this context

All four are derivatives of the price with respect to the probability `p`
(or to `sigma` / time), and each is also returned as a curve over `p` for
charting.

| Greek | Definition here | Meaning in a binary market |
|-------|-----------------|----------------------------|
| **delta** | `d(price)/dp` | How the option value moves as the market's YES probability moves. `Phi(d)` for a call (in `[0, 1]`), `Phi(d) - 1` for a put (in `[-1, 0]`). Literal and well-defined. |
| **gamma** | `d^2(price)/dp^2` | `phi(d)/sigma`. Convexity of value in the probability; peaks when `p == K` (the option is most sensitive to probability moves right at the strike). Literal. |
| **vega** | `d(price)/dsigma` | `phi(d)`, always `>= 0`. Sensitivity to how unresolved the market is. This is an **analogy**: `sigma` is a dispersion we define, not a traded implied vol. |
| **theta** | `-d(price)/d(tau)` | `-(vega * sigma / (2*sqrt(tau)))`, `<= 0` for a long option. Time-decay as resolution nears and dispersion shrinks. Also an **analogy**, since "time" here is a modeled resolution horizon, not a listed expiry. |

Delta and gamma are literal quantities of this model. Vega and theta are
analogies whose precise magnitude depends on the `sigma`/`tau` mapping above;
their **signs and shapes** (vega `>= 0`, gamma peaks at the strike, theta `<= 0`,
delta bounded) are the robust, reportable facts.

## Order book

The order book is used two ways, both optional and documented:

* `implied_probability` prefers the **top-of-book mid** when both sides are
  present, falling back to the raw YES price `p`.
* `sigma` is widened by the **half-spread**, as described above.

If no book is supplied, the engine runs purely off `p` and the caller's `sigma`.
