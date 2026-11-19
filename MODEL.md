# Modeling assumptions

This engine computes **synthetic** derivatives on prediction-market probabilities.
Nothing here trades and nothing here is advice. It is analytics: given a market's
current YES price and (optionally) its order book, it produces payoff curves,
break-even points, a fair price, a chance of profit, an implied probability, and
option Greeks, as plottable data.

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

## Held to resolution: an exact, model-free price

An option **held to resolution** is a *scaled binary*: at resolution the
settlement is `Bernoulli(p)` (`x = 1` with probability `p`, `x = 0` with
probability `1 - p`). Its value is therefore exact and needs no model:

```
call (held to resolution) = p * (1 - K)
put  (held to resolution) = (1 - p) * K
short = the negative of long
```

This is the correct no-arbitrage price for a binary that settles at `{0, 1}`. In
particular the YES share itself is the `K = 0` call: `p * (1 - 0) = p`. These
formulas are what the engine returns at `tau = 1` (see `tau` below), and they are
the basis labelled `pricing_basis = "held_to_resolution"` in the bundle.

## Before resolution: a logit-normal martingale

For an option whose expiry is **before** resolution, we need the distribution of
the YES probability `P_t` at that expiry. We model its path with a process
**bounded in `[0, 1]`** — a logit-normal martingale:

```
P_t = logistic(logit(p) + m + s * Z),   Z ~ Normal(0, 1)
```

* `logit(q) = log(q / (1 - q))`, `logistic` is its inverse, so `P_t` is always in
  `(0, 1)` — it can never leave the probability range (the Bachelier/normal model
  used previously could, which is the no-arbitrage bug the audit found).
* The drift `m` is solved so that `E[P_t] = p`. This makes the process a
  **martingale**, which is what forces put-call parity to hold exactly (below).
* `s` is the dispersion of the log-odds path; it grows with the time left.

The option price is the expectation of the intrinsic payoff:

```
call = E[max(P_t - K, 0)]
put  = E[max(K - P_t, 0)]
```

### Why the exact formula is the limit, not a special case

As `s -> infinity` the terminal law tends to `Bernoulli(p)`. `P_t` is bounded in
`[0, 1]` and the identity is a bounded continuous function, so the martingale
constraint `E[P_t] = p` forces **every** weak limit to have mean `p`. The mass in
the interior transition region vanishes (its width in `log-odds` is `O(1)` while
`s -> infinity`, so the interior probability is `O(1/s)`), leaving a two-point law
on `{0, 1}` with mean `p` — exactly `Bernoulli(p)`. So:

```
E[max(P_t - K, 0)]  ->  p * (1 - K)      as s -> infinity
E[max(K - P_t, 0)]  ->  (1 - p) * K
```

The held-to-resolution value is the `s -> infinity` limit of the same model, not
a separate rule. We return the closed form at `tau = 1` (the limit) and the
quadrature below for `tau < 1`.

**Convergence rate.** The approach is slow: the interior mass, and the gap
between the model price (or chance of profit) and the exact value, shrink like
**`~1/s`**. A representative check at `p = 0.6` (solving `m` so `E[P_t] = p`):

| `s`  | `Phi(m/s)` | interior mass in `(0.05, 0.95)` |
|------|-----------|---------------------------------|
| 2    | 0.631     | 0.826                           |
| 5    | 0.606     | 0.420                           |
| 20   | 0.6004    | 0.110                           |
| 100  | 0.6000    | 0.022                           |
| 400  | 0.6000    | 0.0055                          |

So near the very top of the `tau` slider the model value is within a small
`~1/s` tolerance of the exact value and the two meet exactly at `tau = 1`; the
tests pin this (`tau = 1` exact, `tau = 1 - eps` close within the `~1/s` band).

### `tau` — the time-left slider

`tau in (0, 1]` is the fraction of time remaining to resolution. The dispersion
is mapped from it as

```
s(tau) = s_base * sqrt(tau / (1 - tau))        (widened by the order book, below)
```

so that:

* `tau -> 0` (expiry is now): `s -> 0`, the law collapses to the point `p`, and
  the price tends to the **intrinsic** value `max(p - K, 0)` / `max(K - p, 0)`.
* `tau -> 1` (held to resolution): `s -> infinity`, and the price tends to the
  **exact** value `p * (1 - K)` / `(1 - p) * K`.

`s_base` is a documented default (`2.0`), widened by the order book (below). The
price is non-decreasing in `tau` (more time left is worth more), continuous in
`tau`, and stays inside the no-arbitrage band for every `tau`.

### Quadrature (stable at any `s`)

The expectations are evaluated by integrating in **log-odds space** `u = m + s*Z`,
so `U ~ Normal(m, s^2)`:

```
E[f(logistic(U))] = integral over u in [-U_HI, U_HI] of f(logistic(u)) * N(u; m, s^2) du
                    + P(U >  U_HI) * f(1)
                    + P(U < -U_HI) * f(0)
```

The integral is a fixed-grid composite Simpson rule (`U_HI = 40`, 401 nodes) and
the two boundary tails are exact via the normal survival function. Because we
integrate in `u` rather than in `Z`, the grid resolution does **not** degrade as
`s` grows — this is what lets the model reproduce the slow `~1/s` approach to the
exact limit accurately. The drift `m` scales like `s * Phi^{-1}(p)`, solved by
Newton from that starting guess (with a wide, `s`-scaled bisection fallback).

The chance of profit needs no quadrature: `P(P_t > threshold) = P(U > logit(threshold))`,
an exact normal tail, which at the limit gives `p` for any interior threshold.

## No-arbitrage, parity and bounds (what the tests pin)

For **every** `tau in (0, 1]`:

* **No-arbitrage bounds.** `call in [max(p - K, 0), p * (1 - K)]` and
  `put in [max(K - p, 0), (1 - p) * K]`. The lower bound is the intrinsic value
  (by Jensen); the upper bound is the held-to-resolution value (the payoff is a
  convex function dominated by the chord through `(0, 0)` and `(1, 1 - K)`). The
  premium therefore never exceeds the option's maximum payoff.
* **Put-call parity.** `C - P = p - K`, because `E[P_t] = p` (the martingale
  constraint). This holds for every `tau`, not just at resolution.
* **Monotonic in time.** The price is non-decreasing in `tau` for a long option
  (a mean-preserving spread of a convex payoff can only raise its expectation).

## Chance of profit

`prob_of_profit` uses the **same** `tau` and the **same** premium as the displayed
price. The net payoff is positive on exactly one side of the break-even, so the
chance of profit is the probability mass of the terminal YES probability on that
side: `P(P_t > break_even)` for a long call / short put, `P(P_t < break_even)` for
a long put / short call. As `tau -> 1` this tends to the exact values: a long call
is profitable iff YES (`= p`), a long put iff NO (`= 1 - p`), and the short sides
are the complements. At `p = K = 0.5` held to resolution a long call's chance of
profit is exactly `0.5`.

## The Greeks in this context

All four are derivatives of the **displayed premium**, and each is also returned as
a curve over `p` for charting. They are computed by finite differences of the same
price function the UI shows, so they describe exactly that surface.

| Greek | Definition here | Meaning |
|-------|-----------------|---------|
| **delta** | `d(price)/dp` | How the option value moves as the market's YES probability moves. Bounded in `[0, 1]` (call) / `[-1, 0]` (put). |
| **gamma** | `d^2(price)/dp^2` | Convexity of value in the probability. Peaks near `p == K` at a moderate horizon and flattens as the dispersion widens. `>= 0`. |
| **vega** | `d(price)/ds` | Sensitivity to the path dispersion `s` (how unresolved the market is). `>= 0`. An analogy — `s` is a dispersion we define, not a traded implied vol. |
| **theta** | `d(price)/d(tau)` | The rate of change of the **displayed premium** with the time-left slider. `>= 0` for a long option (more time left is worth more). This is, by construction, the finite-difference slope of the same premium the UI shows — fixing the old bug where the premium ignored `tau` while theta used it. |

At `tau = 1` (held to resolution) the price is linear in `p` and flat in `s`/`tau`,
so the Greeks are those of the exact formula: a call has `delta = 1 - K`,
`gamma = 0`, `theta = 0`; a put has `delta = -K`, `gamma = 0`, `theta = 0`.

Delta and gamma are literal quantities of this model. Vega and theta are
analogies whose magnitude depends on the `s`/`tau` mapping; their signs and shapes
(vega `>= 0`, gamma `>= 0` peaking near the strike, theta `>= 0` for a long option,
delta bounded) are the robust, reportable facts.

## Order book

The order book, when supplied, is used two ways:

* `implied_probability` prefers the **top-of-book mid** when both sides are
  present, falling back to the raw YES price `p`.
* The dispersion `s` is **widened by the half-spread** of the top of book (a wider
  book signals more disagreement about the outcome, i.e. a less-resolved market):
  `s_base -> s_base + 6 * half_spread`. So the dispersion **varies per market**,
  as this section documents — a market with a wide book prices a higher
  time-value premium than a tight one. A test pins this behaviour.

If no book is supplied (or it is empty or one-sided), the engine runs purely off
`p` and the default `s_base`, a clean fallback.

## Break-even

The break-even is the outcome value where the net payoff crosses zero:
`K + premium` for a call, `K - premium` for a put. If that value falls outside
`[0, 1]` it is reported as **absent** (an empty list from the engine, `null` in the
API bundle) — never clamped to a boundary, because a clamped value would be a
break-even that does not actually exist. With the exact held-to-resolution premium
the break-even is always in range; the null path guards the general case and any
inflated-premium input.

## Honest limits

* The model is a **deliberately simple** one-parameter (`s`) description of how a
  prediction-market probability might drift before a sub-resolution expiry. `s` is
  not calibrated to historical volatility; it is a documented function of the
  time-left slider and the order-book spread.
* `vega` and `theta` are **analogies**: `s` is a dispersion we define and "time"
  is a modeled resolution horizon, not a listed expiry.
* The approach to the exact held-to-resolution value is **slow (`~1/s`)**; the
  model meets the exact value precisely only at `tau = 1`, and is within a `~1/s`
  tolerance just below it.
* This is analytics on synthetic options, not a tradeable quote and not advice.
