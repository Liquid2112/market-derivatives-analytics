# UI + API layer

A thin [FastAPI](https://fastapi.tiangolo.com/) backend plus a dependency-light
web UI for the Market Derivatives Analytics tool.

> **Analytics and modeling only.** Nothing here places a trade or connects to a
> brokerage. This is not financial advice.

## Run it

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Then open **http://127.0.0.1:8000/** in a browser.

Search any market across Polymarket and Kalshi, pick one to see its live
YES/NO price and order book, then define a synthetic option (call/put,
long/short, strike as a probability) and watch the payoff curve and all four
Greeks — delta, gamma, theta, vega — update as curves plus current scalar
values.

## Run the tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## API (all read-only JSON)

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Liveness + whether mock backends are in use. |
| `GET` | `/api/markets?q=&provider=&limit=` | Search/list markets across both providers. |
| `GET` | `/api/market/{market_id}` | One market plus its order book. |
| `GET` | `/api/derivative/{market_id}?option_type=&side=&strike=` | Synthetic-option bundle: payoff curve, break-even, implied prob, and the four Greeks (scalars + curves). |

`provider` is `polymarket`, `kalshi`, or omitted for both. `option_type` is
`call`/`put`, `side` is `long`/`short`, `strike` is a probability in `[0, 1]`.

## How this connects to the sibling layers

The data layer (`schema.py` / `SCHEMA.md`) and the derivatives engine
(`MODEL.md`) are built by sibling workers in this same repository. This layer
reaches both **only through `app/adapter.py`**, which defines the `DataLayer`
and `Engine` contracts it expects and ships deterministic **mock**
implementations so the UI and tests run standalone.

**Final wiring is one file.** Set `MDA_USE_REAL_BACKENDS=1` and implement
`_real_data_layer()` / `_real_engine()` in `app/adapter.py` to import the real
siblings. Everything downstream (endpoints, UI, tests) stays unchanged as long
as the siblings satisfy the two Protocols documented there. See
`INTERFACES.md` for the exact expected shapes.
