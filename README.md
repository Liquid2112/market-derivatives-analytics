# Market Derivatives Analytics

A **read-only** analytics tool for prediction markets. Pull any market from
**Polymarket** or **Kalshi** (each trades as a probability — a YES price between
0 and 1), view its live order book, define a **synthetic option/derivative** on
that market, and see its **payoff curve** and the four **option Greeks**
(delta, gamma, theta, vega) rendered as live charts with current scalar values.

> **Analytics and modeling only.** This project does not place trades, does not
> connect to any brokerage, and is not financial advice. Every number shown is a
> model of a hypothetical synthetic position on a prediction-market probability.
> All market access is read-only (GET requests only).

## What it does

1. **Search** real prediction markets across Polymarket and Kalshi.
2. **Inspect** a market's live YES-outcome **order book** (bids and asks).
3. **Model a synthetic option** on the market probability — pick call/put,
   long/short, and a strike (a probability in 0..1).
4. **See the analytics**: fair-value premium, break-even, implied probability,
   a payoff curve, and all four Greeks — each as a chart over the 0..1 outcome
   range plus the current scalar value. The Greeks are the headline feature and
   are computed from the real market price, not placeholders.

## Components

The project is built from three cooperating layers in this one repository:

- **Data layer** (`mdlab/`) — fetches and normalizes markets and order books
  from Polymarket (Gamma + CLOB) and Kalshi (public trade-api) into one shared
  schema. Read-only. Documented in `SCHEMA.md`.
- **Derivatives engine** (`derivatives_engine/`) — pure Python, no I/O, that
  computes payoff, break-even, implied probability, a documented fair price and
  the four Greeks under a Bachelier (normal) model on the probability.
  Documented in `MODEL.md`.
- **UI + API** (`app/`) — a thin FastAPI backend exposing read-only JSON
  endpoints (`/api/markets`, `/api/market/{id}`, `/api/derivative/{id}`), plus a
  dependency-light web UI with the order book, payoff chart and four Greek
  charts. The single seam `app/adapter.py` wires the data layer and the engine
  together and reconciles their shapes.

## Run it

Requires Python 3.10+.

```bash
# 1. (optional) create and activate a virtual environment
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. start the server
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then open **http://127.0.0.1:8000** in your browser.

By default the app runs against the **real** Polymarket and Kalshi data (live
read-only HTTP). To run fully offline with the built-in deterministic demo
markets — no network — set an environment variable before starting the server:

```bash
# Windows (PowerShell):  $env:MDA_USE_REAL_BACKENDS = "0"
# macOS/Linux:           export MDA_USE_REAL_BACKENDS=0
```

The `/api/health` endpoint reports which mode is active (`using_mock_backends`).

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The suite covers the engine math, the data-layer normalizers (against saved
fixtures), the API against the offline demo backends, and an **integration**
suite (`tests/test_integration_real.py`) that exercises the real
`mdlab → engine → bundle` wiring with the network replaced by saved fixtures.
CI (`.github/workflows/ci.yml`) runs the whole suite on Python 3.10 and 3.12.
