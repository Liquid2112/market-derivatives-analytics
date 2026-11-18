# Market Derivatives Analytics

A **read-only** analytics tool for prediction markets. Pull any market from
**Polymarket** or **Kalshi** (each trades as a probability — a YES price between
0 and 1), view its live order book, define a **synthetic option/derivative** on
that market, and see its **payoff curve** and **option Greeks** (delta, gamma,
theta, vega) as live charts.

> **Analytics and modeling only.** This project does not place trades, does not
> connect to any brokerage, and is not financial advice. Everything shown is a
> model of a hypothetical synthetic position.

## Components

The project is built in three cooperating layers that share this repository:

- **Data layer** — fetches and normalizes markets and order books from
  Polymarket and Kalshi into a common schema (`schema.py`, documented in
  `SCHEMA.md`).
- **Derivatives engine** — pure Python that computes payoff, break-even,
  implied probability and the four Greeks, returning JSON-serializable bundles
  (documented in `MODEL.md`).
- **UI + API** — a thin FastAPI backend that wires the above together behind
  read-only JSON endpoints, plus a polished, dependency-light web UI.

See `app/README.md` once the UI + API layer lands for how to run the server.
