# Interface contracts the UI/API layer assumes

The UI + API layer (this PR) depends on two sibling layers that are being
built in parallel in this same repository. Until their PRs land, the UI/API
layer talks to **mocks** behind `app/adapter.py`. This file records the exact
shapes the adapter expects, so final wiring is a mechanical swap. If a
sibling's real signature differs, adapt the thin wrapper in
`app/adapter.py::_real_data_layer` / `_real_engine` — not the callers.

## Data layer (`schema.py` / `SCHEMA.md`)

```python
class DataLayer(Protocol):
    def search_markets(self, query: str, provider: str | None = None,
                       limit: int = 25) -> list[dict]: ...
    def get_market(self, market_id: str) -> dict: ...
```

**Market summary** (`search_markets` item):

```json
{
  "id": "poly-btc-100k-eoy",
  "provider": "polymarket",          // "polymarket" | "kalshi"
  "title": "Bitcoin above $100k at year end?",
  "yes_price": 0.62,                  // probability in [0,1]
  "no_price": 0.38,                   // 1 - yes_price
  "volume": 12440000.0,
  "close_time": "2026-12-31T23:59:00Z",
  "category": "Crypto"
}
```

**Full market** (`get_market`): the summary above plus an `order_book`:

```json
{
  "order_book": {
    "bids": [[0.61, 2400.0], [0.60, 1800.0]],   // [price, size], price in [0,1]
    "asks": [[0.63, 2100.0], [0.64, 1500.0]]
  }
}
```

## Derivatives engine (`MODEL.md`)

```python
class Engine(Protocol):
    def bundle(self, yes_price: float, strike: float, option_type: str,
               side: str, premium: float | None = None) -> dict: ...
```

`yes_price`, `strike` are probabilities in `[0,1]`. `option_type` is
`"call"`/`"put"`, `side` is `"long"`/`"short"`.

**Bundle** (JSON-serializable):

```json
{
  "underlying_prob": 0.62,
  "strike": 0.5,
  "option_type": "call",
  "side": "long",
  "premium": 0.14,
  "break_even": 0.64,
  "implied_prob": 0.5,
  "payoff_curve": [{"p": 0.0, "payoff": -0.14}, "... 101 points over p=0..1"],
  "greeks": {"delta": 0.72, "gamma": 1.9, "theta": -0.3, "vega": 0.21},
  "greek_curves": {
    "delta": [{"p": 0.0, "value": 0.01}, "... 101 points"],
    "gamma": [{"p": 0.0, "value": 0.00}],
    "theta": [{"p": 0.0, "value": 0.00}],
    "vega":  [{"p": 0.0, "value": 0.00}]
  }
}
```

The mock engine in `app/adapter.py` uses a Bachelier (normal) model over the
probability underlying for a smooth premium/Greeks surface. The real engine
may use a different model; the UI only relies on the **shape** above and that
each curve spans `p = 0..1` in 101 points.
