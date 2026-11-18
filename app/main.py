"""Thin FastAPI backend for the Market Derivatives Analytics UI.

All endpoints are read-only JSON. Nothing here places a trade; the service
only reads markets (via the data layer) and computes models (via the engine),
both reached through ``app.adapter``.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import adapter

app = FastAPI(
    title="Market Derivatives Analytics",
    version="0.1.0",
    description=(
        "Read-only analytics for prediction markets. Analytics and modeling "
        "only — not trading, not financial advice."
    ),
)

_STATIC_DIR = Path(__file__).parent / "static"

_PROVIDERS = {"polymarket", "kalshi"}
_OPTION_TYPES = {"call", "put"}
_SIDES = {"long", "short"}


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "using_mock_backends": adapter.using_mock_backends()}


@app.get("/api/markets")
def list_markets(
    q: str = Query("", description="Search text; empty returns featured markets."),
    provider: str | None = Query(None, description="polymarket | kalshi | omit for both"),
    limit: int = Query(25, ge=1, le=100),
) -> JSONResponse:
    if provider is not None and provider not in _PROVIDERS:
        raise HTTPException(status_code=422, detail="provider must be 'polymarket' or 'kalshi'")
    data = adapter.get_data_layer()
    # Use the warning-aware search so a provider outage surfaces instead of
    # being swallowed into an empty "no markets match". Both the real and the
    # mock data layers implement it.
    result = data.search_markets_result(q, provider=provider, limit=limit)
    markets = result["markets"]
    warnings = result.get("warnings", [])
    # Total provider outage with nothing to show: a 502, not a misleading
    # empty 200 that reads to the user as "no markets match".
    if result.get("all_failed") and not markets:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "all market-data providers are currently unavailable",
                "warnings": warnings,
            },
        )
    # `markets` and `count` stay unchanged; `warnings` is additive.
    body: dict = {"markets": markets, "count": len(markets)}
    if warnings:
        body["warnings"] = warnings
    return JSONResponse(body)


@app.get("/api/market/{market_id}")
def get_market(market_id: str) -> JSONResponse:
    try:
        market = adapter.get_data_layer().get_market(market_id)
    except adapter.ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"market '{market_id}' not found")
    return JSONResponse(market)


@app.get("/api/derivative/{market_id}")
def get_derivative(
    market_id: str,
    option_type: str = Query("call", description="call | put"),
    side: str = Query("long", description="long | short"),
    strike: float = Query(..., ge=0.0, le=1.0, description="Strike as a probability 0..1"),
    tau: float = Query(
        1.0,
        gt=0.0,
        le=1.0,
        description=(
            "Modeled time-to-resolution in (0, 1] (a resolution horizon "
            "analogue, not a listed expiry). Default 1.0. Shrinking it toward "
            "0 tightens the model dispersion, so the model-based Greeks and "
            "probability-of-profit update accordingly."
        ),
    ),
) -> JSONResponse:
    option_type = option_type.lower()
    side = side.lower()
    if option_type not in _OPTION_TYPES:
        raise HTTPException(status_code=422, detail="option_type must be 'call' or 'put'")
    if side not in _SIDES:
        raise HTTPException(status_code=422, detail="side must be 'long' or 'short'")
    try:
        market = adapter.get_data_layer().get_market(market_id)
    except adapter.ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"market '{market_id}' not found")

    yes_price = market.get("yes_price")
    if yes_price is None:
        # The market exists but has no usable YES price yet (never traded, no
        # resting quote). Pricing a derivative off it would be meaningless, so
        # say so clearly instead of crashing on float(None).
        raise HTTPException(
            status_code=422,
            detail=f"market '{market_id}' has no YES price yet; cannot price a derivative",
        )

    bundle = adapter.get_engine().bundle(
        yes_price=float(yes_price),
        strike=float(strike),
        option_type=option_type,
        side=side,
        tau=float(tau),
    )
    bundle["market"] = {
        "id": market["id"],
        "provider": market["provider"],
        "title": market["title"],
        "yes_price": market["yes_price"],
        "no_price": market.get("no_price"),
    }
    return JSONResponse(bundle)


# --------------------------------------------------------------------------
# Static UI
# --------------------------------------------------------------------------
@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
