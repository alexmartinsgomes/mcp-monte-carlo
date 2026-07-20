"""MCP server: EGARCH + skewed-t Monte Carlo asset price forecasts."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

# Keep MCP handshake quiet and fast (heavy libs load only when the tool runs).
logging.getLogger("mcp").setLevel(logging.WARNING)
os.environ.setdefault("FASTMCP_LOG_LEVEL", "WARNING")

from mcp.server.fastmcp import FastMCP

PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)
MDD_PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)

# Calendar labels mapped to approximate trading-day horizons.
HORIZONS: dict[str, int] = {
    "7d": 5,
    "30d": 21,
    "3m": 63,
    "6m": 126,
    "1y": 252,
    "3y": 756,
    "5y": 1260,
    "10y": 2520,
}

MAX_HORIZON = max(HORIZONS.values())
MIN_OBS = 252
TRADING_DAYS_PER_YEAR = 252.0

mcp = FastMCP("mcp-monte-carlo")

def _extract_close(raw, ticker: str):
    """Return a 1-D Close series from a yfinance download."""
    import pandas as pd

    if isinstance(raw, pd.Series):
        close = raw.dropna()
        close.name = "Close"
        return close

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"]
        elif "Close" in raw.columns.get_level_values(-1):
            close = raw.xs("Close", axis=1, level=-1)
        else:
            raise ValueError(f"No Close column found for ticker {ticker!r}.")
        if isinstance(close, pd.DataFrame):
            if ticker in close.columns:
                close = close[ticker]
            else:
                close = close.iloc[:, 0]
    else:
        if "Close" not in raw.columns:
            raise ValueError(f"No Close column found for ticker {ticker!r}.")
        close = raw["Close"]

    close = pd.to_numeric(close, errors="coerce").dropna()
    close.name = "Close"
    return close

def download_closes(ticker: str):
    """Download maximum daily Close history (split/dividend adjusted)."""
    import yfinance as yf

    symbol = ticker.strip().upper()
    if not symbol:
        raise ValueError("Ticker must be a non-empty string.")

    raw = yf.download(
        symbol,
        period="max",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw is None or len(raw) == 0:
        raise ValueError(f"No price data returned for ticker {symbol!r}.")

    close = _extract_close(raw, symbol)
    if len(close) < MIN_OBS:
        raise ValueError(
            f"Need at least {MIN_OBS} daily closes for {symbol!r}; got {len(close)}."
        )
    return close

def fit_egarch_skewt(log_returns):
    """Fit EGARCH(1,1) with leverage (o=1), constant mean, skewed-t innovations."""
    from arch import arch_model

    pct_returns = log_returns.dropna() * 100.0
    model = arch_model(
        pct_returns,
        mean="Constant",
        vol="EGARCH",
        p=1,
        o=1,
        q=1,
        dist="skewt",
    )
    return model.fit(disp="off")

def prepare_model(ticker: str) -> dict[str, Any]:
    """Download data and fit EGARCH+skewt; shared by both MCP tools."""
    import numpy as np

    symbol = ticker.strip().upper()
    close = download_closes(symbol)
    log_returns = np.log(close).diff().dropna()
    result = fit_egarch_skewt(log_returns)
    last_price = float(close.iloc[-1])
    params = {str(k): float(v) for k, v in result.params.items()}

    # arch stores conditional vol in percent-return units.
    last_cond_vol_pct = float(result.conditional_volatility.iloc[-1])
    annualized_cond_vol_pct = last_cond_vol_pct * np.sqrt(TRADING_DAYS_PER_YEAR)

    std_resid = result.std_resid.dropna()
    return {
        "ticker": symbol,
        "close": close,
        "log_returns": log_returns,
        "result": result,
        "last_price": last_price,
        "params": params,
        "n_obs": int(len(log_returns)),
        "first_date": str(close.index[0].date()),
        "last_date": str(close.index[-1].date()),
        "last_conditional_vol_daily_pct": last_cond_vol_pct,
        "last_conditional_vol_annualized_pct": float(annualized_cond_vol_pct),
        "residual_skewness": float(std_resid.skew()),
        "residual_kurtosis": float(std_resid.kurtosis()),  # excess kurtosis (pandas)
        "loglikelihood": float(result.loglikelihood),
        "aic": float(result.aic),
        "bic": float(result.bic),
    }

def simulate_price_paths(
    result,
    last_price: float,
    n_paths: int,
    horizon: int = MAX_HORIZON,
):
    """Simulate price paths with fitted EGARCH(1,1)+leverage + skewed-t shocks."""
    import numpy as np

    params = result.params
    mu = float(params["mu"])
    omega = float(params["omega"])
    alpha = float(params["alpha[1]"])
    gamma = float(params["gamma[1]"])
    beta = float(params["beta[1]"])
    eta = float(params["eta"])
    lam = float(params["lambda"])

    norm_const = float(np.sqrt(2.0 / np.pi))

    dist = result.model.distribution
    rng = dist.simulate(np.array([eta, lam], dtype=float))
    z = np.asarray(rng((n_paths, horizon)), dtype=float)

    last_sigma = float(result.conditional_volatility.iloc[-1])
    last_z = float(result.std_resid.iloc[-1])
    if not np.isfinite(last_z):
        last_z = 0.0

    log_var = np.full(n_paths, np.log(last_sigma**2), dtype=float)
    prev_z = np.full(n_paths, last_z, dtype=float)
    pct_returns = np.empty((n_paths, horizon), dtype=float)

    for t in range(horizon):
        # Nelson EGARCH: α(|z|-E|z|) + γ z  (γ < 0 → leverage for equities)
        log_var = (
            omega
            + alpha * (np.abs(prev_z) - norm_const)
            + gamma * prev_z
            + beta * log_var
        )
        log_var = np.clip(log_var, -50.0, 50.0)
        sigma = np.exp(0.5 * log_var)
        shock = z[:, t]
        pct_returns[:, t] = mu + sigma * shock
        prev_z = shock

    log_returns = pct_returns / 100.0
    cumulative = np.cumsum(log_returns, axis=1)
    return last_price * np.exp(cumulative)

def _path_max_drawdowns(paths_to_h, last_price: float):
    """Max drawdown (positive fraction) for each path from last_price through horizon."""
    import numpy as np

    n_paths = paths_to_h.shape[0]
    start = np.full((n_paths, 1), last_price, dtype=float)
    series = np.concatenate([start, paths_to_h], axis=1)
    peak = np.maximum.accumulate(series, axis=1)
    drawdown = (series - peak) / peak
    return -drawdown.min(axis=1)  # positive fractions, e.g. 0.20 = 20% MDD

def summarize_paths(paths, last_price: float) -> dict[str, Any]:
    """Build per-horizon price/return/vol/MDD/probability summary."""
    import numpy as np

    horizons: dict[str, Any] = {}
    for label, days in HORIZONS.items():
        prices = paths[:, days - 1]
        returns_pct = (prices / last_price - 1.0) * 100.0
        log_horizon_returns = np.log(prices / last_price)
        ann_vol_pct = float(
            np.std(log_horizon_returns, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR / days) * 100.0
        )

        mdd = _path_max_drawdowns(paths[:, :days], last_price)
        mdd_pct = mdd * 100.0

        price_pct = {str(p): float(np.percentile(prices, p)) for p in PERCENTILES}
        return_pct = {str(p): float(np.percentile(returns_pct, p)) for p in PERCENTILES}
        mdd_table = {str(p): float(np.percentile(mdd_pct, p)) for p in MDD_PERCENTILES}

        horizons[label] = {
            "trading_days": days,
            "percentiles": price_pct,  # prices (backward compatible)
            "price_percentiles": price_pct,
            "return_percentiles_pct": return_pct,
            "annualized_volatility_pct": ann_vol_pct,
            "max_drawdown_percentiles_pct": mdd_table,
            "probabilities": {
                "end_below_start": float(np.mean(prices < last_price)),
                "end_down_at_least_20pct": float(np.mean(returns_pct <= -20.0)),
                "end_up_at_least_20pct": float(np.mean(returns_pct >= 20.0)),
                "max_drawdown_over_20pct": float(np.mean(mdd_pct >= 20.0)),
            },
        }
    return horizons

def run_forecast(ticker: str, n_paths: int = 5000) -> dict[str, Any]:
    """Full pipeline: download → fit → simulate → enriched JSON-ready dict."""
    if n_paths < 100:
        raise ValueError("n_paths must be at least 100.")

    prepared = prepare_model(ticker)
    paths = simulate_price_paths(
        prepared["result"],
        prepared["last_price"],
        n_paths=n_paths,
    )
    return {
        "ticker": prepared["ticker"],
        "last_price": prepared["last_price"],
        "n_paths": int(n_paths),
        "n_obs": prepared["n_obs"],
        "model": {
            "mean": "Constant",
            "vol": "EGARCH(1,1)+leverage",
            "dist": "skewt",
            "params": prepared["params"],
        },
        "horizons": summarize_paths(paths, prepared["last_price"]),
    }

def run_inspect(ticker: str) -> dict[str, Any]:
    """Fit-only diagnostics (no Monte Carlo paths)."""
    prepared = prepare_model(ticker)
    return {
        "ticker": prepared["ticker"],
        "last_price": prepared["last_price"],
        "n_obs": prepared["n_obs"],
        "history_start": prepared["first_date"],
        "history_end": prepared["last_date"],
        "model": {
            "mean": "Constant",
            "vol": "EGARCH(1,1)+leverage",
            "dist": "skewt",
            "params": prepared["params"],
            "loglikelihood": prepared["loglikelihood"],
            "aic": prepared["aic"],
            "bic": prepared["bic"],
        },
        "volatility": {
            "last_conditional_daily_pct": prepared["last_conditional_vol_daily_pct"],
            "last_conditional_annualized_pct": prepared[
                "last_conditional_vol_annualized_pct"
            ],
        },
        "standardized_residuals": {
            "skewness": prepared["residual_skewness"],
            "excess_kurtosis": prepared["residual_kurtosis"],
        },
        "notes": (
            "Use this tool to check data coverage and whether the EGARCH+skewt fit "
            "looks sane before trusting a long-horizon Monte Carlo forecast. "
            "High excess kurtosis or extreme lambda/eta can mean fat-tail estimates "
            "are unstable. Prefer forecast_asset_monte_carlo for price paths and risk."
        ),
    }

# Keep old name for local smoke tests / imports.
def run(ticker: str, n_paths: int = 5000) -> dict[str, Any]:
    return run_forecast(ticker, n_paths=n_paths)

@mcp.tool()
def forecast_asset_monte_carlo(ticker: str, n_paths: int = 5000) -> str:
    """Run a forward Monte Carlo forecast of an asset's future price distribution.

    Use this when the user wants scenario ranges, risk, or path statistics for a
    ticker (e.g. SPY, AAPL): price/return percentiles, annualized volatility,
    max-drawdown percentiles, and loss/gain probabilities at 7d, 30d, 3m, 6m,
    1y, 3y, 5y, and 10y trading-day horizons.

    Downloads max adjusted daily closes, fits EGARCH(1,1) with leverage
    (o=1) and skewed-t innovations (historical mean drift), then simulates
    ``n_paths`` paths.

    Prefer ``inspect_asset_model`` first only when you need fit/data diagnostics
    without simulating paths.

    Args:
        ticker: Yahoo Finance ticker symbol (e.g. SPY, AAPL).
        n_paths: Number of Monte Carlo paths (default 5000, minimum 100).
    """
    try:
        result = run_forecast(ticker=ticker, n_paths=n_paths)
        return json.dumps(result, indent=2)
    except Exception as exc:  # noqa: BLE001 — surface any failure to the agent
        return json.dumps({"error": str(exc)}, indent=2)

@mcp.tool()
def inspect_asset_model(ticker: str) -> str:
    """Inspect the EGARCH + skewed-t model fit for a ticker WITHOUT simulating paths.

    Call this when you need to validate data quality or model sanity before (or
    instead of) a full Monte Carlo forecast — for example: Is there enough
    history? What is today's conditional volatility? Do residuals look heavily
    skewed/fat-tailed? What are the fitted EGARCH and skew-t parameters?

    Do NOT use this for forward price scenarios, percentiles, drawdowns, or
    probabilities — use ``forecast_asset_monte_carlo`` for those.

    Returns JSON with history span, last price, fitted parameters, AIC/BIC,
    last conditional volatility (daily and annualized), and residual
    skewness/excess kurtosis.

    Args:
        ticker: Yahoo Finance ticker symbol (e.g. SPY, AAPL).
    """
    try:
        result = run_inspect(ticker=ticker)
        return json.dumps(result, indent=2)
    except Exception as exc:  # noqa: BLE001 — surface any failure to the agent
        return json.dumps({"error": str(exc)}, indent=2)

def main() -> None:
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
