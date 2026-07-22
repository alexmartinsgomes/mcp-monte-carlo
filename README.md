# mcp-monte-carlo

**Give any AI agent the power to run a serious Monte Carlo forecast for a stock or ETF — in one tool call.**

This is an [MCP](https://modelcontextprotocol.io/) (Model Context Protocol) server. Connect it once to Hermes, Claude Desktop, Cursor, or any MCP-capable agent, and the agent can download market history, fit a volatility model, simulate thousands of future price paths, and return percentiles, drawdowns, and risk probabilities — without you writing a single line of simulation code.

```text
You:  "What does a bad year look like for SPY over the next 12 months?"
Agent → forecast_asset_monte_carlo("SPY")
      → EGARCH + skewed-t Monte Carlo (5,000 paths by default)
You ← JSON: price/return percentiles, vol, max drawdowns, loss probabilities
```

---

## Why this matters

Large language models are excellent at reasoning and explanation. They are **not** engines for sampling fat-tailed returns under time-varying volatility. Left alone, an agent might invent plausible-looking percentiles or hand-wave “historical vol $\times\sqrt{T}$”.

This server closes that gap:

| Without this MCP | With this MCP |
|---|---|
| Agent guesses ranges or quotes stale numbers | Agent calls a reproducible statistical pipeline |
| No consistent treatment of crashes / fat tails | Skewed-t innovations model skewness and fat tails |
| Constant-vol assumptions ignore clustering | EGARCH captures shock-driven, asymmetric volatility |
| Hard to compare 7-day vs 10-year risk | Same model, same paths, many horizons in one JSON |

The agent stays in charge of *interpretation* and *conversation*. The MCP owns *estimation* and *simulation*.

---

## What it does (pipeline)

```text
Yahoo Finance (max history)
        │  adjusted daily Close
        ▼
  Log returns
        │
        ▼
  Fit EGARCH(1,1) + leverage  +  skewed-t shocks
        │  constant mean drift (historical mean)
        ▼
  Simulate N paths  (default 5,000) out to 10 years
        │
        ▼
  Summarize each horizon → percentiles, vol, MDD, probabilities
```

### 1. Data

Uses [`yfinance`](https://github.com/ranaroussi/yfinance) to pull the **maximum** available daily history. The `Close` field is already adjusted for splits and dividends, so returns are suitable for long-horizon compounding.

### 2. Returns and drift

Prices are converted to **log returns**:

$$
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
$$

The mean model is **constant**: each simulated day has drift equal to the fitted historical average $\mu$. That is a simple, transparent assumption — not a crystal ball for future expected return.

### 3. Volatility: EGARCH with leverage

Equity volatility is neither constant nor symmetric:

- **Volatility clustering** — turbulent days tend to follow turbulent days.
- **Leverage effect** — large *down* moves tend to raise future vol more than equally large *up* moves.

This server fits **EGARCH(1,1) with leverage** ($p=1$, $o=1$, $q=1$) via the [`arch`](https://arch.readthedocs.io/) package. Conditionally, log-variance evolves roughly as:

$$
\ln(\sigma_t^2)
=
\omega
+ \alpha\bigl(\lvert z_{t-1}\rvert - \mathbb{E}[\lvert z\rvert]\bigr)
+ \gamma z_{t-1}
+ \beta\ln(\sigma_{t-1}^2)
$$

For equities, the leverage coefficient $\gamma$ is typically **negative**: a negative shock $z$ increases tomorrow’s volatility.

### 4. Shocks: skewed Student-t

Gaussian shocks understate crash risk. Standardized innovations are drawn from a **skewed t** distribution, so simulated paths can show:

- fat tails (extreme moves more often than a normal),
- skewness (asymmetric left/right risk).

### 5. Monte Carlo paths

Given the fitted parameters, the server simulates $N$ forward trajectories (`n_paths`; vectorized NumPy loop for stability out to multi-year horizons). Each path is a full price series; horizons are slices of those same paths so short- and long-term stats are coherent.

### 6. Horizons (trading days)

| Label | Trading days | Rough calendar |
|------:|-------------:|----------------|
| `7d`  | 5            | ~1 week        |
| `30d` | 21           | ~1 month       |
| `3m`  | 63           | ~3 months      |
| `6m`  | 126          | ~6 months      |
| `1y`  | 252          | ~1 year        |
| `3y`  | 756          | ~3 years       |
| `5y`  | 1260         | ~5 years       |
| `10y` | 2520         | ~10 years      |

---

## Tools

### `forecast_asset_monte_carlo(ticker, n_paths=5000)`

**When to use:** The user wants forward scenarios, risk ranges, or path statistics for a ticker (e.g. `SPY`, `AAPL`).

For **each** horizon, the JSON includes:

- **Price percentiles** — `1, 5, 10, 25, 50, 75, 90, 95, 99`
- **Return percentiles (%)** — same grid, vs today’s price
- **Annualized volatility (%)** — cross-sectional vol of path outcomes at that horizon
- **Max-drawdown percentiles (%)** — peak-to-trough loss along each path up to that horizon
- **Probabilities** — end below start, ±20% moves, max drawdown over 20%

`n_paths` defaults to `5000` (minimum `100`). More paths → smoother percentile estimates, slower run.

### `inspect_asset_model(ticker)`

**When to use:** Validate data quality or model sanity *before* (or instead of) a full forecast — enough history? sensible parameters? how fat are residual tails?

Returns history span, last price, fitted EGARCH + skew-t parameters, AIC/BIC, last conditional volatility (daily and annualized), and residual skewness / excess kurtosis.

Does **not** simulate paths. Prefer `forecast_asset_monte_carlo` for percentiles and drawdowns.

---

## Requirements

- macOS, Linux, or Windows
- [uv](https://docs.astral.sh/uv/) (recommended)
- Python **≥ 3.12** (declared in `pyproject.toml`)
- Network access (Yahoo Finance download)

---

## Quick start (local)

```bash
cd /path/to/mcp-monte-carlo
uv sync
```

Smoke-test without MCP:

```bash
uv run python -c "
from server import run_inspect, run
import json
print(json.dumps(run_inspect('SPY'), indent=2))
print(json.dumps(run('SPY', 200)['horizons']['1y'], indent=2))
"
```

Run the MCP server on stdio:

```bash
uv run mcp-monte-carlo
# or, from a published clone / path:
uvx --from /path/to/mcp-monte-carlo mcp-monte-carlo
```

---

## Connect an AI agent

### Hermes Agent (`~/.hermes/config.yaml`)

Prefer `uv run` against a synced project (faster and more reliable than a cold `uvx`):

```yaml
mcp_servers:
  mcp-monte-carlo:
    command: /opt/homebrew/bin/uv   # which uv  → paste absolute path
    args:
      - run
      - --directory
      - /ABSOLUTE/PATH/TO/mcp-monte-carlo
      - mcp-monte-carlo
    connect_timeout: 120
    timeout: 300
```

Then: `hermes mcp test mcp-monte-carlo` or `/reload-mcp` in a chat.

### Cursor / Claude Desktop

```json
{
  "mcpServers": {
    "mcp-monte-carlo": {
      "command": "uvx",
      "args": [
        "--from",
        "/ABSOLUTE/PATH/TO/mcp-monte-carlo",
        "mcp-monte-carlo"
      ]
    }
  }
}
```

Once published on GitHub, others can point `--from` at the repo URL or clone locally and use the same pattern.

---

## Example agent prompts

- “Inspect the EGARCH model for `QQQ`, then forecast with 2,000 paths.”
- “For `AAPL`, what is the 5th percentile price in 1 year, and the probability of a >20% max drawdown?”
- “Compare 1-year median and 95th percentile max drawdown for `SPY` vs `TLT`.”

---

## Project layout

```text
mcp-monte-carlo/
├── server.py          # MCP tools + EGARCH/skew-t Monte Carlo (single module)
├── pyproject.toml     # package metadata, deps, console entry point
├── uv.lock            # locked dependency versions
├── README.md
└── .gitignore
```

One Python file keeps the project easy to read, audit, and ship.

---

## Model caveats (read this)

This is a **research / educational risk tool**, not investment advice and not a guarantee of future prices.

- Past drift $\mu$ is **not** a forecast of expected return; long-horizon medians inherit that assumption.
- EGARCH(1,1)+leverage and skewed-t are strong defaults for many liquid equities/ETFs — not universally “optimal” for every ticker.
- Yahoo data quality and corporate actions can affect results; always check `inspect_asset_model` on unfamiliar symbols.
- Extremely long horizons (5y–10y) compound model risk; treat tails as illustrative, not certainties.

---

## License / authorship

Created by Alexandre Martins. Use and adapt freely for personal agents and learning; if you redistribute, keep attribution and these caveats visible.
