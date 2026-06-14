# DRAM-lessons research workflow

3 small scripts built directly from the lessons of the 2026-06-05 DRAM
short-put fiasco. Run any of them as standalone CLI tools.

## Quick start

```bash
bash research/setup.sh           # installs deps in a new sandbox
python research/sec_monitor.py AVGO NVDA DRAM --risk
python research/options_backtest.py DRAM --years 2 --strike-pct 0.98
python research/realtime_options.py AVGO
```

## The 3 workflows

### 1. `sec_monitor.py`  —  SEC EDGAR filings monitor
*Lesson: read the 10-K Risk Factors before selling puts on a ticker.*

```bash
# list 5 latest 10-K/10-Q/8-K
python research/sec_monitor.py AVGO NVDA DRAM

# also print latest 10-K Item 1A Risk Factors
python research/sec_monitor.py AVGO --risk

# bulk-download all 10-Ks to ./sec-edgar-filings/ for offline reading
python research/sec_monitor.py AVGO NVDA --forms 10-K --limit 3 --download
```

Sets `EDGAR_IDENTITY` from env (`SEC_EDGAR_USER_AGENT`) — SEC requires
a UA header identifying you.

### 2. `options_backtest.py`  —  Weekly short-put strategy backtest
*Lesson: would systematically selling weekly puts on DRAM (or AVGO, SPY, …)
have actually been profitable? Find out **before** placing the first trade.*

```bash
# Default: 2-year backtest, K = round(spot * 0.98), weekly DTE, 10-day hold
python research/options_backtest.py DRAM

# More conservative strike, 1 year
python research/options_backtest.py SPY --strike-pct 0.95 --years 1
```

Outputs:
- `reports/<TICKER>_short_put_<years>y.csv` — per-trade log
- `reports/<TICKER>_short_put_<years>y.html` — QuantStats tearsheet
- Terminal summary: total PnL, # expires/assigns, vs buy-and-hold

⚠️ **Caveats**
- Premiums are estimated via Black-Scholes using realized 30d vol, not actual
  historical option EOD prices. P&L magnitude is approximate; the *shape* of
  the equity curve and assignment frequency are realistic.
- For exact premiums, swap in Polygon `/v2/aggs/.../1/day/...` via
  `qqq_put_real.py` (already in repo).

### 3. `realtime_options.py`  —  Live options snapshot with fallback
*Lesson: yfinance returned $0 bid/ask + 0% IV for DRAM 8/21 puts today.
We had to guess. This script tries Polygon first when configured.*

```bash
python research/realtime_options.py AVGO
python research/realtime_options.py DRAM --expiries 2026-07-17 2026-08-21
```

Source priority:
1. Polygon via `PROXY_URL` + `PROXY_KEY` (best — real IV/OI/Greeks)
2. Polygon direct via `POLYGON_API_KEY`
3. yfinance (fallback — IV/OI may be zero for small ETF options)

Prints ATM straddle implied move, 25Δ skew, top-OI strikes near spot.

## Environment variables

```bash
export SEC_EDGAR_USER_AGENT="Your Name your@email.com"   # SEC requirement
export PROXY_URL="http://43.206.151.58:8080"             # your Polygon proxy
export PROXY_KEY="..."                                    # X-Proxy-Key header
export POLYGON_API_KEY="..."                              # optional direct fallback
```

## What this workflow does NOT do

- No automated trading (Robinhood MCP is a separate tool, not invoked here)
- No live alerts / cron monitoring (sandbox is ephemeral)
- No machine-learning models (use Qlib/FinRL for that — heavier install)
