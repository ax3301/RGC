"""QQQ put-selling backtest with REAL historical option prices from Massive API.

Strategy:
  Each trading day T-1 at close, sell a put expiring T (1-day to expiry) with
  strike ≈ 0.98 * close(T-1) (rounded to nearest $1 since QQQ strikes are $1).
  Premium = put's EOD close price on T-1.
  If QQQ close(T) <= strike: assigned, hold 10 trading days, then resume.

Requires env: POLYGON_API_KEY (or hard-coded fallback for this session).
"""
import os, math, json, time
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from datetime import date, timedelta
import yfinance as yf
import pandas as pd
import numpy as np

API = os.environ.get("POLYGON_API_KEY", "MhDRBDpwT6rNOWzyMN0uuiYBewlgYfhp")
BASE = "https://api.massive.com"
END = date(2026, 5, 18)
START = END - timedelta(days=400)
HOLD_DAYS = 10
STRIKE_PCT = 0.98

# ---------- pull QQQ EOD ----------
qqq = yf.download("QQQ", start=START.isoformat(), end=END.isoformat(),
                  auto_adjust=False, progress=False)
if isinstance(qqq.columns, pd.MultiIndex):
    qqq.columns = qqq.columns.get_level_values(0)
qqq = qqq[["Open", "High", "Low", "Close"]].copy()
qqq["PrevClose"] = qqq["Close"].shift(1)
qqq = qqq.dropna()
one_year_ago = END - timedelta(days=365)
qqq = qqq[qqq.index.date >= one_year_ago]
print(f"QQQ window: {qqq.index[0].date()} → {qqq.index[-1].date()}  ({len(qqq)} days)")

# ---------- option ticker helper ----------
def occ_put_ticker(expiry: date, strike_dollars: float) -> str:
    s = f"{int(round(strike_dollars * 1000)):08d}"
    return f"O:QQQ{expiry.strftime('%y%m%d')}P{s}"

def fetch_eod(ticker: str, sell_date: date) -> float | None:
    """Get close price of the option contract on `sell_date`."""
    url = (f"{BASE}/v2/aggs/ticker/{ticker}/range/1/day/"
           f"{sell_date.isoformat()}/{sell_date.isoformat()}?apiKey={API}")
    try:
        with urlopen(Request(url), timeout=20) as r:
            data = json.load(r)
    except Exception as e:
        return None
    if data.get("resultsCount", 0) == 0:
        return None
    return float(data["results"][0]["c"])

# ---------- map trade-date → next-trade-date for 1DTE expiry ----------
dates = list(qqq.index.date)
next_dt = {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}

# ---------- pull premiums ----------
records = []
miss = 0
for i, (ts, row) in enumerate(qqq.iterrows()):
    sell_d = ts.date()
    exp = next_dt.get(sell_d)
    if exp is None:
        continue
    strike = round(STRIKE_PCT * float(row["Close"]))   # nearest $1
    # try the round-dollar strike; if missing, try +/-1
    premium = None
    used_strike = None
    for adj in (0, -1, 1, -2, 2):
        k = strike + adj
        tk = occ_put_ticker(exp, k)
        p = fetch_eod(tk, sell_d)
        if p is not None:
            premium = p
            used_strike = k
            break
    if premium is None:
        miss += 1
    records.append({"sell_date": sell_d, "expiry": exp,
                    "prev_close": float(row["Close"]),
                    "strike": used_strike, "premium": premium})

print(f"Premium lookups: {len(records)}   missing: {miss}")
prem_df = pd.DataFrame(records).set_index("sell_date")
prem_df.to_csv("/tmp/qqq_put_premiums.csv")

# ---------- simulate ----------
# Walk forward: each day, we are either selling (using premium captured T-1)
# or holding QQQ from a prior assignment.
hold_remaining = 0
last_close = None
pnl = 0.0
notional_sum = 0.0
days_put = days_hold = breaches = 0
log_rows = []
prev_idx = None

for ts in qqq.index:
    d = ts.date()
    close = float(qqq.loc[ts, "Close"])
    if hold_remaining > 0:
        day_pnl = close - last_close
        pnl += day_pnl
        log_rows.append((d, "HOLD", None, None, close, day_pnl, pnl))
        last_close = close
        days_hold += 1
        hold_remaining -= 1
        continue
    # Selling: premium was set on the prior trading day (sell_date = prev_idx)
    if prev_idx is None:
        prev_idx = d
        continue
    rec = prem_df.loc[prev_idx] if prev_idx in prem_df.index else None
    if rec is None or pd.isna(rec.get("premium")) or rec.get("strike") is None:
        prev_idx = d
        continue
    strike = float(rec["strike"])
    premium = float(rec["premium"])
    notional_sum += strike
    days_put += 1
    if close <= strike:
        day_pnl = premium - (strike - close)
        pnl += day_pnl
        breaches += 1
        hold_remaining = HOLD_DAYS
        last_close = close
        log_rows.append((d, "ASSIGN", strike, premium, close, day_pnl, pnl))
    else:
        pnl += premium
        log_rows.append((d, "EXPIRE", strike, premium, close, premium, pnl))
    prev_idx = d

avg_K = notional_sum / max(days_put, 1)
ret_pct = pnl / avg_K * 100
bh = (qqq["Close"].iloc[-1] / qqq["Close"].iloc[0] - 1) * 100

print(f"\n=== RESULTS (real Massive option prices) ===")
print(f"Days selling puts: {days_put}")
print(f"Days holding QQQ:  {days_hold}")
print(f"Assignments:       {breaches}")
print(f"Total P&L / share: ${pnl:.2f}")
print(f"Avg strike (cash-secured base): ${avg_K:.2f}")
print(f"Return on cash-secured base:    {ret_pct:+.2f}%")
print(f"\nBuy-and-hold QQQ same window:   {bh:+.2f}% "
      f"({qqq['Close'].iloc[0]:.2f} → {qqq['Close'].iloc[-1]:.2f})")

# Save detailed log
log = pd.DataFrame(log_rows, columns=["date", "action", "strike",
                                       "premium", "close", "day_pnl", "cum_pnl"])
log.to_csv("/tmp/qqq_put_log.csv", index=False)

# Show assignment days
print("\n--- Assignment days ---")
print(log[log["action"] == "ASSIGN"].to_string(index=False))

# Premium stats
prem_clean = prem_df.dropna(subset=["premium"])
print(f"\n--- Premium stats ({len(prem_clean)} sell days) ---")
print(f"Mean premium:   ${prem_clean['premium'].mean():.3f} "
      f"({prem_clean['premium'].mean() / prem_clean['prev_close'].mean() * 100:.3f}% of spot)")
print(f"Median premium: ${prem_clean['premium'].median():.3f}")
print(f"Max premium:    ${prem_clean['premium'].max():.3f}")
print(f"Min premium:    ${prem_clean['premium'].min():.3f}")
