"""Backtest: sell 1-day 2% OTM QQQ put each day. On breach (close <= strike),
get assigned, hold QQQ for 10 trading days, then resume selling puts.

Premium models:
  (A) Black-Scholes with 30-day realized vol * 1.2 (rough vol-risk-premium)
  (B) Constant premium = X% of spot per day, swept across a range.

Notional / capital base: strike price each day (cash-secured put). Returns
reported on that basis.
"""
import math
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, timedelta

END = date(2026, 5, 18)
START = END - timedelta(days=400)
HOLD_DAYS = 10
STRIKE_PCT = 0.98   # 2% OTM
RFR = 0.04          # short-rate assumption for BS (effect is tiny at T=1d)
VRP_MULT = 1.2      # implied vol = realized vol * this multiplier

# ------------ data ------------
df = yf.download("QQQ", start=START.isoformat(), end=END.isoformat(),
                 auto_adjust=False, progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df = df[["Open", "High", "Low", "Close"]].copy()
df["PrevClose"] = df["Close"].shift(1)
df["LogRet"] = np.log(df["Close"] / df["PrevClose"])
df["RV30"] = df["LogRet"].rolling(30).std() * math.sqrt(252)  # annualised
df = df.dropna(subset=["PrevClose", "RV30"])

# ------------ BS put pricer (1-day expiry, T=1/252) ------------
from math import log, sqrt, exp, erf
def norm_cdf(x): return 0.5 * (1 + erf(x / sqrt(2)))

def bs_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return max(K - S, 0)
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return K * exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

# ------------ filter to the trailing 1-year window ------------
one_year_ago = END - timedelta(days=365)
df = df[df.index.date >= one_year_ago].copy()
n = len(df)
print(f"Window: {df.index.min().date()} → {df.index.max().date()}   ({n} trading days)\n")

# ------------ simulation engine ------------
def simulate(premium_fn, label):
    """premium_fn(spot, iv_annual) -> per-share premium in dollars.
    iv_annual may be None if not used.
    """
    pnl_per_share = 0.0
    notional_used = 0.0      # sum of strikes used as denominator
    days_put = 0
    days_hold = 0
    breaches = 0

    hold_remaining = 0
    hold_entry_close = None   # close on the assignment day (cost basis after MTM)
    last_close = None

    for ts, row in df.iterrows():
        prev_close = row["PrevClose"]
        close = row["Close"]
        iv = row["RV30"] * VRP_MULT

        if hold_remaining > 0:
            # We own 1 share. P&L = today's close - last close (1-day MTM).
            pnl_per_share += close - last_close
            days_hold += 1
            hold_remaining -= 1
            if hold_remaining == 0:
                # exit at today's close; nothing extra to add (already MTM'd)
                pass
            last_close = close
        else:
            # Sell a 0DTE put at strike 0.98 * prev_close. Spot for pricing = prev_close
            strike = STRIKE_PCT * prev_close
            premium = premium_fn(prev_close, iv)
            notional_used += strike
            days_put += 1
            # Outcome at today's close:
            if close <= strike:
                # Assigned: we receive premium, owe (strike - close)
                pnl_per_share += premium - (strike - close)
                breaches += 1
                # Now we own a share at "effective cost" = strike, currently worth `close`.
                # Begin 10-trading-day hold of QQQ from today's close.
                hold_remaining = HOLD_DAYS
                last_close = close
            else:
                pnl_per_share += premium

    # If still holding at the end of the window, mark to last available close (already done daily).
    avg_notional = notional_used / max(days_put, 1)
    # Total return: total P&L / avg strike (rough single-share basis)
    total_return_pct = pnl_per_share / avg_notional * 100
    return {
        "label": label,
        "days_put": days_put,
        "days_hold": days_hold,
        "breaches": breaches,
        "pnl_per_share": pnl_per_share,
        "avg_notional": avg_notional,
        "total_return_pct": total_return_pct,
    }

# ------------ premium models ------------
def premium_bs(S, iv):
    return bs_put(S, STRIKE_PCT * S, T=1/252, r=RFR, sigma=iv)

def premium_const_pct(pct):
    return lambda S, iv: S * pct

results = []
results.append(simulate(premium_bs, f"BS (RV30 * {VRP_MULT})"))
for pct in (0.0005, 0.0010, 0.0015, 0.0020, 0.0030):
    results.append(simulate(premium_const_pct(pct), f"const {pct*100:.2f}% of spot"))

print(f"{'Model':<28}{'Puts':>6}{'Hold':>6}{'Breach':>8}"
      f"{'P&L/sh ($)':>14}{'AvgK ($)':>12}{'Return %':>12}")
print("-" * 86)
for r in results:
    print(f"{r['label']:<28}{r['days_put']:>6}{r['days_hold']:>6}{r['breaches']:>8}"
          f"{r['pnl_per_share']:>14.2f}{r['avg_notional']:>12.2f}{r['total_return_pct']:>11.2f}%")

# ------------ buy-and-hold benchmark ------------
first_close = df["Close"].iloc[0]
last_close = df["Close"].iloc[-1]
bh = (last_close / first_close - 1) * 100
print(f"\nBuy-and-hold QQQ over same window: {bh:+.2f}%   "
      f"({first_close:.2f} → {last_close:.2f})")
