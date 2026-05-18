"""Backtest: sell 1-day 2% OTM QQQ put each day. On breach (close <= strike),
get assigned, hold QQQ for 10 trading days, then resume selling puts.

Premium pricing:
  Black-Scholes using *market-implied* volatility. We use VXN (CBOE Nasdaq-100
  Volatility Index, the IV equivalent of VIX for NDX/QQQ) as the 30-day ATM IV
  base, with a `SKEW_MULT` knob to scale up to the 0DTE 2% OTM put IV
  (which trades richer due to skew + gamma). Reported across a sweep of skew
  multipliers since we don't have a real 0DTE chain.

Capital base: strike price each day (cash-secured put). Returns reported on
that basis.
"""
import math
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
from math import log, sqrt, exp, erf

END = date(2026, 5, 18)
START = END - timedelta(days=400)
HOLD_DAYS = 10
STRIKE_PCT = 0.98   # 2% OTM
RFR = 0.04

# --- price + IV data ---
qqq = yf.download("QQQ", start=START.isoformat(), end=END.isoformat(),
                  auto_adjust=False, progress=False)
if isinstance(qqq.columns, pd.MultiIndex):
    qqq.columns = qqq.columns.get_level_values(0)
qqq = qqq[["Open", "High", "Low", "Close"]].copy()

vxn = yf.download("^VXN", start=START.isoformat(), end=END.isoformat(),
                  auto_adjust=False, progress=False)
if isinstance(vxn.columns, pd.MultiIndex):
    vxn.columns = vxn.columns.get_level_values(0)
qqq["VXN"] = vxn["Close"] / 100.0   # VXN is quoted *100

qqq["PrevClose"] = qqq["Close"].shift(1)
qqq["PrevVXN"] = qqq["VXN"].shift(1)
qqq["LogRet"] = np.log(qqq["Close"] / qqq["PrevClose"])
qqq["RV30"] = qqq["LogRet"].rolling(30).std() * math.sqrt(252)
qqq = qqq.dropna(subset=["PrevClose", "PrevVXN", "RV30"])

# --- BS put pricer (1-day expiry) ---
def norm_cdf(x): return 0.5 * (1 + erf(x / sqrt(2)))

def bs_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return max(K - S, 0)
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return K * exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

# --- trim to 1-yr window ---
one_year_ago = END - timedelta(days=365)
df = qqq[qqq.index.date >= one_year_ago].copy()
n = len(df)
print(f"Window: {df.index.min().date()} → {df.index.max().date()}  ({n} trading days)")
print(f"VXN mean over window: {df['VXN'].mean()*100:.2f}   "
      f"min: {df['VXN'].min()*100:.2f}   max: {df['VXN'].max()*100:.2f}\n")

# --- simulator ---
def simulate(premium_fn, label):
    pnl = 0.0
    notional_sum = 0.0
    days_put = days_hold = breaches = 0
    hold_remaining = 0
    last_close = None
    premia = []
    for _, row in df.iterrows():
        prev_close = row["PrevClose"]
        close = row["Close"]
        iv = row["PrevVXN"]
        if hold_remaining > 0:
            pnl += close - last_close
            last_close = close
            days_hold += 1
            hold_remaining -= 1
        else:
            strike = STRIKE_PCT * prev_close
            premium = premium_fn(prev_close, iv)
            premia.append(premium)
            notional_sum += strike
            days_put += 1
            if close <= strike:
                pnl += premium - (strike - close)
                breaches += 1
                hold_remaining = HOLD_DAYS
                last_close = close
            else:
                pnl += premium
    avg_K = notional_sum / max(days_put, 1)
    avg_prem_pct = (sum(premia) / len(premia)) / df["PrevClose"].mean() * 100 if premia else 0
    return dict(label=label, days_put=days_put, days_hold=days_hold,
                breaches=breaches, pnl=pnl, avg_K=avg_K,
                ret_pct=pnl / avg_K * 100, avg_prem_pct=avg_prem_pct)

# --- premium models ---
def premium_vxn(skew):
    return lambda S, iv: bs_put(S, STRIKE_PCT * S, T=1/252, r=RFR, sigma=iv * skew)

models = []
for skew in (1.0, 1.3, 1.5, 1.8, 2.0):
    models.append(simulate(premium_vxn(skew), f"VXN IV × {skew:.1f}"))
# also keep a realized-vol baseline for comparison
def premium_rv(S, iv_unused):
    return bs_put(S, STRIKE_PCT * S, T=1/252, r=RFR, sigma=None)
# RV-based uses df["RV30"] - re-implement inline:
def simulate_rv():
    pnl = 0.0; notional_sum = 0.0
    days_put = days_hold = breaches = 0
    hold_remaining = 0; last_close = None; premia = []
    for _, row in df.iterrows():
        prev_close = row["PrevClose"]; close = row["Close"]
        iv = row["RV30"] * 1.2
        if hold_remaining > 0:
            pnl += close - last_close; last_close = close
            days_hold += 1; hold_remaining -= 1
        else:
            strike = STRIKE_PCT * prev_close
            premium = bs_put(prev_close, strike, 1/252, RFR, iv)
            premia.append(premium); notional_sum += strike; days_put += 1
            if close <= strike:
                pnl += premium - (strike - close); breaches += 1
                hold_remaining = HOLD_DAYS; last_close = close
            else:
                pnl += premium
    avg_K = notional_sum / max(days_put, 1)
    avg_prem_pct = (sum(premia) / len(premia)) / df["PrevClose"].mean() * 100
    return dict(label="RV30 × 1.2 (baseline)", days_put=days_put,
                days_hold=days_hold, breaches=breaches, pnl=pnl, avg_K=avg_K,
                ret_pct=pnl / avg_K * 100, avg_prem_pct=avg_prem_pct)
models.insert(0, simulate_rv())

# --- output ---
print(f"{'IV model':<24}{'Puts':>6}{'Hold':>6}{'Brch':>6}"
      f"{'AvgPrem%':>11}{'P&L/sh':>10}{'AvgK':>9}{'Return':>10}")
print("-" * 82)
for m in models:
    print(f"{m['label']:<24}{m['days_put']:>6}{m['days_hold']:>6}{m['breaches']:>6}"
          f"{m['avg_prem_pct']:>10.3f}%{m['pnl']:>10.2f}{m['avg_K']:>9.2f}"
          f"{m['ret_pct']:>9.2f}%")

bh = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
print(f"\nBuy-and-hold QQQ: {bh:+.2f}% "
      f"({df['Close'].iloc[0]:.2f} → {df['Close'].iloc[-1]:.2f})")
