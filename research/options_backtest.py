#!/usr/bin/env python3
"""
Short-put strategy backtest — directly inspired by today's DRAM lesson.

Strategy under test
-------------------
At every Friday close, sell a 1-week put at `spot * strike_pct`.
At next Friday close:
  - put expires worthless  -> keep premium
  - put assigned (S<K)     -> bought at K, hold `hold_days`, then sell
  - while assigned, no new put is sold

Premium uses a Black-Scholes estimate with realized 30-day vol
(because retrieving 1 year of real EOD option prices for a small ETF
is API-heavy; this is an approximation for *strategy* P&L, not exact
historical fills).

Output
------
- per-trade CSV (date, action, K, premium, S_next, pnl, cum_pnl)
- QuantStats HTML tearsheet ./reports/<ticker>_short_put_<years>y.html
- Terminal summary vs buy-and-hold

Usage
-----
    python options_backtest.py DRAM
    python options_backtest.py DRAM --strike-pct 0.95 --years 2
    python options_backtest.py SPY --strike-pct 0.98 --hold-days 10
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# Silence matplotlib font warnings (sandbox has no Arial) and noisy libs
logging.getLogger("matplotlib").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Black-Scholes put pricing (for premium estimation only)
# --------------------------------------------------------------------------- #
def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _phi(-d2) - S * _phi(-d1)


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
def run_backtest(ticker: str, years: float, strike_pct: float,
                 hold_days: int, dte_days: int, risk_free: float) -> dict:
    end = date.today()
    start = end - timedelta(days=int(365 * (years + 0.25)))
    df = yf.download(ticker, start=start.isoformat(),
                     end=(end + timedelta(days=1)).isoformat(),
                     auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).date
    if len(df) < 40:
        raise RuntimeError(f"insufficient data for {ticker} ({len(df)} bars)")

    close = df["Close"].astype(float)
    log_ret = np.log(close / close.shift(1)).dropna()

    # rolling 30-day annualized vol (used as the IV proxy)
    rv30 = log_ret.rolling(30).std() * np.sqrt(252)

    # Restrict to the requested window
    target_start = end - timedelta(days=int(365 * years))
    start_idx = next((i for i, d in enumerate(close.index)
                      if d >= target_start), None)
    # If history doesn't reach back that far, use earliest available + min lookback
    if start_idx is None or start_idx < 30:
        start_idx = min(30, len(close) - 10)

    rows = []
    held = False
    basis = 0.0
    held_until = -1
    cum = 0.0

    dates = list(close.index)
    for i in range(start_idx, len(dates) - 1):
        d0 = dates[i]
        d1 = dates[min(i + dte_days, len(dates) - 1)]
        S0 = float(close.iloc[i])
        S1 = float(close.iloc[min(i + dte_days, len(dates) - 1)])

        if held:
            if i >= held_until:
                realized = (S0 - basis) * 100  # 1 contract
                cum += realized
                rows.append({"date": d0, "action": "SELL_STK",
                             "K": basis, "premium": 0,
                             "S_next": S0, "pnl": round(realized, 2),
                             "cum_pnl": round(cum, 2)})
                held = False
            else:
                rows.append({"date": d0, "action": "HOLD",
                             "K": basis, "premium": 0,
                             "S_next": S0, "pnl": 0,
                             "cum_pnl": round(cum, 2)})
            continue

        sigma = float(rv30.iloc[i]) if not np.isnan(rv30.iloc[i]) else 0.40
        K = round(S0 * strike_pct, 0)
        T = dte_days / 252
        prem = bs_put(S0, K, T, risk_free, sigma)
        prem_cash = prem * 100

        if S1 > K:
            cum += prem_cash
            rows.append({"date": d1, "action": "EXPIRE",
                         "K": K, "premium": round(prem, 2),
                         "S_next": S1, "pnl": round(prem_cash, 2),
                         "cum_pnl": round(cum, 2)})
        else:
            cum += prem_cash
            held = True
            basis = K
            held_until = min(i + dte_days + hold_days, len(dates) - 1)
            rows.append({"date": d1, "action": "ASSIGN",
                         "K": K, "premium": round(prem, 2),
                         "S_next": S1, "pnl": round(prem_cash, 2),
                         "cum_pnl": round(cum, 2)})

    trades = pd.DataFrame(rows)
    if trades.empty:
        return {"trades": trades, "summary": "no trades"}

    notional = float(close.iloc[start_idx]) * 100
    final_cum = float(trades["cum_pnl"].iloc[-1])
    bh = (float(close.iloc[-1]) / float(close.iloc[start_idx]) - 1) * 100

    n_expire = (trades.action == "EXPIRE").sum()
    n_assign = (trades.action == "ASSIGN").sum()

    summary = {
        "ticker": ticker,
        "window": f"{close.index[start_idx]} -> {close.index[-1]}",
        "trading_days": len(close) - start_idx,
        "trades_opened": int(n_expire + n_assign),
        "expired_worthless": int(n_expire),
        "assigned": int(n_assign),
        "total_pnl_dollars": round(final_cum, 2),
        "notional_dollars": round(notional, 2),
        "strategy_return_pct": round(final_cum / notional * 100, 2),
        "buy_and_hold_pct": round(bh, 2),
        "alpha_pct": round(final_cum / notional * 100 - bh, 2),
    }
    return {"trades": trades, "summary": summary,
            "equity_curve": _equity_curve(trades, notional)}


def _equity_curve(trades: pd.DataFrame, notional: float) -> pd.Series:
    s = trades.set_index("date")["cum_pnl"].astype(float)
    s.index = pd.to_datetime(s.index)
    s = s[~s.index.duplicated(keep="last")]
    daily = s.reindex(pd.date_range(s.index.min(), s.index.max(),
                                    freq="B"), method="ffill").fillna(0)
    return (daily / notional)  # return series


def write_quantstats_report(returns: pd.Series, ticker: str,
                            out_path: str) -> None:
    try:
        import quantstats as qs
    except ImportError:
        print("  quantstats not installed — skipping HTML report")
        return
    # convert cumulative-return series to daily returns
    daily = returns.diff().fillna(returns.iloc[0])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    qs.reports.html(daily, title=f"{ticker} Weekly Short-Put Strategy",
                    output=out_path, download_filename=out_path)
    print(f"  ✓ wrote {out_path}")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker")
    ap.add_argument("--years", type=float, default=2.0)
    ap.add_argument("--strike-pct", type=float, default=0.98,
                    help="K = round(spot * strike_pct)")
    ap.add_argument("--hold-days", type=int, default=10,
                    help="trading days to hold assigned shares")
    ap.add_argument("--dte-days", type=int, default=5,
                    help="trading days to expiry (weekly = 5)")
    ap.add_argument("--rfr", type=float, default=0.05,
                    help="annualized risk-free rate")
    ap.add_argument("--out-dir", default="reports")
    args = ap.parse_args()

    print(f"Backtesting {args.ticker.upper()} weekly short-put "
          f"@ {args.strike_pct*100:.0f}% strike, {args.years}y window")
    res = run_backtest(args.ticker.upper(), args.years,
                       args.strike_pct, args.hold_days, args.dte_days,
                       args.rfr)
    summary = res["summary"]
    trades = res["trades"]
    print()
    for k, v in summary.items():
        print(f"  {k:<22}: {v}")
    print()

    Path(args.out_dir).mkdir(exist_ok=True)
    csv_path = Path(args.out_dir) / \
        f"{args.ticker}_short_put_{args.years}y.csv"
    trades.to_csv(csv_path, index=False)
    print(f"  ✓ trades -> {csv_path}")

    html_path = Path(args.out_dir) / \
        f"{args.ticker}_short_put_{args.years}y.html"
    write_quantstats_report(res["equity_curve"], args.ticker,
                            str(html_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
