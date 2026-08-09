#!/usr/bin/env python3
"""
VIX threshold-crossing backtest on QQQ.

Strategy / signal
-----------------
For each threshold T in {17, 18, 19, 20, 21, 22}:
  A signal fires on day D when:
    * VIX High on day D-1 (previous trading day) < T
    * VIX High on day D              (current day) >= T
  On a signal, buy QQQ at day D's close.
  Measure the forward return of QQQ close over the next 5 / 20 / 60 trading days.

For each threshold and horizon we report:
  sample count, mean, median, win rate, best, worst.

Benchmark: QQQ buy & hold average forward return over the same 5/20/60-day
horizons, computed over every trading day (so the strategy's edge can be
compared against "just being in the market on a random day").

Data: Yahoo Finance daily OHLC (^VIX and QQQ).
"""

import io
import json
import sys

import numpy as np
import pandas as pd
import requests

CA_BUNDLE = "/root/.ccr/ca-bundle.crt"
THRESHOLDS = [17, 18, 19, 20, 21, 22]
HORIZONS = [5, 20, 60]
HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch_yahoo(symbol):
    """Return a DataFrame indexed by date with OHLC columns for a Yahoo symbol."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"period1": 0, "period2": 9999999999, "interval": "1d"}
    r = requests.get(url, params=params, headers=HEADERS,
                     timeout=60, verify=CA_BUNDLE)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({
        "open": q["open"],
        "high": q["high"],
        "low": q["low"],
        "close": q["close"],
    }, index=pd.to_datetime(ts, unit="s").normalize())
    df.index.name = "date"
    df = df.dropna(subset=["high", "close"])
    return df


def main():
    print("Fetching data from Yahoo Finance ...", file=sys.stderr)
    vix = fetch_yahoo("%5EVIX")   # ^VIX
    qqq = fetch_yahoo("QQQ")

    # Align to the common trading calendar (QQQ starts 1999-03-10).
    df = pd.DataFrame({
        "vix_high": vix["high"],
        "qqq_close": qqq["close"],
    }).dropna()
    df = df.sort_index()

    df["vix_high_prev"] = df["vix_high"].shift(1)

    # Forward returns of QQQ close for each horizon.
    for h in HORIZONS:
        df[f"fwd_{h}"] = df["qqq_close"].shift(-h) / df["qqq_close"] - 1.0

    n_days = len(df)
    span = f"{df.index[0].date()} to {df.index[-1].date()}"
    print(f"Aligned trading days: {n_days}  ({span})\n")

    # ---- Buy & Hold benchmark (every trading day) ----
    bh = {}
    for h in HORIZONS:
        s = df[f"fwd_{h}"].dropna()
        bh[h] = {
            "n": int(s.shape[0]),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "win": float((s > 0).mean()),
            "best": float(s.max()),
            "worst": float(s.min()),
        }

    # ---- Strategy per threshold ----
    results = {}
    for T in THRESHOLDS:
        signal = (df["vix_high_prev"] < T) & (df["vix_high"] >= T)
        sig = df[signal]
        per_h = {}
        for h in HORIZONS:
            s = sig[f"fwd_{h}"].dropna()
            if s.empty:
                per_h[h] = None
                continue
            per_h[h] = {
                "n": int(s.shape[0]),
                "mean": float(s.mean()),
                "median": float(s.median()),
                "win": float((s > 0).mean()),
                "best": float(s.max()),
                "worst": float(s.min()),
            }
        results[T] = {
            "total_signals": int(signal.sum()),
            "horizons": per_h,
            "signal_dates": [d.strftime("%Y-%m-%d") for d in sig.index],
        }

    out = {
        "data_span": span,
        "trading_days": n_days,
        "thresholds": THRESHOLDS,
        "horizons": HORIZONS,
        "buy_and_hold": bh,
        "strategy": results,
    }

    with open("vix_qqq_results.json", "w") as f:
        json.dump(out, f, indent=2)

    # ---- Pretty print ----
    def pct(x):
        return f"{x * 100:+6.2f}%"

    print("=" * 78)
    print("QQQ BUY & HOLD BENCHMARK (average forward return over ALL trading days)")
    print("=" * 78)
    print(f"{'Horizon':>8} {'N':>7} {'Mean':>9} {'Median':>9} {'Win%':>7} "
          f"{'Best':>9} {'Worst':>9}")
    for h in HORIZONS:
        b = bh[h]
        print(f"{h:>6}d {b['n']:>7} {pct(b['mean']):>9} {pct(b['median']):>9} "
              f"{b['win'] * 100:>6.1f}% {pct(b['best']):>9} {pct(b['worst']):>9}")

    for T in THRESHOLDS:
        r = results[T]
        print("\n" + "=" * 78)
        print(f"THRESHOLD {T}   (VIX High crosses up through {T}: "
              f"prev High < {T} and today High >= {T})")
        print(f"Total signal days: {r['total_signals']}")
        print("=" * 78)
        print(f"{'Horizon':>8} {'N':>7} {'Mean':>9} {'Median':>9} {'Win%':>7} "
              f"{'Best':>9} {'Worst':>9} {'  vs B&H(mean)':>14}")
        for h in HORIZONS:
            d = r["horizons"][h]
            if d is None:
                print(f"{h:>6}d      -  (no completed samples)")
                continue
            edge = d["mean"] - bh[h]["mean"]
            print(f"{h:>6}d {d['n']:>7} {pct(d['mean']):>9} {pct(d['median']):>9} "
                  f"{d['win'] * 100:>6.1f}% {pct(d['best']):>9} {pct(d['worst']):>9} "
                  f"{pct(edge):>14}")

    print("\nSaved detailed results (incl. signal dates) to vix_qqq_results.json")


if __name__ == "__main__":
    main()
