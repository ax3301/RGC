#!/usr/bin/env python3
"""
Compare several long/flat QQQ trading strategies against buy & hold.

Execution model (no look-ahead):
  * A strategy outputs a target position (1 = long QQQ, 0 = cash) that is
    DECIDED using data up to and including day t's close.
  * That position earns the QQQ close-to-close return from day t to t+1.
  * A round-trip costs COST per side (applied when the position changes).

Metrics per strategy:
  total return, CAGR, annual vol, Sharpe (rf=0), max drawdown,
  exposure (% of days invested), number of trades, average trades/year.

Strategies:
  BH   Buy & Hold
  SMA  200-day trend filter: long when Close > SMA200 else cash
  RSI2 Connors-style mean reversion: (Close>SMA200) & RSI(2)<10 -> long,
       exit when Close>SMA5 or RSI(2)>70  (frequent, short holds)
  DIP  Above SMA200, buy after 3 consecutive down days, exit on first up day
  VIXR VIX spike dip-buy: above SMA200, enter when VIX High >= 1.10 * prior
       VIX close, hold 10 trading days (overlaps extend the hold)
"""

import sys
import numpy as np
import pandas as pd
import requests

CA_BUNDLE = "/root/.ccr/ca-bundle.crt"
HEADERS = {"User-Agent": "Mozilla/5.0"}
COST = 0.0002          # 0.02% per side
TRADING_DAYS = 252


def fetch_yahoo(symbol):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"period1": 0, "period2": 9999999999, "interval": "1d"}
    r = requests.get(url, params=params, headers=HEADERS, timeout=60, verify=CA_BUNDLE)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame(
        {"open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"]},
        index=pd.to_datetime(res["timestamp"], unit="s").normalize(),
    ).dropna(subset=["high", "close"])
    df.index.name = "date"
    return df


def rsi(series, n):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = (-delta.clip(upper=0)).rolling(n).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build(df):
    d = pd.DataFrame(index=df.index)
    d["close"] = df["qqq_close"]
    d["ret"] = d["close"].pct_change()
    d["sma200"] = d["close"].rolling(200).mean()
    d["sma5"] = d["close"].rolling(5).mean()
    d["rsi2"] = rsi(d["close"], 2)
    d["up"] = d["close"].diff() > 0
    d["vix_high"] = df["vix_high"]
    d["vix_close"] = df["vix_close"]
    return d


# --- position generators: return a 0/1 Series aligned to d.index ---

def pos_bh(d):
    return pd.Series(1.0, index=d.index)


def pos_sma(d):
    return (d["close"] > d["sma200"]).astype(float)


def pos_rsi2(d):
    above = d["close"] > d["sma200"]
    pos = np.zeros(len(d))
    invested = False
    r2 = d["rsi2"].values
    c = d["close"].values
    s5 = d["sma5"].values
    ab = above.values
    for i in range(len(d)):
        if invested:
            if c[i] > s5[i] or r2[i] > 70:
                invested = False
        else:
            if ab[i] and r2[i] < 10:
                invested = True
        pos[i] = 1.0 if invested else 0.0
    return pd.Series(pos, index=d.index)


def pos_dip(d):
    above = d["close"] > d["sma200"]
    down = ~d["up"]
    three_down = down & down.shift(1) & down.shift(2)
    pos = np.zeros(len(d))
    invested = False
    td = three_down.fillna(False).values
    upv = d["up"].values
    ab = above.values
    for i in range(len(d)):
        if invested:
            if upv[i]:
                invested = False
        else:
            if ab[i] and td[i]:
                invested = True
        pos[i] = 1.0 if invested else 0.0
    return pd.Series(pos, index=d.index)


def pos_vixr(d, hold=10):
    above = d["close"] > d["sma200"]
    spike = d["vix_high"] >= 1.10 * d["vix_close"].shift(1)
    entry = (above & spike).fillna(False).values
    pos = np.zeros(len(d))
    remaining = 0
    for i in range(len(d)):
        if entry[i]:
            remaining = hold
        if remaining > 0:
            pos[i] = 1.0
            remaining -= 1
    return pd.Series(pos, index=d.index)


STRATS = {
    "BH": pos_bh,
    "SMA": pos_sma,
    "RSI2": pos_rsi2,
    "DIP": pos_dip,
    "VIXR": pos_vixr,
}


def evaluate(d, pos):
    # position decided at t's close earns t->t+1 return: shift position by 1.
    held = pos.shift(1).fillna(0.0)
    trades_series = pos.diff().abs().fillna(0.0)  # 1 each time exposure flips
    cost = trades_series * COST
    strat_ret = held * d["ret"].fillna(0.0) - cost
    equity = (1 + strat_ret).cumprod()

    n = len(d)
    years = n / TRADING_DAYS
    total = equity.iloc[-1] - 1
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = strat_ret.std() * np.sqrt(TRADING_DAYS)
    sharpe = (strat_ret.mean() * TRADING_DAYS) / vol if vol > 0 else np.nan
    dd = (equity / equity.cummax() - 1).min()
    exposure = held.mean()
    # count entries (0->1) as trades
    entries = int(((pos > 0) & (pos.shift(1).fillna(0) == 0)).sum())
    return {
        "total": total, "cagr": cagr, "vol": vol, "sharpe": sharpe,
        "maxdd": dd, "exposure": exposure, "trades": entries,
        "trades_yr": entries / years if years > 0 else np.nan,
        "equity": equity,
    }


def run(d, label):
    print("=" * 92)
    print(f"{label}   ({d.index[0].date()} -> {d.index[-1].date()}, "
          f"{len(d)} trading days)")
    print("=" * 92)
    hdr = (f"{'Strat':>5} {'TotRet':>9} {'CAGR':>8} {'Vol':>7} {'Sharpe':>7} "
           f"{'MaxDD':>8} {'Expo':>6} {'Trades':>7} {'Tr/yr':>6}")
    print(hdr)
    for name, fn in STRATS.items():
        m = evaluate(d, fn(d))
        print(f"{name:>5} {m['total']*100:>8.1f}% {m['cagr']*100:>7.1f}% "
              f"{m['vol']*100:>6.1f}% {m['sharpe']:>7.2f} {m['maxdd']*100:>7.1f}% "
              f"{m['exposure']*100:>5.0f}% {m['trades']:>7} {m['trades_yr']:>6.1f}")
    print()


def main():
    vix = fetch_yahoo("%5EVIX")
    qqq = fetch_yahoo("QQQ")
    base = pd.DataFrame({
        "vix_high": vix["high"],
        "vix_close": vix["close"],
        "qqq_close": qqq["close"],
    }).dropna().sort_index()
    d_full = build(base)

    # 5-year window; keep SMA200 warm-up by computing indicators on full history
    # then slicing.
    start = pd.Timestamp("2021-08-10")
    d_5y = d_full[d_full.index >= start].copy()

    run(d_full, "FULL HISTORY")
    run(d_5y, "LAST 5 YEARS")

    print("Legend: BH=buy&hold  SMA=200d trend filter  RSI2=Connors mean-revert")
    print("        DIP=3-down-days dip buy  VIXR=VIX-spike 10d dip buy")
    print("Execution: next-day fills, 0.02%/side cost, long/flat only.")


if __name__ == "__main__":
    main()
