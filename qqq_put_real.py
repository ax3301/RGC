#!/usr/bin/env python3
"""QQQ 1DTE short-put backtest using real Polygon option EOD via proxy.

Env:
  PROXY_URL  e.g. http://43.206.151.58:8080
  PROXY_KEY  sent as X-Proxy-Key header

Outputs in CWD:
  qqq_put_premiums.csv   one row per opened trade
  qqq_put_log.csv        per-day action log with cumulative PnL
  prints summary to stdout
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
import yfinance as yf

PROXY_URL = os.environ.get("PROXY_URL", "").rstrip("/")
PROXY_KEY = os.environ.get("PROXY_KEY", "")
if not PROXY_URL or not PROXY_KEY:
    sys.exit("PROXY_URL and PROXY_KEY must be set in env")

HEADERS = {"X-Proxy-Key": PROXY_KEY}
CACHE_DIR = Path(".polygon_cache")
CACHE_DIR.mkdir(exist_ok=True)

END_DATE = date(2026, 5, 18)
WINDOW_START = END_DATE - timedelta(days=365)
HISTORY_START = END_DATE - timedelta(days=400)

CONTRACT_SIZE = 100
STRIKE_PCT = 0.98
HOLD_DAYS = 10


def _cache_path(url: str) -> Path:
    return CACHE_DIR / f"{hashlib.sha1(url.encode()).hexdigest()}.json"


def http_get_json(url: str) -> dict:
    fp = _cache_path(url)
    if fp.exists():
        return json.loads(fp.read_text())
    last_err = None
    for attempt in range(4):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                data = r.json()
                fp.write_text(json.dumps(data))
                return data
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"GET {url} -> {r.status_code} {r.text[:300]}")
        except requests.RequestException as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"GET {url} failed: {last_err}")


def _proxy_rewrite(next_url: str) -> str:
    p = urlparse(next_url)
    q = f"?{p.query}" if p.query else ""
    return f"{PROXY_URL}{p.path}{q}"


def list_put_contracts(expiration: date) -> list[dict]:
    url = (
        f"{PROXY_URL}/v3/reference/options/contracts"
        f"?underlying_ticker=QQQ&contract_type=put"
        f"&expiration_date={expiration.isoformat()}"
        f"&expired=true&limit=1000&order=asc&sort=strike_price"
    )
    out: list[dict] = []
    while url:
        data = http_get_json(url)
        out.extend(data.get("results", []) or [])
        nxt = data.get("next_url")
        url = _proxy_rewrite(nxt) if nxt else None
    return out


def get_option_close(ticker: str, day: date) -> float | None:
    d = day.isoformat()
    url = (
        f"{PROXY_URL}/v2/aggs/ticker/{ticker}/range/1/day/{d}/{d}"
        f"?adjusted=true&sort=asc&limit=1"
    )
    data = http_get_json(url)
    results = data.get("results") or []
    if not results:
        return None
    return float(results[0]["c"])


def load_qqq_bars() -> pd.DataFrame:
    df = yf.download(
        "QQQ",
        start=HISTORY_START.isoformat(),
        end=(END_DATE + timedelta(days=1)).isoformat(),
        auto_adjust=False,
        progress=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).date
    return df[["Open", "High", "Low", "Close"]].dropna()


def main() -> None:
    bars = load_qqq_bars()
    days = list(bars.index)

    premium_rows: list[dict] = []
    log_rows: list[dict] = []
    cum_pnl = 0.0
    assigns = 0
    strikes_used: list[float] = []

    i = 0
    while i < len(days) - 1:
        t_prev = days[i]
        t = days[i + 1]
        if t < WINDOW_START:
            i += 1
            continue
        if t > END_DATE:
            break

        spot = float(bars.loc[t_prev, "Close"])
        target = spot * STRIKE_PCT

        contracts = list_put_contracts(t)
        if not contracts:
            log_rows.append({
                "date": t.isoformat(), "action": "SKIP_NO_CHAIN",
                "spot_T-1": spot, "strike": "", "premium": "",
                "qqq_close_T": float(bars.loc[t, "Close"]),
                "day_pnl": 0.0, "cum_pnl": cum_pnl,
            })
            i += 1
            continue

        best = min(contracts, key=lambda c: abs(float(c["strike_price"]) - target))
        strike = float(best["strike_price"])
        ticker = best["ticker"]

        premium = get_option_close(ticker, t_prev)
        if premium is None:
            log_rows.append({
                "date": t.isoformat(), "action": "SKIP_NO_PREMIUM",
                "spot_T-1": spot, "strike": strike, "premium": "",
                "qqq_close_T": float(bars.loc[t, "Close"]),
                "day_pnl": 0.0, "cum_pnl": cum_pnl,
            })
            i += 1
            continue

        premium_rows.append({
            "T-1": t_prev.isoformat(), "T": t.isoformat(),
            "spot_T-1": round(spot, 4), "target_strike": round(target, 4),
            "strike": strike, "premium": premium, "ticker": ticker,
        })

        qqq_T = float(bars.loc[t, "Close"])
        strikes_used.append(strike)

        if qqq_T > strike:
            day_pnl = premium * CONTRACT_SIZE
            cum_pnl += day_pnl
            log_rows.append({
                "date": t.isoformat(), "action": "EXPIRE",
                "spot_T-1": spot, "strike": strike, "premium": premium,
                "qqq_close_T": qqq_T,
                "day_pnl": day_pnl, "cum_pnl": cum_pnl,
            })
            i += 1
            continue

        assigns += 1
        day_pnl = (premium + (qqq_T - strike)) * CONTRACT_SIZE
        cum_pnl += day_pnl
        log_rows.append({
            "date": t.isoformat(), "action": "ASSIGN",
            "spot_T-1": spot, "strike": strike, "premium": premium,
            "qqq_close_T": qqq_T,
            "day_pnl": day_pnl, "cum_pnl": cum_pnl,
        })

        t_idx = i + 1
        last_hold_idx = t_idx
        for h in range(1, HOLD_DAYS + 1):
            k = t_idx + h
            if k >= len(days):
                break
            d = days[k]
            prev_close = float(bars.loc[days[k - 1], "Close"])
            cur_close = float(bars.loc[d, "Close"])
            day_pnl_h = (cur_close - prev_close) * CONTRACT_SIZE
            cum_pnl += day_pnl_h
            action = "SELL" if h == HOLD_DAYS else "HOLD"
            log_rows.append({
                "date": d.isoformat(), "action": action,
                "spot_T-1": "", "strike": strike, "premium": "",
                "qqq_close_T": cur_close,
                "day_pnl": day_pnl_h, "cum_pnl": cum_pnl,
            })
            last_hold_idx = k
        i = last_hold_idx

    premium_fields = ["T-1", "T", "spot_T-1", "target_strike", "strike", "premium", "ticker"]
    with open("qqq_put_premiums.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=premium_fields)
        w.writeheader()
        for r in premium_rows:
            w.writerow(r)

    log_fields = ["date", "action", "spot_T-1", "strike", "premium", "qqq_close_T", "day_pnl", "cum_pnl"]
    with open("qqq_put_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log_fields)
        w.writeheader()
        for r in log_rows:
            w.writerow(r)

    in_window = [d for d in days if WINDOW_START <= d <= END_DATE]
    if in_window:
        qqq_start = float(bars.loc[in_window[0], "Close"])
        qqq_end = float(bars.loc[in_window[-1], "Close"])
        bh_pnl = (qqq_end - qqq_start) * CONTRACT_SIZE
        bh_ret_pct = (qqq_end / qqq_start - 1.0) * 100
        notional = qqq_start * CONTRACT_SIZE
        strat_ret_pct = cum_pnl / notional * 100
    else:
        bh_pnl = float("nan")
        bh_ret_pct = float("nan")
        strat_ret_pct = float("nan")

    avg_strike = sum(strikes_used) / len(strikes_used) if strikes_used else float("nan")

    print(f"window:                  {WINDOW_START} -> {END_DATE}")
    print(f"trades opened:           {len(premium_rows)}")
    print(f"assigns:                 {assigns}")
    print(f"avg strike:              {avg_strike:.2f}")
    print(f"strategy total PnL ($):  {cum_pnl:.2f}")
    print(f"strategy return vs start notional: {strat_ret_pct:.2f}%")
    print(f"QQQ buy&hold PnL ($):    {bh_pnl:.2f}")
    print(f"QQQ buy&hold return:     {bh_ret_pct:.2f}%")


if __name__ == "__main__":
    main()
