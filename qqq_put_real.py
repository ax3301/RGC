"""QQQ 1DTE short-put backtest using Polygon EOD option data via proxy."""
from __future__ import annotations

import csv
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

PROXY_URL = os.environ.get("PROXY_URL", "").rstrip("/")
PROXY_KEY = os.environ.get("PROXY_KEY", "")
END_DATE = os.environ.get("END_DATE", "2026-05-18")
LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "365"))
STRIKE_RATIO = float(os.environ.get("STRIKE_RATIO", "0.98"))
HOLD_DAYS = int(os.environ.get("HOLD_DAYS", "10"))
CONTRACT_MULT = 100

LOG_CSV = "qqq_put_log.csv"
PREM_CSV = "qqq_put_premiums.csv"


def _require_proxy() -> None:
    if not PROXY_URL or not PROXY_KEY:
        sys.exit("ERROR: PROXY_URL and PROXY_KEY env vars must be set")


def _polygon_get(path: str, params: Optional[dict] = None, retries: int = 3) -> dict:
    url = f"{PROXY_URL}{path}"
    headers = {"X-Proxy-Key": PROXY_KEY}
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(1 + attempt)
    raise RuntimeError(f"polygon GET failed {path}: {last_err}")


def fetch_qqq(end: date, days: int) -> pd.DataFrame:
    start = end - timedelta(days=days + 30)
    df = yf.download("QQQ", start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
                     progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].dropna()
    df.index = pd.to_datetime(df.index).date
    cutoff = end - timedelta(days=days)
    return df[df.index >= cutoff]


def list_put_contracts(expiry: date) -> list[dict]:
    out: list[dict] = []
    params = {
        "underlying_ticker": "QQQ",
        "expiration_date": expiry.isoformat(),
        "contract_type": "put",
        "limit": 1000,
        "expired": "true",
    }
    resp = _polygon_get("/v3/reference/options/contracts", params)
    out.extend(resp.get("results", []) or [])
    while resp.get("next_url"):
        nxt = resp["next_url"]
        path = nxt.split("polygon.io", 1)[-1] if "polygon.io" in nxt else nxt
        resp = _polygon_get(path)
        out.extend(resp.get("results", []) or [])
    return out


def pick_strike(contracts: list[dict], target: float) -> Optional[dict]:
    if not contracts:
        return None
    return min(contracts, key=lambda c: abs(float(c["strike_price"]) - target))


def fetch_eod_close(ticker: str, day: date) -> Optional[float]:
    d = day.isoformat()
    path = f"/v2/aggs/ticker/{ticker}/range/1/day/{d}/{d}"
    resp = _polygon_get(path, {"adjusted": "true"})
    results = resp.get("results") or []
    if not results:
        return None
    return float(results[0]["c"])


@dataclass
class Position:
    entry_date: date
    strike: float
    premium: float
    expiry: date
    contract: str
    days_held: int = 0
    assigned: bool = False


def run_backtest() -> None:
    _require_proxy()
    end = datetime.strptime(END_DATE, "%Y-%m-%d").date()
    px = fetch_qqq(end, LOOKBACK_DAYS)
    if px.empty:
        sys.exit("ERROR: empty QQQ price series")

    dates = list(px.index)
    closes = {d: float(px.loc[d, "Close"]) for d in dates}

    log_rows: list[dict] = []
    prem_rows: list[dict] = []
    cum_pnl = 0.0
    n_expire = 0
    n_assign = 0
    strikes_used: list[float] = []
    pos: Optional[Position] = None

    for i in range(len(dates) - 1):
        t_minus_1 = dates[i]
        t = dates[i + 1]

        if pos is not None and pos.assigned:
            pos.days_held += 1
            day_pnl_per_share = closes[t] - closes[t_minus_1]
            day_pnl = day_pnl_per_share * CONTRACT_MULT
            cum_pnl += day_pnl
            action = "HOLD"
            if pos.days_held >= HOLD_DAYS:
                action = "SELL_SHARES"
                pos = None
            log_rows.append({
                "date": t.isoformat(),
                "action": action,
                "qqq_close": round(closes[t], 4),
                "strike": pos.strike if pos else "",
                "premium": "",
                "contract": "",
                "day_pnl": round(day_pnl, 2),
                "cum_pnl": round(cum_pnl, 2),
            })
            continue

        spot = closes[t_minus_1]
        target_strike = spot * STRIKE_RATIO
        try:
            contracts = list_put_contracts(t)
        except Exception as e:
            print(f"[{t_minus_1}] contracts fetch failed: {e}", file=sys.stderr)
            continue
        chosen = pick_strike(contracts, target_strike)
        if chosen is None:
            print(f"[{t_minus_1}] no put contracts for expiry {t}", file=sys.stderr)
            continue

        ticker = chosen["ticker"]
        strike = float(chosen["strike_price"])
        try:
            premium = fetch_eod_close(ticker, t_minus_1)
        except Exception as e:
            print(f"[{t_minus_1}] premium fetch failed for {ticker}: {e}", file=sys.stderr)
            continue
        if premium is None:
            print(f"[{t_minus_1}] no EOD bar for {ticker}", file=sys.stderr)
            continue

        prem_rows.append({
            "trade_date": t_minus_1.isoformat(),
            "expiry": t.isoformat(),
            "qqq_close": round(spot, 4),
            "target_strike": round(target_strike, 2),
            "strike": strike,
            "premium": premium,
            "contract": ticker,
        })
        strikes_used.append(strike)

        qqq_t = closes[t]
        if qqq_t > strike:
            pnl = premium * CONTRACT_MULT
            cum_pnl += pnl
            n_expire += 1
            log_rows.append({
                "date": t.isoformat(),
                "action": "EXPIRE",
                "qqq_close": round(qqq_t, 4),
                "strike": strike,
                "premium": premium,
                "contract": ticker,
                "day_pnl": round(pnl, 2),
                "cum_pnl": round(cum_pnl, 2),
            })
        else:
            day_pnl_per_share = (qqq_t - strike) + premium
            day_pnl = day_pnl_per_share * CONTRACT_MULT
            cum_pnl += day_pnl
            n_assign += 1
            pos = Position(
                entry_date=t,
                strike=strike,
                premium=premium,
                expiry=t,
                contract=ticker,
                days_held=0,
                assigned=True,
            )
            log_rows.append({
                "date": t.isoformat(),
                "action": "ASSIGN",
                "qqq_close": round(qqq_t, 4),
                "strike": strike,
                "premium": premium,
                "contract": ticker,
                "day_pnl": round(day_pnl, 2),
                "cum_pnl": round(cum_pnl, 2),
            })

    with open(LOG_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "date", "action", "qqq_close", "strike", "premium",
            "contract", "day_pnl", "cum_pnl",
        ])
        w.writeheader()
        w.writerows(log_rows)

    with open(PREM_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "trade_date", "expiry", "qqq_close",
            "target_strike", "strike", "premium", "contract",
        ])
        w.writeheader()
        w.writerows(prem_rows)

    first_close = closes[dates[0]]
    last_close = closes[dates[-1]]
    bh_pnl = (last_close - first_close) * CONTRACT_MULT
    bh_ret = (last_close / first_close - 1.0) * 100
    notional = first_close * CONTRACT_MULT
    strat_ret = cum_pnl / notional * 100 if notional else 0.0
    avg_strike = sum(strikes_used) / len(strikes_used) if strikes_used else 0.0

    print("=" * 60)
    print(f"QQQ 1DTE short-put backtest  {dates[0]} -> {dates[-1]}")
    print(f"trades opened:       {len(prem_rows)}")
    print(f"expired worthless:   {n_expire}")
    print(f"assigned:            {n_assign}")
    print(f"avg strike:          {avg_strike:.2f}")
    print(f"strategy cum PnL:    ${cum_pnl:,.2f}  ({strat_ret:.2f}% on ${notional:,.0f} notional)")
    print(f"buy & hold PnL:      ${bh_pnl:,.2f}  ({bh_ret:.2f}%)")
    print(f"log written:         {LOG_CSV}")
    print(f"premiums written:    {PREM_CSV}")


if __name__ == "__main__":
    run_backtest()
