"""
QQQ 1DTE short-put backtest with real Polygon option prices.

Strategy
--------
For each trading day T-1 in the look-back window:
  - Sell 1 put on QQQ expiring on the next trading day T.
  - Strike = listed strike closest to T-1 close * 0.98 (slight OTM).
  - Premium = T-1 EOD close of that option contract (Polygon).
  - On T:
      * If QQQ close > strike  -> option expires worthless, P&L = +premium*100
      * If QQQ close <= strike -> assigned 100 shares at strike, cost basis = strike,
        hold for HOLD_DAYS=10 trading days, mark-to-market daily, then sell at close.
        Resume selling puts on the day after the position is closed.

Window: trailing ~1y up to RUN_DATE (default = today).

Data
----
  QQQ daily bars        : yfinance
  Option chain & EOD    : Polygon REST via user proxy
        PROXY_URL  env (e.g. http://43.206.151.58:8080)
        PROXY_KEY  env -> sent as `X-Proxy-Key`
        Endpoints:
          GET /v3/reference/options/contracts?underlying_ticker=QQQ&expiration_date=YYYY-MM-DD&contract_type=put
          GET /v2/aggs/ticker/{O:ticker}/range/1/day/{from}/{to}

Outputs
-------
  qqq_put_log.csv       daily action log (EXPIRE / ASSIGN / HOLD / SELL + cumulative PnL)
  qqq_put_premiums.csv  T-1 -> strike / contract ticker / premium
  stdout summary        total PnL, #assigns, avg strike, return vs QQQ buy-and-hold
"""

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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TICKER = "QQQ"
STRIKE_PCT = 0.98              # T-1 close * 0.98
HOLD_DAYS = 10                 # trading days to hold after assignment
LOOKBACK_DAYS = 365            # calendar days
CONTRACT_MULTIPLIER = 100      # shares per option contract

RUN_DATE = os.environ.get("RUN_DATE")  # YYYY-MM-DD; default = today
PROXY_URL = os.environ.get("PROXY_URL", "").rstrip("/")
PROXY_KEY = os.environ.get("PROXY_KEY", "")

LOG_PATH = "qqq_put_log.csv"
PREMIUM_PATH = "qqq_put_premiums.csv"


# ---------------------------------------------------------------------------
# Polygon client (via proxy)
# ---------------------------------------------------------------------------

class PolygonProxy:
    def __init__(self, base_url: str, key: str):
        if not base_url:
            raise RuntimeError("PROXY_URL env var is required")
        if not key:
            raise RuntimeError("PROXY_KEY env var is required")
        self.base = base_url
        self.session = requests.Session()
        self.session.headers.update({"X-Proxy-Key": key})

    def _get(self, path: str, params: Optional[dict] = None, tries: int = 4) -> dict:
        url = f"{self.base}{path}"
        last = None
        for i in range(tries):
            try:
                r = self.session.get(url, params=params, timeout=20)
                if r.status_code == 429:
                    time.sleep(2 ** i)
                    continue
                r.raise_for_status()
                return r.json()
            except requests.RequestException as e:
                last = e
                time.sleep(2 ** i)
        raise RuntimeError(f"GET {url} failed: {last}")

    def list_put_contracts(self, expiry: date) -> list[dict]:
        """All QQQ put contracts expiring on the given date."""
        out: list[dict] = []
        params = {
            "underlying_ticker": TICKER,
            "contract_type": "put",
            "expiration_date": expiry.isoformat(),
            "expired": "true",        # include expired contracts (historical)
            "limit": 1000,
        }
        path = "/v3/reference/options/contracts"
        while True:
            data = self._get(path, params=params)
            out.extend(data.get("results", []) or [])
            next_url = data.get("next_url")
            if not next_url:
                break
            # next_url is absolute Polygon URL; replay the path+query through the proxy
            after_root = next_url.split("polygon.io", 1)[-1]
            path = after_root
            params = None
        return out

    def eod_close(self, option_ticker: str, day: date) -> Optional[float]:
        """Single-day close for an option contract on `day`."""
        path = f"/v2/aggs/ticker/{option_ticker}/range/1/day/{day.isoformat()}/{day.isoformat()}"
        data = self._get(path, params={"adjusted": "true"})
        results = data.get("results") or []
        if not results:
            return None
        return float(results[0]["c"])


# ---------------------------------------------------------------------------
# Strike selection
# ---------------------------------------------------------------------------

def pick_strike(contracts: list[dict], target: float) -> Optional[dict]:
    """Return the contract whose strike is closest to `target`."""
    if not contracts:
        return None
    return min(contracts, key=lambda c: abs(float(c["strike_price"]) - target))


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

@dataclass
class DailyRow:
    date: str
    qqq_close: float
    action: str         # SELL_PUT, EXPIRE, ASSIGN, HOLD, SELL_SHARES, SKIP
    strike: float | str = ""
    premium: float | str = ""
    contract: str = ""
    day_pnl: float = 0.0
    cum_pnl: float = 0.0
    note: str = ""


def load_qqq(end: date, lookback_days: int) -> pd.DataFrame:
    start = end - timedelta(days=lookback_days + 30)  # pad so we have full window
    df = yf.download(
        TICKER,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        progress=False,
        auto_adjust=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).date
    df = df[df.index <= end]
    df = df.tail(lookback_days + 1)  # keep ~1y of bars
    return df


def run_backtest(qqq: pd.DataFrame, poly: PolygonProxy) -> tuple[list[DailyRow], list[dict]]:
    log: list[DailyRow] = []
    premiums: list[dict] = []
    cum_pnl = 0.0

    dates = list(qqq.index)
    closes = qqq["Close"].astype(float).to_dict()

    # state for an open short put: (sell_date_idx, expiry_idx, strike, premium, contract)
    open_put = None
    # state for assigned shares: (assign_date_idx, cost_basis, days_held)
    holding = None

    i = 0
    while i < len(dates) - 1:
        d = dates[i]
        c = closes[d]

        # -- 1) If currently holding shares, MTM and possibly sell --
        if holding is not None:
            assign_idx, cost_basis, days_held = holding
            prev_close = closes[dates[i - 1]] if i > 0 else cost_basis
            day_pnl = (c - prev_close) * CONTRACT_MULTIPLIER
            cum_pnl += day_pnl
            days_held += 1
            holding = (assign_idx, cost_basis, days_held)

            if days_held >= HOLD_DAYS:
                # Sell at today's close. day_pnl above already covered today's MTM,
                # so the sale itself is flat -- realized P&L = sum of MTMs since entry.
                log.append(DailyRow(
                    date=d.isoformat(), qqq_close=c, action="SELL_SHARES",
                    day_pnl=day_pnl, cum_pnl=cum_pnl,
                    note=f"close 100sh @ {c:.2f}, cost {cost_basis:.2f}, held {days_held}d",
                ))
                holding = None
            else:
                log.append(DailyRow(
                    date=d.isoformat(), qqq_close=c, action="HOLD",
                    day_pnl=day_pnl, cum_pnl=cum_pnl,
                    note=f"day {days_held}/{HOLD_DAYS}, cost {cost_basis:.2f}",
                ))
            i += 1
            continue

        # -- 2) Sell a new 1DTE put at today's close, expiring next trading day --
        expiry = dates[i + 1]
        target_strike = c * STRIKE_PCT

        try:
            contracts = poly.list_put_contracts(expiry)
        except Exception as e:
            log.append(DailyRow(
                date=d.isoformat(), qqq_close=c, action="SKIP",
                note=f"chain fetch failed: {e}",
            ))
            i += 1
            continue

        pick = pick_strike(contracts, target_strike)
        if pick is None:
            log.append(DailyRow(
                date=d.isoformat(), qqq_close=c, action="SKIP",
                note=f"no put expiring {expiry}",
            ))
            i += 1
            continue

        strike = float(pick["strike_price"])
        option_ticker = pick["ticker"]
        premium = poly.eod_close(option_ticker, d)
        if premium is None:
            log.append(DailyRow(
                date=d.isoformat(), qqq_close=c, action="SKIP",
                strike=strike, contract=option_ticker,
                note="no EOD price",
            ))
            i += 1
            continue

        premiums.append({
            "sell_date": d.isoformat(),
            "expiry": expiry.isoformat(),
            "qqq_close": round(c, 4),
            "target_strike": round(target_strike, 4),
            "strike": strike,
            "premium": premium,
            "contract": option_ticker,
        })
        log.append(DailyRow(
            date=d.isoformat(), qqq_close=c, action="SELL_PUT",
            strike=strike, premium=premium, contract=option_ticker,
            day_pnl=0.0, cum_pnl=cum_pnl,
            note=f"sell put exp {expiry}",
        ))

        # -- 3) Resolve on expiry day --
        exp_close = closes[expiry]
        if exp_close > strike:
            day_pnl = premium * CONTRACT_MULTIPLIER
            cum_pnl += day_pnl
            log.append(DailyRow(
                date=expiry.isoformat(), qqq_close=exp_close, action="EXPIRE",
                strike=strike, premium=premium, contract=option_ticker,
                day_pnl=day_pnl, cum_pnl=cum_pnl,
                note=f"close {exp_close:.2f} > {strike:.2f}, keep premium",
            ))
        else:
            # Assigned at strike. Realized cash flow today:
            #   + premium*100 (kept)
            #   shares at cost basis = strike, MTM from strike to today's close
            mtm = (exp_close - strike) * CONTRACT_MULTIPLIER
            day_pnl = premium * CONTRACT_MULTIPLIER + mtm
            cum_pnl += day_pnl
            log.append(DailyRow(
                date=expiry.isoformat(), qqq_close=exp_close, action="ASSIGN",
                strike=strike, premium=premium, contract=option_ticker,
                day_pnl=day_pnl, cum_pnl=cum_pnl,
                note=f"close {exp_close:.2f} <= {strike:.2f}, take 100sh @ {strike:.2f}",
            ))
            holding = (i + 1, strike, 0)

        i += 2  # advance past expiry day

    return log, premiums


# ---------------------------------------------------------------------------
# IO + summary
# ---------------------------------------------------------------------------

def write_log(log: list[DailyRow], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "qqq_close", "action", "strike", "premium",
                    "contract", "day_pnl", "cum_pnl", "note"])
        for r in log:
            w.writerow([r.date, f"{r.qqq_close:.4f}", r.action, r.strike, r.premium,
                        r.contract, f"{r.day_pnl:.2f}", f"{r.cum_pnl:.2f}", r.note])


def write_premiums(rows: list[dict], path: str) -> None:
    if not rows:
        open(path, "w").close()
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def print_summary(log: list[DailyRow], premiums: list[dict], qqq: pd.DataFrame) -> None:
    actions = [r.action for r in log]
    n_sells = actions.count("SELL_PUT")
    n_expire = actions.count("EXPIRE")
    n_assign = actions.count("ASSIGN")
    total_pnl = log[-1].cum_pnl if log else 0.0

    avg_strike = (sum(p["strike"] for p in premiums) / len(premiums)) if premiums else 0.0
    avg_premium = (sum(p["premium"] for p in premiums) / len(premiums)) if premiums else 0.0

    qqq_start = float(qqq["Close"].iloc[0])
    qqq_end = float(qqq["Close"].iloc[-1])
    bh_pnl_per_100sh = (qqq_end - qqq_start) * CONTRACT_MULTIPLIER
    bh_ret = (qqq_end / qqq_start - 1.0) * 100

    # Strategy "notional" anchor for return %: cash to cover assignment at avg strike.
    notional = avg_strike * CONTRACT_MULTIPLIER if avg_strike else qqq_start * CONTRACT_MULTIPLIER
    strat_ret = (total_pnl / notional) * 100 if notional else 0.0

    print("=" * 60)
    print(f"QQQ 1DTE short-put backtest  |  window {qqq.index[0]} -> {qqq.index[-1]}")
    print("=" * 60)
    print(f"trading days seen        : {len(qqq)}")
    print(f"puts sold                : {n_sells}")
    print(f"  expired worthless      : {n_expire}")
    print(f"  assigned               : {n_assign}")
    print(f"avg strike               : {avg_strike:.2f}")
    print(f"avg premium (per share)  : {avg_premium:.3f}")
    print(f"total PnL  (1 contract)  : ${total_pnl:,.2f}")
    print(f"strategy return on notnl : {strat_ret:+.2f}%   (notional ~${notional:,.0f})")
    print(f"QQQ buy-and-hold PnL/100 : ${bh_pnl_per_100sh:,.2f}  ({bh_ret:+.2f}%)")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    end = date.fromisoformat(RUN_DATE) if RUN_DATE else date.today()
    print(f"[info] window end = {end}, lookback = {LOOKBACK_DAYS}d")

    qqq = load_qqq(end, LOOKBACK_DAYS)
    print(f"[info] QQQ bars: {len(qqq)} ({qqq.index[0]} -> {qqq.index[-1]})")

    try:
        poly = PolygonProxy(PROXY_URL, PROXY_KEY)
    except RuntimeError as e:
        print(f"[fatal] {e}")
        print("        set PROXY_URL and PROXY_KEY env vars; the proxy host must be")
        print("        on the sandbox outbound allowlist.")
        return 2

    log, premiums = run_backtest(qqq, poly)
    write_log(log, LOG_PATH)
    write_premiums(premiums, PREMIUM_PATH)
    print(f"[info] wrote {LOG_PATH} ({len(log)} rows)")
    print(f"[info] wrote {PREMIUM_PATH} ({len(premiums)} rows)")
    print_summary(log, premiums, qqq)
    return 0


if __name__ == "__main__":
    sys.exit(main())
