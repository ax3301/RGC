#!/usr/bin/env python3
"""
QQQ 1DTE short-put backtest using real historical option EOD prices.

Strategy
--------
For every trading day t-1 (close), sell 1 QQQ put expiring on the next
trading day t.
  - strike   : listed strike closest to round(close[t-1] * 0.98)
  - premium  : EOD close of that put on day t-1 (Polygon /v2/aggs)
  - settle   :
      * QQQ_close[t] >  strike  -> expires worthless, keep premium
      * QQQ_close[t] <= strike  -> assigned at strike, hold 10 trading
        days, then sell at the close of the 10th day.
        While holding, no new puts are sold.

P&L per contract is x100 (one option = 100 shares). All amounts are USD.

Data sources
------------
  - QQQ daily OHLC : yfinance
  - Option chain   : Polygon  /v3/reference/options/contracts
  - Option EOD     : Polygon  /v2/aggs/ticker/{O:...}/range/1/day/{from}/{to}

Polygon is reached through a proxy:
  PROXY_URL   -- e.g. http://43.206.151.58:8080
  PROXY_KEY   -- sent as X-Proxy-Key request header
(Optionally a direct POLYGON_API_KEY is supported as a fallback.)

Outputs
-------
  qqq_put_premiums.csv : per-day (date, spot, target_strike, strike,
                        contract_ticker, premium)
  qqq_put_log.csv      : per-day action log (EXPIRE / ASSIGN / HOLD / SELL),
                        cumulative PnL
  Terminal            : total PnL, # assigns, avg strike, strategy return
                        vs QQQ buy-and-hold over the same window.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable

import pandas as pd
import requests
import yfinance as yf


# --------------------------------------------------------------------------- #
# Polygon client (via proxy)
# --------------------------------------------------------------------------- #

PROXY_URL = os.environ.get("PROXY_URL", "").rstrip("/")
PROXY_KEY = os.environ.get("PROXY_KEY", "")
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "")

DIRECT_BASE = "https://api.polygon.io"


def _polygon_get(path: str, params: dict | None = None) -> dict:
    """GET a Polygon endpoint via the configured proxy, fall back to direct."""
    params = dict(params or {})
    last_err = None

    if PROXY_URL and PROXY_KEY:
        url = f"{PROXY_URL}{path}"
        try:
            r = requests.get(
                url,
                params=params,
                headers={"X-Proxy-Key": PROXY_KEY},
                timeout=30,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e

    if POLYGON_API_KEY:
        url = f"{DIRECT_BASE}{path}"
        params["apiKey"] = POLYGON_API_KEY
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    raise RuntimeError(
        f"Polygon request failed: {last_err}. "
        "Set PROXY_URL/PROXY_KEY or POLYGON_API_KEY."
    )


def list_qqq_puts_for_expiry(expiry: date) -> pd.DataFrame:
    """Return all QQQ puts expiring on `expiry` (active + expired)."""
    rows: list[dict] = []
    params = {
        "underlying_ticker": "QQQ",
        "contract_type": "put",
        "expiration_date": expiry.isoformat(),
        "limit": 1000,
        "expired": "true",
    }
    path = "/v3/reference/options/contracts"
    while True:
        data = _polygon_get(path, params)
        rows.extend(data.get("results") or [])
        nxt = data.get("next_url")
        if not nxt:
            break
        # next_url is absolute; strip the host so we keep going through proxy
        if "polygon.io" in nxt:
            path = nxt.split("polygon.io", 1)[1]
            params = {}
        else:
            path = nxt
            params = {}
        time.sleep(0.05)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["strike_price"] = df["strike_price"].astype(float)
    return df[["ticker", "strike_price", "expiration_date"]].sort_values(
        "strike_price"
    )


def option_eod_close(ticker: str, day: date) -> float | None:
    """Polygon daily aggregate close for one option contract on `day`."""
    iso = day.isoformat()
    path = f"/v2/aggs/ticker/{ticker}/range/1/day/{iso}/{iso}"
    data = _polygon_get(path, {"adjusted": "true"})
    results = data.get("results") or []
    if not results:
        return None
    return float(results[0]["c"])


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #

CONTRACT_MULT = 100  # one option = 100 shares


@dataclass
class DayRow:
    date: date
    spot: float
    target_strike: float
    strike: float | None
    contract: str | None
    premium: float | None


def pick_strike(chain: pd.DataFrame, target: float) -> pd.Series | None:
    """Closest listed strike to `target` (ties -> lower strike, safer)."""
    if chain.empty:
        return None
    diff = (chain["strike_price"] - target).abs()
    idx = diff.idxmin()
    return chain.loc[idx]


def load_qqq(start: date, end: date) -> pd.DataFrame:
    df = yf.download(
        "QQQ",
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=False,
        progress=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).date
    return df[["Open", "High", "Low", "Close"]].copy()


def run_backtest(start: date, end: date) -> None:
    qqq = load_qqq(start, end)
    dates: list[date] = list(qqq.index)
    closes = qqq["Close"]

    print(f"loaded {len(dates)} QQQ trading days "
          f"({dates[0]} -> {dates[-1]})", flush=True)

    premiums_rows: list[dict] = []
    log_rows: list[dict] = []

    cum_pnl = 0.0
    n_expire = 0
    n_assign = 0
    strikes_used: list[float] = []

    # Assignment state
    held_shares = 0          # 0 or 100
    held_basis = 0.0         # strike we were assigned at
    held_until_idx = -1      # dates[index] when we exit the hold

    HOLD_DAYS = 10

    # Iterate over T-1 (signal day). T is the next trading day.
    for i in range(len(dates) - 1):
        t_minus_1 = dates[i]
        t = dates[i + 1]
        spot = float(closes.iloc[i])
        spot_t = float(closes.iloc[i + 1])

        # If currently holding assigned shares, just mark to market.
        if held_shares:
            day_pnl = (spot_t - float(closes.iloc[i])) * held_shares
            # we account daily MTM only to make log readable; realized PnL
            # is booked when we sell. Track realized only.
            action = "HOLD"
            if i + 1 >= held_until_idx:
                exit_price = spot_t
                realized = (exit_price - held_basis) * held_shares
                cum_pnl += realized
                action = "SELL"
                log_rows.append({
                    "date": t.isoformat(),
                    "action": action,
                    "strike": held_basis,
                    "contract": "",
                    "premium": "",
                    "qqq_close": spot_t,
                    "pnl": round(realized, 2),
                    "cum_pnl": round(cum_pnl, 2),
                })
                held_shares = 0
                held_basis = 0.0
                held_until_idx = -1
            else:
                log_rows.append({
                    "date": t.isoformat(),
                    "action": action,
                    "strike": held_basis,
                    "contract": "",
                    "premium": "",
                    "qqq_close": spot_t,
                    "pnl": 0.0,
                    "cum_pnl": round(cum_pnl, 2),
                })
            continue

        # --- sell a put at T-1 close, expiring T ---
        target = round(spot * 0.98, 0)  # whole-dollar target

        try:
            chain = list_qqq_puts_for_expiry(t)
        except Exception as e:
            print(f"  {t_minus_1}: chain fetch failed: {e}", flush=True)
            log_rows.append({
                "date": t_minus_1.isoformat(),
                "action": "SKIP_NO_CHAIN",
                "strike": "", "contract": "", "premium": "",
                "qqq_close": spot,
                "pnl": 0.0, "cum_pnl": round(cum_pnl, 2),
            })
            continue

        row = pick_strike(chain, target)
        if row is None:
            log_rows.append({
                "date": t_minus_1.isoformat(),
                "action": "SKIP_NO_STRIKE",
                "strike": "", "contract": "", "premium": "",
                "qqq_close": spot,
                "pnl": 0.0, "cum_pnl": round(cum_pnl, 2),
            })
            continue

        contract = row["ticker"]
        strike = float(row["strike_price"])

        try:
            premium = option_eod_close(contract, t_minus_1)
        except Exception as e:
            print(f"  {t_minus_1}: premium fetch failed for {contract}: {e}",
                  flush=True)
            premium = None

        if premium is None:
            log_rows.append({
                "date": t_minus_1.isoformat(),
                "action": "SKIP_NO_PREM",
                "strike": strike, "contract": contract, "premium": "",
                "qqq_close": spot,
                "pnl": 0.0, "cum_pnl": round(cum_pnl, 2),
            })
            continue

        premiums_rows.append({
            "date": t_minus_1.isoformat(),
            "spot": round(spot, 4),
            "target_strike": target,
            "strike": strike,
            "contract": contract,
            "premium": premium,
        })
        strikes_used.append(strike)

        # --- settle on T ---
        if spot_t > strike:
            pnl = premium * CONTRACT_MULT
            cum_pnl += pnl
            n_expire += 1
            log_rows.append({
                "date": t.isoformat(),
                "action": "EXPIRE",
                "strike": strike,
                "contract": contract,
                "premium": premium,
                "qqq_close": spot_t,
                "pnl": round(pnl, 2),
                "cum_pnl": round(cum_pnl, 2),
            })
        else:
            # Assigned. Cash flow = premium received - (strike - close)*100
            # We model it as: receive premium, take 100 shares at `strike`,
            # then sell at close of day t+HOLD_DAYS.
            pnl_assignment = premium * CONTRACT_MULT
            cum_pnl += pnl_assignment
            n_assign += 1
            held_shares = CONTRACT_MULT
            held_basis = strike
            held_until_idx = min(i + 1 + HOLD_DAYS, len(dates) - 1)
            log_rows.append({
                "date": t.isoformat(),
                "action": "ASSIGN",
                "strike": strike,
                "contract": contract,
                "premium": premium,
                "qqq_close": spot_t,
                "pnl": round(pnl_assignment, 2),
                "cum_pnl": round(cum_pnl, 2),
            })

    # Force-close any still-held shares at the final close.
    if held_shares:
        final_close = float(closes.iloc[-1])
        realized = (final_close - held_basis) * held_shares
        cum_pnl += realized
        log_rows.append({
            "date": dates[-1].isoformat(),
            "action": "SELL_EOD",
            "strike": held_basis,
            "contract": "",
            "premium": "",
            "qqq_close": final_close,
            "pnl": round(realized, 2),
            "cum_pnl": round(cum_pnl, 2),
        })

    # --- write CSVs ---
    with open("qqq_put_premiums.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["date", "spot", "target_strike",
                        "strike", "contract", "premium"],
        )
        w.writeheader()
        w.writerows(premiums_rows)

    with open("qqq_put_log.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["date", "action", "strike", "contract",
                        "premium", "qqq_close", "pnl", "cum_pnl"],
        )
        w.writeheader()
        w.writerows(log_rows)

    # --- terminal summary ---
    qqq_start = float(closes.iloc[0])
    qqq_end = float(closes.iloc[-1])
    bh_return = (qqq_end / qqq_start - 1) * 100

    # Strategy return is benchmarked against 100 shares of QQQ at t0.
    # That mirrors covered-put-style sizing (1 contract = 100 shares).
    notional = qqq_start * CONTRACT_MULT
    strat_return = cum_pnl / notional * 100

    avg_strike = sum(strikes_used) / len(strikes_used) if strikes_used else 0.0

    print()
    print("=" * 56)
    print(f"window         : {dates[0]} -> {dates[-1]}  "
          f"({len(dates)} trading days)")
    print(f"trades opened  : {len(premiums_rows)}")
    print(f"  expired      : {n_expire}")
    print(f"  assigned     : {n_assign}")
    print(f"avg strike     : {avg_strike:.2f}")
    print(f"total PnL ($)  : {cum_pnl:,.2f}  (per 1-contract notional)")
    print(f"strategy ret % : {strat_return:.2f}%")
    print(f"QQQ B&H ret %  : {bh_return:.2f}%   "
          f"({qqq_start:.2f} -> {qqq_end:.2f})")
    print("=" * 56)


# --------------------------------------------------------------------------- #

def main(argv: list[str]) -> int:
    end = date(2026, 5, 18)
    start = end - timedelta(days=365)
    run_backtest(start, end)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
