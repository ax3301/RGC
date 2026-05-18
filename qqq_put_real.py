"""
QQQ 1DTE short-put backtest with REAL option EOD prices.

Strategy
--------
At close of trading day T-1:
  * Sell 1 QQQ put expiring at T's close.
  * Strike = listed strike closest to round(T-1_close * 0.98, 2).
  * Premium = that contract's EOD close on T-1 (from Polygon aggregates).

Settlement at T's close:
  * QQQ_close > strike -> expire worthless. PnL += premium * 100.
  * QQQ_close <= strike -> assigned 100 shares at `strike`.
        Hold for HOLD_DAYS trading days, mark-to-market daily,
        sell at day T+HOLD close. No new put is sold during the hold.

Data
----
QQQ daily prices: yfinance.
Option contracts + EOD: Polygon, tunneled through a private HTTP proxy.

Env vars (no hardcoded secrets):
  PROXY_URL  e.g. http://1.2.3.4:8080
  PROXY_KEY  bearer-style key sent in X-Proxy-Key header

Outputs (written to cwd):
  qqq_put_log.csv       daily action log + cumulative PnL
  qqq_put_premiums.csv  per-trade strike / premium / contract ticker

Usage:
  PROXY_URL=... PROXY_KEY=... python3 qqq_put_real.py \
      --start 2025-05-18 --end 2026-05-18
"""
from __future__ import annotations

import argparse
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


CONTRACT_MULT = 100
HOLD_DAYS = 10
STRIKE_PCT = 0.98


# --------------------------------------------------------------------------- #
# Polygon client (via proxy)
# --------------------------------------------------------------------------- #
class Polygon:
    def __init__(self, base: str, key: str, timeout: int = 20):
        if not base or not key:
            raise SystemExit(
                "PROXY_URL and PROXY_KEY must both be set in env "
                "before running this script."
            )
        self.base = base.rstrip("/")
        self.headers = {"X-Proxy-Key": key, "Accept": "application/json"}
        self.timeout = timeout
        self.s = requests.Session()

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = path if path.startswith("http") else f"{self.base}{path}"
        for attempt in range(4):
            try:
                r = self.s.get(
                    url, params=params, headers=self.headers, timeout=self.timeout
                )
                if r.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                return r.json()
            except requests.RequestException as e:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("unreachable")

    def list_put_contracts(self, underlying: str, expiry: date) -> list[dict]:
        out: list[dict] = []
        params = {
            "underlying_ticker": underlying,
            "contract_type": "put",
            "expiration_date": expiry.isoformat(),
            "expired": "true",
            "limit": 1000,
        }
        data = self._get("/v3/reference/options/contracts", params=params)
        out.extend(data.get("results", []) or [])
        while data.get("next_url"):
            data = self._get(data["next_url"])
            out.extend(data.get("results", []) or [])
        return out

    def eod_close(self, option_ticker: str, on: date) -> float | None:
        d = on.isoformat()
        data = self._get(
            f"/v2/aggs/ticker/{option_ticker}/range/1/day/{d}/{d}",
            params={"adjusted": "true"},
        )
        bars = data.get("results") or []
        if not bars:
            return None
        return float(bars[0]["c"])


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
@dataclass
class Row:
    date: str
    action: str          # SELL / EXPIRE / ASSIGN / HOLD / CLOSE
    qqq_close: float
    strike: float | None
    premium: float | None
    contract: str | None
    daily_pnl: float
    cum_pnl: float
    note: str = ""


def pick_strike(contracts: list[dict], target: float) -> dict | None:
    if not contracts:
        return None
    return min(contracts, key=lambda c: abs(float(c["strike_price"]) - target))


def run(start: date, end: date, out_log: str, out_prem: str) -> None:
    poly = Polygon(os.environ.get("PROXY_URL", ""), os.environ.get("PROXY_KEY", ""))

    # Pull QQQ daily, with a small buffer at both ends so we can index neighbors.
    px = yf.download(
        "QQQ",
        start=(start - timedelta(days=10)).isoformat(),
        end=(end + timedelta(days=20)).isoformat(),
        progress=False,
        auto_adjust=False,
    )
    if isinstance(px.columns, pd.MultiIndex):
        px.columns = px.columns.get_level_values(0)
    px = px[["Close"]].dropna()
    px.index = pd.to_datetime(px.index).date
    dates: list[date] = sorted(d for d in px.index if start <= d <= end)
    if not dates:
        raise SystemExit("no QQQ trading days in the requested window")

    log: list[Row] = []
    prem_rows: list[dict] = []
    cum = 0.0

    # state machine: when assigned, we hold shares for HOLD_DAYS and skip puts
    assigned_until: date | None = None
    assigned_strike: float | None = None
    assigned_share_basis: float | None = None  # the strike we paid
    last_mark: float | None = None

    all_idx = list(px.index)
    idx_of = {d: i for i, d in enumerate(all_idx)}

    for d in dates:
        i = idx_of[d]
        qqq_close = float(px.loc[d, "Close"])

        # 1) if we are inside an assignment hold, mark-to-market and maybe close
        if assigned_until is not None:
            assert assigned_strike is not None
            assert assigned_share_basis is not None
            prev = last_mark if last_mark is not None else assigned_share_basis
            day_pnl = (qqq_close - prev) * CONTRACT_MULT
            cum += day_pnl
            last_mark = qqq_close

            if d >= assigned_until:
                # close at today's close (already marked); flat next day
                log.append(Row(
                    d.isoformat(), "CLOSE", qqq_close,
                    assigned_strike, None, None, day_pnl, cum,
                    note=f"sell @ close, basis {assigned_share_basis:.2f}",
                ))
                assigned_until = None
                assigned_strike = None
                assigned_share_basis = None
                last_mark = None
            else:
                log.append(Row(
                    d.isoformat(), "HOLD", qqq_close,
                    assigned_strike, None, None, day_pnl, cum,
                    note=f"basis {assigned_share_basis:.2f}",
                ))
            continue

        # 2) otherwise we are flat: sell a 1DTE put at today's close that
        #    expires on the NEXT trading day.
        if i + 1 >= len(all_idx):
            log.append(Row(d.isoformat(), "SKIP", qqq_close, None, None, None, 0.0, cum,
                           note="no next session"))
            continue
        t_next = all_idx[i + 1]
        if t_next > end:
            log.append(Row(d.isoformat(), "SKIP", qqq_close, None, None, None, 0.0, cum,
                           note="next session beyond window"))
            continue

        target = round(qqq_close * STRIKE_PCT, 2)
        try:
            contracts = poly.list_put_contracts("QQQ", t_next)
        except Exception as e:
            log.append(Row(d.isoformat(), "ERR", qqq_close, None, None, None, 0.0, cum,
                           note=f"contracts: {e}"))
            continue
        c = pick_strike(contracts, target)
        if c is None:
            log.append(Row(d.isoformat(), "SKIP", qqq_close, None, None, None, 0.0, cum,
                           note=f"no contract for {t_next}"))
            continue
        strike = float(c["strike_price"])
        otk = c["ticker"]
        try:
            premium = poly.eod_close(otk, d)
        except Exception as e:
            log.append(Row(d.isoformat(), "ERR", qqq_close, strike, None, otk, 0.0, cum,
                           note=f"eod: {e}"))
            continue
        if premium is None or premium <= 0:
            log.append(Row(d.isoformat(), "SKIP", qqq_close, strike, premium, otk, 0.0, cum,
                           note="no EOD premium"))
            continue

        prem_rows.append({
            "date": d.isoformat(),
            "expiry": t_next.isoformat(),
            "qqq_close": qqq_close,
            "target_strike": target,
            "strike": strike,
            "premium": premium,
            "contract": otk,
        })
        log.append(Row(d.isoformat(), "SELL", qqq_close, strike, premium, otk, 0.0, cum,
                       note=f"expires {t_next.isoformat()}"))

        # 3) settle on t_next
        qqq_next = float(px.loc[t_next, "Close"])
        if qqq_next > strike:
            day_pnl = premium * CONTRACT_MULT
            cum += day_pnl
            log.append(Row(t_next.isoformat(), "EXPIRE", qqq_next, strike,
                           premium, otk, day_pnl, cum,
                           note="OTM at expiry"))
        else:
            # assigned: shares cost = strike, plus we keep the premium.
            # Premium PnL is realized now; share PnL accrues over the hold.
            day_pnl_prem = premium * CONTRACT_MULT
            # mark-to-market the first day too: (close - strike) * 100
            day_pnl_mtm = (qqq_next - strike) * CONTRACT_MULT
            day_pnl = day_pnl_prem + day_pnl_mtm
            cum += day_pnl
            assigned_strike = strike
            assigned_share_basis = strike
            last_mark = qqq_next
            j = idx_of[t_next]
            j_close = min(j + HOLD_DAYS, len(all_idx) - 1)
            assigned_until = all_idx[j_close]
            log.append(Row(t_next.isoformat(), "ASSIGN", qqq_next, strike,
                           premium, otk, day_pnl, cum,
                           note=f"hold until {assigned_until.isoformat()}"))

    # write CSVs
    with open(out_log, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "action", "qqq_close", "strike", "premium",
                    "contract", "daily_pnl", "cum_pnl", "note"])
        for r in log:
            w.writerow([r.date, r.action, f"{r.qqq_close:.4f}",
                        "" if r.strike is None else f"{r.strike:.2f}",
                        "" if r.premium is None else f"{r.premium:.4f}",
                        r.contract or "", f"{r.daily_pnl:.2f}",
                        f"{r.cum_pnl:.2f}", r.note])
    with open(out_prem, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "date", "expiry", "qqq_close", "target_strike",
            "strike", "premium", "contract",
        ])
        w.writeheader()
        for r in prem_rows:
            w.writerow(r)

    # terminal summary
    n_sell = sum(1 for r in log if r.action == "SELL")
    n_expire = sum(1 for r in log if r.action == "EXPIRE")
    n_assign = sum(1 for r in log if r.action == "ASSIGN")
    strikes = [r["strike"] for r in prem_rows]
    avg_strike = sum(strikes) / len(strikes) if strikes else float("nan")

    first_close = float(px.loc[dates[0], "Close"])
    last_close = float(px.loc[dates[-1], "Close"])
    bh_pnl_per_share = last_close - first_close
    bh_return = bh_pnl_per_share / first_close

    # express strategy return on a notional of one contract = 100 * first_close
    notional = first_close * CONTRACT_MULT
    strat_return = cum / notional if notional else float("nan")

    print(f"window:           {dates[0]} -> {dates[-1]}  ({len(dates)} sessions)")
    print(f"trades sold:      {n_sell}")
    print(f"expired worthless:{n_expire}")
    print(f"assignments:      {n_assign}")
    print(f"avg strike:       {avg_strike:.2f}" if strikes else "avg strike:       -")
    print(f"total PnL ($):    {cum:,.2f}")
    print(f"strategy return:  {strat_return*100:.2f}%  (vs 1-contract notional ${notional:,.0f})")
    print(f"QQQ buy & hold:   {bh_return*100:.2f}%   ({first_close:.2f} -> {last_close:.2f})")


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main(argv: Iterable[str]) -> int:
    p = argparse.ArgumentParser()
    today = date.today()
    p.add_argument("--start", type=parse_date,
                   default=today - timedelta(days=365))
    p.add_argument("--end", type=parse_date, default=today)
    p.add_argument("--log", default="qqq_put_log.csv")
    p.add_argument("--premiums", default="qqq_put_premiums.csv")
    args = p.parse_args(list(argv))
    run(args.start, args.end, args.log, args.premiums)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
