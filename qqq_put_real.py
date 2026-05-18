"""QQQ 1DTE short-put backtest using real historical option prices.

Strategy
--------
* On every trading day T-1, sell 1 QQQ put expiring next trading day (T).
* Strike: the listed strike closest to T-1 close * 0.98.
* Premium: T-1 EOD close of that contract (real Polygon EOD, not BS).
* Settlement on T:
    - T close >  strike  -> option expires worthless; keep premium.
    - T close <= strike  -> assigned at strike; hold 10 trading days,
                            mark-to-market daily, then sell at close.
                            No new put sold during the hold.
* Window: trailing ~1 year ending today (CLI overridable).

Data sources
------------
* QQQ daily bars:   yfinance.
* Option chain + option EOD: Polygon REST API.
    - Direct mode:  POLYGON_API_KEY  -> https://api.polygon.io
    - Proxy mode:   PROXY_URL + PROXY_KEY (sent as X-Proxy-Key header).

Outputs
-------
* qqq_put_log.csv       per-day actions and cumulative PnL.
* qqq_put_premiums.csv  per-day strike / premium / contract ticker.
* Terminal summary.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import requests

try:
    import yfinance as yf  # type: ignore
except ImportError:
    print("Need `pip install yfinance pandas`.", file=sys.stderr)
    raise


CONTRACTS_PATH = "/v3/reference/options/contracts"
AGGS_PATH_TEMPLATE = "/v2/aggs/ticker/{ticker}/range/1/day/{frm}/{to}"


# --------------------------------------------------------------------------- #
# Polygon client (proxy-aware)
# --------------------------------------------------------------------------- #

class PolygonClient:
    def __init__(self) -> None:
        proxy_url = os.environ.get("PROXY_URL", "").rstrip("/")
        proxy_key = os.environ.get("PROXY_KEY", "")
        api_key = os.environ.get("POLYGON_API_KEY", "")

        if proxy_url and proxy_key:
            self.base = proxy_url
            self.headers = {"X-Proxy-Key": proxy_key}
            self.api_key_param: Optional[str] = api_key or None
            self.mode = "proxy"
        elif api_key:
            self.base = "https://api.polygon.io"
            self.headers = {"Authorization": f"Bearer {api_key}"}
            self.api_key_param = api_key
            self.mode = "direct"
        else:
            raise SystemExit(
                "No credentials. Set either PROXY_URL+PROXY_KEY or "
                "POLYGON_API_KEY in the environment."
            )

        self.session = requests.Session()

    def get(self, path: str, params: Optional[dict] = None) -> dict:
        url = self.base + path if path.startswith("/") else path
        params = dict(params or {})
        if self.api_key_param and "apiKey" not in params:
            params["apiKey"] = self.api_key_param
        for attempt in range(5):
            try:
                r = self.session.get(
                    url, params=params, headers=self.headers, timeout=30
                )
            except requests.RequestException as e:
                wait = 2 ** attempt
                print(f"  ! request error {e}; retry in {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            if r.status_code == 429:
                wait = 2 ** attempt
                print(f"  ! 429 rate-limited; sleep {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            if r.status_code >= 500:
                wait = 2 ** attempt
                print(f"  ! {r.status_code}; retry in {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"GET {url} failed after retries")

    def get_full(self, path: str, params: Optional[dict] = None) -> list[dict]:
        """Follow Polygon `next_url` pagination."""
        out: list[dict] = []
        params = dict(params or {})
        first = True
        url = path
        while True:
            if first:
                data = self.get(url, params=params)
                first = False
            else:
                # next_url already contains query string; append apiKey
                data = self.get(url, params={})
            out.extend(data.get("results") or [])
            nxt = data.get("next_url")
            if not nxt:
                break
            url = nxt
        return out


# --------------------------------------------------------------------------- #
# Underlying daily bars
# --------------------------------------------------------------------------- #

def load_qqq_bars(start: dt.date, end: dt.date):
    import pandas as pd

    # yfinance occasional 429s -> simple retry
    last_err = None
    for attempt in range(5):
        try:
            df = yf.download(
                "QQQ",
                start=start.isoformat(),
                end=(end + dt.timedelta(days=1)).isoformat(),
                auto_adjust=False,
                progress=False,
            )
            if not df.empty:
                break
        except Exception as e:
            last_err = e
        time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"yfinance failed: {last_err}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df[["Open", "High", "Low", "Close"]].dropna()
    df.index = [d.date() if hasattr(d, "date") else d for d in df.index]
    return df


# --------------------------------------------------------------------------- #
# Option contract selection + premium lookup
# --------------------------------------------------------------------------- #

def find_1dte_put(
    cli: PolygonClient,
    underlying: str,
    asof: dt.date,
    expiry: dt.date,
    target_strike: float,
) -> Optional[dict]:
    """Pick the listed PUT (expiring `expiry`) whose strike is closest to target.

    `as_of` filters to contracts that existed on T-1, so we don't accidentally
    use a strike that was added intraday on T.
    """
    params = {
        "underlying_ticker": underlying,
        "contract_type": "put",
        "expiration_date": expiry.isoformat(),
        "as_of": asof.isoformat(),
        "expired": "true",  # include expired contracts (essential for backtest)
        "limit": 1000,
    }
    results = cli.get_full(CONTRACTS_PATH, params)
    if not results:
        return None
    best = min(results, key=lambda r: abs(float(r["strike_price"]) - target_strike))
    return best


def get_eod_close(
    cli: PolygonClient, ticker: str, day: dt.date
) -> Optional[float]:
    path = AGGS_PATH_TEMPLATE.format(
        ticker=ticker, frm=day.isoformat(), to=day.isoformat()
    )
    data = cli.get(path, params={"adjusted": "true"})
    res = data.get("results") or []
    if not res:
        return None
    return float(res[0]["c"])


# --------------------------------------------------------------------------- #
# Backtest engine
# --------------------------------------------------------------------------- #

@dataclass
class LogRow:
    date: dt.date
    action: str
    qqq_close: float
    contract: str
    strike: float
    premium: float
    pnl_today: float
    cum_pnl: float
    note: str


def run_backtest(
    start: dt.date,
    end: dt.date,
    hold_days: int = 10,
    moneyness: float = 0.98,
):
    cli = PolygonClient()
    print(f"Polygon client mode: {cli.mode}", file=sys.stderr)

    print(f"Loading QQQ daily bars {start} to {end}...", file=sys.stderr)
    bars = load_qqq_bars(start, end)
    dates = list(bars.index)
    closes = {d: float(bars.loc[d, "Close"]) for d in dates}
    print(f"  {len(dates)} trading days", file=sys.stderr)

    log: list[LogRow] = []
    premiums: list[dict] = []
    cum = 0.0
    assigned_n = 0
    expired_n = 0

    i = 0
    while i < len(dates) - 1:
        t_minus_1 = dates[i]
        t = dates[i + 1]

        ref_close = closes[t_minus_1]
        target_strike = ref_close * moneyness

        print(
            f"[{t_minus_1}] QQQ={ref_close:.2f} target~{target_strike:.2f} "
            f"-> expiry {t}",
            file=sys.stderr,
        )

        contract = find_1dte_put(cli, "QQQ", t_minus_1, t, target_strike)
        if contract is None:
            print("  no contract found; skip", file=sys.stderr)
            log.append(
                LogRow(t_minus_1, "SKIP", ref_close, "", 0.0, 0.0, 0.0, cum,
                       "no 1DTE put listed")
            )
            i += 1
            continue

        ticker = contract["ticker"]
        strike = float(contract["strike_price"])
        premium = get_eod_close(cli, ticker, t_minus_1)
        if premium is None:
            print(f"  no EOD premium for {ticker} on {t_minus_1}", file=sys.stderr)
            log.append(
                LogRow(t_minus_1, "SKIP", ref_close, ticker, strike, 0.0, 0.0,
                       cum, "no premium EOD")
            )
            i += 1
            continue

        premiums.append({
            "date": t_minus_1.isoformat(),
            "expiry": t.isoformat(),
            "qqq_close": round(ref_close, 4),
            "target_strike": round(target_strike, 4),
            "strike": strike,
            "premium": premium,
            "ticker": ticker,
        })

        # SELL the put at T-1 close: log "SELL"
        log.append(
            LogRow(t_minus_1, "SELL_PUT", ref_close, ticker, strike, premium,
                   0.0, cum,
                   f"target_strike={target_strike:.2f}")
        )

        # Resolve on T
        t_close = closes[t]
        if t_close > strike:
            # premium kept, option worthless
            pnl = premium * 100.0
            cum += pnl
            expired_n += 1
            log.append(
                LogRow(t, "EXPIRE", t_close, ticker, strike, premium, pnl, cum,
                       f"close {t_close:.2f} > strike {strike:.2f}")
            )
            i += 1
            continue

        # Assignment: get 100 shares at `strike`, plus we kept the premium.
        # Realized at moment of assignment: +premium*100  (cash, regardless of
        # subsequent stock move). Then hold the stock for `hold_days` and mark
        # daily PnL relative to assignment basis.
        assigned_n += 1
        pnl_assign_day = (premium - max(strike - t_close, 0.0)) * 100.0
        # Equivalent formulation: cash from premium minus paper loss vs market
        # = (t_close - strike + premium) * 100. We use the canonical form.
        pnl_assign_day = (t_close - strike + premium) * 100.0
        cum += pnl_assign_day
        log.append(
            LogRow(t, "ASSIGN", t_close, ticker, strike, premium,
                   pnl_assign_day, cum,
                   f"assigned at {strike:.2f}; basis includes premium")
        )

        # Hold the shares for `hold_days` more trading days (mark-to-market),
        # then sell at close on the final day.
        prev_close = t_close
        for k in range(1, hold_days + 1):
            j = i + 1 + k
            if j >= len(dates):
                # ran out of data: liquidate at last available close
                if i + 1 + k - 1 < len(dates):
                    last = dates[-1]
                    last_close = closes[last]
                    daily = (last_close - prev_close) * 100.0
                    cum += daily
                    log.append(
                        LogRow(last, "HOLD_END_OF_DATA", last_close, ticker,
                               strike, 0.0, daily, cum,
                               "ran out of data; liquidate at last close")
                    )
                break
            d = dates[j]
            c = closes[d]
            daily = (c - prev_close) * 100.0
            cum += daily
            action = "SELL_STOCK" if k == hold_days else "HOLD"
            note = (
                f"sold @ {c:.2f}" if k == hold_days
                else f"mark @ {c:.2f}"
            )
            log.append(
                LogRow(d, action, c, ticker, strike, 0.0, daily, cum, note)
            )
            prev_close = c

        # Resume selling puts on the day AFTER the holding window ends
        i = i + 1 + hold_days  # i+1 was T (assignment day); +hold_days = exit
        # Next iteration uses dates[i] as new T-1. If i+hold_days is out of
        # range the outer while-loop terminates.

    return log, premiums, cum, expired_n, assigned_n, bars


# --------------------------------------------------------------------------- #
# CSV writers + summary
# --------------------------------------------------------------------------- #

def write_log_csv(path: str, log: list[LogRow]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "date", "action", "qqq_close", "contract", "strike",
            "premium", "pnl_today", "cum_pnl", "note",
        ])
        for r in log:
            w.writerow([
                r.date.isoformat(), r.action, f"{r.qqq_close:.4f}",
                r.contract, f"{r.strike:.4f}" if r.strike else "",
                f"{r.premium:.4f}" if r.premium else "",
                f"{r.pnl_today:.2f}", f"{r.cum_pnl:.2f}", r.note,
            ])


def write_premiums_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        if not rows:
            f.write("date,expiry,qqq_close,target_strike,strike,premium,ticker\n")
            return
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def print_summary(
    cum: float, expired: int, assigned: int,
    premiums: list[dict], bars,
) -> None:
    n_trades = expired + assigned
    avg_strike = (
        sum(p["strike"] for p in premiums) / len(premiums) if premiums else 0.0
    )
    avg_premium = (
        sum(p["premium"] for p in premiums) / len(premiums) if premiums else 0.0
    )

    first_close = float(bars["Close"].iloc[0])
    last_close = float(bars["Close"].iloc[-1])
    bh_pnl_per_share = last_close - first_close
    # Compare on per-contract scale (100 shares).
    bh_pnl_100 = bh_pnl_per_share * 100.0

    print()
    print("=" * 60)
    print("QQQ 1DTE SHORT-PUT BACKTEST SUMMARY")
    print("=" * 60)
    print(f"Trading days covered:    {len(bars)}")
    print(f"Trades opened:           {n_trades}")
    print(f"  Expired worthless:     {expired}")
    print(f"  Assigned:              {assigned}")
    if n_trades:
        print(f"  Assignment rate:       {assigned/n_trades*100:.1f}%")
    print(f"Avg strike:              {avg_strike:.2f}")
    print(f"Avg premium:             {avg_premium:.4f}")
    print(f"Total PnL (per 1 contract / 100 shares):  ${cum:,.2f}")
    print(f"Buy-and-hold QQQ (100 sh):                ${bh_pnl_100:,.2f}")
    print(f"  start {bars.index[0]} close {first_close:.2f}")
    print(f"  end   {bars.index[-1]} close {last_close:.2f}")
    print("=" * 60)


# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    today = dt.date.today()
    p.add_argument("--end", default=today.isoformat(),
                   help="end date (inclusive)")
    p.add_argument("--start", default=(today - dt.timedelta(days=370)).isoformat(),
                   help="start date (default: ~1y back)")
    p.add_argument("--hold-days", type=int, default=10)
    p.add_argument("--moneyness", type=float, default=0.98)
    p.add_argument("--log-out", default="qqq_put_log.csv")
    p.add_argument("--premiums-out", default="qqq_put_premiums.csv")
    args = p.parse_args()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)

    log, premiums, cum, expired_n, assigned_n, bars = run_backtest(
        start, end, hold_days=args.hold_days, moneyness=args.moneyness
    )
    write_log_csv(args.log_out, log)
    write_premiums_csv(args.premiums_out, premiums)
    print(f"wrote {args.log_out} ({len(log)} rows)", file=sys.stderr)
    print(f"wrote {args.premiums_out} ({len(premiums)} rows)", file=sys.stderr)
    print_summary(cum, expired_n, assigned_n, premiums, bars)


if __name__ == "__main__":
    main()
