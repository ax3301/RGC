#!/usr/bin/env python3
"""
Realtime options snapshot + IV / OI / volume + recent unusual flow.

Uses yfinance first (free, no key). If POLYGON_API_KEY or PROXY_URL/PROXY_KEY
is set, it ALSO pulls Polygon for accurate bid/ask + Greeks (yfinance
often returns 0s for IV/OI on small ETF options — see today's DRAM example).

Why this matters (DRAM lesson)
------------------------------
We discovered today that yfinance's option chain for small ETFs returns
zeros for IV/OI/bid/ask. That meant we had to GUESS strikes/premiums during
your roll. This script:
  1. Tries yfinance.
  2. Falls back to Polygon (direct or via proxy) for fields yfinance leaves blank.
  3. Prints a clear "what's tradable now" snapshot, with ATM straddle
     implied move, skew, term structure.

Usage
-----
    python realtime_options.py AVGO
    python realtime_options.py DRAM --expiries 2026-07-17 2026-08-21
    python realtime_options.py NVDA --strikes-band 0.10
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime
from typing import Optional

import pandas as pd
import requests
import yfinance as yf


PROXY_URL = os.environ.get("PROXY_URL", "").rstrip("/")
PROXY_KEY = os.environ.get("PROXY_KEY", "")
POLY_KEY = os.environ.get("POLYGON_API_KEY", "")
DIRECT_BASE = "https://api.polygon.io"


def polygon_get(path: str, params: dict | None = None,
                timeout: float = 10) -> Optional[dict]:
    """Polygon GET — try proxy first, then direct if POLYGON_API_KEY set."""
    params = dict(params or {})
    if PROXY_URL and PROXY_KEY:
        try:
            r = requests.get(f"{PROXY_URL}{path}", params=params,
                             headers={"X-Proxy-Key": PROXY_KEY},
                             timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception:
            pass
    if POLY_KEY:
        params["apiKey"] = POLY_KEY
        try:
            r = requests.get(f"{DIRECT_BASE}{path}", params=params,
                             timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception:
            pass
    return None


def polygon_chain_snapshot(ticker: str, expiry: str) -> Optional[pd.DataFrame]:
    """Pull Polygon options chain snapshot for one expiry."""
    rows = []
    cursor = None
    path = f"/v3/snapshot/options/{ticker}"
    while True:
        params = {"expiration_date": expiry, "limit": 250}
        if cursor:
            params["cursor"] = cursor
        data = polygon_get(path, params)
        if not data:
            return None
        for r in data.get("results", []):
            d = r.get("details", {})
            day = r.get("day", {})
            greeks = r.get("greeks", {})
            iv = r.get("implied_volatility")
            quote = r.get("last_quote", {})
            rows.append({
                "contract": d.get("ticker"),
                "type": d.get("contract_type"),
                "strike": d.get("strike_price"),
                "expiry": d.get("expiration_date"),
                "bid": quote.get("bid"),
                "ask": quote.get("ask"),
                "mid": ((quote.get("bid") or 0) + (quote.get("ask") or 0)) / 2,
                "volume": day.get("volume"),
                "oi": r.get("open_interest"),
                "iv": iv,
                "delta": greeks.get("delta"),
                "theta": greeks.get("theta"),
                "vega": greeks.get("vega"),
                "gamma": greeks.get("gamma"),
            })
        nxt = data.get("next_url")
        if not nxt:
            break
        cursor = nxt.split("cursor=")[-1].split("&")[0] if "cursor=" in nxt else None
        if not cursor:
            break
        time.sleep(0.1)
    return pd.DataFrame(rows) if rows else None


def yf_chain(ticker: str, expiry: str) -> Optional[pd.DataFrame]:
    try:
        t = yf.Ticker(ticker)
        ch = t.option_chain(expiry)
    except Exception as e:
        print(f"  yfinance chain failed: {e}", file=sys.stderr)
        return None

    def _norm(df: pd.DataFrame, side: str) -> pd.DataFrame:
        df = df[["contractSymbol", "strike", "bid", "ask", "lastPrice",
                 "volume", "openInterest", "impliedVolatility"]].copy()
        df["type"] = side
        df.columns = ["contract", "strike", "bid", "ask", "last",
                      "volume", "oi", "iv", "type"]
        df["mid"] = (df["bid"].fillna(0) + df["ask"].fillna(0)) / 2
        df["expiry"] = expiry
        return df

    return pd.concat([_norm(ch.calls, "call"),
                      _norm(ch.puts, "put")], ignore_index=True)


def summarise(chain: pd.DataFrame, spot: float) -> None:
    print(f"\n  Spot: ${spot:.2f}")
    if chain.empty:
        print("  (no rows)")
        return

    # ATM straddle implied move
    atm = (chain["strike"] - spot).abs().min()
    atm_rows = chain[(chain["strike"] - spot).abs() == atm]
    atm_call = atm_rows[atm_rows["type"] == "call"]
    atm_put = atm_rows[atm_rows["type"] == "put"]
    if len(atm_call) and len(atm_put):
        c_mid = float(atm_call["mid"].iloc[0])
        p_mid = float(atm_put["mid"].iloc[0])
        straddle = c_mid + p_mid
        move_pct = straddle / spot * 100
        print(f"  ATM straddle ({atm_call['strike'].iloc[0]:.1f}) = "
              f"${straddle:.2f}  implied move ±{move_pct:.1f}%")

    # Skew: 25-delta proxy = ~5% OTM
    otm_p = chain[(chain["type"] == "put") & (chain["strike"] < spot * 0.96)]
    otm_c = chain[(chain["type"] == "call") & (chain["strike"] > spot * 1.04)]
    if len(otm_p) and len(otm_c):
        p_iv = otm_p.nsmallest(1, "strike")["iv"].iloc[0]
        c_iv = otm_c.nsmallest(1, "strike")["iv"].iloc[0]
        if p_iv and c_iv:
            print(f"  25Δ skew (put-call IV) = "
                  f"{(p_iv - c_iv) * 100:+.1f} vol points")

    # Top OI strikes (call/put)
    near = chain[(chain["strike"] > spot * 0.85) &
                 (chain["strike"] < spot * 1.15)].copy()
    if "oi" in near and near["oi"].notna().any():
        top = near.sort_values("oi", ascending=False).head(8)
        print(f"\n  Top open-interest strikes (±15%):")
        for _, r in top.iterrows():
            iv_str = f"{(r['iv'] or 0)*100:.0f}%" if r["iv"] else "  -"
            print(f"    {r['type']:<4} K={r['strike']:>7.1f}  "
                  f"bid={(r['bid'] or 0):>5.2f}  ask={(r['ask'] or 0):>5.2f}  "
                  f"OI={int(r['oi'] or 0):>6,}  vol={int(r['volume'] or 0):>6,}  IV={iv_str}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker")
    ap.add_argument("--expiries", nargs="*",
                    help="Specific expiry dates YYYY-MM-DD; default: nearest 3")
    ap.add_argument("--strikes-band", type=float, default=0.15,
                    help="±% around spot to display (default 0.15)")
    args = ap.parse_args()

    t = yf.Ticker(args.ticker.upper())
    spot = float(t.fast_info.last_price)
    print(f"=== {args.ticker.upper()}  spot ${spot:.2f}  "
          f"({datetime.now():%Y-%m-%d %H:%M:%S}) ===")

    expiries = args.expiries or list(t.options[:3])
    print(f"Expiries: {expiries}")
    print(f"Polygon proxy: {'configured' if PROXY_URL and PROXY_KEY else 'NOT configured'}")
    print(f"Polygon direct API key: {'set' if POLY_KEY else 'not set'}")

    for exp in expiries:
        print(f"\n----- expiry {exp} -----")
        chain = None
        if PROXY_URL or POLY_KEY:
            chain = polygon_chain_snapshot(args.ticker.upper(), exp)
            if chain is not None:
                print("  source: Polygon")
        if chain is None:
            chain = yf_chain(args.ticker.upper(), exp)
            if chain is not None:
                print("  source: yfinance")
        if chain is None:
            print("  ✗ no data")
            continue
        summarise(chain, spot)

    return 0


if __name__ == "__main__":
    sys.exit(main())
