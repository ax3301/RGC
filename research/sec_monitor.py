#!/usr/bin/env python3
"""
SEC EDGAR monitor — list & summarize the most recent filings for a ticker.

Usage:
    python sec_monitor.py AVGO NVDA DRAM
    python sec_monitor.py AVGO --forms 10-K 10-Q 8-K --limit 5
    python sec_monitor.py AVGO --download

Why this exists (from the DRAM lesson)
--------------------------------------
You sold puts on DRAM without reading the issuer's risk factors. A 5-minute
look at the 10-K "Risk Factors" + the last few 8-Ks would have flagged:
  - Memory pricing cyclicality
  - Customer concentration
  - Leverage / cash burn
This script makes that read take literally seconds.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

# edgartools needs an identity header per SEC rules.
os.environ.setdefault("EDGAR_IDENTITY",
                       os.environ.get("SEC_EDGAR_USER_AGENT",
                                      "Personal Research research@example.com"))

try:
    from edgar import Company, set_identity
except ImportError as e:
    print(f"edgartools not installed: {e}", file=sys.stderr)
    sys.exit(1)


def list_filings(ticker: str, forms: list[str], limit: int) -> None:
    set_identity(os.environ["EDGAR_IDENTITY"])
    try:
        co = Company(ticker)
    except Exception as e:
        print(f"  {ticker}: lookup failed ({e})")
        return

    print(f"\n=== {ticker} — {co.name} (CIK {co.cik}) ===")
    filings = co.get_filings(form=forms).head(limit)
    if not len(filings):
        print(f"  no {forms} filings found")
        return

    for f in filings:
        print(f"  {f.filing_date}  {f.form:<6}  {f.accession_no}  {f.primary_document}")


def show_risk_factors(ticker: str, max_chars: int = 4000) -> None:
    """Pull the latest 10-K Item 1A 'Risk Factors' and print a snippet."""
    set_identity(os.environ["EDGAR_IDENTITY"])
    try:
        co = Company(ticker)
        ten_k = co.get_filings(form="10-K").latest(1)
        if ten_k is None:
            print(f"  {ticker}: no 10-K found")
            return
        ten_k_obj = ten_k.obj() if hasattr(ten_k, "obj") else ten_k
        # edgartools 2.x exposes Item-level access
        risk = ten_k_obj.risk_factors if hasattr(ten_k_obj, "risk_factors") else None
        if not risk:
            print(f"  {ticker}: risk_factors not parseable, showing filing summary instead")
            print(str(ten_k_obj)[:max_chars])
            return
        text = str(risk)
        print(f"\n--- {ticker} 10-K Risk Factors (first {max_chars} chars) ---")
        print(text[:max_chars] + ("..." if len(text) > max_chars else ""))
    except Exception as e:
        print(f"  {ticker}: risk-factor extraction failed ({e})")


def download_filings(tickers: list[str], forms: list[str], limit: int,
                     out_dir: str) -> None:
    from sec_edgar_downloader import Downloader
    dl = Downloader("PersonalResearch",
                    os.environ["EDGAR_IDENTITY"].split()[-1],
                    out_dir)
    for t in tickers:
        for form in forms:
            try:
                n = dl.get(form, t, limit=limit)
                print(f"  {t}/{form}: downloaded {n} filings into {out_dir}")
            except Exception as e:
                print(f"  {t}/{form}: download failed ({e})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="+")
    ap.add_argument("--forms", nargs="+", default=["10-K", "10-Q", "8-K"])
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--risk", action="store_true",
                    help="Print the latest 10-K risk factors")
    ap.add_argument("--download", action="store_true",
                    help="Bulk-download filings to ./sec-edgar-filings/")
    ap.add_argument("--out", default="./sec-edgar-filings")
    args = ap.parse_args()

    for t in args.tickers:
        list_filings(t.upper(), args.forms, args.limit)
        if args.risk:
            show_risk_factors(t.upper())

    if args.download:
        print("\nDownloading filings...")
        download_filings([t.upper() for t in args.tickers],
                         args.forms, args.limit, args.out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
