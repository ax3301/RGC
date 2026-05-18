"""Find days in the past year when QQQ dropped more than 2% (close-to-close)."""
import yfinance as yf
import pandas as pd
from datetime import date, timedelta

END = date(2026, 5, 18)
START = END - timedelta(days=400)

df = yf.download("QQQ", start=START.isoformat(), end=END.isoformat(),
                 auto_adjust=False, progress=False)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df[["Open", "High", "Low", "Close"]].copy()
df["PrevClose"] = df["Close"].shift(1)
df["PctChange"] = (df["Close"] / df["PrevClose"] - 1) * 100
df["IntradayLowPct"] = (df["Low"] / df["PrevClose"] - 1) * 100

one_year_ago = END - timedelta(days=365)
df = df[df.index.date >= one_year_ago]

print(f"QQQ analysis window: {df.index.min().date()} → {df.index.max().date()}")
print(f"Total trading days: {len(df)}\n")

drops_close = df[df["PctChange"] <= -2].copy()
print(f"=== Close-to-close drops ≤ -2% : {len(drops_close)} day(s) ===")
if len(drops_close):
    out = drops_close[["PrevClose", "Close", "PctChange"]].round(2)
    out.index = out.index.strftime("%Y-%m-%d")
    print(out.to_string())

print()
drops_intraday = df[df["IntradayLowPct"] <= -2].copy()
print(f"=== Intraday drops (Low vs prior Close) ≤ -2% : {len(drops_intraday)} day(s) ===")
if len(drops_intraday):
    out = drops_intraday[["PrevClose", "Low", "Close", "IntradayLowPct", "PctChange"]].round(2)
    out.index = out.index.strftime("%Y-%m-%d")
    print(out.to_string())
