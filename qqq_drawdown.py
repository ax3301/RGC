"""Find days in the past year when QQQ dropped more than 2% (close-to-close),
and measure the forward return 10 trading days later."""
import yfinance as yf
import pandas as pd
from datetime import date, timedelta

END = date(2026, 5, 18)
START = END - timedelta(days=400)
FORWARD_DAYS = 10

df = yf.download("QQQ", start=START.isoformat(), end=END.isoformat(),
                 auto_adjust=False, progress=False)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df[["Open", "High", "Low", "Close"]].copy()
df["PrevClose"] = df["Close"].shift(1)
df["PctChange"] = (df["Close"] / df["PrevClose"] - 1) * 100
df["IntradayLowPct"] = (df["Low"] / df["PrevClose"] - 1) * 100

# Forward N-trading-day close, and forward return relative to the drop day's close.
df["FwdClose"] = df["Close"].shift(-FORWARD_DAYS)
df["FwdDate"] = df.index.to_series().shift(-FORWARD_DAYS)
df["FwdReturnPct"] = (df["FwdClose"] / df["Close"] - 1) * 100

one_year_ago = END - timedelta(days=365)
window = df[df.index.date >= one_year_ago]

print(f"QQQ analysis window: {window.index.min().date()} → {window.index.max().date()}")
print(f"Total trading days: {len(window)}\n")

drops = window[window["PctChange"] <= -2].copy()
print(f"=== Close-to-close drops ≤ -2% : {len(drops)} day(s) ===")
print(f"=== Forward return over next {FORWARD_DAYS} trading days ===\n")

view = drops[["PrevClose", "Close", "PctChange", "FwdDate", "FwdClose", "FwdReturnPct"]].copy()
view["FwdDate"] = view["FwdDate"].dt.strftime("%Y-%m-%d").fillna("(not yet)")
view.index = view.index.strftime("%Y-%m-%d")
view = view.round(2)
view.columns = ["PrevClose", "Close", "Drop%", f"+{FORWARD_DAYS}d Date",
                f"+{FORWARD_DAYS}d Close", f"+{FORWARD_DAYS}d Return%"]
print(view.to_string())

realized = drops.dropna(subset=["FwdReturnPct"])
if len(realized):
    print(f"\nMean fwd return: {realized['FwdReturnPct'].mean():.2f}%")
    print(f"Median fwd return: {realized['FwdReturnPct'].median():.2f}%")
    print(f"Win rate (>0): {(realized['FwdReturnPct'] > 0).mean() * 100:.0f}%")
