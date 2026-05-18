"""QQQ put-selling backtest with REAL historical option prices.

Strategy:
  Each trading day T-1 at close, sell a 1DTE put expiring T with
  strike ≈ 0.98 * close(T-1) (rounded to nearest available QQQ strike).
  Premium = put's EOD close price on T-1.
  If QQQ close(T) <= strike: assigned, hold 10 trading days, then resume.

Connection: reads PROXY_URL and PROXY_KEY from env (preferred) or
falls back to direct POLYGON_API_KEY. Nothing sensitive is hard-coded.
"""
import os, json, time
from urllib.request import urlopen, Request
from urllib.error import HTTPError
from datetime import date, timedelta
import yfinance as yf
import pandas as pd

PROXY_URL = os.environ.get("PROXY_URL")
PROXY_KEY = os.environ.get("PROXY_KEY")
POLY_KEY  = os.environ.get("POLYGON_API_KEY")

if PROXY_URL and PROXY_KEY:
    BASE = PROXY_URL.rstrip("/")
    HEADERS = {"X-Proxy-Key": PROXY_KEY}
    AUTH_QS = ""
    print(f"Using proxy: {BASE}")
elif POLY_KEY:
    BASE = "https://api.massive.com"
    HEADERS = {}
    AUTH_QS = f"&apiKey={POLY_KEY}"
    print(f"Using direct Polygon/Massive API")
else:
    raise SystemExit("Set PROXY_URL+PROXY_KEY or POLYGON_API_KEY env vars")

END   = date(2026, 5, 18)
START = END - timedelta(days=400)
HOLD_DAYS = 10
STRIKE_PCT = 0.98

def http_get(url: str, retries: int = 3) -> dict | None:
    for i in range(retries):
        try:
            req = Request(url, headers=HEADERS)
            with urlopen(req, timeout=30) as r:
                return json.load(r)
        except HTTPError as e:
            if e.code == 429:
                time.sleep(15 * (i + 1))
                continue
            return None
        except Exception:
            time.sleep(2)
    return None

# ---------- QQQ EOD via yfinance ----------
qqq = yf.download("QQQ", start=START.isoformat(), end=END.isoformat(),
                  auto_adjust=False, progress=False)
if isinstance(qqq.columns, pd.MultiIndex):
    qqq.columns = qqq.columns.get_level_values(0)
qqq = qqq[["Open", "High", "Low", "Close"]].copy()
qqq["PrevClose"] = qqq["Close"].shift(1)
qqq = qqq.dropna()
one_year_ago = END - timedelta(days=365)
qqq = qqq[qqq.index.date >= one_year_ago]
print(f"QQQ window: {qqq.index[0].date()} → {qqq.index[-1].date()}  ({len(qqq)} days)")

# ---------- list available QQQ put expiries (with expired=true) ----------
# We pre-fetch every put contract for each expiration date we care about,
# so we can pick the strike that's actually listed rather than guessing.
trading_dates = [ts.date() for ts in qqq.index]
expiries = set(trading_dates)  # we use 1DTE: sell at T-1, expires T

def fetch_contracts(expiration: date) -> list[dict]:
    out = []
    url = (f"{BASE}/v3/reference/options/contracts"
           f"?underlying_ticker=QQQ&contract_type=put"
           f"&expiration_date={expiration.isoformat()}"
           f"&expired=true&limit=1000{AUTH_QS}")
    while url:
        d = http_get(url)
        if not d: break
        out.extend(d.get("results", []))
        nxt = d.get("next_url")
        url = (nxt + AUTH_QS) if nxt else None
    return out

print("\nFetching contract lists per expiry...")
contracts_by_exp = {}
for i, exp in enumerate(sorted(expiries)):
    contracts_by_exp[exp] = fetch_contracts(exp)
    if (i + 1) % 25 == 0:
        print(f"  {i+1}/{len(expiries)}  exp={exp}  n_puts={len(contracts_by_exp[exp])}")

n_with = sum(1 for v in contracts_by_exp.values() if v)
print(f"Expiries with put contracts: {n_with}/{len(contracts_by_exp)}")

# ---------- pick nearest-strike put per sell-day ----------
def nearest_put(exp: date, target_strike: float):
    pool = contracts_by_exp.get(exp, [])
    if not pool: return None
    best = min(pool, key=lambda r: abs(float(r["strike_price"]) - target_strike))
    return best

def fetch_close(ticker: str, sell_date: date) -> float | None:
    url = (f"{BASE}/v2/aggs/ticker/{ticker}/range/1/day/"
           f"{sell_date.isoformat()}/{sell_date.isoformat()}?{AUTH_QS.lstrip('&')}")
    d = http_get(url)
    if not d or d.get("resultsCount", 0) == 0:
        return None
    return float(d["results"][0]["c"])

# Map sell-day -> next-trading-day (= expiry we use)
next_dt = {trading_dates[i]: trading_dates[i + 1]
           for i in range(len(trading_dates) - 1)}

records = []
print("\nFetching put EOD prices per sell-day...")
for i, sell_d in enumerate(trading_dates):
    exp = next_dt.get(sell_d)
    if exp is None: continue
    prev_close = float(qqq.loc[qqq.index.date == sell_d, "Close"].iloc[0])
    target = STRIKE_PCT * prev_close
    contract = nearest_put(exp, target)
    if contract is None:
        records.append({"sell_date": sell_d, "expiry": exp,
                        "prev_close": prev_close, "strike": None,
                        "premium": None, "moneyness": None})
        continue
    strike = float(contract["strike_price"])
    ticker = contract["ticker"]
    premium = fetch_close(ticker, sell_d)
    records.append({"sell_date": sell_d, "expiry": exp,
                    "prev_close": prev_close, "strike": strike,
                    "ticker": ticker, "premium": premium,
                    "moneyness": strike / prev_close - 1})
    if (i + 1) % 25 == 0:
        print(f"  {i+1}/{len(trading_dates)}  sell={sell_d}  "
              f"K={strike}  prem={premium}")

prem_df = pd.DataFrame(records).set_index("sell_date")
prem_df.to_csv("/tmp/qqq_put_premiums.csv")
n_ok = prem_df["premium"].notna().sum()
print(f"\nPremium lookups: {len(prem_df)}   succeeded: {n_ok}")

# ---------- simulate ----------
hold_remaining = 0
last_close = None
pnl = 0.0
notional_sum = 0.0
days_put = days_hold = breaches = 0
log_rows = []
prev_idx = None

for ts in qqq.index:
    d = ts.date()
    close = float(qqq.loc[ts, "Close"])
    if hold_remaining > 0:
        day_pnl = close - last_close
        pnl += day_pnl
        log_rows.append((d, "HOLD", None, None, close, day_pnl, pnl))
        last_close = close
        days_hold += 1
        hold_remaining -= 1
        continue
    if prev_idx is None:
        prev_idx = d; continue
    if prev_idx not in prem_df.index:
        prev_idx = d; continue
    rec = prem_df.loc[prev_idx]
    if pd.isna(rec.get("premium")) or rec.get("strike") is None:
        prev_idx = d; continue
    strike = float(rec["strike"]); premium = float(rec["premium"])
    notional_sum += strike
    days_put += 1
    if close <= strike:
        day_pnl = premium - (strike - close)
        pnl += day_pnl
        breaches += 1
        hold_remaining = HOLD_DAYS
        last_close = close
        log_rows.append((d, "ASSIGN", strike, premium, close, day_pnl, pnl))
    else:
        pnl += premium
        log_rows.append((d, "EXPIRE", strike, premium, close, premium, pnl))
    prev_idx = d

avg_K = notional_sum / max(days_put, 1)
ret_pct = pnl / avg_K * 100
bh = (qqq["Close"].iloc[-1] / qqq["Close"].iloc[0] - 1) * 100

print(f"\n=== RESULTS (real option prices) ===")
print(f"Days selling puts: {days_put}")
print(f"Days holding QQQ:  {days_hold}")
print(f"Assignments:       {breaches}")
print(f"Total P&L / share: ${pnl:.2f}")
print(f"Avg strike:        ${avg_K:.2f}")
print(f"Return:            {ret_pct:+.2f}%")
print(f"Buy-and-hold QQQ:  {bh:+.2f}%")

log = pd.DataFrame(log_rows, columns=["date","action","strike",
                                       "premium","close","day_pnl","cum_pnl"])
log.to_csv("/tmp/qqq_put_log.csv", index=False)

print("\n--- Assignment days ---")
print(log[log["action"] == "ASSIGN"].to_string(index=False))

clean = prem_df.dropna(subset=["premium"])
print(f"\n--- Premium stats ({len(clean)} sell days) ---")
print(f"Mean:   ${clean['premium'].mean():.3f} "
      f"({clean['premium'].mean() / clean['prev_close'].mean() * 100:.3f}% of spot)")
print(f"Median: ${clean['premium'].median():.3f}")
print(f"Max:    ${clean['premium'].max():.3f}")
print(f"Min:    ${clean['premium'].min():.3f}")
print(f"Mean moneyness (K/S - 1): {clean['moneyness'].mean()*100:.2f}%")
