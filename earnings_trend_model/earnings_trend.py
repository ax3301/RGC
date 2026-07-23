"""
Earnings profit -> 3-5 day post-earnings price trend model.

Core, reusable logic shared by the notebook and any automation.

Pipeline
--------
1. For each ticker, pull quarterly earnings events (EPS estimate vs. reported)
   and quarterly income-statement lines (revenue, net income).
2. Engineer profit features per earnings event:
      - EPS surprise (%)                  how much profit beat/missed expectations
      - Revenue YoY growth (%)            top-line trend
      - Net-income YoY growth (%)         bottom-line (profit) trend
      - Net margin & its YoY change       profitability quality
      - Pre-earnings momentum (5d / 20d)  what the tape already priced in
3. Label each event with the forward price trend over the next 3 and 5
   trading days (the post-earnings drift), plus its direction (up/down).
4. Train classifiers (direction) and regressors (magnitude) on the pooled,
   time-ordered event table and evaluate out-of-sample.

Data source is yfinance so the whole thing runs with no API keys. Every
network call is wrapped so a single bad ticker never kills the run.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Trading-day horizons for the "3-5 day trend".
HORIZONS = (3, 5)

# Default universe: liquid large caps with clean, regular earnings history.
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "AMD", "NFLX", "JPM",
]

# Optional HTTP session. In most environments yfinance's default backend is
# fine. Behind a TLS-terminating corporate/agent proxy, curl_cffi's browser
# impersonation can get reset -- call set_session() with a plain requests
# session in that case (see README).
_SESSION = None


def set_session(session):
    """Route all yfinance calls through a caller-supplied HTTP session."""
    global _SESSION
    _SESSION = session


FEATURE_COLS = [
    "eps_surprise_pct",
    "revenue_yoy",
    "net_income_yoy",
    "net_margin",
    "net_margin_yoy_chg",
    "pre_5d_return",
    "pre_20d_return",
]


# --------------------------------------------------------------------------- #
# Data collection
# --------------------------------------------------------------------------- #
def _safe(fn, default=None):
    """Run a yfinance call, swallowing network/parse errors."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - yfinance raises many things
        print(f"    ! {exc}")
        return default


def get_price_history(ticker: str, start=None, end=None) -> pd.DataFrame:
    """Daily OHLCV for a ticker as a tz-naive, date-indexed frame."""
    import yfinance as yf

    kw = {"session": _SESSION} if _SESSION is not None else {}
    df = _safe(
        lambda: yf.download(
            ticker, start=start, end=end, progress=False, auto_adjust=True, **kw
        ),
        default=pd.DataFrame(),
    )
    if df is None or df.empty:
        return pd.DataFrame()
    # yfinance may return a MultiIndex column frame for a single ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def get_earnings_dates(ticker: str, limit: int = 40) -> pd.DataFrame:
    """Earnings events with EPS estimate / reported / surprise(%)."""
    import yfinance as yf

    tk = yf.Ticker(ticker, session=_SESSION) if _SESSION is not None else yf.Ticker(ticker)
    df = _safe(lambda: tk.get_earnings_dates(limit=limit))
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.rename(
        columns={
            "EPS Estimate": "eps_estimate",
            "Reported EPS": "eps_reported",
            "Surprise(%)": "surprise_pct",
        }
    )
    return df.sort_index()


def get_quarterly_income(ticker: str) -> pd.DataFrame:
    """Quarterly income statement, oriented as date-indexed rows."""
    import yfinance as yf

    tk = yf.Ticker(ticker, session=_SESSION) if _SESSION is not None else yf.Ticker(ticker)
    stmt = _safe(lambda: tk.quarterly_income_stmt)
    if stmt is None or stmt.empty:
        return pd.DataFrame()
    df = stmt.T  # rows = report dates, columns = line items
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index()


# --------------------------------------------------------------------------- #
# Feature / label engineering
# --------------------------------------------------------------------------- #
def _nearest_income_row(income: pd.DataFrame, when: pd.Timestamp):
    """Income-statement quarter whose report date is closest to `when`."""
    if income.empty:
        return None, None
    diffs = (income.index - when).to_series().abs()
    pos = diffs.values.argmin()
    if abs((income.index[pos] - when).days) > 100:
        return None, None
    return income.index[pos], pos


def _line(row, *names):
    """First matching income-statement line item, else NaN."""
    if row is None:
        return np.nan
    for n in names:
        if n in row.index and pd.notna(row[n]):
            return float(row[n])
    return np.nan


def build_events(ticker: str) -> pd.DataFrame:
    """One row per past earnings event with profit features + forward trend."""
    earnings = get_earnings_dates(ticker)
    if earnings.empty:
        return pd.DataFrame()

    # Only events that have already been reported.
    earnings = earnings.dropna(subset=["eps_reported"])
    if earnings.empty:
        return pd.DataFrame()

    prices = get_price_history(
        ticker,
        start=(earnings.index.min() - pd.Timedelta(days=40)),
        end=(earnings.index.max() + pd.Timedelta(days=20)),
    )
    if prices.empty or "Close" not in prices:
        return pd.DataFrame()

    income = get_quarterly_income(ticker)
    close = prices["Close"]
    trading_days = close.index

    rows = []
    for edate, e in earnings.iterrows():
        # Entry day D = first trading day on/after the announcement.
        after = trading_days[trading_days >= edate]
        if len(after) == 0:
            continue
        d = after[0]
        di = trading_days.get_loc(d)
        if isinstance(di, slice):
            di = di.start

        # Pre-earnings momentum (what the tape already priced in).
        pre_5d = _ret(close, di, -5)
        pre_20d = _ret(close, di, -20)

        # Forward post-earnings drift over each horizon (the 3-5 day "trend").
        fwd = {h: _ret(close, di, h) for h in HORIZONS}
        if all(pd.isna(v) for v in fwd.values()):
            continue

        # Profit fundamentals for the matching quarter (+ the year-ago quarter).
        _, pos = _nearest_income_row(income, edate)
        rev = ni = rev_yoy = ni_yoy = margin = margin_yoy = np.nan
        if pos is not None:
            cur = income.iloc[pos]
            rev = _line(cur, "Total Revenue", "TotalRevenue", "Operating Revenue")
            ni = _line(cur, "Net Income", "NetIncome",
                       "Net Income Common Stockholders")
            margin = (ni / rev * 100.0) if rev not in (0, np.nan) else np.nan
            if pos - 4 >= 0:  # year-ago quarter for YoY
                yago = income.iloc[pos - 4]
                rev_y = _line(yago, "Total Revenue", "TotalRevenue",
                              "Operating Revenue")
                ni_y = _line(yago, "Net Income", "NetIncome",
                             "Net Income Common Stockholders")
                rev_yoy = _pct(rev, rev_y)
                ni_yoy = _pct(ni, ni_y)
                margin_y = (ni_y / rev_y * 100.0) if rev_y else np.nan
                margin_yoy = (margin - margin_y) if pd.notna(margin_y) else np.nan

        rows.append(
            {
                "ticker": ticker,
                "earnings_date": edate,
                "entry_date": d,
                "eps_estimate": e.get("eps_estimate", np.nan),
                "eps_reported": e.get("eps_reported", np.nan),
                "eps_surprise_pct": e.get("surprise_pct", np.nan),
                "revenue_yoy": rev_yoy,
                "net_income_yoy": ni_yoy,
                "net_margin": margin,
                "net_margin_yoy_chg": margin_yoy,
                "pre_5d_return": pre_5d,
                "pre_20d_return": pre_20d,
                **{f"fwd_ret_{h}d": fwd[h] for h in HORIZONS},
            }
        )

    df = pd.DataFrame(rows)
    for h in HORIZONS:
        df[f"up_{h}d"] = (df[f"fwd_ret_{h}d"] > 0).astype(int)
    return df


def _ret(close: pd.Series, i: int, offset: int):
    """Return from bar i to bar i+offset (or i+offset..i if offset<0)."""
    j = i + offset
    if j < 0 or j >= len(close) or i < 0 or i >= len(close):
        return np.nan
    a, b = (close.iloc[j], close.iloc[i]) if offset < 0 else (close.iloc[i], close.iloc[j])
    if a == 0 or pd.isna(a) or pd.isna(b):
        return np.nan
    return float(b / a - 1.0) * 100.0


def _pct(cur, prev):
    if pd.isna(cur) or pd.isna(prev) or prev == 0:
        return np.nan
    return float((cur - prev) / abs(prev) * 100.0)


def build_dataset(tickers=DEFAULT_TICKERS) -> pd.DataFrame:
    """Pooled, time-ordered event table across all tickers."""
    frames = []
    for t in tickers:
        print(f"  - {t}")
        ev = build_events(t)
        if not ev.empty:
            frames.append(ev)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("earnings_date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Modeling
# --------------------------------------------------------------------------- #
@dataclass
class ModelResult:
    horizon: int
    features: list = field(default_factory=list)
    clf: object = None
    reg: object = None
    imputer: object = None
    scaler: object = None
    test_accuracy: float = float("nan")
    baseline_accuracy: float = float("nan")
    reg_r2: float = float("nan")
    reg_mae: float = float("nan")
    importances: pd.Series = None
    report: str = ""


def train_models(df: pd.DataFrame, horizon: int, test_frac: float = 0.25):
    """Fit direction classifier + magnitude regressor for one horizon.

    Split is chronological (train on the past, test on the most recent
    events) so the score reflects real forward use, not leakage.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler

    ycol, ucol = f"fwd_ret_{horizon}d", f"up_{horizon}d"
    data = df.dropna(subset=[ycol]).copy()
    feats = [c for c in FEATURE_COLS if c in data.columns]

    n = len(data)
    if n < 20:
        return ModelResult(horizon=horizon, features=feats,
                           report=f"Only {n} events - too few to model reliably.")

    split = int(n * (1 - test_frac))
    train, test = data.iloc[:split], data.iloc[split:]

    # keep_empty_features keeps a column that is all-NaN in the train split
    # (filled with 0) so the fitted feature vector always matches `feats`.
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(train[feats])
    sc = StandardScaler().fit(imp.transform(train[feats]))
    Xtr = sc.transform(imp.transform(train[feats]))
    Xte = sc.transform(imp.transform(test[feats]))

    # Direction classifier.
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=4, min_samples_leaf=3, random_state=42
    ).fit(Xtr, train[ucol])
    pred = clf.predict(Xte)
    acc = accuracy_score(test[ucol], pred)
    # Baseline = always predict the majority class from training.
    majority = int(train[ucol].mean() >= 0.5)
    base = accuracy_score(test[ucol], np.full(len(test), majority))

    # Magnitude regressor.
    reg = Ridge(alpha=1.0).fit(Xtr, train[ycol])
    yhat = reg.predict(Xte)
    r2 = r2_score(test[ycol], yhat) if len(test) > 1 else float("nan")
    mae = mean_absolute_error(test[ycol], yhat)

    imp_series = (
        pd.Series(clf.feature_importances_, index=feats).sort_values(ascending=False)
    )
    rep = classification_report(test[ucol], pred, zero_division=0)

    return ModelResult(
        horizon=horizon, features=feats, clf=clf, reg=reg, imputer=imp, scaler=sc,
        test_accuracy=acc, baseline_accuracy=base, reg_r2=r2, reg_mae=mae,
        importances=imp_series, report=rep,
    )


def score_upcoming(ticker: str, model: ModelResult) -> dict:
    """Score the latest available fundamentals for a ticker with a fitted model.

    Uses the most recent reported earnings event's features as a stand-in for
    the next one. Returns predicted up-probability and expected N-day return.
    """
    ev = build_events(ticker)
    if ev.empty or model.clf is None:
        return {}
    row = ev.iloc[[-1]]
    X = model.scaler.transform(model.imputer.transform(row[model.features]))
    prob_up = float(model.clf.predict_proba(X)[0][1])
    exp_ret = float(model.reg.predict(X)[0])
    return {
        "ticker": ticker,
        "as_of_earnings": row["earnings_date"].iloc[0],
        "horizon_days": model.horizon,
        "prob_up": round(prob_up, 3),
        "expected_return_pct": round(exp_ret, 2),
    }
