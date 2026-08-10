#!/usr/bin/env python3
"""
Multiple-testing-aware strategy discovery pipeline for QQQ.

Guards against data-mining bias (Harvey & Liu, 2014; "...and the Cross-Section
of Expected Returns"). Steps:

  1. Enumerate a LARGE grid of candidate long/flat timing strategies
     (trend filter x entry trigger x holding period).
  2. In-sample (IS) test: for each candidate compute the daily net-of-cost P&L
     and a HAC (Newey-West) t-statistic for H0: mean daily return <= 0.
     HAC standard errors account for the autocorrelation induced by
     overlapping holding windows.
  3. Multiple-testing correction on the IS one-sided p-values:
     Benjamini-Hochberg FDR (assumes positive dependence) AND the more
     conservative Benjamini-Yekutieli FDR (valid under arbitrary dependence).
  4. Out-of-sample (OOS) confirmation on a held-out period the search never
     touched: survivors must stay profitable and significant there.
  5. Strategies passing BOTH gates are flagged 'LIVE-ELIGIBLE'.

Educational research tooling, not investment advice. Backtests overstate
live performance (slippage, regime change, survivorship).
"""

import itertools
import sys
import numpy as np
import pandas as pd
import requests

CA_BUNDLE = "/root/.ccr/ca-bundle.crt"
HEADERS = {"User-Agent": "Mozilla/5.0"}
COST = 0.0002
TD = 252
IS_END = pd.Timestamp("2016-12-31")   # in-sample: start .. 2016
OOS_START = pd.Timestamp("2017-01-01")  # out-of-sample: 2017 .. now
FDR_Q = 0.10
OOS_ALPHA = 0.05


# ---------------- data ----------------
def fetch_yahoo(symbol):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"period1": 0, "period2": 9999999999, "interval": "1d"}
    r = requests.get(url, params=params, headers=HEADERS, timeout=60, verify=CA_BUNDLE)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame(
        {"open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"]},
        index=pd.to_datetime(res["timestamp"], unit="s").normalize(),
    ).dropna(subset=["high", "close"])
    df.index.name = "date"
    return df


def rsi(series, n):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = (-delta.clip(upper=0)).rolling(n).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def load():
    vix = fetch_yahoo("%5EVIX")
    qqq = fetch_yahoo("QQQ")
    d = pd.DataFrame({
        "vix_high": vix["high"], "vix_close": vix["close"], "close": qqq["close"],
    }).dropna().sort_index()
    d["ret"] = d["close"].pct_change()
    for n in (50, 100, 200):
        d[f"sma{n}"] = d["close"].rolling(n).mean()
    d["rsi2"] = rsi(d["close"], 2)
    d["down"] = d["close"].diff() < 0
    d["roll_max20"] = d["close"].rolling(20).max()
    return d


# ---------------- candidate signals ----------------
def trend_ok(d, filt):
    if filt is None or (isinstance(filt, float) and pd.isna(filt)):
        return pd.Series(True, index=d.index)
    return d["close"] > d[f"sma{int(filt)}"]


def entry_signal(d, kind, param):
    if kind == "vix_spike":
        return d["vix_high"] >= param * d["vix_close"].shift(1)
    if kind == "vix_cross":
        return (d["vix_high"].shift(1) < param) & (d["vix_high"] >= param)
    if kind == "rsi2":
        return d["rsi2"] < param
    if kind == "down_days":
        s = d["down"]
        cond = s.copy()
        for k in range(1, int(param)):
            cond = cond & s.shift(k)
        return cond
    if kind == "pullback":
        return d["close"] <= d["roll_max20"] * (1 - param / 100.0)
    raise ValueError(kind)


def position(d, filt, kind, param, hold):
    sig = (entry_signal(d, kind, param) & trend_ok(d, filt)).fillna(False).values
    pos = np.zeros(len(d))
    remaining = 0
    for i in range(len(d)):
        if sig[i]:
            remaining = hold
        if remaining > 0:
            pos[i] = 1.0
            remaining -= 1
    return pd.Series(pos, index=d.index)


def candidates():
    filters = [None, 50, 100, 200]
    triggers = (
        [("vix_spike", k) for k in (1.05, 1.08, 1.10, 1.12, 1.15)]
        + [("vix_cross", L) for L in (17, 18, 19, 20, 22, 25)]
        + [("rsi2", th) for th in (5, 10, 15, 20)]
        + [("down_days", n) for n in (2, 3, 4)]
        + [("pullback", x) for x in (3, 5, 7)]
    )
    holds = [5, 10, 20]
    out = []
    for filt, (kind, param), hold in itertools.product(filters, triggers, holds):
        out.append({"filt": filt, "kind": kind, "param": param, "hold": hold})
    return out


# ---------------- stats ----------------
def hac_tstat(x, lag):
    """Newey-West HAC t-stat for H0: mean(x) <= 0 (one-sided)."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 30:
        return np.nan, np.nan
    xbar = x.mean()
    xc = x - xbar
    gamma0 = np.dot(xc, xc) / n
    lrv = gamma0
    for l in range(1, lag + 1):
        w = 1 - l / (lag + 1)
        g = np.dot(xc[l:], xc[:-l]) / n
        lrv += 2 * w * g
    if lrv <= 0:
        return np.nan, np.nan
    se = np.sqrt(lrv / n)
    t = xbar / se
    return t, se


def norm_sf(z):
    # one-sided upper-tail p-value via erfc (no scipy dependency)
    from math import erfc, sqrt
    return 0.5 * erfc(z / sqrt(2))


def metrics(pnl):
    pnl = pnl.dropna()
    n = len(pnl)
    if n == 0:
        return {}
    eq = (1 + pnl).cumprod()
    yrs = n / TD
    total = eq.iloc[-1] - 1
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan
    vol = pnl.std() * np.sqrt(TD)
    sharpe = (pnl.mean() * TD) / vol if vol > 0 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    return {"total": total, "cagr": cagr, "sharpe": sharpe, "maxdd": dd}


def strat_pnl(d, spec):
    pos = position(d, spec["filt"], spec["kind"], spec["param"], spec["hold"])
    trades = pos.diff().abs().fillna(0.0)
    held = pos.shift(1).fillna(0.0)
    ret = d["ret"].fillna(0.0)
    pnl = held * ret - trades * COST
    # Timing-alpha (active) return: strips the passive equity premium by
    # benchmarking against a constant-exposure position at the strategy's own
    # average exposure. mean>0 <=> the strategy is invested MORE on up days than
    # its average -- i.e. genuine market-timing skill, not just being long.
    active = (held - held.mean()) * ret
    return pnl, active, pos


# ---------------- FDR ----------------
def bh_reject(pvals, q):
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    thresh = q * (np.arange(1, m + 1)) / m
    passed = p[order] <= thresh
    kmax = np.where(passed)[0].max() + 1 if passed.any() else 0
    rej = np.zeros(m, bool)
    if kmax > 0:
        rej[order[:kmax]] = True
    return rej


def by_reject(pvals, q):
    m = len(pvals)
    c_m = np.sum(1.0 / np.arange(1, m + 1))  # harmonic number
    return bh_reject(pvals, q / c_m)


# ---------------- run ----------------
def main():
    print("Loading data ...", file=sys.stderr)
    d = load()
    d_is = d[d.index <= IS_END]
    d_oos = d[d.index >= OOS_START]
    print(f"IS : {d_is.index[0].date()} -> {d_is.index[-1].date()} ({len(d_is)} days)")
    print(f"OOS: {d_oos.index[0].date()} -> {d_oos.index[-1].date()} ({len(d_oos)} days)\n")

    specs = candidates()
    print(f"Candidate strategies: {len(specs)}\n")

    rows = []
    bh_is = metrics(d_is["ret"])
    bh_oos = metrics(d_oos["ret"])

    for i, spec in enumerate(specs):
        pnl_is, active_is, pos_is = strat_pnl(d_is, spec)
        lag = spec["hold"] + 5
        t_ret, _ = hac_tstat(pnl_is.fillna(0.0).values, lag)
        t_a, _ = hac_tstat(active_is.fillna(0.0).values, lag)
        if np.isnan(t_ret) or np.isnan(t_a):
            continue
        m_is = metrics(pnl_is)
        rows.append({
            **spec,
            "p_ret": norm_sf(t_ret),   # raw return > 0 (captures beta)
            "p": norm_sf(t_a),          # timing-alpha > 0 (skill) -- FDR gate
            "is_cagr": m_is["cagr"], "is_sharpe": m_is["sharpe"],
            "is_maxdd": m_is["maxdd"],
        })

    res = pd.DataFrame(rows)
    res["name"] = res.apply(
        lambda r: f"{r['kind']}={r['param']}|H{r['hold']}|"
                  f"F{r['filt'] if r['filt'] else '-'}", axis=1)

    # FDR correction on IS p-values
    res["bh"] = bh_reject(res["p"].values, FDR_Q)
    res["by"] = by_reject(res["p"].values, FDR_Q)
    n_beta = int((res["p_ret"] < 0.05).sum())
    n_naive = int((res["p"] < 0.05).sum())
    print("IS gate = market-timing ALPHA (raw-return premium stripped out)")
    print(f"  would pass on raw return p<0.05 (beta-driven) : {n_beta}/{len(res)}")
    print(f"  significant timing-alpha at naive p<0.05      : {n_naive}/{len(res)}")
    print(f"  survive Benjamini-Hochberg FDR<{FDR_Q} (pos. dep.) : {int(res['bh'].sum())}")
    print(f"  survive Benjamini-Yekutieli FDR<{FDR_Q} (arb. dep.): {int(res['by'].sum())}\n")

    survivors = res[res["bh"]].copy()

    # OOS confirmation on BH survivors
    oos_rec = []
    for _, r in survivors.iterrows():
        spec = {"filt": r["filt"], "kind": r["kind"],
                "param": r["param"], "hold": r["hold"]}
        pnl_oos, active_oos, _ = strat_pnl(d_oos, spec)
        t_o, _ = hac_tstat(active_oos.fillna(0.0).values, spec["hold"] + 5)
        p_o = norm_sf(t_o) if not np.isnan(t_o) else np.nan
        m_o = metrics(pnl_oos)
        oos_rec.append({
            "name": r["name"], "is_sharpe": r["is_sharpe"], "is_p": r["p"],
            "oos_cagr": m_o["cagr"], "oos_sharpe": m_o["sharpe"],
            "oos_maxdd": m_o["maxdd"], "oos_p": p_o,
            "by": bool(r["by"]),
            # OOS gate is also on timing-alpha, not raw return.
            "live": (not np.isnan(p_o)) and (p_o < OOS_ALPHA) and (m_o["cagr"] > 0),
        })
    oos = pd.DataFrame(oos_rec).sort_values("oos_sharpe", ascending=False)

    pd.set_option("display.width", 200)
    print("=" * 100)
    print(f"BENCHMARK  Buy&Hold  IS: CAGR {bh_is['cagr']*100:.1f}% Sharpe "
          f"{bh_is['sharpe']:.2f} MaxDD {bh_is['maxdd']*100:.1f}%   |   "
          f"OOS: CAGR {bh_oos['cagr']*100:.1f}% Sharpe {bh_oos['sharpe']:.2f} "
          f"MaxDD {bh_oos['maxdd']*100:.1f}%")
    print("=" * 100)

    if oos.empty:
        print("No strategies survived the FDR gate.")
    else:
        print(f"\nBH-FDR survivors carried to OOS ({len(oos)}):\n")
        show = oos.copy()
        for c in ["is_sharpe", "oos_sharpe"]:
            show[c] = show[c].map(lambda v: f"{v:.2f}")
        for c in ["oos_cagr", "oos_maxdd"]:
            show[c] = show[c].map(lambda v: f"{v*100:+.1f}%")
        for c in ["is_p", "oos_p"]:
            show[c] = show[c].map(lambda v: f"{v:.3f}" if pd.notnull(v) else "na")
        print(show[["name", "is_sharpe", "is_p", "oos_cagr", "oos_sharpe",
                    "oos_maxdd", "oos_p", "by", "live"]].to_string(index=False))

        live = oos[oos["live"]]
        print("\n" + "=" * 100)
        print(f"LIVE-ELIGIBLE (passed FDR IS-gate AND OOS p<{OOS_ALPHA} with "
              f"positive OOS CAGR): {len(live)}")
        print("=" * 100)
        if not live.empty:
            for _, r in live.iterrows():
                tag = "  [also survives BY]" if r["by"] else ""
                print(f"  {r['name']:<28} OOS CAGR {r['oos_cagr']*100:+.1f}%  "
                      f"Sharpe {r['oos_sharpe']:.2f}  MaxDD {r['oos_maxdd']*100:.1f}%"
                      f"{tag}")

    res.to_csv("pipeline_is_results.csv", index=False)
    if not oos.empty:
        oos.to_csv("pipeline_oos_survivors.csv", index=False)
    print("\nSaved pipeline_is_results.csv and pipeline_oos_survivors.csv")


if __name__ == "__main__":
    main()
