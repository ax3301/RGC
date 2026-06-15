# Portfolio Put-Selling Scan (6-hourly)

Scan my Robinhood portfolio and identify the 3 best put-selling opportunities given current market conditions and stock-specific news. Run end-to-end without asking me follow-up questions.

## STEP 1 — Get my positions

Call `mcp__robinhood__get_accounts`, find the default margin individual account, then call `mcp__robinhood__get_equity_positions` and `mcp__robinhood__get_portfolio` for it. If the Robinhood MCP is not available, fall back to the most-recent positions snapshot in this conversation history (NVDA 10, INTC 100, MU 8, MRVL 12, COHR 5.27, NOK 200, AMD 2, CPSH 100, FOTO 100, DRAM 132, MUU 6, AMDL 5, AVGX 10, MRVU 4, TSMX 5, ARMG 7, SOXX 4, SMCI 2.26, GFS 2). Also report my current cash / buying power.

## STEP 2 — Macro snapshot (yfinance)

Pull today's % change and last price for: `^GSPC`, `^IXIC`, `^VIX`, `SOXX`, `QQQ`. If VIX > 25 or SPY down >2% today, mark the environment as "risk-off — be cautious".

## STEP 3 — Per-position scoring

For my top 10 holdings by market value (skip positions worth < $200), gather:

| Field | Source |
|---|---|
| Last price, day % | `yf.Ticker(t).fast_info` |
| 30-day realized vol (annualized) | log returns × √252 over last 30 trading days |
| 6-month max drawdown | `(close.cummax() - close) / close.cummax()` |
| Nearest earnings date | `yf.Ticker(t).calendar` ('Earnings Date') |
| ATM IV proxy | yfinance options nearest weekly mid IV (skip if 0) |
| Weekly options OI | `option_chain(nearest_friday)` — sum if available |
| 50-day SMA, 20-day SMA | from history |
| Latest news headlines | one `WebSearch` per ticker, query `"<ticker> stock news today"`, take 1-2 most recent |

Skip any ticker where:
- earnings ≤ 14 days away (vega blowup risk)
- 6-month max drawdown > 60% AND today's % < -3% (catching a knife)
- last price < 20-day SMA AND 20-day SMA < 50-day SMA (downtrend)
- weekly options volume < 100 (illiquid)
- news contains: "fraud", "investigation", "delisting", "going concern", "missed", "guides down"

## STEP 4 — Rank surviving candidates

Score 0–10 on each axis, sum:
- **IV-rich** (5 pts): realized vol ≥ 50% → 5, 30–50 → 3, < 30 → 1
- **Trend** (2 pts): above both SMAs → 2, between → 1, below both → 0
- **Capital fit** (1 pt): 1 contract notional ≤ 50% of my BP → 1
- **Hold-already** (1 pt): I already hold the ticker (covered-put logic) → 1
- **News-clean** (1 pt): no negative headlines in last 24h → 1

Drop anything scoring < 5.

## STEP 5 — Produce concrete trades

For the top 3, propose ONE specific short put each:

- **Strike**: spot × 0.95 if IV ≥ 50%, else spot × 0.97
- **Expiry**: nearest monthly (21–35 DTE) — premiums richer than weeklies, less gamma risk
- **Estimated premium**: ATM IV × √(DTE/365) × spot × put delta ≈ 0.30 — round to nearest $0.10
- **Capital required**: strike × 100
- **Break-even**: strike – premium
- **Annualized return** if put expires worthless: (premium / capital) × (365 / DTE) × 100

## STEP 6 — Output

```
═══════════════════════════════════════════
PORTFOLIO PUT-SELLING SCAN — <UTC time>
═══════════════════════════════════════════

Account: $<NLV> total, $<BP> buying power, <N> positions
Macro: SPX <±x%>, NDX <±x%>, VIX <level> → <regime>

TOP 3 PUT-SELLING OPPORTUNITIES
───────────────────────────────

1) <TICKER>  score X/10
   spot $X.XX  IV XX%  next-earnings <date>
   Trade: Sell 1 <TICKER> <expiry> $<strike> P
          ~$X.XX credit  ($XXX)
          BE $XX.XX  ann.return ~XX%
   Why: <1 line>
   News: <1 line summary>

2) ... (same)
3) ... (same)

SKIPPED (and why) — bullet list, one per skipped ticker.

FLAGS ON EXISTING DRAM SHORT-PUT POSITIONS
───────────────────────────────────────────
For each open DRAM short put (RH 8/21 $60P naked × 1, RH 8/21 $55/$60 PCS × 5,
Fidelity 10/16 $60P × 2, Fidelity 7/17 $56P × 1), report mark-to-market PnL
and a 1-line status (on-track / at-risk / urgent).

ACTION REQUIRED?
────────────────
YES / NO. If yes, one sentence with the specific action and the trigger.
```

## Hard rules

- Do NOT place any actual orders — analysis only.
- Do NOT recommend selling puts on tickers where I'm already long > $5,000 worth (concentration risk).
- Do NOT recommend leveraged single-stock ETFs (MUU, AMDL, MRVU, AVGX, TSMX, ARMG) — too volatile for my account size, even if backtest looks good.
- If Robinhood MCP is offline, say so clearly at the top and use the fallback list above.
- If no candidates score ≥ 5, output "NO TRADES TODAY — best to wait" and explain why.
- Keep total output under 60 lines — this gets read on a phone.
- Append the run to `research/scan_history.csv` with columns: timestamp,top1_ticker,top1_score,top1_strike,top1_premium,action_required.

End of prompt.
