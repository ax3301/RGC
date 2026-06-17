#!/usr/bin/env bash
# Quick-start for new Claude Code sessions.
# Usage:  bash research/setup.sh
set -e

cd "$(dirname "$0")"
pip install -q -r requirements.txt
echo "✓ All packages installed."

# SEC requires a UA header identifying you
export SEC_EDGAR_USER_AGENT="${SEC_EDGAR_USER_AGENT:-Personal Research research@example.com}"
echo "✓ SEC_EDGAR_USER_AGENT set."

# Polygon proxy (set in env or fall back)
export PROXY_URL="${PROXY_URL:-http://43.206.151.58:8080}"
export PROXY_KEY="${PROXY_KEY:-}"
echo "✓ Proxy env vars ready (PROXY_URL=$PROXY_URL)."

echo
echo "Workflows:"
echo "  1. SEC monitor:        python research/sec_monitor.py AVGO NVDA DRAM"
echo "  2. Short-put backtest: python research/options_backtest.py DRAM --strike-pct 0.98"
echo "  3. Realtime options:   python research/realtime_options.py AVGO"
