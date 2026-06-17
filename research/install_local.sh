#!/usr/bin/env bash
# RGC research workflow — one-command local installer
# Usage:
#   curl -fsSL <raw URL of this script> | bash
#   OR: clone + bash research/install_local.sh
set -e

cd "$(dirname "$0")/.."

echo "=== RGC research workflow — local installer ==="
echo

# 1. Python 检查
if ! command -v python3 &>/dev/null; then
    echo "❌ Python3 not found. Install:"
    echo "   Mac:   brew install python3"
    echo "   Linux: sudo apt install python3 python3-pip"
    echo "   Win:   https://python.org/downloads"
    exit 1
fi
echo "✓ Python: $(python3 --version)"

# 2. 装依赖
echo
echo "Installing Python packages (1-2 min)..."
pip3 install --user --quiet -r research/requirements.txt
echo "✓ Packages installed"

# 3. 写环境变量到 shell rc
RC="$HOME/.zshrc"
[ -f "$HOME/.bashrc" ] && [ ! -f "$RC" ] && RC="$HOME/.bashrc"

if ! grep -q "PROXY_URL=http://43.206.151.58" "$RC" 2>/dev/null; then
    echo "" >> "$RC"
    echo "# RGC research workflow" >> "$RC"
    echo 'export PROXY_URL="http://43.206.151.58:8080"' >> "$RC"
    echo 'export PROXY_KEY="e5c39778fab32834e0b149f861a93157daf4f0ddb94ba710aa002211"' >> "$RC"
    echo 'export SEC_EDGAR_USER_AGENT="Personal Research research@example.com"' >> "$RC"
    echo "✓ env vars added to $RC"
else
    echo "✓ env vars already in $RC"
fi

# 4. 测试代理 (本地应该通!)
echo
echo "Testing Polygon proxy from your local machine..."
export PROXY_URL="http://43.206.151.58:8080"
export PROXY_KEY="e5c39778fab32834e0b149f861a93157daf4f0ddb94ba710aa002211"

RESULT=$(curl -s --max-time 10 -H "X-Proxy-Key: $PROXY_KEY" \
    "$PROXY_URL/v3/reference/tickers/DRAM" \
    -w "\nHTTP_CODE=%{http_code}")
CODE=$(echo "$RESULT" | grep "HTTP_CODE=" | cut -d= -f2)

if [ "$CODE" = "200" ]; then
    echo "✅ Polygon proxy WORKS from your machine!"
    echo "$RESULT" | head -2
else
    echo "⚠️  Proxy returned HTTP $CODE — check VPN/firewall"
fi

# 5. 测试 yfinance
echo
echo "Testing yfinance..."
python3 -c "
import yfinance as yf
t = yf.Ticker('DRAM')
print(f'  ✓ DRAM spot: \${t.fast_info.last_price:.2f}')
"

# 6. 测试 SEC
echo
echo "Testing SEC EDGAR..."
python3 -c "
import os
os.environ.setdefault('EDGAR_IDENTITY', 'Personal Research research@example.com')
from edgar import Company, set_identity
set_identity(os.environ['EDGAR_IDENTITY'])
co = Company('NVDA')
print(f'  ✓ NVDA: CIK {co.cik}, {co.name}')
"

echo
echo "════════════════════════════════════════════════"
echo "✅ Install complete!"
echo ""
echo "Try these commands:"
echo "  python3 research/realtime_options.py DRAM"
echo "  python3 research/options_backtest.py SMCI --years 0.45"
echo "  python3 research/sec_monitor.py AVGO --risk"
echo ""
echo "For auto-scan every 6 hours, run:"
echo "  ./research/setup_cron.sh"
echo "════════════════════════════════════════════════"
