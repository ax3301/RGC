#!/usr/bin/env bash
# 设置 6 小时自动 scan (cron)
# Usage: ./research/setup_cron.sh

set -e
cd "$(dirname "$0")/.."
RGC_DIR="$(pwd)"

mkdir -p logs

# 写 daily scan 脚本
cat > "$RGC_DIR/scripts/scheduled_scan.sh" << SCANEOF
#!/usr/bin/env bash
cd "$RGC_DIR"
source ~/.zshrc 2>/dev/null || source ~/.bashrc 2>/dev/null || true
export PROXY_URL="http://43.206.151.58:8080"
export PROXY_KEY="e5c39778fab32834e0b149f861a93157daf4f0ddb94ba710aa002211"

TIMESTAMP=\$(date -u +%Y-%m-%dT%H:%M:%SZ)
LOG="logs/scan_\$(date +%Y%m%d_%H%M).log"

echo "=== Scan started \$TIMESTAMP ===" > "\$LOG"

# 你 RH 主要持仓
TICKERS="NVDA AMD MU MRVL SMCI GFS AVGO INTC SOXX SOXL AMDL"

for t in \$TICKERS; do
    echo "" >> "\$LOG"
    echo "=== \$t ===" >> "\$LOG"
    python3 research/realtime_options.py "\$t" --expiries "\$(date -d '+30 days' +%Y-%m-%d 2>/dev/null || date -v+30d +%Y-%m-%d)" >> "\$LOG" 2>&1
done

echo "" >> "\$LOG"
echo "=== DRAM short put 状态 ===" >> "\$LOG"
python3 research/realtime_options.py DRAM --expiries 2026-07-17 2026-08-21 2026-10-16 >> "\$LOG" 2>&1

echo "=== Scan done \$(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "\$LOG"

# 保留最近 50 个日志
ls -t logs/scan_*.log 2>/dev/null | tail -n +51 | xargs rm -f 2>/dev/null
SCANEOF

mkdir -p scripts
chmod +x "$RGC_DIR/scripts/scheduled_scan.sh"

# 检查 cron 是否已经有这个 job
CRON_LINE="0 */6 * * * $RGC_DIR/scripts/scheduled_scan.sh"
if crontab -l 2>/dev/null | grep -q "scheduled_scan.sh"; then
    echo "✓ Cron job already exists"
else
    (crontab -l 2>/dev/null; echo "$CRON_LINE") | crontab -
    echo "✓ Cron job added: 每 6 小时跑一次"
fi

echo
echo "查看 cron jobs:  crontab -l"
echo "查看最新日志:    ls -t logs/scan_*.log | head -1 | xargs cat"
echo "手动跑一次:      ./scripts/scheduled_scan.sh"
echo "停止自动跑:      crontab -e (删除带 scheduled_scan.sh 的行)"
