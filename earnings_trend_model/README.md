# Earnings Profit → 3–5 Day Trend Model / 财报利润 → 3–5 天趋势模型

用**财报利润数据**预测个股在财报公布后 **3–5 个交易日**的价格走势（post-earnings drift，财报漂移）。

## 这是什么

给每一次历史财报事件构造一组“利润特征”，打上“未来 3/5 天涨跌”的标签，
在按时间排序的事件表上训练一个方向分类器 + 一个幅度回归器，并对最新财报打分。

- **利润特征**：EPS 惊喜度(%)、营收同比、净利润同比、净利率及其同比变化、财报前 5/20 日动量
- **标签**：入场日（公告当天或之后第一个交易日）起 3 日 / 5 日的前向收益与涨跌方向
- **模型**：`RandomForestClassifier`（方向）+ `Ridge`（幅度），按时间顺序切分训练/测试，避免未来数据泄露
- **数据源**：`yfinance`，无需 API key

## 文件

| 文件 | 说明 |
|------|------|
| `earnings_trend_model.ipynb` | 主 notebook：一键跑通数据→EDA→建模→打分（Colab 友好） |
| `earnings_trend.py` | 核心可复用模块，供 notebook 和自动化脚本调用 |
| `README.md` | 本文件 |

## 快速开始

```bash
pip install yfinance lxml pandas numpy scikit-learn matplotlib
```

```python
import earnings_trend as et

df = et.build_dataset(["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"])   # 拉数据 + 造事件表
model = et.train_models(df, horizon=5)                            # 训练 5 日模型
print("方向准确率:", round(model.test_accuracy, 3),
      " vs 基线:", round(model.baseline_accuracy, 3))
print(et.score_upcoming("AAPL", model))                           # 对最新财报打分
```

或直接打开 `earnings_trend_model.ipynb`，改顶部的 `TICKERS` 即可。

## 受限网络 / 代理环境

`yfinance` 默认用 `curl_cffi` 做浏览器 TLS 指纹；企业或 agent 代理会重置这类连接
（报 `Connection reset by peer`）。此时改用普通 `requests` 会话：

```python
import requests, earnings_trend as et
s = requests.Session(); s.headers.update({"User-Agent": "Mozilla/5.0"})
# 若代理有自签 CA:  import os; os.environ["REQUESTS_CA_BUNDLE"] = "/path/ca.crt"
et.set_session(s)
```

## 重要限制（务必先读）

- **免费 yfinance 的利润表只有约 5 个季度**，所以 `net_income_yoy` / `revenue_yoy` 等同比特征
  对**较早的**财报事件多为 NaN。实测里能稳定驱动模型的是 **EPS 惊喜度 + 财报前动量**
  （`get_earnings_dates` 提供约 25 个季度，覆盖完整）。想让基本面同比特征真正生效，需换一个
  历史更长的利润数据源（见下）。
- **样本量**：每只股票只有十几个季度财报。想要统计显著性，请把 `TICKERS` 扩到几十、上百只。
  真实的后财报漂移信号很弱，方向准确率通常就在 50% 附近波动 —— 代码如实汇报，不做粉饰。
- **财报时点**：盘前/盘后公布会影响入场日。本模型统一取“公告当天或之后第一个交易日”入场，
  衡量随后 3/5 日漂移。
- 仅用于研究，**不构成投资建议**。

## 换用 Robinhood 数据源（可选）

若已在本会话授权 Robinhood 连接器，可把利润与财报数据换成覆盖更全的接口：

- `get_earnings_results` → 替换 `get_earnings_dates`（EPS 估计/实际）
- `get_financials` → 替换 `quarterly_income_stmt`（营收、净利润）
- `get_equity_historicals` → 替换 `yf.download`（价格）

只需在 `earnings_trend.py` 里改 `get_earnings_dates` / `get_quarterly_income` /
`get_price_history` 三个函数的实现，其余特征工程与建模逻辑不变。

## 下一步可做

- walk-forward / 滚动交叉验证，替代单次时间切分
- 加入交易成本、滑点、隔夜跳空的象限分析
- 用分位数分组回测（大幅超预期 vs 大幅不及）的分组收益与胜率
- 接更长历史的基本面源，激活同比利润特征
