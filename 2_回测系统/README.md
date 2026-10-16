# 2_回测系统

【轨道 B】把 `core/` 的数据层与引擎跑成**可复核的策略回测**。

| 脚本 | 做什么 | 产物 |
| --- | --- | --- |
| `s02_backtest.py` | 5 个主策略 + 12 组口径对照实验（不复权 / 忽略涨跌停 / 忽略成本 / 放开参与率） | `out/runs/`、`out/tables/02_*.csv` |
| `s03_report.py` | 汇总成报告与自包含看板 | `out/backtest_report.md`、`out/dashboard.html`、`out/figures/03_*.png` |
| `s04_vol_volume.py` | 波动率+成交量 13 因子：IC → 单因子 → 合成（样本内定方向、样本外检验） | `out/vol_report.md`、`out/tables/04_*.csv` |
| `s05_robustness.py` | **稳健性压测**：19 组参数扰动 / 成本盈亏平衡 / 随机对照 / 引擎不变量 | `out/tables/05_*.csv`、`out/figures/04_robust.png` |
| `s06_report.py` | 波动率策略看板 | `out/vol_dashboard.html`、`out/figures/05_*.png` |

## 运行

```bash
cd ..                       # 回到项目根
python 2_回测系统/s02_backtest.py
python 2_回测系统/s03_report.py
python 2_回测系统/s04_vol_volume.py
python 2_回测系统/s05_robustness.py
python 2_回测系统/s06_report.py
```

## 引擎的关键口径

- **复权记账**：用「复权份额」而不是「原始股数 × 复权价」——
  后者会把持仓价值凭空虚增（000001.SZ 的 cumadj 达 98 倍）。
- **涨跌停方向性**：一字涨停禁买、一字跌停禁卖；**反向允许**（跌停时卖盘堆积，买方反而容易成交）。
- **T+1 / 容量 / 成本**：先卖后买、参与率上限（默认 10% 当日成交额）、佣金+印花税+过户费+滑点+平方根冲击。
- 被挡下的委托**不假装成交**，逐日落库可审计。

细节见项目根 `README.md` 第一至七章。
