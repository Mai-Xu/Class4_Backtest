# out/ 产物索引

本目录是所有脚本的输出根（**轨道 A 除外**，它自带 `../1_数据真实性验证/out/`）。

## 报告与看板（双击即开）

| 文件 | 产出脚本 | 内容 |
| --- | --- | --- |
| `backtest_report.md` | `../2_回测系统/s03_report.py` | 主策略回测报告（复权/涨跌停/成本口径 + 对照实验） |
| `dashboard.html` | 同上 | 主策略自包含看板 |
| `vol_report.md` | `../2_回测系统/s04_vol_volume.py` + `s05_robustness.py` | 13 因子波动率+成交量策略 + 稳健性压测 |
| `vol_dashboard.html` | `../2_回测系统/s06_report.py` | 同上，自包含看板 |
| `factor_lab.html` | `../3_因子挖掘/s07_factor_lab.py` | 因子挖掘实验室（交互式 Plotly 看板） |
| `factor_lab_report.md` | 同上 | 实验室文字小结 |
| `mining_report.md` | `../3_因子挖掘/s08_factor_mining.py` + `s09_mining_report.py` | **70 因子全市场挖掘**：IC 筛选 / 去冗余 / 合成 / 规模敏感性 / 中性化鉴定 |
| `multifactor_report.md` | `../3_因子挖掘/s10_multifactor.py` | **多因子策略**：家族 ICIR 打分 + 风险预算仓位模型 + 执行层诊断 + 完整稳健性 |

## 子目录

| 目录 | 说明 | 是否入库 |
| --- | --- | --- |
| `figures/` | 全部图（编号与脚本对应：`03_*`=回测系统，`04_*`/`05_*`=波动率策略与稳健性，`06_*`=因子挖掘，`10_*`=多因子策略） | 是 |
| `tables/` | 全部表（`02_*`~`10_*` 与脚本编号对应） | 是 |
| `runs/` | 每个组合的回测明细（nav / daily / trades / blocks / holdings），共 30+ 组 | **否**（可再生，已 gitignore） |
| `cache/` | 因子面板 parquet 缓存，带 mtime 指纹自动失效 | **否**（可再生，已 gitignore） |

> 核验产物不在这里：`../核验报告.md`、`../3_因子挖掘/audit/`、`../1_数据真实性验证/audit/`。
> 核验用的 `naive_*.py` 会逐格重跑因子，耗时 30 秒~4 分钟不等。

> `runs/` 与 `cache/` 合计约 770 MB，都是可再生的中间产物，
> 已经写进 `.gitignore`；删掉后重跑对应脚本即可重建。
