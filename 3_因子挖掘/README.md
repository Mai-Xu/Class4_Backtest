# 3_因子挖掘

【轨道 C】全市场低频因子挖掘：从 70 个因子里筛出真正有信息的那几个。

| 脚本 | 做什么 | 产物 |
| --- | --- | --- |
| `s07_factor_lab.py` | **因子实验室**：把挖矿全过程做成交互式 Plotly 看板（IC 热力图 / 衰减光谱 / 十分层回测 / 留一法 / 随机对照） | `out/factor_lab.html`、`out/factor_lab_report.md`、`out/tables/07_*.csv` |
| `s08_factor_mining.py` | **挖掘主线**：70 因子 → 因果校验 → IC 筛选 → 去冗余 → 合成 → 指定口径回测 | `out/mining_report.md`、`out/tables/08_*.csv`、`out/runs/zoo_*` |
| `s09_mining_report.py` | 净值图 + 规模敏感性与规模中性化鉴定 | `out/figures/06_mining.png`（追加进报告 §7-8） |
| `s10_multifactor.py` | **【轨道 D】多因子策略**：家族 ICIR 打分 + 风险预算仓位模型（EW/IVOL60/ERC/MVO）+ 波动目标 + 完整稳健性 | `out/multifactor_report.md`、`out/figures/10_*.png`、`out/tables/10_*.csv`、`out/runs/MF5_*` |
| `audit/` | **【轨道 E】核验**：对 5 个因子模块与 s07 另写一套朴素实现逐格重算 | `audit/*_audit.md`、`audit/naive_*.py` |

## 运行

```bash
cd ..
python 3_因子挖掘/s07_factor_lab.py
python 3_因子挖掘/s08_factor_mining.py
python 3_因子挖掘/s09_mining_report.py
python 3_因子挖掘/s10_multifactor.py     # 多因子策略（约 100 秒）
```

## 因子库

`../factor_zoo/` 下按 5 个方向实现 **70 个因子**，全部通过「把数组截到 t 之后重算、逐位比对」
的因果性检验：

| 模块 | 因子数 | 内容 |
| --- | ---: | --- |
| `mom_rev.py` | 12 | 短期反转、3-1/6-1/12-1 动量、长期反转、52 周高点、隔夜/日内拆分、动量加速度 |
| `risk.py` | 14 | 波动率多窗口、特质波动率、下行波动、上下行比、最大回撤、偏度、峰度、波动率变化 |
| `liquidity.py` | 14 | 成交额水平/活跃度、Amihud、量能趋势与斜率、变异系数、单笔均额系列、流动性冲击 |
| `flow.py` | 14 | 量价相关、成交额加权动量、量价背离、OBV/MFI/CMF、筹码分布代理（获利盘/成本/集中度） |
| `trend.py` | 16 | 均线偏离与排列、回归斜率与 R²、ADX、布林带/唐奇安、路径效率、连续涨跌、跳空 |

单模块自检：

```bash
python -m factor_zoo.selfcheck risk    # 或 mom_rev / liquidity / flow / trend
```

> **数据口径**：本项目数据集**没有成交量**，只有成交额 `amt` 与成交笔数；
> 且没有股本 → **无法构造真实换手率**。OBV/MFI/CMF 等一律用成交额替代成交量，
> 所有"换手代理"实为成交额代理，源码注释已逐条标注。

## 回测口径（本次任务指定）

非 ST 全市场 · 初始资金 **10 亿元** · 佣金单边**万一** · 印花税**万五** ·
**2025-01-02 建仓 → 2026-09-30 清仓离场**（信号取 2024-12-31 收盘，次日开盘执行）。

**因子方向与筛选只用 2021-01 ~ 2024-12 的数据**，回测区间不参与选择。

核心结论见项目根 `README.md` 第十一/十二章、`out/mining_report.md` 与 `out/multifactor_report.md`。

## 单段 vs 两段加权

- `s08` 是**单段 IC 加权**（所有入选因子按样本内 IC 幅度直接加权），
  问题是流动性族占了入选因子的一半，等于把组合押在规模上；
- `s10` 是**两段式**：家族内按 ICIR 合成、家族间交给风险预算模型，并且显式检验了
  「规模中性化后还剩多少」，答案见 `out/multifactor_report.md` §5b。
