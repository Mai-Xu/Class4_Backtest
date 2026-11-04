# 自进化因子研究总览

本文件只做导航，详细报告在各工作目录。

## 组合层（`03_因子挖掘/evo/`）

| 版本 | 做了什么 | 结论 |
|---|---|---|
| V1 | 选因子 + 权重，59 进化 + 59 随机，冻结后一次 OOS | 随机在验证段经常更强；8 因子 OOS 年化 24.0%、夏普 1.22 |
| V2 | 5 种子、Bonferroni、block bootstrap、失败台账、成本敏感性 | 随机搜索 5/5 胜过进化组合 |
| V3 | inner holdout 放回搜索，加 gap/retention/换手惩罚 | 随机仍 5/5 胜；只能留作未来数据协议 |

导航：`03_因子挖掘/evo/README.md`、`out/03_因子挖掘/evo/README.md`、`03_因子挖掘/evo/ISSUE10_REVIEW.md`。

## 公式层（`03_因子挖掘/evo_formula/`）

| 版本 | 做了什么 | 结论 |
|---|---|---|
| V1 | 53 个因果基元 + GP 公式树；135 进化 + 135 随机 | 验证段主公式 ICIR 2.60，OOS 方向翻转为 −2.11 |
| V2 | 5 个滚动 fold、5 种子、同预算随机、因果投毒、反事实、成本审计、策略记忆 | 随机 4 胜 1 平；完整公式 outer 2.34，不如单叶 `intra20` 的 6.12 |

导航：`03_因子挖掘/evo_formula/README.md`、`out/03_因子挖掘/evo_formula/README.md`、`out/03_因子挖掘/evo_formula/v2/report.md`。

## 关键边界

- 2025–2026 已经被 V1 消费，**不能再称为新的盲测**；V2 只把它用于已看样本的成本诊断。
- 下一步真正需要的是未触碰的未来数据、walk-forward 重新训练、成本/容量约束和同预算随机基线。
- 当前成果不是“稳定 alpha”，而是一个可审计的负结果与可复用的研究 Harness。

## 一键复现

```powershell
python 03_因子挖掘/evo/train_evo.py
python 03_因子挖掘/evo/run_oos.py
python 03_因子挖掘/evo_formula/train_formula.py
python 03_因子挖掘/evo_formula/run_formula_oos.py
python 03_因子挖掘/evo_formula/formula_walkforward.py
python 03_因子挖掘/evo_formula/formula_cost_audit.py
python 03_因子挖掘/evo_formula/plot_v2_results.py
```

> 注意：`run_oos.py` 和 `run_formula_oos.py` 会消费已看过的 OOS；默认重复运行会被脚本阻止。V2 不应再读取 2025–2026 做新选择。
