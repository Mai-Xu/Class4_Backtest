# 公式级因子进化：产物索引

## V1：单次公式搜索（已揭盲）

- `train_formula_report.md`：2021–22 训练 / 2023–24 验证；
- `freeze.json`：V1 冻结记录；
- `oos_formula_report.md`：2025–26 一次性揭盲；
- `oos_formula_metrics.csv`、`oos_formula_nav.csv`；
- `figures/formula_evolution_summary.png`：V1 汇总图。

V1 结论：3 节点公式 `-(amtlog60 + max5d20)` 验证段有效，但 OOS 方向翻转；同预算随机公式也翻转。

## V2：Walk-Forward + 因果/反事实/成本审计（当前主结果）

位于 `v2/`：

- `report.md`：完整设计与结论；
- `walkforward_multiseed.csv`：5 个种子的进化 vs 随机；
- `walkforward_trials.csv`：全部候选与全部滚动 fold 指标；
- `failure_taxonomy.csv`：失败原因台账；
- `causal_audit.csv`：`Ctx.truncated` 因果投毒；
- `counterfactual.csv`：叶子反事实拆解；
- `pool_increment.json`：组合池增量；
- `cost_sensitivity.csv`、`cost_audit.md`：已消费样本上的成本诊断；
- `policy_memory.json`：下一轮算子权重与约束；
- `freeze_future.json`：未来窗口的预注册公式；
- `figures/v2_summary.png`：V2 汇总图。

V2 结论：随机对照 4 胜 1 平；完整公式反而不如单叶 `intra20`；因果性通过；成本暴露显著。公式搜索闭环完成，但没有证明优于同预算随机搜索。

## 阅读顺序

1. `v2/report.md`
2. `v2/figures/v2_summary.png`
3. `v2/freeze_future.json`
4. V1 `oos_formula_report.md`（已消费样本，仅作历史记录）
