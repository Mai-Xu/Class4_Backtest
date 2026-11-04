# 组合层自进化：产物索引

## V1：单次组合搜索（已揭盲）

- `train_report.md`：训练/验证；
- `freeze.json`：冻结记录；
- `oos_report.md`、`oos_metrics.csv`、`oos_nav.csv`：2025–26 一次性揭盲。

## V2：多种子审计 + 成本敏感性

- `v2_audit_report.md`
- `v2_multiseed.csv`
- `v2_failure_taxonomy.csv`
- `v2_cost_sensitivity.csv`

## V3：inner holdout 放回搜索

- `v3_report.md`
- `v3_multiseed.csv`
- `v3_freeze_future.json`

## 结论

组合层 V2/V3 都没有证明进化稳定优于同预算随机搜索；V3 只能在未来数据上再做一次真正的独立确认，不能重复使用 2025–26。
