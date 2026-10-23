# feathers/ · 原始行情数据（不在仓库里）

这个目录**不随仓库分发**（约 226 MB）。请把课程下发的日频行情解压到这里，或放到任意位置后
用命令行 / 环境变量指过来。目录下必须齐备这 7 个 Feather（Feather V2 / Arrow IPC）：

| 文件 | 含义 | 单位/口径 |
| --- | --- | --- |
| `open.feather` | 开盘价 | 元，**未复权** |
| `high.feather` | 最高价 | 元，未复权 |
| `low.feather` | 最低价 | 元，未复权 |
| `close.feather` | 收盘价 | 元，未复权 |
| `amt.feather` | 成交额 | 元 |
| `num_trades.feather` | 成交笔数 | 笔 |
| `cumadj.feather` | 累计复权因子 | `复权价 = 原始价 × cumadj` |

- 形状：`index = 交易日`（2020-01-02 ~ 2026-10-16，1643 行），`columns = 证券代码`（5451 列，6 位代码 + `.SH/.SZ/.BJ`）。
- 停牌日：OHLC 为前收盘携程价、`amt = 0`；未上市区间为 `NaN`。
- 当前项目的全部结果都基于这份数据；换数据后请重跑 `01 → 02 → 03 → 04` 流水线。

## 三种指向方式

`core/lib.py::resolve_dir()` 的优先级（从高到低）：

1. **命令行参数**：`python 03_因子挖掘/s21_factor_eval.py --data-dir "D:/quant/feathers"`
2. **环境变量**：`CLASS4_DATA_DIR` 或 `FEATHERS_DIR`
   （PowerShell：`$env:CLASS4_DATA_DIR="D:/quant/feathers"`；bash：`export CLASS4_DATA_DIR=/data/feathers`）
3. **自动搜索**：项目根 `feathers/` → 当前工作目录 → 逐级向上找含 `close.feather` 的目录

代码里**没有任何本机绝对路径**，报告里也只打印相对路径（`core.lib.portable_path`）。
