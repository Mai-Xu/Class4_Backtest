# 1_数据真实性验证

对 `../feathers/` 的 7 个日频矩阵做**真实性核查**：证明这份数据不仅「自洽」，而且「正确」。

回测系统（`../engine.py` / `../strategies.py` / `../s02_backtest.py` / `../s03_report.py`）
是另一套代码，不在本目录范围内。

## 三层验证

| 层级 | 手段 | 入口 | 结论 |
| --- | --- | --- | --- |
| **结构自洽** | 29 项单元测试 | `tests/test_lib.py` | 全部通过 |
| **内部一致** | 掩码对齐 / 复权去噪 / 涨跌停逐格违约 / 极端日对照真实事件 | `s01_audit.py` | 见 `out/data_audit.md` |
| **外部真值** | 与公开行情源逐格比对 | `s02_external_check.py` | 见 `out/external_check.md` |

## 目录

| 文件 | 说明 |
| --- | --- |
| `s01_audit.py` | 数据体检 → `out/data_audit.md`（9 节，正文全部为实测数字） |
| `s02_external_check.py` | 外部真值比对 → `out/external_check.md` |
| `tests/test_lib.py` | 29 项单元测试，把每个「坑」固化成断言 |
| `out/data_audit.md` | 审计报告：数据事实、口径、验证结果、已知限制 |
| `out/external_check.md` | 外部核对报告：逐股逐格比对明细 |
| `out/figures/01_*.png` | 4 张图（8 个面板）：股票池规模、停牌/一字板、复权噪声、涨跌停、行情对照 |
| `out/tables/01_*.csv` | 掩码统计、涨跌停诊断、流动性、逐日市场状态 |
| `out/tables/02_external_*.csv` | 外部比对的逐股明细（价格 / 成交额 / 复权收益） |

## 运行

```bash
cd 1_数据真实性验证

python s01_audit.py                                   # 数据体检（约 17 秒）
python s02_external_check.py                          # 外部比对（联网，约 30 秒）
python s02_external_check.py --sample 12 --random 18  # 扩到 30 只样本
python -m unittest discover -s tests -p "test_lib.py" -v
```

依赖 `pandas` / `numpy` / `pyarrow` / `matplotlib`（Python 3.13）。
`s02_external_check.py` 需要联网，**回测本身不依赖网络**。

## 外部核对结果（30 只证券：12 只固定大盘股 + 18 只随机抽取）

| 核对项 | 外部来源 | 规模 | 结果 |
| --- | --- | --- | --- |
| 不复权开 / 高 / 低 / 收 | 腾讯行情日线 | 44,131 个证券日 = **176,524 格** | **逐格完全一致，最大价差 0.000000 元** |
| 成交额（`VWAP` 必落在 `[low, high]` 的数学约束） | 腾讯行情成交量 | 44,131 个证券日 | **0 越界** |
| 复权收益 | 腾讯前复权 | 640 日 x 30 只，89 个除权日 | 与外部一致；同样这些日子用**未复权**收益比会差到 **30.5%** |

## 关于 `lib.py`

数据层 `lib.py` **保留在上一级目录**，没有搬进来：

- `engine.py` / `strategies.py` / `s02_backtest.py` / `s03_report.py` / `tests/test_engine.py`
  都 `from lib import ...`，搬走会直接报错；
- `lib.py` 定义了项目共用的 `ROOT / FIG / TAB` 输出约定，搬家会把回测系统的产物
  一起挪位置。

本目录的脚本通过把上级目录挂进 `sys.path` 来引用它，并把 `savefig` / `savetab`
的输出重定向到本目录的 `out/`，因此**两边互不干扰**。
