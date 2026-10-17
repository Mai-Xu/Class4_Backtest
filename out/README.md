# out/ 产物索引

**所有产物按工作分目录**，不再混在一起。每项工作只往 `out/<自己>/` 写，
`out/` 根目录下只有 4 个工作目录 + 一个共享缓存。

| 目录 | 属于 | 内容 |
| --- | --- | --- |
| `01_数据核验/` | 工作① | `data_audit.md`、`external_check.md`、`figures/01_*`、`tables/01_*`、`tables/02_external_*` |
| `02_回测系统/` | 工作② | `backtest_report.md`、`vol_report.md`、`dashboard.html`、`vol_dashboard.html`、`figures/03~05_*`、`tables/02~05_*`、`runs/` |
| `03_因子挖掘/` | 工作③ | `factor_lab.html`、`factor_lab_report.md`、`mining_report.md`、`figures/06_*`、`tables/07~08_*`、`runs/zoo_*` |
| `04_多因子策略/` | 工作④ | `multifactor_report.md`、`figures/10_*`、`tables/10_*`、`runs/MF5_*`、`runs/MF6_*` |
| `cache/` | 共享 | 因子面板 parquet 缓存，带 mtime 指纹自动失效。**4 项工作共用**，不按工作分。 |

## 每个工作目录里的固定结构

```
out/<工作名>/
├── figures/     图（编号与脚本编号对应：03_* ← s03，10_* ← s10）
├── tables/      表
└── runs/        回测明细（每个组合一个子目录：nav / daily / trades / blocks / holdings）
```

## 哪些进版本库

| 内容 | 是否入库 |
| --- | --- |
| 报告（`.md`）、看板（`.html`）、`figures/`、`tables/` | **是** |
| `runs/`（回测明细，共 50+ 组，体积大且可重跑） | 否（`.gitignore`：`out/*/runs/`） |
| `cache/`（因子面板 parquet，可重算） | 否（`out/cache/`） |
| `*.log`（运行日志） | 否（`out/*/*.log`） |

> 删掉 `runs/` 与 `cache/` 后重跑对应脚本即可重建；`cache/` 只影响速度，不影响结果。

## 产物怎么路由的

`core/lib.py` 在 **import 时**读环境变量 `CLASS4_TRACK`：

```python
os.environ.setdefault("CLASS4_TRACK", "04_多因子策略")   # 必须在 import core.lib 之前
from core.lib import OUT, FIG, TAB, RUNS                  # → out/04_多因子策略/{figures,tables,runs}
```

每个脚本开头都有一行这样的 `setdefault`，所以脚本从任意 cwd 运行、产物都归到自己的工作目录。
`savefig` / `savetab` 在**调用时**读模块级 `FIG` / `TAB`，目录不存在会自动创建。

唯一的跨工作读取：`03_因子挖掘/s07_factor_lab.py` 复用工作②的 `vv_*` 回测明细与 `04_/05_` 表，
所以它显式指向 `out/02_回测系统/`（源码里叫 `SRC_RUNS` / `SRC_TAB`）。
