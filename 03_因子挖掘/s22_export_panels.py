# -*- coding: utf-8 -*-
"""s22 导出：给下游「多因子打分选股程序」用的因子字典 + 面板 + 读取 API。

产物
----
out/03_因子挖掘/tables/22_factor_dict.csv    因子字典（口径/家族/IC/簇/代表标记）
out/cache/zoo2/factor_dict.json              同上（JSON，供程序读取）
out/cache/zoo2/panel_manifest.csv            因子名 -> 面板 parquet 路径 + sha1 + 形状
out/03_因子挖掘/tables/22_usage.md           下游用法说明（含代码片段）
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.environ.setdefault("CLASS4_TRACK", "03_因子挖掘")

import argparse
import glob
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

from core.lib import MarketData, CACHE, TAB, savetab
from factor_zoo import registry
from factor_zoo.ctx import Ctx
from factor_zoo.panels import build_panels_parallel
import mining_core as mc

PANEL_DIR = os.path.join(CACHE, "zoo2")


def sha1(p, nbytes=1 << 20):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        while True:
            b = f.read(nbytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    t0 = time.time()
    md = MarketData()
    ts = mc.monthly_grid(md, mc.GRID_START, mc.BT_END)
    facts = registry.load_all(strict=False)
    print("注册因子 %d 个，网格 %d 个日期" % (len(facts), len(ts)))
    panels = build_panels_parallel(md, facts, ts, cache_dir=PANEL_DIR,
                                   workers=a.workers, verbose=True, force=a.force)

    stats = None
    p = os.path.join(TAB, "21_factor_stats.csv")
    if os.path.exists(p):
        stats = pd.read_csv(p).set_index("factor")

    rows = []
    for nm, spec in facts.items():
        f = sorted(glob.glob(os.path.join(PANEL_DIR, "zoo2_%s_*.parquet" % nm)))
        P = panels[nm].values if nm in panels else None
        cov = float(np.isfinite(P).mean()) if P is not None else np.nan
        rec = dict(factor=nm, family=spec["family"], module=spec["module"],
                   desc=spec.get("desc", ""), direction_prior=int(spec.get("direction", -1)),
                   panel_file=(os.path.basename(f[-1]) if f else ""),
                   panel_rows=(P.shape[0] if P is not None else 0),
                   panel_cols=(P.shape[1] if P is not None else 0),
                   coverage_panel=cov)
        if stats is not None and nm in stats.index:
            s = stats.loc[nm]
            rec.update(dict(coverage_is=s.get("coverage_is"), ic20=s.get("ic20"),
                            icir=s.get("icir"), ic20_adj=s.get("ic20_adj"),
                            oos_ic20=s.get("oos_ic20"),
                            direction_is=s.get("direction_is"),
                            ls_net_ann=s.get("ls_net_ann"),
                            stability=s.get("stability"),
                            cluster=s.get("cluster"), is_rep=s.get("is_rep")))
        rows.append(rec)
    d = pd.DataFrame(rows)
    savetab(d, "22_factor_dict.csv", index=False)

    man = []
    for nm, spec in facts.items():
        f = sorted(glob.glob(os.path.join(PANEL_DIR, "zoo2_%s_*.parquet" % nm)))
        if not f:
            continue
        fp = f[-1]
        man.append(dict(factor=nm, file=os.path.relpath(fp, CACHE), bytes=os.path.getsize(fp),
                        sha1=sha1(fp)))
    savetab(pd.DataFrame(man), "panel_manifest.csv")

    with open(os.path.join(PANEL_DIR, "factor_dict.json"), "w", encoding="utf-8") as fh:
        json.dump(d.replace({np.nan: None}).to_dict(orient="records"), fh,
                  ensure_ascii=False, indent=1)

    usage = """# 下游「多因子打分选股程序」用法

因子面板与字典都在 `out/cache/zoo2/`：

- `zoo2_<因子名>_<模块mtime>.parquet`：`index=调仓信号日(月末最后一个交易日)`、
  `columns=证券代码`、`values=因子原值(float32)`。
- `factor_dict.json` / `../03_因子挖掘/tables/22_factor_dict.csv`：每个因子的家族、口径、
  样本内 IC/ICIR、样本内方向 `direction_is`、冗余簇 `cluster`、是否簇代表 `is_rep`。
- `panel_manifest.csv`：面板文件清单 + sha1。

推荐用法（`factor_zoo/api.py` 已封装）：

```python
from factor_zoo.api import load_factor_dict, load_panel_matrix, universe_mask

fd = load_factor_dict()                     # DataFrame，index=因子名
names = fd[(fd.is_rep) & (fd.ic20.abs() >= 0.02)].index.tolist()
X = load_panel_matrix(names)                # {因子名: DataFrame(日期 x 证券)}
uni = universe_mask()                       # DataFrame(bool)，同一日期索引
```

打分要点：

1. **只用样本内（2021-01 ~ 2024-12）的 IC/ICIR 定方向与权重**，`direction_is` 已经给出符号；
2. 横截面先做秩 z（`mining_core.cs_rank_z`），再按家族合成；
3. 冗余簇（`cluster`）只取代表（`is_rep=True`），避免同一 bet 重复下注；
4. 需要规模中性化时对 `log(ADV20)` 做横截面回归取残差（见 `s23_multifactor_bt.py`）。
"""
    with open(os.path.join(TAB, "22_usage.md"), "w", encoding="utf-8") as fh:
        fh.write(usage)

    print("导出因子 %d 个，面板文件 %d 个，用时 %.1fs" % (len(d), len(man), time.time() - t0))
    print("  字典：%s" % os.path.join(PANEL_DIR, "factor_dict.json"))
    print("  清单：%s" % os.path.join(TAB, "panel_manifest.csv"))
    return d


if __name__ == "__main__":
    main()
