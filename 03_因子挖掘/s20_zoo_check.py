# -*- coding: utf-8 -*-
"""s20 因子库自检：注册表 -> 形状/覆盖率/因果性/常量性 -> 质量门。

对**全部**注册因子在抽样调仓日上跑 `registry.check_one`（含 `Ctx.truncated` 因果性检验），
输出逐因子质量表；同时打印家族/模块分布与耗时。

用法
----
    python 03_因子挖掘/s20_zoo_check.py [--dates 12] [--workers 6]
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.environ.setdefault("CLASS4_TRACK", "03_因子挖掘")

import argparse
import os
import time

import numpy as np
import pandas as pd

from core.lib import MarketData, TAB, savetab
from core.engine import Backtest
from factor_zoo.ctx import Ctx
from factor_zoo import registry
import mining_core as mc


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", type=int, default=12)
    ap.add_argument("--start", default="2021-01-01")
    a = ap.parse_args(argv)

    t0 = time.time()
    md = MarketData()
    C = Ctx(md)
    facts = registry.load_all(strict=False)
    print("注册因子 %d 个" % len(facts))

    grid = mc.monthly_grid(md, a.start, mc.BT_END)
    step = max(1, len(grid) // a.dates)
    ts = grid[::step][: a.dates]
    print("自检日期 %d 个：%s ~ %s" % (len(ts), md.dates[ts[0]].date(), md.dates[ts[-1]].date()))

    rows = []
    for i, (nm, spec) in enumerate(facts.items(), 1):
        t1 = time.time()
        r = registry.check_one(nm, spec, C, ts)
        rows.append(dict(factor=nm, family=spec["family"], module=spec["module"],
                         desc=spec.get("desc", ""), direction_prior=spec.get("direction", -1),
                         ok=r["ok"], causal=r["causal"], coverage=r["coverage"],
                         n_valid_days=r["n_valid_days"], worst_diff=r["worst_diff"],
                         reason=r["reason"], seconds=time.time() - t1))
        if not r["ok"]:
            print("  [FAIL] %-24s %s" % (nm, r["reason"]))
    tab = pd.DataFrame(rows).sort_values(["ok", "coverage"], ascending=[True, False])
    savetab(tab, "20_zoo_check.csv", index=False)

    n_ok = int(tab["ok"].sum())
    print("\n通过 %d/%d（因果性失败 %d，覆盖率不足 %d）"
          % (n_ok, len(tab), int((~tab["causal"]).sum()),
             int(((~tab["ok"]) & tab["causal"]).sum())))
    print("\n按模块：")
    g = tab.groupby("module").agg(n=("factor", "size"), ok=("ok", "sum"),
                                  覆盖中位=("coverage", "median"))
    print(g.to_string())
    print("\n按家族：")
    print(tab.groupby("family").agg(n=("factor", "size"), ok=("ok", "sum")).to_string())
    print("\n总耗时 %.1fs，明细：%s" % (time.time() - t0, os.path.join(TAB, "20_zoo_check.csv")))
    return tab


if __name__ == "__main__":
    main()
