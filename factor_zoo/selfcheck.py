# -*- coding: utf-8 -*-
"""单模块自检：python -m factor_zoo.selfcheck mom_rev

会打印每个因子的覆盖率、有效天数、是否通过因果性检验。
写因子时请反复跑这个命令，**未通过因果性检验的因子会被主线丢弃**。
"""
from __future__ import annotations
# --- 项目根入 sys.path：core / factor_zoo 都是根目录下的包 ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import importlib
import sys
import time

import numpy as np

from core.lib import MarketData
from core.engine import Backtest
from factor_zoo.ctx import Ctx
from factor_zoo.registry import check_one, FAMILY


def main(mod_name: str):
    t0 = time.time()
    md = MarketData()
    C = Ctx(md)
    # 用月频调仓日做检查（也是实际使用频率）
    ts = sorted(Backtest._rebalance_index(md.dates, "M"))
    ts = [t for t in ts if md.dates[t] >= np.datetime64("2021-01-01")][::3]   # 抽样，快一点
    mod = importlib.import_module("factor_zoo." + mod_name)
    facts = mod.FACTORS
    print("%s（%s）共 %d 个因子，检查 %d 个日期" %
          (mod_name, FAMILY.get(mod_name, "?"), len(facts), len(ts)))
    print("%-22s %8s %8s %10s %6s %s" % ("因子", "覆盖率", "有效日", "截断差", "因果", "说明"))
    bad = 0
    for name, spec in facts.items():
        r = check_one(name, spec, C, ts)
        if not r["ok"]:
            bad += 1
        print("%-22s %7.1f%% %8d %10.2e %6s %s" %
              (name, 100 * r["coverage"], r["n_valid_days"], r["worst_diff"],
               "OK" if r["causal"] else "FAIL",
               r["reason"] or spec.get("desc", "")))
    print("\n%d/%d 通过，用时 %.1fs" % (len(facts) - bad, len(facts), time.time() - t0))
    return bad


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1] if len(sys.argv) > 1 else "mom_rev") == 0 else 1)
