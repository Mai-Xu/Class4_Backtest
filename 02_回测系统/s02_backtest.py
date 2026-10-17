# -*- coding: utf-8 -*-
"""s02 跑回测：主策略组 + 复权/涨跌停的对照实验。

产物
    out/runs/<策略>/{nav,daily,trades,blocks}.csv  stats.json
    out/summary.csv        主策略与对照实验的绩效汇总
运行
    python s02_backtest.py
"""
from __future__ import annotations
# --- 项目根入 sys.path：core / factor_zoo 都是根目录下的包 ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- 产物归档到本工作的目录：out/02_回测系统/ ---
_os.environ.setdefault("CLASS4_TRACK", "02_回测系统")


import json
import os
import time

import numpy as np
import pandas as pd

from core.lib import MarketData, ROOT, TAB, savetab, RUNS
from core.engine import Backtest, CostModel
from core.strategies import (Momentum, Reversal, LowVol, Illiquidity, EqualWeightUniverse)
from core import metrics

os.makedirs(RUNS, exist_ok=True)

START, END = "2021-01-04", "2026-10-16"
CASH = 1e8
REBAL = "M"
N_HOLD = 50
MAX_W = 0.05

HONEST = dict(exec_price="open", max_participation=0.10, min_trade_notional=5e3,
              rebalance_band=2e-3, gross_limit=0.98, cash=CASH,
              use_adjusted=True, enforce_limits=True)


def dump_run(md, res, tag):
    d = os.path.join(RUNS, tag)
    os.makedirs(d, exist_ok=True)
    res.nav.rename("nav").rename_axis("date").to_frame().to_csv(
        os.path.join(d, "nav.csv"), encoding="utf-8")
    res.daily.to_csv(os.path.join(d, "daily.csv"), encoding="utf-8")
    if len(res.trades):
        res.trades.to_csv(os.path.join(d, "trades.csv"), index=False, encoding="utf-8")
    if len(res.blocks):
        res.blocks.to_csv(os.path.join(d, "blocks.csv"), index=False, encoding="utf-8")
    if len(res.holdings):
        res.holdings.to_csv(os.path.join(d, "holdings.csv"), index=False, encoding="utf-8")
    if len(res.target_log):
        res.target_log.to_csv(os.path.join(d, "target.csv"), encoding="utf-8")
    return d


def summarize(md, res, bench=None, tag=""):
    st = metrics.perf_stats(res.nav)
    row = dict(策略=tag or res.name, 年化收益=st["年化收益"], 年化波动=st["年化波动"],
               夏普=st["夏普"], 最大回撤=st["最大回撤"], 卡玛=st["卡玛"],
               日胜率=st["日胜率"])
    if len(res.daily):
        row["年换手"] = float(res.daily.turnover.mean() * 252)
        row["费用占净值"] = float(res.trades.fee.sum() / CASH) if len(res.trades) else 0.0
        row["成交笔数"] = int(len(res.trades))
        row["被挡金额万"] = float(res.blocks.dv.sum() / 1e4) if len(res.blocks) else 0.0
        row["持仓数中位"] = float(res.daily.n_holdings.median())
    if bench is not None:
        row.update(metrics.excess_stats(res.ret, bench))
    return row, st


def main():
    t_all = time.time()
    md = MarketData(verbose=True)
    bench = md.equal_weight_ret().loc[START:END]
    bench.name = "bench"

    jobs = [
        ("反转5", Reversal(lookback=5, n_hold=N_HOLD, max_weight=MAX_W)),
        ("动量20", Momentum(lookback=20, n_hold=N_HOLD, max_weight=MAX_W)),
        ("动量20_跳5", Momentum(lookback=20, skip=5, n_hold=N_HOLD, max_weight=MAX_W)),
        ("低波60", LowVol(lookback=60, n_hold=N_HOLD, max_weight=MAX_W)),
        ("低流动性20", Illiquidity(lookback=20, n_hold=N_HOLD, max_weight=MAX_W)),
    ]

    rows, stats_by = [], {}
    results = {}
    for tag, st in jobs:
        t0 = time.time()
        bt = Backtest(md, **HONEST)
        res = bt.run(st, start=START, end=END, rebalance=REBAL)
        results[tag] = res
        dump_run(md, res, tag)
        row, s = summarize(md, res, bench, tag)
        rows.append(row); stats_by[tag] = s
        print("%-12s 年化 %7.2f%%  夏普 %6.2f  回撤 %7.2f%%  用时 %.1fs"
              % (tag, 100 * s["年化收益"], s["夏普"], 100 * s["最大回撤"], time.time() - t0))

    # ---------------- 对照实验：复权 & 涨跌停 ----------------
    print("\n对照实验（以 反转5 / 动量20 为例）...")
    for tag, st in [("低流动性20", jobs[4][1]), ("反转5", jobs[0][1]), ("动量20", jobs[1][1])]:
        for label, kw in [
            ("no_adj", dict(HONEST, use_adjusted=False)),
            ("no_limit", dict(HONEST, enforce_limits=False)),
            ("no_cost", dict(HONEST, enforce_limits=False, use_adjusted=False,
                             max_participation=1.0,
                             cost=CostModel(commission_rate=0, commission_min=0,
                                            transfer_fee_rate=0, stamp_duty_rate=0,
                                            stamp_duty_rate_old=0, slippage_bps=0,
                                            impact_coef=0.0))),
            ("particip0", dict(HONEST, max_participation=1.0)),
        ]:
            name = "%s@%s" % (tag, label)
            if name in results:
                continue
            bt = Backtest(md, **kw)
            res = bt.run(st, start=START, end=END, rebalance=REBAL)
            results[name] = res
            dump_run(md, res, name)
            row, s = summarize(md, res, bench, name)
            rows.append(row)
            print("%-16s 年化 %7.2f%%  夏普 %6.2f  回撤 %7.2f%%"
                  % (name, 100 * s["年化收益"], s["夏普"], 100 * s["最大回撤"]))

    sm = pd.DataFrame(rows).set_index("策略")
    savetab(sm.round(6), "02_backtest_summary.csv")
    bench.rename("ew_ret").rename_axis("date").to_frame().to_csv(
        os.path.join(RUNS, "benchmark.csv"), encoding="utf-8")
    with open(os.path.join(RUNS, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump({"n_hold": N_HOLD, "max_weight": MAX_W, "rebal": REBAL,
                   "start": START, "end": END, "cash": CASH,
                   "honest": {k: v for k, v in HONEST.items() if k != "cash"},
                   "stats": {k: {kk: (None if isinstance(vv, float) and not np.isfinite(vv) else vv)
                                 for kk, vv in v.items()} for k, v in stats_by.items()}},
                  fh, ensure_ascii=False, indent=1, default=str)

    print("\n=== 汇总 ===")
    show = sm[["年化收益", "年化波动", "夏普", "最大回撤", "超额年化", "信息比率",
               "年换手", "费用占净值"]].copy()
    for c in show.columns:
        show[c] = show[c].astype(float).round(4)
    print(show.to_string())
    print("\n总用时 %.1fs  产物目录 %s" % (time.time() - t_all, RUNS))
    return sm


if __name__ == "__main__":
    main()
