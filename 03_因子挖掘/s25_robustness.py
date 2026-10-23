# -*- coding: utf-8 -*-
"""s25 稳健性：参数敏感性 / 成本 / 家族留一 / 随机对照分布。

    python 03_因子挖掘/s25_robustness.py [--workers 6]

产物：out/03_因子挖掘/tables/25_*.csv
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.environ.setdefault("CLASS4_TRACK", "03_因子挖掘")

import argparse
import importlib.util
import os
import time

import numpy as np
import pandas as pd

from core.lib import MarketData, CACHE, TAB, RUNS, savetab
from core.engine import Backtest
from core import metrics
from factor_zoo import registry
from factor_zoo.ctx import Ctx
from factor_zoo.panels import build_panels_parallel
import mining_core as mc

_here = _os.path.dirname(_os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("s23", _os.path.join(_here, "s23_multifactor_bt.py"))
s23 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s23)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--random-controls", type=int, default=0,
                    help="随机对照组数（默认 0：随机对照由 s23 统一产出）")
    ap.add_argument("--quick", action="store_true", help="只跑基础与成本两组")
    a = ap.parse_args(argv)

    t0 = time.time()
    md = MarketData()
    C = Ctx(md)
    ts = mc.monthly_grid(md, mc.GRID_START, mc.BT_END)
    dates = md.dates[ts]
    facts = registry.load_all(strict=False)
    panels = build_panels_parallel(md, facts, ts, cache_dir=os.path.join(CACHE, "zoo2"),
                                   workers=a.workers, verbose=False)
    uni_df = md.universe(**mc.UNI_KW)
    uni = pd.DataFrame(np.asarray(uni_df.values, bool), index=md.dates, columns=md.symbols)
    uni_g = np.asarray(uni.values, bool)[list(ts)]
    adv_log = pd.DataFrame(np.log(np.maximum(md.adv(20).values.astype("float64"), 1.0)),
                           index=md.dates, columns=md.symbols)
    adv_g = np.asarray(adv_log.values, dtype="float64")[list(ts)]
    stats = pd.read_csv(os.path.join(TAB, "21_factor_stats.csv"))

    sig = [d for d in dates if pd.Timestamp(mc.SIGNAL0) <= d <= pd.Timestamp("2026-08-31")]
    liq = pd.Timestamp(mc.LIQ_SIGNAL)
    rebal = sorted(set(sig + [liq]))
    bench_ret = md.ret.where(uni).mean(axis=1)
    # 基准绩效（同一起点，用于和组合 CAGR 相减得到超额）
    _b0 = pd.Timestamp(mc.SIGNAL0)
    _b1 = pd.Timestamp(mc.BT_END)          # 必须截到回测末日，否则基准多算了 10 月上半月的下跌
    _bench = bench_ret.loc[_b0:_b1]
    _bn = (1 + _bench.fillna(0)).cumprod()
    STB = metrics.perf_stats(_bn, _bench)

    score_cache = {}

    def get_score(ns=False, exclude=None):
        key = (ns, tuple(sorted(exclude or [])))
        if key not in score_cache:
            sc, det = s23.build_score_panels(
                md, C, facts, panels, stats, ts, uni_g, adv_g if ns else None,
                exclude_families=exclude, verbose=False)
            score_cache[key] = (sc, det)
        return score_cache[key]

    def one(name, ns=False, exclude=None, n_hold=1700, cap=0.02, buffer=1.0,
            exec_window=5, participation=0.10, slippage=0.0, impact=0.0,
            commission=mc.COMMISSION, n_rand=0):
        sc, det = get_score(ns, exclude)
        strat = s23.ScoreStrategy(sc, uni, dates, n_hold=n_hold, max_weight=cap,
                                  buffer_mult=buffer, liq_date=liq)
        cost = s23.CostModel(commission_rate=commission, commission_min=0.0,
                             transfer_fee_rate=0.0, stamp_duty_rate=mc.STAMP,
                             stamp_duty_rate_old=mc.STAMP, slippage_bps=slippage,
                             impact_coef=impact)
        res = s23.run_backtest(md, strat, cost, rebalance=rebal,
                               exec_window=exec_window, participation=participation,
                               n_hold=n_hold)
        st = metrics.perf_stats(res.nav, res.ret)
        rr = res.ret.loc[res.ret.index > _b0]
        ex = metrics.excess_stats(rr, bench_ret.reindex(rr.index))
        excess = st["年化收益"] - STB["年化收益"]
        te = ex.get("跟踪误差", np.nan)
        row = dict(配置=name, 年化=st["年化收益"], 夏普=st["夏普"], 最大回撤=st["最大回撤"],
                   卡玛=st["卡玛"], 超额年化=excess,
                   信息比率=(excess / te) if (te and np.isfinite(te) and te > 0) else np.nan,
                   年换手=res.daily["turnover"].sum() / (len(res.daily) / 252.0),
                   平均持仓数=float(res.daily["n_holdings"].mean()),
                   平均仓位=float(1 - (res.daily["cash"] / res.daily["nav"]).mean()),
                   费用万元=float(res.trades["fee"].sum() / 1e4) if len(res.trades) else 0.0,
                   被挡金额万元=float(res.blocks["dv"].sum() / 1e4) if len(res.blocks) else 0.0)
        if n_rand:
            rr = []
            rng = np.random.default_rng(20260101)
            for i in range(n_rand):
                rs = s23.RandomStrategy(sc, uni, dates, n_hold=n_hold, max_weight=cap,
                                        buffer_mult=1.0, liq_date=liq,
                                        seed=int(rng.integers(1 << 30)))
                r = s23.run_backtest(md, rs, cost, rebalance=rebal,
                                     exec_window=exec_window, participation=participation,
                                     n_hold=n_hold)
                s2 = metrics.perf_stats(r.nav, r.ret)
                rr.append(dict(run=i, 年化=s2["年化收益"], 夏普=s2["夏普"],
                               最大回撤=s2["最大回撤"]))
            rd = pd.DataFrame(rr)
            savetab(rd, "25_random_controls.csv", index=False)
            row["随机年化中位"] = float(rd["年化"].median())
            row["随机年化95分位"] = float(rd["年化"].quantile(.95))
            row["分位排名"] = float((rd["年化"] < st["年化收益"]).mean())
        return row

    rows = []
    print("== 基础 ==")
    rows.append(one("基准 MF1700_raw", ns=False))
    rows.append(one("基准 MF1700_ns", ns=True))
    if not a.quick:
        print("== 持仓数 ==")
        for n in (100, 200, 400, 800, 2500):
            rows.append(one("持仓数 %d" % n, n_hold=n))
        print("== 缓冲带 ==")
        for b in (1.5, 2.0):
            rows.append(one("缓冲带 %.1f" % b, buffer=b))
        print("== 执行窗口 ==")
        for ew in (3, 10):
            rows.append(one("执行窗口 %d 日" % ew, exec_window=ew))
        print("== 参与率 ==")
        for p in (0.05, 0.20):
            rows.append(one("参与率 %.0f%%" % (100 * p), participation=p))
        print("== 成本 ==")
        rows.append(one("佣金万二", commission=2e-4))
        rows.append(one("滑点10bp+冲击", slippage=10.0, impact=0.01))
        rows.append(one("滑点20bp+冲击", slippage=20.0, impact=0.02))
        print("== 家族留一（raw）==")
        for fam in sorted(stats["family"].unique()):
            rows.append(one("留一：去掉%s" % fam, exclude=[fam]))
    if a.random_controls > 0:
        print("== 随机对照分布（%d 组）==" % a.random_controls)
        rows.append(one("基准+随机对照", n_rand=a.random_controls))

    tab = pd.DataFrame(rows)
    savetab(tab, "25_robustness.csv", index=False)
    print("\n" + tab.round(4).to_string(index=False))
    print("\n用时 %.1fs" % (time.time() - t0))
    return tab


if __name__ == "__main__":
    main()
