# -*- coding: utf-8 -*-
"""s26 机器学习打分：用树模型把 354 个因子自动组合，并与 ICIR 加权打分对照。

思路（防过拟合）
----------------
- 特征：s21 筛出的非冗余因子（`is_rep` 且样本内 |IC20|≥0.02），当日横截面秩 z；
- 标签：**可成交前视收益** `ADJ_O[t+21]/ADJ_O[t+1]-1`，并在当日横截面内**去均值**（学相对强弱）；
- 训练集：只用 2021-01 ~ 2024-12 的样本内数据（严格 OOS）；
- 模型：`HistGradientBoostingRegressor`（原生支持 NaN，无需插补）；
- 报告：IS 的 3 折时序 CV IC 与 OOS IC（看有没有过拟合），再跑同一口径回测与 ICIR 打分对照。

产物：out/03_因子挖掘/tables/26_*.csv、runs/26_ml/
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.environ.setdefault("CLASS4_TRACK", "03_因子挖掘")

import argparse
import importlib.util
import json
import os
import time

import numpy as np
import pandas as pd

from core.lib import MarketData, CACHE, TAB, RUNS, savetab
from core import metrics
from factor_zoo import registry
from factor_zoo.ctx import Ctx
from factor_zoo.panels import build_panels_parallel
import mining_core as mc

_here = _os.path.dirname(_os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("s23", _os.path.join(_here, "s23_multifactor_bt.py"))
s23 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s23)


def build_xy(panels, names, ts, uni, fwd20):
    """把 (期数 x 证券 x 因子) 摊成行样本：X (N, n_factors)、y (N,)、date_idx (N,)、col_idx (N,)。"""
    Xs, ys, di, ci = [], [], [], []
    for i, t in enumerate(ts):
        ok = uni[i] & np.isfinite(fwd20[i])
        idx = np.flatnonzero(ok)
        if len(idx) < 100:
            continue
        rows = np.empty((len(idx), len(names)), dtype="float32")
        for a, nm in enumerate(names):
            v = np.asarray(panels[nm].values[i], dtype="float64")
            z = np.full(v.shape, np.nan)
            m = np.flatnonzero(uni[i] & np.isfinite(v))
            if len(m) > 20:
                z[m] = mc.cs_rank_z(v[m])
            rows[:, a] = z[idx].astype("float32")
        y = fwd20[i][idx]
        Xs.append(rows)
        ys.append(y - np.nanmean(y))          # 横截面去均值：学相对强弱
        di.append(np.full(len(idx), i))
        ci.append(idx)
    return (np.vstack(Xs), np.concatenate(ys), np.concatenate(di), np.concatenate(ci))


def rank_ic_by_date(pred, y, di, dates_idx):
    out = {}
    for i in np.unique(di):
        m = di == i
        out[i] = mc.spearman(pred[m], y[m], min_n=50)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--n-hold", type=int, default=1700)
    ap.add_argument("--min-abs-ic", type=float, default=0.02)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import TimeSeriesSplit

    t0 = time.time()
    md = MarketData()
    C = Ctx(md)
    ts = mc.monthly_grid(md, mc.GRID_START, mc.BT_END)
    dates = md.dates[ts]
    facts = registry.load_all(strict=False)
    panels = build_panels_parallel(md, facts, ts, cache_dir=os.path.join(CACHE, "zoo2"),
                                   workers=a.workers, verbose=False)
    uni = np.asarray(md.universe(**mc.UNI_KW).values, bool)[list(ts)]
    fwd20 = mc.fwd_returns(md, ts, (20,))[20]
    stats = pd.read_csv(os.path.join(TAB, "21_factor_stats.csv"))
    sel = stats[(stats["valid"]) & (stats["coverage_is"] >= 0.6)
                & (stats["ic20"].abs() >= a.min_abs_ic)]
    if "is_rep" in sel.columns:
        sel = sel[sel["is_rep"].astype(bool)]
    names = [n for n in sel["factor"].tolist() if n in panels]
    print("特征因子 %d 个" % len(names))

    # 训练集只能用「标签已经实现」的日期：标签是 t+21 个交易日后的开盘价，
    # 所以 2024-12-31 之后才实现的标签不能进训练集，否则 2024-12-31 当天的预测
    # 会用到 2025-01 的信息（细节泄漏）。这样切分后，2024-12-31（建仓信号日）
    # 也是可预测的，回测才能在 2025-01-02 开盘建仓，与线性口径完全可比。
    pos_end = int(md.dates.get_loc(pd.Timestamp(mc.IS_END)))
    is_rows = np.array([int(t) + 21 <= pos_end for t in ts])       # 训练（标签已实现）
    oos_rows = ~is_rows                                            # 预测（含建仓信号日）
    print("训练用信号日 %d 个（末位 %s），预测用 %d 个（首位 %s）"
          % (is_rows.sum(), dates[is_rows][-1].date(), oos_rows.sum(), dates[oos_rows][0].date()))
    X, y, di, ci = build_xy(panels, names, ts, uni, fwd20)
    tr = is_rows[di]
    print("样本：训练 %d 行 / 预测 %d 行，特征 %d 个（%.1fs）"
          % (tr.sum(), (~tr).sum(), X.shape[1], time.time() - t0))

    def mk():
        return HistGradientBoostingRegressor(
            max_iter=a.max_iter, learning_rate=a.lr, max_depth=a.max_depth,
            l2_regularization=a.l2, early_stopping=False, random_state=20260101)

    # ---- IS 时序 CV（3 折）看样本内泛化 ----
    cv_rows = []
    tscv = TimeSeriesSplit(n_splits=3)
    Xis, yis, diis = X[tr], y[tr], di[tr]
    udates = np.unique(diis)
    for k, (i_tr, i_va) in enumerate(tscv.split(udates), 1):
        d_tr, d_va = set(udates[i_tr]), set(udates[i_va])
        m_tr = np.isin(diis, list(d_tr))
        m_va = np.isin(diis, list(d_va))
        mdl = mk().fit(Xis[m_tr], yis[m_tr])
        va_idx = np.flatnonzero(m_va)
        p = mdl.predict(Xis[m_va])
        ics = [mc.spearman(p[diis[va_idx] == d], yis[va_idx][diis[va_idx] == d], min_n=50)
               for d in sorted(d_va)]
        cv_rows.append(dict(fold=k, n_train=int(m_tr.sum()), n_valid=int(m_va.sum()),
                            ic=float(np.nanmean(ics)), n_dates=len(d_va)))
        print("  CV 折 %d：训练 %d 行 / 验证 %d 行，验证期平均 IC %.4f"
              % (k, m_tr.sum(), m_va.sum(), np.nanmean(ics)))
    savetab(pd.DataFrame(cv_rows), "26_ml_cv.csv", index=False)

    # ---- 最终模型：IS 全量训练 -> OOS 预测 ----
    mdl = mk().fit(X[tr], y[tr])
    pred = np.full(len(y), np.nan)
    pred[~tr] = mdl.predict(X[~tr])
    oos_ic = [mc.spearman(pred[~tr & (di == d)], y[~tr & (di == d)], min_n=50)
              for d in np.unique(di[~tr])]
    tr_idx = np.flatnonzero(tr)
    is_pred = mdl.predict(X[tr])
    is_ic = [mc.spearman(is_pred[di[tr_idx] == d], y[tr_idx][di[tr_idx] == d], min_n=50)
             for d in np.unique(di[tr])]
    print("  IS 全量拟合 IC %.4f（含拟合内样本，偏高）| OOS IC %.4f"
          % (np.nanmean(is_ic), np.nanmean(oos_ic)))

    # ---- 组装打分面板 ----
    sc = np.full((len(ts), len(md.symbols)), np.nan)
    for j in range(len(y)):
        if not tr[j]:
            sc[di[j], ci[j]] = pred[j]
    for i in range(len(ts)):
        m = np.flatnonzero(uni[i] & np.isfinite(sc[i]))
        if len(m) > 20:
            sc[i, m] = mc.cs_rank_z(sc[i, m])
    score = pd.DataFrame(sc, index=dates, columns=md.symbols)

    # ---- 回测（与 s23 同口径）----
    uni_df = pd.DataFrame(uni, index=dates, columns=md.symbols)
    sig = [d for d in dates if pd.Timestamp(mc.SIGNAL0) <= d <= pd.Timestamp("2026-08-31")]
    liq = pd.Timestamp(mc.LIQ_SIGNAL)
    rebal = sorted(set(sig + [liq]))
    strat = s23.ScoreStrategy(score, uni_df, dates, n_hold=a.n_hold, max_weight=0.02,
                              buffer_mult=1.0, liq_date=liq)
    strat.name = "ML%d%s" % (a.n_hold, a.tag)
    res = s23.run_backtest(md, strat, s23.cost_model(), rebalance=rebal,
                           exec_window=5, participation=0.10, n_hold=a.n_hold)
    res_slip = s23.run_backtest(md, strat, s23.cost_model(slippage_bps=10.0, impact_coef=0.01),
                                rebalance=rebal, exec_window=5, participation=0.10,
                                n_hold=a.n_hold)
    res_slip.name = res.name + "_slip10bp"
    bench_ret = md.ret.where(md.universe(**mc.UNI_KW)).mean(axis=1)
    bench_ret = bench_ret.loc[res.nav.index[0]:res.nav.index[-1]]

    stb = metrics.perf_stats((1 + bench_ret.fillna(0)).cumprod(), bench_ret)
    rows = []
    for nm, r in ((res.name, res), (res_slip.name, res_slip)):
        st = metrics.perf_stats(r.nav, r.ret)
        rr = r.ret.loc[r.ret.index > pd.Timestamp(mc.SIGNAL0)]
        ex = metrics.excess_stats(rr, bench_ret.reindex(rr.index))
        excess = st["年化收益"] - stb["年化收益"]
        te = ex.get("跟踪误差", np.nan)
        rows.append(dict(组合=nm, 累计=st["累计收益"], 年化=st["年化收益"], 波动=st["年化波动"],
                         夏普=st["夏普"], 最大回撤=st["最大回撤"], 卡玛=st["卡玛"],
                         日胜率=st["日胜率"], 超额年化=excess, 跟踪误差=te,
                         信息比率=(excess / te) if (te and np.isfinite(te) and te > 0) else np.nan,
                         年换手=r.daily["turnover"].sum() / (len(r.daily) / 252.0)))
    rows.append(dict(组合="BENCH_池内等权", 累计=stb["累计收益"], 年化=stb["年化收益"],
                     波动=stb["年化波动"], 夏普=stb["夏普"], 最大回撤=stb["最大回撤"],
                     卡玛=stb["卡玛"], 日胜率=stb["日胜率"], 超额年化=0.0,
                     跟踪误差=0.0, 信息比率=np.nan, 年换手=0.0))
    summ = pd.DataFrame(rows)
    savetab(summ, "26_summary%s.csv" % a.tag, index=False)


    run_dir = os.path.join(RUNS, "26_ml%s" % a.tag)
    os.makedirs(run_dir, exist_ok=True)
    for nm, r in ((res.name, res), (res_slip.name, res_slip)):
        d = os.path.join(run_dir, nm)
        os.makedirs(d, exist_ok=True)
        r.nav.rename("nav").to_frame().to_csv(os.path.join(d, "nav.csv"), encoding="utf-8-sig")
        r.daily.to_csv(os.path.join(d, "daily.csv"), encoding="utf-8-sig")
        r.trades.to_csv(os.path.join(d, "trades.csv"), index=False, encoding="utf-8-sig")
        r.blocks.to_csv(os.path.join(d, "blocks.csv"), index=False, encoding="utf-8-sig")
    score.to_parquet(os.path.join(run_dir, "score_panel.parquet"))
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(dict(vars(a), n_features=len(names), n_train=int(tr.sum()),
                       is_cv_ic=[float(r["ic"]) for r in cv_rows],
                       oos_ic=float(np.nanmean(oos_ic))), f, ensure_ascii=False, indent=1)

    print("\n===== ML 打分汇总 =====")
    print(summ.round(4).to_string(index=False))
    print("\n分年度：")
    print(metrics.annual_table(res.ret, bench_ret).to_string())
    print("\n执行质量：")
    print(metrics.execution_report(res).to_string())
    print("\n用时 %.1fs，产物 %s" % (time.time() - t0, run_dir))
    return summ


if __name__ == "__main__":
    main()
