# -*- coding: utf-8 -*-
"""s23 多因子打分选股 + 指定口径回测。

用户指定口径
------------
- 非 ST 全市场选股（动态池，含流动性/停牌/一字板/次新股过滤）
- 初始资金 10 亿人民币
- 交易费用：佣金单边万一 + 印花税万五（卖出）；基准情形不加滑点
- 2025 年第一个交易日（2025-01-02）开盘建仓 → 2026-09-30 开盘清仓离场

方法（全部只用样本内 2021-01 ~ 2024-12 定方向与权重）
------------------------------------------------------
1. 因子池：s21 产出的统计表里 `valid & |IC20|>=0.02 & |ICIR|>=0.5` 且非冗余簇代表；
2. 家族内：按 |ICIR| 加权合成家族分数（先做当日横截面秩 z）；
3. 家族间：用家族分数在**样本内**的 ICIR 做正权重加权；
4. 选股：分数前 N 名，带缓冲带（老持仓跌出 N×buffer 才卖）降换手，单票权重上限；
5. 可选 `--neutralize-adv`：把最终分数对 log(ADV20) 做横截面回归取残差（规模中性化）。

产物
----
out/03_因子挖掘/runs/23_<variant>/   nav / trades / holdings / blocks / 汇总
out/03_因子挖掘/tables/23_*.csv
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.environ.setdefault("CLASS4_TRACK", "03_因子挖掘")

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from core.lib import MarketData, CACHE, TAB, RUNS, savetab
from core.engine import Backtest, CostModel
from core import metrics
from factor_zoo import registry
from factor_zoo.ctx import Ctx
from factor_zoo.panels import build_panels_parallel
import mining_core as mc

PANEL_DIR = os.path.join(CACHE, "zoo2")


# ---------------------------------------------------------------- 组合构造
def _zscore_masked(x, ok):
    """在 ok 掩码内做秩 z；掩码外 NaN。"""
    z = np.full(x.shape, np.nan, dtype="float64")
    idx = np.flatnonzero(ok & np.isfinite(x))
    if len(idx) < 20:
        return z
    z[idx] = mc.cs_rank_z(x[idx])
    return z


def _neutralize(y, x, ok):
    """在 ok 掩码内把 y 对 x 做横截面线性回归，返回残差（规模中性化）。"""
    out = np.full(y.shape, np.nan, dtype="float64")
    idx = np.flatnonzero(ok & np.isfinite(y) & np.isfinite(x))
    if len(idx) < 30:
        return out
    a, b = x[idx], y[idx]
    a = a - a.mean()
    b = b - b.mean()
    den = float((a * a).sum())
    if den <= 0:
        out[idx] = b
        return out
    beta = float((a * b).sum()) / den
    out[idx] = b - beta * a
    return out


def build_score_panels(md, C, specs, panels, stats, ts, uni, adv_log,
                       min_abs_ic=0.02, min_abs_icir=0.5, min_cov=0.6,
                       family_weight_cap=0.45, exclude_families=None, verbose=True):
    """按样本内统计构造「家族 -> 总分」两级打分面板。

    返回 (score DataFrame, 明细 dict)
    """
    dates = md.dates[ts]
    uni = np.asarray(uni, dtype=bool)
    assert uni.shape == (len(ts), len(md.symbols)), "掩码必须先按网格切片：%s" % (uni.shape,)
    is_rows = np.array([d <= pd.Timestamp(mc.IS_END) for d in dates])
    fwd20 = mc.fwd_returns(md, ts, (20,))[20]

    sel = stats[(stats["valid"]) & (stats["coverage_is"] >= min_cov)
                & (stats["ic20"].abs() >= min_abs_ic)
                & (stats["icir"].abs() >= min_abs_icir)]
    if "is_rep" in sel.columns:
        sel = sel[sel["is_rep"].astype(bool)]
    if exclude_families:
        sel = sel[~sel["family"].isin(list(exclude_families))]
    sel = sel[sel["factor"].isin(panels.keys())].copy()
    sel["sign"] = np.sign(sel["ic20"]).replace(0, np.nan)
    sel["w_fam"] = sel["icir"].abs()
    sel = sel[sel["sign"].notna()]
    if verbose:
        print("  入选因子 %d 个（家族分布见下）" % len(sel))
        print(sel.groupby("family").size().to_string())

    # ---- 1) 家族内合成 ----
    fam_score = {}
    for fam, g in sel.groupby("family"):
        w = g["w_fam"].values
        w = w / w.sum()
        acc = np.zeros((len(ts), len(md.symbols)), dtype="float64")
        cnt = np.zeros((len(ts), len(md.symbols)), dtype="float64")
        for (_, row), wi in zip(g.iterrows(), w):
            P = np.asarray(panels[row["factor"]].values, dtype="float64")
            for i in range(len(ts)):
                z = _zscore_masked(P[i], uni[i])
                acc[i] += wi * row["sign"] * np.nan_to_num(z, nan=0.0)
                cnt[i] += np.isfinite(z).astype("float64")
        S = np.where(cnt > 0, acc, np.nan)
        # 家族分数再做一次横截面标准化，量纲统一
        for i in range(len(ts)):
            z = _zscore_masked(S[i], uni[i])
            S[i] = z
        fam_score[fam] = S

    # ---- 2) 家族权重（用样本内 ICIR） ----
    fam_rows = []
    for fam, S in fam_score.items():
        ic = mc.ic_series(S, fwd20, uni)
        st = mc.ic_stats(ic[is_rows])
        fam_rows.append(dict(family=fam, n_factors=int((sel["family"] == fam).sum()),
                             ic_is=st["ic"], icir_is=st["icir"], t_is=st["t"]))
    fam_tab = pd.DataFrame(fam_rows).sort_values("icir_is", key=lambda s: -s.abs())
    pos = fam_tab[fam_tab["icir_is"] > 0].copy()
    if len(pos) == 0:
        pos = fam_tab.head(3).copy()
    w = pos["icir_is"].abs().values
    w = w / w.sum()
    w = np.minimum(w, family_weight_cap)
    w = w / w.sum()
    pos["weight"] = w
    fam_tab = fam_tab.merge(pos[["family", "weight"]], on="family", how="left")

    total = np.zeros((len(ts), len(md.symbols)), dtype="float64")
    cnt = np.zeros((len(ts), len(md.symbols)), dtype="float64")
    for (_, r) in pos.iterrows():
        S = fam_score[r["family"]]
        m = np.isfinite(S)
        total = np.where(m, total + r["weight"] * np.nan_to_num(S, nan=0.0), total)
        cnt += m
    score = np.where(cnt > 0, total, np.nan)
    for i in range(len(ts)):
        score[i] = _zscore_masked(score[i], uni[i])

    # ---- 3) 可选：规模中性化（对 log ADV20 回归取残差） ----
    if adv_log is not None:
        for i in range(len(ts)):
            score[i] = _neutralize(score[i], adv_log[i], uni[i])
        for i in range(len(ts)):
            score[i] = _zscore_masked(score[i], uni[i])

    Sdf = pd.DataFrame(score, index=dates, columns=md.symbols)
    detail = dict(fam_tab=fam_tab, selected=sel)
    return Sdf, detail


# ---------------------------------------------------------------- 策略
class ScoreStrategy:
    """按打分选前 N，带缓冲带与单票上限。"""

    def __init__(self, score: pd.DataFrame, uni: pd.DataFrame, dates,
                 n_hold=100, max_weight=0.02, buffer_mult=1.2, liq_date=None):
        self.score = score
        self.uni = uni
        self.dates = dates
        self.n_hold = int(n_hold)
        self.max_weight = float(max_weight)
        self.buffer_mult = float(buffer_mult)
        self.liq_date = pd.Timestamp(liq_date) if liq_date is not None else None
        self._prev = []
        self.name = "MF%d_%s" % (n_hold, "score")

    def weights(self, t, md):
        date = md.dates[int(t)]
        if self.liq_date is not None and date >= self.liq_date:
            self._prev = []
            return pd.Series(dtype="float64")
        if date not in self.score.index:
            return pd.Series(dtype="float64")
        s = self.score.loc[date].values.astype("float64")
        ok = self.uni.loc[date].values.astype(bool) & np.isfinite(s)
        cand = np.flatnonzero(ok)
        if len(cand) == 0:
            return pd.Series(dtype="float64")
        order = cand[np.argsort(-s[cand], kind="stable")]
        rank = np.full(len(s), 10 ** 9, dtype="int64")
        rank[order] = np.arange(len(order))
        n = min(self.n_hold, len(order))
        prev = self._prev if (self.buffer_mult and self._prev) else None
        if prev is None:
            pick = order[:n]
        else:
            exit_rank = min(max(int(round(n * self.buffer_mult)), n + 1), len(order))
            keep = [j for j in prev if ok[j] and rank[j] < exit_rank]
            pick_set = set(keep)
            for j in order:
                if len(pick_set) >= n:
                    break
                pick_set.add(int(j))
            pick = np.array(sorted(pick_set), dtype="int64")
        self._prev = pick.tolist()
        w = np.zeros(len(s))
        w[pick] = 1.0 / len(pick)
        if self.max_weight > 0:
            w = np.minimum(w, self.max_weight)
        ssum = w.sum()
        if ssum > 0:
            w = w / ssum
        return pd.Series(w[w > 0], index=md.symbols[w > 0])


class RandomStrategy(ScoreStrategy):
    """随机对照：每期在池内随机抽 N 只（同口径、同约束）。"""

    def __init__(self, *a, seed=0, **kw):
        super().__init__(*a, **kw)
        self.rng = np.random.default_rng(seed)
        self.name = kw.get("name", "RANDOM")

    def weights(self, t, md):
        date = md.dates[int(t)]
        if self.liq_date is not None and date >= self.liq_date:
            return pd.Series(dtype="float64")
        if date not in self.uni.index:
            return pd.Series(dtype="float64")
        ok = self.uni.loc[date].values.astype(bool)
        cand = np.flatnonzero(ok)
        if len(cand) == 0:
            return pd.Series(dtype="float64")
        pick = self.rng.choice(cand, size=min(self.n_hold, len(cand)), replace=False)
        w = np.zeros(len(ok))
        w[pick] = 1.0 / len(pick)
        w = np.minimum(w, self.max_weight)
        w = w / w.sum()
        return pd.Series(w[w > 0], index=md.symbols[w > 0])


# ---------------------------------------------------------------- 主流程
def cost_model(slippage_bps=0.0, impact_coef=0.0):
    return CostModel(commission_rate=mc.COMMISSION, commission_min=0.0,
                     transfer_fee_rate=0.0, stamp_duty_rate=mc.STAMP,
                     stamp_duty_rate_old=mc.STAMP, slippage_bps=slippage_bps,
                     impact_coef=impact_coef)


def run_backtest(md, strategy, cost, start=mc.SIGNAL0, end=mc.BT_END,
                 rebalance=None, exec_window=5, participation=0.10,
                 n_hold=100, band_frac=0.25):
    """start 必须设在**建仓信号日**（2024-12-31）而不是 2025-01-02：

    引擎的信号是在循环内 `if i in rebal` 生成的，从 2025-01-02 起跑会让
    2024-12-31 的信号永远不产生（整个 1 月空仓）。从 2024-12-31 起跑时，
    该日收盘生成目标权重、次日（2025-01-02）开盘成交，正是用户要求的
    「2025 年第一个交易日当日开始建仓」。
    """
    ew = None if (exec_window is None or int(exec_window) < 0) else int(exec_window)
    # 调仓死区按「单票目标权重」缩放：band = band_frac / N（相对 NAV），
    # 否则 N 大时单票市值 < 死区 -> 一笔都不成交（N>490 时 2e-3 的死区就是这种情况）。
    band = min(2e-3, float(band_frac) / max(int(n_hold), 1))
    min_notional = min(2e5, 0.2 * mc.CASH / max(int(n_hold), 1))
    bt = Backtest(md, cost=cost, exec_price="open", max_participation=participation,
                  min_trade_notional=min_notional, rebalance_band=band, gross_limit=0.98,
                  cash=mc.CASH, use_adjusted=True, enforce_limits=True,
                  exec_window=ew)
    return bt.run(strategy, start=start, end=end, rebalance=rebalance)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n-hold", type=int, default=1700,
                    help="持仓只数（1700 ≈ 打分前 40%，见 README 的 N 敏感性）")
    ap.add_argument("--cap", type=float, default=0.02)
    ap.add_argument("--buffer", type=float, default=1.0,
                    help="缓冲带倍数（1.0 = 每月完全按目标权重重排）")
    ap.add_argument("--min-abs-ic", type=float, default=0.02)
    ap.add_argument("--min-abs-icir", type=float, default=0.5)
    ap.add_argument("--min-cov", type=float, default=0.6)
    ap.add_argument("--neutralize-adv", action="store_true")
    ap.add_argument("--random-controls", type=int, default=10)
    ap.add_argument("--exec-window", type=int, default=5,
                    help="信号后允许下单的交易日数；-1 = 一直工作到下一个信号（持续跟踪）")
    ap.add_argument("--participation", type=float, default=0.10, help="单日单票成交额参与率上限")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)

    t_all = time.time()
    md = MarketData(verbose=False)
    C = Ctx(md)
    ts = mc.monthly_grid(md, mc.GRID_START, mc.BT_END)
    dates = md.dates[ts]

    facts = registry.load_all(strict=False)
    print("注册因子 %d 个" % len(facts))
    panels = build_panels_parallel(md, facts, ts, cache_dir=PANEL_DIR,
                                   workers=a.workers, verbose=True)
    uni_df = md.universe(**mc.UNI_KW)
    uni = pd.DataFrame(np.asarray(uni_df.values, dtype=bool), index=md.dates, columns=md.symbols)
    adv_log = pd.DataFrame(np.log(np.maximum(md.adv(20).values.astype("float64"), 1.0)),
                           index=md.dates, columns=md.symbols)

    stats = pd.read_csv(os.path.join(TAB, "21_factor_stats.csv"))
    variant = "ns" if a.neutralize_adv else "raw"
    print("\n构造打分面板（variant=%s）..." % variant)
    uni_g = np.asarray(uni.values, dtype=bool)[list(ts)]
    adv_g = np.asarray(adv_log.values, dtype="float64")[list(ts)] if a.neutralize_adv else None
    score, detail = build_score_panels(md, C, facts, panels, stats, ts, uni_g, adv_g,
                                       min_abs_ic=a.min_abs_ic,
                                       min_abs_icir=a.min_abs_icir, min_cov=a.min_cov)
    savetab(detail["fam_tab"], "23_family_weights_%s.csv" % variant, index=False)
    savetab(detail["selected"][["factor", "family", "ic20", "icir", "sign", "w_fam"]],
            "23_selected_factors_%s.csv" % variant, index=False)

    # ---------------- 回测日程 ----------------
    # 信号日：2024-12-31（2025-01-02 开盘建仓）... 2026-08-31，外加 2026-09-29 清仓信号
    sig = [d for d in dates if pd.Timestamp(mc.SIGNAL0) <= d <= pd.Timestamp("2026-08-31")]
    liq = pd.Timestamp(mc.LIQ_SIGNAL)
    rebal = sorted(set(sig + [liq]))
    print("调仓信号 %d 个：%s ~ %s（末位清仓）"
          % (len(rebal), rebal[0].date(), rebal[-1].date()))

    strat = ScoreStrategy(score, uni, dates, n_hold=a.n_hold, max_weight=a.cap,
                          buffer_mult=a.buffer, liq_date=liq)
    strat.name = "MF%d_%s%s" % (a.n_hold, variant, a.tag)

    out = {}
    res = run_backtest(md, strat, cost_model(), rebalance=rebal,
                       exec_window=a.exec_window, participation=a.participation,
                       n_hold=a.n_hold)
    out[res.name] = res

    # ---------------- 基准与对照 ----------------
    bench_ret = md.ret.where(uni).mean(axis=1)
    bench_ret = bench_ret.loc[res.nav.index[0]:res.nav.index[-1]]
    bench_nav = (1.0 + bench_ret.fillna(0.0)).cumprod()

    rng = np.random.default_rng(20260101)
    rand_stats = []
    for i in range(max(0, a.random_controls)):
        rs = RandomStrategy(score, uni, dates, n_hold=a.n_hold, max_weight=a.cap,
                            buffer_mult=1.0, liq_date=liq, seed=int(rng.integers(1 << 30)))
        rs.name = "RANDOM%02d" % i
        rr = run_backtest(md, rs, cost_model(), rebalance=rebal,
                          exec_window=a.exec_window, participation=a.participation,
                          n_hold=a.n_hold)
        st = metrics.perf_stats(rr.nav, rr.ret)
        rand_stats.append(dict(run=rs.name, 累计=st["累计收益"], 年化=st["年化收益"],
                               夏普=st["夏普"], 最大回撤=st["最大回撤"]))
        print("  随机对照 %02d 年化 %.2f%%" % (i, 100 * st["年化收益"]))

    # 加滑点/冲击的敏感性
    res_slip = run_backtest(md, strat, cost_model(slippage_bps=10.0, impact_coef=0.01),
                            rebalance=rebal, exec_window=a.exec_window,
                            participation=a.participation, n_hold=a.n_hold)
    res_slip.name = res.name + "_slip10bp"
    out[res_slip.name] = res_slip

    # ---------------- 汇总 ----------------
    # 基准绩效（与组合同一起点 2024-12-31：当天组合还全是现金，NAV=1，基准已含当日市场收益）
    st_b = metrics.perf_stats(bench_nav, bench_ret)
    rows = []
    for nm, r in out.items():
        st = metrics.perf_stats(r.nav, r.ret)
        # 超额与跟踪误差只在**组合真正建仓之后**（2025-01-02 起）计算：
        # 2024-12-31 组合是现金、基准有涨跌，把它算进超额会白送 2.5pp。
        rr = r.ret.loc[r.ret.index > pd.Timestamp(mc.SIGNAL0)]
        ex = metrics.excess_stats(rr, bench_ret.reindex(rr.index))
        excess = st["年化收益"] - st_b["年化收益"]        # 同一起点、同一窗口的 CAGR 之差
        te = ex.get("跟踪误差", np.nan)
        rows.append(dict(组合=nm, 累计=st["累计收益"], 年化=st["年化收益"], 波动=st["年化波动"],
                         夏普=st["夏普"], 最大回撤=st["最大回撤"], 卡玛=st["卡玛"],
                         日胜率=st["日胜率"], 超额年化=excess,
                         跟踪误差=te,
                         信息比率=(excess / te) if (te and np.isfinite(te) and te > 0) else np.nan,
                         年换手=r.daily["turnover"].sum() / (len(r.daily) / 252.0)))
    rows.append(dict(组合="BENCH_池内等权", 累计=st_b["累计收益"], 年化=st_b["年化收益"],
                     波动=st_b["年化波动"], 夏普=st_b["夏普"], 最大回撤=st_b["最大回撤"],
                     卡玛=st_b["卡玛"], 日胜率=st_b["日胜率"], 超额年化=0.0,
                     跟踪误差=0.0, 信息比率=np.nan, 年换手=0.0))
    summary = pd.DataFrame(rows)
    savetab(summary, "23_summary_%s%s.csv" % (variant, a.tag), index=False)
    if rand_stats:
        savetab(pd.DataFrame(rand_stats), "23_random_controls_%s%s.csv" % (variant, a.tag), index=False)

    # ---------------- 落盘 ----------------
    run_dir = os.path.join(RUNS, "23_%s%s" % (variant, a.tag))
    os.makedirs(run_dir, exist_ok=True)
    for nm, r in out.items():
        d = os.path.join(run_dir, nm)
        os.makedirs(d, exist_ok=True)
        r.nav.rename("nav").to_frame().to_csv(os.path.join(d, "nav.csv"), encoding="utf-8-sig")
        r.daily.to_csv(os.path.join(d, "daily.csv"), encoding="utf-8-sig")
        r.trades.to_csv(os.path.join(d, "trades.csv"), index=False, encoding="utf-8-sig")
        r.blocks.to_csv(os.path.join(d, "blocks.csv"), index=False, encoding="utf-8-sig")
        r.holdings.to_csv(os.path.join(d, "holdings.csv"), index=False, encoding="utf-8-sig")
    bench_nav.rename("nav").to_frame().to_csv(os.path.join(run_dir, "bench_nav.csv"), encoding="utf-8-sig")
    score.to_parquet(os.path.join(run_dir, "score_panel.parquet"))
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(dict(vars(a), rebalance=[str(d.date()) for d in rebal],
                       n_factors=len(facts), n_selected=int(len(detail["selected"])),
                       exec_window=a.exec_window, participation=a.participation,
                       cash=mc.CASH, commission=mc.COMMISSION, stamp=mc.STAMP),
                  f, ensure_ascii=False, indent=1)

    print("\n===== 汇总（%s）=====" % variant)
    print(summary.to_string(index=False))
    print("\n===== 分年度 =====")
    print(metrics.annual_table(res.ret, bench_ret).to_string())
    print("\n===== 执行质量 =====")
    print(metrics.execution_report(res).to_string())
    print("\n用时 %.1fs  产物：%s" % (time.time() - t_all, run_dir))
    return summary


if __name__ == "__main__":
    main()
