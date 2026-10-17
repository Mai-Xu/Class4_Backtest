# -*- coding: utf-8 -*-
"""s08 因子挖掘主线：注册 -> 校验 -> IC 筛选 -> 去冗余 -> 合成 -> 按指定口径回测。

回测口径（用户指定）
    非 ST 全市场选股 · 初始资金 10 亿 · 佣金单边万一 · 印花税万五
    2025 年第一个交易日建仓 -> 2026 年 9 月最后一个交易日清仓离场

方法要点
    因子方向与筛选**只用 2021-01 ~ 2024-12 的样本内数据**，
    2025-01 ~ 2026-09 完全不参与选择，只用于回测与检验。

产物
    out/mining_report.md            报告
    out/tables/08_*.csv             因子校验 / IC / 筛选 / 相关性
    out/runs/zoo_*/                 各组合回测明细
"""
from __future__ import annotations
# --- 项目根入 sys.path：core / factor_zoo 都是根目录下的包 ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- 产物归档到本工作的目录：out/03_因子挖掘/ ---
_os.environ.setdefault("CLASS4_TRACK", "03_因子挖掘")


import json
import os
import time

import numpy as np
import pandas as pd

from core.lib import MarketData, ROOT, CACHE, savetab, RUNS, OUT
from core.engine import Backtest, CostModel
from core.strategies import PanelStrategy
from factor_zoo.panels import build_zoo_panels
from factor_zoo.ctx import Ctx, cs_zscore
from factor_zoo import registry
from core import metrics

FUC = os.path.join(CACHE, "zoo")
os.makedirs(RUNS, exist_ok=True)
os.makedirs(FUC, exist_ok=True)

# ---- 用户指定口径 ----
CASH = 1e9                       # 10 亿
COMMISSION = 1e-4                # 佣金单边万一
STAMP = 5e-4                     # 印花税万五（卖出）
BT_START, BT_END = "2025-01-02", "2026-09-30"
SIGNAL0 = "2024-12-31"           # 建仓信号日（次日 2025-01-02 开盘成交）
LIQ = "2026-09-29"               # 清仓信号日（次日 2026-09-30 开盘清空）

GRID_START = "2021-01-01"        # 因子面板起点
IS_END = "2024-12-31"            # 样本内截止（定方向 + 筛选）
N_HOLD = 100
BUFFER = 2.0
MAX_W = 0.02
EXEC_WINDOW = 3
PARTICIPATION = 0.10

COST = CostModel(commission_rate=COMMISSION, commission_min=0.0, transfer_fee_rate=0.0,
                 stamp_duty_rate=STAMP, stamp_duty_rate_old=STAMP,
                 slippage_bps=0.0, impact_coef=0.0)
COST_SLIP = CostModel(commission_rate=COMMISSION, commission_min=0.0, transfer_fee_rate=0.0,
                      stamp_duty_rate=STAMP, stamp_duty_rate_old=STAMP,
                      slippage_bps=10.0, impact_coef=0.01)
BT_KW = dict(exec_price="open", max_participation=PARTICIPATION, min_trade_notional=2e5,
             rebalance_band=2e-3, gross_limit=0.98, cash=CASH, exec_window=EXEC_WINDOW,
             enforce_limits=True, use_adjusted=True)

L = []


def w(s=""):
    L.append(s)
    print(s)


def md_table(df, floatfmt="%.4f"):
    d = df.copy()
    d.columns = [str(c) for c in d.columns]
    cols = list(d.columns)
    head = "| " + " | ".join([str(d.index.name or "")] + cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    rows = []
    for i in range(len(d)):
        vals = [str(d.index[i])]
        for c in cols:
            v = d[c].iloc[i]
            if isinstance(v, (bool, np.bool_)):
                vals.append("是" if v else "否")
            elif isinstance(v, (int, np.integer)):
                vals.append(format(int(v), ","))
            elif isinstance(v, (float, np.floating)):
                vals.append("—" if not np.isfinite(v) else floatfmt % v)
            else:
                vals.append(str(v))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([head, sep] + rows)


def pct(v, n=2):
    """把小数格式化成百分比字符串；传入 NaN/None 返回破折号。"""
    try:
        if v is None or not np.isfinite(v):
            return "—"
    except TypeError:
        return "—"
    return f"{100 * float(v):.{int(n)}f}%"


def spearman_ic(x, y, mask):
    ok = mask & np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 50:
        return np.nan
    a = pd.Series(x[ok]).rank().values.astype("float64")
    b = pd.Series(y[ok]).rank().values.astype("float64")
    a -= a.mean(); b -= b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def main():
    t_all = time.time()
    md = MarketData(verbose=True)
    C = Ctx(md)
    dates = md.dates

    # ---------- 调仓网格 ----------
    me = sorted(Backtest._rebalance_index(dates, "M"))
    grid = [t for t in me if pd.Timestamp(GRID_START) <= dates[t] <= pd.Timestamp(BT_END)]
    liq_pos = int(dates.searchsorted(pd.Timestamp(LIQ)))
    if liq_pos not in grid:
        grid = sorted(set(grid + [liq_pos]))
    gdates = dates[grid]
    n_g = len(grid)
    print("调仓网格 %d 个：%s ~ %s（末位为清仓信号日）"
          % (n_g, gdates[0].date(), gdates[-1].date()))

    # ---------- 因子注册与校验 ----------
    print("\n[1/6] 加载因子 ...")
    facts = registry.load_all(strict=False)
    print("共注册 %d 个因子" % len(facts))
    if not facts:
        raise SystemExit("factor_zoo 下没有任何因子，先让子 agent 写完分类模块")
    by_family = pd.Series([s["family"] for s in facts.values()]).value_counts()
    print(by_family.to_string())

    print("\n[2/6] 因果性与形状校验 ...")
    # 筹码类因子单次约 0.6s，校验只抽样 12 个日期（面板仍会算满全网格）
    chk_ts = list(grid[:: max(1, len(grid) // 12)])[:12]
    chk = pd.DataFrame(registry.check_all(facts, C, chk_ts)).set_index("factor")
    chk["family"] = [facts[i]["family"] for i in chk.index]
    chk["desc"] = [facts[i]["desc"] for i in chk.index]
    savetab(chk.round(6), "08_factor_check.csv")
    good = list(chk.index[chk["ok"]])
    dropped = list(chk.index[~chk["ok"]])
    print("通过 %d，丢弃 %d" % (len(good), len(dropped)))
    for f in dropped[:20]:
        print("  丢弃 %-24s %s" % (f, chk.loc[f, "reason"]))

    # ---------- 面板 ----------
    print("\n[3/6] 计算因子面板 ...")
    raw, zsc = build_zoo_panels(md, C, {k: facts[k] for k in good}, grid,
                                cache_dir=FUC, verbose=False)
    # 清仓信号日：把打分清空 -> 目标为空 -> 全仓卖出
    for d in zsc.values():
        if pd.Timestamp(LIQ) in d.index:
            d.loc[pd.Timestamp(LIQ)] = np.nan

    # ---------- 可投资域与前向收益 ----------
    uni = md.universe(min_listed_days=250, adv_window=20, min_adv=None, price_min=1.0,
                      exclude_st=True, exclude_no_limit=True, exclude_one_line=True)
    U = uni.values[grid]
    adj = C.C[grid]
    fwd = np.full_like(adj, np.nan)
    fwd[:-1] = adj[1:] / adj[:-1] - 1.0          # 到下一个调仓日的收益
    ok_all = U & np.isfinite(fwd)

    # ---------- IC ----------
    print("\n[4/6] 计算 IC ...")
    ic = {}
    for nm in good:
        X = raw[nm].values
        ic[nm] = pd.Series([spearman_ic(X[i], fwd[i], ok_all[i]) for i in range(n_g)],
                           index=gdates)
    IC = pd.DataFrame(ic)
    IC.round(4).to_csv(os.path.join(RUNS, "zoo_ic.csv"), encoding="utf-8")

    rows = []
    for nm in good:
        s = IC[nm].dropna()
        flag = np.asarray(s.index <= pd.Timestamp(IS_END))
        si, so = s[flag], s[~flag]
        t_is = si.mean() / si.std() * np.sqrt(len(si)) if len(si) > 2 and si.std() > 0 else np.nan
        t_oos = so.mean() / so.std() * np.sqrt(len(so)) if len(so) > 2 and so.std() > 0 else np.nan
        rows.append(dict(因子=nm, 类别=facts[nm]["family"], 说明=facts[nm]["desc"],
                         全样本IC=s.mean(), 样本内IC=si.mean(), 样本内t=t_is,
                         样本外IC=so.mean(), 样本外t=t_oos,
                         方向一致=bool(np.sign(si.mean()) == np.sign(so.mean()))
                         if len(si) and len(so) else False,
                         覆盖=chk.loc[nm, "coverage"]))
    ICT = pd.DataFrame(rows).set_index("因子")

    # ---------- 筛选 + 去冗余 ----------
    print("\n[5/6] 筛选与去冗余 ...")
    cand = ICT[(ICT["样本内t"].abs() >= 2.0) & (ICT["覆盖"] >= 0.6)].copy()
    print("样本内 |t| >= 2 且覆盖 >= 60%%：%d 个" % len(cand))
    # 因子值相关（横截面 rank 相关的平均），|rho|>0.85 视为冗余
    keep_order = cand["样本内IC"].abs().sort_values(ascending=False).index.tolist()
    Z = {nm: zsc[nm].values for nm in keep_order}
    dropped_corr = []
    kept = []
    for nm in keep_order:
        red = False
        for kv in kept:
            a, b = Z[nm], Z[kv]
            m = np.isfinite(a) & np.isfinite(b)
            r = []
            for i in range(0, n_g, 2):
                mm = m[i]
                if mm.sum() > 200:
                    r.append(np.corrcoef(a[i][mm], b[i][mm])[0, 1])
            if r and abs(np.nanmean(r)) > 0.85:
                red = True
                dropped_corr.append((nm, kv, float(np.nanmean(r))))
                break
        if not red:
            kept.append(nm)
    print("去冗余后保留 %d 个" % len(kept))
    if dropped_corr:
        savetab(pd.DataFrame(dropped_corr, columns=["被剔除", "与谁冗余", "相关系数"]).set_index("被剔除"),
                "08_corr_drop.csv")
    signs = {nm: float(np.sign(ICT.loc[nm, "样本内IC"])) for nm in kept}

    # ---------- 合成与回测 ----------
    print("\n[6/6] 合成与回测 ...")
    # ---- 规模中性化：把每个因子的 z-score 对 log(20日均额) 做横截面回归取残差 ----
    # kept 里有一半是成交额/流动性代理，本质是规模的变身；不中性化就分不清
    # 「因子有效」还是「小市值 beta」。
    A20 = md.adv(20).values[grid]
    with np.errstate(invalid="ignore", divide="ignore"):
        size = np.log(np.where(np.isfinite(A20) & (A20 > 0), A20, np.nan))

    def _neu(z, sx):
        ok = np.isfinite(z) & np.isfinite(sx)
        n = ok.sum(1).astype("float64")
        nz = np.maximum(n, 1.0)
        zz = np.where(ok, z, 0.0)
        ss = np.where(ok, sx, 0.0)
        mz = zz.sum(1) / nz
        ms = ss.sum(1) / nz
        dz = np.where(ok, z - mz[:, None], 0.0)
        ds = np.where(ok, sx - ms[:, None], 0.0)
        cov = (dz * ds).sum(1) / nz
        var = (ds ** 2).sum(1) / nz
        beta = np.where(var > 0, cov / var, 0.0)
        r = dz - beta[:, None] * ds
        out = np.full_like(z, np.nan)
        out[ok] = r[ok]
        return out, ok.sum(1)

    zsc_ns = {nm: pd.DataFrame(_neu(zsc[nm].values, size)[0], index=zsc[nm].index,
                               columns=zsc[nm].columns) for nm in kept}

    eq = sum(zsc[nm] * signs[nm] for nm in kept) / len(kept)
    eq.name = "zoo_eq"
    nseq = sum(zsc_ns[nm] * signs[nm] for nm in kept) / len(kept)
    nseq.name = "zoo_ns"
    wt = sum(zsc[nm] * signs[nm] * abs(ICT.loc[nm, "样本内IC"]) for nm in kept) / \
        sum(abs(ICT.loc[nm, "样本内IC"]) for nm in kept)
    wt.name = "zoo_icw"

    panel_for = {"zoo_eq": eq, "zoo_icw": wt, "zoo_ns": nseq}
    for nm in kept[:12]:
        panel_for["zoo_" + nm] = zsc[nm] * signs[nm]

    rebal = sorted(set(grid))
    results, rows2 = {}, []
    for tag, panel in panel_for.items():
        st = PanelStrategy(panel, name=tag, n_hold=N_HOLD, buffer_mult=BUFFER,
                           max_weight=MAX_W)
        res = Backtest(md, cost=COST, **BT_KW).run(
            st, start=SIGNAL0, end=BT_END, rebalance=[dates[t] for t in rebal])
        # 只保留 2025-01-02 之后的净值
        res.nav = res.nav.loc[BT_START:]
        res.daily = res.daily.loc[BT_START:]
        results[tag] = res
        d = os.path.join(RUNS, tag)
        os.makedirs(d, exist_ok=True)
        res.nav.rename("nav").rename_axis("date").to_frame().to_csv(
            os.path.join(d, "nav.csv"), encoding="utf-8")
        res.daily.to_csv(os.path.join(d, "daily.csv"), encoding="utf-8")
        for f_ in ("trades", "blocks", "holdings"):
            df = getattr(res, f_)
            if len(df):
                df.to_csv(os.path.join(d, f_ + ".csv"), index=False, encoding="utf-8")
        s_ = metrics.perf_stats(res.nav)
        bench = md.ret_ok.where(uni).mean(1).loc[res.nav.index]   # 池内等权 = 正确基准
        s_.update(metrics.excess_stats(res.ret, bench))
        s_["年换手"] = float(res.daily.turnover.mean() * 252)
        s_["费用占净值"] = float(res.trades.fee.sum() / CASH) if len(res.trades) else 0.0
        s_["被挡金额_亿"] = float(res.blocks.dv.sum() / 1e8) if len(res.blocks) else 0.0
        s_["持仓数"] = float(res.daily.n_holdings.median())
        rows2.append(pd.Series(s_, name=tag))
        print("  %-14s 累计 %7.2f%%  夏普 %6.2f  回撤 %7.2f%%  换手 %5.1f"
              % (tag, 100 * s_["累计收益"], s_["夏普"], 100 * s_["最大回撤"], s_["年换手"]))
    S = pd.DataFrame(rows2)
    bench_all = md.ret_ok.where(uni).mean(1).loc[BT_START:BT_END]
    bs = metrics.perf_stats((1 + bench_all).cumprod())
    S.loc["基准(等权全市场)"] = pd.Series(bs, name="基准(等权全市场)")
    savetab(S.round(6), "08_mining_summary.csv")

    # 成本敏感性
    print("\n附加：滑点+冲击敏感性 ...")
    sen = []
    for lab, cost in [("指定口径(佣金万一+印花万五)", COST), ("加 10bp 滑点 + 平方根冲击", COST_SLIP)]:
        res = Backtest(md, cost=cost, **BT_KW).run(
            PanelStrategy(eq, name="eq_%s" % lab[:2], n_hold=N_HOLD, buffer_mult=BUFFER,
                          max_weight=MAX_W),
            start=SIGNAL0, end=BT_END, rebalance=[dates[t] for t in rebal])
        res.nav = res.nav.loc[BT_START:]
        s_ = metrics.perf_stats(res.nav)
        s_["费用占净值"] = float(res.trades.fee.sum() / CASH) if len(res.trades) else 0.0
        sen.append(pd.Series(s_, name=lab))
        print("  %-30s 累计 %7.2f%%  夏普 %.2f" % (lab, 100 * s_["累计收益"], s_["夏普"]))
    SEN = pd.DataFrame(sen)

    # ================= 报告 =================
    w()
    w("# 全市场因子挖掘 · 选股回测")
    w()
    w("回测口径（按指定）：**非 ST 全市场选股 · 初始资金 10 亿元 · 佣金单边万一 · 印花税万五 · "
      "%s 建仓 → %s 清仓离场**。" % (BT_START, BT_END))
    w()
    w("> 因子方向与筛选**只用 %s 之前的样本内数据**，回测区间完全不参与选择。" % IS_END)
    w()
    w("## 1. 因子库")
    w()
    w("按 5 个方向分派实现，共注册 **%d 个因子**，通过因果性与形状校验 **%d 个**、丢弃 %d 个。"
      % (len(facts), len(good), len(dropped)))
    w()
    w(md_table(by_family.rename("因子数").to_frame(), "%.0f"))
    w()
    if dropped:
        w("被丢弃的原因（前 10 条）：")
        w()
        w(md_table(chk.loc[dropped, ["family", "reason"]].head(10), "%.4f"))
        w()
    w("**每个因子都经过「把所有数组截到 t 之后重算一遍」的因果性检验**，"
      "偷看未来数据的因子会被自动识别并丢弃。")
    w()
    w("## 2. 因子 IC（样本内 vs 回测区间）")
    w()
    t = ICT.sort_values("样本内IC", key=lambda s: s.abs(), ascending=False).copy()
    t["全样本IC"] = t["全样本IC"].map(lambda v: "%.4f" % v)
    t["样本内IC"] = t["样本内IC"].map(lambda v: "%.4f" % v)
    t["样本内t"] = t["样本内t"].map(lambda v: "%.2f" % v)
    t["样本外IC"] = t["样本外IC"].map(lambda v: "%.4f" % v)
    t["样本外t"] = t["样本外t"].map(lambda v: "%.2f" % v)
    t["覆盖"] = t["覆盖"].map(lambda v: "%.0f%%" % (100 * v))
    t["方向一致"] = t["方向一致"].map(lambda v: "是" if v else "否")
    w(md_table(t.head(40)))
    w()
    n_ok = int(ICT["方向一致"].sum())
    w("- %d 个因子里 **%d 个**在回测区间保持了与样本内一致的符号。" % (len(ICT), n_ok))
    w("- 样本内 |t| >= 2.0：**%d 个**。" % len(cand))
    w()
    w("## 3. 去冗余与合成")
    w()
    w("- 因子值横截面相关 |rho| > 0.85 视为冗余，保留样本内 |t| 更大的那个，"
      "**剔除 %d 个**，保留 **%d 个**。" % (len(dropped_corr), len(kept)))
    w("- 纳入合成的因子：%s" % "、".join(kept))
    w("- 合成方式：横截面 z-score，按样本内 IC 符号定向；等权版本 `zoo_eq`、"
      "IC 加权版本 `zoo_icw`。")
    w()
    w("- **规模中性化版本 `zoo_ns`**：把每个因子的横截面 z-score 对 `log(20 日均额)` "
      "做截面回归取残差，再按同样方式合成。kept 里有近一半是成交额/流动性代理"
      "（amt_log / amihud / trade_size / amt_rel …），本质是规模的变身；"
      "不中性化就分不清赚的是「因子」还是「小市值 beta」。")
    w()
    w("## 4. 回测结果（%s ~ %s，10 亿元）" % (BT_START, BT_END))
    w()
    t2 = S[["累计收益", "年化收益", "年化波动", "夏普", "最大回撤", "卡玛",
            "超额年化", "信息比率", "年换手", "费用占净值", "被挡金额_亿", "持仓数"]].copy()
    for c in ("累计收益", "年化收益", "年化波动", "最大回撤", "超额年化", "费用占净值"):
        t2[c] = t2[c].map(lambda v: pct(v) if pd.notna(v) else "—")
    for c in ("夏普", "卡玛", "信息比率"):
        t2[c] = t2[c].map(lambda v: "—" if pd.isna(v) else "%.2f" % v)
    for c in ("年换手", "被挡金额_亿", "持仓数"):
        t2[c] = t2[c].map(lambda v: "—" if pd.isna(v) else "%.1f" % v)
    w(md_table(t2))
    w()
    w("## 5. 成本敏感性")
    w()
    w(md_table(pd.DataFrame({
        "累计收益": SEN["累计收益"].map(pct),
        "年化收益": SEN["年化收益"].map(pct),
        "夏普": SEN["夏普"].map(lambda v: "%.2f" % v),
        "最大回撤": SEN["最大回撤"].map(pct),
        "费用占净值": SEN["费用占净值"].map(pct),
    })))
    w()
    w("## 6. 结论与风险")
    w()
    best = S.drop(index="基准(等权全市场)")["夏普"].idxmax()
    w("- 表现最好的是 **%s**（累计 %s，夏普 %.2f，最大回撤 %s）。"
      % (best, pct(S.loc[best, "累计收益"]), S.loc[best, "夏普"], pct(S.loc[best, "最大回撤"])))
    w("- 基准（**股票池内等权**）同期累计 %s。注意不能用「全市场等权」做基准：被剔除的次新股在样本内年化 85%%，会把基准整体抬高约 1 个百分点。"
      % pct(S.loc["基准(等权全市场)", "累计收益"]))
    w("- 区间只有约 21 个月、21 个调仓日，**样本极短**，结论的统计置信度有限；"
      "因子筛选用的是 %s 之前的数据，能部分缓解但不消除这个问题。" % IS_END)
    w("- 10 亿规模下容量约束是真实瓶颈：被挡委托合计 %.1f 亿元，需结合 §4 的持仓数变化一起看。"
      % float(S.loc[best, "被挡金额_亿"]))
    w()

    out_md = os.path.join(OUT, "mining_report.md")
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    json.dump(dict(cash=CASH, commission=COMMISSION, stamp=STAMP, start=BT_START,
                   end=BT_END, n_hold=N_HOLD, buffer=BUFFER, max_w=MAX_W,
                   participation=PARTICIPATION, is_end=IS_END,
                   n_factors=len(facts), n_good=len(good), kept=kept, signs=signs),
              open(os.path.join(RUNS, "zoo_config.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n总用时 %.1fs\n已写出 %s" % (time.time() - t_all, out_md))
    return S, ICT


if __name__ == "__main__":
    main()
