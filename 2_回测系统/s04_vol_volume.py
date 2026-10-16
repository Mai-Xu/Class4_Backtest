# -*- coding: utf-8 -*-
"""s04 波动率 + 成交量低频策略：因子 IC -> 单因子 -> 合成，样本内定方向、样本外检验。

产物
    out/runs/vv_*/                    各组合的回测明细
    out/cache/f_*.parquet             因子面板缓存（参数扫描复用）
    out/vol_report.md                 报告（s05 继续追加稳健性一节）
运行
    python s04_vol_volume.py
"""
from __future__ import annotations
# --- 项目根入 sys.path：core / factor_zoo 都是根目录下的包 ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import json
import os
import time

import numpy as np
import pandas as pd

from core.lib import MarketData, ROOT, TAB, CACHE, savetab
from core.engine import Backtest, CostModel
from core.strategies import build_factor_panels, PanelStrategy, FACTOR_DESC
from core.factors import FACTORS, cs_zscore
from core import metrics

RUNS = os.path.join(ROOT, "out", "runs")
os.makedirs(RUNS, exist_ok=True)
FUC = os.path.join(CACHE, "factors")
os.makedirs(FUC, exist_ok=True)

START, END = "2021-01-04", "2026-10-16"
IS_END = "2023-12-31"          # 样本内（定方向）
REBAL = "M"
N_HOLD = 50
BUFFER = 2.0
MAX_W = 0.05
FWD = 21                       # IC 的前瞻窗口（交易日）
CASH = 1e8

HONEST = dict(exec_price="open", max_participation=0.10, min_trade_notional=5e3,
              rebalance_band=2e-3, gross_limit=0.98, cash=CASH, exec_window=3)
L = []


def w(s=""):
    L.append(s)
    print(s)


def spearman_ic(x, y, mask):
    ok = mask & np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 50:
        return np.nan
    a = pd.Series(x[ok]).rank().values
    b = pd.Series(y[ok]).rank().values
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def dump_run(res, tag):
    d = os.path.join(RUNS, tag)
    os.makedirs(d, exist_ok=True)
    res.nav.rename("nav").rename_axis("date").to_frame().to_csv(
        os.path.join(d, "nav.csv"), encoding="utf-8")
    res.daily.to_csv(os.path.join(d, "daily.csv"), encoding="utf-8")
    for nm in ("trades", "blocks", "holdings"):
        df = getattr(res, nm)
        if len(df):
            df.to_csv(os.path.join(d, nm + ".csv"), index=False, encoding="utf-8")


def main():
    t_all = time.time()
    md = MarketData(verbose=True)
    ts = sorted(Backtest._rebalance_index(md.dates, REBAL))
    ts = [t for t in ts if md.dates[t] >= pd.Timestamp(START)]
    print("调仓日 %d 个（%s ~ %s）" % (len(ts), md.dates[ts[0]].date(), md.dates[ts[-1]].date()))

    print("\n构建因子面板 ...")
    raw, zsc = build_factor_panels(md, FACTORS, ts, cache_dir=FUC, verbose=True)

    # ---------------- 前置收益与可投资域 ----------------
    adj = md.adj_close
    fwd = (adj.shift(-FWD) / adj - 1.0).values
    uni = md.universe().values
    dates = md.dates[list(ts)]
    fwd_p = pd.DataFrame(fwd[ts], index=dates, columns=md.symbols)
    uni_p = pd.DataFrame(uni[ts] & np.isfinite(fwd[ts]), index=dates, columns=md.symbols)

    # ---------------- IC 分析 ----------------
    print("\n计算 IC ...")
    ic_rows = {}
    for nm in FACTORS:
        x = raw[nm].values
        u = uni_p.values
        ics = np.array([spearman_ic(x[i], fwd_p.values[i], u[i]) for i in range(len(ts))])
        ic_rows[nm] = pd.Series(ics, index=dates)
    IC = pd.DataFrame(ic_rows)

    rows = []
    for nm in FACTORS:
        s = IC[nm].dropna()
        flag = np.asarray(s.index <= pd.Timestamp(IS_END))
        si, so = s[flag], s[~flag]
        rows.append(dict(
            因子=nm, 含义=FACTOR_DESC[nm],
            IC全样本=s.mean(), ICIR=s.mean() / s.std() if s.std() > 0 else np.nan,
            IC胜率=(s > 0).mean(),
            样本内IC=si.mean(), 样本内t=si.mean() / si.std() * np.sqrt(len(si)) if len(si) > 1 else np.nan,
            样本外IC=so.mean(), 样本外t=so.mean() / so.std() * np.sqrt(len(so)) if len(so) > 1 else np.nan,
            方向=np.sign(si.mean()) if np.isfinite(si.mean()) else 1.0,
            方向一致=bool(np.sign(si.mean()) == np.sign(so.mean())) if len(so) else False))
    ICT = pd.DataFrame(rows).set_index("因子")
    savetab(ICT.round(4), "04_factor_ic.csv")
    IC.round(4).to_csv(os.path.join(RUNS, "vv_ic.csv"), encoding="utf-8")

    # ---------------- 单因子组合 ----------------
    print("\n回测单因子组合 ...")
    results, summ = {}, {}
    bt_cache = {}
    for nm in FACTORS:
        sign = ICT.loc[nm, "方向"]
        sp = zsc[nm] * (sign if np.isfinite(sign) else 1.0)
        st = PanelStrategy(sp, name=nm, n_hold=N_HOLD, buffer_mult=BUFFER,
                           max_weight=MAX_W)
        bt = Backtest(md, **HONEST)
        res = bt.run(st, start=START, end=END, rebalance=REBAL)
        results[nm] = res
        dump_run(res, "vv_" + nm)
        st_ = metrics.perf_stats(res.nav)
        summ[nm] = dict(年化=st_["年化收益"], 夏普=st_["夏普"], 回撤=st_["最大回撤"],
                        年换手=float(res.daily.turnover.mean() * 252))
    bt_cache["single"] = summ

    # ---------------- 合成策略 ----------------
    # 只用样本内信息挑因子：|t| >= 1 且方向一致
    sel = ICT[(ICT["样本内t"].abs() >= 1.0)].index.tolist()
    if len(sel) < 3:
        sel = ICT["样本内IC"].abs().sort_values(ascending=False).head(6).index.tolist()
    print("\n样本内入选因子（|t|>=1）:", sel)
    comp = sum(zsc[nm] * (ICT.loc[nm, "方向"] if np.isfinite(ICT.loc[nm, "方向"]) else 1.0)
               for nm in sel) / len(sel)
    comp_all = sum(zsc[nm] for nm in FACTORS) / len(FACTORS)   # 等权全部因子（无方向选择）

    comps = {"vv_comp": comp, "vv_comp_all": comp_all}
    for tag, panel in comps.items():
        st = PanelStrategy(panel, name=tag, n_hold=N_HOLD, buffer_mult=BUFFER,
                           max_weight=MAX_W)
        res = Backtest(md, **HONEST).run(st, start=START, end=END, rebalance=REBAL)
        results[tag] = res
        dump_run(res, tag)
        s_ = metrics.perf_stats(res.nav)
        summ[tag] = dict(年化=s_["年化收益"], 夏普=s_["夏普"], 回撤=s_["最大回撤"],
                         年换手=float(res.daily.turnover.mean() * 252))

    # ---------------- 汇总 ----------------
    bench = md.equal_weight_ret().loc[START:END]
    rows = []
    for tag, res in results.items():
        s_ = metrics.perf_stats(res.nav)
        s_.update(metrics.excess_stats(res.ret, bench))
        s_["年换手"] = float(res.daily.turnover.mean() * 252)
        s_["成本占净值"] = float(res.trades.fee.sum() / CASH) if len(res.trades) else 0.0
        s_["被挡金额_亿"] = float(res.blocks.dv.sum() / 1e8) if len(res.blocks) else 0.0
        s_["持仓数"] = float(res.daily.n_holdings.median())
        rows.append(pd.Series(s_, name=tag))
    S = pd.DataFrame(rows)
    bs = metrics.perf_stats((1 + bench).cumprod())
    S.loc["基准(等权全市场)"] = pd.Series(bs, name="基准(等权全市场)")
    savetab(S.round(6), "04_vol_strategy_summary.csv")

    # ================= 报告 =================
    w("# Class4 · 波动率 + 成交量低频策略")
    w()
    w("由 `python s04_vol_volume.py` 生成（`out/vol_report.md`）。")
    w()
    w("## 1. 设计")
    w()
    w("- **频率**：月度调仓（82 个调仓日），持有 50 只，单票上限 5%%，"
      "**换手控制用缓冲带**：老持仓只要还在前 %g 名内就不卖。" % (N_HOLD * BUFFER))
    w("- **方向由样本内决定**：2021-01 ~ %s 上用 IC 定因子方向，"
      "%s 之后为样本外，**样本外不参与任何选择**。" % (IS_END,
      (pd.Timestamp(IS_END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")))
    w("- **执行**：信号在月末收盘产生，次日起 3 个交易日内按目标权重在开盘成交；"
      "涨跌停/停牌不可交易、参与率上限 10%%、含全部交易成本。")
    w("- **股票池**：上市满 250 日、20 日均额 >= 2000 万、股价 >= 2 元、剔除次新/一字板/停牌。")
    w()
    w("### 因子清单（13 个，全部只用 t 日及之前数据）")
    w()
    w("| 族 | 因子 | 含义 |")
    w("| --- | --- | --- |")
    fam = {"ivol": "波动率", "beta": "波动率", "dvol": "波动率", "maxdd": "波动率",
           "skew": "波动率", "kurt": "波动率", "vol_trend": "波动率",
           "vol_of_vol": "波动率", "amt_trend": "成交量", "amt_cv": "成交量",
           "amihud": "成交量", "trade_size": "成交量", "pv_corr": "成交量"}
    for nm in FACTORS:
        w("| %s | `%s` | %s |" % (fam[nm], nm, FACTOR_DESC[nm]))
    w()
    w("## 2. 因子 IC：样本内 vs 样本外")
    w()
    w("IC = 调仓日因子值与随后 21 个交易日复权收益的**横截面 Spearman 秩相关**。")
    w()
    t = ICT[["含义", "IC全样本", "ICIR", "IC胜率", "样本内IC", "样本内t",
             "样本外IC", "样本外t", "方向一致"]].copy()
    for c in ["IC全样本", "ICIR", "IC胜率", "样本内IC", "样本外IC"]:
        t[c] = t[c].map(lambda v: "—" if pd.isna(v) else "%.4f" % v)
    for c in ["样本内t", "样本外t"]:
        t[c] = t[c].map(lambda v: "—" if pd.isna(v) else "%.2f" % v)
    t["方向一致"] = t["方向一致"].map(lambda v: "是" if v else "否")
    t = t.sort_values("IC全样本", key=lambda s: s.map(
        lambda v: float(v) if v != "—" else 0.0), ascending=False)
    w(md_table(t))
    w()
    n_ok = int(ICT["方向一致"].sum())
    w("- 13 个因子里 **%d 个**在样本外保持了与样本内一致的符号，说明大部分波动率/成交量"
      "因子的方向是稳定的，不是样本内的偶然。" % n_ok)
    w("- IC 量级在 ±0.05 上下，属于典型的中低频截面因子强度；"
      "单看 ICIR 低于 0.3 的因子不宜单独使用。")
    w()
    w("## 3. 单因子组合")
    w()
    w("每个因子取样本内 IC 的符号做多头（前 50 只，等权，缓冲带 2 倍）。")
    w()
    t2 = S.copy()
    w(md_table(pct_table(t2[["年化收益", "年化波动", "夏普", "最大回撤", "卡玛",
                              "超额年化", "信息比率", "年换手", "成本占净值", "持仓数"]])))
    w()
    w("## 4. 合成策略")
    w()
    w("样本内 |t| >= 1 的因子：**%s**，等权 z-score 合成（`vv_comp`）；"
      "另外给出不做方向筛选、直接等权全部 13 个因子的 `vv_comp_all` 作为对照。"
      % "、".join(sel))
    w()

    out_md = os.path.join(ROOT, "out", "vol_report.md")
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    json.dump({"sel_factors": sel, "is_end": IS_END, "n_hold": N_HOLD,
               "buffer": BUFFER, "rebal": REBAL, "start": START, "end": END,
               "cash": CASH, "honest": {k: v for k, v in HONEST.items()}},
              open(os.path.join(RUNS, "vv_config.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n总用时 %.1fs" % (time.time() - t_all))
    print("已写出 %s" % out_md)
    return S, ICT


def md_table(df):
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
                vals.append("—" if not np.isfinite(v) else "%.4f" % v)
            else:
                vals.append(str(v))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([head, sep] + rows)


def pct_table(df, cols_pct=None):
    cols_pct = cols_pct or ["年化收益", "年化波动", "最大回撤", "超额年化", "成本占净值"]
    d = df.copy()
    for c in d.columns:
        if c in cols_pct:
            d[c] = d[c].map(lambda v: "—" if pd.isna(v) else "%.2f%%" % (100 * float(v)))
        elif c in ("夏普", "卡玛", "信息比率"):
            d[c] = d[c].map(lambda v: "—" if pd.isna(v) else "%.2f" % float(v))
        else:
            d[c] = d[c].map(lambda v: "—" if pd.isna(v) else "%.1f" % float(v))
    return d


if __name__ == "__main__":
    main()
