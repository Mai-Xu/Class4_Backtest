# -*- coding: utf-8 -*-
"""s09 挖掘结果出图与补录：净值图 + 规模敏感性表格并入 mining_report.md。

    python s09_mining_report.py
"""
from __future__ import annotations
# --- 项目根入 sys.path：core / factor_zoo 都是根目录下的包 ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.lib import MarketData, ROOT, FIG, savetab, savefig

RUNS = os.path.join(ROOT, "out", "runs")
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
TAGS = {"zoo_ns": ("规模中性化 38 因子", "#d94801"),
        "zoo_icw": ("IC 加权 38 因子", "#2c7fb8"),
        "zoo_eq": ("等权 38 因子", "#31a354"),
        "zoo_lq_amt_log20": ("纯规模因子(成交额水平)", "#969696")}


def load(tag):
    p = os.path.join(RUNS, tag, "nav.csv")
    return pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0]


def md_table(df, fmts):
    d = df.copy()
    head = "| " + " | ".join([str(d.index.name or "")] + [str(c) for c in d.columns]) + " |"
    sep = "|" + "---|" * (len(d.columns) + 1)
    rows = []
    for i in range(len(d)):
        vals = [str(d.index[i])]
        for c in d.columns:
            v = d[c].iloc[i]
            vals.append(fmts.get(c, "%s") % v if isinstance(fmts.get(c), str) and fmts.get(c) != "%s" else str(v))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([head, sep] + rows)


def main():
    md = MarketData()
    idx_bt = md.dates[(md.dates >= np.datetime64("2025-01-02")) & (md.dates <= np.datetime64("2026-09-30"))]
    uni = md.universe(min_listed_days=250, adv_window=20, min_adv=None, price_min=1.0,
                      exclude_st=True, exclude_no_limit=True, exclude_one_line=True)
    bench = md.ret_ok.where(uni).mean(1).loc[idx_bt]
    bnav = (1 + bench).cumprod()
    navs = {t: load(t).loc[idx_bt] for t in TAGS if os.path.exists(os.path.join(RUNS, t, "nav.csv"))}

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for t, (lab, col) in TAGS.items():
        if t not in navs:
            continue
        nv = navs[t]
        ax[0].plot(nv.index, nv / nv.iloc[0], lw=2.0 if t == "zoo_ns" else 1.3,
                   color=col, label=lab)
        ax[1].plot(nv.index, (nv / nv.cummax() - 1) * 100, lw=2.0 if t == "zoo_ns" else 1.1,
                   color=col)
    ax[0].plot(bnav.index, bnav / bnav.iloc[0], lw=1.6, ls="--", color="#252525",
               label="基准(股票池内等权)")
    ax[0].axhline(1, color="k", lw=0.6)
    ax[0].set_title("图16  因子挖掘组合净值（2025-01-02 ~ 2026-09-30，10 亿元）")
    ax[0].set_ylabel("净值（初始=1）"); ax[0].legend(fontsize=8, loc="upper left")
    ax[1].plot(bnav.index, (bnav / bnav.cummax() - 1) * 100, lw=1.6, ls="--", color="#252525")
    ax[1].set_title("图17  回撤：规模中性化后回撤近乎腰斩")
    ax[1].set_ylabel("%")
    savefig(fig, "06_mining.png")

    # 规模敏感性
    p = os.path.join(ROOT, "out", "tables", "08_scale_sensitivity.csv")
    sens = pd.read_csv(p, encoding="utf-8-sig") if os.path.exists(p) else None

    L = []
    L.append("")
    L.append("## 7. 规模敏感性：10 亿是这条策略的硬边界")
    L.append("")
    L.append("同一套等权合成信号，只改初始资金：")
    L.append("")
    if sens is not None:
        s2 = sens.copy()
        s2["资金"] = s2["cash"].map(lambda v: "%.0f 亿" % (v / 1e8))
        s2["累计收益"] = (100 * s2["cum"]).round(2).astype(str) + "%"
        s2["超额年化"] = (100 * s2["ex"]).round(2).astype(str) + "%"
        s2["信息比率"] = s2["ir"].round(2)
        s2["夏普"] = s2["ir"].round(0).astype(int) * 0 + sens["ir"].round(0) * 0
        s2["持仓数中位"] = s2["nh"].round(0).astype(int)
        s2["被挡委托(亿元)"] = s2["blk"].round(0).astype(int)
        t = s2[["资金", "累计收益", "超额年化", "信息比率", "持仓数中位", "被挡委托(亿元)"]]
        t = t.set_index("资金")
        head = "| 资金 | 累计收益 | 超额年化 | 信息比率 | 持仓数中位 | 被挡委托(亿元) |"
        L.append(head)
        L.append("|---|---|---|---|---|---|")
        for i, r in t.iterrows():
            L.append("| %s | %s | %s | %s | %s | %s |" % (i, r["累计收益"], r["超额年化"],
                                                           r["信息比率"], r["持仓数中位"],
                                                           r["被挡委托(亿元)"]))
    L.append("")
    L.append("读法：**资金从 1 亿加到 30 亿，超额年化从 +1.30% 一路掉到 −1.58%，"
             "持仓数从 105 只膨胀到 471 只，被挡委托从 0 涨到 1,507 亿元。**")
    L.append("")
    L.append("10 亿元下参与率上限（单日单票不超过当日成交额的 10%）截断了 4,478 笔、"
             "215 亿元的委托，组合被迫从目标的 100 只分散到 440 只 —— "
             "**被容量约束硬生生压成了「近似全市场等权」，这正是超额被抹平的原因。**")
    L.append("")
    L.append("## 8. 规模中性化：alpha 的真假鉴定")
    L.append("")
    L.append("入选的 38 个因子里有近一半是成交额/流动性代理（`lq_amt_log20`、`lq_amihud20`、"
             "`lq_trade_size20`、`lq_amt_rel20_60` …），**它们本质是规模的变身**。"
             "把每个因子的横截面 z-score 对 `log(20 日均额)` 做截面回归取残差后再合成：")
    L.append("")
    L.append("| 组合 | 累计 | 年化 | 夏普 | 最大回撤 | 超额年化 | 信息比率 | 被挡(亿) | 持仓数 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for tag, lab in [("zoo_icw", "IC 加权 38 因子"), ("zoo_eq", "等权 38 因子"),
                     ("zoo_ns", "**规模中性化 38 因子**"), ("zoo_lq_amt_log20", "纯规模因子")]:
        S = pd.read_csv(os.path.join(ROOT, "out", "tables", "08_mining_summary.csv"),
                        encoding="utf-8-sig", index_col=0)
        if tag in S.index:
            r = S.loc[tag]
            L.append("| %s | %.2f%% | %.2f%% | %.2f | %.2f%% | %.2f%% | %.2f | %.0f | %.0f |" % (
                lab, 100 * r["累计收益"], 100 * r["年化收益"], r["夏普"], 100 * r["最大回撤"],
                100 * r["超额年化"], r["信息比率"], r["被挡金额_亿"], r["持仓数"]))
    r = S.loc["基准(等权全市场)"]
    L.append("| 基准(股票池内等权) | %.2f%% | %.2f%% | %.2f | %.2f%% | — | — | — | — |" % (
        100 * r["累计收益"], 100 * r["年化收益"], r["夏普"], 100 * r["最大回撤"]))
    L.append("")
    L.append("**这是本次挖掘最重要的结论：**")
    L.append("")
    L.append("1. 规模中性化之后，**超额年化从 +1.22% 变成 −0.63%，信息比率从 0.11 变成 −0.04** —— "
             "38 因子合成的全部超额都来自**规模暴露（小市值 beta）**，而不是因子本身。")
    L.append("2. 纯规模因子 `lq_amt_log20`（买 20 日均成交额最小的股票）单独就有 **+20.03% 年化超额、"
             "IR 1.92**，而且被挡 441 亿元 —— 这是**不可投资的小市值 beta**，不是 alpha。")
    L.append("3. 但规模中性化带来一个**真实且可投资的好处**：最大回撤从 −21.87% 降到 **−12.36%**"
             "（近乎腰斩），年化波动从 19.18% 降到 17.65%，**被挡委托从 246 亿元降到 11 亿元** —— "
             "容量问题一并消失，夏普 1.10 仍高于基准 0.96。")
    L.append("")
    L.append("换句话说：**这条流水线挖出来的不是选股 alpha，而是一个容量友好、低回撤的"
             "风险控制型组合。** 想要真正的 alpha，需要往因子之外找（财务、事件、分析师、"
             "分钟级量价），或者先做行业中性化 —— 而本数据集既无财务也无行业。")
    L.append("")
    L.append("## 9. 复现")
    L.append("")
    L.append("```bash")
    L.append("python s08_factor_mining.py   # 70 因子 -> 校验 -> IC 筛选 -> 去冗余 -> 合成 -> 回测")
    L.append("python s09_mining_report.py  # 净值图 + 规模敏感性与中性化结论")
    L.append("```")
    L.append("")

    out = os.path.join(ROOT, "out", "mining_report.md")
    with open(out, "a", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("已追加 %s" % out)
    print("已写出 %s" % os.path.join(FIG, "06_mining.png"))


if __name__ == "__main__":
    main()
