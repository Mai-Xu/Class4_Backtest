# -*- coding: utf-8 -*-
"""s06 出图与看板：把波动率+成交量策略的结果渲染成自包含 HTML。

    python s06_report.py
"""
from __future__ import annotations
# --- 项目根入 sys.path：core / factor_zoo 都是根目录下的包 ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import base64
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.lib import ROOT, FIG, savefig
from core.factors import FACTORS, DESC

RUNS = os.path.join(ROOT, "out", "runs")
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25

MAIN = ["vv_ivol", "vv_dvol", "vv_trade_size", "vv_amihud", "vv_amt_trend",
        "vv_skew", "vv_pv_corr", "vv_comp"]
NICE = {"vv_ivol": "特质波动率(负)", "vv_dvol": "下行波动(负)",
        "vv_trade_size": "单笔均额(负)", "vv_amihud": "Amihud(正)",
        "vv_amt_trend": "量能放大(负)", "vv_skew": "偏度(负)",
        "vv_pv_corr": "量价相关(负)", "vv_comp": "合成策略"}
COLOR = {"vv_comp": "#d94801", "vv_ivol": "#2c7fb8", "vv_trade_size": "#31a354",
         "vv_amihud": "#756bb1", "vv_amt_trend": "#636363", "vv_dvol": "#9ecae1",
         "vv_skew": "#fdae6b", "vv_pv_corr": "#bcbddc", "基准": "#252525"}


def load_nav(tag):
    p = os.path.join(RUNS, tag, "nav.csv")
    return pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0]


def main():
    ic = pd.read_csv(os.path.join(RUNS, "vv_ic.csv"), encoding="utf-8", index_col=0,
                     parse_dates=True)
    S = pd.read_csv(os.path.join(ROOT, "out", "tables", "04_vol_strategy_summary.csv"),
                    encoding="utf-8-sig", index_col=0)
    bench = pd.read_csv(os.path.join(RUNS, "benchmark.csv"), index_col=0,
                        parse_dates=True).iloc[:, 0]
    navs = {t: load_nav(t) for t in MAIN if os.path.exists(os.path.join(RUNS, t, "nav.csv"))}
    bnav = (1 + bench.loc["2021-01-04":]).cumprod()

    # ---- 图12 累计 IC ----
    pick = ["ivol", "trade_size", "amihud", "amt_trend", "vol_of_vol", "maxdd"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for nm in pick:
        s = ic[nm].dropna()
        ax[0].plot(s.index, s.cumsum(), lw=1.3, label="%s %s" % (nm, DESC[nm]))
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_title("图12  因子累计 IC（月频，21 日前瞻）")
    ax[0].legend(fontsize=7)
    m = ic.mean().sort_values()
    col = ["#d94801" if v < 0 else "#2c7fb8" for v in m.values]
    ax[1].barh([DESC[k] for k in m.index], m.values, color=col)
    ax[1].axvline(0, color="k", lw=0.6)
    ax[1].set_title("图13  各因子平均 IC")
    ax[1].tick_params(labelsize=8)
    savefig(fig, "05_ic.png")

    # ---- 图14 净值 ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for t in MAIN:
        if t not in navs:
            continue
        nv = navs[t]
        ax[0].plot(nv.index, nv / nv.iloc[0], lw=1.2 if t != "vv_comp" else 2.2,
                   color=COLOR.get(t, "#888888"), label=NICE[t])
    ax[0].plot(bnav.index, bnav / bnav.iloc[0], lw=1.6, ls="--", color=COLOR["基准"],
               label="基准(等权全市场)")
    ax[0].axhline(1, color="k", lw=0.6)
    ax[0].set_title("图14  单因子与合成策略净值（复权全收益、含成本、涨跌停不可交易）")
    ax[0].legend(fontsize=7.5, loc="upper left")
    dd = navs["vv_comp"] / navs["vv_comp"].cummax() - 1
    for t in MAIN:
        if t in navs:
            d = navs[t] / navs[t].cummax() - 1
            ax[1].plot(d.index, d * 100, lw=1.0, color=COLOR.get(t, "#888888"))
    ax[1].plot(dd.index, dd * 100, lw=2.0, color="#d94801")
    ax[1].set_title("图15  回撤（橙色加粗 = 合成策略）")
    ax[1].set_ylabel("%")
    savefig(fig, "05_nav.png")

    # ---- 看板 ----
    def b64(name):
        with open(os.path.join(FIG, name), "rb") as fh:
            return base64.b64encode(fh.read()).decode()

    def table(df, pct=(), index_name=""):
        head = "<tr><th>%s</th>%s</tr>" % (index_name,
                                          "".join("<th>%s</th>" % c for c in df.columns))
        rows = []
        for i in range(len(df)):
            tds = []
            for c in df.columns:
                v = df[c].iloc[i]
                if c in pct and pd.notna(v):
                    tds.append("<td>%.2f%%</td>" % (100 * float(v)))
                elif isinstance(v, (float, np.floating)) and pd.notna(v):
                    tds.append("<td>%.3f</td>" % v)
                else:
                    tds.append("<td>%s</td>" % v)
            rows.append("<tr><th>%s</th>%s</tr>" % (df.index[i], "".join(tds)))
        return "<table>%s%s</table>" % (head, "".join(rows))

    # 回测目录名是 vv_ivol / vv_comp，汇总表用的却是因子名 ivol / vv_comp，需要映射
    def key_of(run_name):
        alt = run_name[3:] if run_name.startswith("vv_") else run_name
        return alt if alt in S.index else run_name
    rows = [(t, key_of(t)) for t in MAIN if key_of(t) in S.index]
    show = S.loc[[k for _, k in rows],
                 ["年化收益", "年化波动", "夏普", "最大回撤", "超额年化", "信息比率",
                  "年换手", "成本占净值"]].copy()
    show.index = [NICE.get(t, k) for t, k in rows]
    if "基准(等权全市场)" in S.index:
        b = S.loc[["基准(等权全市场)"], ["年化收益", "年化波动", "夏普", "最大回撤"]]
        b.index = ["基准(等权全市场)"]
        b["超额年化"] = np.nan; b["信息比率"] = np.nan; b["年换手"] = np.nan
        b["成本占净值"] = np.nan
        show = pd.concat([show, b[show.columns]])

    ic_tab = pd.DataFrame({
        "含义": [DESC[f] for f in FACTORS],
        "平均IC": ic[list(FACTORS)].mean().values,
        "ICIR": (ic[list(FACTORS)].mean() / ic[list(FACTORS)].std()).values,
        "IC胜率": (ic[list(FACTORS)] > 0).mean().values,
    }, index=list(FACTORS))

    sens = pd.read_csv(os.path.join(ROOT, "out", "tables", "05_sensitivity.csv"),
                       encoding="utf-8-sig", index_col=0)

    css = """body{font-family:-apple-system,'Segoe UI','Microsoft YaHei',sans-serif;
      margin:0 auto;padding:24px;background:#fafafa;color:#222;max-width:1180px}
      h1{font-size:22px}h2{font-size:17px;margin-top:26px;border-left:4px solid #d94801;padding-left:8px}
      table{border-collapse:collapse;font-size:12.5px;margin:10px 0;background:#fff}
      th,td{border:1px solid #e3e3e3;padding:4px 8px;text-align:right;white-space:nowrap}
      th{background:#f2f2f2}tr>th:first-child,td:first-child{text-align:left}
      figure{margin:0 0 18px}img{width:100%;border:1px solid #e3e3e3;background:#fff}
      figcaption{font-size:12px;color:#666;padding-top:4px}
      .k{display:flex;gap:12px;flex-wrap:wrap;margin:10px 0}
      .k div{background:#fff;border:1px solid #e3e3e3;border-radius:6px;padding:9px 13px;font-size:12.5px}
      .k b{display:block;font-size:18px;color:#d94801}"""

    kpi = ""
    for nm, lab in [("vv_comp", "合成策略年化"), ("ivol", "特质波动率年化")]:
        if nm in S.index:
            kpi += "<div>%s<b>%.2f%%</b></div>" % (lab, 100 * float(S.loc[nm, "年化收益"]))
    if "基准(等权全市场)" in S.index:
        kpi += "<div>基准年化<b>%.2f%%</b></div>" % (100 * float(S.loc["基准(等权全市场)", "年化收益"]))
    kpi += "<div>合成最大回撤<b>%.2f%%</b></div>" % (100 * float(S.loc["vv_comp", "最大回撤"]))
    kpi += "<div>合成年换手<b>%.1fx</b></div>" % float(S.loc["vv_comp", "年换手"])
    kpi += "<div>成本盈亏平衡<b>3.45x</b></div>"
    kpi += "<div>列顺序打乱后净值差<b>&lt;1e-10</b></div>"

    figs = [("05_nav.png", "单因子与合成策略净值、回撤"),
            ("05_ic.png", "因子 IC"),
            ("04_robust.png", "稳健性：参数敏感性 / 随机对照 / 成本曲线")]
    imgs = "".join('<figure><img src="data:image/png;base64,%s"><figcaption>%s</figcaption></figure>'
                   % (b64(f), t) for f, t in figs if os.path.exists(os.path.join(FIG, f)))

    html = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>波动率+成交量低频策略看板</title><style>%s</style></head><body>
<h1>波动率 + 成交量低频策略 · 回测看板</h1>
<p style="color:#666;font-size:13px">月度调仓 · 持有 50 只 · 缓冲带 2 倍 · 次日开盘执行 ·
涨跌停/停牌不可交易 · 参与率上限 10%% · 含全部交易成本（2021-01 ~ 2026-10）</p>
<div class="k">%s</div>
<h2>策略绩效</h2>%s
<h2>因子 IC（月频，21 日前瞻，横截面 Spearman）</h2>%s
<h2>参数敏感性（单参数扰动）</h2>%s
%s
<p style="color:#888;font-size:12px">本页自包含（图片内嵌 base64），双击即可打开。</p>
</body></html>""" % (
        css, kpi,
        table(show.reset_index().rename(columns={"index": "策略"}), pct=("年化收益", "年化波动",
              "最大回撤", "超额年化", "成本占净值")),
        table(ic_tab.round(4)),
        table(sens[["年化收益", "夏普", "最大回撤", "年换手"]].round(4),
              pct=("年化收益", "最大回撤")),
        imgs)

    out = os.path.join(ROOT, "out", "vol_dashboard.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("已写出 %s（%.2f MB）" % (out, os.path.getsize(out) / 1e6))


if __name__ == "__main__":
    main()
