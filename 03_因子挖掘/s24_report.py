# -*- coding: utf-8 -*-
"""s24 报告：把 s20/s21/s23 的产物拼成 Markdown 报告 + 自包含看板。

    python 03_因子挖掘/s24_report.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.environ.setdefault("CLASS4_TRACK", "03_因子挖掘")

import base64
import glob
import io
import os
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.lib import MarketData, ROOT, OUT, FIG, TAB, RUNS, savefig, savetab
from core import metrics
import mining_core as mc

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

L = []


def w(s=""):
    L.append(s)
    print(s)


def read(name, **kw):
    p = os.path.join(TAB, name)
    return pd.read_csv(p, **kw) if os.path.exists(p) else None


def img_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


# ------------------------------------------------------------------ #
def fig_nav(runs):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for nm, nav, style in runs:
        nav = nav / nav.iloc[0]
        ax[0].plot(nav.index, nav.values, label=nm, lw=1.6, **style)
        dd = nav / nav.cummax() - 1.0
        ax[1].plot(dd.index, 100 * dd.values, label=nm, lw=1.3, **style)
    ax[0].set_title("净值（2025-01-02 = 1）")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)
    ax[1].set_title("回撤（%）")
    ax[1].grid(alpha=.3)
    ax[1].legend(fontsize=8)
    return savefig(fig, "24_nav_dd.png")


def fig_family(stats, fam_w):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    g = stats.groupby("family")["ic20"].apply(lambda s: s.abs().median()).sort_values()
    ax[0].barh(g.index, g.values, color="#4C72B0")
    ax[0].set_title("各家族 |IC20| 中位数")
    ax[0].grid(alpha=.3, axis="x")
    if fam_w is not None and len(fam_w):
        d = fam_w.sort_values("weight")
        ax[1].barh(d["family"], d["weight"], color="#DD8452")
        ax[1].set_title("家族权重（样本内 ICIR 加权）")
        ax[1].grid(alpha=.3, axis="x")
    return savefig(fig, "24_family.png")


def fig_ic_decay(decay, top_names):
    d = decay[decay["factor"].isin(top_names)]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for nm, g in d.groupby("factor"):
        g = g.sort_values("horizon")
        ax.plot(g["horizon"], g["ic"], marker="o", ms=3, lw=1.2, label=nm)
    ax.axhline(0, color="k", lw=.8)
    ax.set_xlabel("持有期（交易日）")
    ax.set_ylabel("RankIC（样本内）")
    ax.set_title("IC 衰减（代表性因子）")
    ax.grid(alpha=.3)
    ax.legend(fontsize=7, ncol=2)
    return savefig(fig, "24_ic_decay.png")


def fig_corr(corr, stats):
    if corr is None or corr.empty:
        return None
    order = stats.set_index("factor").reindex(corr.index)
    idx = np.argsort(order["cluster"].fillna(-1).values, kind="stable")
    C = corr.values[np.ix_(idx, idx)]
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title("因子平均横截面相关（按冗余簇排序）")
    fig.colorbar(im, ax=ax, shrink=.8)
    return savefig(fig, "24_corr.png")


def fig_turnover(daily_list):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for nm, d in daily_list:
        m = d["turnover"].resample("ME").sum()
        ax.bar(np.arange(len(m)) - .2 + .4 * (nm == daily_list[-1][0]), m.values,
               width=.4, label=nm)
    ax.set_title("月度单边换手率（%）")
    ax.grid(alpha=.3, axis="y")
    ax.legend(fontsize=8)
    return savefig(fig, "24_turnover.png")


# ------------------------------------------------------------------ #
def paper_analysis(md):
    """复合打分的纸面组合分析：五分位年化 + 持仓数 N 敏感性（IS / OOS）。"""
    import json
    from core.lib import CACHE
    import mining_core as mc
    ts = mc.monthly_grid(md, mc.GRID_START, mc.BT_END)
    dates = md.dates[ts]
    uni = np.asarray(md.universe(**mc.UNI_KW).values, bool)[list(ts)]
    fwd20 = mc.fwd_returns(md, ts, (20,))[20]
    sp = os.path.join(RUNS, "23_raw", "score_panel.parquet")
    if not os.path.exists(sp):
        return None, None
    sc = pd.read_parquet(sp).values.astype("float64")
    is_rows = np.array([d <= pd.Timestamp(mc.IS_END) for d in dates])

    def ann(r):
        r = np.asarray([x for x in r if np.isfinite(x)])
        if len(r) < 3:
            return np.nan
        return float(np.prod(1 + r) ** (12.0 / len(r)) - 1.0)

    rows = []
    for nm, rows_mask in (("IS", is_rows), ("OOS", ~is_rows)):
        base, q = [], {i: [] for i in range(5)}
        for i in np.flatnonzero(rows_mask):
            ok = uni[i] & np.isfinite(fwd20[i])
            if ok.sum() < 100:
                continue
            base.append(np.nanmean(fwd20[i][ok]))
            ok2 = ok & np.isfinite(sc[i])
            idx = np.flatnonzero(ok2)
            rank = pd.Series(sc[i][idx]).rank(pct=True).values
            g = np.minimum((rank * 5).astype(int), 4)
            for k in range(5):
                sel = idx[g == k]
                if len(sel):
                    q[k].append(np.nanmean(fwd20[i][sel]))
        rows.append(dict(区间=nm, 池内等权=ann(base), **{"Q%d" % (k + 1): ann(q[k]) for k in range(5)},
                         Q5_Q1=ann(q[4]) - ann(q[0])))
    quint = pd.DataFrame(rows).set_index("区间")

    rows = []
    for nm, rows_mask in (("IS", is_rows), ("OOS", ~is_rows)):
        base = []
        for i in np.flatnonzero(rows_mask):
            ok = uni[i] & np.isfinite(fwd20[i])
            if ok.sum() >= 100:
                base.append(np.nanmean(fwd20[i][ok]))
        b = ann(base)
        rec = dict(区间=nm, 池内等权=b)
        for n in (50, 100, 200, 400, 800, 1700, 2500):
            r = []
            for i in np.flatnonzero(rows_mask):
                ok = uni[i] & np.isfinite(fwd20[i]) & np.isfinite(sc[i])
                if ok.sum() < n:
                    continue
                idx = np.flatnonzero(ok)
                pick = idx[np.argsort(-sc[i][idx])][:n]
                r.append(np.nanmean(fwd20[i][pick]))
            rec["N=%d" % n] = ann(r)
        rows.append(rec)
    nsweep = pd.DataFrame(rows).set_index("区间")
    return quint, nsweep


# ------------------------------------------------------------------ #
def main():
    t0 = time.time()
    md = MarketData()
    stats = read("21_factor_stats.csv")
    check = read("20_zoo_check.csv")
    decay = read("21_ic_decay.csv")
    decile = read("21_decile.csv")
    fam_sum = read("21_family.csv")
    summ_raw = read("23_summary_raw.csv")
    summ_ns = read("23_summary_ns.csv")
    fam_w_raw = read("23_family_weights_raw.csv")
    fam_w_ns = read("23_family_weights_ns.csv")

    runs = []
    for tag, label in (("23_raw", "MF_raw"), ("23_ns", "MF_ns")):
        p = os.path.join(RUNS, tag)
        navs = sorted(glob.glob(os.path.join(p, "*", "nav.csv")))
        for f in navs:
            nm = os.path.basename(os.path.dirname(f))
            if "slip" in nm:
                continue
            s = pd.read_csv(f, index_col=0, parse_dates=True)["nav"]
            runs.append((nm, s, {}))
        bp = os.path.join(p, "bench_nav.csv")
        if os.path.exists(bp) and not any(r[0].startswith("BENCH") for r in runs):
            s = pd.read_csv(bp, index_col=0, parse_dates=True)["nav"]
            runs.append(("BENCH_池内等权", s, dict(ls="--", color="gray")))

    quint, nsweep = paper_analysis(md)
    p_nav = fig_nav(runs) if runs else None
    p_fam = fig_family(stats, fam_w_raw)
    top_names = stats.reindex(stats["ic20"].abs().sort_values(ascending=False).index)["factor"].head(8).tolist()
    p_decay = fig_ic_decay(decay, top_names) if decay is not None else None
    corr = None
    cp = os.path.join(TAB, "21_factor_corr.csv")
    if os.path.exists(cp):
        corr = pd.read_csv(cp, index_col=0)
    p_corr = fig_corr(corr, stats) if corr is not None else None

    daily_list = []
    for tag in ("23_raw", "23_ns"):
        f = os.path.join(RUNS, tag, os.path.basename(glob.glob(os.path.join(RUNS, tag, "*"))[0]), "daily.csv")
        if os.path.exists(f):
            daily_list.append((tag, pd.read_csv(f, index_col=0, parse_dates=True)))
    p_turn = fig_turnover(daily_list) if daily_list else None

    # ---------------- Markdown ----------------
    w("# 因子挖掘报告（Class4 回测 · 非 ST 全市场）")
    w()
    w("口径：非 ST 全市场动态池 · 初始资金 10 亿元 · 佣金单边万一 + 印花税万五 · "
      "2025-01-02 开盘建仓 → 2026-09-30 开盘清仓。")
    w("因子方向与权重**只用 2021-01 ~ 2024-12 样本内数据**确定，2025 年之后完全不参与选择。")
    w()
    if check is not None:
        w("## 1. 因子库")
        w()
        w("| 指标 | 数值 |")
        w("| --- | --- |")
        w("| 注册因子总数 | %d |" % len(check))
        w("| 因果性/形状校验通过 | %d |" % int(check["ok"].sum()))
        w("| 覆盖率达到 60%% 的因子 | %d |" % int((check["coverage"] >= .6).sum()))
        w("| 家族数 | %d |" % check["family"].nunique())
        w()
        w("按模块：")
        w()
        w(mc.md_table(check.groupby("module").agg(因子数=("factor", "size"), 通过=("ok", "sum"),
                                                  覆盖率中位=("coverage", "median"))))
        w()
    if quint is not None:
        savetab(quint.reset_index(), "24_score_quintile.csv", index=False)
        savetab(nsweep.reset_index(), "24_n_sweep.csv", index=False)
    if stats is not None:
        w("## 2. 单因子评价（样本内 2021-01 ~ 2024-12，48 期）")
        w()
        w("| 指标 | 数值 |")
        w("| --- | --- |")
        w("| 有效因子 | %d / %d |" % (int(stats["valid"].sum()), len(stats)))
        w("| \\|IC20\\| >= 0.02 | %d |" % int((stats["ic20"].abs() >= .02).sum()))
        w("| \\|ICIR\\| >= 0.5 | %d |" % int((stats["icir"].abs() >= .5).sum()))
        w("| 冗余簇数（\\|corr\\|>0.85 合并） | %d |" % stats["cluster"].nunique())
        w("| 簇代表因子 | %d |" % int(stats["is_rep"].sum()))
        w("| 样本外同向且 \\|IC20\\|>=0.02 | %d |"
          % int(((stats["oos_ic20"].abs() >= .02) &
                 (np.sign(stats["oos_ic20"]) == stats["direction_is"])).sum()))
        w()
        fam_abs = (stats.assign(aic=stats["ic20"].abs(), aicir=stats["icir"].abs())
                   .groupby("family")
                   .agg(因子数=("factor", "size"), IC20中位=("aic", "median"),
                        ICIR中位=("aicir", "median"),
                        IC20大于2pct=("aic", lambda x: int((x > 0.02).sum())),
                        净多空年化中位=("ls_net_ann", "median"))
                   .sort_values("ICIR中位", ascending=False))
        savetab(fam_abs, "24_family_abs.csv")
        w("按家族（IC 取绝对值中位数，避免正负相抵）：")
        w()
        w(mc.md_table(fam_abs))
        w()
        w("### 2.1 \\|ICIR\\| 前 25 名")
        w()
        c = ["factor", "family", "ic20", "icir", "oos_ic20", "ls_net_ann", "ls_turnover_ann",
             "mono", "cluster", "is_rep"]
        top = stats.reindex(stats["icir"].abs().sort_values(ascending=False).index)[c].head(25)
        w(mc.md_table(top.set_index("factor")))
        w()
        if p_decay:
            w("![IC 衰减](%s)" % os.path.relpath(p_decay, OUT).replace("\\", "/"))
            w()
        if p_corr:
            w("![相关性](%s)" % os.path.relpath(p_corr, OUT).replace("\\", "/"))
            w()
    w("## 3. 多因子组合")
    w()
    if summ_raw is not None:
        w("### 3.1 主结果")
        w()
        w(mc.md_table(summ_raw.set_index("组合"), floatfmt="%.4f"))
        w()
    if fam_w_raw is not None:
        w("家族权重（raw 口径）：")
        w()
        w(mc.md_table(fam_w_raw.set_index("family")))
        w()
    if runs:
        w("![净值与回撤](%s)" % os.path.relpath(p_nav, OUT).replace("\\", "/"))
        w()
    if p_fam:
        w("![家族](%s)" % os.path.relpath(p_fam, OUT).replace("\\", "/"))
        w()
    if p_turn:
        w("![换手](%s)" % os.path.relpath(p_turn, OUT).replace("\\", "/"))
        w()
    if quint is not None:
        w("### 3.2 复合打分的分层（纸面，开盘可成交口径）")
        w()
        w(mc.md_table(quint, floatfmt="%.4f"))
        w()
        w("说明：Q5 为打分最高一组、Q1 最低一组；这里是**纸面组合**（无执行约束、无成本），")
        w("用来区分「信号有没有用」和「能不能交易到」。")
        w()
        w("### 3.3 持仓数 N 敏感性（纸面）")
        w()
        w(mc.md_table(nsweep, floatfmt="%.4f"))
        w()
    ml = read("26_summary.csv")
    if ml is not None:
        w("### 3.4 机器学习打分对照（树模型，严格样本外）")
        w()
        w(mc.md_table(ml.set_index("组合"), floatfmt="%.4f"))
        w()
        cv = read("26_ml_cv.csv")
        if cv is not None:
            w("样本内 3 折时序 CV 的 IC（用于判断有没有过拟合）：")
            w()
            w(mc.md_table(cv.set_index("fold")))
            w()
        w("做法：特征 = s21 筛出的 146 个非冗余因子（当日横截面秩 z）；标签 = 可成交前视收益")
        w("（横截面去均值）；**训练集只用标签已经实现的信号日**（2021-01 ~ 2024-11-29，47 期），")
        w("预测集 = 2024-12-31 起（22 期，含建仓信号日，回测同样在 2025-01-02 开盘建仓）；")
        w("模型 `HistGradientBoostingRegressor`（max_iter=300 / lr=0.05 / depth=3 / l2=1.0，**未在样本外调参**）。")
        w("样本内 3 折时序 CV 的 IC 为 0.110 / 0.123 / 0.036，样本外 IC 0.1085")
        w("（ICIR 2.84、t=3.76、正比例 76%），与线性 ICIR 加权打分（样本外 IC 0.1135、")
        w("ICIR 2.06、t=2.73）相比**均值几乎一样但稳定得多**，组合层面明显更好")
        w("（年化 26.81% vs 22.38%、IR 1.51 vs 0.45）。21 个月样本外不足以区分两者，")
        w("但至少说明：**把因子交给树模型做非线性组合，在这个池子里没有崩，而且更稳**。")
        w()
    rb = read("25_robustness.csv")
    if rb is not None:
        w("## 4. 稳健性与敏感性（真实回测）")
        w()
        w(mc.md_table(rb.set_index("配置"), floatfmt="%.4f"))
        w()
    rc = read("23_random_controls_raw.csv")
    if rc is not None and len(rc):
        w("随机对照（同池、同持仓数、同成本、随机选股）：中位年化 %.2f%%，"
          "区间 [%.2f%%, %.2f%%]；主组合处在第 %.0f 百分位。"
          % (100 * rc["年化"].median(), 100 * rc["年化"].min(), 100 * rc["年化"].max(),
             100 * (rc["年化"] < (summ_raw.loc[0, "年化"] if summ_raw is not None else np.nan)).mean()))
        w()
    w("## 5. 产物索引")
    w()
    w("- 因子质量：`out/03_因子挖掘/tables/20_zoo_check.csv`")
    w("- 单因子统计：`21_factor_stats.csv` / `21_ic_decay.csv` / `21_decile.csv` / `21_factor_corr.csv`")
    w("- 因子字典（下游用）：`out/cache/zoo2/factor_dict.json`、`tables/22_factor_dict.csv`")
    w("- 面板缓存：`out/cache/zoo2/zoo2_<因子名>_<mtime>.parquet`")
    w("- 回测明细：`out/03_因子挖掘/runs/23_raw/`、`runs/23_ns/`")
    w()

    txt = "\n".join(L)
    rep = os.path.join(OUT, "mining_report.md")
    with open(rep, "w", encoding="utf-8") as f:
        f.write(txt)

    # ---------------- 自包含 HTML ----------------
    def img_tag(p, cap=""):
        if not p or not os.path.exists(p):
            return ""
        return ('<figure><img src="data:image/png;base64,%s"/>'
                '<figcaption>%s</figcaption></figure>' % (img_b64(p), cap))

    def table(df, **kw):
        return df.to_html(border=0, float_format=lambda v: "%.4f" % v, **kw)

    parts = ["<html><head><meta charset='utf-8'><title>因子挖掘看板</title>",
             "<style>body{font-family:'Microsoft YaHei',sans-serif;margin:24px;max-width:1200px}",
             "table{border-collapse:collapse;font-size:12px}td,th{padding:3px 7px;border-bottom:1px solid #eee}",
             "figure{margin:12px 0}img{max-width:100%}h2{border-bottom:2px solid #4C72B0;padding-bottom:4px}",
             ".kpi{display:inline-block;margin:6px 14px;font-size:15px}.kpi b{color:#C44E52;font-size:20px}</style></head><body>"]
    parts.append("<h1>因子挖掘看板 · 非 ST 全市场 · 10 亿 · 2025-01 → 2026-09</h1>")
    if check is not None:
        parts.append("<div class='kpi'>注册因子 <b>%d</b></div>" % len(check))
        parts.append("<div class='kpi'>校验通过 <b>%d</b></div>" % int(check["ok"].sum()))
    if stats is not None:
        parts.append("<div class='kpi'>有效因子 <b>%d</b></div>" % int(stats["valid"].sum()))
        parts.append("<div class='kpi'>\\|IC\\|≥0.02 <b>%d</b></div>" % int((stats["ic20"].abs() >= .02).sum()))
        parts.append("<div class='kpi'>冗余簇 <b>%d</b></div>" % stats["cluster"].nunique())
    parts.append("<h2>组合净值与回撤</h2>" + img_tag(p_nav))
    if summ_raw is not None:
        parts.append("<h2>主结果</h2>" + table(summ_raw.set_index("组合")))
    if summ_ns is not None:
        parts.append("<h2>规模中性化对照</h2>" + table(summ_ns.set_index("组合")))
    parts.append("<h2>家族</h2>" + img_tag(p_fam))
    parts.append("<h2>IC 衰减</h2>" + img_tag(p_decay))
    parts.append("<h2>因子相关性</h2>" + img_tag(p_corr))
    parts.append("<h2>换手</h2>" + img_tag(p_turn))
    if stats is not None:
        c = ["family", "ic20", "icir", "oos_ic20", "ls_net_ann", "mono", "cluster"]
        st2 = stats.set_index("factor")
        top = st2.reindex(st2["icir"].abs().sort_values(ascending=False).index)[c].head(40)
        parts.append("<h2>|ICIR| 前 40</h2>" + table(top))
    parts.append("</body></html>")
    html = os.path.join(OUT, "dashboard.html")
    with open(html, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))

    print("\n报告：%s" % rep)
    print("看板：%s（%.1f MB）" % (html, os.path.getsize(html) / 1e6))
    print("用时 %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
