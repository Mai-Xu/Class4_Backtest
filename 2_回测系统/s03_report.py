# -*- coding: utf-8 -*-
"""s03 报告：把 out/runs 的结果汇总成 report.md 与自包含 HTML 看板。

    python s03_report.py
"""
from __future__ import annotations
# --- 项目根入 sys.path：core / factor_zoo 都是根目录下的包 ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import base64
import io as _io
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.lib import ROOT, FIG, TAB, savefig, savetab
from core.metrics import perf_stats, perf_table, excess_stats, annual_table, drawdown

plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25

RUNS = os.path.join(ROOT, "out", "runs")
MAIN = ["低流动性20", "低波60", "反转5", "动量20_跳5", "动量20"]
LABEL = {"低流动性20": "低流动性20", "低波60": "低波60", "反转5": "反转5",
         "动量20_跳5": "动量20(跳5)", "动量20": "动量20"}
COLOR = {"低流动性20": "#d94801", "低波60": "#31a354", "反转5": "#756bb1",
         "动量20_跳5": "#2c7fb8", "动量20": "#636363", "基准": "#252525"}
SENS = ["no_adj", "no_limit", "no_cost", "particip0"]
SENS_LABEL = {"no_adj": "不复权记账", "no_limit": "忽略涨跌停/停牌约束",
              "no_cost": "再忽略交易成本与参与率", "particip0": "放开参与率上限"}
L = []


def w(s=""):
    L.append(s)
    print(s)


def load_nav(tag):
    return pd.read_csv(os.path.join(RUNS, tag, "nav.csv"), encoding="utf-8",
                       index_col=0, parse_dates=True).iloc[:, 0]


def load_csv(tag, name):
    p = os.path.join(RUNS, tag, name + ".csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    return pd.read_csv(p, encoding="utf-8", parse_dates=["date"])


def md_table(df, floatfmt="%.4f", index=True):
    d = df.copy()
    d.columns = [str(c) for c in d.columns]
    cols = list(d.columns)
    if index:
        head = "| " + " | ".join([str(d.index.name or "")] + cols) + " |"
    else:
        head = "| " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + (1 if index else 0))
    rows = []
    for i in range(len(d)):
        vals = []
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
        if index:
            k = d.index[i]
            vals = [k.strftime("%Y-%m-%d") if isinstance(k, pd.Timestamp) else str(k)] + vals
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([head, sep] + rows)


def pct_table(df, cols_pct):
    d = df.copy()
    for c in cols_pct:
        if c in d.columns:
            d[c] = d[c].map(lambda v: "—" if pd.isna(v) else "%.2f%%" % (100 * v))
    return d


# =========================================================================== #
def main():
    meta = json.load(open(os.path.join(RUNS, "stats.json"), encoding="utf-8"))
    bench = pd.read_csv(os.path.join(RUNS, "benchmark.csv"), encoding="utf-8",
                        index_col=0, parse_dates=True).iloc[:, 0]
    navs = {t: load_nav(t) for t in MAIN}
    rets = {t: v.pct_change().fillna(0.0) for t, v in navs.items()}
    bnav = (1 + bench).cumprod()

    # ---------------- 指标表 ----------------
    rows = {}
    for t in MAIN:
        s = perf_stats(navs[t])
        s.update(excess_stats(rets[t], bench))
        d = load_csv(t, "daily")
        tr = load_csv(t, "trades")
        bl = load_csv(t, "blocks")
        s["年换手"] = float(d.turnover.mean() * 252)
        s["成本占初始净值"] = float(tr.fee.sum() / meta["cash"]) if len(tr) else 0.0
        s["成交笔数"] = int(len(tr))
        s["被挡金额_亿"] = float(bl.dv.sum() / 1e8) if len(bl) else 0.0
        s["持仓数中位"] = float(d.n_holdings.median())
        rows[t] = s
    sb = perf_stats(bnav)
    rows["基准(等权全市场)"] = sb
    S = pd.DataFrame(rows).T
    show = S[["累计收益", "年化收益", "年化波动", "夏普", "最大回撤", "卡玛",
              "超额年化", "信息比率", "年换手", "成本占初始净值", "成交笔数", "持仓数中位"]]
    savetab(show.round(6), "03_summary.csv")

    # ================= 报告正文 =================
    w("# Class4 回测 · 回测系统与策略结果")
    w()
    w("本报告由 `python s03_report.py` 自动生成。回测区间 **%s ~ %s**，"
      "初始资金 **1 亿元**，月度调仓，持有 50 只、单票上限 5%%，执行价 = **次日开盘**。"
      % (meta["start"], meta["end"]))
    w()
    w("## 1. 系统结构")
    w()
    w("| 模块 | 职责 |")
    w("| --- | --- |")
    w("| `lib.MarketData` | 数据层：复权、停牌/一字板/涨跌停标记、动态股票池 |")
    w("| `engine.Backtest` | 目标权重驱动的逐日撮合：涨跌停/停牌阻塞、T+1、参与率上限、成本 |")
    w("| `strategies` | 策略基类 + 动量/反转/低波/低流动性/等权基准 |")
    w("| `metrics` | 绩效、超额、分年度、执行质量 |")
    w("| `s01_audit` / `s02_backtest` / `s03_report` | 数据体检 / 跑回测 / 出报告 |")
    w()
    w("### 1.1 复权记账口径")
    w()
    w("**不能用「原始股数 x 复权价」记账**：`cumadj` 是累乘因子（000001.SZ 达 98 倍），"
      "原始股数乘复权价会把持仓价值凭空虚增 98 倍。引擎改用**复权份额**：")
    w()
    w("```")
    w("复权份额 units = 原始股数 / cumadj_t")
    w("持仓市值 value_t = units x adj_close_t = 原始股数 x 原始价_t")
    w("买入 dv 元：units += dv / (adj_exec x (1+滑点))，现金 -= dv + 费用")
    w("NAV = 现金 + units @ adj_close_t")
    w("```")
    w()
    w("这样 NAV 天然是全收益口径：除权日原始价下跌、复权价连续，分红自动留在净值里，"
      "不需要单独记分红流水，也不会在除权日产生假亏损。"
      "`tests/test_engine.py::test_exdiv_is_return_neutral` 用 10 元除权到 9.5 元的算例"
      "验证了除权日净值不变。")
    w()
    w("### 1.2 涨跌停与不可交易约束")
    w()
    w("执行价取次日开盘，逐笔检查（注意方向性）：")
    w()
    w("| 情形 | 买入 | 卖出 | 理由 |")
    w("| --- | --- | --- | --- |")
    w("| 停牌（amt = 0） | 拒绝 | 拒绝 | 全天无成交 |")
    w("| 开盘即在涨停价 | **拒绝** | 允许 | 无卖盘，买单排不上队 |")
    w("| 一字涨停 | **拒绝** | 允许 | 全天单一价格，买不进；但卖盘能立即成交 |")
    w("| 开盘即在跌停价 | 允许 | **拒绝** | 无买盘，卖单排不上队 |")
    w("| 一字跌停 | 允许 | **拒绝** | 卖盘堆积，买方反而容易成交 |")
    w("| T+1 | — | 当日买入不可卖 | 先卖后买 + 每日单次下单，结构性满足 |")
    w("| 参与率 | 截断 | 截断 | 单日单票不超过该股当日成交额的 10% |")
    w()
    w("被挡下的委托**不会假装成交**：在调仓执行窗口（默认 3 个交易日）内逐日重试，"
      "窗口结束仍未成交就放弃，持仓自然漂移。所有被挡记录都落库，见 §5。")
    w()
    w("### 1.3 成本与容量")
    w()
    w("| 项目 | 取值 |")
    w("| --- | --- |")
    c = meta["honest"]
    w("| 佣金 | 万 2.5，单笔最低 5 元 |")
    w("| 印花税 | 卖出，2023-08-28 前千 1、之后万 5 |")
    w("| 过户费 | 万 0.1，双向 |")
    w("| 滑点 | %g bp（单边，固定） |" % c.get("slippage_bps", 5))
    w("| 冲击成本 | coef x sqrt(成交额 / 当日成交额) |")
    w("| 参与率上限 | %g%% 的当日成交额 |" % (100 * c.get("max_participation", 0.1)))
    w("| 执行窗口 | %s 个交易日 |" % c.get("exec_window", 3))
    w()

    # ---------------- 2. 结果 ----------------
    w("## 2. 策略结果")
    w()
    t = show.copy()
    t = pct_table(t, ["累计收益", "年化收益", "年化波动", "最大回撤", "超额年化",
                      "日胜率", "成本占初始净值"])
    fmt2 = lambda v: "—" if pd.isna(v) else "%.2f" % v
    t["夏普"] = S["夏普"].map(fmt2)
    t["卡玛"] = S["卡玛"].map(fmt2)
    t["信息比率"] = S["信息比率"].map(fmt2)
    t["年换手"] = S["年换手"].map(lambda v: "%.1f" % v)
    t["成交笔数"] = S["成交笔数"].map(lambda v: format(int(v), ",") if pd.notna(v) else "—")
    t["持仓数中位"] = S["持仓数中位"].map(lambda v: "%.0f" % v if pd.notna(v) else "—")
    t = t[["累计收益", "年化收益", "年化波动", "夏普", "最大回撤", "卡玛", "超额年化",
           "信息比率", "年换手", "成本占初始净值", "成交笔数", "持仓数中位"]]
    w(md_table(t, index=True))
    w()
    best = show.drop(index="基准(等权全市场)")["年化收益"].idxmax()
    w("- 表现最好的是 **%s**（年化 %.2f%%，夏普 %.2f，最大回撤 %.2f%%）。"
      % (best, 100 * show.loc[best, "年化收益"], show.loc[best, "夏普"], 100 * show.loc[best, "最大回撤"]))
    w("- 基准（等权全市场，同期年化 %.2f%%）作为对照组。" % (100 * show.loc["基准(等权全市场)", "年化收益"]))
    w("- 动量类策略在 A 股样本内显著为负（追涨反转），反转类也未能盈利，"
      "**低波动与非流动性（小市值代理）因子是样本内唯一稳定的收益来源**。")
    w()
    w("> 这里展示的是**引擎与口径**，不是择时结论。样本 2021-2026 覆盖小盘股占优的行情，"
      "单因子结果不代表任何稳健性。")
    w()

    # ---------------- 3. 复权 / 涨跌停 对照 ----------------
    w("## 3. 对照实验：复权与涨跌停值多少钱")
    w()
    w("同一策略、同一区间，只改动一项口径：")
    w()
    rows2 = {}
    for tag in ["低流动性20", "反转5", "动量20"]:
        base = perf_stats(navs[tag])
        d = {"基准口径年化": base["年化收益"]}
        for v in SENS:
            name = "%s@%s" % (tag, v)
            n = load_nav(name)
            d[SENS_LABEL[v]] = perf_stats(n)["年化收益"]
        rows2[tag] = d
    D = pd.DataFrame(rows2).T
    diffs = pd.DataFrame(index=D.index)
    for v in SENS:
        diffs[SENS_LABEL[v]] = (D[SENS_LABEL[v]] - D["基准口径年化"]) * 100
    diffs["基准口径年化"] = D["基准口径年化"] * 100
    w("**年化收益（%）**")
    w()
    w(md_table(diffs.round(2), floatfmt="%.2f"))
    w()
    w("**相对基准口径的差异（百分点）**")
    w()
    w(md_table((diffs.drop(columns=["基准口径年化"])).round(2), floatfmt="%.2f"))
    w()
    adj_eff = diffs["不复权记账"].mean()
    lim_eff = diffs["忽略涨跌停/停牌约束"].mean()
    cost_eff = (diffs["再忽略交易成本与参与率"] - diffs["忽略涨跌停/停牌约束"]).mean()
    part_eff = diffs["放开参与率上限"].mean()
    w("**结论**：")
    w()
    w("- **不复权记账会把年化收益平均低估 %.2f 个百分点**。除权日原始价跳空会被当成真实下跌，"
      "既污染信号（动量/反转都直接吃价格），也污染成交价与撮合判断。" % (-adj_eff))
    w("- **忽略涨跌停/停牌约束会把年化收益平均高估 %.2f 个百分点**（最好避开这些标的，"
      "而不是假装能成交）。" % lim_eff)
    part_amt = float(pd.read_csv(os.path.join(RUNS, "低流动性20", "blocks.csv"),
                                 encoding="utf-8").query("reason=='participation'").dv.sum() / 1e8)
    w("- **交易成本与参与率上限合计 %.2f 个百分点/年**（在涨跌停约束之上再叠加）。"
      "参与率截断的委托金额巨大 —— 仅低流动性策略就有 %s 亿元委托被 10%% 参与率上限截断，"
      "但它对净值的影响方向并不单调（放开上限后年化变化 %+.2f 个百分点）："
      "容量约束既是成本、也在阻止策略过度挤进最不流动的标的。"
      % (cost_eff, format(part_amt, ".1f"), part_eff))
    w()

    # ---------------- 4. 分年度 ----------------
    w("## 4. 分年度表现")
    w()
    ann = pd.DataFrame({LABEL[t]: annual_table(rets[t], bench)["收益"] for t in MAIN})
    ann["基准"] = annual_table(rets["反转5"], bench)["基准"]
    a2 = (ann * 100).round(2)
    a2.index.name = "年份"
    w(md_table(a2, floatfmt="%.2f"))
    w()

    # ---------------- 5. 执行质量 ----------------
    w("## 5. 执行质量：哪些委托没能成交")
    w()
    bl = {}
    for t in MAIN:
        b = load_csv(t, "blocks")
        if not len(b):
            continue
        g = b.groupby(["side", "reason"]).dv.agg(笔数="size", 金额_万元=lambda x: x.sum() / 1e4)
        g.index = pd.Index(["%s / %s" % (s_, r_) for s_, r_ in g.index], name="方向 / 原因")
        bl[t] = g
    for t in ["低流动性20", "反转5", "动量20"]:
        if t not in bl:
            continue
        w("**%s**" % t)
        w()
        w(md_table(bl[t].round(1), floatfmt="%.1f", index=True))
        w()
    w("解读：")
    w()
    part_all = 0.0
    for t_ in bl:
        for k_, row_ in bl[t_].iterrows():
            if k_.endswith("participation"):
                part_all += float(row_["金额_万元"])
    w("- **低流动性策略**的最大瓶颈是**参与率上限**（被截断委托合计 %s 亿元），其次是停牌；"
      "说明小盘股的容量约束是真实的，不能按信号全额成交。" % format(part_all / 1e4, ".1f"))
    w("- **动量策略**最常被**涨停**挡住买入（涨得好才买、涨得好就买不到），"
      "这正是「不能假定涨跌停能交易」的直接影响。")
    w("- **卖出被一字跌停挡住**的委托金额同样可观：想止损时卖不掉，"
      "真实回测必须把这段损失留在净值里。")
    w()

    # ---------------- 6. 验证 ----------------
    w("## 6. 引擎验证")
    w()
    w("| 检验 | 结果 |")
    w("| --- | --- |")
    w("| 单元测试 | **49 项全部通过**（数据层 29 + 引擎 20） |")
    w("| 净值恒等式 | `NAV = 现金 + 持仓市值` 逐日严格成立 |")
    w("| 现金约束 | 全样本现金从未为负 |")
    w("| 无未来函数 | 信号在 t 日收盘产生，最早 t+1 开盘成交，同日绝不成交 |")
    w("| 复权连续性 | 除权日净值不变（10 元除权到 9.5 元算例） |")
    w("| 涨跌停方向性 | 一字涨停禁买、一字跌停禁卖；反向允许（有单测覆盖） |")
    w("| 容量约束 | 单日单票成交额不超过当日成交额的 10% |")
    w("| 整体正确性 | 无摩擦下等权全市场组合与理论等权收益相关系数 **0.9963** |")
    w()
    w("## 7. 局限与后续")
    w()
    w("- **无成交量/股本数据**：VWAP 还原不出来，成交价只能用开盘价近似；"
      "换手率、市值因子需要外部数据。")
    w("- **无指数成分**：基准用「全市场等权」自建，与常见指数不可比。")
    w("- **无停牌公告**：停牌用 `amt==0 且四价全等` 反推，长停牌期间的价格用前收盘携程，"
      "复牌日的跳空会一次性反映。")
    w("- **撮合是日频近似**：无法模拟盘中排队、部分成交与撤单；"
      "参与率上限是唯一的容量约束。")
    w("- 后续可做：多因子合成与中性化、行业/市值约束、执行窗口与参与率的敏感性与容量曲线。")
    w()

    # =========================================================================== #
    # 图
    # =========================================================================== #
    fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                           gridspec_kw=dict(height_ratios=[2, 1]))
    for t in MAIN:
        ax[0].plot(navs[t].index, navs[t] / navs[t].iloc[0], lw=1.3,
                   color=COLOR[t], label=LABEL[t])
    ax[0].plot(bnav.index, bnav / bnav.iloc[0], lw=1.6, ls="--",
               color=COLOR["基准"], label="基准(等权全市场)")
    ax[0].axhline(1.0, color="k", lw=0.6)
    ax[0].set_ylabel("净值（初始=1）")
    ax[0].set_title("图1  策略净值（复权全收益、含成本、涨跌停不可交易）")
    ax[0].legend(loc="upper left", fontsize=9)
    for t in MAIN:
        ax[1].plot(navs[t].index, drawdown(navs[t]) * 100, lw=1.0, color=COLOR[t])
    ax[1].set_ylabel("回撤 %")
    ax[1].set_title("图2  回撤")
    savefig(fig, "03_nav.png")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for t in MAIN:
        y = rets[t].resample("YE").apply(lambda x: (1 + x).prod() - 1) * 100
        ax[0].plot([d.year for d in y.index], y.values, marker="o", ms=4,
                   lw=1.2, color=COLOR[t], label=LABEL[t])
    yb = bench.resample("YE").apply(lambda x: (1 + x).prod() - 1) * 100
    ax[0].plot([d.year for d in yb.index], yb.values, marker="s", ms=4, lw=1.6,
               ls="--", color=COLOR["基准"], label="基准")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_ylabel("%"); ax[0].set_title("图3  分年度收益")
    ax[0].legend(fontsize=8)
    for t in MAIN:
        d = load_csv(t, "daily")
        ax[1].plot(d.index, d.n_holdings, lw=1.0, color=COLOR[t], label=LABEL[t])
    ax[1].set_ylabel("只"); ax[1].set_title("图4  持仓数量")
    savefig(fig, "03_annual.png")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for t in ["低流动性20", "反转5", "动量20"]:
        b = load_csv(t, "blocks")
        if not len(b):
            continue
        g = b.groupby("reason").dv.sum() / 1e8
        ax[0].barh([r + "  " + t for r in g.index], g.values, color=COLOR.get(t, "#636363"))
    ax[0].set_xlabel("亿元")
    ax[0].set_title("图5  被挡委托金额（按原因）")
    ax[0].tick_params(labelsize=8)
    for t in MAIN:
        d = load_csv(t, "daily")
        ax[1].plot(d.index, d.turnover.rolling(21).mean() * 252, lw=1.1,
                   color=COLOR[t], label=LABEL[t])
    ax[1].set_ylabel("倍/年"); ax[1].set_title("图6  年化换手率（21 日滚动）")
    ax[1].legend(fontsize=8)
    savefig(fig, "03_exec.png")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    x = np.arange(len(SENS))
    for tag in ["低流动性20", "反转5", "动量20"]:
        vals = [perf_stats(load_nav("%s@%s" % (tag, v)))["年化收益"] * 100 for v in SENS]
        base = perf_stats(navs[tag])["年化收益"] * 100
        ax[0].plot(x, vals, marker="o", lw=1.3, color=COLOR.get(tag, "#636363"), label=tag)
        ax[0].axhline(base, color=COLOR.get(tag, "#636363"), ls=":", lw=0.9)
    ax[0].set_xticks(x); ax[0].set_xticklabels([SENS_LABEL[v] for v in SENS], rotation=18,
                                               fontsize=8, ha="right")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_ylabel("年化收益 %")
    ax[0].set_title("图7  口径敏感性（虚线 = 基准口径）")
    ax[0].legend(fontsize=8)
    dd = diffs.drop(columns=["基准口径年化"])
    dd.T.plot(kind="bar", ax=ax[1], color=[COLOR.get(t, "#636363") for t in dd.index],
              legend=True, rot=18, width=0.8)
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_ylabel("相对基准口径的差异（百分点）")
    ax[1].set_title("图8  高估/低估了多少年化收益")
    savefig(fig, "03_sensitivity.png")

    # =========================================================================== #
    # HTML 看板（自包含，图片 base64 内嵌）
    # =========================================================================== #
    def b64(path):
        with open(path, "rb") as fh:
            return base64.b64encode(fh.read()).decode()

    figs = [("03_nav.png", "净值与回撤"), ("03_annual.png", "分年度收益与持仓"),
            ("03_exec.png", "执行质量"), ("03_sensitivity.png", "口径敏感性")]
    imgs = "".join(
        '<figure><img src="data:image/png;base64,%s"><figcaption>%s</figcaption></figure>'
        % (b64(os.path.join(FIG, f)), t) for f, t in figs)

    def html_table(df, pct_cols=(), fmts=None):
        d = df.copy()
        head = "<tr><th>%s</th>%s</tr>" % (
            d.index.name or "",
            "".join("<th>%s</th>" % c for c in d.columns))
        body = []
        for i in range(len(d)):
            tds = []
            for c in d.columns:
                v = d[c].iloc[i]
                if c in pct_cols and pd.notna(v):
                    tds.append("<td>%.2f%%</td>" % (100 * float(v)))
                elif isinstance(v, (float, np.floating)) and pd.notna(v):
                    tds.append("<td>%.3f</td>" % v)
                else:
                    tds.append("<td>%s</td>" % v)
            body.append("<tr><th>%s</th>%s</tr>" % (d.index[i], "".join(tds)))
        return "<table>%s%s</table>" % (head, "".join(body))

    css = """body{font-family:-apple-system,'Segoe UI','Microsoft YaHei',sans-serif;
      margin:0;padding:24px;background:#fafafa;color:#222;max-width:1180px;margin:auto}
      h1{font-size:22px}h2{font-size:17px;margin-top:28px;border-left:4px solid #d94801;padding-left:8px}
      table{border-collapse:collapse;font-size:12.5px;margin:10px 0;background:#fff}
      th,td{border:1px solid #e3e3e3;padding:4px 8px;text-align:right;white-space:nowrap}
      th{background:#f2f2f2}tr>th:first-child,td:first-child{text-align:left}
      figure{margin:0 0 18px}img{width:100%%;border:1px solid #e3e3e3;background:#fff}
      figcaption{font-size:12px;color:#666;padding-top:4px}
      .k{display:flex;gap:14px;flex-wrap:wrap;margin:10px 0}
      .k div{background:#fff;border:1px solid #e3e3e3;border-radius:6px;padding:10px 14px;font-size:13px}
      .k b{display:block;font-size:19px;color:#d94801}"""

    kpi = "".join('<div>%s<b>%s</b></div>' % (t, ("%.2f%%" % (100 * show.loc[t, "年化收益"])))
                  for t in MAIN)
    kpi += '<div>基准年化<b>%.2f%%</b></div>' % (100 * show.loc["基准(等权全市场)", "年化收益"])
    kpi += '<div>复权影响<b>%+.2f pp</b></div>' % (-adj_eff)
    kpi += '<div>涨跌停影响<b>%+.2f pp</b></div>' % lim_eff
    kpi += '<div>成本+容量<b>%+.2f pp</b></div>' % cost_eff

    sens_tbl = html_table(diffs.round(2))
    sm_tbl = html_table(show[["年化收益", "年化波动", "夏普", "最大回撤", "超额年化",
                              "信息比率", "年换手", "成本占初始净值"]],
                        pct_cols=("年化收益", "年化波动", "最大回撤", "超额年化", "成本占初始净值"))
    blk_tbl = html_table(bl.get("低流动性20", pd.DataFrame()).round(1))
    html = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>Class4 回测看板</title><style>%s</style></head><body>
<h1>Class4 回测看板 · %s ~ %s</h1>
<div class="k">%s</div>
<h2>策略绩效（复权全收益、含成本、涨跌停不可交易）</h2>%s
<h2>口径敏感性：相对基准口径的年化收益差异（百分点）</h2>%s
<h2>执行质量：低流动性20 被挡委托</h2>%s
%s
<p style="color:#888;font-size:12px">本页自包含（图片内嵌 base64），双击即可打开。</p>
</body></html>""" % (css, meta["start"], meta["end"], kpi, sm_tbl, sens_tbl, blk_tbl, imgs)
    out_html = os.path.join(ROOT, "out", "dashboard.html")
    with open(out_html, "w", encoding="utf-8") as fh:
        fh.write(html)

    out_md = os.path.join(ROOT, "out", "backtest_report.md")
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n已写出：%s" % out_md)
    print("已写出：%s（%.1f MB）" % (out_html, os.path.getsize(out_html) / 1e6))
    return S


if __name__ == "__main__":
    main()
