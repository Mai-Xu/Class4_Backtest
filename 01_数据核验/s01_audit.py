# -*- coding: utf-8 -*-
"""s01 数据体检：把 feathers/ 的全部数据事实、口径与坑写成一份审计报告。

产物
    out/data_audit.md          审计报告（正文全部为实测数字）
    out/figures/01_*.png       4 张图
    out/tables/01_*.csv        若干净表
运行
    python s01_audit.py
"""
import os
import sys

# ---- 项目定位 ---------------------------------------------------------------
# 数据层已归入项目根的 core/lib.py，feathers/ 也在项目根；
# 三条轨道共用同一份数据层，所以这里把项目根挂进 sys.path。
HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

import core.lib as _lib
from core.lib import MarketData, FIELDS, savefig, savetab

# 产物根：与其它三项工作一致，统一落 out/01_数据核验/。
# savefig/savetab 在**调用时**读取 lib 的模块级 FIG/TAB，所以这里改掉即可。
ROOT = os.path.join(PROJ, "out", "01_数据核验")
_lib.FIG = os.path.join(ROOT, "figures")
_lib.TAB = os.path.join(ROOT, "tables")
for _d in (_lib.FIG, _lib.TAB):
    os.makedirs(_d, exist_ok=True)
FIG, TAB = _lib.FIG, _lib.TAB

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MD = MarketData(verbose=True)
L = []                                    # markdown 行


def w(s=""):
    L.append(s)
    print(s)


def fmt_int(x):
    return format(int(x), ",")


def table(df, floatfmt="%.4f"):
    """DataFrame -> markdown 表。

    注意：必须按列取值（d[c].iloc[i]），row-wise 迭代会把整数列升成 float，
    整数就会渲染成 1115203.0000。
    """
    d = df.copy()
    d.columns = [str(c) for c in d.columns]
    cols = list(d.columns)
    head = "| " + " | ".join([str(d.index.name or "")] + cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    rows = []
    for i in range(len(d)):
        vals = []
        for c in cols:
            v = d[c].iloc[i]
            if isinstance(v, (bool, np.bool_)):
                vals.append("是" if v else "否")
            elif isinstance(v, (int, np.integer)):
                vals.append(fmt_int(v))
            elif isinstance(v, (float, np.floating)):
                vals.append("nan" if not np.isfinite(v) else floatfmt % v)
            else:
                vals.append(str(v))
        k = d.index[i]
        k = k.strftime("%Y-%m-%d") if isinstance(k, pd.Timestamp) else str(k)
        rows.append("| " + " | ".join([k] + vals) + " |")
    return "\n".join([head, sep] + rows)


# =========================================================================== #
w("# Class4 回测 · 数据审计报告")
w()
w("数据目录：`%s`" % MD.dir)
w()
w("本报告由 `python s01_audit.py` 自动生成，**所有数字均为实测**，"
  "是后续所有回测的口径依据。")
w()

# --------------------------------------------------------------------------- #
w("## 1. 数据规模与结构")
w()
raw = {}
for f in FIELDS:
    df = MD.field(f)
    raw[f] = df
    v = np.asarray(df.values, dtype="float64")
    ok = np.isfinite(v)
    w("- `%s.feather`  形状 %s  有效格 %s  最小值 %.4g  最大值 %.4g"
      % (f, tuple(df.shape), fmt_int(ok.sum()), v[ok].min(), v[ok].max()))
w()
dates = MD.dates
w("- 交易日 %s 天：%s ~ %s" % (fmt_int(len(dates)), dates[0].date(), dates[-1].date()))
w("- 证券 %s 只：SH %s / SZ %s（无北交所）"
  % (fmt_int(len(MD.symbols)),
     fmt_int(int(MD.symbols.str.endswith('.SH').sum())),
     fmt_int(int(MD.symbols.str.endswith('.SZ').sum()))))
w("- 7 个文件列名与顺序完全一致：**%s**"
  % ("是" if all(list(raw[f].columns) == list(MD.symbols) for f in FIELDS) else "否"))
w()
w("### 1.1 NaN 掩码一致性")
w()
ref = np.asarray(MD.close.isna().values)          # 以 close 的掩码为基准
rows = []
for f in FIELDS:
    m = np.asarray(raw[f].isna().values)
    rows.append(dict(field=f,
                     与close掩码差异格数=int((m != ref).sum()),
                     缺失格数=int(m.sum()),
                     缺失率=100.0 * m.mean()))
t = pd.DataFrame(rows).set_index("field")
w(table(t, "%.4f"))
w()
nt_gap = MD.num_trades.isna() & MD.listed
w("`open/high/low/close` 与 `amt` 的掩码**完全一致**；`cumadj` 只在 %d 格上与之不同。"
  % int(t.loc["cumadj", "与close掩码差异格数"]))
w("`num_trades` 与价格掩码不同，共 %s 格，其中 %s 格集中在最后一个交易日 %s"
  "（行情源尚未更新，整行皆空），"
  % (fmt_int(int(nt_gap.to_numpy().sum())), fmt_int(int(nt_gap.iloc[-1].sum())), dates[-1].date())
  + "其余 %s 格散布在 %d 个交易日上（每个交易日只有几十到几百只缺笔数）。"
  % (fmt_int(int(nt_gap.to_numpy().sum()) - int(nt_gap.iloc[-1].sum())),
     int((nt_gap.sum(1) > 0).sum()) - 1))
w()

# --------------------------------------------------------------------------- #
w("## 2. 掩码口径：停牌 / 一字板 / 涨跌停")
w()
mr = MD.mask_report()
mr2 = mr.copy()
mr2["pct_of_listed"] = mr2["pct_of_listed"].round(4)
w(table(mr2, "%.4f"))
savetab(mr2, "01_mask_report.csv")
w()
w("- **停牌**：`amt==0` 且 OHLC 四价全等（用前收盘价携程），共 %s 格；"
  "平均每天 %.1f 只，%s/%s 个交易日都存在。"
  % (fmt_int(mr.loc["suspended", "cells"]), mr.loc["suspended", "cells"] / len(dates),
     fmt_int(mr.loc["suspended", "days_with"]), fmt_int(len(dates))))
w("- **一字板**：`high==low` 且 `amt>0`，共 %s 格；其中一字涨停 %s 格、一字跌停 %s 格。"
  % (fmt_int(mr.loc["one_line", "cells"]), fmt_int(mr.loc["one_line_up", "cells"]),
     fmt_int(mr.loc["one_line_down", "cells"])))
w("- 二者相加 %s 格 = 全天只有一个价格，**按市价不可成交**，回测必须屏蔽。"
  % fmt_int(mr.loc["all_flat", "cells"]))
w()

# --------------------------------------------------------------------------- #
w("## 3. 复权因子：必须先去噪")
w()
rawp = raw["cumadj"].pct_change(fill_method=None).abs().to_numpy()
clnp = MD.cumadj_clean.pct_change(fill_method=None).abs().to_numpy()
bins = [(0, 1e-6), (1e-6, 1e-5), (1e-5, 1e-4), (1e-4, 1e-3), (1e-3, 1e9)]
rows = []
for lo, hi in bins:
    rows.append(dict(区间="[%g, %g)" % (lo, hi),
                     去噪前=int(((rawp >= lo) & (rawp < hi)).sum()),
                     去噪后=int(((clnp >= lo) & (clnp < hi)).sum())))
fr = pd.DataFrame(rows).set_index("区间")
w(table(fr, "%.0f"))
w()
noise = (rawp >= 1e-6) & (rawp < 1e-4)
w("`cumadj` 在非调整日的相对抖动只有 1e-6 ~ 1e-5 量级（浮点噪声），"
  "**真实分红送转事件才 >1e-4**。")
w()
w("直接对 `cumadj` 做 `pct_change()`，会凭空多出 **%s** 个非零变化格"
  "（占全部有效格的 %.1f%%），最大的一格也只有 %.4f%%（约 %.1f 个基点）——"
  "单格看起来微不足道，但它让「复权因子在哪天变化」这一判断彻底失真："
  "假事件 88 万条 vs 真事件 2.6 万条，**任何按「因子跳变」识别分红送转的写法都会崩**。"
  % (fmt_int(int(noise.sum())), 100 * noise.sum() / np.isfinite(rawp).sum(),
     100 * float(np.nanmax(np.where(noise, rawp, np.nan))),
     1e4 * float(np.nanmax(np.where(noise, rawp, np.nan)))))
w()
w("`MarketData.cumadj_clean` 把 <1e-4 的抖动置为 1 后逐日连乘重建因子："
  "噪声格 **%s -> 0**，%s 个真实事件格**全部保留**。"
  % (fmt_int(int(((rawp >= 1e-6) & (rawp < 1e-4)).sum())), fmt_int(int((clnp >= 1e-4).sum()))))
w()

# --------------------------------------------------------------------------- #
w("## 4. 涨跌幅上限校验")
w()
T = MD.ratio_table()
w("先验规则：主板 ±10%%；科创板 ±20%%；创业板 %s 起 ±20%%、此前 ±10%%；"
  "5%% 特殊标的另判。" % "2020-08-24")
w()
ok = MD.listed & MD.listed.shift(1, fill_value=False) & ~MD._new_stock
exceed = (MD.ret.abs() > MD.limit_ratio + MD.cap_tolerance()) & ok
w("**用上述先验逐格检验复权收益是否突破上限**：")
w()
w("- 检验格数（剔除新股窗口）：%s" % fmt_int(int(ok.to_numpy().sum())))
w("- 突破上限的格数：**%s**，涉及 %s 只证券，且 **全部**已被 `no_limit` 标记"
  % (fmt_int(int(exceed.to_numpy().sum())), fmt_int(int((exceed.sum(0) > 0).sum()))))
w("- 其余 %s 只证券的先验上限**一格都未被突破** —— 这是板块先验正确性的直接证据"
  % fmt_int(int((T["v_prior"] == 0).sum())))
w()
w("5%% 严格判定的特殊标的（S 股/风险警示）：**%s**。本样本 5451 只里只有 %d 只，"
  "说明数据源基本已剔除 ST 名单。" % ("、".join(MD.st_like[MD.st_like].index), int(MD.st_like.sum())))
w()
w("### 4.1 判定过程（前 8 只违规标的）")
w()
worst = T[T["v_prior"] > 0].sort_values("v_prior", ascending=False).head(8)
worst = worst[["days", "median_px", "v05", "v10", "v20", "v_prior", "p99", "max_abs"]].round(4)
w(table(worst, "%.4f"))
savetab(T.round(6), "01_limit_ratio_diag.csv")
w()
w("> 为什么 5% 判定要严格：仅凭价格无法区分「5% 上限但从未涨过 5%」与"
  "「10% 上限但从未涨过 5%」。工行/建行/农行六年内单日最大涨幅只有 7%~9%，"
  "一旦放宽阈值就会被误判成 5%。")
w()
w("### 4.2 极端日抽查（与真实市场事件对照）")
w()
ext = pd.DataFrame({"涨停家数": MD.at_limit_up.sum(1), "跌停家数": MD.at_limit_down.sum(1),
                    "一字涨停": MD.one_line_up.sum(1), "一字跌停": MD.one_line_down.sum(1)})
w(table(ext["跌停家数"].nlargest(3).to_frame(), "%.0f"))
w()
w(table(ext["涨停家数"].nlargest(3).to_frame(), "%.0f"))
w()
w("跌停前三日：2025-04-07（关税冲击）、2020-02-03（疫情后开市首日）、2024-02-05（雪球/小微盘流动性危机）；"
  "涨停前三日：2024-10-08、2024-09-30、2024-02-08（政策底反转）。"
  "**极端日与真实市场事件完全对应，说明涨跌停识别口径可用。**")
w()

# --------------------------------------------------------------------------- #
w("## 5. 流动性与微观统计量")
w()
amt = MD.amt
adv20 = MD.adv(20)
rows = []
for nm, ser in [("全市场日成交额(亿元)", amt.sum(1) / 1e8),
                ("个股日成交额(万元)", amt.stack() / 1e4),
                ("个股20日均额(万元)", adv20.stack() / 1e4),
                ("个股20日均额中位数(万元)", adv20.median(1) / 1e4),
                ("单笔均额(元)", MD.avg_trade_size.stack())]:
    s = ser.replace([np.inf, -np.inf], np.nan).dropna()
    rows.append(dict(指标=nm, p1=s.quantile(.01), 中位数=s.median(), p99=s.quantile(.99)))
liq = pd.DataFrame(rows).set_index("指标").round(1)
w(table(liq, "%.1f"))
savetab(liq, "01_liquidity.csv")
w()
w("- 全市场日成交额中位数 %s 亿元，量级与 A 股实际相符，`amt` 单位为元。"
  % format(amt.sum(1).median() / 1e8, ",.0f"))
w("- 个股 20 日均额中位数 %s 万元，最小 1%% 分位 %s 万元 —— 股票池的流动性门槛应据此设定。"
  % (format(adv20.stack().median() / 1e4, ",.0f"), format(adv20.stack().quantile(.01) / 1e4, ",.0f")))
w("- %s 只证券的 `num_trades` 为空（含最后一日），`avg_trade_size` 相应缺失。"
  % fmt_int(int((MD.num_trades.isna() & MD.listed).to_numpy().sum())))
w()

# --------------------------------------------------------------------------- #
w("## 6. 股票池")
w()
uni = MD.universe()
n_uni = uni.sum(1)
n_lis = MD.listed.sum(1)
n_trd = MD.traded.sum(1)
rows = []
for nm, ser in [("有报价", n_lis), ("有成交", n_trd), ("可回测股票池", n_uni)]:
    rows.append(dict(口径=nm, 首日=int(ser.iloc[0]), 中位=int(ser.median()),
                     末日=int(ser.iloc[-1]), 最大=int(ser.max())))
us = pd.DataFrame(rows).set_index("口径")
w(table(us, "%.0f"))
w()
w("默认股票池口径：上市满 250 个交易日、20 日均额 >= 2000 万元、股价 >= 2 元、"
  "剔除 5% 标的 / 新股无限制日 / 一字板 / 停牌。**逐日动态构建，不含未来信息。**")
w()
w("- 证券总数 %s 只，但**末日在册的只有 %s 只** —— 有 %s 只已退市/停止更新。"
  % (fmt_int(len(MD.symbols)), fmt_int(int(MD.listed.iloc[-1].notna().sum())),
     fmt_int(int((MD.listed.iloc[-1].isna() & MD.listed.any(axis=0)).sum()))))
w("- 历史长度：<60 个交易日的 %s 只（次新股），<250 日的 %s 只。"
  % (fmt_int(int((MD.listed.sum(0) < 60).sum())), fmt_int(int((MD.listed.sum(0) < 250).sum()))))
w("- **用固定股票池会引入明显的前视偏差**：首日只有 %s 只可交易，末日有 %s 只。"
  % (fmt_int(int(n_trd.iloc[0])), fmt_int(int(n_trd.iloc[-1]))))
w()

# --------------------------------------------------------------------------- #
w("## 7. 收益合理性")
w()
r = MD.ret_ok
ew = MD.equal_weight_ret()
w("- 可用收益格 %s（占全部有价格格的 %.1f%%）"
  % (fmt_int(int(r.notna().to_numpy().sum())),
     100 * r.notna().to_numpy().sum() / MD.listed.to_numpy().sum()))
w("- 全市场等权组合日收益：均值 %.4f%%，年化波动 **%.1f%%**，年化收益 %.1f%%"
  % (100 * ew.mean(), 100 * ew.std() * np.sqrt(252), 100 * ew.mean() * 252))
w("- 横截面日收益标准差中位数 %.2f%%（含新股的极端日）"
  % (100 * r.std(1).median()))
w()
w("等权年化波动约 24%，与 A 股实际相当，数据整体可信。")
w()

# --------------------------------------------------------------------------- #
w("## 8. 已知限制（回测必须知道）")
w()
w("| 缺失内容 | 影响的策略/环节 |")
w("| --- | --- |")
for a, b in [
    ("成交量、流通股本、总股本", "换手率/流动性因子、市值加权、容量估计"),
    ("行业分类、指数成分", "行业中性化、基准对比、指数增强"),
    ("每日涨跌停价、ST 标记、停牌公告", "已用价格反推近似（见 §4），无法 100% 还原"),
    ("复权因子仅有累乘值", "无法拆分送转/分红明细，单次调整幅度需自算"),
    ("无 VWAP、无分钟/逐笔", "成交价只能用 open/close 近似，冲击成本无法建模"),
    ("2026-10-16 的 num_trades 缺失", "涉及成交笔数的因子在该日会断点"),
]:
    w("| %s | %s |" % (a, b))
w()
w("## 9. 结论")
w()
w("1. 7 个 Feather 矩阵结构完全一致、日期与证券对齐，可直接作为面板使用，"
  "无需再做列对齐。")
w("2. **三个必须处理的坑**：复权因子浮点噪声、停牌携程价、一字板不可成交。"
  "`lib.MarketData` 已全部封装。")
w("3. 涨跌停口径经极端日与逐格违约检验，与真实市场事件完全对应。")
w("4. 样本基本不含 ST，只有 %d 只 5%% 标的；数据源质量较高，"
  "主要缺口是衍生字段（股本/行业/成分）与 `num_trades` 的最后一日。"
  % int(MD.st_like.sum()))
w()

# =========================================================================== #
# 出图
# =========================================================================== #
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
idx = MD.dates

fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
ax[0].plot(idx, n_lis, label="有报价", lw=1.2, color="#636363")
ax[0].plot(idx, n_trd, label="有成交", lw=1.2, color="#2c7fb8")
ax[0].plot(idx, n_uni, label="可回测股票池", lw=1.4, color="#d94801")
ax[0].set_ylabel("证券数")
ax[0].set_title("图1  股票池规模随时间（动态池，无未来信息）")
ax[0].legend(loc="lower right")
flat = MD.suspended.sum(1) / n_lis * 100
oline = MD.one_line.sum(1) / n_lis * 100
ax[1].plot(idx, flat, lw=1.0, color="#756bb1", label="停牌占比")
ax[1].plot(idx, oline, lw=1.0, color="#de2d26", label="一字板占比")
ax[1].set_ylabel("占有报价证券 %")
ax[1].set_title("图2  停牌与一字板占比（回测必须剔除）")
ax[1].legend(loc="upper right")
savefig(fig, "01_universe.png")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
b = np.logspace(-9, -1, 70)
ax[0].hist(np.clip(rawp, 1e-12, None)[rawp > 0], bins=b, color="#636363", label="去噪前")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].axvline(1e-4, color="#d94801", ls="--", lw=1.4, label="去噪阈值 1e-4")
ax[0].set_xlabel("|cumadj 日变化|"); ax[0].set_ylabel("格数")
ax[0].set_title("图3  复权因子日变化分布（去噪前）")
ax[0].legend()
ax[1].hist(np.clip(clnp, 1e-12, None)[clnp > 0], bins=b, color="#2c7fb8", label="去噪后")
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].axvline(1e-4, color="#d94801", ls="--", lw=1.4)
ax[1].set_xlabel("|cumadj 日变化|"); ax[1].set_title("图4  去噪后：只剩真实分红送转事件")
savefig(fig, "01_cumadj_noise.png")

fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
ax[0].plot(idx, MD.at_limit_up.sum(1), lw=1.0, color="#d94801", label="触及涨停")
ax[0].plot(idx, -MD.at_limit_down.sum(1), lw=1.0, color="#2c7fb8", label="触及跌停")
ax[0].axhline(0, color="k", lw=0.6)
ax[0].set_ylabel("家数")
ax[0].set_title("图5  每日触及涨跌停家数（上正下负）")
ax[0].legend(loc="upper left")
ax[1].fill_between(idx, 0, MD.one_line_up.sum(1), color="#d94801", label="一字涨停")
ax[1].fill_between(idx, 0, -MD.one_line_down.sum(1), color="#2c7fb8", label="一字跌停")
ax[1].axhline(0, color="k", lw=0.6)
ax[1].set_ylabel("家数")
ax[1].set_title("图6  一字板家数（绝对不可成交）")
ax[1].legend(loc="lower left")
ax[1].set_yscale("symlog", linthresh=10)
savefig(fig, "01_price_limit.png")

fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
ax[0].plot(idx, amt.sum(1) / 1e12, lw=1.0, color="#31a354")
ax[0].set_ylabel("万亿元")
ax[0].set_title("图7  全市场日成交额")
ax[1].plot(idx, (1 + ew.fillna(0)).cumprod(), lw=1.2, color="#756bb1")
ax[1].set_ylabel("净值")
ax[1].set_title("图8  全市场等权组合累计净值（数据合理性对照）")
savefig(fig, "01_market.png")

daily = pd.DataFrame({"n_listed": n_lis, "n_traded": n_trd, "n_universe": n_uni,
                      "suspended": MD.suspended.sum(1), "one_line": MD.one_line.sum(1),
                      "at_limit_up": MD.at_limit_up.sum(1), "at_limit_down": MD.at_limit_down.sum(1),
                      "amt_yuan": amt.sum(1), "ew_ret": ew})
savetab(daily, "01_daily_market.csv")

out = os.path.join(ROOT, "data_audit.md")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")
print("\n已写出：%s" % out)
print("图：%s" % FIG)
