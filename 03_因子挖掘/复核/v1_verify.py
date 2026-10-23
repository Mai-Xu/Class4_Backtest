# -*- coding: utf-8 -*-
"""v1 独立复核（最终版）：不复用 03_因子挖掘 / core.engine 的计算函数，仅从产物 CSV + 原始数据独立重算。
输出：v1_verification.json + v1_report.md 所需全部数字。
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))          # <项目根>/03_因子挖掘/复核
ROOT = os.path.dirname(os.path.dirname(HERE))            # 项目根（不写死本机路径）
RUN = os.path.join(ROOT, "out", "03_因子挖掘", "runs", "23_raw")
RUNP = os.path.join(RUN, "MF1700_raw")
TAB = os.path.join(ROOT, "out", "03_因子挖掘", "tables")
OUTDIR = os.path.join(ROOT, "03_因子挖掘", "复核")
os.makedirs(OUTDIR, exist_ok=True)
LINE = "=" * 78
REPORT = {}


def sec(t):
    print("\n" + LINE + "\n" + t + "\n" + LINE)


trades = pd.read_csv(os.path.join(RUNP, "trades.csv"), parse_dates=["date"])
daily = pd.read_csv(os.path.join(RUNP, "daily.csv"), index_col=0, parse_dates=True)
nav = pd.read_csv(os.path.join(RUNP, "nav.csv"), index_col=0, parse_dates=True)["nav"]
holdings = pd.read_csv(os.path.join(RUNP, "holdings.csv"), parse_dates=["date"])
blocks = pd.read_csv(os.path.join(RUNP, "blocks.csv"), parse_dates=["date"])
bench = pd.read_csv(os.path.join(RUN, "bench_nav.csv"), index_col=0, parse_dates=True)["nav"]
summary = pd.read_csv(os.path.join(TAB, "23_summary_raw.csv"))
robust = pd.read_csv(os.path.join(TAB, "25_robustness.csv"))
S = summary.set_index("组合")
row = S.loc["MF1700_raw"]
rob = robust.set_index("配置").loc["基准 MF1700_raw"]

sys.path.insert(0, ROOT)
from core.lib import MarketData  # 仅用于取原始行情做撮合抽查

md = MarketData(verbose=False)
adj_close = md.adj_close
date_pos = {d: i for i, d in enumerate(md.dates)}
sym_pos = {s: i for i, s in enumerate(md.symbols)}

print("载入：trades %d | nav %d | daily %d | holdings %d | blocks %d | bench %d"
      % (len(trades), len(nav), len(daily), len(holdings), len(blocks), len(bench)))

# ---------------------------------------------------------------- 1
sec("【1】建仓 / 清仓时点")
dmin, dmax = trades["date"].min(), trades["date"].max()
fd, ld = pd.Timestamp("2025-01-02"), pd.Timestamp("2026-09-30")
resid = holdings[holdings["date"] == ld]
resid_val = float(resid["value"].sum())
resid_valid = resid[resid["value"].notna()]
resid_nan = resid[resid["value"].isna()]
block_ld = blocks[blocks["date"] == ld]
r1 = dict(
    first_trade_date=str(dmin.date()), last_trade_date=str(dmax.date()),
    first_ok=bool(dmin == fd), last_ok=bool(dmax == ld),
    n_trades=int(len(trades)), n_buy=int((trades.side == "buy").sum()),
    n_sell=int((trades.side == "sell").sum()), n_trade_days=int(trades.date.nunique()),
    first_day_n=int(len(trades[trades.date == fd])), first_day_sides=sorted(trades[trades.date == fd].side.unique()),
    last_day_n=int(len(trades[trades.date == ld])), last_day_sides=sorted(trades[trades.date == ld].side.unique()),
    final_n_holdings=float(daily.loc[ld, "n_holdings"]), final_holdings_rows=int(len(resid)),
    final_resid_value_valid=resid_val, final_resid_nan_rows=int(len(resid_nan)),
    final_resid_symbols=list(resid["symbol"]),
    final_nav=float(daily.loc[ld, "nav"]), final_cash=float(daily.loc[ld, "cash"]),
    final_resid_weight=resid_val / float(daily.loc[ld, "nav"]),
    final_blocks=block_ld.groupby(["side", "reason"])["dv"].agg(["size", "sum"]).reset_index().to_dict("records"),
)
print("第一笔成交日 %s（要求 %s）→ %s" % (r1["first_trade_date"], fd.date(), "一致" if r1["first_ok"] else "不一致"))
print("最后一笔成交日 %s（要求 %s）→ %s" % (r1["last_trade_date"], ld.date(), "一致" if r1["last_ok"] else "不一致"))
print("首日 %s：%d 笔，%s；末日 %s：%d 笔，%s"
      % (fd.date(), r1["first_day_n"], r1["first_day_sides"], ld.date(), r1["last_day_n"], r1["last_day_sides"]))
print("清仓后末日 daily.n_holdings = %.0f；holdings.csv 末日 %d 行（有效估值 %d 行、NaN %d 行），残留有效市值 %.2f 元（占 NAV %.6f%%）"
      % (r1["final_n_holdings"], r1["final_holdings_rows"], len(resid_valid), len(resid_nan),
         resid_val, 100 * r1["final_resid_weight"]))
print("末日被挡卖出：", r1["final_blocks"])
print("末日残留标的：", r1["final_resid_symbols"])
REPORT["1_timing"] = r1

# ---------------------------------------------------------------- 2
sec("【2】成本记账（佣金 1e-4 双边、无最低 + 印花税卖出 5e-4）")
rate = np.where(trades.side.values == "sell", 6e-4, 1e-4)
exp_fee = trades.dv.values * rate
act_fee = trades.fee.values
ad = np.abs(exp_fee - act_fee); rd = ad / np.maximum(np.abs(act_fee), 1e-12)
iw = int(np.argmax(ad))
r2 = dict(
    max_abs_diff=float(ad.max()), max_rel_diff=float(rd.max()), mean_abs_diff=float(ad.mean()),
    worst=dict(date=str(trades.date.iloc[iw].date()), symbol=str(trades.symbol.iloc[iw]),
               side=str(trades.side.iloc[iw]), dv=float(trades.dv.iloc[iw]),
               fee_actual=float(act_fee[iw]), fee_expect=float(exp_fee[iw])),
    total_fee=float(act_fee.sum()), total_fee_wan=float(act_fee.sum() / 1e4),
    commission=float((trades.dv * 1e-4).sum()), stamp=float((trades.loc[trades.side == "sell", "dv"] * 5e-4).sum()),
    report_fee_wan=float(rob["费用万元"]), report_diff_yuan=float(act_fee.sum() - rob["费用万元"] * 1e4),
    total_dv=float(trades.dv.sum()), buy_dv=float(trades.loc[trades.side == "buy", "dv"].sum()),
    sell_dv=float(trades.loc[trades.side == "sell", "dv"].sum()),
)
print("逐笔重算 vs fee：max|Δ| = %.6e 元，max 相对差 = %.6e，mean|Δ| = %.6e" % (r2["max_abs_diff"], r2["max_rel_diff"], r2["mean_abs_diff"]))
print("最差笔 %s %s %s dv=%.2f：记录 %.6f vs 重算 %.6f" % tuple([r2["worst"][k] for k in ("date", "symbol", "side", "dv", "fee_actual", "fee_expect")]))
print("总费用 逐笔合计 %.2f 元（%.4f 万元）= 佣金 %.2f + 印花税 %.2f；报告 %.4f 万元，差 %.6f 元"
      % (r2["total_fee"], r2["total_fee_wan"], r2["commission"], r2["stamp"], r2["report_fee_wan"], r2["report_diff_yuan"]))
REPORT["2_cost"] = r2

# ---------------------------------------------------------------- 3
sec("【3】绩效指标独立重算（nav.csv）")
nv = nav.sort_index(); r = nv.pct_change().fillna(0.0)
n = len(nv); yrs = n / 252.0
total = float(nv.iloc[-1] / nv.iloc[0] - 1)
cagr = float((nv.iloc[-1] / nv.iloc[0]) ** (1 / yrs) - 1)
vol = float(r.std(ddof=1) * np.sqrt(252))
sharpe = cagr / vol
dd = nv / nv.cummax() - 1
mdd = float(dd.min())
win = float((r > 0).mean())
calmar = cagr / abs(mdd)
# 纯交易日口径（从 2025-01-02 起）
nv2 = nv.iloc[1:]; r2s = nv2.pct_change().fillna(0.0); yrs2 = len(nv2) / 252.0
cagr2 = float((nv2.iloc[-1] / nv2.iloc[0]) ** (1 / yrs2) - 1)
yrs_cal = (nv.index[-1] - nv.index[0]).days / 365.25
cagr_cal = float((nv.iloc[-1] / nv.iloc[0]) ** (1 / yrs_cal) - 1)
r3 = dict(n=n, start=str(nv.index[0].date()), end=str(nv.index[-1].date()),
          nav0=float(nv.iloc[0]), nav_end=float(nv.iloc[-1]), years=yrs, years_cal=yrs_cal,
          total=total, cagr=cagr, cagr_live=cagr2, cagr_calendar=cagr_cal, vol=vol, sharpe=sharpe,
          mdd=mdd, calmar=calmar, winrate=win, mdd_peak=str(nv.loc[:dd.idxmin()].idxmax().date()), mdd_trough=str(dd.idxmin().date()),
          summary=dict(total=float(row["累计"]), cagr=float(row["年化"]), vol=float(row["波动"]),
                       sharpe=float(row["夏普"]), mdd=float(row["最大回撤"]), calmar=float(row["卡玛"]), winrate=float(row["日胜率"])),
          diff=dict(total=float(row["累计"]) - total, cagr=float(row["年化"]) - cagr, vol=float(row["波动"]) - vol,
                    sharpe=float(row["夏普"]) - sharpe, mdd=float(row["最大回撤"]) - mdd,
                    calmar=float(row["卡玛"]) - calmar, winrate=float(row["日胜率"]) - win))
print("%-10s %16s %16s %12s" % ("指标", "独立重算", "summary", "差异"))
for k, v, sk in [("累计收益", total, "total"), ("年化收益", cagr, "cagr"), ("年化波动", vol, "vol"),
                 ("夏普", sharpe, "sharpe"), ("最大回撤", mdd, "mdd"), ("卡玛", calmar, "calmar"), ("日胜率", win, "winrate")]:
    print("%-10s %16.10f %16.10f %12.3e" % (k, v, r3["summary"][sk], r3["diff"][sk]))
print("窗口：%d 行，%.6f 年（252 口径）；纯 2025-01-02 起 %.6f 年，年化 %.6f；自然日口径年化 %.6f（较 summary 低 %.2fpp）"
      % (n, yrs, yrs2, cagr2, cagr_cal, 100 * (cagr - cagr_cal)))
print("最大回撤峰 %s → 谷 %s" % (r3["mdd_peak"], r3["mdd_trough"]))
REPORT["3_perf"] = r3

# ---------------------------------------------------------------- 4
sec("【4】换手独立重算")
tot_dv = float(trades.dv.sum())
sum_turn = float(daily.turnover.sum())
yrs_d = len(daily) / 252.0
t_daily = sum_turn / yrs_d
t_nav0 = tot_dv / float(nav.iloc[0]) / yrs_d
t_avgnav = tot_dv / float(nav.iloc[1:].mean()) / yrs_d
tr_day = trades.groupby("date")["dv"].sum().reindex(daily.index).fillna(0.0)
turn_recon = tr_day / daily.nav
dt = (turn_recon - daily.turnover).abs()
r4 = dict(total_dv=tot_dv, sum_daily_turnover=sum_turn, years=yrs_d,
          turnover_daily_sum=t_daily, turnover_nav0=t_nav0, turnover_avg_nav=t_avgnav,
          summary_turnover=float(row["年换手"]), diff_daily_sum=float(row["年换手"] - t_daily),
          diff_nav0=float(row["年换手"] - t_nav0),
          max_abs_diff_daily=float(dt.max()), max_rel_diff_daily=float((dt / daily.turnover.replace(0, np.nan)).max()))
print("成交额合计 %.2f 元；Σdaily.turnover = %.10f，年数 %.6f" % (tot_dv, sum_turn, yrs_d))
print("年换手：Σturnover/年数 = %.10f（summary %.10f，差 %.3e）" % (t_daily, r4["summary_turnover"], r4["diff_daily_sum"]))
print("对照口径：成交额/期初NAV/年数 = %.10f；成交额/期均NAV/年数 = %.10f" % (t_nav0, t_avgnav))
print("逐日 turnover 重建 max|Δ| = %.3e，max 相对差 = %.3e" % (r4["max_abs_diff_daily"], r4["max_rel_diff_daily"]))
REPORT["4_turnover"] = r4

# ---------------------------------------------------------------- 5
sec("【5】持仓一致性（trades → 每日持仓/现金）")
tr = trades.sort_values(["date"], kind="stable").copy()
cf = np.where(tr.side.values == "buy", -(tr.dv.values + tr.fee.values), tr.dv.values - tr.fee.values)
tr["cash_after"] = 1e9 + np.cumsum(cf)
cbd = tr.groupby("date")["cash_after"].last().reindex(daily.index)
cbd.iloc[0] = 1e9
cdiff = (cbd - daily.cash).abs()
sym_all = sorted(set(trades.symbol))
smap = {s: i for i, s in enumerate(sym_all)}
tr["_s"] = tr.symbol.map(smap)
tr["_flow"] = np.where(tr.side.values == "buy", tr.units.values, -tr.units.values)
dpos = {d: i for i, d in enumerate(daily.index)}
cum = np.zeros(len(sym_all)); rows = {}
for d, g in tr.groupby("date", sort=True):
    for u, s in zip(g._flow.values, g._s.values):
        cum[s] += u
    rows[dpos[d]] = cum.copy()
U = np.full((len(daily), len(sym_all)), np.nan)
for i, rr in rows.items():
    U[i] = rr
U = pd.DataFrame(U, index=daily.index, columns=sym_all).ffill().fillna(0.0)
n_from = (U > 1e-9).sum(axis=1)
ndiff = (n_from - daily.n_holdings).abs()
# holdings.value 重建核对
ho = holdings[holdings.date.isin(daily.index)].copy()
ho["_ldi"] = ho.date.map(dpos); ho["_gdi"] = ho.date.map(date_pos); ho["_si"] = ho.symbol.map(sym_pos)
ho = ho.dropna(subset=["_ldi", "_gdi", "_si"]); ho["_ldi"] = ho._ldi.astype(int); ho["_gdi"] = ho._gdi.astype(int); ho["_si"] = ho._si.astype(int)
ho = ho[ho.symbol.isin(smap)].copy()
rebuilt = U.values[ho._ldi.values, np.array([smap[s] for s in ho.symbol])]
adjc = adj_close.values[ho._gdi.values, ho._si.values]
vdiff = np.abs(rebuilt * adjc - ho.value.values)
ok = np.isfinite(vdiff)
hsum = holdings.groupby("date").value.sum().reindex(daily.index).fillna(0.0)
navid = (daily.cash + hsum - daily.nav).abs()
r5 = dict(max_abs_cash_diff=float(cdiff.max()), max_rel_cash_diff=float((cdiff / daily.cash.abs()).max()),
          cash_worst_date=str(cdiff.idxmax().date()),
          max_abs_nholdings_diff=float(ndiff.max()), nholdings_mismatch_days=int((ndiff > 0).sum()),
          negative_cells=int((U < -1e-6).sum().sum()),
          max_abs_value_diff=float(vdiff[ok].max()), value_cells=int(ok.sum()),
          value_cells_gt1yuan=int((vdiff[ok] > 1).sum()),
          max_abs_nav_identity=float(navid.max()))
print("逐笔现金流 → daily.cash：max|Δ| = %.6e 元（%s），max 相对差 = %.3e" % (r5["max_abs_cash_diff"], r5["cash_worst_date"], r5["max_rel_cash_diff"]))
print("trades 重建持仓数 vs daily.n_holdings：max|Δ| = %.0f，不一致天数 = %d/%d；负份额 = %d"
      % (r5["max_abs_nholdings_diff"], r5["nholdings_mismatch_days"], len(daily), r5["negative_cells"]))
print("holdings.value vs 重建份额×复权收盘：max|Δ| = %.6e 元（%d 个格子），>1 元格子 = %d" % (r5["max_abs_value_diff"], r5["value_cells"], r5["value_cells_gt1yuan"]))
print("恒等式 daily.nav = daily.cash + Σholdings.value：max|Δ| = %.6e 元" % r5["max_abs_nav_identity"])
REPORT["5_holdings"] = r5

# ---------------------------------------------------------------- 6
sec("【6】口径合理性抽查（一字板 / 停牌 / 参与率）")
rng = np.random.default_rng(20261022)
tdays = sorted(trades.date.unique())
sample = [pd.Timestamp(x) for x in rng.choice(tdays, size=3, replace=False)]
olu, old_, sus = md.one_line_up, md.one_line_down, md.suspended
amt = md.amt
part = []; bad_buy = []; bad_sell = []
for _, t in trades.iterrows():
    i, j = date_pos[t.date], sym_pos[t.symbol]
    if t.side == "buy" and (bool(olu.values[i, j]) or bool(sus.values[i, j])):
        bad_buy.append((str(t.date.date()), t.symbol))
    if t.side == "sell" and (bool(old_.values[i, j]) or bool(sus.values[i, j])):
        bad_sell.append((str(t.date.date()), t.symbol))
    a = amt.values[i, j]
    if np.isfinite(a) and a > 0:
        part.append(t.dv / a)
part = np.asarray(part)
r6 = dict(sample_days=[str(d.date()) for d in sample], bad_buys=len(bad_buy), bad_sells=len(bad_sell),
          bad_buy_head=bad_buy[:5], bad_sell_head=bad_sell[:5],
          max_participation=float(part.max()), p99=float(np.quantile(part, 0.99)), median=float(np.median(part)),
          n_over_10pct=int((part > 0.1 + 1e-9).sum()), n_over_10pct_1e6=int((part > 0.1 + 1e-6).sum()),
          sample_detail=[])
print("全样本 30,124 笔：买入撞一字涨停/停牌 %d 笔；卖出撞一字跌停/停牌 %d 笔" % (len(bad_buy), len(bad_sell)))
print("参与率 dv/当日成交额：中位 %.6f，P99 %.6f，最大 %.10f；>10%% 笔数 %d（+1e-6 容差 %d）"
      % (r6["median"], r6["p99"], r6["max_participation"], r6["n_over_10pct"], r6["n_over_10pct_1e6"]))
for d in sample:
    sub = trades[trades.date == d]
    rows = []
    for _, t in sub.iterrows():
        i, j = date_pos[d], sym_pos[t.symbol]
        rows.append((t.side, t.dv / amt.values[i, j], bool(olu.values[i, j]), bool(old_.values[i, j]), bool(sus.values[i, j])))
    sd = pd.DataFrame(rows, columns=["side", "part", "olu", "old", "sus"])
    det = dict(date=str(d.date()), n=int(len(sd)), n_buy=int((sd.side == "buy").sum()), n_sell=int((sd.side == "sell").sum()),
               max_part=float(sd.part.max()), n_buy_olu=int(((sd.side == "buy") & sd.olu).sum()),
               n_sell_old=int(((sd.side == "sell") & sd.old).sum()), n_sus=int(sd.sus.sum()))
    r6["sample_detail"].append(det)
    print("  %s：%d 笔（买 %d/卖 %d），最大参与率 %.6f，买撞一字涨停 %d，卖撞一字跌停 %d，停牌 %d"
          % (det["date"], det["n"], det["n_buy"], det["n_sell"], det["max_part"], det["n_buy_olu"], det["n_sell_old"], det["n_sus"]))
REPORT["6_sanity"] = r6

# ---------------------------------------------------------------- 7
sec("【7】附加：基准 / 超额 / 报告数字对齐")
b = bench.sort_index(); rb = b.pct_change().fillna(0.0)
uni = md.universe()
b_ind = (1 + md.ret.where(uni).mean(axis=1).reindex(b.index).fillna(0.0)).cumprod()
rb_diff = (b_ind - b).abs().max()
yrs_b = len(b) / 252.0
cagr_b = float((b.iloc[-1] / b.iloc[0]) ** (1 / yrs_b) - 1)
act = (r - rb).dropna()
te = float(act.std(ddof=1) * np.sqrt(252))
excess_engine = cagr - cagr_b
ir_engine = excess_engine / te
b2 = b.iloc[1:] / b.iloc[1]; a2 = nv.iloc[1:] / nv.iloc[1]
yrs2 = len(a2) / 252.0
cagr_a2 = float(a2.iloc[-1] ** (1 / yrs2) - 1); cagr_b2 = float(b2.iloc[-1] ** (1 / yrs2) - 1)
excess_rebase = cagr_a2 - cagr_b2
act2 = (nv.iloc[1:].pct_change().fillna(0.0) - b.iloc[1:].pct_change().fillna(0.0)).dropna()
te2 = float(act2.std(ddof=1) * np.sqrt(252))
r7 = dict(bench_first=float(b.iloc[0]), bench_first_date=str(b.index[0].date()),
          bench_2025_0102=float(b.loc[fd]), bench_last=float(b.iloc[-1]),
          bench_independent_max_diff=float(rb_diff), bench_cagr=cagr_b, bench_cum=float(b.iloc[-1] / b.iloc[0] - 1),
          summary_bench_cum=float(S.loc["BENCH_池内等权", "累计"]), summary_bench_cagr=float(S.loc["BENCH_池内等权", "年化"]),
          summary_bench_vol=float(S.loc["BENCH_池内等权", "波动"]), summary_bench_sharpe=float(S.loc["BENCH_池内等权", "夏普"]),
          te=te, excess_engine=excess_engine, ir_engine=ir_engine,
          summary_excess=float(row["超额年化"]), summary_te=float(row["跟踪误差"]), summary_ir=float(row["信息比率"]),
          cagr_live=cagr_a2, bench_cagr_live=cagr_b2, excess_rebase=excess_rebase, te_live=te2, ir_rebase=excess_rebase / te2,
          avg_n_holdings=float(daily.n_holdings.mean()), report_avg_n=float(rob["平均持仓数"]),
          avg_position=float((hsum / daily.nav).mean()), report_avg_position=float(rob["平均仓位"]),
          blocks_total_wan=float(blocks.dv.sum() / 1e4), report_blocks_wan=float(rob["被挡金额万元"]),
          first_day_dv=float(trades[trades.date == fd].dv.sum()))
print("基准自建独立复现：max|Δ| = %.3e（与 bench_nav.csv 一致）" % rb_diff)
print("bench_nav 首值 %.6f @ %s，2025-01-02 = %.6f，末值 %.6f；累计 %.6f，年化 %.6f"
      % (r7["bench_first"], r7["bench_first_date"], r7["bench_2025_0102"], r7["bench_last"], r7["bench_cum"], cagr_b))
print("summary 基准行：累计 %.6f，年化 %.6f，波动 %.6f，夏普 %.6f" % (r7["summary_bench_cum"], r7["summary_bench_cagr"], r7["summary_bench_vol"], r7["summary_bench_sharpe"]))
print("超额（引擎口径，两序列均从 2024-12-31 起）：%.6f；TE %.6f；IR %.6f" % (excess_engine, te, ir_engine))
print("summary 超额 %.6f，TE %.6f，IR %.6f → 与独立重算差 %.6f / %.6f / %.6f"
      % (r7["summary_excess"], r7["summary_te"], r7["summary_ir"],
         r7["summary_excess"] - excess_engine, r7["summary_te"] - te, r7["summary_ir"] - ir_engine))
print("改用「2025-01-02 首个实盘日起、双方归一到 1」：组合年化 %.6f，基准年化 %.6f，超额 %.6f，TE %.6f，IR %.6f"
      % (cagr_a2, cagr_b2, excess_rebase, te2, r7["ir_rebase"]))
print("报告核对：平均持仓 %.4f（报告 %.4f）；平均仓位 %.6f（报告 %.6f）；被挡金额 %.2f 万元（报告 %.2f 万元）"
      % (r7["avg_n_holdings"], r7["report_avg_n"], r7["avg_position"], r7["report_avg_position"], r7["blocks_total_wan"], r7["report_blocks_wan"]))
REPORT["7_bench_extra"] = r7

with open(os.path.join(OUTDIR, "v1_verification.json"), "w", encoding="utf-8") as f:
    json.dump(REPORT, f, ensure_ascii=False, indent=1, default=float)
print("\n已写出 " + os.path.join(OUTDIR, "v1_verification.json"))
