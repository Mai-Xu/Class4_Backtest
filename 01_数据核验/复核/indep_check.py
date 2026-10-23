# -*- coding: utf-8 -*-
"""A 轨独立核验（audit/indep_check.py）

只读 feathers/ 原始 feather 文件 + 公开行情接口；**不复用** s01_audit.py /
s02_external_check.py / core.lib 的任何函数（去噪、重建、收益全部独立实现）。

核验内容
  A. OHLC 排序约束：low <= min(open,close) <= max(open,close) <= high（全样本逐格）
  B. cumadj 单调性 + 浮点噪声计数（880,546 假事件 / 26,295 真事件）+ 独立去噪重建
  C. adj_close 日收益尾部：99.99 分位、最大值、Top5 极端值归因
  D. 报告关键数字独立复算：规模 / 掩码 / 停牌 / 一字板 / 涨跌停极端日 / 股票池量级
  E. 第二外部源（东方财富）：4 只股票 OHLC 逐格 + 成交额逐格 + 前复权收益

运行：python 1_数据真实性验证/audit/indep_check.py
"""
from __future__ import annotations

import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))     # .../1_数据真实性验证/audit
ATRACK = os.path.dirname(HERE)                         # .../1_数据真实性验证
ROOT = os.path.dirname(ATRACK)                         # 项目根


def _resolve_data() -> str:
    """数据目录：与主代码同一优先级（环境变量 → 项目根 feathers/ → 当前目录）。

    本脚本刻意不 import core.lib（独立核验），但数据定位规则保持一致，
    这样换机器 / 数据放别处时同样能跑。
    """
    for var in ("CLASS4_DATA_DIR", "FEATHERS_DIR"):
        p = os.environ.get(var)
        if p and os.path.isfile(os.path.join(p, "close.feather")):
            return os.path.abspath(os.path.expanduser(p))
    for cand in (os.path.join(ROOT, "feathers"), os.getcwd()):
        if os.path.isfile(os.path.join(cand, "close.feather")):
            return os.path.abspath(cand)
    return os.path.join(ROOT, "feathers")


DATA = _resolve_data()
OUTJSON = os.path.join(HERE, "indep_check_results.json")
OUTCSV = os.path.join(HERE, "indep_em_check.csv")
OUTCSVEV = os.path.join(HERE, "indep_em_events.csv")

EPS = 1e-4        # cumadj 去噪阈值（数值与原脚本相同，但实现完全独立）
FIELD_ORDER = ("open", "high", "low", "close", "amt", "num_trades", "cumadj")


def log(*a):
    print(*a)


def load_fields(names):
    out = {}
    for f in names:
        df = pd.read_feather(os.path.join(DATA, f + ".feather"))
        if "date" in df.columns:
            df = df.set_index("date")
        out[f] = df
    return out


def prev_row(arr):
    p = np.full_like(arr, np.nan)
    p[1:] = arr[:-1]
    return p


def board_ratio(sym, d):
    """板块先验：主板 10%；创业板 2020-08-24 起 20%；科创板 20%。"""
    if sym.startswith("688") or sym.startswith("689"):
        return 0.20
    if sym.startswith("300") or sym.startswith("301"):
        return 0.20 if d >= pd.Timestamp("2020-08-24") else 0.10
    return 0.10
# =========================================================================== #
def part_a_ohlc(o, h, l, c, dates, syms):
    log("\n" + "=" * 100)
    log("PART A  OHLC 排序约束（全样本逐格）")
    log("=" * 100)
    R = {}
    valid = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
    lo_min = np.minimum(o, c)
    hi_max = np.maximum(o, c)
    bad_low = valid & (l > lo_min)                 # low 高于 min(o,c)
    bad_high = valid & (hi_max > h)                # high 低于 max(o,c)
    bad_any = bad_low | bad_high
    R.update(valid_cells=int(valid.sum()),
             viol_low_gt_minoc=int(bad_low.sum()),
             viol_maxoc_gt_high=int(bad_high.sum()),
             viol_any=int(bad_any.sum()),
             viol_low_gt_minoc_tol1e9=int((valid & (l > lo_min + 1e-9)).sum()),
             viol_maxoc_gt_high_tol1e9=int((valid & (hi_max > h + 1e-9)).sum()),
             also_l_gt_h=int((valid & (l > h)).sum()))
    # 最严重的越界幅度
    over_low = np.where(bad_low, l - lo_min, 0.0)
    over_high = np.where(bad_high, hi_max - h, 0.0)
    R["max_overshoot_low_gt_minoc"] = float(over_low.max())
    R["max_overshoot_maxoc_gt_high"] = float(over_high.max())
    log("valid cells (o/h/l/c all finite): %s" % format(R["valid_cells"], ","))
    log("violations  low > min(open,close)      : %s" % format(R["viol_low_gt_minoc"], ","))
    log("violations  max(open,close) > high     : %s" % format(R["viol_maxoc_gt_high"], ","))
    log("violations  any of the two             : %s" % format(R["viol_any"], ","))
    log("max overshoot (low - min(o,c))         : %.10g" % R["max_overshoot_low_gt_minoc"])
    log("max overshoot (max(o,c) - high)        : %.10g" % R["max_overshoot_maxoc_gt_high"])
    samples = []
    if bad_any.any():
        ii, jj = np.where(bad_any)
        for k in range(min(5, len(ii))):
            i, j = int(ii[k]), int(jj[k])
            samples.append(dict(date=str(dates[i].date()), symbol=str(syms[j]),
                                open=float(o[i, j]), high=float(h[i, j]),
                                low=float(l[i, j]), close=float(c[i, j])))
    R["samples"] = samples
    for s in samples:
        log("sample: %s" % s)
    return R


# =========================================================================== #
def part_b_cumadj(cum, close, o, h, l, a, dates, syms):
    log("\n" + "=" * 100)
    log("PART B  cumadj 单调性 / 浮点噪声 vs 真实事件 / 独立去噪重建")
    log("=" * 100)
    R = {}
    n = cum.shape[0]
    prev = prev_row(cum)
    both = np.isfinite(cum) & np.isfinite(prev)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(both, cum / prev, np.nan)
    absd = np.abs(rel - 1.0)

    dec = both & (rel < 1.0)
    dec_tol = both & (rel < 1.0 - 1e-12)
    R.update(pairs=int(both.sum()),
             raw_decreases_exact=int(dec.sum()),
             raw_decreases_gt1e12=int(dec_tol.sum()),
             raw_min_rel=float(np.nanmin(rel)) if both.any() else None)
    log("valid consecutive pairs            : %s" % format(R["pairs"], ","))
    log("raw cumadj decreases (rel < 1)     : %s" % format(R["raw_decreases_exact"], ","))
    log("raw cumadj decreases (<1-1e-12)    : %s" % format(R["raw_decreases_gt1e12"], ","))
    log("min rel (raw)                      : %.12g" % R["raw_min_rel"])
    # 减幅分布（把「下降」按幅度分档）
    bands = [(0, 1e-12), (1e-12, 1e-8), (1e-8, 1e-6), (1e-6, 1e-4), (1e-4, 1e9)]
    R["decrease_bands"] = {("[%g,%g)" % b): int((dec & (1.0 - rel >= b[0]) & (1.0 - rel < b[1])).sum()) for b in bands}
    log("decrease magnitude bands           : %s" % R["decrease_bands"])

    # 噪声 / 真事件计数（与报告口径对齐：|rel-1| 分档）
    eb = [(0, 1e-6), (1e-6, 1e-5), (1e-5, 1e-4), (1e-4, 1e-3), (1e-3, 1e9)]
    R["event_bands"] = {("[%g,%g)" % b): int((both & (absd >= b[0]) & (absd < b[1])).sum()) for b in eb}
    R["noise_1e6_1e4"] = int((both & (absd >= 1e-6) & (absd < 1e-4)).sum())
    R["noise_gt0_lt1e4"] = int((both & (absd > 0) & (absd < 1e-4)).sum())
    R["real_ge1e4"] = int((both & (absd >= 1e-4)).sum())
    R["real_ge1e4_up"] = int((both & (absd >= 1e-4) & (rel > 1)).sum())
    R["real_ge1e4_down"] = int((both & (absd >= 1e-4) & (rel < 1)).sum())
    R["noise_max_abs"] = float(np.nanmax(np.where(both & (absd > 0) & (absd < 1e-4), absd, np.nan)))
    log("bands |rel-1|                      : %s" % R["event_bands"])
    log("noise cells 1e-6 <= |rel-1| < 1e-4 : %s  (= claimed 880,546)" % format(R["noise_1e6_1e4"], ","))
    log("noise cells 0 < |rel-1| < 1e-4     : %s" % format(R["noise_gt0_lt1e4"], ","))
    log("real events |rel-1| >= 1e-4        : %s  (= claimed 26,295; up %s / down %s)"
        % (format(R["real_ge1e4"], ","), format(R["real_ge1e4_up"], ","), format(R["real_ge1e4_down"], ",")))

    # ---- 独立去噪重建（我自己写：<1e-4 的抖动置 1，逐列连乘）----
    nrow, ncol = cum.shape
    clean = np.full((nrow, ncol), np.nan)
    for j in range(ncol):
        col = cum[:, j]
        ok = np.flatnonzero(np.isfinite(col))
        if ok.size == 0:
            continue
        v = col[ok]
        r = np.ones_like(v)
        r[1:] = v[1:] / v[:-1]
        step = np.where(np.abs(r - 1.0) >= EPS, r, 1.0)
        clean[ok, j] = v[0] * np.cumprod(step)
    np.save(os.path.join(HERE, "_clean_factor.npy"), clean)     # 供 C 段复用（我自己算的）

    prev_c = prev_row(clean)
    both_c = np.isfinite(clean) & np.isfinite(prev_c)
    with np.errstate(invalid="ignore", divide="ignore"):
        relc = np.where(both_c, clean / prev_c, np.nan)
    absc = np.abs(relc - 1.0)
    R.update(clean_decreases_gt1e12=int((both_c & (relc < 1.0 - 1e-12)).sum()),
             clean_min_rel=float(np.nanmin(relc)),
             clean_noise_gt0_lt1e4=int((both_c & (absc > 0) & (absc < 1e-4)).sum()),
             clean_real_ge1e4=int((both_c & (absc >= 1e-4)).sum()))
    log("after my own denoise: decreases(>1e-12) : %s" % format(R["clean_decreases_gt1e12"], ","))
    log("after my own denoise: min rel           : %.12g" % R["clean_min_rel"])
    log("after my own denoise: 0<|rel-1|<1e-4   : %s" % format(R["clean_noise_gt0_lt1e4"], ","))
    log("after my own denoise: real events       : %s" % format(R["clean_real_ge1e4"], ","))

    # 去噪前后事件格是否一一保留：真事件格在 clean 里是否仍是事件
    keep = int((both & (absd >= 1e-4) & (absc >= 1e-4)).sum())
    R["real_events_preserved"] = keep
    log("real event cells preserved by rebuild    : %s / %s" % (format(keep, ","), format(R["real_ge1e4"], ",")))
    # ---- 真实事件格里的「因子下降」明细 ----
    ii, jj = np.where(both & (absd >= 1e-4) & (rel < 1.0))
    dec_detail = []
    for i, j in zip(ii, jj):
        k = np.flatnonzero(np.isfinite(close[:i, j]))
        ppos = int(k[-1]) if k.size else None
        dec_detail.append(dict(date=str(dates[i].date()), symbol=str(syms[j]),
                               factor_rel=float(rel[i, j]),
                               prev_close=float(close[ppos, j]) if ppos is not None else None,
                               close=float(close[i, j]),
                               prev_valid_date=str(dates[ppos].date()) if ppos is not None else None))
    R["real_decreases"] = dec_detail
    log("real (>1e-4) factor DECREASES: %d" % len(dec_detail))
    for d in dec_detail:
        log("  %s %s factor_rel=%.6f prev_close=%s close=%s prev_valid=%s"
            % (d["date"], d["symbol"], d["factor_rel"], d["prev_close"], d["close"], d["prev_valid_date"]))
    return R, clean
# =========================================================================== #
def part_c_returns(clean, close, o, h, l, a, cum, dates, syms):
    log("\n" + "=" * 100)
    log("PART C  adj_close 日收益尾部（独立实现 adj = close x 我算的去噪因子）")
    log("=" * 100)
    R = {}
    nrow, ncol = close.shape
    adj = close * clean
    preva = prev_row(adj)
    both = np.isfinite(adj) & np.isfinite(preva)
    with np.errstate(invalid="ignore", divide="ignore"):
        ret = np.where(both, adj / preva - 1.0, np.nan)
    flat = ret[np.isfinite(ret)]
    q = np.percentile(flat, [0.01, 0.1, 99.9, 99.99])
    aq = np.percentile(np.abs(flat), [99.9, 99.99, 99.999])
    R.update(n_returns=int(flat.size),
             p0001=float(q[0]), p001=float(q[1]), p999=float(q[2]), p9999=float(q[3]),
             abs_p999=float(aq[0]), abs_p9999=float(aq[1]), abs_p99999=float(aq[2]),
             max_abs=float(np.abs(flat).max()), max_ret=float(flat.max()), min_ret=float(flat.min()))
    for thr in (0.11, 0.21, 0.50, 1.0, 5.0):
        R["n_abs_gt_%g" % thr] = int((np.abs(flat) > thr).sum())
    log("valid daily adj-returns            : %s" % format(R["n_returns"], ","))
    log("quantiles 0.01%% / 0.1%% / 99.9%% / 99.99%%: %.6f%%  %.6f%%  %.6f%%  %.6f%%"
        % (100 * q[0], 100 * q[1], 100 * q[2], 100 * q[3]))
    log("|ret| quantiles 99.9%% / 99.99%% / 99.999%% : %.6f%%  %.6f%%  %.6f%%"
        % (100 * aq[0], 100 * aq[1], 100 * aq[2]))
    log("max |ret| = %.4f%%   max ret = %.4f%%   min ret = %.4f%%"
        % (100 * R["max_abs"], 100 * R["max_ret"], 100 * R["min_ret"]))
    log("count |ret|>11%%: %s ; >21%%: %s ; >50%%: %s ; >100%%: %s"
        % (format(R["n_abs_gt_0.11"], ","), format(R["n_abs_gt_0.21"], ","),
           format(R["n_abs_gt_0.5"], ","), format(R["n_abs_gt_1"], ",")))

    first_pos = np.full(ncol, -1, dtype=int)
    fin = np.isfinite(close)
    for j in range(ncol):
        w = np.flatnonzero(fin[:, j])
        if w.size:
            first_pos[j] = int(w[0])

    # Top 5 by |ret|
    ii, jj = np.where(np.isfinite(ret))
    vals = np.abs(ret[ii, jj])
    k = min(5, vals.size)
    sel = np.argpartition(-vals, k - 1)[:k]
    sel = sel[np.argsort(-vals[sel])]
    known = {pd.Timestamp(x) for x in ["2020-02-03", "2024-02-05", "2024-02-08",
                                       "2024-09-30", "2024-10-08", "2025-04-07"]}
    tops = []
    log("Top 5 |adj daily return|:")
    for s in sel:
        i, j = int(ii[s]), int(jj[s])
        d = dates[i]
        sym = str(syms[j])
        r = float(ret[i, j])
        rr = float(close[i, j] / close[i - 1, j] - 1.0) if np.isfinite(close[i - 1, j]) else np.nan
        fr = float(clean[i, j] / clean[i - 1, j]) if np.isfinite(clean[i - 1, j]) else np.nan
        ratio = board_ratio(sym, d)
        info = dict(date=str(d.date()), symbol=sym, ret_adj=r, ret_raw=rr, factor_rel=fr,
                    open=float(o[i, j]), high=float(h[i, j]), low=float(l[i, j]), close=float(c_prev := close[i - 1, j]),
                    close_t=float(close[i, j]), amt=float(a[i, j]),
                    one_line=bool(h[i, j] == l[i, j] and a[i, j] > 0),
                    days_since_first=int(i - first_pos[j]),
                    prior_limit=ratio, beyond_prior=bool(abs(r) > ratio + 0.005 / close[i - 1, j]),
                    known_event_day=bool(d in known))
        tops.append(info)
        log("  %s %s ret_adj=%+.4f%% ret_raw=%+.4f%% factor_rel=%.6f days_since_first=%d "
            "prior=%.0f%% beyond=%s one_line=%s known_day=%s px %.2f->%.2f"
            % (info["date"], sym, 100 * r, 100 * rr, fr, info["days_since_first"],
               100 * ratio, info["beyond_prior"], info["one_line"], info["known_event_day"],
               info["close"], info["close_t"]))
    R["top5"] = tops

    # 归因统计：极端值里有多少落在上市初期（无涨跌幅窗口）/ 已知事件日
    big = np.abs(ret) > 0.21
    bi, bj = np.where(big)
    early = sum(1 for i, j in zip(bi, bj) if i - first_pos[j] < 5)
    R["n_abs_gt_21pct"] = int(big.sum())
    R["n_abs_gt_21pct_days_since_first_lt5"] = int(early)
    bigd = set(str(dates[i].date()) for i in bi)
    R["n_days_with_abs_gt_21pct"] = len(bigd)
    topdays = pd.Series({str(dates[i].date()): int((np.abs(ret[i]) > 0.21).sum()) for i in np.unique(bi)})
    R["top_days_abs_gt_21pct"] = topdays.sort_values(ascending=False).head(10).to_dict()
    log("|ret|>21%% cells: %s ; of which days_since_first<5: %s ; on %s distinct days"
        % (format(big.sum(), ","), format(early, ","), len(bigd)))
    log("days with most >21%% moves: %s" % R["top_days_abs_gt_21pct"])
    # ---- |ret|>21% 逐条归类：次新 / 长期停牌复牌 / 其他 ----
    trd2 = np.isfinite(a) & (a > 0)
    cats, details = {"new": 0, "resume": 0, "other": 0}, []
    for i, j in zip(bi, bj):
        k = np.flatnonzero(trd2[:i, j])
        ppos = int(k[-1]) if k.size else None
        gap = (i - ppos) if ppos is not None else None
        if i - first_pos[j] < 5:
            cat = "new"
        elif gap is None or gap >= 20:
            cat = "resume"
        else:
            cat = "other"
        cats[cat] += 1
        details.append(dict(date=str(dates[i].date()), symbol=str(syms[j]), ret=float(ret[i, j]),
                            days_since_first=int(i - first_pos[j]),
                            days_since_last_trade=(int(gap) if gap is not None else None),
                            cat=cat,
                            prev_close=float(close[ppos, j]) if ppos is not None else None,
                            close=float(close[i, j])))
    R["big_move_categories"] = cats
    R["big_moves_top20"] = sorted(details, key=lambda d: -abs(d["ret"]))[:20]
    log("|ret|>21%% classification: %s" % cats)
    for d in R["big_moves_top20"][:12]:
        log("  %s %s ret=%+.2f%% cat=%-6s days_since_first=%d days_since_last_trade=%s"
            % (d["date"], d["symbol"], 100 * d["ret"], d["cat"], d["days_since_first"],
               d["days_since_last_trade"]))
    return R
# =========================================================================== #
def part_d_scale(F, arrays, dates, syms):
    log("\n" + "=" * 100)
    log("PART D  报告关键数字独立复算（规模 / 掩码 / 停牌 / 一字板 / 涨跌停极端日 / 股票池）")
    log("=" * 100)
    R = {}
    o, h, l, c, a, nt, cum = arrays
    valid = np.isfinite(c)
    R["shape"] = list(c.shape)
    R["n_dates"] = int(c.shape[0])
    R["first_date"] = str(dates[0].date())
    R["last_date"] = str(dates[-1].date())
    R["n_symbols"] = int(c.shape[1])
    R["n_sh"] = int(sum(1 for s in syms if str(s).endswith(".SH")))
    R["n_sz"] = int(sum(1 for s in syms if str(s).endswith(".SZ")))
    log("panel: %d dates x %d symbols ; %s ~ %s ; SH %d / SZ %d"
        % (R["n_dates"], R["n_symbols"], R["first_date"], R["last_date"], R["n_sh"], R["n_sz"]))
    R["valid_cells"] = {f: int(np.isfinite(np.asarray(F[f].values)).sum()) for f in FIELD_ORDER}
    log("valid cells per field: %s" % R["valid_cells"])

    # 掩码
    mc = ~np.isfinite(c)
    R["mask_diff_vs_close"] = {f: int((~np.isfinite(np.asarray(F[f].values)) != mc).sum()) for f in FIELD_ORDER}
    R["missing_cells"] = {f: int((~np.isfinite(np.asarray(F[f].values))).sum()) for f in FIELD_ORDER}
    log("mask sym-diff vs close: %s" % R["mask_diff_vs_close"])
    nt_missing = ~np.isfinite(nt)
    R["nt_missing_total"] = int(nt_missing.sum())
    R["nt_missing_and_close_valid"] = int((nt_missing & valid).sum())
    R["nt_missing_close_valid_last_day"] = int((nt_missing & valid)[-1].sum())
    R["nt_missing_close_valid_other_days"] = R["nt_missing_and_close_valid"] - R["nt_missing_close_valid_last_day"]
    R["nt_days_with_gap"] = int(((nt_missing & valid).sum(1) > 0).sum())
    log("num_trades missing: total %s ; close-valid %s ; last day %s ; other days %s on %d days"
        % (format(R["nt_missing_total"], ","), format(R["nt_missing_and_close_valid"], ","),
           format(R["nt_missing_close_valid_last_day"], ","), format(R["nt_missing_close_valid_other_days"], ","),
           R["nt_days_with_gap"]))

    # 停牌 / 一字板
    flat = valid & (o == h) & (h == l) & (l == c)
    susp = flat & np.isfinite(a) & (a == 0)
    oline = valid & (h == l) & np.isfinite(a) & (a > 0)
    R["suspended_cells"] = int(susp.sum())
    R["one_line_cells"] = int(oline.sum())
    R["flat_cells"] = int((susp | oline).sum())
    # 一字板方向（用当日原始收益符号近似）
    prev_c = prev_row(c)
    rr = np.where(np.isfinite(c) & np.isfinite(prev_c) & (prev_c > 0), c / prev_c - 1.0, np.nan)
    R["one_line_up"] = int((oline & (rr > 0)).sum())
    R["one_line_down"] = int((oline & (rr < 0)).sum())
    log("suspended (amt==0 & flat) : %s ; one_line (h==l & amt>0): %s ; flat total: %s"
        % (format(R["suspended_cells"], ","), format(R["one_line_cells"], ","), format(R["flat_cells"], ",")))
    log("one_line split by raw return sign: up %s / down %s (report: 11,923 / 9,908)"
        % (format(R["one_line_up"], ","), format(R["one_line_down"], ",")))

    # 每日规模
    n_listed = valid.sum(1)
    n_traded = (valid & np.isfinite(a) & (a > 0)).sum(1)
    R["n_listed_first"], R["n_listed_median"], R["n_listed_last"], R["n_listed_max"] = (
        int(n_listed[0]), int(np.median(n_listed)), int(n_listed[-1]), int(n_listed.max()))
    R["n_traded_first"], R["n_traded_median"], R["n_traded_last"], R["n_traded_max"] = (
        int(n_traded[0]), int(np.median(n_traded)), int(n_traded[-1]), int(n_traded.max()))
    log("n_listed first/median/last/max: %s / %s / %s / %s (report 3,760 / 4,994 / 5,226 / 5,226)"
        % (format(R["n_listed_first"], ","), format(R["n_listed_median"], ","),
           format(R["n_listed_last"], ","), format(R["n_listed_max"], ",")))
    log("n_traded first/median/last/max: %s / %s / %s / %s (report 3,740 / 4,965 / 5,216 / 5,216)"
        % (format(R["n_traded_first"], ","), format(R["n_traded_median"], ","),
           format(R["n_traded_last"], ","), format(R["n_traded_max"], ",")))

    # 股票池粗复算（独立近似：上市>=250日 & 20日均额>=2000万 & 价>=2 & 可交易）
    ls_days = valid.cumsum(0)
    adv20 = pd.DataFrame(a).rolling(20, min_periods=1).mean().to_numpy()
    uni = valid & np.isfinite(a) & (a > 0) & ~susp & ~oline & (ls_days >= 250) & (adv20 >= 2e7) & (c >= 2.0)
    nu = uni.sum(1)
    R["n_universe_first"], R["n_universe_median"], R["n_universe_last"], R["n_universe_max"] = (
        int(nu[0]), int(np.median(nu)), int(nu[-1]), int(nu.max()))
    log("my universe approx first/median/last/max: %s / %s / %s / %s (report 0 / 4,101 / 4,917 / 5,044)"
        % (format(R["n_universe_first"], ","), format(R["n_universe_median"], ","),
           format(R["n_universe_last"], ","), format(R["n_universe_max"], ",")))

    # 涨跌停极端日（独立粗口径）
    star = np.array([str(s).startswith(("688", "689")) for s in syms])
    gem = np.array([str(s).startswith(("300", "301")) for s in syms])
    gd = dates >= pd.Timestamp("2020-08-24")
    ratio = np.where(star[None, :], 0.20, np.where(gem[None, :], np.where(gd[:, None], 0.20, 0.10), 0.10))
    adj = c * np.load(os.path.join(HERE, "_clean_factor.npy"))
    preva = prev_row(adj)
    botha = np.isfinite(adj) & np.isfinite(preva)
    with np.errstate(invalid="ignore", divide="ignore"):
        ret = np.where(botha, adj / preva - 1.0, np.nan)
    up = botha & (ret >= ratio - 0.003)
    dn = botha & (ret <= -(ratio - 0.003))
    nup, ndn = up.sum(1), dn.sum(1)
    R["crude_limit_up_max_per_day"] = int(nup.max())
    R["crude_limit_down_max_per_day"] = int(ndn.max())
    log("crude limit-up   max/day: %s (report max 2,498)" % format(int(nup.max()), ","))
    log("crude limit-down max/day: %s (report max 3,638)" % format(int(ndn.max()), ","))
    # ---- 涨跌停「触及」口径（0.01 元价格网格 + 半格容差），独立实现 ----
    cl = np.load(os.path.join(HERE, "_clean_factor.npy"))
    ref = np.full_like(c, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        ref[1:] = (c[:-1] * cl[:-1]) / cl[1:]
    lup = np.floor(ref * (1.0 + ratio) * 100.0 + 0.5) / 100.0
    ldn = np.floor(ref * (1.0 - ratio) * 100.0 + 0.5) / 100.0
    fp = np.full(c.shape[1], -1, dtype=int)
    finc = np.isfinite(c)
    for jj2 in range(c.shape[1]):
        w = np.flatnonzero(finc[:, jj2])
        if w.size:
            fp[jj2] = int(w[0])
    dsf = np.arange(c.shape[0])[:, None] - fp[None, :]
    new_win = (dsf < 5) & (fp[None, :] > 0)
    excl2 = new_win | (botha & (np.abs(ret) > ratio + 0.003))
    tup = np.isfinite(h) & (np.abs(h - lup) < 0.005) & ~excl2
    tdn = np.isfinite(l) & (np.abs(l - ldn) < 0.005) & ~excl2
    ntup, ntdn = tup.sum(1), tdn.sum(1)
    R["price_grid_limit_up_max_per_day"] = int(ntup.max())
    R["price_grid_limit_down_max_per_day"] = int(ntdn.max())
    log("price-grid touch limit-up   max/day: %s (report 2,498)" % format(int(ntup.max()), ","))
    log("price-grid touch limit-down max/day: %s (report 3,638)" % format(int(ntdn.max()), ","))
    R["price_grid_extreme_days"] = {}
    _days = {"2024-10-08": ("up", 2498), "2024-09-30": ("up", 1157), "2024-02-08": ("up", 800),
             "2025-04-07": ("down", 3638), "2020-02-03": ("down", 3338), "2024-02-05": ("down", 1928)}
    for ds, (kind, claimed) in _days.items():
        i = int(np.flatnonzero(dates == pd.Timestamp(ds))[0])
        got = int(ntup[i] if kind == "up" else ntdn[i])
        R["price_grid_extreme_days"][ds] = dict(kind=kind, price_grid=got, report=claimed)
        log("  %s %-4s price_grid=%s  report=%s" % (ds, kind, format(got, ","), format(claimed, ",")))
    # 全样本突破先验上限的格数（近似报告 §4 的 266 格 / 153 只）
    with np.errstate(invalid="ignore", divide="ignore"):
        tol2 = 0.002 + 0.005 / np.where(np.isfinite(ref) & (ref > 0), ref, np.nan)
    ex2 = botha & ~new_win & (np.abs(ret) > ratio + tol2)
    R["breach_prior_cells"] = int(np.nansum(ex2))
    R["breach_prior_symbols"] = int((ex2.sum(0) > 0).sum())
    log("approx cells breaching prior limit: %s in %s symbols (report 266 / 153)"
        % (format(R["breach_prior_cells"], ","), format(R["breach_prior_symbols"], ",")))
    days = {"2024-10-08": ("up", 2498), "2024-09-30": ("up", 1157), "2024-02-08": ("up", 800),
            "2025-04-07": ("down", 3638), "2020-02-03": ("down", 3338), "2024-02-05": ("down", 1928)}
    R["crude_extreme_days"] = {}
    for ds, (kind, claimed) in days.items():
        i = int(np.flatnonzero(dates == pd.Timestamp(ds))[0])
        got = int(nup[i] if kind == "up" else ndn[i])
        R["crude_extreme_days"][ds] = dict(kind=kind, crude=got, report=claimed)
        log("  %s %-4s crude=%s  report=%s" % (ds, kind, format(got, ","), format(claimed, ",")))
    return R
# =========================================================================== #
EMURL = ("http://push2his.eastmoney.com/api/qt/stock/kline/get?secid={secid}"
         "&fields1=f1,f2,f3,f4,f5&fields2=f51,f52,f53,f54,f55,f56,f57&klt=101"
         "&fqt={fq}&beg=20200101&end=20301231&lmt=1000000")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def _get_json(url, retry=3, pause=2.5):
    last = None
    for k in range(retry):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA,
                                                       "Referer": "https://quote.eastmoney.com/"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.loads(r.read().decode("utf-8", "ignore"))
        except Exception as e:                                   # noqa: BLE001
            last = e
            time.sleep(pause * (k + 1))
    raise RuntimeError("%r <- %s" % (last, url[:80]))


def em_kline(sym, fq=0):
    code, mk = sym.split(".")
    secid = ("1." if mk == "SH" else "0.") + code
    d = _get_json(EMURL.format(secid=secid, fq=fq))
    kl = (d.get("data") or {}).get("klines") or []
    if not kl:
        return pd.DataFrame()
    df = pd.DataFrame([x.split(",") for x in kl],
                      columns=["date", "open", "close", "high", "low", "volume", "amount"])
    df["date"] = pd.to_datetime(df["date"])
    for cc in ("open", "close", "high", "low", "volume", "amount"):
        df[cc] = pd.to_numeric(df[cc], errors="coerce")
    return df.set_index("date").sort_index()


def part_e_external(check_syms, events, o, h, l, c, a, clean, dates, syms):
    log("\n" + "=" * 100)
    log("PART E  第二外部源（东方财富 http）逐格复核：OHLC + 成交额 + 前复权收益 + 事件日")
    log("=" * 100)
    R = {"note": "东方财富 push2his kline（http），与 s02 的腾讯源相互独立；"
                 "HTTPS 端口在本机被拒（RemoteDisconnected），HTTP 可用"}
    rows = []
    R["symbols"] = check_syms
    for sym in check_syms:
        try:
            em0 = em_kline(sym, 0)
            time.sleep(0.6)
            em1 = em_kline(sym, 1)
            time.sleep(0.6)
        except Exception as e:                                   # noqa: BLE001
            log("- %s 拉取失败，未能验证：%s" % (sym, e))
            rows.append(dict(symbol=sym, status="fetch_failed", error=repr(e)[:120]))
            continue
        if em0.empty:
            log("- %s 外部源无数据" % sym)
            rows.append(dict(symbol=sym, status="no_data"))
            continue
        common = em0.index.intersection(pd.DatetimeIndex(dates))
        loc = pd.DatetimeIndex(dates).get_indexer(common)
        j = syms.index(sym)
        rec = dict(symbol=sym, status="ok", days=len(common))
        for f, arr in (("open", o), ("high", h), ("low", l), ("close", c)):
            mine = arr[loc, j]
            ext = em0[f].loc[common].to_numpy()
            bad = ~np.isclose(mine, ext, rtol=0, atol=1e-6)
            rec["%s_mismatch" % f] = int(bad.sum())
            rec["%s_maxdiff" % f] = float(np.nanmax(np.abs(mine - ext)))
        mine_amt = a[loc, j]
        ext_amt = em0["amount"].loc[common].to_numpy()
        okm = np.isfinite(mine_amt) & np.isfinite(ext_amt) & (ext_amt > 0)
        rel = np.abs(mine_amt[okm] - ext_amt[okm]) / ext_amt[okm]
        rec["amt_days"] = int(okm.sum())
        rec["amt_rel_gt_1e6"] = int((rel > 1e-6).sum())
        rec["amt_max_rel"] = float(rel.max()) if rel.size else None
        rec["amt_exact_eq"] = int((mine_amt[okm] == ext_amt[okm]).sum())
        if not em1.empty:
            common2 = em1.index.intersection(common)
            loc2 = pd.DatetimeIndex(dates).get_indexer(common2)
            emr = em1["close"].loc[common2].pct_change(fill_method=None).to_numpy()
            adj = c * clean
            preva = prev_row(adj)
            pc = prev_row(c)
            with np.errstate(invalid="ignore", divide="ignore"):
                myr = np.where(np.isfinite(adj[loc2, j]) & np.isfinite(preva[loc2, j]),
                               adj[loc2, j] / preva[loc2, j] - 1.0, np.nan)
                myraw = np.where(np.isfinite(c[loc2, j]) & np.isfinite(pc[loc2, j]),
                                 c[loc2, j] / pc[loc2, j] - 1.0, np.nan)
            dd = np.abs(myr - emr)
            good = np.isfinite(dd)
            rec["ret_days"] = int(good.sum())
            rec["ret_median_diff"] = float(np.median(dd[good])) if good.any() else None
            rec["ret_max_diff"] = float(np.max(dd[good])) if good.any() else None
            pcl = prev_row(clean)
            fc = np.where(np.isfinite(clean[loc2, j]) & np.isfinite(pcl[loc2, j]),
                          clean[loc2, j] / pcl[loc2, j], np.nan)
            ev = (np.abs(fc - 1.0) >= 1e-4) & good
            rec["exdiv_days"] = int(ev.sum())
            rec["exdiv_max_adj_diff"] = float(np.max(dd[ev])) if ev.any() else None
            rawd = np.abs(myraw - emr)
            rec["exdiv_max_raw_diff"] = float(np.max(rawd[ev])) if ev.any() else None
        rows.append(rec)
        log("- %s days=%s mismatch(o/h/l/c)=%s/%s/%s/%s maxdiff=%.6f amt_rel>1e-6=%s/%s amt_max_rel=%s"
            % (sym, format(rec["days"], ","), rec["open_mismatch"], rec["high_mismatch"],
               rec["low_mismatch"], rec["close_mismatch"],
               max(rec["open_maxdiff"], rec["high_maxdiff"], rec["low_maxdiff"], rec["close_maxdiff"]),
               format(rec["amt_rel_gt_1e6"], ","), format(rec["amt_days"], ","), rec["amt_max_rel"]))
        if rec.get("ret_days"):
            log("  qfq returns: days=%s median|diff|=%.2e max=%.4f%% | factor-event days=%s adj_max=%.4f%% raw_max=%.4f%%"
                % (format(rec["ret_days"], ","), rec["ret_median_diff"] or 0,
                   100 * (rec["ret_max_diff"] or 0), rec["exdiv_days"],
                   100 * (rec["exdiv_max_adj_diff"] or 0), 100 * (rec["exdiv_max_raw_diff"] or 0)))
    R["rows"] = rows
    pd.DataFrame(rows).to_csv(OUTCSV, index=False, encoding="utf-8-sig")

    # ---- 事件日独立复核：复牌 / 退市整理首日 / 因子重算日 ----
    log("event-day external re-check:")
    erows = []
    for sym, ds in events:
        try:
            em0 = em_kline(sym, 0)
            time.sleep(0.6)
            em1 = em_kline(sym, 1)
            time.sleep(0.6)
        except Exception as e:                                   # noqa: BLE001
            log("  %s %s 拉取失败，未能验证：%s" % (sym, ds, e))
            erows.append(dict(symbol=sym, date=ds, status="fetch_failed", error=repr(e)[:120]))
            continue
        d0 = pd.Timestamp(ds)
        if d0 not in em0.index or sym not in syms:
            log("  %s %s 外部无当日数据" % (sym, ds))
            erows.append(dict(symbol=sym, date=ds, status="no_data"))
            continue
        pos = int(em0.index.get_loc(d0))
        prev = em0.index[pos - 1] if pos > 0 else None
        j = syms.index(sym)
        i = int(pd.DatetimeIndex(dates).get_indexer([d0])[0])
        rec = dict(symbol=sym, date=ds, em_prev_date=str(prev.date()) if prev is not None else None)
        for f, arr in (("open", o), ("high", h), ("low", l), ("close", c)):
            rec["local_%s" % f] = float(arr[i, j])
            rec["em_%s" % f] = float(em0[f].loc[d0])
        rec["local_prev_close"] = float(c[i - 1, j]) if np.isfinite(c[i - 1, j]) else None
        rec["em_prev_close"] = float(em0["close"].loc[prev]) if prev is not None else None
        rec["local_amt"] = float(a[i, j])
        rec["em_amount"] = float(em0["amount"].loc[d0])
        rec["local_raw_ret"] = ((rec["local_close"] / rec["local_prev_close"] - 1.0)
                                if rec["local_prev_close"] else None)
        rec["em_raw_ret"] = ((rec["em_close"] / rec["em_prev_close"] - 1.0)
                             if rec["em_prev_close"] else None)
        if prev is not None and d0 in em1.index:
            rec["em_qfq_ret"] = float(em1["close"].loc[d0] / em1["close"].loc[prev] - 1.0)
        cl = c[i, j] * clean[i, j]
        clp = c[i - 1, j] * clean[i - 1, j]
        rec["local_adj_ret"] = float(cl / clp - 1.0) if np.isfinite(cl) and np.isfinite(clp) else None
        rec["factor_rel"] = (float(clean[i, j] / clean[i - 1, j])
                             if np.isfinite(clean[i, j]) and np.isfinite(clean[i - 1, j]) else None)
        erows.append(rec)
        okraw = (rec["local_raw_ret"] is not None and rec["em_raw_ret"] is not None
                 and abs(rec["local_raw_ret"] - rec["em_raw_ret"]) < 1e-6)
        log("  %s %s raw local %+.2f%% = EM %+.2f%% (%s) | local adj %+.2f%% vs EM qfq %+.2f%% | factor_rel=%.6f"
            % (sym, ds, 100 * (rec["local_raw_ret"] or 0), 100 * (rec["em_raw_ret"] or 0),
               "MATCH" if okraw else "CHECK", 100 * (rec["local_adj_ret"] or 0),
               100 * (rec.get("em_qfq_ret") or 0), rec["factor_rel"] or 0))
    R["event_rows"] = erows
    pd.DataFrame(erows).to_csv(OUTCSVEV, index=False, encoding="utf-8-sig")
    return R


# =========================================================================== #
def main():
    import datetime
    log("=" * 100)
    log("A-track independent check  |  data dir: %s" % DATA)
    log("started: %s" % datetime.datetime.now().isoformat(timespec="seconds"))
    log("=" * 100)
    D = load_fields(FIELD_ORDER)
    dates = pd.DatetimeIndex(D["close"].index)
    syms = [str(s) for s in D["close"].columns]
    o, h, l, c, a, nt, cum = (D[f].to_numpy() for f in FIELD_ORDER)
    R = dict(script="1_数据真实性验证/audit/indep_check.py",
             env=dict(python=__import__("sys").version,
                      pandas=pd.__version__, numpy=np.__version__,
                      data_dir=DATA, when=datetime.datetime.now().isoformat(timespec="seconds")))
    R["A"] = part_a_ohlc(o, h, l, c, dates, syms)
    R["B"], clean = part_b_cumadj(cum, c, o, h, l, a, dates, syms)
    R["C"] = part_c_returns(clean, c, o, h, l, a, cum, dates, syms)
    R["D"] = part_d_scale(D, (o, h, l, c, a, nt, cum), dates, syms)
    R["E"] = part_e_external(["600519.SH", "000001.SZ", "300750.SZ", "688981.SH"],
                             [("000670.SZ", "2022-08-22"), ("600145.SH", "2020-06-30"),
                              ("600766.SH", "2024-06-05")],
                             o, h, l, c, a, clean, dates, syms)
    for _tmp in ("_clean_factor.npy",):
        _p = os.path.join(HERE, _tmp)
        if os.path.exists(_p):
            os.remove(_p)
    with open(OUTJSON, "w", encoding="utf-8") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=1)
    log("\nJSON 写出：%s" % OUTJSON)
    log("CSV  写出：%s" % OUTCSV)


if __name__ == "__main__":
    main()