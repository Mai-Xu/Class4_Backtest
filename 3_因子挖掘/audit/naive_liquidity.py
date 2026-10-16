# -*- coding: utf-8 -*-
"""独立复核 factor_zoo/liquidity.py 的 14 个流动性因子（只读被审代码，不修改）。

方法（全部是"另一条实现路径"）：
  A. 逐列 python 循环 + numpy.polyfit / numpy.corrcoef 重写每个因子，
     窗口端点显式写死 [t-w+1, t]；在 6 个交易日上与模块输出逐格比对
     （max|diff| + NaN 掩码是否完全一致）。
  B. 全样本扫描：shape / dtype / 是否出现 inf / 覆盖率。
  C. 端点扰动扫描：逐 lag 改写某一行 A/NT/R，实测因子真正使用的最长滞后，
     独立确定窗口端点（不依赖读源码）。
  D. 未来扰动检验：改写 t 之后 30 行的数据，输出必须完全不变。
  E. 停牌污染量化：把 amt==0 / num_trades==0 当有效观测时，会改变多少只股票的
     因子值（模块本身是否安全）。
  F. 专项：Amihud 分子用复权还是未复权收益；成交额单位；rel20_60 分母是否含 t。

运行（项目根目录）:
    $env:PYTHONIOENCODING="utf-8"; python 3_因子挖掘/audit/naive_liquidity.py
    $env:PYTHONIOENCODING="utf-8"; python 3_因子挖掘/audit/naive_liquidity.py --step 5  # 全局扫描抽样
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.lib import MarketData          # noqa: E402
from factor_zoo.ctx import Ctx           # noqa: E402
import factor_zoo.liquidity as LQ        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_DATES = ("2020-03-31", "2021-06-30", "2022-12-30",
              "2023-07-31", "2024-06-28", "2025-06-30")

# 从代码读出的窗口行数（仅用于停牌污染量化；端点由扰动扫描独立复核）
WIN_ROWS = {
    "lq_amt_log20": 20, "lq_amt_rel20_60": 60,
    "lq_amihud20": 20, "lq_amihud60": 60,
    "lq_amt_trend20_120": 120, "lq_amt_slope60": 60,
    "lq_amt_cv60": 60, "lq_amt_stab60": 61,
    "lq_trade_size20": 20, "lq_trade_size_chg": 120, "lq_trade_size_skew60": 60,
    "lq_amt_impact60": 61,
    "lq_nt_log20": 20, "lq_nt_trend20_120": 120,
}

# ==================================================================== #
# 朴素实现：逐列、显式窗口 [t-w+1, t]，全部用 numpy 基础函数重算
# ==================================================================== #
def win(col, t, w):
    """返回 (窗口切片, 起始行)——与 Ctx.sl 无关的独立实现。"""
    lo = max(int(t) - int(w) + 1, 0)
    return col[lo:int(t) + 1], lo


def okv(v, bad):
    fin = np.isfinite(v)
    return fin if bad else (fin & (v > 0.0))


def mmean(v, bad, mincnt):
    ok = okv(v, bad)
    if int(ok.sum()) < int(mincnt):
        return np.nan
    with np.errstate(all="ignore"):
        return float(v[ok].mean())


def mstd(v, bad, mincnt):
    ok = okv(v, bad)
    if int(ok.sum()) < int(mincnt):
        return np.nan
    with np.errstate(all="ignore"):
        return float(v[ok].std(ddof=1))


def mcorr(x, y, mincnt):
    if len(x) < int(mincnt) or len(x) < 2:
        return np.nan
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return np.nan
    sx, sy = float(np.std(x)), float(np.std(y))
    if not np.isfinite(sx) or not np.isfinite(sy) or sx == 0.0 or sy == 0.0:
        return np.nan
    with np.errstate(all="ignore"):
        return float(np.corrcoef(np.asarray(x, dtype="float64"),
                                 np.asarray(y, dtype="float64"))[0, 1])


def mslope(i, y, mincnt):
    if len(i) < int(mincnt) or not np.isfinite(y).all():
        return np.nan
    if float(np.ptp(i)) == 0.0:
        return np.nan
    with np.errstate(all="ignore"):
        return float(np.polyfit(np.asarray(i, dtype="float64"),
                                np.asarray(y, dtype="float64"), 1)[0])


# -------------------------------------------------------------------- #
def n_lq_amt_log20(a, nt, r, t, bad):
    v, _ = win(a, t, 20)
    m = mmean(v, bad, 10)
    if np.isfinite(m) and m > 0.0:
        return float(np.log(m))
    if bad and np.isfinite(m) and m == 0.0:
        return -np.inf
    return np.nan


def n_lq_amt_rel20_60(a, nt, r, t, bad):
    v, _ = win(a, t, 60)
    den = mmean(v, bad, 30)
    a0 = a[int(t)]
    if not np.isfinite(a0) or (a0 <= 0.0 and not bad):
        return np.nan
    if not np.isfinite(den) or den <= 0.0:
        return -np.inf if (bad and np.isfinite(den)) else np.nan
    return float(a0 / den - 1.0)


def n_lq_amt_rel20_60_clean(a, nt, r, t, bad):
    """分母改为 [t-60, t-1]（不含 t）的对照口径，仅用于量化口径影响。"""
    if int(t) < 1:
        return np.nan
    v, _ = win(a[:int(t)], int(t) - 1, 60)     # 行 t-60 .. t-1
    den = mmean(v, bad, 30)
    a0 = a[int(t)]
    if not np.isfinite(a0) or a0 <= 0.0:
        return np.nan
    if not np.isfinite(den) or den <= 0.0:
        return np.nan
    return float(a0 / den - 1.0)


def n_amihud(a, r, t, w, mc, bad):
    v, _ = win(a, t, w)
    rr, _ = win(r, t, w)
    ok = np.isfinite(v) & np.isfinite(rr)
    if not bad:
        ok &= v > 0.0
    if int(ok.sum()) < int(mc):
        return np.nan
    with np.errstate(all="ignore"):
        val = float((np.abs(rr[ok]) / v[ok]).mean())
    return val * 1e8


def n_lq_amihud20(a, nt, r, t, bad):
    return n_amihud(a, r, t, 20, 12, bad)


def n_lq_amihud60(a, nt, r, t, bad):
    return n_amihud(a, r, t, 60, 36, bad)


def n_lq_amt_trend20_120(a, nt, r, t, bad):
    v20, _ = win(a, t, 20)
    v120, _ = win(a, t, 120)
    s = mmean(v20, bad, 12)
    l = mmean(v120, bad, 60)
    if not (np.isfinite(s) and np.isfinite(l)):
        return np.nan
    if l <= 0.0:
        return np.inf if (bad and l == 0.0) else np.nan
    return float(s / l - 1.0)


def n_lq_amt_slope60(a, nt, r, t, bad):
    v, _ = win(a, t, 60)
    ok = okv(v, bad)
    if int(ok.sum()) < 36:
        return np.nan
    i = np.arange(len(v), dtype="float64")[ok]
    with np.errstate(all="ignore"):
        y = np.log(v[ok])
    b = mslope(i, y, 36)
    return np.nan if not np.isfinite(b) else float(b * 100.0)


def n_lq_amt_cv60(a, nt, r, t, bad):
    v, _ = win(a, t, 60)
    s = mstd(v, bad, 36)
    m = mmean(v, bad, 36)
    if np.isfinite(s) and np.isfinite(m) and m > 0.0:
        return float(s / m)
    return np.nan


def n_lq_amt_stab60(a, nt, r, t, bad):
    v, _ = win(a, t, 61)                    # 61 行 -> 60 对
    ok = okv(v, bad)
    pair = ok[1:] & ok[:-1]
    if int(pair.sum()) < 36:
        return np.nan
    with np.errstate(all="ignore"):
        lx = np.log(v[:-1][pair])
        ly = np.log(v[1:][pair])
    return mcorr(lx, ly, 36)


def n_size(a, nt, t, w, mc, bad):
    va, _ = win(a, t, w)
    vn, _ = win(nt, t, w)
    ok = np.isfinite(va) & np.isfinite(vn)
    if not bad:
        ok &= (va > 0.0) & (vn > 0.0)
    if int(ok.sum()) < int(mc):
        return np.nan
    with np.errstate(all="ignore"):
        sz = np.log(va[ok] / vn[ok])
    if not np.isfinite(sz).all():
        return np.nan
    return float(sz.mean())


def n_lq_trade_size20(a, nt, r, t, bad):
    return n_size(a, nt, t, 20, 12, bad)


def n_lq_trade_size_chg(a, nt, r, t, bad):
    s = n_size(a, nt, t, 20, 12, bad)
    l = n_size(a, nt, t, 120, 60, bad)
    if np.isfinite(s) and np.isfinite(l):
        return float(s - l)
    return np.nan


def n_skew(a, nt, t, w, mc, bad, style="pop"):
    """style='pop' = 教科书三阶矩偏度 m3/m2^1.5（中心矩 1/n）；
       style='sample' = 按模块的 ddof=1 归一化（等价 pop*sqrt((n-1)/n)）。"""
    va, _ = win(a, t, w)
    vn, _ = win(nt, t, w)
    ok = np.isfinite(va) & np.isfinite(vn)
    if not bad:
        ok &= (va > 0.0) & (vn > 0.0)
    n = int(ok.sum())
    if n < int(mc):
        return np.nan
    with np.errstate(all="ignore"):
        sz = np.log(va[ok] / vn[ok])
    if not np.isfinite(sz).all():
        return np.nan
    c = sz - sz.mean()
    m2 = float((c ** 2).mean())
    m3 = float((c ** 3).mean())
    if not np.isfinite(m2) or m2 <= 0.0:
        return np.nan
    sk = m3 / m2 ** 1.5
    if style == "sample":
        sk *= float(np.sqrt((n - 1.0) / n))
    return float(sk)


def n_lq_trade_size_skew60(a, nt, r, t, bad):
    return n_skew(a, nt, t, 60, 36, bad, style="pop")


def n_lq_trade_size_skew60_modstyle(a, nt, r, t, bad):
    return n_skew(a, nt, t, 60, 36, bad, style="sample")


def n_lq_amt_impact60(a, nt, r, t, bad):
    va, _ = win(a, t, 61)
    vr, _ = win(r, t, 61)
    ok = okv(va, bad)
    pair = ok[1:] & ok[:-1] & np.isfinite(vr[1:])
    if int(pair.sum()) < 36:
        return np.nan
    with np.errstate(all="ignore"):
        d = np.log(va[1:][pair]) - np.log(va[:-1][pair])
    c = mcorr(d, vr[1:][pair], 36)
    return np.nan if not np.isfinite(c) else float(-c)


def n_lq_nt_log20(a, nt, r, t, bad):
    v, _ = win(nt, t, 20)
    m = mmean(v, bad, 10)
    if np.isfinite(m) and m > 0.0:
        return float(np.log(m))
    if bad and np.isfinite(m) and m == 0.0:
        return -np.inf
    return np.nan


def n_lq_nt_trend20_120(a, nt, r, t, bad):
    v20, _ = win(nt, t, 20)
    v120, _ = win(nt, t, 120)
    s = mmean(v20, bad, 12)
    l = mmean(v120, bad, 60)
    if np.isfinite(s) and np.isfinite(l) and l > 0.0:
        return float(s / l - 1.0)
    return np.nan


NAIVE = {
    "lq_amt_log20": n_lq_amt_log20,
    "lq_amt_rel20_60": n_lq_amt_rel20_60,
    "lq_amihud20": n_lq_amihud20,
    "lq_amihud60": n_lq_amihud60,
    "lq_amt_trend20_120": n_lq_amt_trend20_120,
    "lq_amt_slope60": n_lq_amt_slope60,
    "lq_amt_cv60": n_lq_amt_cv60,
    "lq_amt_stab60": n_lq_amt_stab60,
    "lq_trade_size20": n_lq_trade_size20,
    "lq_trade_size_chg": n_lq_trade_size_chg,
    "lq_trade_size_skew60": n_lq_trade_size_skew60,
    "lq_amt_impact60": n_lq_amt_impact60,
    "lq_nt_log20": n_lq_nt_log20,
    "lq_nt_trend20_120": n_lq_nt_trend20_120,
}

NAIVE_ALT = {
    "lq_trade_size_skew60": n_lq_trade_size_skew60_modstyle,
    "lq_amt_rel20_60": n_lq_amt_rel20_60_clean,
}


# ==================================================================== #
# 对比工具
# ==================================================================== #
def cross_section(name, A, NT, R, t, bad=False, alt=False):
    """逐列调用朴素实现，得到全横截面。"""
    fn = NAIVE_ALT[name] if alt else NAIVE[name]
    k = A.shape[1]
    out = np.empty(k, dtype="float64")
    for j in range(k):
        out[j] = fn(A[:, j], NT[:, j], R[:, j], int(t), bad)
    return out


def cmp_cells(mod, ref):
    """模块输出 vs 朴素输出：NaN 掩码不一致格数 + 双方有限格的 max|diff|。"""
    m_nan = np.isnan(mod)
    r_nan = np.isnan(ref)
    mis = int((m_nan != r_nan).sum())
    both = (~m_nan) & (~r_nan)
    fin = both & np.isfinite(mod) & np.isfinite(ref)
    mx = 0.0
    big = 0
    if fin.any():
        d = np.abs(mod[fin] - ref[fin])
        mx = float(d.max())
        big = int((d > 1e-9 * np.maximum(1.0, np.maximum(np.abs(mod[fin]), np.abs(ref[fin])))).sum())
    return dict(nan_mismatch=mis, max_abs_diff=mx, finite_pairs=int(fin.sum()),
                big_diff=big, mod_inf=int(np.isinf(mod).sum()), ref_inf=int(np.isinf(ref).sum()))


def changed_count(base, got, tol=1e-9):
    """两种口径下因子值发生变化的股票数（NaN 位置变化、inf 出现、数值变化都算）。"""
    b_nan = np.isnan(base)
    g_nan = np.isnan(got)
    diff = b_nan != g_nan
    both = (~b_nan) & (~g_nan)
    nonfin = both & (~np.isfinite(base) | ~np.isfinite(got))
    fin = both & np.isfinite(base) & np.isfinite(got)
    val = np.zeros(base.shape, dtype=bool)
    if fin.any():
        x, y = base[fin], got[fin]
        val[fin] = np.abs(x - y) > tol * np.maximum(1.0, np.maximum(np.abs(x), np.abs(y)))
    return int((diff | nonfin | val).sum())


def clone_ctx(C):
    c = Ctx.__new__(Ctx)
    c.md = C.md
    for an in ("R", "C", "RAW", "O", "H", "L", "A", "NT", "ADJ_O", "ADJ_H",
               "ADJ_L", "listed", "mkt", "mkt_fill", "VWAP_PROXY"):
        setattr(c, an, getattr(C, an))
    c.n, c.k = C.n, C.k
    return c


def endpoint_sweep(name, spec, C, t, maxlag=135):
    """逐个 lag 扰动第 t-k 行，实测该因子真正读取到的最长滞后。"""
    base = np.asarray(spec["func"](int(t), C.md, C), dtype="float64")
    Cw = clone_ctx(C)
    Cw.A = C.A.copy()
    Cw.NT = C.NT.copy()
    Cw.R = C.R.copy()
    hits = []
    t = int(t)
    for k in range(int(maxlag) + 1):
        row = t - k
        if row < 0:
            hits.append(0)
            continue
        a0 = Cw.A[row].copy()
        n0 = Cw.NT[row].copy()
        r0 = Cw.R[row].copy()
        va = Cw.A[row]
        fin = np.isfinite(va) & (va > 0.0)
        va[fin] = va[fin] * 1.07 + 1000.0
        vn = Cw.NT[row]
        finn = np.isfinite(vn) & (vn > 0.0)
        vn[finn] = vn[finn] * 1.07 + 100.0
        vr = Cw.R[row]
        finr = np.isfinite(vr)
        vr[finr] = vr[finr] + 0.013
        try:
            got = np.asarray(spec["func"](t, C.md, Cw), dtype="float64")
        finally:
            Cw.A[row] = a0
            Cw.NT[row] = n0
            Cw.R[row] = r0
        hits.append(changed_count(base, got))
    last = max([k for k, h in enumerate(hits) if h > 0], default=-1)
    return dict(lag_hits=hits, max_lag_changed=last, window_rows=last + 1 if last >= 0 else 0,
                hits_at_boundary=hits[last + 1] if last + 1 < len(hits) else None)


def future_test(name, spec, C, t, horizon=30):
    """改写 t 之后 horizon 行，输出必须完全不变（独立因果性检验）。"""
    t = int(t)
    base = np.asarray(spec["func"](t, C.md, C), dtype="float64")
    Cw = clone_ctx(C)
    Cw.A = C.A.copy()
    Cw.NT = C.NT.copy()
    Cw.R = C.R.copy()
    Cw.C = C.C.copy()
    hi = min(t + int(horizon), C.n - 1)
    sl = slice(t + 1, hi + 1)
    with np.errstate(all="ignore"):
        Cw.A[sl] = np.where(np.isfinite(Cw.A[sl]), Cw.A[sl] * 1.31 + 777.0, Cw.A[sl])
        Cw.NT[sl] = np.where(np.isfinite(Cw.NT[sl]), Cw.NT[sl] * 1.31 + 77.0, Cw.NT[sl])
        Cw.R[sl] = np.where(np.isfinite(Cw.R[sl]), Cw.R[sl] + 0.02, Cw.R[sl])
        Cw.C[sl] = Cw.C[sl] * 1.07
        got = np.asarray(spec["func"](t, C.md, Cw), dtype="float64")
    return changed_count(base, got)


def amihud_with_returns(A, Rmat, t, w, mc):
    """用给定收益矩阵重算 Amihud（用于 复权 vs 未复权 对比）。"""
    lo = max(int(t) - int(w) + 1, 0)
    k = A.shape[1]
    out = np.full(k, np.nan)
    for j in range(k):
        v = A[lo:int(t) + 1, j]
        rr = Rmat[lo:int(t) + 1, j]
        ok = np.isfinite(v) & (v > 0.0) & np.isfinite(rr)
        if int(ok.sum()) >= int(mc):
            with np.errstate(all="ignore"):
                out[j] = float((np.abs(rr[ok]) / v[ok]).mean()) * 1e8
    return out


def _nearest_idx(dates, d):
    i = int(dates.searchsorted(pd.Timestamp(d)))
    i = min(max(i, 0), len(dates) - 1)
    if i > 0 and abs(dates[i] - pd.Timestamp(d)) > abs(dates[i - 1] - pd.Timestamp(d)):
        i -= 1
    return i


def main():
    ap = argparse.ArgumentParser(description="factor_zoo/liquidity.py 独立复核")
    ap.add_argument("--step", type=int, default=1, help="全局扫描步长（交易日）")
    ap.add_argument("--skip-scan", action="store_true", help="跳过全局 inf 扫描")
    ap.add_argument("--lag", type=int, default=135, help="端点扰动扫描最大 lag")
    args = ap.parse_args()

    md = MarketData(verbose=False)
    C = Ctx(md)
    A, NT, R = C.A, C.NT, C.R
    facts = LQ.FACTORS
    names = list(facts)
    rows_txt = []
    def say(s=""):
        print(s)
        rows_txt.append(str(s))

    say("=" * 110)
    say("被审文件: %s" % LQ.__file__)
    say("面板: %d 交易日 x %d 证券 | %s ~ %s | 因子 %d 个"
        % (C.n, C.k, md.dates[0].date(), md.dates[-1].date(), len(names)))
    say("=" * 110)

    # ---------------- 0) 因子清单 ----------------
    say("\n[0] 因子清单（desc 为注册表声称；win 为源码阅读的窗口行数，稍后扰动扫描独立复核）")
    for n in names:
        say("  %-22s win=%-4d dir=%+d  %s" % (n, WIN_ROWS[n], facts[n]["direction"], facts[n]["desc"]))

    # ---------------- A) 6 日独立重算 ----------------
    idxs = [_nearest_idx(md.dates, d) for d in TEST_DATES]
    say("\n[A] 独立重算：逐格对比（%s）"
        % "  ".join("%s(t=%d)" % (md.dates[i].date(), i) for i in idxs))
    t0 = time.time()
    rows = []
    for n in names:
        spec = facts[n]
        for i in idxs:
            mod = np.asarray(spec["func"](i, md, C), dtype="float64")
            ref = cross_section(n, A, NT, R, i)
            r = cmp_cells(mod, ref)
            r.update(factor=n, date=str(md.dates[i].date()), t=i)
            rows.append(r)
    D = pd.DataFrame(rows)
    piv = D.pivot(index="factor", columns="date", values="max_abs_diff").reindex(names)
    S = D.groupby("factor").agg(
        max_diff=("max_abs_diff", "max"),
        nan_mismatch=("nan_mismatch", "sum"),
        big_diff=("big_diff", "sum"),
        min_pairs=("finite_pairs", "min"),
        mod_inf=("mod_inf", "sum"),
    ).reindex(names)
    say("\n  max|diff|（行=因子，列=日期，空格=NaN）:")
    say(piv.to_string(float_format=lambda x: "%.3e" % x))
    say("\n  汇总（6 日合计）:")
    say(S.to_string(float_format=lambda x: "%.3e" % x))
    say("  用时 %.1fs" % (time.time() - t0))

    # A1.5) 随机日期复核
    n_extra = int(os.environ.get("NAIVE_EXTRA_DATES", "8"))
    if n_extra > 0:
        rng = np.random.default_rng(20261016)
        ex = sorted(rng.choice(np.arange(150, C.n), size=n_extra, replace=False).tolist())
        say("\n[A1.5] 随机日期复核（%s）" % ", ".join(str(md.dates[i].date()) for i in ex))
        agg = []
        for n in names:
            spec = facts[n]
            mxd = 0.0
            mis = 0
            big = 0
            for i in ex:
                mod = np.asarray(spec["func"](i, md, C), dtype="float64")
                ref = cross_section(n, A, NT, R, i)
                r = cmp_cells(mod, ref)
                mxd = max(mxd, r["max_abs_diff"])
                mis += r["nan_mismatch"]
                big += r["big_diff"]
            agg.append(dict(factor=n, max_diff=mxd, nan_mismatch=mis, big_diff=big))
        AG = pd.DataFrame(agg).set_index("factor")
        say(AG.to_string(float_format=lambda x: "%.3e" % x))

    # A2) skew 归一化专项
    say("\n[A2] lq_trade_size_skew60 归一化专项（模块 = 教科书三阶矩偏度 × sqrt((n-1)/n)）")
    say("     %-12s %14s %14s %14s" % ("date", "max|mod-pop|", "max|mod-sty|", "max rel dev"))
    for i in idxs:
        mod = np.asarray(facts["lq_trade_size_skew60"]["func"](i, md, C), dtype="float64")
        pop = cross_section("lq_trade_size_skew60", A, NT, R, i)
        sty = cross_section("lq_trade_size_skew60", A, NT, R, i, alt=True)
        m = np.isfinite(mod)
        d_pop = float(np.nanmax(np.abs(mod[m] - pop[m]))) if m.any() else float("nan")
        d_sty = float(np.nanmax(np.abs(mod[m] - sty[m]))) if m.any() else float("nan")
        both = m & np.isfinite(pop) & (np.abs(pop) > 1e-6)
        rel = float(np.max(np.abs(mod[both] / pop[both] - 1.0))) if both.any() else float("nan")
        say("     %-12s %14.3e %14.3e %14.3e"
            % (md.dates[i].date(), d_pop, d_sty, rel))

    # A3) rel20_60 分母含 t 的影响
    say("\n[A3] lq_amt_rel20_60 分母含 t 的影响（对照：分母改为 [t-60, t-1]）")
    say("     %-12s %10s %14s %10s %10s" % ("date", "changed", "max|diff|", "pearson", "spearman"))
    for i in idxs:
        mod = np.asarray(facts["lq_amt_rel20_60"]["func"](i, md, C), dtype="float64")
        cln = cross_section("lq_amt_rel20_60", A, NT, R, i, alt=True)
        chg = changed_count(mod, cln)
        both = np.isfinite(mod) & np.isfinite(cln)
        mx = float(np.max(np.abs(mod[both] - cln[both]))) if both.any() else float("nan")
        pe = float(np.corrcoef(mod[both], cln[both])[0, 1]) if both.sum() > 2 else float("nan")
        sp = float(pd.Series(mod[both]).corr(pd.Series(cln[both]), method="spearman")) if both.sum() > 2 else float("nan")
        say("     %-12s %10d %14.4f %10.4f %10.4f" % (md.dates[i].date(), chg, mx, pe, sp))

    # ---------------- C) 端点扰动扫描 ----------------
    say("\n[C] 端点扰动扫描（t=%d, %s；逐 lag 改写一行 A/NT/R，实测最长生效滞后）"
        % (idxs[-1], md.dates[idxs[-1]].date()))
    t0 = time.time()
    say("     %-22s %8s %8s %8s %10s   %s"
        % ("factor", "win_src", "win_emp", "hit@w-1", "hit@w", "判定"))
    for n in names:
        r = endpoint_sweep(n, facts[n], C, idxs[-1], maxlag=args.lag)
        h = r["lag_hits"]
        w = r["window_rows"]
        a = h[w - 1] if 0 <= w - 1 < len(h) else None
        b = h[w] if 0 <= w < len(h) else None
        ok = "OK" if w == WIN_ROWS[n] else "!!! 与源码阅读不一致"
        say("     %-22s %8d %8d %8d %10s   %s"
            % (n, WIN_ROWS[n], w, -1 if a is None else a, "-" if b is None else b, ok))
    say("     用时 %.1fs" % (time.time() - t0))

    # ---------------- D) 未来扰动检验 ----------------
    say("\n[D] 未来扰动检验：改写 t+1..t+30 的 A/NT/R/C，输出必须不变")
    say("     %-22s %10s %10s" % ("factor", "t=%d" % idxs[1], "t=%d" % idxs[-1]))
    for n in names:
        c1 = future_test(n, facts[n], C, idxs[1])
        c2 = future_test(n, facts[n], C, idxs[-1])
        say("     %-22s %10d %10d" % (n, c1, c2))

    # ---------------- E) 停牌污染量化 ----------------
    say("\n[E] 停牌/零值污染量化")
    zA = (A == 0)
    zNT = (NT == 0)
    say("     面板总量: amt==0 共 %d 格（%.1f%% of listed），涉及 %d 只股票、%d 个交易日；"
        "num_trades==0 共 %d 格；两者同时为 0: %d 格"
        % (int(zA.sum()), 100.0 * zA.sum() / max(md.listed.values.sum(), 1),
           int(zA.any(0).sum()), int(zA.any(1).sum()), int(zNT.sum()), int((zA & zNT).sum())))
    say("     注：amt==0 且 listed=True 即框架定义的停牌格 = %d" % int((zA & md.listed.values).sum()))
    say("     (a) 停牌行是否被排除：模块 vs 把 amt==0 / num_trades==0 当有效观测的朴素实现")
    say("         零格=窗口内零格数; 涉零=窗口内有零格的股票数; 变了=因子值变化的股票数(6日最大);")
    say("         变NaN/变inf=因零值污染由(有值/缺失)变为 NaN / ±inf 的股票数; NaN位=NaN 位置变化数; max|diff| 只统计双方有限格")
    say("         %-22s %8s %8s %8s %8s %8s %8s %10s"
        % ("factor", "零格", "涉零", "变了", "变NaN", "变inf", "NaN位", "max|diff|"))
    cont_rows = []
    ED = []
    for n in names:
        w = WIN_ROWS[n]
        src = NT if n.startswith("lq_nt") else A
        for i in idxs:
            lo = max(i - w + 1, 0)
            sub = src[lo:i + 1]
            zc = int((sub == 0).sum())
            stk = int(((sub == 0).any(0)).sum())
            mod = np.asarray(facts[n]["func"](i, md, C), dtype="float64")
            bad = cross_section(n, A, NT, R, i, bad=True,
                                alt=(n == "lq_trade_size_skew60"))
            chg = changed_count(mod, bad)
            n_bad_nan = int((np.isnan(bad) & np.isfinite(mod)).sum())
            n_bad_inf = int((np.isinf(bad) & ~np.isinf(mod)).sum())
            n_nanmis = int((np.isnan(mod) != np.isnan(bad)).sum())
            both = np.isfinite(mod) & np.isfinite(bad)
            mx = float(np.max(np.abs(mod[both] - bad[both]))) if both.any() else 0.0
            ED.append(dict(factor=n, date=str(md.dates[i].date()), zero_cells=zc,
                           stocks_with_zero=stk, stocks_changed=chg,
                           bad_nan=n_bad_nan, bad_inf=n_bad_inf,
                           nan_pos_changed=n_nanmis, max_abs_diff_finite=mx))
        d = pd.DataFrame([r for r in ED if r["factor"] == n])
        cont_rows.append(dict(factor=n, zero_cells=int(d.zero_cells.max()),
                              stocks_with_zero=int(d.stocks_with_zero.max()),
                              stocks_changed=int(d.stocks_changed.max()),
                              bad_nan_max=int(d.bad_nan.max()),
                              bad_inf_max=int(d.bad_inf.max()),
                              bad_nan_last=int(d.bad_nan.iloc[-1]),
                              bad_inf_last=int(d.bad_inf.iloc[-1]),
                              nan_pos_max=int(d.nan_pos_changed.max()),
                              max_abs_diff=float(d.max_abs_diff_finite.max())))
        say("         %-22s %8d %8d %8d %8d %8d %8d %10.4g"
            % (n, int(d.zero_cells.max()), int(d.stocks_with_zero.max()),
               int(d.stocks_changed.max()), int(d.bad_nan.max()),
               int(d.bad_inf.max()), int(d.nan_pos_changed.max()),
               float(d.max_abs_diff_finite.max())))
    E_DET = pd.DataFrame(ED)
    CONT = pd.DataFrame(cont_rows).set_index("factor")

    say("     (b) 停牌当天（A[t]==0）模块取值：以 %s 为例（%d 只当日停牌）"
        % (md.dates[idxs[-1]].date(), int(zA[idxs[-1]].sum())))
    i = idxs[-1]
    susp = zA[i]
    say("         %-22s %10s %10s" % ("factor", "当日停牌数", "其中非NaN"))
    for n in names:
        v = np.asarray(facts[n]["func"](i, md, C), dtype="float64")
        say("         %-22s %10d %10d" % (n, int(susp.sum()), int(np.isfinite(v[susp]).sum())))

    # ---------------- F) 专项 ----------------
    say("\n[F1] Amihud 分子：复权收益(R) vs 未复权收益(ret_raw)")
    raw = np.asarray(md.ret_raw.values, dtype="float64")
    say("     %-12s %10s %14s %10s %10s" % ("date", "changed", "max|diff|", "adj_nan", "raw_nan"))
    for i in (idxs[1], idxs[3], idxs[-1]):
        for n, w, mc in (("lq_amihud20", 20, 12), ("lq_amihud60", 60, 36)):
            mod = np.asarray(facts[n]["func"](i, md, C), dtype="float64")
            adj = amihud_with_returns(A, R, i, w, mc)
            rw = amihud_with_returns(A, raw, i, w, mc)
            chg = changed_count(adj, rw)
            both = np.isfinite(adj) & np.isfinite(rw)
            mx = float(np.max(np.abs(adj[both] - rw[both]))) if both.any() else float("nan")
            say("     %-12s %10d %14.4g %10d %10d  [%s]"
                % (md.dates[i].date(), chg, mx, int(np.isnan(mod).sum()), int(np.isnan(rw).sum()), n))
    say("     模块输出是否正确使用复权口径: %s"
        % ("是" if all(cmp_cells(np.asarray(facts[n]['func'](i, md, C)),
                                 amihud_with_returns(A, R, i, w, mc))['max_abs_diff'] < 1e-9
                       for i in idxs for n, w, mc in (("lq_amihud20", 20, 12), ("lq_amihud60", 60, 36)))
           else "否"))

    say("\n[F2] 成交额单位自检（amt/num_trades 单笔均额应落在 A 股合理量级）")
    fin = np.isfinite(A) & np.isfinite(NT) & (NT > 0) & (A > 0)
    per = A[fin] / NT[fin]
    say("     amt 分位(元): p10=%.3g p50=%.3g p90=%.3g" % tuple(np.percentile(A[fin], [10, 50, 90])))
    say("     单笔均额 amt/NT 分位(元): p10=%.3g p50=%.3g p90=%.3g" % tuple(np.percentile(per, [10, 50, 90])))
    say("     Amihud 因子中位量级: " + ", ".join(
        "%s=%.3g" % (n, float(np.nanmedian(np.asarray(facts[n]["func"](idxs[-1], md, C), dtype="float64"))))
        for n in ("lq_amihud20", "lq_amihud60")))

    say("\n[F4] lq_trade_size_chg 口径歧义：代码=mean(log sz) 之差；另一读法=log(mean sz) 之差")
    def size_chg_logmean(i):
        k = C.k
        out = np.full(k, np.nan)
        for j in range(k):
            def mm(w, mc):
                va, _ = win(A[:, j], i, w)
                vn, _ = win(NT[:, j], i, w)
                ok = np.isfinite(va) & np.isfinite(vn) & (va > 0.0) & (vn > 0.0)
                if int(ok.sum()) < int(mc):
                    return np.nan
                return float((va[ok] / vn[ok]).mean())
            s_, l_ = mm(20, 12), mm(120, 60)
            if np.isfinite(s_) and np.isfinite(l_) and s_ > 0.0 and l_ > 0.0:
                out[j] = float(np.log(s_) - np.log(l_))
        return out
    say("     %-12s %10s %14s %10s %10s" % ("date", "changed", "max|diff|", "pearson", "spearman"))
    for i in idxs:
        mod = np.asarray(facts["lq_trade_size_chg"]["func"](i, md, C), dtype="float64")
        alt = size_chg_logmean(i)
        chg = changed_count(mod, alt)
        both = np.isfinite(mod) & np.isfinite(alt)
        mx = float(np.max(np.abs(mod[both] - alt[both]))) if both.any() else float("nan")
        pe = float(np.corrcoef(mod[both], alt[both])[0, 1]) if both.sum() > 2 else float("nan")
        sp = float(pd.Series(mod[both]).corr(pd.Series(alt[both]), method="spearman")) if both.sum() > 2 else float("nan")
        say("     %-12s %10d %14.4f %10.4f %10.4f" % (md.dates[i].date(), chg, mx, pe, sp))

    say("\n[F3] 源码静态扫描（只读）")
    src = open(LQ.__file__, encoding="utf-8").read().splitlines()
    for pat, tag in ((r"int\(t\)\s*\+\s*1", "切片上界（含 t）"),
                     (r"\[1:\]|\[:-1\]", "对已截断窗口切片"),
                     (r"md\.", "直接访问 md")):
        hit = [(i + 1, ln.strip()) for i, ln in enumerate(src) if __import__("re").search(pat, ln)]
        say("     %-18s 命中 %d 处%s" % (tag, len(hit),
                                         (": " + " | ".join("%d:%s" % h for h in hit[:4])) if hit else ""))

    # ---------------- B) 全局扫描 ----------------
    if not args.skip_scan:
        say("\n[B] 全样本扫描（步长 %d；每因子 %d 个日期）" % (args.step, len(range(20, C.n, args.step))))
        t0 = time.time()
        scan = []
        for n in names:
            f = facts[n]["func"]
            n_inf = 0
            n_shape = 0
            n_dtype = 0
            ndays = 0
            nv = []
            for t in range(20, C.n, args.step):
                v = np.asarray(f(t, md, C))
                if v.shape != (C.k,):
                    n_shape += 1
                    continue
                if v.dtype != np.float64:
                    n_dtype += 1
                if np.isinf(v).any():
                    n_inf += 1
                c = int(np.isfinite(v).sum())
                if c > 100:
                    ndays += 1
                nv.append(c)
            scan.append(dict(factor=n, days=len(nv), cov=float(np.mean(nv)) / C.k,
                             inf_days=n_inf, bad_shape=n_shape, bad_dtype=n_dtype,
                             valid_days=ndays))
        SCAN = pd.DataFrame(scan).set_index("factor")
        say(SCAN.to_string(float_format=lambda x: "%.4f" % x))
        say("     0 个 inf 格: %s；用时 %.1fs"
            % ("是" if int(SCAN["inf_days"].sum()) == 0 else "否", time.time() - t0))
    else:
        SCAN = pd.DataFrame()

    # ---------------- 保存 ----------------
    D.to_csv(os.path.join(HERE, "liquidity_naive_detail.csv"), index=False, encoding="utf-8-sig")
    CONT.to_csv(os.path.join(HERE, "liquidity_zero_impact.csv"), encoding="utf-8-sig")
    E_DET.to_csv(os.path.join(HERE, "liquidity_zero_impact_detail.csv"), index=False, encoding="utf-8-sig")
    if len(SCAN):
        SCAN.to_csv(os.path.join(HERE, "liquidity_scan.csv"), encoding="utf-8-sig")
    say("\n明细已保存: liquidity_naive_detail.csv / liquidity_zero_impact.csv / "
        "liquidity_zero_impact_detail.csv / liquidity_scan.csv")
    return S, CONT, SCAN


if __name__ == "__main__":
    main()
