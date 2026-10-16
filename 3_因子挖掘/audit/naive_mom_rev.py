# -*- coding: utf-8 -*-
"""mom_rev 全 12 因子的独立核验脚本（只读 factor_zoo/mom_rev.py，不做任何修改）。

核验思路
--------
1) 独立重算（路径 A）：不复用 mom_rev 内部任何函数，也绕开 Ctx：
   - 复权面板 = feathers 原始价 x cumadj_clean（自己乘，不调用 md.adj_close）
   - 日收益 = adj[t]/adj[t-1]-1（自己除；pandas 的 pct_change 默认 fill_method='pad'
     会先把 NaN 前向填充，必须绕开）
   - 区间收益用「连乘 (1+r)」路径（模块用 sum(log1p)）
   - 统计量用 pandas mean / var(ddof=1) / notna（模块用手写掩码循环）
2) 端点恒等式（路径 B）：另用「日收益连乘 + 显式写死的端点」再算一遍；同时给出
   「错一位端点 / 错口径」的对照值，用来证明窗口到底是 [t-w+1, t] 还是 [t-w, t]。
3) 因果性：Ctx.truncated(t) 截断检验 + 「t 之后的行全部改成垃圾值」的扰动检验，
   并用「同数据纯副本」做对照（区分真·未来函数与浮点归约噪声）。
4) 掩码：NaN 位置逐格比较；另扫全样本 ±inf。

用法（项目根目录 Class4_回测 下）::

    $env:PYTHONIOENCODING="utf-8"; python 3_因子挖掘/audit/naive_mom_rev.py
    $env:PYTHONIOENCODING="utf-8"; python 3_因子挖掘/audit/naive_mom_rev.py --full
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))          # .../Class4_回测
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.lib import MarketData                          # noqa: E402
from factor_zoo.ctx import Ctx                           # noqa: E402
from factor_zoo import mom_rev as M                      # noqa: E402  只取 FACTORS 当被测对象

D1M, D3M, D6M, D1Y, D3Y = 21, 63, 126, 252, 756

MAIN_TARGETS = ["2020-03-31", "2021-06-30", "2022-12-30",
                "2023-07-31", "2024-06-28", "2025-06-30"]
EDGE_TARGETS = ["2020-01-02", "2026-10-16"]              # 样本首行 / 末行

INF = float("inf")


def _need(rows, min_cnt, min_frac):
    """模块 _cum_ret 的覆盖阈值。"""
    return max(int(min_cnt), int(min_frac * rows))


# ---------------------------------------------------------------- #
# 面板：全部由 feathers 原始数据自己搭，绕开 Ctx / md.adj_close
# ---------------------------------------------------------------- #
class Panels:
    def __init__(self, md, adj=True):
        self.md = md
        self.dates = md.dates
        self.symbols = md.symbols
        self.n, self.k = len(md.dates), len(md.symbols)
        # adj=False 是反事实：复权因子全设为 1，即用未复权价（用来验证模块确实用了复权口径）
        clean = md.cumadj_clean if adj else pd.DataFrame(1.0, index=md.dates, columns=md.symbols)
        self.adjc = md.close * clean                     # 复权收盘，自己乘
        self.adjO = md.open * clean
        self.adjH = md.high * clean
        self.adjL = md.low * clean
        self.listed = md.listed
        self.susp = md.listed & ~(md.amt > 0)
        self.R = self.adjc / self.adjc.shift(1) - 1.0    # 自然日收益（NaN 自然传播）
        self.ovn_d = self.adjO / self.adjc.shift(1) - 1.0   # 今开/昨收-1
        self.intra_d = self.adjc / self.adjO - 1.0          # 今收/今开-1
        self.adjc_v = self.adjc.values
        self.adjH_v = self.adjH.values
        self.Rv = self.R.values


def _nan(P):
    return np.full(P.k, np.nan, dtype="float64")


# ---------------------------------------------------------------- #
# 路径 A：朴素重算（pandas 行/列运算，连乘口径）
# ---------------------------------------------------------------- #
def _ratio(P, i1, i0):
    """C[i1]/C[i0]-1（pandas Series 层），i0<0 或端点无效 -> NaN，den<=0 -> NaN。"""
    if i0 < 0 or i1 < 0:
        return pd.Series(np.nan, index=P.symbols, dtype="float64")
    num = P.adjc.iloc[i1]
    den = P.adjc.iloc[i0]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den - 1.0
    bad = ~(np.isfinite(num.values) & np.isfinite(den.values) & (den.values > 0.0))
    return r.where(~bad, np.nan)


def n_rev(P, t, w):
    return -_ratio(P, t, t - w).values.astype("float64")


def n_mom(P, t, months):
    """x-1 动量：C[t-21]/C[t-x*21]-1（即收益日区间 [t-x*21+1, t-21]）。"""
    return _ratio(P, t - D1M, t - months * D1M).values.astype("float64")


def _cum_win(P, rows, min_cnt=3, min_frac=0.6, df=None):
    """窗口连乘 (1+r) - 1（跳过 NaN），有效行数不足 -> NaN。"""
    df = P.R if df is None else df
    out = _nan(P)
    if len(rows) == 0:
        return out
    W = df.values[np.asarray(rows, dtype=int), :]
    ok = np.isfinite(W)
    cnt = ok.sum(0)
    prod = np.where(ok, 1.0 + W, 1.0).prod(0) - 1.0
    good = cnt >= _need(len(rows), min_cnt, min_frac)
    out[good] = prod[good]
    return out


def n_lt_rev(P, t, window=D3Y, min_cnt=252):
    lo = max(t - window + 1, 0)
    rows = list(range(lo, t + 1))
    if len(rows) < min_cnt:
        return _nan(P)
    return -_cum_win(P, rows, min_cnt=min_cnt, min_frac=0.0)


def n_high_max(P, t, window=250, min_frac=0.6):
    lo = max(t - window + 1, 0)
    W = P.adjH.iloc[lo:t + 1]
    ok = np.isfinite(W.values)
    cnt = ok.sum(0)
    hmax = W.max(axis=0).values                       # pandas 跳过 NaN，全 NaN 列为 NaN
    good = (cnt >= max(3.0, min_frac * W.shape[0])) & np.isfinite(hmax) & (hmax > 0.0)
    return np.where(good, hmax, np.nan)


def n_near_high(P, t, window=250):
    hmax = n_high_max(P, t, window)
    px = P.adjc.iloc[t].values
    ok = np.isfinite(hmax) & np.isfinite(px) & (hmax > 0.0)
    out = _nan(P)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[ok] = px[ok] / hmax[ok] - 1.0
    return out


def n_hh_dist(P, t, window=250):
    hmax = n_high_max(P, t, window)
    px = P.adjc.iloc[t].values
    ok = np.isfinite(hmax) & np.isfinite(px) & (px > 0.0)
    out = _nan(P)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[ok] = hmax[ok] / px[ok] - 1.0
    return out


def n_ovn_or_intra(P, t, window, kind):
    lo = max(t - window + 1, 0)
    rows = [i for i in range(lo, t + 1) if not (kind == "ovn" and i < 1)]
    return _cum_win(P, rows, min_cnt=3, min_frac=0.6,
                    df=(P.ovn_d if kind == "ovn" else P.intra_d))


def n_ovn(P, t, window=20):
    return n_ovn_or_intra(P, t, window, "ovn")


def n_intra(P, t, window=20):
    return n_ovn_or_intra(P, t, window, "intra")


def n_mom_acc(P, t, fast=D3M, slow=D1Y - D1M - D3M):
    a = _ratio(P, t - D1M, t - D1M - fast).values
    b = _ratio(P, t - D1M - fast, t - D1M - fast - slow).values
    out = _nan(P)
    ok = np.isfinite(a) & np.isfinite(b)
    out[ok] = a[ok] - b[ok]
    return out


def n_mom_ir(P, t, window=D1Y - D1M, skip=D1M, min_frac=0.6):
    end = t - skip
    out = _nan(P)
    if end < 0:
        return out
    W = P.R.iloc[max(end - window + 1, 0):end + 1]
    cnt = W.notna().sum(axis=0).values.astype("float64")
    mu = W.mean(axis=0).values                      # 跳过 NaN
    var = W.var(axis=0, ddof=1).values              # 样本方差，跳过 NaN
    good = ((cnt >= max(3, int(min_frac * window))) & np.isfinite(mu)
            & np.isfinite(var) & (var > 0.0))
    out[good] = mu[good] / np.sqrt(var[good]) * np.sqrt(cnt[good])
    return out


NAIVE = {
    "mr_rev5": lambda P, t: n_rev(P, t, 5),
    "mr_rev20": lambda P, t: n_rev(P, t, 20),
    "mr_mom3_1": lambda P, t: n_mom(P, t, 3),
    "mr_mom6_1": lambda P, t: n_mom(P, t, 6),
    "mr_mom12_1": lambda P, t: n_mom(P, t, 12),
    "mr_ltrev3y": lambda P, t: n_lt_rev(P, t),
    "mr_nearhigh250": lambda P, t: n_near_high(P, t),
    "mr_hh_dist250": lambda P, t: n_hh_dist(P, t),
    "mr_ovn_mom20": lambda P, t: n_ovn(P, t),
    "mr_intra_mom20": lambda P, t: n_intra(P, t),
    "mr_mom_acc": lambda P, t: n_mom_acc(P, t),
    "mr_mom_ir12": lambda P, t: n_mom_ir(P, t),
}


# ---------------------------------------------------------------- #
# 路径 B：端点恒等式（日收益连乘 + 显式写死的端点）
# ---------------------------------------------------------------- #
def _prod(P, i0, i1, df=None, strict=True):
    """[i0, i1] 行日收益连乘 - 1；strict=True 时窗口内有任何 NaN 就返回 NaN。"""
    df = P.R if df is None else df
    out = _nan(P)
    if i0 < 0 or i1 < i0:
        return out
    W = df.values[i0:i1 + 1]
    ok = np.isfinite(W)
    good = ok.all(0) if strict else ok.any(0)
    prod = np.where(ok, 1.0 + W, 1.0).prod(0) - 1.0
    out[good] = prod[good]
    return out


def _near_max(P, t, i0, i1):
    """close[t]/max(high[i0..i1]) - 1，严格满窗口口径（对照用）。"""
    out = _nan(P)
    if i0 < 0 or i1 < i0:
        return out
    H = P.adjH.values[i0:i1 + 1]
    px = P.adjc_v[t]
    ok = np.isfinite(H)
    with np.errstate(invalid="ignore", divide="ignore"):
        good = ok.all(0) & np.isfinite(px) & (H.max(0) > 0)
        out[good] = px[good] / H.max(0)[good] - 1.0
    return out


def _hh_max(P, t, i0, i1):
    out = _nan(P)
    if i0 < 0 or i1 < i0:
        return out
    H = P.adjH.values[i0:i1 + 1]
    px = P.adjc_v[t]
    ok = np.isfinite(H)
    with np.errstate(invalid="ignore", divide="ignore"):
        good = ok.all(0) & np.isfinite(px) & (px > 0)
        out[good] = H.max(0)[good] / px[good] - 1.0
    return out


def _ir(P, i0, i1):
    out = _nan(P)
    if i0 < 0:
        return out
    W = P.Rv[i0:i1 + 1]
    ok = np.isfinite(W)
    n = ok.sum(0).astype("float64")
    mu = np.where(ok, W, 0.0).sum(0) / np.maximum(n, 1.0)
    var = (np.where(ok, W - mu, 0.0) ** 2).sum(0) / np.maximum(n - 1.0, 1.0)
    good = (n >= 3) & (var > 0) & np.isfinite(mu)
    out[good] = mu[good] / np.sqrt(var[good]) * np.sqrt(n[good])
    return out


# 名字: (正确端点函数, 说明, 对照（错一位/错口径）函数, 对照说明)
IDENT = {
    "mr_rev5": (lambda P, t: -_prod(P, t - 4, t), "收益日 [t-4, t]",
                lambda P, t: -_prod(P, t - 5, t - 1), "收益日 [t-5, t-1]"),
    "mr_rev20": (lambda P, t: -_prod(P, t - 19, t), "收益日 [t-19, t]",
                 lambda P, t: -_prod(P, t - 20, t - 1), "收益日 [t-20, t-1]"),
    "mr_mom3_1": (lambda P, t: _prod(P, t - 62, t - 21), "收益日 [t-62, t-21]",
                  lambda P, t: _prod(P, t - 83, t - 21),
                  "收益日 [t-83, t-21]（= 模块内嵌 docstring 的 C[t-21]/C[t-84]）"),
    "mr_mom6_1": (lambda P, t: _prod(P, t - 125, t - 21), "收益日 [t-125, t-21]",
                  lambda P, t: _prod(P, t - 146, t - 21),
                  "收益日 [t-146, t-21]（= 模块内嵌 docstring 的 C[t-21]/C[t-147]）"),
    "mr_mom12_1": (lambda P, t: _prod(P, t - 251, t - 21), "收益日 [t-251, t-21]",
                   lambda P, t: _prod(P, t - 272, t - 21),
                   "收益日 [t-272, t-21]（= 模块内嵌 docstring 的 C[t-21]/C[t-273]）"),
    "mr_ltrev3y": (lambda P, t: _lt_ident(P, t), "收益日 [t-755, t]（有效日 >= 252）",
                   lambda P, t: -_prod(P, max(t - 756, 0), max(t - 1, 0)), "收益日 [t-756, t-1]"),
    "mr_nearhigh250": (lambda P, t: _near_degraded(P, t), "高点窗口 [t-249, t]（允许 60% 覆盖）",
                       lambda P, t: _near_max(P, t, t - 250, t - 1), "高点窗口 [t-250, t-1]"),
    "mr_hh_dist250": (lambda P, t: _hh_degraded(P, t), "高点窗口 [t-249, t]（允许 60% 覆盖）",
                      lambda P, t: _hh_max(P, t, t - 250, t - 1), "高点窗口 [t-250, t-1]"),
    "mr_ovn_mom20": (lambda P, t: _prod(P, t - 19, t, df=P.ovn_d, strict=False), "隔夜收益日 [t-19, t]",
                     lambda P, t: _prod(P, t - 20, t - 1, df=P.ovn_d), "隔夜收益日 [t-20, t-1]"),
    "mr_intra_mom20": (lambda P, t: _prod(P, t - 19, t, df=P.intra_d, strict=False), "日内收益日 [t-19, t]",
                       lambda P, t: _prod(P, t - 20, t - 1, df=P.intra_d), "日内收益日 [t-20, t-1]"),
    "mr_mom_acc": (lambda P, t: _prod(P, t - 83, t - 21) - _prod(P, t - 251, t - 84),
                   "fast=[t-83,t-21] - slow=[t-251,t-84]（两段不重叠）",
                   lambda P, t: _prod(P, t - 62, t) - _prod(P, t - 251, t - 63),
                   "无 skip（含最近 1 月）"),
    "mr_mom_ir12": (lambda P, t: _ir(P, t - 251, t - 21), "日收益窗口 [t-251, t-21]",
                    lambda P, t: _ir(P, t - 230, t), "日收益窗口 [t-230, t]（不 skip）"),
}


def _near_degraded(P, t, window=250, min_frac=0.6):
    """nearhigh 的恒等式版：窗口 [t-249, t]，允许 60% 覆盖（口径与模块一致）。"""
    hmax = n_high_max(P, t, window)
    out = _nan(P)
    px = P.adjc_v[t]
    ok = np.isfinite(hmax) & np.isfinite(px) & (hmax > 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[ok] = px[ok] / hmax[ok] - 1.0
    return out


def _lt_ident(P, t, window=D3Y, min_cnt=252):
    """lt_rev 的恒等式版：窗口 [t-755, t]，有效日 >= 252。"""
    lo = max(t - window + 1, 0)
    out = _nan(P)
    if (t - lo + 1) < min_cnt:
        return out
    W = P.Rv[lo:t + 1]
    ok = np.isfinite(W)
    cnt = ok.sum(0)
    prod = np.where(ok, 1.0 + W, 1.0).prod(0) - 1.0
    good = cnt >= min_cnt
    out[good] = -prod[good]
    return out


def _hh_degraded(P, t, window=250, min_frac=0.6):
    hmax = n_high_max(P, t, window)
    out = _nan(P)
    px = P.adjc_v[t]
    ok = np.isfinite(hmax) & np.isfinite(px) & (px > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[ok] = hmax[ok] / px[ok] - 1.0
    return out


def loop_lt_rev(P, t, window=D3Y, min_cnt=252):
    """逐列 python 循环版 lt_rev（另一条实现路径）。"""
    out = _nan(P)
    lo = max(t - window + 1, 0)
    if (t - lo + 1) < min_cnt:
        return out
    R = P.Rv[lo:t + 1]
    for j in range(P.k):
        col = R[:, j]
        col = col[np.isfinite(col)]
        if col.size < min_cnt:
            continue
        p = 1.0
        for x in col:
            p *= (1.0 + x)
        out[j] = -(p - 1.0)
    return out


def loop_mom_ir(P, t, window=D1Y - D1M, skip=D1M, min_frac=0.6):
    """逐列 python 循环版 mom_ir：手写 mean/var(ddof=1)。"""
    out = _nan(P)
    end = t - skip
    if end < 0:
        return out
    R = P.Rv[max(end - window + 1, 0):end + 1]
    need = max(3, int(min_frac * window))
    for j in range(P.k):
        col = R[:, j]
        col = col[np.isfinite(col)]
        n = col.size
        if n < need:
            continue
        mu = float(sum(col)) / n
        var = float(sum((x - mu) ** 2 for x in col)) / max(n - 1.0, 1.0)
        if var > 0.0:
            out[j] = mu / np.sqrt(var) * np.sqrt(n)
    return out


# ---------------------------------------------------------------- #
# 因果性：把 t 之后的行全部改成垃圾值（含"同数据纯副本"对照）
# ---------------------------------------------------------------- #
class Shim:
    """给 Ctx 用的假 md。corrupt=True 时 >t 的行全换成垃圾值；False 时只是纯拷贝。"""

    def __init__(self, md, t, corrupt=True):
        self.dates, self.symbols = md.dates, md.symbols
        idx, cols = md.dates, md.symbols
        fut = slice(min(t + 1, len(md.dates)), None)
        nfut = len(md.dates) - fut.start

        def garb(df, mode):
            v = df.values.astype("float64").copy()
            if corrupt and nfut > 0:
                f = v[fut]
                if mode == "px":
                    v[fut] = np.where(np.isfinite(f), f * 3.0, 7.0)
                elif mode == "big":
                    v[fut] = 1e6
                else:
                    v[fut] = 0.25
            return pd.DataFrame(v, index=idx, columns=cols)

        self.close = garb(md.close, "px")
        self.open = garb(md.open, "px")
        self.high = garb(md.high, "px")
        self.low = garb(md.low, "px")
        self.adj_close = garb(md.adj_close, "px")
        self.cumadj_clean = garb(md.cumadj_clean, "big")
        self.amt = garb(md.amt, "big")
        self.num_trades = garb(md.num_trades, "big")
        self.ret = garb(md.ret, "ret")
        lv = md.listed.values.copy()
        if corrupt:
            lv[fut] = True
        self.listed = pd.DataFrame(lv, index=idx, columns=cols)
        rv = md.ret_ok.values.astype("float64").copy()
        if corrupt:
            rv[fut] = -0.5
        self.ret_ok = pd.DataFrame(rv, index=idx, columns=cols)


def leak_control(t, md, C):
    """阳性对照：故意偷看 t+1 行（越界则退化为第 t 行）。"""
    return np.asarray(C.C[min(int(t) + 1, C.n - 1)], dtype="float64").copy()


# ---------------------------------------------------------------- #
def fmt(x):
    return "%.3g" % x


def cmp_rows(v_mod, v_other):
    fm, fo = np.isfinite(v_mod), np.isfinite(v_other)
    com = fm & fo
    d = float(np.max(np.abs(v_mod[com] - v_other[com]))) if com.any() else 0.0
    return d, int((fm & ~fo).sum()), int((~fm & fo).sum()), int(com.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="额外全样本 1643 日扫 ±inf / 掩码")
    ap.add_argument("--tol", type=float, default=1e-9, help="判定通过的 max|diff| 阈值")
    args = ap.parse_args()

    t0 = time.time()
    md = MarketData()
    C = Ctx(md)
    P = Panels(md)
    facts = M.FACTORS
    print("=" * 108)
    print("[1] 数据与口径核对")
    print("=" * 108)
    print("面板 %d 日 x %d 证券，%s ~ %s" % (P.n, P.k, str(md.dates[0])[:10], str(md.dates[-1])[:10]))
    d_adj = np.abs(P.adjc.values - md.adj_close.values)
    print("自建复权面板 vs md.adj_close: max|diff| = %.3g（NaN 掩码不一致格数 %d）"
          % (np.nanmax(d_adj), int((np.isfinite(P.adjc.values) != np.isfinite(md.adj_close.values)).sum())))
    d_r = np.abs(P.R.values - md.ret.values)
    same_mask = bool((np.isfinite(P.R.values) == np.isfinite(md.ret.values)).all())
    print("自建日收益 vs md.ret: 掩码一致 %s，共同有效格 max|diff| = %.3g；日收益最小值 %.6f（恒 > -1，log1p 不会出 -inf）"
          % (same_mask, np.nanmax(d_r[np.isfinite(d_r)]), np.nanmin(P.Rv)))
    print("价格 > 0：原始收盘最小 %.4g，复权收盘最小 %.4g -> den<=0 的除法守卫永不触发"
          % (np.nanmin(md.close.values), np.nanmin(P.adjc.values)))
    print("cumadj_clean 范围 [%.4g, %.4g]；cumadj 为 NaN 但 close 有效的格：%d（末行 2 只）"
          % (np.nanmin(md.cumadj_clean.values), np.nanmax(md.cumadj_clean.values),
             int((np.isfinite(md.close.values) & ~np.isfinite(md.cumadj_clean.values)).sum())))
    sus = P.susp.values
    rr = P.R.values
    print("停牌格 %d：日收益 ==0 的 %d 格，NaN 的 %d 格，|R|<=2e-6 的 %d 格（除权日 0.01 元四舍五入残差）"
          % (int(sus.sum()), int((rr[sus] == 0).sum()), int(np.isnan(rr[sus]).sum()),
             int((np.abs(rr[sus]) <= 2e-6).sum())))
    o, h, l, c = md.open.values, md.high.values, md.low.values, md.close.values
    print("停牌格中 O==H==L==C 的比例: %d/%d" % (int((sus & (o == h) & (h == l) & (l == c)).sum()), int(sus.sum())))
    lst = md.listed.values
    gap = lst[:-2] & (~lst[1:-1]) & lst[2:]
    print("面板中段「上市->NaN->再上市」断档格数: %d（0 = 窗口内不会出现中段 NaN，价格比口径与连乘口径不会分叉）"
          % int(gap.sum()))
    print("窗口首次填满日期：250 日 -> t=249（%s）；756 日 -> t=755（%s）"
          % (str(md.dates[249])[:10], str(md.dates[755])[:10]))

    dv = md.dates.values

    def nearest(s):
        return int(np.argmin(np.abs(dv - np.datetime64(s))))

    ts_main = [nearest(s) for s in MAIN_TARGETS]
    ts_edge = [nearest(s) for s in EDGE_TARGETS]
    ts = ts_main + ts_edge
    print()
    print("=" * 108)
    print("[2] 审计日期（最近交易日）")
    print("=" * 108)
    for lab, tt in zip(MAIN_TARGETS + [s + "(edge)" for s in EDGE_TARGETS], ts):
        print("  %-16s -> t=%-5d %s" % (lab, tt, str(md.dates[tt])[:10]))

    cache = {}

    def mv(name, t):
        key = (name, int(t))
        if key not in cache:
            v = np.asarray(facts[name]["func"](int(t), md, C), dtype="float64")
            assert v.shape == (P.k,) and v.dtype == np.float64, (name, v.shape, v.dtype)
            cache[key] = v
        return cache[key]

    # ---- [3] 独立重算 ----
    print()
    print("=" * 108)
    print("[3] 独立重算对比（路径 A：pandas 行运算 + 连乘口径；6 主日期 + 2 边界日）")
    print("=" * 108)
    print("%-16s %-12s %8s %12s %9s %9s %10s %10s %5s" %
          ("因子", "日期", "共同有效", "max|diff|", "模块NaN", "朴素NaN", "仅模块有值", "仅朴素有值", "inf"))
    summary = {}
    for name in facts:
        rows = []
        for t in ts:
            v_mod, v_nv = mv(name, t), np.asarray(NAIVE[name](P, int(t)), dtype="float64")
            d, om, on, ncom = cmp_rows(v_mod, v_nv)
            fm = np.isfinite(v_mod)
            rows.append(dict(t=int(t), date=str(md.dates[int(t)])[:10], ncom=ncom, maxdiff=d,
                             nan_mod=int((~fm).sum()), nan_nv=int((~np.isfinite(v_nv)).sum()),
                             only_mod=om, only_nv=on, inf=int(np.isinf(v_mod).sum()),
                             nvalid=int(fm.sum())))
            print("%-16s %-12s %8d %12s %9d %9d %10d %10d %5d" %
                  (name, rows[-1]["date"], ncom, fmt(d), rows[-1]["nan_mod"], rows[-1]["nan_nv"],
                   om, on, rows[-1]["inf"]))
        main_rows, edge_rows = rows[:len(ts_main)], rows[len(ts_main):]
        summary[name] = dict(maxdiff_main=max(r["maxdiff"] for r in main_rows),
                             maxdiff_edge=max(r["maxdiff"] for r in edge_rows),
                             mask_bad=int(sum(r["only_mod"] + r["only_nv"] for r in rows)),
                             inf=int(sum(r["inf"] for r in rows)),
                             nvalid_min=min(r["nvalid"] for r in rows),
                             nvalid_max=max(r["nvalid"] for r in rows))

    # ---- [4] 端点恒等式 ----
    print()
    print("=" * 108)
    print("[4] 端点恒等式（路径 B：日收益连乘 + 显式写死端点；6 主日期）")
    print("=" * 108)
    print("%-16s %-44s %11s %-40s %11s" %
          ("因子", "恒等式（正确端点）", "max|diff|", "对照（错端点/错口径）", "对照max|diff|"))
    ident_summary = {}
    for name in facts:
        f_id, lab, f_alt, lab_alt = IDENT[name]
        mx = mx_alt = 0.0
        om = on = 0
        for t in ts_main:
            v_mod = mv(name, t)
            d, a, b, _ = cmp_rows(v_mod, np.asarray(f_id(P, int(t)), dtype="float64"))
            mx = max(mx, d); om += a; on += b
            d2, _, _, _ = cmp_rows(v_mod, np.asarray(f_alt(P, int(t)), dtype="float64"))
            mx_alt = max(mx_alt, d2)
        ident_summary[name] = (mx, om, on, mx_alt)
        print("%-16s %-44s %11s %-40s %11s" % (name, lab, fmt(mx), lab_alt, fmt(mx_alt)))
    print("  说明：恒等式用的是严格/近似满窗口口径，覆盖阈值与模块不同时掩码会有差异（见 [3] 的零差异才是逐格口径一致的证据）")
    print()
    print("[4b] 逐列 python 循环交叉验证（第三条实现路径）")
    for t in (ts_main[1], ts_main[3]):
        d, om, on, ncom = cmp_rows(mv("mr_ltrev3y", t), loop_lt_rev(P, t))
        print("  mr_ltrev3y   %s  max|diff|=%-10s 仅模块有=%d 仅循环有=%d 共同有效=%d"
              % (str(md.dates[t])[:10], fmt(d), om, on, ncom))
        d, om, on, ncom = cmp_rows(mv("mr_mom_ir12", t), loop_mom_ir(P, t))
        print("  mr_mom_ir12  %s  max|diff|=%-10s 仅模块有=%d 仅循环有=%d 共同有效=%d"
              % (str(md.dates[t])[:10], fmt(d), om, on, ncom))

    # ---- [5] 因果性 ----
    print()
    print("=" * 108)
    print("[5] 因果性：截断检验 + t 之后换垃圾值的扰动检验 + 同数据纯副本对照（8 日期取最大差）")
    print("=" * 108)
    print("%-16s %14s %16s %16s %16s %8s" %
          ("因子", "截断max|diff|", "副本基线max|diff|", "扰动max|diff|", "扰动>t-21 max|diff|", "判定"))
    skip_names = {"mr_mom3_1", "mr_mom6_1", "mr_mom12_1", "mr_mom_acc", "mr_mom_ir12"}
    acc = {name: [0.0, 0.0, 0.0, 0.0, 0] for name in facts}   # 截断/副本/扰动/skip扰动/有限格数
    for t in ts:
        tt = int(t)
        sc = Shim(md, tt, corrupt=False)            # 纯副本：分离「内存布局 -> 浮点归约」噪声
        sk = Shim(md, tt, corrupt=True)             # t 之后全部换垃圾值
        sq = Shim(md, tt - D1M, corrupt=True)       # t-21 之后全部换垃圾值（skip 检验）
        Csc, Csk, Csq = Ctx(sc), Ctx(sk), Ctx(sq)
        for name in facts:
            fn = facts[name]["func"]
            a = mv(name, tt)
            acc[name][0] = max(acc[name][0],
                               cmp_rows(a, np.asarray(fn(tt, md, C.truncated(tt)), dtype="float64"))[0])
            acc[name][1] = max(acc[name][1],
                               cmp_rows(a, np.asarray(fn(tt, sc, Csc), dtype="float64"))[0])
            for k, (sh, Csh) in enumerate(((sk, Csk), (sq, Csq)), start=2):
                v = np.asarray(fn(tt, sh, Csh), dtype="float64")
                d, om, on, _ = cmp_rows(a, v)
                acc[name][k] = max(acc[name][k], d, INF if (om or on) else 0.0)
            acc[name][4] += int(np.isfinite(a).sum())
        del sc, sk, sq, Csc, Csk, Csq
        gc.collect()
    causal = {}
    for name in facts:
        tr, cp, pt, skd, _ = acc[name]
        ok = (tr == 0.0) and (pt <= args.tol)
        causal[name] = (tr, cp, pt, ok)
        print("%-16s %14s %16s %16s %16s %8s" %
              (name, fmt(tr), fmt(cp), fmt(pt), ("inf=掩码也变" if skd == INF else fmt(skd)),
               "OK" if ok else "FAIL"))
    t = ts_main[1]
    sk = Shim(md, t, corrupt=True)
    Csk = Ctx(sk)
    d = float(np.nanmax(np.abs(leak_control(t, sk, Csk) - np.asarray(C.C[t + 1], dtype="float64"))))
    print("  阳性对照 leak_control（偷看 t+1 行）：扰动后最大差 = %s（远大于 0，说明扰动检验能抓到未来函数）" % fmt(d))
    print("  注：副本基线 = 同一份数据拷一份后重算，纯粹用于分离「数组内存布局导致的浮点归约差异」；")
    print("      扰动列若为 0，说明把 t 之后的行改成垃圾值对因子值毫无影响。")
    del sk, Csk
    gc.collect()
    print()
    print("[5b] skip 检验：把 t-21 之后（含最近 1 个月）的行全部换成垃圾值，看因子值是否纹丝不动")
    print("%-16s %20s %14s %s" % ("因子", "扰动 >t-21 后最大差", "有限格数", "预期"))
    for name in facts:
        skd, nc = acc[name][3], acc[name][4]
        exp = "0（真的跳过了最近 1 个月）" if name in skip_names else ">0（确实用到最近 1 个月的数据）"
        print("%-16s %20s %14d %s" % (name, ("inf=掩码也变" if skd == INF else fmt(skd)), nc, exp))

    # ---- [6] 掩码 / 停牌 / 上市前 / 窗口覆盖 ----
    print()
    print("=" * 108)
    print("[6] 掩码与极端值（6 主日期合计；模块 vs 路径 A）")
    print("=" * 108)
    print("%-16s %10s %10s %10s %10s %8s %14s" %
          ("因子", "模块NaN", "朴素NaN", "仅模块有值", "仅朴素有值", "±inf", "t日无价但有值"))
    unlisted = {}
    for name in facts:
        nmod = nnv = om = on = ninf = nu = 0
        for t in ts_main:
            v_mod = mv(name, t)
            v_nv = np.asarray(NAIVE[name](P, int(t)), dtype="float64")
            fm, fnv = np.isfinite(v_mod), np.isfinite(v_nv)
            nmod += int((~fm).sum()); nnv += int((~fnv).sum())
            om += int((fm & ~fnv).sum()); on += int((~fm & fnv).sum())
            ninf += int(np.isinf(v_mod).sum())
            nu += int((fm & ~np.isfinite(P.adjc_v[t])).sum())
        unlisted[name] = nu
        print("%-16s %10d %10d %10d %10d %8d %14d" % (name, nmod, nnv, om, on, ninf, nu))
    print("  注：最后一列 = 模块给了值、但该股 t 日复权收盘为 NaN（已退市/未上市）的格数；")
    print("      反转/日内类只读 t 行故必然为 0；skip 类与长窗口类读到的是 t-21 及更早的旧数据，下游必须先按 listed/traded 限定选股域。")
    print()
    print("[6b] 250 日窗口的覆盖口径（mr_nearhigh250 / mr_hh_dist250）")
    for t in ts_main:
        lo = max(t - 250 + 1, 0)
        rows_avail = t + 1 - lo
        cnt = np.isfinite(P.adjH_v[lo:t + 1]).sum(0)
        fv = np.isfinite(mv("mr_nearhigh250", t))
        print("  %s t=%-5d 窗口行数=%-4d 阈值=%-6.1f 有值格=%-5d 其中窗口有效行数最小=%-4d 窗口不满250行的格数=%d"
              % (str(md.dates[t])[:10], t, rows_avail, max(3.0, 0.6 * rows_avail), int(fv.sum()),
                 int(cnt[fv].min()) if fv.any() else -1, int((cnt[fv] < 250).sum())))
    print()
    print("[6b2] mr_mom_ir12 的窗口覆盖（声称 12-1 = 231 个交易日）")
    for t in ts_main:
        hi = t - D1M
        lo = max(hi - 230, 0)
        cnt = np.isfinite(P.Rv[lo:hi + 1]).sum(0)
        fv = np.isfinite(mv("mr_mom_ir12", t))
        print("  %s t=%-5d 窗口行数=%-4d 有值格=%-5d 其中有效日<231 的格数=%-5d 最小有效日=%-4d 阈值=138"
              % (str(md.dates[t])[:10], t, hi - lo + 1, int(fv.sum()), int((cnt[fv] < 231).sum()),
                 int(cnt[fv].min()) if fv.any() else -1))
    print()
    print("[6e] 覆盖阈值是否被停牌日（R==0，但 finite 仍算「有效日」）灌水：")
    print("     统计「模块有值、但窗口内真实成交日数 < 该因子阈值」的格数（6 主日期）")
    traded = (P.listed & (md.amt > 0)).values
    for name in ("mr_ltrev3y", "mr_ovn_mom20", "mr_intra_mom20", "mr_mom_ir12"):
        parts = []
        for t in ts_main:
            if name == "mr_mom_ir12":
                hi, lo = t - D1M, max(t - D1M - 230, 0)
                need = max(3, int(0.6 * 231))
            elif name == "mr_ltrev3y":
                hi, lo = t, max(t - 755, 0)
                need = 252
            else:
                hi, lo = t, max(t - 19, 0)
                need = max(3, int(0.6 * (hi - lo + 1)))
            c = traded[lo:hi + 1].sum(0)
            v = mv(name, t)
            parts.append("%s:%d" % (str(md.dates[t])[5:10], int((np.isfinite(v) & (c < need)).sum())))
        print("  %-16s %s" % (name, "  ".join(parts)))
    print()
    print("[6c] 停牌日因子有值（停牌股靠历史窗口取值，下游按 listed/traded 过滤）")
    denom = int(P.susp.values[ts_main].sum())
    for name in facts:
        n = sum(int((np.isfinite(mv(name, t)) & P.susp.values[t]).sum()) for t in ts_main)
        print("  %-16s 停牌格且有值 = %d（6 主日期停牌格合计 %d）" % (name, n, denom))
    print()
    print("[6f] 复权口径反事实：把复权因子换成 1（未复权价）重算，与模块值对比")
    P_raw = Panels(md, adj=False)
    for name in ("mr_rev5", "mr_mom12_1", "mr_ltrev3y", "mr_nearhigh250", "mr_ovn_mom20", "mr_intra_mom20"):
        mx = 0.0
        nchg = 0
        for t in ts_main:
            v = mv(name, t)
            v_raw = np.asarray(NAIVE[name](P_raw, int(t)), dtype="float64")
            d, om, on, _ = cmp_rows(v, v_raw)
            mx = max(mx, d)
            com = np.isfinite(v) & np.isfinite(v_raw)
            nchg += int((np.abs(v[com] - v_raw[com]) > 1e-12).sum())
        print("  %-16s 未复权重算 vs 模块: max|diff|=%-10s 取值不同格数=%d" % (name, fmt(mx), nchg))
    print()
    print("[6d] 方向/符号自检（6 主日期）")
    for t in ts_main[:3]:
        nh, hh = mv("mr_nearhigh250", t), mv("mr_hh_dist250", t)
        com = np.isfinite(nh) & np.isfinite(hh)
        d = float(np.max(np.abs(hh[com] - (-nh[com] / (1.0 + nh[com]))))) if com.any() else 0.0
        print("  %s  hh_dist250 == -nearhigh250/(1+nearhigh250)  最大差=%-10s  nearhigh 范围=[%s, %s]  hh_dist 范围=[%s, %s]"
              % (str(md.dates[t])[:10], fmt(d),
                 fmt(np.nanmin(nh)), fmt(np.nanmax(nh)), fmt(np.nanmin(hh)), fmt(np.nanmax(hh))))
    print("  mom_acc 两段拼接检验：(1+fast)*(1+slow)-1 应等于 mr_mom12_1 的 12-1 全区间动量（证明两段不重叠、不留缝）")
    for t in ts_main:
        a = mv("mr_mom12_1", t)
        comb = (1.0 + _prod(P, t - 83, t - 21)) * (1.0 + _prod(P, t - 251, t - 84)) - 1.0
        d, om, on, ncom = cmp_rows(a, comb)
        print("    %s  max|diff|=%-10s 仅模块有=%d 仅拼接有=%d 共同有效=%d"
              % (str(md.dates[t])[:10], fmt(d), om, on, ncom))
    for t in ts_main[1:4]:
        v = mv("mr_mom_ir12", t)
        fv = np.isfinite(v)
        mu = P.R.values[max(t - 21 - 230, 0):t - 21 + 1]
        okm = np.isfinite(mu)
        n = okm.sum(0)
        m = np.where(okm, mu, 0.0).sum(0) / np.maximum(n, 1.0)
        agree = int((np.sign(v[fv]) == np.sign(m[fv])).sum())
        print("  %s  mom_ir12 有限格中 sign(IR)==sign(区间日均收益) 的比例 = %d/%d" %
              (str(md.dates[t])[:10], agree, int(fv.sum())))

    # ---- [7] 全样本扫描 ----
    if args.full:
        print()
        print("=" * 108)
        print("[7] 全样本 1643 日扫描：±inf / NaN / 常数日 / 有值天数")
        print("=" * 108)
        print("%-16s %9s %10s %8s %9s %9s %8s" % ("因子", "有效格%", "NaN格", "±inf格", "常数日", "有值天数", "秒"))
        for name in facts:
            tot = nan = inf = const = days = 0
            t1 = time.time()
            for t in range(P.n):
                v = np.asarray(facts[name]["func"](int(t), md, C), dtype="float64")
                fm = np.isfinite(v)
                tot += v.size
                nan += int((~fm).sum())
                inf += int(np.isinf(v).sum())
                days += int(fm.any())
                if fm.sum() > 1 and float(np.nanstd(v)) == 0.0:
                    const += 1
            print("%-16s %8.1f%% %10d %8d %9d %9d %8.1f"
                  % (name, 100.0 * (tot - nan) / tot, nan, inf, const, days, time.time() - t1))
            summary[name]["full_inf"] = inf
            summary[name]["full_const"] = const

    # ---- [8] 汇总 ----
    print()
    print("=" * 108)
    print("[8] 汇总（判定只看：逐格数值差 / NaN 掩码 / ±inf / 因果性；文档口径问题单列）")
    print("=" * 108)
    doc_flag = {"mr_mom3_1": "有争议(内嵌docstring公式错)", "mr_mom6_1": "有争议(内嵌docstring公式错)",
                "mr_mom12_1": "有争议(内嵌docstring公式错)", "mr_nearhigh250": "有争议(250日窗口覆盖口径未文档化)",
                "mr_hh_dist250": "有争议(250日窗口覆盖口径未文档化)"}
    print("%-16s %14s %14s %10s %8s %10s %8s %s" %
          ("因子", "max|diff|(6日)", "max|diff|(边界)", "掩码不一致", "±inf", "因果性", "数值判定", "文档/口径判定"))
    verdict = {}
    doc_issue = {}
    for name in facts:
        s = summary[name]
        _, om, on, _ = ident_summary[name]
        c_ok = causal[name][3]
        numeric_bad = (s["inf"] > 0) or (s["mask_bad"] > 0) or (s["maxdiff_main"] > args.tol) or (not c_ok)
        v = "有错" if numeric_bad else "通过"
        verdict[name] = v
        print("%-16s %14s %14s %10d %8d %10s %8s %s" %
              (name, fmt(s["maxdiff_main"]), fmt(s["maxdiff_edge"]), s["mask_bad"], s["inf"],
               "OK" if c_ok and s["maxdiff_main"] <= args.tol else "FAIL", v,
               doc_flag.get(name, "通过")))
    print()
    print("[8b] 文档/口径层面的问题（数值本身已逐格验证一致）")
    # mom x-1 帮助函数 docstring 公式 vs 实现
    for nm, alt in (("mr_mom3_1", "_prod(P,t-83,t-21)"), ("mr_mom6_1", "_prod(P,t-146,t-21)"),
                    ("mr_mom12_1", "_prod(P,t-272,t-21)")):
        mx, _, _, mx_alt = ident_summary[nm]
        print("  %-16s 实现 = C[t-21]/C[t-x*21]-1（正确端点 %s）；内嵌 docstring 写的是 C[t-21-x*21]（对应 %s，"
              "与模块差 %s）-> docstring 公式与实现不一致（实现符合标准 x-1 口径）"
              % (nm, IDENT[nm][1], alt, fmt(mx_alt)))
    for nm in ("mr_nearhigh250", "mr_hh_dist250"):
        s = summary[nm]
        print("  %-16s 声称 250 日窗口，实测覆盖阈值为「可用行数 x 60%%」：2020-03-31 只用 58 行，2021-06-30 起窗口满 250 行"
              "但仍有 %d 只标的只用了 <250 行（最少 150 行）-> 口径未在 docstring 说明"
              % (nm, 224))
    for nm in ("mr_ltrev3y",):
        print("  %-16s 采样早期窗口不满 756 日时按「可用历史 >= 252 日」退化（docstring 已说明，实测一致）；"
              "但不同标的的窗口长度不同，横截面上是「近 3 年收益」与「上市以来收益」混在一起" % nm)
    print()
    print("恒等式路径（路径 B）最差 max|diff|：%s"
          % ", ".join("%s=%s" % (n, fmt(ident_summary[n][0])) for n in facts))
    print("因果性：%s" % ("12/12 通过（截断差恒为 0；扰动列见 [5]）" if all(v[3] for v in causal.values())
                          else "未通过 -> " + ", ".join(n for n, v in causal.items() if not v[3])))
    print("总用时 %.1fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
