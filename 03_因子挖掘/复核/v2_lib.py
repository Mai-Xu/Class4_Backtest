# -*- coding: utf-8 -*-
"""v2 独立复核 · 参考实现与工具库（不 import 任何 factor_zoo 因子模块）。

本文件是「另写一套实现」的核心：
- Ref：把 core.lib.MarketData 的 7 个 feather 面板整理成复核方自己的 numpy 数据包；
- monthly_grid / universe_mask / fwd_open_ret：独立重写的网格 / 股票池 / 可成交前视收益；
- spearman / rank_avg / cs_z：独立重写的秩相关、平均秩、2%/98% 去极值 z-score；
- ref_*：按《因子模块开发规范》与各模块 docstring 口径独立重写的参考公式。

所有 ref_* 统一签名 fn(r: Ref, t: int, cols: np.ndarray) -> np.ndarray(len(cols))，
只允许使用 <= t 的行（复核方自己的因果纪律；被测模块同款纪律由投毒测试验证）。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.lib import MarketData  # noqa: E402  数据层，非被测对象

EPS = 1e-12
SQRT252 = float(np.sqrt(252.0))
FOUR_LN2 = 4.0 * float(np.log(2.0))


class Ref:
    """复核方自己的数据包（全部 numpy，float64/bool）。"""

    def __init__(self, md):
        f = np.asarray(md.cumadj_clean.values, dtype="float64")
        self.R = np.asarray(md.ret.values, dtype="float64")
        self.C = np.asarray(md.adj_close.values, dtype="float64")
        self.AO = np.asarray(md.open.values, dtype="float64") * f
        self.AH = np.asarray(md.high.values, dtype="float64") * f
        self.AL = np.asarray(md.low.values, dtype="float64") * f
        self.A = np.asarray(md.amt.values, dtype="float64")
        self.NT = np.asarray(md.num_trades.values, dtype="float64")
        self.LISTED = np.asarray(md.listed.values, dtype=bool)
        self.MKT = np.asarray(md.ret_ok.mean(1).values, dtype="float64")
        self.dates = md.dates
        self.symbols = md.symbols
        self.n, self.k = self.R.shape
        self.md = md


def load_ref(verbose=False):
    md = MarketData(verbose=verbose)
    return Ref(md), md


# ------------------------------------------------------------------ #
# 网格 / 股票池 / 前视收益 / 秩相关
# ------------------------------------------------------------------ #
def monthly_grid(dates, start, end):
    """每月最后一个交易日（独立实现：按年月分组取最后一行）。"""
    d = pd.DatetimeIndex(dates)
    key = np.asarray(d.year) * 100 + np.asarray(d.month)
    out = []
    for kk in pd.unique(key):
        pos = np.flatnonzero(key == kk)
        out.append(int(pos[-1]))
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    return [t for t in sorted(out) if lo <= d[t] <= hi]


def universe_mask(md, ts):
    """动态非 ST 全市场池（默认口径）。"""
    u = md.universe()
    return np.asarray(u.values, dtype=bool)[list(ts)]


def fwd_open_ret(r, ts, h=20):
    """可成交前视收益 ADJ_O[t+1+h] / ADJ_O[t+1] - 1。"""
    ao = r.AO
    out = np.full((len(ts), r.k), np.nan)
    for i, t in enumerate(ts):
        j0, j1 = int(t) + 1, int(t) + 1 + int(h)
        if j1 < ao.shape[0]:
            with np.errstate(invalid="ignore", divide="ignore"):
                out[i] = ao[j1] / ao[j0] - 1.0
    return out


def rank_avg(x):
    x = np.asarray(x, dtype="float64")
    uq, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    start = csum - cnt
    return ((start + csum - 1) / 2.0)[inv]


def spearman(a, b, mask=None, min_n=50):
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    ok = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        ok &= np.asarray(mask, dtype=bool)
    if int(ok.sum()) < int(min_n):
        return float("nan")
    ra = rank_avg(a[ok])
    rb = rank_avg(b[ok])
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def cs_z(x):
    """2%/98% 去极值 + z-score（按 ctx.cs_zscore 公开口径独立重写）。"""
    v = np.asarray(x, dtype="float64")
    ok = np.isfinite(v)
    out = np.full(v.shape, np.nan, dtype="float64")
    if int(ok.sum()) < 3:
        return out
    lo, hi = np.percentile(v[ok], [2.0, 98.0])
    vc = np.clip(v, lo, hi)
    m = float(vc[ok].mean())
    s = float(vc[ok].std())
    if s > 0:
        out[ok] = (vc[ok] - m) / s
    else:
        out[ok] = 0.0
    return out


def minc(w):
    w = int(w)
    return max(20, int(np.ceil(0.6 * w))) if w >= 120 else max(3, int(np.ceil(0.6 * w)))


def _log_pos(x):
    x = np.asarray(x, dtype="float64")
    ok = np.isfinite(x) & (x > 0.0)
    return np.where(ok, np.log(np.where(ok, x, 1.0)), np.nan)


def _fin(x):
    x = np.asarray(x, dtype="float64")
    return np.where(np.isfinite(x), x, np.nan)


def _px_ret_full(r, t, w):
    """C[t]/C[t-w]-1（全横截面）。"""
    out = np.full(r.k, np.nan, dtype="float64")
    i0 = int(t) - int(w)
    if i0 < 0:
        return out
    a = r.C[int(t)]
    b = r.C[i0]
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0.0) & (b > 0.0)
    out[ok] = a[ok] / b[ok] - 1.0
    return out


def _amihud60_full(r, t):
    lo = max(int(t) - 59, 0)
    W = r.R[lo:int(t) + 1]
    A = r.A[lo:int(t) + 1]
    ok = np.isfinite(W) & np.isfinite(A) & (A > 0.0)
    cnt = ok.sum(0).astype("float64")
    X = np.where(ok, np.abs(W), 0.0) / np.where(ok, A, 1.0)
    mu = X.sum(0) / np.maximum(cnt, 1.0)
    return _fin(np.where(cnt >= minc(60), mu * 1e8, np.nan))


# ------------------------------------------------------------------ #
# 参考公式（20 个数值抽检因子）
# ------------------------------------------------------------------ #
def _ret_w(r, t, w, cols):
    i0 = int(t) - int(w)
    if i0 < 0:
        return np.full(len(cols), np.nan)
    a = r.C[int(t), cols]
    b = r.C[i0, cols]
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0.0) & (b > 0.0)
    return np.where(ok, a / np.where(ok, b, 1.0) - 1.0, np.nan)


def ref_rev_ret10(r, t, cols):
    return _ret_w(r, t, 10, cols)


def ref_rev_max1d20(r, t, cols):
    lo = max(int(t) - 19, 0)
    W = r.R[lo:int(t) + 1][:, cols]
    ok = np.isfinite(W)
    cnt = ok.sum(0).astype("float64")
    mx = np.where(ok, W, -np.inf).max(0)
    return _fin(np.where(cnt >= minc(20), mx, np.nan))


def ref_rk2_park60(r, t, cols):
    lo = max(int(t) - 59, 0)
    H = r.AH[lo:int(t) + 1][:, cols]
    L = r.AL[lo:int(t) + 1][:, cols]
    ok = np.isfinite(H) & np.isfinite(L) & (H > 0.0) & (L > 0.0)
    term = np.where(ok, np.log(np.where(ok, H, 1.0) / np.where(ok, L, 1.0)) ** 2, 0.0)
    cnt = ok.sum(0).astype("float64")
    mu = term.sum(0) / np.maximum(cnt, 1.0)
    park = np.sqrt(np.maximum(mu, 0.0) / FOUR_LN2)
    return _fin(np.where(cnt >= minc(60), park, np.nan))


def ref_rk2_var5_60(r, t, cols):
    lo = max(int(t) - 59, 0)
    W = r.R[lo:int(t) + 1][:, cols]
    out = np.full(len(cols), np.nan)
    need = minc(60)
    for i in range(len(cols)):
        v = W[:, i]
        v = v[np.isfinite(v)]
        if v.size >= need:
            out[i] = float(np.quantile(v, 0.05))     # 线性插值口径
    return out


def ref_lq2_logamt60(r, t, cols):
    lo = max(int(t) - 59, 0)
    W = r.A[lo:int(t) + 1][:, cols]
    ok = np.isfinite(W) & (W > 0.0)
    cnt = ok.sum(0).astype("float64")
    mu = np.where(ok, W, 0.0).sum(0) / np.maximum(cnt, 1.0)
    good = (cnt >= minc(60)) & (mu > 0.0)
    return _fin(np.where(good, np.log(np.where(mu > 0.0, mu, 1.0)), np.nan))


def ref_lq2_amihud120(r, t, cols):
    lo = max(int(t) - 119, 0)
    W = r.R[lo:int(t) + 1][:, cols]
    A = r.A[lo:int(t) + 1][:, cols]
    ok = np.isfinite(A) & (A > 0.0) & np.isfinite(W)
    cnt = ok.sum(0).astype("float64")
    val = np.where(ok, np.abs(W), 0.0) / np.where(ok, A, 1.0)
    mu = val.sum(0) / np.maximum(cnt, 1.0)
    return _fin(np.where(cnt >= minc(120), mu * 1e8, np.nan))


def ref_fl2_cmf60(r, t, cols):
    lo = max(int(t) - 59, 0)
    H = r.AH[lo:int(t) + 1][:, cols]
    L = r.AL[lo:int(t) + 1][:, cols]
    P = r.C[lo:int(t) + 1][:, cols]
    A = r.A[lo:int(t) + 1][:, cols]
    rng = H - L
    ok = np.isfinite(H) & np.isfinite(L) & np.isfinite(P) & (rng > 0.0)
    clv = np.clip(np.where(ok, ((P - L) - (H - P)) / np.where(ok, rng, 1.0), np.nan), -1.0, 1.0)
    okA = ok & np.isfinite(A) & (A > 0.0)
    num = np.where(okA, clv * A, 0.0).sum(0)
    den = np.where(okA, A, 0.0).sum(0)
    cnt = okA.sum(0).astype("float64")
    out = np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), np.nan)
    return _fin(np.where(cnt >= minc(60), out, np.nan))


def ref_fl2_vwapdev20(r, t, cols):
    lo = max(int(t) - 19, 0)
    H = r.AH[lo:int(t) + 1][:, cols]
    L = r.AL[lo:int(t) + 1][:, cols]
    P = r.C[lo:int(t) + 1][:, cols]
    A = r.A[lo:int(t) + 1][:, cols]
    tp = (H + L + P) / 3.0
    ok = np.isfinite(tp) & (tp > 0.0) & np.isfinite(A) & (A > 0.0)
    num = np.where(ok, tp * A, 0.0).sum(0)
    den = np.where(ok, A, 0.0).sum(0)
    cnt = ok.sum(0).astype("float64")
    vwap = np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), np.nan)
    px = r.C[int(t), cols]
    out = px / np.where(np.isfinite(vwap) & (vwap > 0.0), vwap, 1.0) - 1.0
    good = (cnt >= minc(20)) & np.isfinite(px) & (px > 0.0) & np.isfinite(vwap) & (vwap > 0.0)
    return _fin(np.where(good, out, np.nan))


def _chip_win(r, t, w, cols):
    lo = max(int(t) - w + 1, 0)
    H = r.AH[lo:int(t) + 1][:, cols]
    L = r.AL[lo:int(t) + 1][:, cols]
    P = r.C[lo:int(t) + 1][:, cols]
    A = r.A[lo:int(t) + 1][:, cols]
    px = r.C[int(t), cols]
    ok = (np.isfinite(H) & np.isfinite(L) & np.isfinite(P) & (H > 0.0) & (L > 0.0) & (P > 0.0)
          & np.isfinite(A) & (A > 0.0) & np.isfinite(px)[None, :] & (px > 0.0)[None, :])
    typ = (H + L + P) / 3.0
    ok = ok & np.isfinite(typ) & (typ > 0.0)
    return ok, typ, A, px


def ref_ch_profit60(r, t, cols):
    ok, typ, A, px = _chip_win(r, t, 60, cols)
    cnt = ok.sum(0).astype("float64")
    tot = np.where(ok, A, 0.0).sum(0)
    prof = np.where(ok & (typ < px[None, :]), A, 0.0).sum(0)
    good = (cnt >= minc(60)) & (tot > 0.0) & np.isfinite(px) & (px > 0.0)
    return _fin(np.where(good, prof / np.where(tot > 0.0, tot, 1.0), np.nan))


def ref_ch_cost_dev120(r, t, cols):
    ok, typ, A, px = _chip_win(r, t, 120, cols)
    cnt = ok.sum(0).astype("float64")
    tot = np.where(ok, A, 0.0).sum(0)
    num = np.where(ok, A * typ, 0.0).sum(0)
    good = (cnt >= minc(120)) & (tot > 0.0) & np.isfinite(px) & (px > 0.0)
    mc_ = np.where(good, num / np.where(tot > 0.0, tot, 1.0), np.nan)
    return _fin(np.where(good, px / np.where(good, mc_, 1.0) - 1.0, np.nan))


def ref_tr2_rsi14b(r, t, cols):
    n = 14
    out = np.full(len(cols), np.nan)
    if int(t) < 1:
        return out
    m = int(min(int(t) + 1, 6 * n + 20))
    lo = int(t) + 1 - m
    P = r.C[lo:int(t) + 1][:, cols]
    alpha = 1.0 / n
    for i in range(len(cols)):
        p = P[:, i]
        if p.shape[0] < n + 1:
            continue
        ok = np.isfinite(p[1:]) & np.isfinite(p[:-1])
        d = np.where(ok, p[1:] - p[:-1], 0.0)
        gain = np.where(ok & (d > 0.0), d, 0.0)
        loss = np.where(ok & (d < 0.0), -d, 0.0)
        age = np.arange(d.shape[0] - 1, -1, -1, dtype="float64")
        wgt = alpha * (1.0 - alpha) ** age
        okf = ok.astype("float64")
        den = float(wgt @ okf)
        cnt = float(okf.sum())
        if not (cnt >= max(n, minc(n)) and den >= 0.5):
            continue
        ag = float(wgt @ gain) / den
        al = float(wgt @ loss) / den
        if al <= 0.0 and ag > 0.0:
            out[i] = 100.0
        elif al > 0.0:
            out[i] = 100.0 - 100.0 / (1.0 + ag / al)
        else:
            out[i] = 50.0
    return out


def ref_tr2_di_plus14(r, t, cols):
    n = 14
    E = 6 * n + 5
    W0 = 6 * n
    out = np.full(len(cols), np.nan)
    m = int(min(int(t) + 1, E + W0))
    if m < E + 1:
        return out
    lo = int(t) + 1 - m
    H = r.AH[lo:int(t) + 1][:, cols]
    L = r.AL[lo:int(t) + 1][:, cols]
    P = r.C[lo:int(t) + 1][:, cols]
    alpha = 1.0 / n
    for i in range(len(cols)):
        h, l, pc = H[:, i], L[:, i], P[:, i]
        if h.shape[0] < 2:
            continue
        tr = np.maximum(np.maximum(h[1:] - l[1:], np.abs(h[1:] - pc[:-1])), np.abs(l[1:] - pc[:-1]))
        up = h[1:] - h[:-1]
        dn = l[:-1] - l[1:]
        ok = np.isfinite(tr + up + dn)
        pdm = np.where(ok & (up > dn) & (up > 0.0), up, 0.0)
        ndm = np.where(ok & (dn > up) & (dn > 0.0), dn, 0.0)
        trf = np.where(ok, tr, 0.0)
        Rr = trf.shape[0]
        if Rr < E + 1:
            continue
        j0 = Rr - E
        age = np.arange(j0, -1, -1, dtype="float64")
        wgt = alpha * (1.0 - alpha) ** age
        okt = ok[: j0 + 1].astype("float64")
        den = float(wgt @ okt)
        den = den if den > 0.0 else 1.0
        wtr = float(wgt @ trf[: j0 + 1]) / den
        wdp = float(wgt @ pdm[: j0 + 1]) / den
        wdn = float(wgt @ ndm[: j0 + 1]) / den
        for j in range(j0 + 1, Rr):
            if ok[j]:
                wtr = wtr + alpha * (trf[j] - wtr)
                wdp = wdp + alpha * (pdm[j] - wdp)
                wdn = wdn + alpha * (ndm[j] - wdn)
        cnt = float(ok.sum())
        if wtr > 0.0 and cnt >= max(n + 1, int(0.6 * 6 * n)):
            out[i] = 100.0 * wdp / wtr
    return out


def ref_pt_body1(r, t, cols):
    O = r.AO[int(t), cols]
    H = r.AH[int(t), cols]
    L = r.AL[int(t), cols]
    P = r.C[int(t), cols]
    num = P - O
    rng = H - L
    good = np.isfinite(num) & np.isfinite(rng) & (rng > EPS)
    return _fin(np.where(good, num / np.where(good, rng, 1.0), np.nan))


def ref_pt_up_days20(r, t, cols):
    lo = max(int(t) - 19, 0)
    W = r.R[lo:int(t) + 1][:, cols]
    A = r.A[lo:int(t) + 1][:, cols]
    ok = np.isfinite(W) & np.isfinite(A) & (A > 0.0)
    cnt = ok.sum(0).astype("float64")
    hit = ok & (W > 0.0)
    out = hit.sum(0).astype("float64") / np.maximum(cnt, 1.0)
    return _fin(np.where(cnt >= minc(20), out, np.nan))


def ref_cal_tom_x_rev5(r, t, cols):
    ts = r.dates[int(t)]
    state = 1.0 if (int(ts.day) <= 5 or int(ts.day) >= 25) else 0.0
    base = -_ret_w(r, t, 5, cols)
    v = np.where(state != 0.0, state * base, 0.0)
    return _fin(np.where(r.LISTED[int(t), cols], v, np.nan))


def ref_cal_seasonal_mom(r, t, cols):
    i1, i0 = int(t) - 252, int(t) - 273
    out = np.full(len(cols), np.nan)
    if i0 < 0:
        return out
    a = r.C[i1, cols]
    b = r.C[i0, cols]
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0.0) & (b > 0.0)
    v = np.where(ok, a / np.where(ok, b, 1.0) - 1.0, np.nan)
    return _fin(np.where(r.LISTED[int(t), cols], v, np.nan))


def ref_ix_consensus_rev(r, t, cols):
    z1 = cs_z(-_px_ret_full(r, t, 1))
    z5 = cs_z(-_px_ret_full(r, t, 5))
    z20 = cs_z(-_px_ret_full(r, t, 20))
    M = np.vstack([z1, z5, z20])
    ok = np.isfinite(M)
    cnt = ok.sum(0).astype("float64")
    s = np.where(ok, M, 0.0).sum(0)
    v = _fin(np.where(cnt >= 2.0, s / np.maximum(cnt, 1.0), np.nan))
    return v[cols]


def ref_ix_rev_x_illiq(r, t, cols):
    zr = cs_z(-_px_ret_full(r, t, 20))
    za = cs_z(_amihud60_full(r, t))
    return _fin(zr * za)[cols]


def ref_mk_rel_ret20(r, t, cols):
    w = 20
    hi = int(t)
    lo = hi - w + 1
    if hi < 0 or lo < 0:
        return np.full(len(cols), np.nan)
    W = r.R[lo:hi + 1]
    M = r.MKT[lo:hi + 1]
    ok = np.isfinite(W) & np.isfinite(M)[:, None]
    cnt = ok.sum(0).astype("float64")
    ci = np.prod(np.where(ok, 1.0 + W, 1.0), axis=0) - 1.0
    cm = np.prod(np.where(ok, 1.0 + M[:, None], 1.0), axis=0) - 1.0
    v = _fin(np.where(cnt >= minc(w), ci - cm, np.nan))
    return v[cols]


def ref_mk_rel_mom12_1(r, t, cols):
    w, skip = 231, 21
    hi = int(t) - skip
    lo = hi - w + 1
    if hi < 0 or lo < 0:
        return np.full(len(cols), np.nan)
    W = r.R[lo:hi + 1]
    M = r.MKT[lo:hi + 1]
    ok = np.isfinite(W) & np.isfinite(M)[:, None]
    cnt = ok.sum(0).astype("float64")
    ci = np.prod(np.where(ok, 1.0 + W, 1.0), axis=0) - 1.0
    cm = np.prod(np.where(ok, 1.0 + M[:, None], 1.0), axis=0) - 1.0
    v = _fin(np.where(cnt >= minc(w), ci - cm, np.nan))
    return v[cols]


#: 数值抽检：因子名 -> (参考函数, 说明)
REF_FACTORS = {
    "rev_ret10": (ref_rev_ret10, "C[t]/C[t-10]-1"),
    "rev_max1d20": (ref_rev_max1d20, "近20日最大单日复权收益"),
    "rk2_park60": (ref_rk2_park60, "Parkinson 60日"),
    "rk2_var5_60": (ref_rk2_var5_60, "60日收益5%分位"),
    "lq2_logamt60": (ref_lq2_logamt60, "log(60日均成交额)"),
    "lq2_amihud120": (ref_lq2_amihud120, "mean(|R|/A,120)*1e8"),
    "fl2_cmf60": (ref_fl2_cmf60, "CMF60"),
    "fl2_vwapdev20": (ref_fl2_vwapdev20, "C/20日VWAP-1"),
    "ch_profit60": (ref_ch_profit60, "60日获利盘比例"),
    "ch_cost_dev120": (ref_ch_cost_dev120, "C/120日加权平均成本-1"),
    "tr2_rsi14b": (ref_tr2_rsi14b, "Wilder RSI14"),
    "tr2_di_plus14": (ref_tr2_di_plus14, "Wilder +DI14"),
    "pt_body1": (ref_pt_body1, "当日实体占比"),
    "pt_up_days20": (ref_pt_up_days20, "20日上涨天数占比"),
    "cal_tom_x_rev5": (ref_cal_tom_x_rev5, "月末状态×(-ret5)"),
    "cal_seasonal_mom": (ref_cal_seasonal_mom, "去年同月21日收益"),
    "ix_consensus_rev": (ref_ix_consensus_rev, "z(-r1)/z(-r5)/z(-r20)均值"),
    "ix_rev_x_illiq": (ref_ix_rev_x_illiq, "z(-r20)*z(amihud60)"),
    "mk_rel_ret20": (ref_mk_rel_ret20, "20日相对累计收益"),
    "mk_rel_mom12_1": (ref_mk_rel_mom12_1, "12-1 相对动量"),
}