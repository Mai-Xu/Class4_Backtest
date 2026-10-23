# -*- coding: utf-8 -*-
"""资金流 / 量价关系 / 强弱资金因子（前缀 fl2_）。

本模块严格只使用 ``Ctx`` 中下标 <= t 的数据；窗口统计全部使用 numpy 矩阵运算，
不做逐股票 Python 循环。数据没有成交量，因此经典量价指标中的 volume 一律用
成交额 ``A`` 替代；VWAP、MFI、CMF 的典型价一律使用复权价
``(ADJ_H + ADJ_L + C) / 3``。
"""
from __future__ import annotations

import numpy as np

from .ctx import Ctx, cs_zscore


# --------------------------------------------------------------------------- #
# 基础工具
def _fin(x) -> np.ndarray:
    """出口统一过滤 inf：非有限值返回 NaN。"""
    x = np.asarray(x, dtype="float64")
    return np.where(np.isfinite(x), x, np.nan)


def _safe_div(num, den) -> np.ndarray:
    """安全除法：分母为 0 / 非有限时返回 NaN，绝不产生 inf。"""
    num = np.asarray(num, dtype="float64")
    den = np.asarray(den, dtype="float64")
    num, den = np.broadcast_arrays(num, den)
    out = np.full(num.shape, np.nan, dtype="float64")
    good = np.isfinite(num) & np.isfinite(den) & (den != 0.0)
    np.divide(num, den, out=out, where=good)
    return out


def _min_cnt(w: int) -> int:
    """窗口最低有效样本数：短窗 0.6w（至少 3），长窗 0.6w（至少 20）。"""
    w = int(w)
    if w >= 120:
        return max(20, int(np.ceil(0.6 * w)))
    return max(3, int(np.ceil(0.6 * w)))


def _amt_ok(C: Ctx, t: int, lo: int) -> np.ndarray:
    """窗口内成交额有效（>0 且有限）的掩码；停牌 / 未上市均为 False。"""
    A = C.A[lo:int(t) + 1]
    return C.listed[lo:int(t) + 1] & np.isfinite(A) & (A > 0.0)


def _count(ok) -> np.ndarray:
    return ok.sum(axis=0).astype("float64")


def _sum_ok(X, ok) -> np.ndarray:
    X = np.asarray(X, dtype="float64")
    return np.where(ok, X, 0.0).sum(axis=0)


def _mean_ok(X, ok) -> np.ndarray:
    return _safe_div(_sum_ok(X, ok), _count(ok))


def _corr(x, y, ok, min_cnt: int) -> np.ndarray:
    """掩码 Pearson 相关；有效样本不足或方差为 0 时返回 NaN。"""
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    ok = np.asarray(ok, dtype=bool)
    n = _count(ok)
    mx = _safe_div(_sum_ok(x, ok), n)
    my = _safe_div(_sum_ok(y, ok), n)
    xc = np.where(ok, x - mx, 0.0)
    yc = np.where(ok, y - my, 0.0)
    num = (xc * yc).sum(axis=0)
    den = np.sqrt((xc * xc).sum(axis=0) * (yc * yc).sum(axis=0))
    out = _safe_div(num, den)
    out = np.clip(out, -1.0, 1.0)
    return _fin(np.where(n >= float(min_cnt), out, np.nan))


def _ols_slope(y, x, ok, min_cnt: int) -> np.ndarray:
    """掩码 OLS 斜率：y = a + b*x，返回 b。"""
    y = np.asarray(y, dtype="float64")
    x = np.asarray(x, dtype="float64")
    ok = np.asarray(ok, dtype=bool)
    n = _count(ok)
    mx = _safe_div(_sum_ok(x, ok), n)
    my = _safe_div(_sum_ok(y, ok), n)
    dx = np.where(ok, x - mx, 0.0)
    dy = np.where(ok, y - my, 0.0)
    sxx = (dx * dx).sum(axis=0)
    sxy = (dx * dy).sum(axis=0)
    out = _safe_div(sxy, sxx)
    good = (n >= float(min_cnt)) & (sxx > 0.0)
    return _fin(np.where(good, out, np.nan))


def _skew(X, ok, min_cnt: int) -> np.ndarray:
    """掩码偏度（总体矩口径，仅用于横截面排序）。"""
    X = np.asarray(X, dtype="float64")
    ok = np.asarray(ok, dtype=bool)
    n = _count(ok)
    mu = _safe_div(_sum_ok(X, ok), n)
    c = np.where(ok, X - mu, 0.0)
    m2 = (c * c).sum(axis=0) / np.maximum(n, 1.0)
    m3 = (c * c * c).sum(axis=0) / np.maximum(n, 1.0)
    out = _safe_div(m3, m2 * np.sqrt(np.maximum(m2, 0.0)))
    good = (n >= float(min_cnt)) & (m2 > 0.0)
    return _fin(np.where(good, out, np.nan))


def _typical(C: Ctx, lo: int, hi: int):
    """复权典型价 (H+L+C)/3 及其有效掩码。"""
    tp = (C.ADJ_H[lo:hi] + C.ADJ_L[lo:hi] + C.C[lo:hi]) / 3.0
    return tp, np.isfinite(tp) & (tp > 0.0)


def _clv(C: Ctx, lo: int, hi: int):
    """CLV = ((C-L)-(H-C))/(H-L) 及其有效掩码。"""
    H = C.ADJ_H[lo:hi]
    L = C.ADJ_L[lo:hi]
    P = C.C[lo:hi]
    rng = H - L
    clv = _safe_div((P - L) - (H - P), rng)
    clv = np.clip(clv, -1.0, 1.0)
    ok = np.isfinite(H) & np.isfinite(L) & np.isfinite(P) & (rng > 0.0)
    return clv, ok


def _regress_time(Y, ok, min_cnt: int):
    """窗口内 Y 对时间（中心化 0..m-1）的回归斜率。"""
    Y = np.asarray(Y, dtype="float64")
    ok = np.asarray(ok, dtype=bool)
    m = Y.shape[0]
    n = _count(ok)
    x = (np.arange(m, dtype="float64") - 0.5 * (m - 1))[:, None]
    mx = _safe_div(_sum_ok(x, ok), n)
    my = _safe_div(_sum_ok(Y, ok), n)
    dx = np.where(ok, x - mx, 0.0)
    dy = np.where(ok, Y - my, 0.0)
    sxx = (dx * dx).sum(axis=0)
    sxy = (dx * dy).sum(axis=0)
    out = _safe_div(sxy, sxx)
    good = (n >= float(min_cnt)) & (sxx > 0.0)
    return _fin(np.where(good, out, np.nan))


# --------------------------------------------------------------------------- #
# 1) 量价相关 / 成交额加权收益
def _pvcorr(t: int, C: Ctx, w: int) -> np.ndarray:
    """过去 w 个日收益与 Δlog 成交额的相关。"""
    lo = max(int(t) - int(w), 0)
    R = C.R[lo:int(t) + 1]
    A = C.A[lo:int(t) + 1]
    okA = _amt_ok(C, t, lo)
    if R.shape[0] < 2:
        return np.full(C.k, np.nan, dtype="float64")
    L = np.zeros_like(A, dtype="float64")
    np.log(A, out=L, where=okA)
    dA = np.diff(L, axis=0)
    ok = okA[1:] & okA[:-1] & np.isfinite(R[1:])
    return _corr(R[1:], dA, ok, _min_cnt(w))


def _amtmom(t: int, C: Ctx, w: int) -> np.ndarray:
    """过去 w 日 Σ(A*R)/ΣA。"""
    lo = max(int(t) - int(w) + 1, 0)
    R = C.R[lo:int(t) + 1]
    A = C.A[lo:int(t) + 1]
    ok = _amt_ok(C, t, lo) & np.isfinite(R)
    Rs = np.where(ok, R, 0.0)
    As = np.where(ok, A, 0.0)
    out = _safe_div((As * Rs).sum(axis=0), As.sum(axis=0))
    return _fin(np.where(_count(ok) >= _min_cnt(w), out, np.nan))


def _sign_amt_ratio(t: int, C: Ctx, w: int) -> np.ndarray:
    """上涨日成交额 / 总成交额。"""
    lo = max(int(t) - int(w) + 1, 0)
    R = C.R[lo:int(t) + 1]
    A = C.A[lo:int(t) + 1]
    ok = _amt_ok(C, t, lo) & np.isfinite(R)
    up = ok & (R > 0.0)
    den = np.where(ok, A, 0.0).sum(axis=0)
    num = np.where(up, A, 0.0).sum(axis=0)
    out = _safe_div(num, den)
    return _fin(np.where(_count(ok) >= _min_cnt(w), out, np.nan))


def _updn_amt(t: int, C: Ctx, w: int) -> np.ndarray:
    """log(上涨日成交额 / 下跌日成交额)。"""
    lo = max(int(t) - int(w) + 1, 0)
    R = C.R[lo:int(t) + 1]
    A = C.A[lo:int(t) + 1]
    ok = _amt_ok(C, t, lo) & np.isfinite(R)
    up = ok & (R > 0.0)
    dn = ok & (R < 0.0)
    us = np.where(up, A, 0.0).sum(axis=0)
    ds = np.where(dn, A, 0.0).sum(axis=0)
    ratio = _safe_div(us, ds)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(np.where(ratio > 0.0, ratio, 1.0))
    good = (_count(ok) >= _min_cnt(w)) & (us > 0.0) & (ds > 0.0)
    return _fin(np.where(good, out, np.nan))


# --------------------------------------------------------------------------- #
# 2) OBV / MFI / CMF / Force / PVT / EMV
def _obv_slope(t: int, C: Ctx, w: int) -> np.ndarray:
    """OBV 窗口回归斜率 / 日均成交额。"""
    lo = max(int(t) - int(w) + 1, 0)
    R = C.R[lo:int(t) + 1]
    A = C.A[lo:int(t) + 1]
    ok = _amt_ok(C, t, lo) & np.isfinite(R)
    flow = np.where(ok, np.sign(np.where(ok, R, 0.0)) * A, 0.0)
    Y = np.cumsum(flow, axis=0)
    slope = _regress_time(Y, ok, _min_cnt(w))
    meanA = _safe_div(np.where(ok, A, 0.0).sum(axis=0), _count(ok))
    out = _safe_div(slope, meanA)
    return _fin(np.where(np.isfinite(slope) & (meanA > 0.0), out, np.nan))


def _obv_z250(t: int, C: Ctx, w: int = 250) -> np.ndarray:
    """OBV 相对 w 日均值的标准化偏离。"""
    lo = max(int(t) - int(w) + 1, 0)
    R = C.R[lo:int(t) + 1]
    A = C.A[lo:int(t) + 1]
    ok = _amt_ok(C, t, lo) & np.isfinite(R)
    flow = np.zeros_like(A, dtype="float64")
    np.multiply(np.sign(R), A, out=flow, where=ok)
    np.cumsum(flow, axis=0, out=flow)
    n = ok.sum(axis=0).astype("float64")
    okf = ok.astype("float64")
    sy = np.einsum("ij,ij->j", flow, okf)
    y2 = np.einsum("ij,ij->j", flow, flow * okf)
    mu = sy / np.maximum(n, 1.0)
    var = (y2 - 2.0 * mu * sy + n * mu * mu) / np.maximum(n - 1.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (flow[-1] - mu) / np.sqrt(np.maximum(var, 0.0))
    good = (n >= _min_cnt(w)) & (var > 0.0) & ok[-1]
    return _fin(np.where(good, out, np.nan))


def _mfi(t: int, C: Ctx, w: int) -> np.ndarray:
    """MFI = 100 - 100/(1+正资金流/负资金流)，典型价用复权价。"""
    lo = max(int(t) - int(w), 0)
    tp, okp = _typical(C, lo, int(t) + 1)
    A = C.A[lo:int(t) + 1]
    base = okp & _amt_ok(C, t, lo)
    if tp.shape[0] < 2:
        return np.full(C.k, np.nan, dtype="float64")
    d = np.diff(tp, axis=0)
    ok = base[1:] & base[:-1] & np.isfinite(d)
    mf = np.where(ok, tp[1:] * A[1:], 0.0)
    pos = np.where(ok & (d > 0.0), mf, 0.0).sum(axis=0)
    neg = np.where(ok & (d < 0.0), mf, 0.0).sum(axis=0)
    ratio = _safe_div(pos, neg)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100.0 - 100.0 / (1.0 + ratio)
    out = _fin(out)
    out = np.where(np.isfinite(neg) & (neg <= 0.0) & (pos > 0.0), 100.0, out)
    out = np.where(np.isfinite(neg) & (neg <= 0.0) & (pos <= 0.0), np.nan, out)
    out = np.clip(out, 0.0, 100.0)
    return _fin(np.where(_count(ok) >= _min_cnt(w), out, np.nan))


def _cmf(t: int, C: Ctx, w: int) -> np.ndarray:
    """CMF = Σ(CLV*A)/ΣA。"""
    cache = C.__dict__.setdefault("_fl2_cmf_cache", {})
    key = (int(t), int(w))
    hit = cache.get(key)
    if hit is not None:
        return hit
    lo = max(int(t) - int(w) + 1, 0)
    clv, okc = _clv(C, lo, int(t) + 1)
    A = C.A[lo:int(t) + 1]
    ok = okc & _amt_ok(C, t, lo)
    clv_s = np.where(ok, clv, 0.0)
    As = np.where(ok, A, 0.0)
    out = _safe_div((clv_s * As).sum(axis=0), As.sum(axis=0))
    out = _fin(np.where(_count(ok) >= _min_cnt(w), out, np.nan))
    if len(cache) > 512:
        cache.clear()
    cache[key] = out
    return out


def _cmf_trend(t: int, C: Ctx) -> np.ndarray:
    """CMF20 - CMF60。"""
    return _fin(_cmf(t, C, 20) - _cmf(t, C, 60))


def _force(t: int, C: Ctx, w: int) -> np.ndarray:
    """Elder Force Index = mean(A*ΔC)/mean(A)。"""
    lo = max(int(t) - int(w), 0)
    P = C.C[lo:int(t) + 1]
    A = C.A[lo:int(t) + 1]
    okp = np.isfinite(P) & (P > 0.0) & _amt_ok(C, t, lo)
    if P.shape[0] < 2:
        return np.full(C.k, np.nan, dtype="float64")
    dP = np.diff(P, axis=0)
    ok = okp[1:] & okp[:-1] & np.isfinite(dP)
    num = np.where(ok, A[1:] * dP, 0.0).sum(axis=0)
    den = np.where(ok, A[1:], 0.0).sum(axis=0)
    out = _safe_div(num, den)
    return _fin(np.where(_count(ok) >= _min_cnt(w), out, np.nan))


def _pvt_slope(t: int, C: Ctx, w: int) -> np.ndarray:
    """PVT 累计序列的窗口回归斜率。"""
    lo = max(int(t) - int(w) + 1, 0)
    R = C.R[lo:int(t) + 1]
    A = C.A[lo:int(t) + 1]
    ok = _amt_ok(C, t, lo) & np.isfinite(R)
    n = _count(ok)
    meanA = _safe_div(np.where(ok, A, 0.0).sum(axis=0), n)
    denA = np.where(meanA > 0.0, meanA, 1.0)
    inc = np.where(ok, np.where(ok, R, 0.0) * np.where(ok, A, 0.0) / denA[None, :], 0.0)
    Y = np.cumsum(inc, axis=0)
    out = _regress_time(Y, ok, _min_cnt(w))
    return _fin(np.where(meanA > 0.0, out, np.nan))


def _emv(t: int, C: Ctx, w: int) -> np.ndarray:
    """Ease of Movement：mean(Δ(复权(H+L)/2)/(A/1e8))。"""
    lo = max(int(t) - int(w), 0)
    mid = (C.ADJ_H[lo:int(t) + 1] + C.ADJ_L[lo:int(t) + 1]) / 2.0
    A = C.A[lo:int(t) + 1]
    okm = np.isfinite(mid) & (mid > 0.0) & _amt_ok(C, t, lo)
    if mid.shape[0] < 2:
        return np.full(C.k, np.nan, dtype="float64")
    dmid = np.diff(mid, axis=0)
    ok = okm[1:] & okm[:-1] & np.isfinite(dmid)
    emv = np.where(ok, dmid * 1e8 / np.where(A[1:] > 0.0, A[1:], 1.0), 0.0)
    out = _safe_div(emv.sum(axis=0), _count(ok))
    return _fin(np.where(_count(ok) >= _min_cnt(w), out, np.nan))


# --------------------------------------------------------------------------- #
# 3) VWAP 偏离 / 量价背离
def _vwap_dev(t: int, C: Ctx, w: int) -> np.ndarray:
    """C[t] / 窗口 VWAP - 1，VWAP 用复权典型价按成交额加权。"""
    lo = max(int(t) - int(w) + 1, 0)
    tp, okp = _typical(C, lo, int(t) + 1)
    A = C.A[lo:int(t) + 1]
    ok = okp & _amt_ok(C, t, lo)
    num = np.where(ok, tp * A, 0.0).sum(axis=0)
    den = np.where(ok, A, 0.0).sum(axis=0)
    vwap = _safe_div(num, den)
    px = C.C[t]
    out = _safe_div(px, vwap) - 1.0
    good = (_count(ok) >= _min_cnt(w)) & np.isfinite(px) & (px > 0.0) & (vwap > 0.0)
    return _fin(np.where(good, out, np.nan))


def _vwap_mom(t: int, C: Ctx, w: int) -> np.ndarray:
    """近 w 日 VWAP / 前 w 日 VWAP - 1。"""
    if int(t) < 2 * int(w) - 1:
        return np.full(C.k, np.nan, dtype="float64")
    lo = int(t) - 2 * int(w) + 1
    tp, okp = _typical(C, lo, int(t) + 1)
    A = C.A[lo:int(t) + 1]
    ok = okp & _amt_ok(C, t, lo)
    n0 = _count(ok[:w])
    n1 = _count(ok[w:])
    v0 = _safe_div(np.where(ok[:w], tp[:w] * A[:w], 0.0).sum(axis=0),
                   np.where(ok[:w], A[:w], 0.0).sum(axis=0))
    v1 = _safe_div(np.where(ok[w:], tp[w:] * A[w:], 0.0).sum(axis=0),
                   np.where(ok[w:], A[w:], 0.0).sum(axis=0))
    out = _safe_div(v1, v0) - 1.0
    good = (n0 >= _min_cnt(w)) & (n1 >= _min_cnt(w)) & (v0 > 0.0) & (v1 > 0.0)
    return _fin(np.where(good, out, np.nan))


def _pvdiv_z(t: int, C: Ctx, w: int) -> np.ndarray:
    """量价背离：z(近 w 日收益) - z(近 w 日成交额均值 / 前 w 日均值)。"""
    if int(t) < 2 * int(w) - 1:
        return np.full(C.k, np.nan, dtype="float64")
    lo = int(t) - 2 * int(w) + 1
    A = C.A[lo:int(t) + 1]
    okA = _amt_ok(C, t, lo)
    n0 = _count(okA[:w])
    n1 = _count(okA[w:])
    m0 = _safe_div(np.where(okA[:w], A[:w], 0.0).sum(axis=0), n0)
    m1 = _safe_div(np.where(okA[w:], A[w:], 0.0).sum(axis=0), n1)
    ratio = _safe_div(m1, m0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dlog = np.log(np.where(ratio > 0.0, ratio, 1.0))
    ret = _safe_div(C.C[t], C.C[int(t) - int(w)]) - 1.0
    good = (n0 >= _min_cnt(w)) & (n1 >= _min_cnt(w)) & np.isfinite(ret) \
        & np.isfinite(dlog) & (ratio > 0.0)
    ret = np.where(good, ret, np.nan)
    dlog = np.where(good, dlog, np.nan)
    return _fin(cs_zscore(ret) - cs_zscore(dlog))


# --------------------------------------------------------------------------- #
# 4) 大单代理 / 资金流强度
def _smart_stats(t: int, C: Ctx, w: int):
    """缓存 (t,w) 的单笔均额均值与有效计数（只缓存小向量）。"""
    cache = C.__dict__.setdefault("_fl2_smart_cache", {})
    key = (int(t), int(w))
    hit = cache.get(key)
    if hit is not None:
        return hit
    lo = max(int(t) - int(w) + 1, 0)
    A = C.A[lo:int(t) + 1]
    N = C.NT[lo:int(t) + 1]
    ok = _amt_ok(C, t, lo) & np.isfinite(N) & (N > 0.0)
    x = np.zeros_like(A, dtype="float64")
    np.divide(A, N, out=x, where=ok)
    n = ok.sum(axis=0).astype("float64")
    m = _safe_div(x.sum(axis=0), n)
    if len(cache) > 512:
        cache.clear()
    cache[key] = (m, n)
    return m, n


def _smart_base(t: int, C: Ctx, w: int) -> np.ndarray:
    """log(近 w 日均单笔成交额) / log(近 250 日均单笔成交额)。"""
    m_w, n_w = _smart_stats(t, C, w)
    m_250, n_250 = _smart_stats(t, C, 250)
    with np.errstate(divide="ignore", invalid="ignore"):
        s_w = np.log(np.where(m_w > 0.0, m_w, 1.0))
        s_250 = np.log(np.where(m_250 > 0.0, m_250, 1.0))
    out = _safe_div(s_w, s_250)
    good = (n_w >= _min_cnt(w)) & (n_250 >= _min_cnt(250)) \
        & (m_w > 0.0) & (m_250 > 0.0) & (s_250 != 0.0)
    return _fin(np.where(good, out, np.nan))

def _smart(t: int, C: Ctx, w: int) -> np.ndarray:
    return _smart_base(t, C, w)


def _smart_ret20(t: int, C: Ctx) -> np.ndarray:
    """sign(20 日收益) * cs_zscore(20 日大单强度)。"""
    w = 20
    if int(t) < w:
        return np.full(C.k, np.nan, dtype="float64")
    strength = _smart_base(t, C, w)
    ret = _safe_div(C.C[t], C.C[int(t) - w]) - 1.0
    z = cs_zscore(strength)
    out = np.sign(np.where(np.isfinite(ret), ret, 0.0)) * z
    return _fin(np.where(np.isfinite(ret) & np.isfinite(z), out, np.nan))


def _mfl(t: int, C: Ctx, w: int) -> np.ndarray:
    """mean(CLV, w) 资金流强度均值口径。"""
    lo = max(int(t) - int(w) + 1, 0)
    clv, okc = _clv(C, lo, int(t) + 1)
    ok = okc & _amt_ok(C, t, lo)
    out = _safe_div(np.where(ok, clv, 0.0).sum(axis=0), _count(ok))
    return _fin(np.where(_count(ok) >= _min_cnt(w), out, np.nan))


# --------------------------------------------------------------------------- #
# 5) 成交额—收益回归 / 放缩量日 / 分布
def _amt_pairs(t: int, C: Ctx, w: int):
    """返回 (Δlog成交额, 当日收益, 配对掩码)。"""
    lo = max(int(t) - int(w), 0)
    A = C.A[lo:int(t) + 1]
    R = C.R[lo:int(t) + 1]
    okA = _amt_ok(C, t, lo)
    if A.shape[0] < 2:
        z = np.zeros((0, C.k), dtype="float64")
        return z, z, np.zeros((0, C.k), dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.log(np.where(okA, A, 1.0))
        dA = L[1:] - L[:-1]
    ok = okA[1:] & okA[:-1] & np.isfinite(R[1:]) & np.isfinite(dA)
    return dA, R[1:], ok


def _amt_ret_elasticity(t: int, C: Ctx, w: int) -> np.ndarray:
    """收益对 Δlog 成交额的回归斜率。"""
    dA, R, ok = _amt_pairs(t, C, w)
    return _ols_slope(R, dA, ok, _min_cnt(w))


def _amt_price_lead(t: int, C: Ctx, w: int) -> np.ndarray:
    """Δlog 成交额对收益的回归斜率（放量领先度）。"""
    dA, R, ok = _amt_pairs(t, C, w)
    return _ols_slope(dA, R, ok, _min_cnt(w))


def _extreme_ret(t: int, C: Ctx, shock: bool = True, w: int = 20, look: int = 60) -> np.ndarray:
    """放量/缩量日的平均收益；以 t 日为止的 look 日均成交额为阈值。"""
    lo_look = max(int(t) - int(look) + 1, 0)
    A_look = C.A[lo_look:int(t) + 1]
    ok_look = _amt_ok(C, t, lo_look)
    n_look = _count(ok_look)
    mean_look = _safe_div(np.where(ok_look, A_look, 0.0).sum(axis=0), n_look)

    lo = max(int(t) - int(w) + 1, 0)
    A = C.A[lo:int(t) + 1]
    R = C.R[lo:int(t) + 1]
    okt = _amt_ok(C, t, lo) & np.isfinite(R) & (n_look >= _min_cnt(look)) \
        & np.isfinite(mean_look)[None, :] & (mean_look > 0.0)[None, :]
    if shock:
        sel = okt & (A > 1.5 * mean_look[None, :])
    else:
        sel = okt & (A < 0.5 * mean_look[None, :])
    num = np.where(sel, R, 0.0).sum(axis=0)
    cnt = sel.sum(axis=0).astype("float64")
    out = _safe_div(num, cnt)
    good = (_count(okt) >= _min_cnt(w)) & (cnt > 0.0)
    return _fin(np.where(good, out, np.nan))

def _amt_skew(t: int, C: Ctx, w: int) -> np.ndarray:
    """窗口内成交额份额偏度（与成交额原始值偏度等价）。"""
    lo = max(int(t) - int(w) + 1, 0)
    A = C.A[lo:int(t) + 1]
    ok = _amt_ok(C, t, lo)
    return _skew(A, ok, _min_cnt(w))


def _flow_persistence(t: int, C: Ctx, w: int) -> np.ndarray:
    """sign(R)*A 序列的一阶自相关。"""
    lo = max(int(t) - int(w) + 1, 0)
    R = C.R[lo:int(t) + 1]
    A = C.A[lo:int(t) + 1]
    ok = _amt_ok(C, t, lo) & np.isfinite(R)
    x = np.where(ok, np.sign(np.where(ok, R, 0.0)) * A, 0.0)
    okp = ok[1:] & ok[:-1]
    return _corr(x[1:], x[:-1], okp, _min_cnt(max(int(w) - 1, 1)))


def _close_flow(t: int, C: Ctx, w: int) -> np.ndarray:
    """近 w 日 sum(CLV*A)/sum(A)。"""
    return _cmf(t, C, w)


# --------------------------------------------------------------------------- #
FACTORS = {
    # 1) 量价相关 / 成交额加权收益
    "fl2_pvcorr10": dict(
        func=lambda t, md, C: _pvcorr(t, C, 10),
        desc="10日 corr(复权收益, Δlog成交额)，量价配合度", direction=-1),
    "fl2_pvcorr120": dict(
        func=lambda t, md, C: _pvcorr(t, C, 120),
        desc="120日 corr(复权收益, Δlog成交额)，量价配合度", direction=-1),
    "fl2_amtmom10": dict(
        func=lambda t, md, C: _amtmom(t, C, 10),
        desc="10日成交额加权收益 sum(A*R)/sum(A)", direction=-1),
    "fl2_amtmom120": dict(
        func=lambda t, md, C: _amtmom(t, C, 120),
        desc="120日成交额加权收益 sum(A*R)/sum(A)", direction=-1),
    "fl2_sign_amt_ratio20": dict(
        func=lambda t, md, C: _sign_amt_ratio(t, C, 20),
        desc="20日上涨日成交额/总成交额", direction=-1),
    "fl2_sign_amt_ratio60": dict(
        func=lambda t, md, C: _sign_amt_ratio(t, C, 60),
        desc="60日上涨日成交额/总成交额", direction=-1),
    "fl2_updn_amt20": dict(
        func=lambda t, md, C: _updn_amt(t, C, 20),
        desc="20日 log(上涨日成交额/下跌日成交额)", direction=-1),

    # 2) OBV / MFI / CMF / Force / PVT / EMV
    "fl2_obvslp60": dict(
        func=lambda t, md, C: _obv_slope(t, C, 60),
        desc="60日OBV回归斜率/日均成交额（成交额替代成交量）", direction=1),
    "fl2_obv_z250": dict(
        func=lambda t, md, C: _obv_z250(t, C, 250),
        desc="OBV相对250日均值的标准化偏离", direction=1),
    "fl2_mfi7": dict(
        func=lambda t, md, C: _mfi(t, C, 7),
        desc="7日MFI，典型价用复权(H+L+C)/3，成交额替代成交量", direction=-1),
    "fl2_mfi60": dict(
        func=lambda t, md, C: _mfi(t, C, 60),
        desc="60日MFI，典型价用复权(H+L+C)/3，成交额替代成交量", direction=-1),
    "fl2_cmf10": dict(
        func=lambda t, md, C: _cmf(t, C, 10),
        desc="10日CMF=sum(CLV*A)/sum(A)，复权高低收", direction=1),
    "fl2_cmf60": dict(
        func=lambda t, md, C: _cmf(t, C, 60),
        desc="60日CMF=sum(CLV*A)/sum(A)，复权高低收", direction=1),
    "fl2_cmf_trend": dict(
        func=lambda t, md, C: _cmf_trend(t, C),
        desc="CMF20-CMF60：资金流强度变化", direction=1),
    "fl2_force13": dict(
        func=lambda t, md, C: _force(t, C, 13),
        desc="13日Elder Force Index=mean(A*ΔC)/mean(A)", direction=1),
    "fl2_force60": dict(
        func=lambda t, md, C: _force(t, C, 60),
        desc="60日Elder Force Index=mean(A*ΔC)/mean(A)", direction=1),
    "fl2_pvt_slope20": dict(
        func=lambda t, md, C: _pvt_slope(t, C, 20),
        desc="20日PVT累计序列回归斜率（R*A/mean(A)累计）", direction=1),
    "fl2_pvt_slope60": dict(
        func=lambda t, md, C: _pvt_slope(t, C, 60),
        desc="60日PVT累计序列回归斜率（R*A/mean(A)累计）", direction=1),
    "fl2_emv20": dict(
        func=lambda t, md, C: _emv(t, C, 20),
        desc="20日Ease of Movement=mean(Δ(复权(H+L)/2)/(A/1e8))", direction=1),

    # 3) VWAP 偏离 / 量价背离
    "fl2_vwapdev5": dict(
        func=lambda t, md, C: _vwap_dev(t, C, 5),
        desc="收盘价/5日成交额加权复权典型价-1", direction=-1),
    "fl2_vwapdev20": dict(
        func=lambda t, md, C: _vwap_dev(t, C, 20),
        desc="收盘价/20日成交额加权复权典型价-1", direction=-1),
    "fl2_vwapmom20": dict(
        func=lambda t, md, C: _vwap_mom(t, C, 20),
        desc="近20日VWAP/前20日VWAP-1（复权典型价）", direction=-1),
    "fl2_pvdiv10": dict(
        func=lambda t, md, C: _pvdiv_z(t, C, 10),
        desc="10日量价背离 z(ret10)-z(Δlog成交额10)", direction=-1),
    "fl2_pvdiv30": dict(
        func=lambda t, md, C: _pvdiv_z(t, C, 30),
        desc="30日量价背离 z(ret30)-z(Δlog成交额30)", direction=-1),

    # 4) 大单代理 / 资金流强度
    "fl2_smart20": dict(
        func=lambda t, md, C: _smart(t, C, 20),
        desc="log(20日均单笔成交额)/log(250日均单笔成交额)，大单代理", direction=1),
    "fl2_smart60": dict(
        func=lambda t, md, C: _smart(t, C, 60),
        desc="log(60日均单笔成交额)/log(250日均单笔成交额)，大单代理", direction=1),
    "fl2_smart_ret20": dict(
        func=lambda t, md, C: _smart_ret20(t, C),
        desc="sign(20日收益)*cs_zscore(20日大单强度)", direction=1),
    "fl2_mfl20": dict(
        func=lambda t, md, C: _mfl(t, C, 20),
        desc="20日CLV均值：资金流强度（等权口径）", direction=1),
    "fl2_mfl60": dict(
        func=lambda t, md, C: _mfl(t, C, 60),
        desc="60日CLV均值：资金流强度（等权口径）", direction=1),

    # 5) 成交额—收益回归 / 放缩量日 / 分布
    "fl2_amt_ret_elasticity60": dict(
        func=lambda t, md, C: _amt_ret_elasticity(t, C, 60),
        desc="60日收益对Δlog成交额的回归斜率", direction=-1),
    "fl2_amt_price_lead60": dict(
        func=lambda t, md, C: _amt_price_lead(t, C, 60),
        desc="60日Δlog成交额对收益的回归斜率（放量领先度）", direction=1),
    "fl2_ret_on_shock20": dict(
        func=lambda t, md, C: _extreme_ret(t, C, True),
        desc="20日内放量日(A>1.5倍60日均额)的平均收益", direction=-1),
    "fl2_quiet_ret20": dict(
        func=lambda t, md, C: _extreme_ret(t, C, False),
        desc="20日内缩量日(A<0.5倍60日均额)的平均收益", direction=-1),
    "fl2_amt_skew60": dict(
        func=lambda t, md, C: _amt_skew(t, C, 60),
        desc="60日成交额分布偏度（成交额份额）", direction=1),
    "fl2_flow_persistence20": dict(
        func=lambda t, md, C: _flow_persistence(t, C, 20),
        desc="20日sign(R)*A序列一阶自相关", direction=1),
    "fl2_close_flow30": dict(
        func=lambda t, md, C: _close_flow(t, C, 30),
        desc="30日CLV成交额加权均值sum(CLV*A)/sum(A)", direction=1),
}