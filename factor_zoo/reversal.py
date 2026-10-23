# -*- coding: utf-8 -*-
"""短期反转 / 彩票 / 日内隔夜拆解因子（factor_zoo 分类：reversal）。

本模块实现开发规范 §5.1 的全部 rev_* 因子。所有价格类计算均使用复权口径：
C.C / C.ADJ_O / C.ADJ_H / C.ADJ_L / C.R；成交额类使用 C.A（元）。停牌日
C.R=0、C.A=0，按数据层约定作为 0 收益参与窗口统计；未上市 / 退市位置为 NaN，
无法计算时安静地返回 NaN。

因果性：所有窗口只切 [0, t] 行，所有统计量都用掩码版矩阵运算，不使用任何
t 之后的数据。连续涨跌天数会在 Ctx 实例上缓存一次前缀扫描结果；Ctx.truncated(t)
是全新对象，因此缓存不会破坏注册表的截断因果性检查。
"""
from __future__ import annotations

import numpy as np

from .ctx import Ctx

_EPS = 1e-12


# ------------------------------------------------------------------ #
# 通用工具
# ------------------------------------------------------------------ #
def _nan(C: Ctx) -> np.ndarray:
    """返回一个全 NaN 的横截面向量。"""
    return np.full(int(C.k), np.nan, dtype="float64")


def _fin(x) -> np.ndarray:
    """出口统一把 ±inf / NaN 收敛成 NaN，保证不向注册表泄漏 inf。"""
    x = np.asarray(x, dtype="float64")
    return np.where(np.isfinite(x), x, np.nan)


def _need(window: int) -> int:
    """窗口最少有效样本数：短窗 max(3, ceil(0.6w))，长窗 max(20, ceil(0.6w))。

    对 w<3 的极短窗口，不能要求超过窗口长度的样本数，因此用 min(w, ...)
    退化为「窗口内全部可用样本」。
    """
    w = int(window)
    if w <= 0:
        return 1
    floor = 20 if w >= 120 else 3
    return max(1, min(w, max(floor, int(np.ceil(0.6 * w)))))


def _div(num, den, positive: bool = False) -> np.ndarray:
    """安全除法：分母无效 / 过小时返回 NaN，绝不返回 inf。

    positive=True 时要求分母 > _EPS；否则只要求 |分母| > _EPS。
    """
    num, den = np.broadcast_arrays(np.asarray(num, dtype="float64"),
                                   np.asarray(den, dtype="float64"))
    good = np.isfinite(num) & np.isfinite(den)
    good = good & ((den > _EPS) if positive else (np.abs(den) > _EPS))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / np.where(good, den, 1.0)
    return np.where(good, out, np.nan)


def _px_ret(t: int, C: Ctx, window: int) -> np.ndarray:
    """复权价区间收益 C[t] / C[t-window] - 1（端点缺失返回 NaN）。"""
    out = _nan(C)
    i1 = int(t)
    i0 = i1 - int(window)
    if i0 < 0:
        return out
    a = C.C[i1]
    b = C.C[i0]
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0.0) & (b > 0.0)
    out[ok] = a[ok] / b[ok] - 1.0
    return _fin(out)


def _compound(X: np.ndarray, need: int) -> np.ndarray:
    """把 (w,k) 的日收益矩阵按列复利累计；有效样本不足返回 NaN。"""
    k = int(X.shape[1])
    out = np.full(k, np.nan, dtype="float64")
    if X.shape[0] == 0:
        return out
    ok = np.isfinite(X) & (X > -1.0)
    cnt = ok.sum(0).astype("float64")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        logs = np.log1p(np.where(ok, X, 0.0))
        s = np.where(ok, logs, 0.0).sum(0)
        val = np.expm1(s / np.maximum(cnt, 1.0))
    good = cnt >= float(need)
    out[good] = val[good]
    return _fin(out)


def _daily_part(t: int, C: Ctx, window: int, kind: str) -> np.ndarray:
    """窗口内每日隔夜 / 日内收益矩阵 (w,k)。

    kind="ovn"  : ADJ_O[e] / C[e-1] - 1
    kind="intra": C[e] / ADJ_O[e] - 1
    """
    lo = max(int(t) - int(window) + 1, 0)
    idx = np.arange(lo, int(t) + 1, dtype=int)
    if kind == "ovn":
        idx = idx[idx >= 1]
        if idx.size == 0:
            return np.zeros((0, C.k), dtype="float64")
        num = C.ADJ_O[idx]
        den = C.C[idx - 1]
    else:
        num = C.C[idx]
        den = C.ADJ_O[idx]
    ok = np.isfinite(num) & np.isfinite(den) & (num > 0.0) & (den > 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(ok, num / np.where(ok, den, 1.0) - 1.0, np.nan)
    return _fin(out)


def _vol(t: int, C: Ctx, window: int) -> np.ndarray:
    """窗口内复权日收益的样本标准差（掩码版）。"""
    lo, W = C.sl(t, window, C.R)
    ok = np.isfinite(W)
    return _fin(C.std(W, ok, _need(window)))


# ------------------------------------------------------------------ #
# 1. 区间收益 / 几何收益 / 风险调整反转
# ------------------------------------------------------------------ #
def _ret20g(t: int, C: Ctx) -> np.ndarray:
    """过去 20 日复权日收益的几何平均：prod(1+R)^(1/20)-1。"""
    lo, W = C.sl(t, 20, C.R)
    ok = np.isfinite(W) & (W > -1.0)
    cnt = ok.sum(0).astype("float64")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        s = np.where(ok, np.log1p(np.where(ok, W, 0.0)), 0.0).sum(0)
        val = np.expm1(s / np.maximum(cnt, 1.0))
    good = cnt >= float(_need(20))
    return _fin(np.where(good, val, np.nan))


def _ret5_adj(t: int, C: Ctx) -> np.ndarray:
    """过去 5 日复权收益 / 过去 60 日复权日收益波动率。"""
    return _fin(_div(_px_ret(t, C, 5), _vol(t, C, 60), positive=True))


# ------------------------------------------------------------------ #
# 2. 市场模型残差累计收益
# ------------------------------------------------------------------ #
def _resid(t: int, C: Ctx, window: int) -> np.ndarray:
    """对等权市场收益做含截距 OLS，累计同窗口残差收益。

    有效样本必须成对（个股收益与市场收益同时有限）；beta 用同一窗口的
    掩码 OLS 估计，残差 = R - alpha - beta * mkt。
    """
    lo, W = C.sl(t, window, C.R)
    M = C.mkt[lo:int(t) + 1]
    ok = np.isfinite(W) & np.isfinite(M)[:, None]
    cnt = ok.sum(0).astype("float64")
    den = np.maximum(cnt, 1.0)
    mx = np.where(ok, W, 0.0).sum(0) / den
    mm = np.where(ok, M[:, None], 0.0).sum(0) / den
    Xc = np.where(ok, W - mx, 0.0)
    Mc = np.where(ok, M[:, None] - mm, 0.0)
    df = np.maximum(cnt - 1.0, 1.0)
    vm = (Mc ** 2).sum(0) / df
    cov = (Xc * Mc).sum(0) / df
    beta = _div(cov, vm, positive=True)
    alpha = mx - beta * mm
    with np.errstate(invalid="ignore"):
        res = np.where(ok, W - alpha[None, :] - beta[None, :] * M[:, None], 0.0)
    cum = res.sum(0)
    good = (cnt >= float(_need(window))) & np.isfinite(cum)
    return _fin(np.where(good, cum, np.nan))


# ------------------------------------------------------------------ #
# 3. 成交额加权收益 / 区间位置
# ------------------------------------------------------------------ #
def _amtmom(t: int, C: Ctx, window: int) -> np.ndarray:
    """窗口内成交额加权收益 sum(A*R) / sum(A)。"""
    lo, _ = C.sl(t, window, C.C)
    R = C.R[lo:int(t) + 1]
    A = C.A[lo:int(t) + 1]
    ok = np.isfinite(R) & np.isfinite(A) & (A > 0.0)
    aw = np.where(ok, A, 0.0)
    num = (aw * np.where(ok, R, 0.0)).sum(0)
    den = aw.sum(0)
    val = _div(num, den, positive=True)
    good = ok.sum(0).astype("float64") >= float(_need(window))
    return _fin(np.where(good, val, np.nan))


def _hl(t: int, C: Ctx, window: int) -> np.ndarray:
    """收盘价在窗口 [min(L), max(H)] 中的位置。"""
    lo, L = C.sl(t, window, C.ADJ_L)
    _, H = C.sl(t, window, C.ADJ_H)
    ok = np.isfinite(L) & np.isfinite(H) & (L > 0.0) & (H > 0.0)
    cnt = ok.sum(0).astype("float64")
    low = np.where(ok, L, np.inf).min(0)
    high = np.where(ok, H, -np.inf).max(0)
    px = C.C[t]
    pos = _div(px - low, high - low, positive=True)
    pos = np.clip(pos, 0.0, 1.0)
    good = (cnt >= float(_need(window))) & np.isfinite(px)
    return _fin(np.where(good, pos, np.nan))


# ------------------------------------------------------------------ #
# 4. 彩票 / 极值收益
# ------------------------------------------------------------------ #
def _extreme(t: int, C: Ctx, window: int, mode: str) -> np.ndarray:
    """窗口内单日收益最大值或最小值（掩码版）。"""
    lo, W = C.sl(t, window, C.R)
    ok = np.isfinite(W)
    cnt = ok.sum(0).astype("float64")
    if mode == "max":
        v = np.where(ok, W, -np.inf).max(0)
    else:
        v = np.where(ok, W, np.inf).min(0)
    good = cnt >= float(_need(window))
    return _fin(np.where(good, v, np.nan))


def _max5d(t: int, C: Ctx) -> np.ndarray:
    """近 20 日内最大 5 日滚动复权收益（向量化价格矩阵）。"""
    tt = int(t)
    lo_e = max(0, tt - 19)
    start = max(0, lo_e - 5)
    if tt - start < 5:
        return _nan(C)
    P = C.C[start:tt + 1]
    num = P[5:]
    den = P[:-5]
    ok = np.isfinite(num) & np.isfinite(den) & (num > 0.0) & (den > 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        V = np.where(ok, num / np.where(ok, den, 1.0) - 1.0, np.nan)
    cnt = ok.sum(0).astype("float64")
    mx = np.where(ok, V, -np.inf).max(0)
    good = cnt >= float(_need(5))
    return _fin(np.where(good, mx, np.nan))



# ------------------------------------------------------------------ #
# 5. 连续涨跌 / 跳空回补 / 反转加速度
# ------------------------------------------------------------------ #
def _streak_masks(C: Ctx):
    """缓存连涨 / 连跌扫描所需的布尔面板（只读，不修改 Ctx 原始数组）。"""
    hit = C.__dict__.get("_rev_streak_masks")
    if hit is not None and hit[0] == int(C.n) and hit[1] == int(C.k):
        return hit[2], hit[3], hit[4]
    finite = np.isfinite(C.R)
    susp = finite & np.isfinite(C.A) & (C.A == 0.0)
    up_add = finite & (C.R > 0.0) & (~susp)
    dn_add = finite & (C.R < 0.0) & (~susp)
    C.__dict__["_rev_streak_masks"] = (int(C.n), int(C.k), susp, up_add, dn_add)
    return susp, up_add, dn_add


def _streak_pair(t: int, C: Ctx):
    """截至 t 的连涨 / 连跌天数（停牌日跳过）。

    从 t 向前扫描，只读 [0, t] 行；停牌日（C.A==0 且 C.R 有限）不增加计数、
    也不打断 streak，真实零收益日（A>0, R==0）会打断 streak。结果按 t 缓存在
    Ctx 实例上，两个因子共享同一次扫描；Ctx.truncated(t) 是全新对象，缓存隔离。
    """
    cache = C.__dict__.setdefault("_rev_streak_t_cache", {})
    key = int(t)
    hit = cache.get(key)
    if hit is not None:
        return hit
    susp, up_add, dn_add = _streak_masks(C)
    k = int(C.k)
    up = np.zeros(k, dtype="float64")
    dn = np.zeros(k, dtype="float64")
    ua = np.ones(k, dtype=bool)
    da = np.ones(k, dtype=bool)
    i = key
    while i >= 0 and (ua.any() or da.any()):
        s = susp[i]
        p = up_add[i]
        n = dn_add[i]
        up += (ua & p)
        dn += (da & n)
        ua = ua & (p | s)
        da = da & (n | s)
        i -= 1
    if len(cache) > 256:
        cache.clear()
    cache[key] = (up, dn)
    return up, dn


def _upstreak(t: int, C: Ctx) -> np.ndarray:
    up, _ = _streak_pair(t, C)
    return _fin(np.where(np.isfinite(C.R[int(t)]), up, np.nan))


def _dnstreak(t: int, C: Ctx) -> np.ndarray:
    _, dn = _streak_pair(t, C)
    return _fin(np.where(np.isfinite(C.R[int(t)]), dn, np.nan))


def _gaprev(t: int, C: Ctx, window: int = 20) -> np.ndarray:
    """跳空缺口回补比例：sum(max(0, prevC-L)) / sum(max(0, ADJ_O-prevC))。"""
    out = _nan(C)
    lo = max(int(t) - int(window) + 1, 1)
    if lo > int(t):
        return out
    idx = np.arange(lo, int(t) + 1, dtype=int)
    o = C.ADJ_O[idx]
    l = C.ADJ_L[idx]
    p = C.C[idx - 1]
    ok = np.isfinite(o) & np.isfinite(l) & np.isfinite(p) \
        & (o > 0.0) & (l > 0.0) & (p > 0.0)
    up = np.where(ok, np.maximum(o - p, 0.0), 0.0)
    dn = np.where(ok, np.maximum(p - l, 0.0), 0.0)
    val = _div(dn.sum(0), up.sum(0), positive=True)
    good = ok.sum(0).astype("float64") >= float(_need(window))
    return _fin(np.where(good, val, np.nan))


def _last5_vs20(t: int, C: Ctx) -> np.ndarray:
    """最近 5 日复权收益 - 前 15 日复权收益。"""
    if int(t) < 20:
        return _nan(C)
    return _fin(_px_ret(t, C, 5) - _px_ret(int(t) - 5, C, 15))


# ------------------------------------------------------------------ #
# 6. 日内 / 隔夜拆解
# ------------------------------------------------------------------ #
def _intra(t: int, C: Ctx, window: int) -> np.ndarray:
    return _compound(_daily_part(t, C, window, "intra"), _need(window))


def _ovn(t: int, C: Ctx, window: int) -> np.ndarray:
    return _compound(_daily_part(t, C, window, "ovn"), _need(window))


def _ovn_last(t: int, C: Ctx) -> np.ndarray:
    out = _nan(C)
    if int(t) < 1:
        return out
    num = C.ADJ_O[int(t)]
    den = C.C[int(t) - 1]
    ok = np.isfinite(num) & np.isfinite(den) & (num > 0.0) & (den > 0.0)
    out[ok] = num[ok] / den[ok] - 1.0
    return _fin(out)


def _intra_last(t: int, C: Ctx) -> np.ndarray:
    out = _nan(C)
    num = C.C[int(t)]
    den = C.ADJ_O[int(t)]
    ok = np.isfinite(num) & np.isfinite(den) & (num > 0.0) & (den > 0.0)
    out[ok] = num[ok] / den[ok] - 1.0
    return _fin(out)


def _close_loc5(t: int, C: Ctx, window: int = 5) -> np.ndarray:
    """近 5 日平均日内收盘位置 (C-L)/(H-L)，无区间的日子跳过。"""
    lo, L = C.sl(t, window, C.ADJ_L)
    _, H = C.sl(t, window, C.ADJ_H)
    _, X = C.sl(t, window, C.C)
    rng = H - L
    ok = np.isfinite(rng) & (rng > _EPS) & np.isfinite(L) & np.isfinite(H) & np.isfinite(X)
    clv = np.where(ok, (X - L) / np.where(ok, rng, 1.0), np.nan)
    clv = np.clip(clv, 0.0, 1.0)
    cnt = ok.sum(0).astype("float64")
    val = np.where(ok, clv, 0.0).sum(0) / np.maximum(cnt, 1.0)
    good = cnt >= float(_need(window))
    return _fin(np.where(good, val, np.nan))


# ------------------------------------------------------------------ #
# 7. 回撤 / 自身分位 / 近高点距离
# ------------------------------------------------------------------ #
def _maxdd(t: int, C: Ctx, window: int = 20) -> np.ndarray:
    """近 20 日最大回撤取负：-max(1 - C/cummax(C))。"""
    lo, W = C.sl(t, window, C.C)
    ok = np.isfinite(W)
    cnt = ok.sum(0).astype("float64")
    cm = np.fmax.accumulate(W, axis=0)
    den = np.where(np.isfinite(cm) & (cm > _EPS), cm, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = 1.0 - W / den
    valid = ok & np.isfinite(cm) & (cm > _EPS) & np.isfinite(dd)
    dd = np.where(valid, np.maximum(dd, 0.0), 0.0)
    mx = dd.max(0)
    good = cnt >= float(_need(window))
    return _fin(np.where(good, -mx, np.nan))


def _ret5_rank60(t: int, C: Ctx) -> np.ndarray:
    """当前 5 日收益在过去 60 日自身 5 日收益分布中的分位（0~1）。"""
    tt = int(t)
    start = max(0, tt - 64)
    if tt - start < 5:
        return _nan(C)
    P = C.C[start:tt + 1]
    num = P[5:]
    den = P[:-5]
    ok = np.isfinite(num) & np.isfinite(den) & (num > 0.0) & (den > 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        V = np.where(ok, num / np.where(ok, den, 1.0) - 1.0, np.nan)
    need = _need(60)
    if V.shape[0] < need:
        return _nan(C)
    cnt = ok.sum(0).astype("float64")
    cur = V[-1]
    le = (ok & (V <= cur[None, :])).sum(0).astype("float64")
    good = (cnt >= float(need)) & np.isfinite(cur)
    val = np.where(good, le / np.maximum(cnt, 1.0), np.nan)
    return _fin(val)


def _high5_60(t: int, C: Ctx) -> np.ndarray:
    """复权收盘 / 近 60 日复权最高价 - 1。"""
    lo, H = C.sl(t, 60, C.ADJ_H)
    ok = np.isfinite(H)
    cnt = ok.sum(0).astype("float64")
    hmax = np.where(ok, H, -np.inf).max(0)
    val = _div(C.C[int(t)], hmax, positive=True) - 1.0
    good = cnt >= float(_need(60))
    return _fin(np.where(good, val, np.nan))


# ------------------------------------------------------------------ #
# 注册表
# ------------------------------------------------------------------ #
def _spec(fn, desc: str, direction: int) -> dict:
    """把 (t, C) 实现包装成注册表要求的 (t, md, C) 签名。"""
    return dict(func=lambda t, md, C, _fn=fn: _fn(int(t), C),
                desc=desc, direction=int(direction))


FACTORS = {
    # --- 区间收益 / 几何收益 ---
    "rev_ret1": _spec(lambda t, C: _px_ret(t, C, 1),
                      "过去1日复权收益 C[t]/C[t-1]-1（短期反转）", -1),
    "rev_ret2": _spec(lambda t, C: _px_ret(t, C, 2),
                      "过去2日复权收益 C[t]/C[t-2]-1（短期反转）", -1),
    "rev_ret3": _spec(lambda t, C: _px_ret(t, C, 3),
                      "过去3日复权收益 C[t]/C[t-3]-1（短期反转）", -1),
    "rev_ret4": _spec(lambda t, C: _px_ret(t, C, 4),
                      "过去4日复权收益 C[t]/C[t-4]-1（短期反转）", -1),
    "rev_ret10": _spec(lambda t, C: _px_ret(t, C, 10),
                       "过去10日复权收益 C[t]/C[t-10]-1（短期反转）", -1),
    "rev_ret20g": _spec(lambda t, C: _ret20g(t, C),
                        "过去20日复权日收益的几何平均 prod(1+R)^(1/20)-1", -1),
    "rev_ret5_adj": _spec(lambda t, C: _ret5_adj(t, C),
                          "过去5日复权收益 / 过去60日日收益波动率（风险调整反转）", -1),
    # --- 市场模型残差 ---
    "rev_resid5": _spec(lambda t, C: _resid(t, C, 5),
                        "过去5日对等权市场收益做含截距市场模型回归后的残差累计收益", -1),
    "rev_resid20": _spec(lambda t, C: _resid(t, C, 20),
                         "过去20日市场模型残差累计收益（含截距，掩码配对OLS）", -1),
    # --- 成交额加权收益 / 区间位置 ---
    "rev_amtmom5": _spec(lambda t, C: _amtmom(t, C, 5),
                         "过去5日成交额加权收益 sum(A*R)/sum(A)", -1),
    "rev_amtmom20": _spec(lambda t, C: _amtmom(t, C, 20),
                          "过去20日成交额加权收益 sum(A*R)/sum(A)", -1),
    "rev_hl5": _spec(lambda t, C: _hl(t, C, 5),
                     "收盘价在近5日区间中的位置 (C-min(L))/(max(H)-min(L))", -1),
    "rev_hl20": _spec(lambda t, C: _hl(t, C, 20),
                      "收盘价在近20日区间中的位置 (C-min(L))/(max(H)-min(L))", -1),
    # --- 彩票 / 极值 ---
    "rev_max1d20": _spec(lambda t, C: _extreme(t, C, 20, "max"),
                         "近20日最大单日复权收益（彩票MAX）", -1),
    "rev_max5d20": _spec(lambda t, C: _max5d(t, C),
                         "近20日内最大5日滚动复权收益", -1),
    "rev_min1d20": _spec(lambda t, C: _extreme(t, C, 20, "min"),
                         "近20日最小单日复权收益", 1),
    # --- 连续涨跌 ---
    "rev_upstreak": _spec(lambda t, C: _upstreak(t, C),
                          "截至t的连续上涨天数（停牌日跳过）", -1),
    "rev_dnstreak": _spec(lambda t, C: _dnstreak(t, C),
                          "截至t的连续下跌天数（停牌日跳过）", 1),
    # --- 跳空 / 反转加速度 ---
    "rev_gaprev20": _spec(lambda t, C: _gaprev(t, C, 20),
                          "近20日跳空回补比例 sum(max(0,prevC-L))/sum(max(0,ADJ_O-prevC))", -1),
    "rev_last5_vs20": _spec(lambda t, C: _last5_vs20(t, C),
                            "最近5日复权收益 - 前15日复权收益（反转加速度）", -1),
    # --- 日内 / 隔夜拆解 ---
    "rev_intra5": _spec(lambda t, C: _intra(t, C, 5),
                        "近5日日内收益 (C/ADJ_O-1) 累计", -1),
    "rev_intra20": _spec(lambda t, C: _intra(t, C, 20),
                         "近20日日内收益 (C/ADJ_O-1) 累计", -1),
    "rev_ovn5": _spec(lambda t, C: _ovn(t, C, 5),
                      "近5日隔夜收益 (ADJ_O/prevC-1) 累计", -1),
    "rev_ovn20": _spec(lambda t, C: _ovn(t, C, 20),
                       "近20日隔夜收益 (ADJ_O/prevC-1) 累计", -1),
    "rev_ovn_last": _spec(lambda t, C: _ovn_last(t, C),
                          "最近1日隔夜收益 ADJ_O[t]/C[t-1]-1", -1),
    "rev_intra_last": _spec(lambda t, C: _intra_last(t, C),
                            "最近1日日内收益 C[t]/ADJ_O[t]-1", -1),
    "rev_close_loc5": _spec(lambda t, C: _close_loc5(t, C, 5),
                            "近5日平均日内收盘位置 (C-L)/(H-L)", -1),
    # --- 回撤 / 自身分位 / 高点距离 ---
    "rev_maxdd20": _spec(lambda t, C: _maxdd(t, C, 20),
                         "近20日最大回撤取负 -max(1-C/cummax(C))", 1),
    "rev_ret5_rank60": _spec(lambda t, C: _ret5_rank60(t, C),
                             "近5日收益在过去60日自身5日收益分布中的分位（0~1）", -1),
    "rev_high5_60": _spec(lambda t, C: _high5_60(t, C),
                          "复权收盘 / 近60日复权最高价 - 1", -1),
}