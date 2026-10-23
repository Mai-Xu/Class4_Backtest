# -*- coding: utf-8 -*-
"""K 线形态 / 日内结构因子（factor_zoo 分类：pattern）。

实现开发规范 §5.7 的全部 pt_* 因子，并在同类范围内补充 5 个形态因子（共 39 个）。
K 线形态一律用**复权** OHLC（C.C / C.ADJ_O / C.ADJ_H / C.ADJ_L），成交额用 C.A（元），
日收益用 C.R（复权日收益）。

口径约定
--------
- 有效 K 线：OHLC 全部有限且 H > L。停牌日与一字板 H==L（振幅 0）形态无意义，
  按缺失处理，对应位置返回 NaN。
- 「放量」：当日成交额 > 其近 60 日滚动均额（含当日）。
- 「跳空」：向上跳空 = 开盘价 > 前一日最高价；向下跳空 = 开盘价 < 前一日最低价。
- 「连阳 / 连阴」按 K 线实体方向（C>O 为阳线、C<O 为阴线）：停牌日跳过不打断，
  十字星（C==O）打断。
- 窗口最少有效样本按规范：短窗 max(3, ceil(0.6w))，长窗(>=120) max(20, ceil(0.6w))。

因果性：所有窗口只切 [0, t] 行（最多多取窗口前 1 行做「前一日」对齐，仍在 t 之内）；
统计量全部走掩码矩阵运算；连阳/连阴长度用 cumsum + maximum.accumulate 矩阵化求解，
没有任何逐股票 Python 循环。K 线几何量按需计算，避免为单个因子付整块窗口的开销。
"""
from __future__ import annotations

import numpy as np

from .ctx import Ctx

_EPS = 1e-12
_DOJI = 0.1          # 十字星阈值 |C-O|/(H-L)
_TOUCH = 0.1         # 「收盘贴近最高/最低」的影线阈值
_LONG_SHADOW = 0.5   # 长影线阈值（影线占振幅比例）
_VOL_WIN = 60        # 「放量」比较用的均额窗口


# ------------------------------------------------------------------ #
# 通用工具
# ------------------------------------------------------------------ #
def _nan(C: Ctx) -> np.ndarray:
    """全 NaN 横截面向量。"""
    return np.full(int(C.k), np.nan, dtype="float64")


def _fin(x) -> np.ndarray:
    """出口统一把 ±inf / NaN 收敛成 NaN。"""
    x = np.asarray(x, dtype="float64")
    return np.where(np.isfinite(x), x, np.nan)


def _need(window: int) -> int:
    """窗口最少有效样本数：短窗 max(3, ceil(0.6w))，长窗 max(20, ceil(0.6w))。"""
    w = int(window)
    if w <= 0:
        return 1
    floor = 20 if w >= 120 else 3
    return max(1, min(w, max(floor, int(np.ceil(0.6 * w)))))


def _div(num, den, positive: bool = False) -> np.ndarray:
    """安全除法：分母无效 / 过小时返回 NaN，绝不返回 inf。"""
    num, den = np.broadcast_arrays(np.asarray(num, dtype="float64"),
                                   np.asarray(den, dtype="float64"))
    good = np.isfinite(num) & np.isfinite(den)
    good = good & ((den > _EPS) if positive else (np.abs(den) > _EPS))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / np.where(good, den, 1.0)
    return np.where(good, out, np.nan)


def _blk(t: int, C: Ctx, w: int) -> dict:
    """窗口 [t-w+1, t] 的 OHLC / 成交额 / 收益块，附前一日对齐数组。

    多取窗口前 1 行仅用于「前一日」对齐（仍在 t 之内）；数据起点不足时块会短于 w。
    """
    t = int(t)
    lo = max(t - int(w) + 1, 0)
    lo2 = max(lo - 1, 0)
    sl = slice(lo2, t + 1)
    Oe = C.ADJ_O[sl]
    He = C.ADJ_H[sl]
    Le = C.ADJ_L[sl]
    Ce = C.C[sl]
    Ae = C.A[sl]
    Re = C.R[sl]
    if lo2 < lo:
        O, H, L, CL, A, R = Oe[1:], He[1:], Le[1:], Ce[1:], Ae[1:], Re[1:]
        pH, pL, pC = He[:-1], Le[:-1], Ce[:-1]
    else:
        O, H, L, CL, A, R = Oe, He, Le, Ce, Ae, Re
        blank = np.full(Oe.shape, np.nan, dtype="float64")
        pH, pL, pC = blank, blank, blank
    rng = H - L
    bar = (np.isfinite(O) & np.isfinite(H) & np.isfinite(L) & np.isfinite(CL)
           & (rng > 0.0))
    return dict(O=O, H=H, L=L, CL=CL, A=A, R=R, pH=pH, pL=pL, pC=pC,
                rng=rng, bar=bar)


def _geom(t: int, C: Ctx, w: int, fields=()) -> dict:
    """在 _blk 基础上**按需**补出 K 线几何量（只计算 fields 点名的面板）。

    面板（窗口内 (w,k) 数组，无效 K 线处为 NaN）：
    body/absbody 实体、upsh/dnsh 上/下影、clv 收盘位置、oloc 开盘位置、
    eodrev 收盘相对当日中枢偏离、rngrel 相对振幅 (H-L)/C。
    """
    B = _blk(t, C, w)
    bar, rng = B["bar"], B["rng"]
    O, H, L, CL = B["O"], B["H"], B["L"], B["CL"]
    px_ok = bar & (CL > 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        if "body" in fields:
            B["body"] = np.where(bar, (CL - O) / rng, np.nan)
        if "absbody" in fields:
            B["absbody"] = np.where(bar, np.abs(CL - O) / rng, np.nan)
        if "upsh" in fields:
            B["upsh"] = np.where(bar, (H - np.maximum(O, CL)) / rng, np.nan)
        if "dnsh" in fields:
            B["dnsh"] = np.where(bar, (np.minimum(O, CL) - L) / rng, np.nan)
        if "clv" in fields:
            B["clv"] = np.where(bar, (CL - L) / rng, np.nan)
        if "oloc" in fields:
            B["oloc"] = np.where(bar, (O - L) / rng, np.nan)
        if "eodrev" in fields:
            B["eodrev"] = np.where(bar, (CL - 0.5 * (H + L)) / rng, np.nan)
        if "rngrel" in fields:
            B["rngrel"] = np.where(px_ok, rng / CL, np.nan)
    return B


def _bar1(t: int, C: Ctx):
    """单日 K 线原始数组。"""
    t = int(t)
    O = C.ADJ_O[t]
    H = C.ADJ_H[t]
    L = C.ADJ_L[t]
    CL = C.C[t]
    return O, H, L, CL, H - L


def _pmean(C: Ctx, B: dict, key: str, w: int) -> np.ndarray:
    """窗口内面板 key 的掩码均值（有效样本 < 最少样本数返回 NaN）。"""
    X = B[key]
    return _fin(C.mean(X, np.isfinite(X), _need(w)))


def _pratio(C: Ctx, hit, ok, w: int) -> np.ndarray:
    """窗口内命中占比：分母为窗口内有效样本数。"""
    return _fin(C.mean(hit.astype("float64"), ok, _need(w)))


def _pcount(C: Ctx, hit, ok, w: int) -> np.ndarray:
    """窗口内命中次数（整数，浮点存储）：有效样本不足返回 NaN。"""
    good = ok.sum(0).astype("float64") >= float(_need(w))
    return _fin(np.where(good, hit.sum(0).astype("float64"), np.nan))


def _roll_mean(X, ok, w: int, need: int) -> np.ndarray:
    """(m,k) 面板的滚动掩码均值（窗口 w，锚定行尾）；有效样本 < need 为 NaN。"""
    m, k = X.shape
    out = np.full((m, k), np.nan, dtype="float64")
    if m == 0:
        return out
    xs = np.where(ok, X, 0.0)
    zero = np.zeros((1, k), dtype="float64")
    cx = np.vstack([zero, np.cumsum(xs, axis=0)])
    cc = np.vstack([zero, np.cumsum(ok, axis=0).astype("float64")])
    i = np.arange(m)
    hi = i + 1
    lo = np.maximum(i + 1 - int(w), 0)
    s = cx[hi] - cx[lo]
    c = cc[hi] - cc[lo]
    val = s / np.maximum(c, 1.0)
    good = c >= float(need)
    out[good] = val[good]
    return out


def _compound(X, need: int) -> np.ndarray:
    """(m,k) 日收益矩阵按列复利累计；有效样本不足返回 NaN。"""
    if X.shape[0] == 0:
        return np.full(X.shape[1], np.nan, dtype="float64")
    ok = np.isfinite(X) & (X > -1.0)
    cnt = ok.sum(0).astype("float64")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        s = np.log1p(np.where(ok, X, 0.0)).sum(0)
        val = np.expm1(s)
    return np.where(cnt >= float(need), val, np.nan)


def _autocorr_last(X, ok, w: int, need: int) -> np.ndarray:
    """最后 w 行的滞后 1 期自相关（配对掩码 Pearson，min_periods=need）。"""
    m, k = X.shape
    if m < 3:
        return np.full(k, np.nan, dtype="float64")
    Xs = X[-int(w):]
    oks = ok[-int(w):]
    a = Xs[1:]
    b = Xs[:-1]
    okp = oks[1:] & oks[:-1] & np.isfinite(a) & np.isfinite(b)
    n = okp.sum(0).astype("float64")
    aa = np.where(okp, a, 0.0)
    bb = np.where(okp, b, 0.0)
    d = np.maximum(n, 1.0)
    sa = aa.sum(0)
    sb = bb.sum(0)
    cov = (aa * bb).sum(0) - sa * sb / d
    va = (aa * aa).sum(0) - sa * sa / d
    vb = (bb * bb).sum(0) - sb * sb / d
    val = _div(cov, np.sqrt(np.maximum(va, 0.0) * np.maximum(vb, 0.0)), positive=True)
    good = (n >= float(need)) & (va > _EPS) & (vb > _EPS)
    return np.where(good, val, np.nan)


def _run_len(add, brk) -> np.ndarray:
    """窗口内最长连续 True 的长度：brk 打断，add/brk 都不成立的日子顺延。

    矩阵化求解：run = cumsum(add) - cumsum(add) 在最近一次 brk 处的取值。
    """
    m, k = add.shape
    if m == 0:
        return np.zeros(k, dtype="float64")
    cum = np.cumsum(add, axis=0).astype("float64")
    idx = np.where(brk, np.arange(1, m + 1, dtype="float64")[:, None], 0.0)
    last = np.maximum.accumulate(idx, axis=0)
    pos = np.maximum(last - 1.0, 0.0).astype(np.intp)
    gather = np.take_along_axis(cum, pos, axis=0)
    gather = np.where(last > 0.0, gather, 0.0)
    return (cum - gather).max(0)


def _runs(t: int, C: Ctx, w: int = 20):
    """窗口内最长连阳 / 连阴长度（停牌跳过，十字星打断）。"""
    B = _geom(t, C, w)
    O, CL, A = B["O"], B["CL"], B["A"]
    valid = np.isfinite(O) & np.isfinite(CL) & (O > 0.0) & (CL > 0.0)
    susp = valid & np.isfinite(A) & (A <= 0.0)
    act = valid & ~susp
    up = _run_len(act & (CL > O), act & (CL <= O))
    dn = _run_len(act & (CL < O), act & (CL >= O))
    return up, dn, act


def _amt_vol_mask(t: int, C: Ctx, w_bars: int):
    """窗口 [t-w_bars+1, t] 的「放量」掩码：成交额 > 近 60 日滚动均额。"""
    t = int(t)
    lo_g = max(t - int(w_bars) + 1, 0)
    lo = max(lo_g - (_VOL_WIN - 1), 0)
    A = C.A[lo:t + 1]
    ok = np.isfinite(A) & (A > 0.0)
    mu = _roll_mean(A, ok, _VOL_WIN, _need(_VOL_WIN))
    n = t + 1 - lo_g
    return A[-n:], ok[-n:], mu[-n:]


def _gap_panels(t: int, C: Ctx, w: int):
    """窗口内跳空中间量：严格缺口口径（开 > 昨高 / 开 < 昨低）。"""
    B = _geom(t, C, w)
    O, L = B["O"], B["L"]
    pH, pL, pC = B["pH"], B["pL"], B["pC"]
    ok = (np.isfinite(O) & (O > 0.0) & np.isfinite(L)
          & np.isfinite(pH) & np.isfinite(pL) & np.isfinite(pC)
          & (pH > 0.0) & (pL > 0.0) & (pC > 0.0))
    return B, ok, ok & (O > pH), ok & (O < pL)


# ------------------------------------------------------------------ #
# 1. 单根 K 线形态
# ------------------------------------------------------------------ #
def _body1(t: int, C: Ctx) -> np.ndarray:
    """当日实体占比 (C-O)/(H-L)。"""
    O, H, L, CL, rng = _bar1(t, C)
    return _fin(_div(CL - O, rng, positive=True))


def _upper_shadow1(t: int, C: Ctx) -> np.ndarray:
    """当日上影线占比 (H-max(O,C))/(H-L)。"""
    O, H, L, CL, rng = _bar1(t, C)
    return _fin(_div(H - np.maximum(O, CL), rng, positive=True))


def _lower_shadow1(t: int, C: Ctx) -> np.ndarray:
    """当日下影线占比 (min(O,C)-L)/(H-L)。"""
    O, H, L, CL, rng = _bar1(t, C)
    return _fin(_div(np.minimum(O, CL) - L, rng, positive=True))


def _pivot_dist(t: int, C: Ctx) -> np.ndarray:
    """(C - (H+L+昨C)/3) / C（枢轴点乖离）。"""
    t = int(t)
    if t < 1:
        return _nan(C)
    O, H, L, CL, rng = _bar1(t, C)
    piv = (H + L + C.C[t - 1]) / 3.0
    return _fin(_div(CL - piv, CL, positive=True))


# ------------------------------------------------------------------ #
# 2. 十字星 / 长影线 / 影线不对称
# ------------------------------------------------------------------ #
def _doji5(t: int, C: Ctx) -> np.ndarray:
    """近 5 日十字星（|C-O|/(H-L) < 0.1）占比。"""
    B = _geom(t, C, 5, ("absbody",))
    ok = B["bar"]
    return _pratio(C, ok & (B["absbody"] < _DOJI), ok, 5)


def _long_upper20(t: int, C: Ctx) -> np.ndarray:
    """近 20 日「长上影（上影>0.5 且放量）」出现次数。"""
    B = _geom(t, C, 20, ("upsh",))
    A, okA, mu = _amt_vol_mask(t, C, 20)
    ok = B["bar"]
    vol = okA & np.isfinite(mu) & (A > mu)
    return _pcount(C, ok & vol & (B["upsh"] > _LONG_SHADOW), ok, 20)


def _reversal_bar20(t: int, C: Ctx) -> np.ndarray:
    """近 20 日「长下影（下影>0.5 且放量）」出现次数。"""
    B = _geom(t, C, 20, ("dnsh",))
    A, okA, mu = _amt_vol_mask(t, C, 20)
    ok = B["bar"]
    vol = okA & np.isfinite(mu) & (A > mu)
    return _pcount(C, ok & vol & (B["dnsh"] > _LONG_SHADOW), ok, 20)


def _hammer20(t: int, C: Ctx) -> np.ndarray:
    """近 20 日锤子线占比（下影>0.5、实体<0.3、收盘位于上半部）。"""
    B = _geom(t, C, 20, ("dnsh", "absbody", "clv"))
    ok = B["bar"]
    hit = ok & (B["dnsh"] > _LONG_SHADOW) & (B["absbody"] < 0.3) & (B["clv"] >= 0.6)
    return _pratio(C, hit, ok, 20)


def _shooting_star20(t: int, C: Ctx) -> np.ndarray:
    """近 20 日射击之星占比（上影>0.5、实体<0.3、收盘位于下半部）。"""
    B = _geom(t, C, 20, ("upsh", "absbody", "clv"))
    ok = B["bar"]
    hit = ok & (B["upsh"] > _LONG_SHADOW) & (B["absbody"] < 0.3) & (B["clv"] <= 0.4)
    return _pratio(C, hit, ok, 20)


def _shadow_asym20(t: int, C: Ctx) -> np.ndarray:
    """近 20 日平均上影 / 平均下影（影线不对称）。"""
    B = _geom(t, C, 20, ("upsh", "dnsh"))
    up = _pmean(C, B, "upsh", 20)
    dn = _pmean(C, B, "dnsh", 20)
    return _fin(_div(up, dn, positive=True))


# ------------------------------------------------------------------ #
# 3. 日内位置 / 实体结构
# ------------------------------------------------------------------ #
def _shape20(t: int, C: Ctx) -> np.ndarray:
    """近 20 日平均日内收盘位置 (C-L)/(H-L)。"""
    B = _geom(t, C, 20, ("clv",))
    return _pmean(C, B, "clv", 20)


def _open_strength20(t: int, C: Ctx) -> np.ndarray:
    """近 20 日平均开盘位置 (O-L)/(H-L)。"""
    B = _geom(t, C, 20, ("oloc",))
    return _pmean(C, B, "oloc", 20)


def _close_vs_open20(t: int, C: Ctx) -> np.ndarray:
    """近 20 日 mean(C-O) / mean(H-L)。"""
    B = _geom(t, C, 20)
    ok = B["bar"]
    num = np.where(ok, B["CL"] - B["O"], 0.0).sum(0)
    den = np.where(ok, B["rng"], 0.0).sum(0)
    val = _div(num, den, positive=True)
    good = ok.sum(0).astype("float64") >= float(_need(20))
    return _fin(np.where(good, val, np.nan))


def _intraday_skew20(t: int, C: Ctx) -> np.ndarray:
    """近 20 日 (C-O)/(H-L) 的标准差（日内强度不稳定度）。"""
    B = _geom(t, C, 20, ("body",))
    X = B["body"]
    return _fin(C.std(X, np.isfinite(X), _need(20), ddof=1))


def _close_loc_std20(t: int, C: Ctx) -> np.ndarray:
    """近 20 日 (C-L)/(H-L) 的标准差（收盘位置不稳定度）。"""
    B = _geom(t, C, 20, ("clv",))
    X = B["clv"]
    return _fin(C.std(X, np.isfinite(X), _need(20), ddof=1))


def _eod_reversal20(t: int, C: Ctx) -> np.ndarray:
    """近 20 日 (C-(H+L)/2)/(H-L) 均值（收盘相对当日中枢的偏离）。"""
    B = _geom(t, C, 20, ("eodrev",))
    return _pmean(C, B, "eodrev", 20)


def _upper_touch20(t: int, C: Ctx) -> np.ndarray:
    """近 20 日收盘贴近当日最高的天数占比（上影<0.1）。"""
    B = _geom(t, C, 20, ("upsh",))
    ok = B["bar"]
    return _pratio(C, ok & (B["upsh"] < _TOUCH), ok, 20)


def _lower_touch20(t: int, C: Ctx) -> np.ndarray:
    """近 20 日收盘贴近当日最低的天数占比（下影<0.1）。"""
    B = _geom(t, C, 20, ("dnsh",))
    ok = B["bar"]
    return _pratio(C, ok & (B["dnsh"] < _TOUCH), ok, 20)


# ------------------------------------------------------------------ #
# 4. 振幅 / 波动收缩 / 通道
# ------------------------------------------------------------------ #
def _range_ratio(t: int, C: Ctx, w: int) -> np.ndarray:
    """近 w 日平均振幅 (H-L)/C。"""
    B = _geom(t, C, w, ("rngrel",))
    return _pmean(C, B, "rngrel", w)


def _range_trend(t: int, C: Ctx) -> np.ndarray:
    """振幅20 / 振幅60 - 1。"""
    r20 = _range_ratio(t, C, 20)
    r60 = _range_ratio(t, C, 60)
    return _fin(_div(r20, r60, positive=True) - 1.0)


def _vol_contract20(t: int, C: Ctx) -> np.ndarray:
    """近 10 日平均振幅 / 近 60 日平均振幅。"""
    r10 = _range_ratio(t, C, 10)
    r60 = _range_ratio(t, C, 60)
    return _fin(_div(r10, r60, positive=True))


def _range_expand20(t: int, C: Ctx) -> np.ndarray:
    """近 5 日平均振幅 / 近 20 日平均振幅 - 1（短期振幅扩张）。"""
    r5 = _range_ratio(t, C, 5)
    r20 = _range_ratio(t, C, 20)
    return _fin(_div(r5, r20, positive=True) - 1.0)


def _nr7_count20(t: int, C: Ctx) -> np.ndarray:
    """近 20 日中「7 日最窄振幅」（NR7）出现次数。"""
    t = int(t)
    lo_g = max(t - 19, 0)
    lo = max(lo_g - 6, 0)
    rng = C.ADJ_H[lo:t + 1] - C.ADJ_L[lo:t + 1]
    ok = np.isfinite(rng) & (rng > 0.0)
    m = rng.shape[0]
    if m < 7:
        return _nan(C)
    best = np.where(ok, rng, np.inf)
    cnt = ok.astype("float64")
    for s in range(1, 7):                       # 7 次整体矩阵移位，非逐股循环
        prev_ok = ok[:-s]
        best[s:] = np.minimum(best[s:], np.where(prev_ok, rng[:-s], np.inf))
        cnt[s:] = cnt[s:] + prev_ok
    nr7 = ok & (cnt >= 7.0) & (best == rng)
    n = t + 1 - lo_g
    good = ok[-n:].sum(0).astype("float64") >= float(_need(20))
    return _fin(np.where(good, nr7[-n:].sum(0).astype("float64"), np.nan))


# ------------------------------------------------------------------ #
# 5. 跳空 / 内外包 / 高低点结构
# ------------------------------------------------------------------ #
def _gap_up20(t: int, C: Ctx) -> np.ndarray:
    """近 20 日向上跳空（开>昨高）天数占比。"""
    _, ok, gup, _ = _gap_panels(t, C, 20)
    return _pratio(C, gup, ok, 20)


def _gap_dn20(t: int, C: Ctx) -> np.ndarray:
    """近 20 日向下跳空（开<昨低）天数占比。"""
    _, ok, _, gdn = _gap_panels(t, C, 20)
    return _pratio(C, gdn, ok, 20)


def _gap_fill_rate20(t: int, C: Ctx) -> np.ndarray:
    """近 20 日向上跳空当日回补的缺口幅度占比（按缺口幅度加权）。"""
    B, ok, gup, _ = _gap_panels(t, C, 20)
    O, L, pH = B["O"], B["L"], B["pH"]
    gapmag = np.where(gup, O - pH, 0.0)
    filled = np.where(gup, np.minimum(O - np.maximum(L, pH), O - pH), 0.0)
    val = _div(filled.sum(0), gapmag.sum(0), positive=True)
    good = (ok.sum(0) >= float(_need(20))) & (gup.sum(0) >= 1)
    return _fin(np.where(good, val, np.nan))


def _inside_days20(t: int, C: Ctx) -> np.ndarray:
    """近 20 日内包线（H<=昨高 且 L>=昨低）天数占比。"""
    B = _geom(t, C, 20)
    H, L, pH, pL = B["H"], B["L"], B["pH"], B["pL"]
    ok = B["bar"] & np.isfinite(pH) & np.isfinite(pL) & (pH > 0.0) & (pL > 0.0)
    return _pratio(C, ok & (H <= pH) & (L >= pL), ok, 20)


def _outside_days20(t: int, C: Ctx) -> np.ndarray:
    """近 20 日外包线（H>=昨高 且 L<=昨低）天数占比。"""
    B = _geom(t, C, 20)
    H, L, pH, pL = B["H"], B["L"], B["pH"], B["pL"]
    ok = B["bar"] & np.isfinite(pH) & np.isfinite(pL) & (pH > 0.0) & (pL > 0.0)
    return _pratio(C, ok & (H >= pH) & (L <= pL), ok, 20)


def _hh_hl_count20(t: int, C: Ctx) -> np.ndarray:
    """近 20 日高点与低点同时抬高（H>昨高 且 L>昨低）天数占比。"""
    B = _geom(t, C, 20)
    H, L, pH, pL = B["H"], B["L"], B["pH"], B["pL"]
    ok = B["bar"] & np.isfinite(pH) & np.isfinite(pL) & (pH > 0.0) & (pL > 0.0)
    return _pratio(C, ok & (H > pH) & (L > pL), ok, 20)


def _ll_lh_count20(t: int, C: Ctx) -> np.ndarray:
    """近 20 日高点与低点同时降低（H<昨高 且 L<昨低）天数占比。"""
    B = _geom(t, C, 20)
    H, L, pH, pL = B["H"], B["L"], B["pH"], B["pL"]
    ok = B["bar"] & np.isfinite(pH) & np.isfinite(pL) & (pH > 0.0) & (pL > 0.0)
    return _pratio(C, ok & (H < pH) & (L < pL), ok, 20)


# ------------------------------------------------------------------ #
# 6. 连续涨跌 / 上涨天数
# ------------------------------------------------------------------ #
def _up_days(t: int, C: Ctx, w: int) -> np.ndarray:
    """近 w 日上涨（R>0）天数占比（仅统计有成交的日子）。"""
    B = _geom(t, C, w)
    A, R = B["A"], B["R"]
    ok = np.isfinite(R) & np.isfinite(A) & (A > 0.0)
    return _pratio(C, ok & (R > 0.0), ok, w)


def _run_up(t: int, C: Ctx) -> np.ndarray:
    """近 20 日最长连阳（C>O 连续天数，停牌跳过、十字星打断）。"""
    up, _, act = _runs(t, C, 20)
    good = act.sum(0).astype("float64") >= float(_need(20))
    return _fin(np.where(good, up, np.nan))


def _run_dn(t: int, C: Ctx) -> np.ndarray:
    """近 20 日最长连阴（C<O 连续天数，停牌跳过、十字星打断）。"""
    _, dn, act = _runs(t, C, 20)
    good = act.sum(0).astype("float64") >= float(_need(20))
    return _fin(np.where(good, dn, np.nan))


# ------------------------------------------------------------------ #
# 7. 路径结构 / 自相关 / 熵 / 日内隔夜拆分
# ------------------------------------------------------------------ #
def _hl_autocorr60(t: int, C: Ctx) -> np.ndarray:
    """近 60 日 (H-L)/C 一阶自相关。"""
    B = _geom(t, C, 60, ("rngrel",))
    X = B["rngrel"]
    return _fin(_autocorr_last(X, np.isfinite(X), 60, _need(60)))


def _ret_autocorr_neg20(t: int, C: Ctx) -> np.ndarray:
    """近 20 日复权日收益一阶自相关取负。"""
    B = _geom(t, C, 20)
    A, R = B["A"], B["R"]
    ok = np.isfinite(R) & np.isfinite(A) & (A > 0.0)
    return _fin(-_autocorr_last(R, ok, 20, _need(20)))


def _path_entropy20(t: int, C: Ctx) -> np.ndarray:
    """近 20 日「收盘位于当日上半部」符号序列的二元 Shannon 熵（0~1）。"""
    B = _geom(t, C, 20, ("clv",))
    ok = B["bar"]
    p = C.mean((ok & (B["clv"] > 0.5)).astype("float64"), ok, _need(20))
    q = np.where(np.isfinite(p), np.clip(p, 0.0, 1.0), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = -(np.where(q > 0.0, q * np.log2(np.where(q > 0.0, q, 1.0)), 0.0)
              + np.where(q < 1.0, (1.0 - q) * np.log2(np.where(q < 1.0, 1.0 - q, 1.0)), 0.0))
    return _fin(np.where(np.isfinite(q), e, np.nan))


def _intra_ovn20(t: int, C: Ctx) -> np.ndarray:
    """近 20 日日内累计收益（复利） − 隔夜累计收益（复利）。"""
    t = int(t)
    lo = max(t - 19, 0)
    lo2 = max(lo - 1, 0)
    idx = np.arange(lo, t + 1)
    Oe = C.ADJ_O[idx]
    intra = _div(C.C[idx], Oe, positive=True) - 1.0
    if lo2 < lo:
        ovn = _div(Oe, C.C[idx - 1], positive=True) - 1.0
    else:
        ovn = np.full(intra.shape, np.nan, dtype="float64")
    need = _need(20)
    return _fin(_compound(intra, need) - _compound(ovn, need))


# ------------------------------------------------------------------ #
# 注册表
# ------------------------------------------------------------------ #
def _spec(fn, desc: str, direction: int) -> dict:
    """把 (t, C) 实现包装成注册表要求的 (t, md, C) 签名。"""
    return dict(func=lambda t, md, C, _fn=fn: _fn(int(t), C),
                desc=desc, direction=int(direction))


FACTORS = {
    # --- 单根 K 线形态 ---
    "pt_body1": _spec(_body1, "当日实体占比 (C-O)/(H-L)", -1),
    "pt_upper_shadow1": _spec(_upper_shadow1, "当日上影线占比 (H-max(O,C))/(H-L)", -1),
    "pt_lower_shadow1": _spec(_lower_shadow1, "当日下影线占比 (min(O,C)-L)/(H-L)", 1),
    "pt_pivot_dist": _spec(_pivot_dist, "(C-(H+L+昨C)/3)/C：枢轴点乖离", -1),
    # --- 十字星 / 长影线 ---
    "pt_doji5": _spec(_doji5, "近5日十字星(|C-O|/(H-L)<0.1)占比", -1),
    "pt_long_upper20": _spec(_long_upper20, "近20日长上影(上影>0.5且放量)出现次数", -1),
    "pt_reversal_bar20": _spec(_reversal_bar20, "近20日长下影(下影>0.5且放量)出现次数", 1),
    "pt_hammer20": _spec(_hammer20, "近20日锤子线占比(下影>0.5且实体<0.3且收盘在上半部)", 1),
    "pt_shooting_star20": _spec(_shooting_star20, "近20日射击之星占比(上影>0.5且实体<0.3且收盘在下半部)", -1),
    "pt_shadow_asym20": _spec(_shadow_asym20, "近20日平均上影/平均下影（影线不对称）", -1),
    # --- 日内位置 / 实体结构 ---
    "pt_shape20": _spec(_shape20, "近20日平均日内收盘位置 (C-L)/(H-L)", 1),
    "pt_open_strength20": _spec(_open_strength20, "近20日平均开盘位置 (O-L)/(H-L)", -1),
    "pt_close_vs_open20": _spec(_close_vs_open20, "近20日 mean(C-O)/mean(H-L)", -1),
    "pt_intraday_skew20": _spec(_intraday_skew20, "近20日 (C-O)/(H-L) 的标准差", -1),
    "pt_close_loc_std20": _spec(_close_loc_std20, "近20日 (C-L)/(H-L) 的标准差", -1),
    "pt_eod_reversal20": _spec(_eod_reversal20, "近20日 (C-(H+L)/2)/(H-L) 均值（尾盘反转强度）", -1),
    "pt_upper_touch20": _spec(_upper_touch20, "近20日收盘贴近当日最高(上影<0.1)天数占比", 1),
    "pt_lower_touch20": _spec(_lower_touch20, "近20日收盘贴近当日最低(下影<0.1)天数占比", -1),
    # --- 振幅 / 收缩 ---
    "pt_range_ratio20": _spec(lambda t, C: _range_ratio(t, C, 20), "近20日平均振幅 (H-L)/C", -1),
    "pt_range_ratio60": _spec(lambda t, C: _range_ratio(t, C, 60), "近60日平均振幅 (H-L)/C", -1),
    "pt_range_trend": _spec(_range_trend, "振幅20/振幅60-1：振幅扩张", -1),
    "pt_vol_contract20": _spec(_vol_contract20, "近10日平均振幅/近60日平均振幅（波动收缩）", -1),
    "pt_range_expand20": _spec(_range_expand20, "近5日平均振幅/近20日平均振幅-1", -1),
    "pt_nr7_count20": _spec(_nr7_count20, "近20日中「7日最窄振幅」(NR7)出现次数", -1),
    # --- 跳空 / 内外包 / 高低点 ---
    "pt_gap_up20": _spec(_gap_up20, "近20日向上跳空(开>昨高)天数占比", -1),
    "pt_gap_dn20": _spec(_gap_dn20, "近20日向下跳空(开<昨低)天数占比", 1),
    "pt_gap_fill_rate20": _spec(_gap_fill_rate20, "近20日向上跳空当日回补的缺口幅度占比", -1),
    "pt_inside_days20": _spec(_inside_days20, "近20日内包线(H<=昨高且L>=昨低)天数占比", -1),
    "pt_outside_days20": _spec(_outside_days20, "近20日外包线(H>=昨高且L<=昨低)天数占比", 1),
    "pt_hh_hl_count20": _spec(_hh_hl_count20, "近20日高低点同时抬高天数占比", 1),
    "pt_ll_lh_count20": _spec(_ll_lh_count20, "近20日高低点同时降低天数占比", -1),
    # --- 连续涨跌 ---
    "pt_up_days20": _spec(lambda t, C: _up_days(t, C, 20), "近20日上涨(R>0)天数占比", -1),
    "pt_up_days60": _spec(lambda t, C: _up_days(t, C, 60), "近60日上涨(R>0)天数占比", -1),
    "pt_run_up": _spec(_run_up, "近20日最长连阳天数(停牌跳过、十字星打断)", -1),
    "pt_run_dn": _spec(_run_dn, "近20日最长连阴天数(停牌跳过、十字星打断)", 1),
    # --- 路径结构 ---
    "pt_hl_autocorr60": _spec(_hl_autocorr60, "近60日 (H-L)/C 一阶自相关", 1),
    "pt_ret_autocorr_neg20": _spec(_ret_autocorr_neg20, "近20日收益一阶自相关取负", 1),
    "pt_path_entropy20": _spec(_path_entropy20, "近20日收盘位置符号序列的二元Shannon熵", -1),
    "pt_intra_vs_ovn20": _spec(_intra_ovn20, "近20日日内累计收益(复利)-隔夜累计收益(复利)", -1),
}