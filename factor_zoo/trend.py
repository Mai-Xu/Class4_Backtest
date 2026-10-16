# -*- coding: utf-8 -*-
"""趋势与形态类因子（trend，前缀 tr_）。

覆盖清单
--------
- 均线偏离   tr_madev20 / tr_madev60
- 均线排列   tr_ma_align
- 趋势强度   tr_linslope60 / tr_linslope120 / tr_linr2_60 / tr_adx14
- 通道突破   tr_bb_pos20 / tr_donch20 / tr_donch55
- 形态统计   tr_upratio60 / tr_maxdd60 / tr_er20 / tr_streak
- 同族扩展   tr_clv20 / tr_gap20

口径
----
- 价格一律用复权口径（C.C / C.ADJ_O / C.ADJ_H / C.ADJ_L / C.R）；
  区间位置类都是同口径的比值，与复权因子无关，高低价直接取 C.ADJ_H / C.ADJ_L。
- 只使用下标 <= t 的行（注册表用 Ctx.truncated 做因果性检验）。
- 窗口统计一律走 Ctx 掩码版 mean / std；有效样本不足返回 NaN，绝不出现 inf。
- 返回 shape=(C.k,) 的 float64。

缺失处理约定
------------
- 停牌 / 未上市位置的价格是 NaN，掩码统计自动跳过。
- 每个窗口要求「有效样本数 >= max(3, ceil(0.6 * 窗口))」，不满足则 NaN。
"""
from __future__ import annotations

import numpy as np

from .ctx import Ctx

_YEAR = 252.0        # 年化交易日
_MIN_FRAC = 0.6      # 窗口最小有效样本占比


# ------------------------------------------------------------------ #
# 通用工具
# ------------------------------------------------------------------ #
def _min_cnt(w: int, frac: float = _MIN_FRAC) -> int:
    """窗口 w 内要求的最少有效样本数。"""
    return max(3, int(np.ceil(frac * w)))


def _nan(C: Ctx) -> np.ndarray:
    return np.full(C.k, np.nan, dtype="float64")


def _masked_ma(t: int, C: Ctx, w: int, X: np.ndarray):
    """窗口 [t-w+1, t] 的掩码均值 -> (MA, 有效样本数)。"""
    _, W = C.sl(t, w, X)
    ok = np.isfinite(W)
    return C.mean(W, ok, _min_cnt(w)), ok.sum(0).astype("float64")


# ------------------------------------------------------------------ #
# 1) 均线偏离：close / MA(w) - 1
# ------------------------------------------------------------------ #
def _ma_dev(t: int, C: Ctx, w: int) -> np.ndarray:
    ma, _ = _masked_ma(t, C, w, C.C)
    px = C.C[t]
    good = np.isfinite(ma) & (ma > 0.0) & np.isfinite(px)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(good, px / np.where(good, ma, 1.0) - 1.0, np.nan)


# ------------------------------------------------------------------ #
# 2) 均线排列：多头排列强度
# ------------------------------------------------------------------ #
def _ma_align(t: int, C: Ctx) -> np.ndarray:
    """(MA5-MA20)/MA20 + (MA20-MA60)/MA60，>0 表示多头排列（MA5>MA20>MA60）。"""
    ma5, _ = _masked_ma(t, C, 5, C.C)
    ma20, _ = _masked_ma(t, C, 20, C.C)
    ma60, _ = _masked_ma(t, C, 60, C.C)
    good = (np.isfinite(ma5) & np.isfinite(ma20) & np.isfinite(ma60)
            & (ma20 > 0.0) & (ma60 > 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        v = ((ma5 - ma20) / np.where(good, ma20, 1.0)
             + (ma20 - ma60) / np.where(good, ma60, 1.0))
    return np.where(good, v, np.nan)


# ------------------------------------------------------------------ #
# 3) 趋势强度：对数价回归斜率 / R² / ADX 代理
# ------------------------------------------------------------------ #
def _linfit(t: int, C: Ctx, w: int):
    """对数复权价对时间（窗口内行号）做掩码最小二乘 -> (年化斜率, R²)。"""
    _, W = C.sl(t, w, C.C)
    ok = np.isfinite(W) & (W > 0.0)
    cnt = ok.sum(0).astype("float64")
    x = np.arange(W.shape[0], dtype="float64")[:, None]
    y = np.where(ok, np.log(np.where(ok, W, 1.0)), 0.0)
    xm = np.where(ok, x, 0.0).sum(0) / np.maximum(cnt, 1.0)
    ym = y.sum(0) / np.maximum(cnt, 1.0)
    dx = np.where(ok, x - xm, 0.0)
    dy = np.where(ok, y - ym, 0.0)
    sxx = (dx * dx).sum(0)
    sxy = (dx * dy).sum(0)
    syy = (dy * dy).sum(0)
    good = (cnt >= _min_cnt(w)) & (sxx > 0.0)
    use = good & (syy > 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = np.where(good, _YEAR * sxy / np.where(good, sxx, 1.0), np.nan)
        r2 = np.where(use, (sxy * sxy) / np.where(use, sxx * syy, 1.0), np.nan)
    return slope, np.clip(r2, 0.0, 1.0)


def _adx_proxy(t: int, C: Ctx, n: int = 14) -> np.ndarray:
    """ADX 代理：+DM / -DM / TR 的 n 日滚动和 -> DI -> DX，再取最近 n 个 DX 的均值。

    经典 Wilder 平滑在窗口内用「n 日滚动和」近似，输出 0~100 的趋势强度；
    只读最近 2n+1 = 29 行，天然满足因果性。
    """
    need = 2 * n + 1
    _, H = C.sl(t, need, C.ADJ_H)
    m = H.shape[0] - 1                       # 可用的日间 TR 个数
    if m < n:
        return _nan(C)
    _, L = C.sl(t, need, C.ADJ_L)
    _, CL = C.sl(t, need, C.C)
    hc, hp = H[1:], H[:-1]
    lc, lp = L[1:], L[:-1]
    cp = CL[:-1]
    ok = (np.isfinite(hc) & np.isfinite(lc) & np.isfinite(hp)
          & np.isfinite(lp) & np.isfinite(cp))
    tr = np.maximum(np.maximum(hc - lc, np.abs(hc - cp)), np.abs(lc - cp))
    up = hc - hp
    dn = lp - lc
    pdm = np.where((up > dn) & (up > 0.0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0.0), dn, 0.0)
    tr = np.where(ok, tr, 0.0)
    pdm = np.where(ok, pdm, 0.0)
    ndm = np.where(ok, ndm, 0.0)
    zero = np.zeros((1, C.k))
    cntc = np.concatenate([zero, np.cumsum(ok.astype("float64"), axis=0)], axis=0)
    trc = np.concatenate([zero, np.cumsum(tr, axis=0)], axis=0)
    pc = np.concatenate([zero, np.cumsum(pdm, axis=0)], axis=0)
    nc = np.concatenate([zero, np.cumsum(ndm, axis=0)], axis=0)
    idx = np.arange(n - 1, m)                # 各滚动 n 日窗口的右端（TR 行号）
    cnt_w = cntc[idx + 1] - cntc[idx + 1 - n]
    trs = trc[idx + 1] - trc[idx + 1 - n]
    ps = pc[idx + 1] - pc[idx + 1 - n]
    ns = nc[idx + 1] - nc[idx + 1 - n]
    good = (cnt_w >= _min_cnt(n, 0.8)) & (trs > 0.0)
    den = np.where(good, trs, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        dip = 100.0 * ps / den
        din = 100.0 * ns / den
        tot = dip + din
        dx = np.where(
            good,
            np.where(tot > 0.0, 100.0 * np.abs(dip - din) / np.where(tot > 0.0, tot, 1.0), 0.0),
            np.nan,
        )
    okdx = np.isfinite(dx)
    adx = C.mean(dx, okdx, max(3, n - 3))
    return np.where(np.isfinite(dx[-1]), adx, np.nan)


# ------------------------------------------------------------------ #
# 4) 通道突破：布林带位置 / 唐奇安通道位置
# ------------------------------------------------------------------ #
def _bb_pos(t: int, C: Ctx, w: int = 20) -> np.ndarray:
    """布林带位置：(close - MA20) / (2 * Std20)，0 在中轨、±1 在上下轨。"""
    _, W = C.sl(t, w, C.C)
    ok = np.isfinite(W)
    ma = C.mean(W, ok, _min_cnt(w))
    sd = C.std(W, ok, _min_cnt(w))
    px = C.C[t]
    good = np.isfinite(ma) & np.isfinite(sd) & (sd > 0.0) & np.isfinite(px)
    with np.errstate(invalid="ignore", divide="ignore"):
        pos = (px - ma) / np.where(good, 2.0 * sd, 1.0)
    return np.where(good, pos, np.nan)


def _donchian(t: int, C: Ctx, w: int) -> np.ndarray:
    """唐奇安通道位置：(close - min(L,w)) / (max(H,w) - min(L,w))，0~1。"""
    _, LW = C.sl(t, w, C.ADJ_L)
    _, HW = C.sl(t, w, C.ADJ_H)
    ok_l = np.isfinite(LW)
    ok_h = np.isfinite(HW)
    cnt = np.minimum(ok_l.sum(0), ok_h.sum(0)).astype("float64")
    lmin = np.where(ok_l, LW, np.inf).min(0)
    hmax = np.where(ok_h, HW, -np.inf).max(0)
    px = C.C[t]
    rng = hmax - lmin
    good = (cnt >= _min_cnt(w)) & np.isfinite(rng) & (rng > 0.0) & np.isfinite(px)
    with np.errstate(invalid="ignore", divide="ignore"):
        pos = (px - lmin) / np.where(good, rng, 1.0)
    return np.where(good, pos, np.nan)


# ------------------------------------------------------------------ #
# 5) 形态统计
# ------------------------------------------------------------------ #
def _up_ratio(t: int, C: Ctx, w: int) -> np.ndarray:
    """过去 w 日上涨（复权收益 > 0）天数占比，0~1。"""
    _, W = C.sl(t, w, C.R)
    ok = np.isfinite(W)
    up = ok & (W > 0.0)
    cnt = ok.sum(0).astype("float64")
    cu = up.sum(0).astype("float64")
    good = cnt >= _min_cnt(w)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cu / np.where(good, cnt, 1.0)
    return np.where(good, r, np.nan)


def _max_dd(t: int, C: Ctx, w: int) -> np.ndarray:
    """过去 w 日区间最大回撤：max(1 - close / cummax(close))，0~1。"""
    _, W = C.sl(t, w, C.C)
    ok = np.isfinite(W)
    cnt = ok.sum(0).astype("float64")
    run = np.fmax.accumulate(np.where(ok, W, -np.inf), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        dd = 1.0 - np.where(ok, W, np.nan) / run
    dd = np.where(np.isfinite(dd), dd, np.nan)
    mx = np.where(np.isfinite(dd), dd, -np.inf).max(0)
    good = (cnt >= max(5, _min_cnt(w))) & np.isfinite(mx)
    return np.where(good, np.clip(mx, 0.0, 1.0), np.nan)


def _eff_ratio(t: int, C: Ctx, w: int) -> np.ndarray:
    """有符号 Kaufman 路径效率：净变动 / 路径总长度，裁剪到 -1~1。"""
    _, W = C.sl(t, w, C.C)
    ok = np.isfinite(W)
    dP = np.diff(W, axis=0)
    okd = np.isfinite(dP)
    path = np.where(okd, np.abs(dP), 0.0).sum(0)
    cntd = okd.sum(0).astype("float64")
    first = np.argmax(ok, axis=0)             # 窗口内首个有效行（全 NaN 列取 0）
    p0 = W[first, np.arange(C.k)]
    good = (ok[-1] & np.isfinite(p0) & (path > 0.0) & (cntd >= _min_cnt(w - 1)))
    with np.errstate(invalid="ignore", divide="ignore"):
        er = (W[-1] - p0) / np.where(good, path, 1.0)
    return np.where(good, np.clip(er, -1.0, 1.0), np.nan)


def _streak(t: int, C: Ctx, w: int = 20) -> np.ndarray:
    """截至 t 的连续涨/跌天数（带符号），连涨 +k、连跌 -k，窗口上限 w。"""
    _, W = C.sl(t, w, C.R)
    ok = np.isfinite(W)
    S = np.where(ok & (W > 0.0), 1.0, np.where(ok & (W < 0.0), -1.0, 0.0))
    sgn = S[-1]
    match = ok & (sgn[None, :] != 0.0) & (S == sgn[None, :])
    run = np.zeros(C.k, dtype="float64")
    active = np.ones(C.k, dtype=bool)
    for i in range(W.shape[0] - 1, -1, -1):  # 自 t 往前数同号连续的个数
        active &= match[i]
        run += active
    return np.where(ok[-1] & (sgn != 0.0), run * sgn, np.nan)


# ------------------------------------------------------------------ #
# 6) 同族扩展：收盘日内位置 / 隔夜跳空
# ------------------------------------------------------------------ #
def _clv(t: int, C: Ctx, w: int = 20) -> np.ndarray:
    """收盘价在当日区间的位置 (C-L)/(H-L) 的 w 日均值，0~1（越大收盘越强）。"""
    _, LW = C.sl(t, w, C.ADJ_L)
    _, HW = C.sl(t, w, C.ADJ_H)
    _, CW = C.sl(t, w, C.C)
    rng = HW - LW
    ok = np.isfinite(rng) & (rng > 0.0) & np.isfinite(CW)
    clv = np.where(ok, (CW - LW) / np.where(ok, rng, 1.0), np.nan)
    clv = np.clip(clv, 0.0, 1.0)             # 防脏数据越界（一字板/停牌已排除）
    return C.mean(clv, ok, _min_cnt(w))


def _gap(t: int, C: Ctx, w: int = 20) -> np.ndarray:
    """隔夜跳空（复权开盘 / 前一日复权收盘 - 1）的 w 日均值，>0 为平均高开。"""
    need = w + 1
    _, OW = C.sl(t, need, C.ADJ_O)
    _, CW = C.sl(t, need, C.C)
    if OW.shape[0] < 2:
        return _nan(C)
    prev = CW[:-1]
    ok = np.isfinite(OW[1:]) & np.isfinite(prev) & (prev > 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        g = OW[1:] / np.where(ok, prev, 1.0) - 1.0
    return C.mean(g, ok, _min_cnt(w))


# ------------------------------------------------------------------ #
# 注册表
# ------------------------------------------------------------------ #
FACTORS = {
    "tr_madev20": dict(func=lambda t, md, C: _ma_dev(t, C, 20),
                       desc="复权收盘 / MA20 - 1（20 日均线偏离，>0 高于均线）", direction=-1),
    "tr_madev60": dict(func=lambda t, md, C: _ma_dev(t, C, 60),
                       desc="复权收盘 / MA60 - 1（60 日均线偏离）", direction=-1),
    "tr_ma_align": dict(func=lambda t, md, C: _ma_align(t, C),
                        desc="多头排列强度 (MA5-MA20)/MA20 + (MA20-MA60)/MA60", direction=1),
    "tr_linslope60": dict(func=lambda t, md, C: _linfit(t, C, 60)[0],
                          desc="对数复权价 60 日回归斜率（年化，趋势方向与速度）", direction=1),
    "tr_linslope120": dict(func=lambda t, md, C: _linfit(t, C, 120)[0],
                           desc="对数复权价 120 日回归斜率（年化）", direction=1),
    "tr_linr2_60": dict(func=lambda t, md, C: _linfit(t, C, 60)[1],
                        desc="60 日对数价回归 R2（趋势直线程度，0~1）", direction=1),
    "tr_adx14": dict(func=lambda t, md, C: _adx_proxy(t, C, 14),
                     desc="ADX 代理（+DM/-DM/TR 滚动平滑 14 日，0~100，越高趋势越强）", direction=1),
    "tr_bb_pos20": dict(func=lambda t, md, C: _bb_pos(t, C, 20),
                        desc="布林带位置 (close - MA20) / (2 * Std20)", direction=-1),
    "tr_donch20": dict(func=lambda t, md, C: _donchian(t, C, 20),
                       desc="唐奇安 20 日通道位置 (close - min(L,20)) / (max(H,20) - min(L,20))",
                       direction=1),
    "tr_donch55": dict(func=lambda t, md, C: _donchian(t, C, 55),
                       desc="唐奇安 55 日通道位置", direction=1),
    "tr_upratio60": dict(func=lambda t, md, C: _up_ratio(t, C, 60),
                         desc="过去 60 日上涨天数占比（0~1）", direction=-1),
    "tr_maxdd60": dict(func=lambda t, md, C: _max_dd(t, C, 60),
                       desc="过去 60 日区间最大回撤（1 - close/cummax，0~1）", direction=-1),
    "tr_er20": dict(func=lambda t, md, C: _eff_ratio(t, C, 20),
                    desc="20 日有符号 Kaufman 路径效率（净变动/路径总长，-1~1）", direction=1),
    "tr_streak": dict(func=lambda t, md, C: _streak(t, C, 20),
                      desc="连续涨(正)/跌(负)天数（20 日窗口内，带符号）", direction=-1),
    "tr_clv20": dict(func=lambda t, md, C: _clv(t, C, 20),
                     desc="20 日均收盘日内位置 (C-L)/(H-L)，0~1（越大收盘越强）", direction=1),
    "tr_gap20": dict(func=lambda t, md, C: _gap(t, C, 20),
                     desc="20 日平均隔夜跳空收益（复权开盘/昨收 - 1）", direction=-1),
}