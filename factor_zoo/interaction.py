# -*- coding: utf-8 -*-
"""因子交互 / 非线性变换因子（factor_zoo 分类：interaction，前缀 ix_）。

本模块实现《因子模块开发规范》§5.9 列出的全部 ix_* 因子，另加 4 个同族扩展，共 24 个。

设计约定
--------
- 基础量（区间收益 / 波动 / 下行波动 / 偏度 / MAX / Amihud 非流动性 / 成交额相对度 /
  beta / 量价相关 / 量价背离 / 筹码获利盘 / 振幅）全部在本模块内自行实现，
  不 import 其它因子模块。
- 交互项统一做法：先算基础量 -> 做**当日横截面** z 分数（``cs_zscore``，2%/98% 去极值）
  -> 逐股相乘。``cs_zscore`` 只使用当日横截面行，不使用任何 t 之后的数据，因此不破坏
  ``Ctx.truncated(t)`` 的因果性检验。
- 窗口统计一律用 ``Ctx.sl / Ctx.stats / Ctx.mean / Ctx.std`` 掩码版矩阵运算，不做逐股票
  Python 循环；无法计算的位置安静返回 NaN；出口统一过 ``_fin``（绝不返回 ±inf）。
- 可复用的中间结果缓存在 ``Ctx`` 实例的 ``_ix_cache`` 上（键含 t）。``Ctx.truncated(t)``
  是全新对象，缓存天然隔离；模块内没有任何模块级 / 全局缓存。
- ``direction`` 是先验方向（+1 = 值越大越好，-1 = 值越小越好），最终以样本内 IC 为准。
  两个同号分量相乘时组合符号天然有歧义，本模块按规范表给定的方向原样填写。
"""
from __future__ import annotations

import numpy as np

from .ctx import Ctx, cs_zscore

_EPS = 1e-12
_D1M = 21


# ------------------------------------------------------------------ #
# 通用工具
# ------------------------------------------------------------------ #
def _nan(k: int) -> np.ndarray:
    """全 NaN 的横截面向量。"""
    return np.full(int(k), np.nan, dtype="float64")


def _fin(x) -> np.ndarray:
    """出口统一把 ±inf / NaN 收敛成 NaN。"""
    x = np.asarray(x, dtype="float64")
    return np.where(np.isfinite(x), x, np.nan)


def _need(w: int) -> int:
    """窗口最低有效样本数：短窗 max(3, ceil(0.6w))，长窗（>=120）max(20, ceil(0.6w))。"""
    w = int(w)
    if w <= 0:
        return 1
    floor = 20 if w >= 120 else 3
    return max(1, min(w, max(floor, int(np.ceil(0.6 * w)))))


def _div(num, den, positive: bool = False) -> np.ndarray:
    """安全除法：分母无效 / 过小时返回 NaN，绝不产生 inf。"""
    num, den = np.broadcast_arrays(np.asarray(num, dtype="float64"),
                                   np.asarray(den, dtype="float64"))
    good = np.isfinite(num) & np.isfinite(den)
    good = good & ((den > _EPS) if positive else (np.abs(den) > _EPS))
    out = np.full(num.shape, np.nan, dtype="float64")
    np.divide(num, den, out=out, where=good)
    return out


def _z(x) -> np.ndarray:
    """当日横截面 z 分数（2%/98% 去极值 + 标准化；常数截面返回 0）。"""
    return cs_zscore(x)


def _mavg(X, ok) -> np.ndarray:
    """掩码均值（不设最小样本数，调用方自己判）。"""
    return _div(np.where(ok, X, 0.0).sum(0), ok.sum(0).astype("float64"), positive=True)


def _mcorr(x, y, ok, min_cnt: int) -> np.ndarray:
    """掩码 Pearson 相关；有效样本不足或方差为 0 时返回 NaN。"""
    cnt = ok.sum(0).astype("float64")
    mx = _mavg(x, ok)
    my = _mavg(y, ok)
    xc = np.where(ok, x - mx, 0.0)
    yc = np.where(ok, y - my, 0.0)
    num = (xc * yc).sum(0)
    den = np.sqrt((xc * xc).sum(0) * (yc * yc).sum(0))
    out = np.clip(_div(num, den, positive=True), -1.0, 1.0)
    return _fin(np.where(cnt >= float(min_cnt), out, np.nan))


def _consensus(vals) -> np.ndarray:
    """多个 z 分数的等权均值（至少 2 个分量有效，否则 NaN）。"""
    M = np.vstack([np.asarray(v, dtype="float64") for v in vals])
    ok = np.isfinite(M)
    cnt = ok.sum(0).astype("float64")
    s = np.where(ok, M, 0.0).sum(0)
    return _fin(np.where(cnt >= 2.0, s / np.maximum(cnt, 1.0), np.nan))


def _cs_rank(x) -> np.ndarray:
    """当日横截面分位（0~1，平均秩），只用当日行。"""
    v = np.asarray(x, dtype="float64")
    out = _nan(v.shape[0])
    idx = np.flatnonzero(np.isfinite(v))
    n = idx.size
    if n < 3:
        return out
    r = np.empty(n, dtype="float64")
    r[np.argsort(v[idx], kind="mergesort")] = np.arange(n, dtype="float64")
    out[idx] = (r + 0.5) / float(n)
    return out


def _cached(fn):
    """把基础量缓存到 Ctx 实例（键含 t 与窗口参数）；禁止模块级缓存。"""
    def wrapper(t, C, *args):
        cache = C.__dict__.setdefault("_ix_cache", {})
        key = (fn.__name__, int(t)) + tuple(int(a) for a in args)
        hit = cache.get(key)
        if hit is None:
            if len(cache) >= 512:      # 有界：超限整体清空（只影响重算，不影响正确性）
                cache.clear()
            hit = fn(t, C, *args)
            cache[key] = hit
        return hit
    wrapper.__name__ = fn.__name__
    return wrapper


# ------------------------------------------------------------------ #
# 基础量（本模块内自实现，只切 [0, t] 行）
# ------------------------------------------------------------------ #
@_cached
def _ret(t: int, C: Ctx, w: int) -> np.ndarray:
    """复权区间收益 C[t] / C[t-w] - 1（端点缺失返回 NaN）。"""
    i1, i0 = int(t), int(t) - int(w)
    if i0 < 0:
        return _nan(C.k)
    a, b = C.C[i1], C.C[i0]
    good = np.isfinite(a) & np.isfinite(b) & (b > _EPS)
    return _fin(np.where(good, a / np.where(good, b, 1.0) - 1.0, np.nan))


@_cached
def _vol(t: int, C: Ctx, w: int = 60) -> np.ndarray:
    """窗口内复权日收益标准差（ddof=1）。"""
    lo, W = C.sl(t, w, C.R)
    ok = np.isfinite(W)
    return _fin(C.std(W, ok, _need(w), ddof=1))


@_cached
def _dvol(t: int, C: Ctx, w: int = 60) -> np.ndarray:
    """下行半偏差 sqrt(sum(min(r,0)^2) / n)，n = 窗口内全部有效样本数。"""
    lo, W = C.sl(t, w, C.R)
    ok = np.isfinite(W)
    cnt = ok.sum(0).astype("float64")
    X = np.where(ok, np.minimum(W, 0.0), 0.0)
    out = np.sqrt((X ** 2).sum(0) / np.maximum(cnt, 1.0))
    return _fin(np.where(cnt >= _need(w), out, np.nan))


@_cached
def _skew(t: int, C: Ctx, w: int = 60) -> np.ndarray:
    """窗口内日收益偏度 m3 / m2^1.5。"""
    s = C.stats(t, w)
    cnt = s["cnt"]
    Z = s["Wc"]
    n = np.maximum(cnt, 1.0)
    m2 = (Z ** 2).sum(0) / n
    m3 = (Z ** 3).sum(0) / n
    good = (cnt >= _need(w)) & (m2 > _EPS)
    return _fin(np.where(good, m3 / np.where(good, m2, 1.0) ** 1.5, np.nan))


@_cached
def _maxret(t: int, C: Ctx, w: int = 20) -> np.ndarray:
    """窗口内最大单日复权收益（彩票 MAX）。"""
    lo, W = C.sl(t, w, C.R)
    ok = np.isfinite(W)
    cnt = ok.sum(0).astype("float64")
    mx = np.where(ok, W, -np.inf).max(0)
    good = (cnt >= _need(w)) & np.isfinite(mx)
    return _fin(np.where(good, mx, np.nan))


@_cached
def _amihud(t: int, C: Ctx, w: int = 60) -> np.ndarray:
    """Amihud 非流动性 mean(|R| / A, w) × 1e8（只统计 A > 0 的交易日）。"""
    lo, W = C.sl(t, w, C.R)
    A = C.A[lo:int(t) + 1]
    ok = np.isfinite(W) & np.isfinite(A) & (A > 0.0)
    X = np.where(ok, np.abs(W), 0.0) / np.where(ok, A, 1.0)
    mu = _mavg(X, ok)
    return _fin(np.where(ok.sum(0) >= _need(w), mu * 1e8, np.nan))


@_cached
def _amt_rel(t: int, C: Ctx, w: int = 60) -> np.ndarray:
    """成交额相对度：A[t] / mean(A, w) - 1。"""
    lo, W = C.sl(t, w, C.A)
    ok = np.isfinite(W) & (W > 0.0)
    mu = _mavg(W, ok)
    good = (ok.sum(0) >= _need(w)) & np.isfinite(mu) & (mu > 0.0)
    a = C.A[int(t)]
    good = good & np.isfinite(a) & (a >= 0.0)
    return _fin(np.where(good, a / np.where(good, mu, 1.0) - 1.0, np.nan))


@_cached
def _logamt(t: int, C: Ctx, w: int = 60) -> np.ndarray:
    """log(窗口平均成交额)，规模 / 关注度代理。"""
    lo, W = C.sl(t, w, C.A)
    ok = np.isfinite(W) & (W > 0.0)
    mu = _mavg(W, ok)
    good = (ok.sum(0) >= _need(w)) & np.isfinite(mu) & (mu > 0.0)
    return _fin(np.where(good, np.log(np.where(good, mu, 1.0)), np.nan))


@_cached
def _mom(t: int, C: Ctx, months: int = 12) -> np.ndarray:
    """x-1 动量：C[t-21] / C[t-21-21x] - 1（跳过最近 1 个月）。"""
    i1 = int(t) - _D1M
    i0 = int(t) - _D1M - int(months) * _D1M
    if i0 < 0:
        return _nan(C.k)
    a, b = C.C[i1], C.C[i0]
    good = np.isfinite(a) & np.isfinite(b) & (b > _EPS)
    return _fin(np.where(good, a / np.where(good, b, 1.0) - 1.0, np.nan))


@_cached
def _beta(t: int, C: Ctx, w: int = 60) -> np.ndarray:
    """市场模型 beta：窗口内 R 对 C.mkt 的掩码配对 OLS 斜率。"""
    lo, W = C.sl(t, w, C.R)
    m = np.asarray(C.mkt[lo:int(t) + 1], dtype="float64")[:, None]
    ok = np.isfinite(W) & np.isfinite(m)
    cnt = ok.sum(0).astype("float64")
    n = np.maximum(cnt, 1.0)
    mx = np.where(ok, m, 0.0).sum(0) / n
    my = np.where(ok, W, 0.0).sum(0) / n
    xc = np.where(ok, m - mx, 0.0)
    yc = np.where(ok, W - my, 0.0)
    sxx = (xc * xc).sum(0)
    sxy = (xc * yc).sum(0)
    good = (cnt >= _need(w)) & (sxx > _EPS)
    return _fin(np.where(good, sxy / np.where(good, sxx, 1.0), np.nan))


@_cached
def _nearhigh(t: int, C: Ctx, w: int = 250) -> np.ndarray:
    """52 周高点接近度 C[t] / max(ADJ_H, w) - 1（<= 0，越接近 0 越强）。"""
    lo, H = C.sl(t, w, C.ADJ_H)
    ok = np.isfinite(H)
    cnt = ok.sum(0).astype("float64")
    hmax = np.where(ok, H, -np.inf).max(0)
    px = C.C[int(t)]
    good = (cnt >= _need(w)) & (hmax > _EPS) & np.isfinite(px)
    return _fin(np.where(good, px / np.where(good, hmax, 1.0) - 1.0, np.nan))


@_cached
def _dlogamt(t: int, C: Ctx, w: int = 20) -> np.ndarray:
    """Δlog 成交额：log(近 w 日均额 / 前 w 日均额)。"""
    if int(t) < 2 * int(w) - 1:
        return _nan(C.k)
    lo = int(t) - 2 * int(w) + 1
    A = C.A[lo:int(t) + 1]
    ok = np.isfinite(A) & (A > 0.0)
    n0 = ok[:w].sum(0).astype("float64")
    n1 = ok[w:].sum(0).astype("float64")
    m0 = np.where(ok[:w], A[:w], 0.0).sum(0) / np.maximum(n0, 1.0)
    m1 = np.where(ok[w:], A[w:], 0.0).sum(0) / np.maximum(n1, 1.0)
    good = (n0 >= _need(w)) & (n1 >= _need(w)) & (m0 > _EPS) & (m1 > _EPS)
    return _fin(np.where(good, np.log(np.where(good, m1, 1.0) / np.where(good, m0, 1.0)),
                         np.nan))


@_cached
def _pvdiv(t: int, C: Ctx, w: int = 20) -> np.ndarray:
    """量价背离：z(近 w 日收益) - z(Δlog 成交额 w)（>0 = 价涨但量能不配合）。"""
    return _fin(_z(_ret(t, C, w)) - _z(_dlogamt(t, C, w)))


@_cached
def _pvcorr(t: int, C: Ctx, w: int = 20) -> np.ndarray:
    """量价相关：corr(R, Δlog A) 的 w 日掩码相关。"""
    i1 = int(t)
    i0 = max(i1 - int(w) + 1, 1)
    if i1 < i0:
        return _nan(C.k)
    A1 = C.A[i0:i1 + 1]
    A0 = C.A[i0 - 1:i1]
    R1 = C.R[i0:i1 + 1]
    ok = (np.isfinite(A1) & np.isfinite(A0) & np.isfinite(R1)
          & (A1 > 0.0) & (A0 > 0.0))
    dl = np.log(np.where(ok, A1, 1.0) / np.where(ok, A0, 1.0))
    return _mcorr(R1, dl, ok, _need(w))


@_cached
def _profit(t: int, C: Ctx, w: int = 60) -> np.ndarray:
    """获利盘比例：窗口内典型价 < 现价的成交额占比（成交额加权）。"""
    lo, H = C.sl(t, w, C.ADJ_H)
    _, L = C.sl(t, w, C.ADJ_L)
    _, Cl = C.sl(t, w, C.C)
    _, A = C.sl(t, w, C.A)
    px = C.C[int(t)]
    TP = (H + L + Cl) / 3.0
    ok = np.isfinite(TP) & np.isfinite(A) & (A > 0.0) & np.isfinite(px)
    tot = np.where(ok, A, 0.0).sum(0)
    num = np.where(ok & (TP < px), A, 0.0).sum(0)
    good = (ok.sum(0) >= _need(w)) & (tot > _EPS)
    return _fin(np.where(good, num / np.where(good, tot, 1.0), np.nan))


@_cached
def _amp(t: int, C: Ctx, w: int = 20) -> np.ndarray:
    """振幅：mean((ADJ_H - ADJ_L) / C, w)。"""
    lo, H = C.sl(t, w, C.ADJ_H)
    _, L = C.sl(t, w, C.ADJ_L)
    _, Cl = C.sl(t, w, C.C)
    ok = np.isfinite(H) & np.isfinite(L) & np.isfinite(Cl) & (Cl > _EPS)
    X = np.where(ok, H - L, 0.0) / np.where(ok, Cl, 1.0)
    mu = _mavg(X, ok)
    return _fin(np.where(ok.sum(0) >= _need(w), mu, np.nan))


# ------------------------------------------------------------------ #
# 交互项
# ------------------------------------------------------------------ #
def _rev5(t, C):
    return -_ret(t, C, 5)


def _rev20(t, C):
    return -_ret(t, C, 20)


def _lowvol(t, C):
    return -_vol(t, C, 60)


def _lowamt(t, C):
    return -_amt_rel(t, C, 60)


def _ix_rev_x_illiq(t, C):
    return _fin(_z(_rev20(t, C)) * _z(_amihud(t, C, 60)))


def _ix_rev_x_lowvol(t, C):
    return _fin(_z(_rev20(t, C)) * _z(_lowvol(t, C)))


def _ix_mom_x_lowvol(t, C):
    return _fin(_z(_mom(t, C, 12)) * _z(_lowvol(t, C)))


def _ix_mom_x_lowamt(t, C):
    return _fin(_z(_mom(t, C, 12)) * _z(_lowamt(t, C)))


def _ix_nearhigh_x_amt(t, C):
    return _fin(_z(_nearhigh(t, C, 250)) * _z(_amt_rel(t, C, 60)))


def _ix_max_x_illiq(t, C):
    return _fin(_z(_maxret(t, C, 20)) * _z(_amihud(t, C, 60)))


def _ix_skew_x_vol(t, C):
    return _fin(_z(_skew(t, C, 60)) * _z(_vol(t, C, 60)))


def _ix_rev_x_size(t, C):
    return _fin(_z(_rev5(t, C)) * _z(_logamt(t, C, 60)))


def _ix_mom_x_beta(t, C):
    return _fin(_z(_mom(t, C, 12)) * _z(_beta(t, C, 60)))


def _ix_pvdiv_x_turn(t, C):
    return _fin(_z(_pvdiv(t, C, 20)) * _z(_amt_rel(t, C, 60)))


def _ix_chip_x_rev(t, C):
    return _fin(_z(_profit(t, C, 60)) * _z(_rev5(t, C)))


def _ix_amt_x_ret20(t, C):
    return _fin(_z(_amt_rel(t, C, 60)) * _z(_ret(t, C, 20)))


def _ix_vol_x_range(t, C):
    return _fin(_z(_vol(t, C, 60)) * _z(_amp(t, C, 20)))


def _ix_illiq_x_lowvol(t, C):
    return _fin(_z(_amihud(t, C, 60)) * _z(_lowvol(t, C)))


def _ix_rev_sq(t, C):
    r = _ret(t, C, 5)
    zr = _z(r)
    sgn = np.where(np.isfinite(r), np.sign(r), np.nan)
    return _fin(sgn * zr * zr)


def _ix_mom_rank_x_vol_rank(t, C):
    return _fin(_cs_rank(_mom(t, C, 12)) * _cs_rank(_vol(t, C, 60)))


def _ix_dnvol_x_rev(t, C):
    return _fin(_z(_dvol(t, C, 60)) * _z(_rev5(t, C)))


def _ix_skew_sq(t, C):
    zs = _z(_skew(t, C, 60))
    return _fin(zs * zs)


def _ix_pvcorr_x_rev(t, C):
    return _fin(_z(_pvcorr(t, C, 20)) * _z(_rev5(t, C)))


def _ix_consensus_rev(t, C):
    return _consensus([_z(-_ret(t, C, 1)), _z(_rev5(t, C)), _z(_rev20(t, C))])


def _ix_consensus_mom(t, C):
    return _consensus([_z(_mom(t, C, 3)), _z(_mom(t, C, 6)), _z(_mom(t, C, 12))])


def _ix_lowvol_x_lowamt(t, C):
    return _fin(_z(_lowvol(t, C)) * _z(_lowamt(t, C)))


def _ix_nearhigh_x_lowvol(t, C):
    return _fin(_z(_nearhigh(t, C, 250)) * _z(_lowvol(t, C)))


def _ix_mom_x_illiq(t, C):
    return _fin(_z(_mom(t, C, 12)) * _z(_amihud(t, C, 60)))


# ------------------------------------------------------------------ #
def _spec(fn, desc: str, direction: int) -> dict:
    """把 (t, C) 实现包装成注册表要求的 (t, md, C) 签名。"""
    return dict(func=lambda t, md, C, _fn=fn: _fn(t, C), desc=desc, direction=int(direction))


FACTORS = {
    # ---- 反转 × 风险 / 流动性 ----
    "ix_rev_x_illiq": _spec(_ix_rev_x_illiq,
                            "z(-20日收益) × z(60日Amihud非流动性)：错杀 + 流动性溢价交互", +1),
    "ix_rev_x_lowvol": _spec(_ix_rev_x_lowvol,
                             "z(-20日收益) × z(-60日波动)：低波动反转交互", +1),
    "ix_dnvol_x_rev": _spec(_ix_dnvol_x_rev,
                            "z(60日下行半偏差) × z(-5日收益)：下行风险下的短期反转", +1),
    "ix_illiq_x_lowvol": _spec(_ix_illiq_x_lowvol,
                               "z(60日Amihud非流动性) × z(-60日波动)：非流动性 + 低波", +1),
    "ix_vol_x_range": _spec(_ix_vol_x_range,
                            "z(60日波动) × z(20日均振幅(H-L)/C)：波动与日内振幅交互", -1),
    "ix_skew_x_vol": _spec(_ix_skew_x_vol,
                           "z(60日收益偏度) × z(60日波动)：偏度与波动交互", -1),
    "ix_skew_sq": _spec(_ix_skew_sq,
                        "z(60日收益偏度)^2：极端偏度惩罚项（越小越好）", -1),
    "ix_max_x_illiq": _spec(_ix_max_x_illiq,
                            "z(20日最大单日收益MAX) × z(60日Amihud非流动性)：彩票 × 非流动性", -1),
    # ---- 动量 × 风险 / 流动性 ----
    "ix_mom_x_lowvol": _spec(_ix_mom_x_lowvol,
                             "z(12-1动量) × z(-60日波动)：低波动量交互", +1),
    "ix_mom_x_lowamt": _spec(_ix_mom_x_lowamt,
                             "z(12-1动量) × z(-成交额相对度)：低关注度动量", +1),
    "ix_mom_x_beta": _spec(_ix_mom_x_beta,
                           "z(12-1动量) × z(60日市场beta)", +1),
    "ix_mom_x_illiq": _spec(_ix_mom_x_illiq,
                            "z(12-1动量) × z(60日Amihud非流动性)", +1),
    "ix_mom_rank_x_vol_rank": _spec(_ix_mom_rank_x_vol_rank,
                                    "12-1动量横截面分位 × 60日波动横截面分位（0~1）", +1),
    # ---- 反转 × 规模 / 量能 / 筹码 ----
    "ix_rev_x_size": _spec(_ix_rev_x_size,
                           "z(-5日收益) × z(log(60日均成交额))：短期反转 × 规模代理", +1),
    "ix_amt_x_ret20": _spec(_ix_amt_x_ret20,
                            "z(成交额相对度) × z(20日收益)：放量上涨交互", -1),
    "ix_chip_x_rev": _spec(_ix_chip_x_rev,
                           "z(60日获利盘比例) × z(-5日收益)：筹码盈利 × 短期反转", +1),
    "ix_pvdiv_x_turn": _spec(_ix_pvdiv_x_turn,
                             "z(20日量价背离) × z(成交额相对度)", -1),
    "ix_pvcorr_x_rev": _spec(_ix_pvcorr_x_rev,
                             "z(20日量价相关) × z(-5日收益)", +1),
    # ---- 形态 / 高点 ----
    "ix_nearhigh_x_amt": _spec(_ix_nearhigh_x_amt,
                               "z(52周高点接近度 C/max(H,250)-1) × z(成交额相对度)", +1),
    "ix_nearhigh_x_lowvol": _spec(_ix_nearhigh_x_lowvol,
                                  "z(52周高点接近度) × z(-60日波动)：强势低波交互", +1),
    # ---- 非线性 / 合成 ----
    "ix_rev_sq": _spec(_ix_rev_sq,
                       "sign(r5) × z(r5)^2：短期反转非线性项（大涨为正、大跌为负）", -1),
    "ix_consensus_rev": _spec(_ix_consensus_rev,
                              "z(-1日)/z(-5日)/z(-20日) 收益的等权均值（多周期反转合成）", +1),
    "ix_consensus_mom": _spec(_ix_consensus_mom,
                              "z(3-1)/z(6-1)/z(12-1) 动量的等权均值（多周期动量合成）", +1),
    "ix_lowvol_x_lowamt": _spec(_ix_lowvol_x_lowamt,
                                "z(-60日波动) × z(-成交额相对度)：低波 + 低活跃度", +1),
}