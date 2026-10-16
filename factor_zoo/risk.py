# -*- coding: utf-8 -*-
"""风险 / 波动率类因子（rk_*）：全部只用第 t 行及之前的数据。

覆盖方向
--------
1. 低波动异象      rk_vol60 / rk_vol120 / rk_vol250
2. 特质波动率      rk_ivol60：对 C.mkt 做市场模型回归，取残差波动
3. 下行风险        rk_dvol60（只统计负收益的半偏差）/ rk_udvol60（下行 ÷ 上行波动）
4. 区间最大回撤    rk_maxdd60 / rk_maxdd250：max(1 - C / cummax(C))
5. 高阶矩          rk_skew60 / rk_skew250 偏度；rk_kurt60 / rk_kurt250 超额峰度
6. 波动率变化      rk_vr60250（短窗波动 / 长窗波动 - 1）/ rk_vov120（波动的波动）

实现约定
--------
- 窗口一律用 Ctx.sl / Ctx.stats 取，统计量一律用 Ctx.mean / Ctx.std 掩码版，不用 np.nanmean
- 每个窗口要求有效样本 >= max(floor, 0.6 * w)，不足即 NaN（停牌、上市不足等自然缺失）
- 出口统一过 _fin()：把 ±inf 收敛成 NaN（契约要求"绝不返回 inf"）
- 不缓存跨日状态、不做整表归约，函数只读 C，保证 Ctx.truncated(t) 复算逐位一致

注：窗口数量按"总数 10~14 个"约束做了取舍——低波动一族保留 60/120/250，
    最大回撤保留 60/250（120 与两侧高度共线，省下的名额给波动率变化一族）。
"""
from __future__ import annotations

import numpy as np

from .ctx import Ctx

#: 二阶矩下限保护，避免除零／负数开方
_EPS = 1e-12


# ------------------------------------------------------------------ #
# 工具
# ------------------------------------------------------------------ #
def _min_cnt(w: int, frac: float = 0.6, floor: int = 20) -> int:
    """窗口内有效样本数下限：max(floor, frac * w)。"""
    return max(int(floor), int(np.ceil(float(frac) * int(w))))


def _fin(x) -> np.ndarray:
    """把 ±inf / NaN 统一收敛成 NaN，保证不会把 inf 传出去。"""
    x = np.asarray(x, dtype="float64")
    return np.where(np.isfinite(x), x, np.nan)


def _half_moments(t: int, C: Ctx, w: int):
    """窗口内上/下半二阶矩：返回 (sum(min(r,0)^2), sum(max(r,0)^2), cnt)。

    用掩码求和，不用 np.nanmean / np.nansum（全 NaN 列会告警且慢）。
    """
    _, W = C.sl(t, w, C.R)
    ok = np.isfinite(W)
    cnt = ok.sum(0).astype("float64")
    dn = np.where(ok & (W < 0.0), W, 0.0)
    up = np.where(ok & (W > 0.0), W, 0.0)
    return (dn ** 2).sum(0), (up ** 2).sum(0), cnt


# ------------------------------------------------------------------ #
# 1. 低波动异象：日收益标准差
# ------------------------------------------------------------------ #
def _vol(t: int, C: Ctx, w: int, frac: float = 0.6, floor: int = 20) -> np.ndarray:
    """过去 w 日复权日收益的样本标准差（掩码版，ddof=1）。"""
    _, W = C.sl(t, w, C.R)
    ok = np.isfinite(W)
    sd = C.std(W, ok, _min_cnt(w, frac, floor))
    return _fin(sd)


# ------------------------------------------------------------------ #
# 2. 特质波动率：市场模型残差波动
# ------------------------------------------------------------------ #
def _ivol(t: int, C: Ctx, w: int = 60, frac: float = 0.6, floor: int = 40) -> np.ndarray:
    """市场模型残差波动。

    beta = cov(r_i, r_m) / var(r_m)（窗口内、掩码配对），
    残差 = 去均值后的 r_i - beta * 去均值后的 r_m，取残差标准差。
    """
    lo, W = C.sl(t, w, C.R)
    M = C.mkt[lo:int(t) + 1]                                  # 等权市场收益（已去停牌/一字板）
    ok = np.isfinite(W) & np.isfinite(M)[:, None]
    cnt = ok.sum(0).astype("float64")
    den = np.maximum(cnt, 1.0)
    Mw = np.where(ok, M[:, None], 0.0)
    mx = np.where(ok, W, 0.0).sum(0) / den                    # 个股窗口均值
    mm = Mw.sum(0) / den                                      # 市场窗口均值（同一有效集合）
    Xc = np.where(ok, W - mx, 0.0)
    Mc = np.where(ok, M[:, None] - mm, 0.0)
    df = np.maximum(cnt - 1.0, 1.0)
    v_m = (Mc ** 2).sum(0) / df                               # 市场方差
    c_im = (Xc * Mc).sum(0) / df                              # 协方差
    good = v_m > _EPS
    beta = np.where(good, c_im / np.where(good, v_m, 1.0), 0.0)
    res = Xc - beta * Mc                                      # 残差（市场项已正交化）
    rv = (res ** 2).sum(0) / df
    out = np.sqrt(np.maximum(rv, 0.0))
    return _fin(np.where(cnt >= _min_cnt(w, frac, floor), out, np.nan))


# ------------------------------------------------------------------ #
# 3. 下行波动
# ------------------------------------------------------------------ #
def _dvol(t: int, C: Ctx, w: int = 60, frac: float = 0.6, floor: int = 20) -> np.ndarray:
    """下行半偏差 sqrt(sum(min(r,0)^2) / n)，n = 窗口内有效样本数。"""
    s_dn, _, cnt = _half_moments(t, C, w)
    out = np.sqrt(s_dn / np.maximum(cnt, 1.0))
    return _fin(np.where(cnt >= _min_cnt(w, frac, floor), out, np.nan))


def _udvol(t: int, C: Ctx, w: int = 60, frac: float = 0.6, floor: int = 20) -> np.ndarray:
    """下行 / 上行波动比：>1 说明下跌比上涨更剧烈（负向风险占优）。"""
    s_dn, s_up, cnt = _half_moments(t, C, w)
    n = np.maximum(cnt, 1.0)
    dn = np.sqrt(s_dn / n)
    up = np.sqrt(s_up / n)
    good = up > _EPS
    r = np.where(good, dn / np.where(good, up, 1.0), np.nan)   # 全为正收益 -> NaN（无穷大更糟）
    return _fin(np.where(cnt >= _min_cnt(w, frac, floor), r, np.nan))


# ------------------------------------------------------------------ #
# 4. 区间最大回撤
# ------------------------------------------------------------------ #
def _maxdd(t: int, C: Ctx, w: int = 60, frac: float = 0.6, floor: int = 20) -> np.ndarray:
    """区间最大回撤 max(1 - C / cummax(C))，用复权收盘价。

    上市前的 NaN 用 -inf 占位，使得 cummax 只在有效样本内滚动（不会把缺失当 0）。
    """
    _, P = C.sl(t, w, C.C)
    ok = np.isfinite(P) & (P > 0.0)
    cnt = ok.sum(0).astype("float64")
    run = np.maximum.accumulate(np.where(ok, P, -np.inf), axis=0)
    good = ok & (run > 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        dd = np.where(good, 1.0 - P / np.where(good, run, 1.0), -np.inf)
    mdd = dd.max(0)                                            # 全无效列 -> -inf
    return _fin(np.where(cnt >= _min_cnt(w, frac, floor), mdd, np.nan))


# ------------------------------------------------------------------ #
# 5. 偏度 / 超额峰度
# ------------------------------------------------------------------ #
def _skew(t: int, C: Ctx, w: int = 60, frac: float = 0.6, floor: int = 20) -> np.ndarray:
    """收益偏度 g1 = m3 / m2^1.5（中心矩由 C.stats 一次给出）。"""
    s = C.stats(t, w)
    cnt = s["cnt"]
    Z = s["Wc"]
    n = np.maximum(cnt, 1.0)
    m2 = (Z ** 2).sum(0) / n
    m3 = (Z ** 3).sum(0) / n
    good = m2 > _EPS
    out = np.where(good, m3 / np.where(good, m2, 1.0) ** 1.5, np.nan)
    return _fin(np.where(cnt >= _min_cnt(w, frac, floor), out, np.nan))


def _kurt(t: int, C: Ctx, w: int = 60, frac: float = 0.6, floor: int = 20) -> np.ndarray:
    """超额峰度 g2 = m4 / m2^2 - 3（正态为 0，越大尾部越厚）。"""
    s = C.stats(t, w)
    cnt = s["cnt"]
    Z = s["Wc"]
    n = np.maximum(cnt, 1.0)
    m2 = (Z ** 2).sum(0) / n
    m4 = (Z ** 4).sum(0) / n
    good = m2 > _EPS
    out = np.where(good, m4 / np.where(good, m2, 1.0) ** 2 - 3.0, np.nan)
    return _fin(np.where(cnt >= _min_cnt(w, frac, floor), out, np.nan))


# ------------------------------------------------------------------ #
# 6. 波动率变化 / 波动率的波动
# ------------------------------------------------------------------ #
def _vr(t: int, C: Ctx, short: int = 60, long: int = 250,
        frac: float = 0.6, floor: int = 20) -> np.ndarray:
    """短窗波动 / 长窗波动 - 1：>0 表示近期波动在放大。"""
    vs = _vol(t, C, short, frac, floor)
    vl = _vol(t, C, long, frac, floor)
    good = vl > _EPS
    r = np.where(good, vs / np.where(good, vl, 1.0) - 1.0, np.nan)
    return _fin(r)


def _vov(t: int, C: Ctx, window: int = 120, sub: int = 20,
         frac: float = 0.6, floor: int = 10) -> np.ndarray:
    """波动率的波动：把长窗切成若干不重叠短段，算各段波动的标准差。

    段数不足（数据太短）时返回全 NaN。
    """
    nseg = max(int(window) // int(sub), 2)
    segs = []
    for j in range(nseg):
        ti = int(t) - j * int(sub)
        if ti < int(sub) - 1:                                  # 不够一个完整短段
            return np.full(C.k, np.nan, dtype="float64")
        segs.append(_vol(ti, C, sub, frac, floor))
    S = np.vstack(segs)
    ok = np.isfinite(S)
    cnt = ok.sum(0).astype("float64")
    mu = np.where(ok, S, 0.0).sum(0) / np.maximum(cnt, 1.0)
    c = np.where(ok, S - mu, 0.0)
    sd = np.sqrt((c ** 2).sum(0) / np.maximum(cnt - 1.0, 1.0))
    need = max(2.0, frac * nseg)
    return _fin(np.where(cnt >= need, sd, np.nan))


# ------------------------------------------------------------------ #
FACTORS = {
    # --- 1. 低波动异象 ---
    "rk_vol60": dict(func=lambda t, md, C: _vol(t, C, 60),
                     desc="过去60日复权日收益标准差（60日已实现波动）", direction=-1),
    "rk_vol120": dict(func=lambda t, md, C: _vol(t, C, 120),
                      desc="过去120日复权日收益标准差", direction=-1),
    "rk_vol250": dict(func=lambda t, md, C: _vol(t, C, 250),
                      desc="过去250日复权日收益标准差（年度已实现波动）", direction=-1),
    # --- 2. 特质波动率 ---
    "rk_ivol60": dict(func=lambda t, md, C: _ivol(t, C, 60),
                      desc="60日特质波动率：对等权市场收益做市场模型回归的残差标准差", direction=-1),
    # --- 3. 下行波动 ---
    "rk_dvol60": dict(func=lambda t, md, C: _dvol(t, C, 60),
                      desc="60日下行半偏差 sqrt(sum(min(r,0)^2)/n)", direction=-1),
    "rk_udvol60": dict(func=lambda t, md, C: _udvol(t, C, 60),
                       desc="60日下行波动/上行波动（涨跌不对称度）", direction=-1),
    # --- 4. 区间最大回撤 ---
    "rk_maxdd60": dict(func=lambda t, md, C: _maxdd(t, C, 60),
                       desc="60日区间最大回撤 max(1 - 复权价/cummax)", direction=-1),
    "rk_maxdd250": dict(func=lambda t, md, C: _maxdd(t, C, 250),
                        desc="250日区间最大回撤 max(1 - 复权价/cummax)", direction=-1),
    # --- 5. 偏度 / 超额峰度 ---
    "rk_skew60": dict(func=lambda t, md, C: _skew(t, C, 60),
                      desc="60日收益偏度 m3/m2^1.5（右偏=彩票偏好）", direction=-1),
    "rk_skew250": dict(func=lambda t, md, C: _skew(t, C, 250),
                       desc="250日收益偏度 m3/m2^1.5", direction=-1),
    "rk_kurt60": dict(func=lambda t, md, C: _kurt(t, C, 60),
                      desc="60日收益超额峰度 m4/m2^2-3", direction=-1),
    "rk_kurt250": dict(func=lambda t, md, C: _kurt(t, C, 250),
                       desc="250日收益超额峰度 m4/m2^2-3", direction=-1),
    # --- 6. 波动率变化 / 波动率的波动 ---
    "rk_vr60250": dict(func=lambda t, md, C: _vr(t, C, 60, 250),
                       desc="波动率变化：60日波动/250日波动-1（>0 波动放大）", direction=-1),
    "rk_vov120": dict(func=lambda t, md, C: _vov(t, C, 120, 20),
                      desc="波动率的波动：120日窗切成6段20日，段间波动率的标准差", direction=-1),
}
