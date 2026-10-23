# -*- coding: utf-8 -*-
"""流动性 / 交易结构因子（lq2_*，共 38 个）。

本模块只使用 Ctx 中下标 <= t 的数据，窗口通过 C.sl 截取；所有统计量都使用
掩码实现，无法计算的位置返回 NaN，绝不返回 ±inf。数据中没有成交量、股本、
市值和逐笔方向，因此本模块明确采用以下代理口径：

* 成交额 A 为正且有限才视为可交易观测；停牌 A==0、未上市/退市 A==NaN 都视为缺失。
* A/C 只能作为“成交股数代理”，且按项目硬性契约使用复权收盘价 C.C。由于复权因子
  会随除权除息变化，A/C 的量纲会受复权影响，不能解释为真实股数。
* 单笔均额 = A/NT，NT 有零星缺口；任一项缺失即该日缺失。
* 成交额份额类因子只在有效正成交额上计算，停牌/缺失不进入份额分母。

关键实现细节
------------
Corwin-Schultz 高低价价差估计（lq2_spread_cs60）
    对窗口内每一对相邻交易日 (i, i+1) 使用复权 H/L：
        beta  = ln(H_i/L_i)^2 + ln(H_{i+1}/L_{i+1})^2
        gamma = ln(max(H_i,H_{i+1}) / min(L_i,L_{i+1}))^2
        alpha = (sqrt(2*beta)-sqrt(beta))/(3-2*sqrt(2))
                - sqrt(gamma/(3-2*sqrt(2)))
        S     = 2*(exp(alpha)-1)/(1+exp(alpha)) = 2*tanh(alpha/2)
    原始估计器规定 alpha<0 时把 S 截断为 0；本模块保留该规则，并对窗口内所有
    相邻日对的 S 取算术平均。只有两天的 A>0、H/L 有效且 H>=L 的日对参与计算。

Roll 价差估计（lq2_spread_roll60）
    令 d_t = R_t - R_{t-1}，计算 d_t 与 d_{t-1} 的样本协方差 cov（有效日对配对）。
    标准 Roll 口径为 S = 2*sqrt(-cov)，要求协方差为负，即根号内 -cov>0；
    若协方差非负（根号内为负），估计量无实数解，本模块返回 NaN。窗口使用
    t-w-1 ... t 的收益构造 w 个 d_t/d_{t-1} 日对，并要求至少 60% 的有效日对。
    Roll 估计量对趋势、停牌和离散报价敏感，本模块只把它当作价差代理。

Kyle lambda（lq2_kyle60）
    在缺少买卖方向/逐笔数据的限制下，使用“无符号”Kyle 代理：以 sqrt(A) 为
    订单流规模代理、|R| 为价格冲击，做窗口内掩码 OLS：
        |R_i| = a + lambda * sqrt(A_i) + e_i
    返回斜率 lambda（不做年化）。这不是严格的结构 Kyle lambda；严格版本需要
    带方向的订单流和中间价。这里只利用“成交额越大、价格冲击越高则流动性越差”
    的方向信息。

其他口径
--------
* Amihud：mean(|R|/A)*1e8；lq2_amihud20b 按需求改用窗口内中位数，降低极端值影响。
* Amivest：log(mean(A/|R|))，R==0 的日期无法形成有限比值，按缺失处理。
* 流动性冲击：log(A_t / mean(A,60))，不是收益因子。
* still/traded 天数：分母只统计 A 有限的上市交易日；A==0 为停牌日。
* 所有因子保持原量纲，不做横截面标准化。
"""
from __future__ import annotations

import numpy as np

from .ctx import Ctx


_EPS = 1e-12
_SQRT2 = float(np.sqrt(2.0))
_CS_DEN = 3.0 - 2.0 * _SQRT2


# ================================================================== #
# 基础工具：掩码统计与安全运算
# ================================================================== #
def _fin(x) -> np.ndarray:
    """把 ±inf / NaN 统一收敛为 NaN。"""
    x = np.asarray(x, dtype="float64")
    return np.where(np.isfinite(x), x, np.nan)


def _min_cnt(w: int, frac: float = 0.6, floor=None) -> int:
    """窗口最低有效样本数：max(floor, ceil(frac*w))。"""
    w = int(w)
    if floor is None:
        floor = 3 if w < 120 else 20
    return max(int(floor), int(np.ceil(float(frac) * w)))


def _positive(x) -> np.ndarray:
    """严格正且有限的掩码；A<=0 / NT<=0 都视为缺失。"""
    x = np.asarray(x, dtype="float64")
    return np.isfinite(x) & (x > 0.0)


def _ratio(num, den) -> np.ndarray:
    """安全除法：分母非正、非有限，或分子非有限时返回 NaN。"""
    num = np.asarray(num, dtype="float64")
    den = np.asarray(den, dtype="float64")
    good = np.isfinite(num) & np.isfinite(den) & (den > 0.0)
    return np.where(good, num / np.where(good, den, 1.0), np.nan)


def _log_positive(x) -> np.ndarray:
    """只对严格正有限值取自然对数，其余返回 NaN。"""
    x = np.asarray(x, dtype="float64")
    good = np.isfinite(x) & (x > 0.0)
    return np.where(good, np.log(np.where(good, x, 1.0)), np.nan)


def _mean_pos_window(C: Ctx, t: int, w: int, X, frac: float = 0.6,
                     floor=None) -> np.ndarray:
    """窗口内严格正值的掩码均值。"""
    lo, W = C.sl(t, w, X)
    ok = _positive(W)
    return C.mean(W, ok, _min_cnt(w, frac, floor))


def _std_pos_window(C: Ctx, t: int, w: int, X, frac: float = 0.6,
                    floor=None) -> np.ndarray:
    """窗口内严格正值的掩码样本标准差（ddof=1）。"""
    lo, W = C.sl(t, w, X)
    ok = _positive(W)
    return C.std(W, ok, _min_cnt(w, frac, floor), ddof=1)


def _median_masked(X, ok, need: int) -> np.ndarray:
    """掩码中位数：按列排序，偶数个取中间两数均值。"""
    X = np.asarray(X, dtype="float64")
    ok = np.asarray(ok, dtype=bool)
    cnt = ok.sum(0).astype("float64")
    Z = np.sort(np.where(ok, X, np.inf), axis=0)
    cnt_i = np.maximum(cnt.astype("int64"), 1)
    lo_idx = ((cnt_i - 1) // 2).astype(np.intp)
    hi_idx = (cnt_i // 2).astype(np.intp)
    lo = np.take_along_axis(Z, lo_idx[None, :], axis=0)[0]
    hi = np.take_along_axis(Z, hi_idx[None, :], axis=0)[0]
    out = 0.5 * (lo + hi)
    return np.where(cnt >= need, out, np.nan)


def _corr_masked(x, y, ok, need: int) -> np.ndarray:
    """掩码 Pearson 相关；分母为零或有效数不足时返回 NaN。"""
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    ok = np.asarray(ok, dtype=bool)
    cnt = ok.sum(0).astype("float64")
    den = np.maximum(cnt, 1.0)
    xb = np.where(ok, x, 0.0).sum(0) / den
    yb = np.where(ok, y, 0.0).sum(0) / den
    xc = np.where(ok, x - xb, 0.0)
    yc = np.where(ok, y - yb, 0.0)
    num = (xc * yc).sum(0)
    den2 = np.sqrt(np.maximum((xc ** 2).sum(0) * (yc ** 2).sum(0), 0.0))
    out = _ratio(num, den2)
    return np.where(cnt >= need, out, np.nan)


def _ols_slope(x, y, ok, need: int) -> np.ndarray:
    """逐列掩码 OLS 斜率 sum((x-xbar)(y-ybar)) / sum((x-xbar)^2)。"""
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    ok = np.asarray(ok, dtype=bool)
    cnt = ok.sum(0).astype("float64")
    den = np.maximum(cnt, 1.0)
    xb = np.where(ok, x, 0.0).sum(0) / den
    yb = np.where(ok, y, 0.0).sum(0) / den
    xc = np.where(ok, x - xb, 0.0)
    yc = np.where(ok, y - yb, 0.0)
    out = _ratio((xc * yc).sum(0), (xc ** 2).sum(0))
    return np.where(cnt >= need, out, np.nan)


def _skew_masked(X, ok, need: int) -> np.ndarray:
    """掩码偏度 m3/m2^1.5（非超额峰度口径）。"""
    X = np.asarray(X, dtype="float64")
    ok = np.asarray(ok, dtype=bool)
    cnt = ok.sum(0).astype("float64")
    den = np.maximum(cnt, 1.0)
    mu = np.where(ok, X, 0.0).sum(0) / den
    Z = np.where(ok, X - mu, 0.0)
    m2 = (Z ** 2).sum(0) / den
    m3 = (Z ** 3).sum(0) / den
    good = m2 > _EPS
    out = np.where(good, m3 / np.where(good, m2, 1.0) ** 1.5, np.nan)
    return np.where(cnt >= need, out, np.nan)


# ================================================================== #
# 成交额水平 / 相对活跃度
# ================================================================== #
def _log_amt(t: int, C: Ctx, w: int) -> np.ndarray:
    """log(近 w 日平均成交额)。"""
    return _log_positive(_mean_pos_window(C, t, w, C.A))


def _share_amt(t: int, C: Ctx, w: int) -> np.ndarray:
    """log(mean(A/C))：日均成交股数代理；C 为复权收盘价。"""
    lo, A = C.sl(t, w, C.A)
    Cc = C.C[lo:int(t) + 1]
    ok = _positive(A) & _positive(Cc)
    val = A / np.where(ok, Cc, 1.0)
    mu = C.mean(val, ok, _min_cnt(w))
    return _log_positive(mu)


def _amt_rel5(t: int, C: Ctx) -> np.ndarray:
    """A[t] / mean(A,60) - 1。"""
    m = _mean_pos_window(C, t, 60, C.A)
    cur = C.A[t]
    good = np.isfinite(cur) & (cur > 0.0) & np.isfinite(m) & (m > 0.0)
    return np.where(good, cur / np.where(good, m, 1.0) - 1.0, np.nan)


def _amt_rel20(t: int, C: Ctx) -> np.ndarray:
    """mean(A,20) / mean(A,120) - 1。"""
    m20 = _mean_pos_window(C, t, 20, C.A)
    m120 = _mean_pos_window(C, t, 120, C.A)
    good = np.isfinite(m20) & np.isfinite(m120) & (m20 > 0.0) & (m120 > 0.0)
    return np.where(good, m20 / np.where(good, m120, 1.0) - 1.0, np.nan)


def _amt_cv(t: int, C: Ctx, w: int) -> np.ndarray:
    """成交额变异系数 std(A)/mean(A)。"""
    lo, A = C.sl(t, w, C.A)
    ok = _positive(A)
    need = _min_cnt(w)
    mu = C.mean(A, ok, need)
    sd = C.std(A, ok, need, ddof=1)
    return _fin(_ratio(sd, mu))


def _amt_ar1(t: int, C: Ctx, w: int) -> np.ndarray:
    """log(A) 一阶自相关。"""
    lo, A = C.sl(t, w + 1, C.A)
    ok = _positive(A)
    logA = np.log(np.where(ok, A, 1.0))
    return _fin(_corr_masked(logA[1:], logA[:-1],
                             ok[1:] & ok[:-1], _min_cnt(w)))


def _amt_hhi(t: int, C: Ctx, w: int) -> np.ndarray:
    """窗口内成交额份额的 Herfindahl 指数 sum(p_i^2)。"""
    lo, A = C.sl(t, w, C.A)
    ok = _positive(A)
    cnt = ok.sum(0).astype("float64")
    tot = np.where(ok, A, 0.0).sum(0)
    p = np.where(ok & (tot > 0.0), A / np.where(tot > 0.0, tot, 1.0), 0.0)
    out = (p ** 2).sum(0)
    return np.where(cnt >= _min_cnt(w), out, np.nan)


def _amt_entropy(t: int, C: Ctx, w: int) -> np.ndarray:
    """窗口内成交额份额的 Shannon 熵 -sum(p_i*ln p_i)。"""
    lo, A = C.sl(t, w, C.A)
    ok = _positive(A)
    cnt = ok.sum(0).astype("float64")
    tot = np.where(ok, A, 0.0).sum(0)
    p = np.where(ok & (tot > 0.0), A / np.where(tot > 0.0, tot, 1.0), 0.0)
    term = np.where(p > 0.0, p * np.log(np.where(p > 0.0, p, 1.0)), 0.0)
    out = -term.sum(0)
    return np.where(cnt >= _min_cnt(w), out, np.nan)


# ================================================================== #
# Amihud / Amivest / Kyle / 价差
# ================================================================== #
def _amihud(t: int, C: Ctx, w: int, median: bool = False) -> np.ndarray:
    """Amihud 非流动性 mean/median(|R|/A)*1e8。"""
    lo, A = C.sl(t, w, C.A)
    R = C.R[lo:int(t) + 1]
    ok = _positive(A) & np.isfinite(R)
    val = np.abs(R) / np.where(ok, A, 1.0)
    need = _min_cnt(w)
    if median:
        out = _median_masked(val, ok, need)
    else:
        out = C.mean(val, ok, need)
    return _fin(out * 1e8)


def _amihud_log(t: int, C: Ctx, w: int) -> np.ndarray:
    """log(Amihud*1e8)。"""
    return _log_positive(_amihud(t, C, w, median=False))


def _amihud_trend(t: int, C: Ctx) -> np.ndarray:
    """amihud20 / amihud120 - 1（均使用均值口径）。"""
    a20 = _amihud(t, C, 20, median=False)
    a120 = _amihud(t, C, 120, median=False)
    good = np.isfinite(a20) & np.isfinite(a120) & (a120 > 0.0)
    return np.where(good, a20 / np.where(good, a120, 1.0) - 1.0, np.nan)


def _illiq_asym(t: int, C: Ctx, w: int) -> np.ndarray:
    """下跌日 |R|/A 均值 / 上涨日 |R|/A 均值。"""
    lo, A = C.sl(t, w, C.A)
    R = C.R[lo:int(t) + 1]
    base = _positive(A) & np.isfinite(R)
    val = np.abs(R) / np.where(base, A, 1.0)
    dn = base & (R < 0.0)
    up = base & (R > 0.0)
    dn_cnt = dn.sum(0).astype("float64")
    up_cnt = up.sum(0).astype("float64")
    dn_mean = _ratio(np.where(dn, val, 0.0).sum(0), dn_cnt)
    up_mean = _ratio(np.where(up, val, 0.0).sum(0), up_cnt)
    out = _ratio(dn_mean, up_mean)
    valid = (base.sum(0) >= _min_cnt(w)) & (dn_cnt > 0.0) & (up_cnt > 0.0)
    return _fin(np.where(valid, out, np.nan))


def _amivest(t: int, C: Ctx, w: int) -> np.ndarray:
    """log(mean(A/|R|))，R==0 的日期按缺失处理。"""
    lo, A = C.sl(t, w, C.A)
    R = C.R[lo:int(t) + 1]
    ar = np.abs(R)
    ok = _positive(A) & np.isfinite(ar) & (ar > _EPS)
    val = A / np.where(ok, ar, 1.0)
    mu = C.mean(val, ok, _min_cnt(w))
    return _log_positive(mu)


def _kyle(t: int, C: Ctx, w: int) -> np.ndarray:
    """无符号 Kyle lambda：|R| 对 sqrt(A) 的窗口 OLS 斜率。"""
    lo, A = C.sl(t, w, C.A)
    R = C.R[lo:int(t) + 1]
    ok = _positive(A) & np.isfinite(R)
    x = np.sqrt(np.where(ok, A, 1.0))
    y = np.abs(np.where(ok, R, 0.0))
    return _fin(_ols_slope(x, y, ok, _min_cnt(w)))


def _cs_spread(t: int, C: Ctx, w: int) -> np.ndarray:
    """Corwin-Schultz 相邻两日高低价价差估计的窗口均值。"""
    lo, H = C.sl(t, w + 1, C.ADJ_H)
    L = C.ADJ_L[lo:int(t) + 1]
    A = C.A[lo:int(t) + 1]
    H1, H2 = H[:-1], H[1:]
    L1, L2 = L[:-1], L[1:]
    A1, A2 = A[:-1], A[1:]
    base = (np.isfinite(H1) & np.isfinite(L1) & np.isfinite(H2) & np.isfinite(L2)
            & np.isfinite(A1) & np.isfinite(A2)
            & (H1 > 0.0) & (L1 > 0.0) & (H2 > 0.0) & (L2 > 0.0)
            & (A1 > 0.0) & (A2 > 0.0) & (H1 >= L1) & (H2 >= L2))
    hh = np.where(base, H1 / np.where(base, L1, 1.0), 1.0)
    hh2 = np.where(base, H2 / np.where(base, L2, 1.0), 1.0)
    hl = np.where(base, np.maximum(H1, H2) / np.minimum(L1, L2), 1.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        lh1 = np.log(hh)
        lh2 = np.log(hh2)
        lg = np.log(hl)
    beta = lh1 ** 2 + lh2 ** 2
    gamma = lg ** 2
    sqrt2b = np.sqrt(np.where(base, 2.0 * beta, 0.0))
    sqrtb = np.sqrt(np.where(base, beta, 0.0))
    sqrtg = np.sqrt(np.where(base, gamma / _CS_DEN, 0.0))
    alpha = (sqrt2b - sqrtb) / _CS_DEN - sqrtg
    valid = base & np.isfinite(alpha)
    # 原始 CS 估计器把 alpha<0 截断为 0；有效日对都参与均值。
    spread = 2.0 * np.tanh(alpha / 2.0)
    spread = np.where(valid & (alpha < 0.0), 0.0, spread)
    cnt = valid.sum(0).astype("float64")
    out = np.where(valid, spread, 0.0).sum(0) / np.maximum(cnt, 1.0)
    return _fin(np.where(cnt >= _min_cnt(w), out, np.nan))


def _roll_spread(t: int, C: Ctx, w: int) -> np.ndarray:
    """Roll 价差 2*sqrt(-cov(d_t,d_{t-1}))；协方差非负时返回 NaN。"""
    # w+2 行收益 -> w+1 个 d_t -> w 个 (d_t,d_{t-1}) 日对。
    lo, R = C.sl(t, w + 2, C.R)
    A = C.A[lo:int(t) + 1]
    ok_r = np.isfinite(R)
    d = R[1:] - R[:-1]
    x = d[1:]
    y = d[:-1]
    # 要求差分及对应成交额都有效。
    ok = (ok_r[2:] & ok_r[1:-1] & ok_r[:-2]
          & (A[2:] > 0.0) & (A[1:-1] > 0.0) & (A[:-2] > 0.0)
          & np.isfinite(A[2:]) & np.isfinite(A[1:-1]) & np.isfinite(A[:-2]))
    cnt = ok.sum(0).astype("float64")
    den = np.maximum(cnt, 1.0)
    xb = np.where(ok, x, 0.0).sum(0) / den
    yb = np.where(ok, y, 0.0).sum(0) / den
    xc = np.where(ok, x - xb, 0.0)
    yc = np.where(ok, y - yb, 0.0)
    cov = (xc * yc).sum(0) / np.maximum(cnt - 1.0, 1.0)
    inside = -cov
    valid = ok.sum(0) >= _min_cnt(w)
    valid = valid & np.isfinite(inside) & (inside > 0.0)
    out = 2.0 * np.sqrt(np.where(valid, inside, 0.0))
    return _fin(np.where(valid, out, np.nan))


def _spread_hl(t: int, C: Ctx, w: int) -> np.ndarray:
    """mean((H-L)/C)，使用复权 H/L/C，只统计有成交的交易日。"""
    lo, H = C.sl(t, w, C.ADJ_H)
    L = C.ADJ_L[lo:int(t) + 1]
    Cc = C.C[lo:int(t) + 1]
    A = C.A[lo:int(t) + 1]
    ok = (np.isfinite(H) & np.isfinite(L) & np.isfinite(Cc) & np.isfinite(A)
          & (H >= L) & (Cc > 0.0) & (A > 0.0))
    val = (H - L) / np.where(ok, Cc, 1.0)
    out = C.mean(val, ok, _min_cnt(w))
    return _fin(out)


# ================================================================== #
# 成交笔数 / 单笔均额 / 成交结构
# ================================================================== #
def _nt_log(t: int, C: Ctx, w: int) -> np.ndarray:
    """log(平均成交笔数)。"""
    return _log_positive(_mean_pos_window(C, t, w, C.NT))


def _trade_size_parts(t: int, C: Ctx, w: int):
    """返回 (每日单笔均额 A/NT, 有效掩码, 最低样本数)。"""
    lo, A = C.sl(t, w, C.A)
    NT = C.NT[lo:int(t) + 1]
    ok = _positive(A) & _positive(NT)
    sz = A / np.where(ok, NT, 1.0)
    return sz, ok, _min_cnt(w)


def _trade_size_mean(t: int, C: Ctx, w: int) -> np.ndarray:
    """单笔均额的窗口均值。"""
    sz, ok, need = _trade_size_parts(t, C, w)
    return C.mean(sz, ok, need)


def _trade_size_median(t: int, C: Ctx, w: int) -> np.ndarray:
    """log(单笔均额 A/NT 的窗口中位数)，用于 20 日中位数口径。"""
    sz, ok, need = _trade_size_parts(t, C, w)
    return _log_positive(_median_masked(sz, ok, need))


def _trade_size_trend(t: int, C: Ctx) -> np.ndarray:
    """log(mean(A/NT),20) - log(mean(A/NT),120)。"""
    m20 = _trade_size_mean(t, C, 20)
    m120 = _trade_size_mean(t, C, 120)
    return _fin(_log_positive(m20) - _log_positive(m120))


def _trade_size_skew(t: int, C: Ctx, w: int) -> np.ndarray:
    """单笔均额 A/NT 的窗口偏度。"""
    sz, ok, need = _trade_size_parts(t, C, w)
    return _fin(_skew_masked(sz, ok, need))


def _nt_per_amt(t: int, C: Ctx, w: int) -> np.ndarray:
    """mean(NT/A)：每元成交额笔数，散户度代理。"""
    lo, A = C.sl(t, w, C.A)
    NT = C.NT[lo:int(t) + 1]
    ok = _positive(A) & _positive(NT)
    val = NT / np.where(ok, A, 1.0)
    out = C.mean(val, ok, _min_cnt(w))
    return _fin(out)


def _still_days(t: int, C: Ctx, w: int) -> np.ndarray:
    """过去 w 日中 A==0（停牌）天数占比；分母为 A 有限的上市交易日。"""
    lo, A = C.sl(t, w, C.A)
    finite = np.isfinite(A)
    valid = finite & (A >= 0.0)
    still = valid & (A == 0.0)
    cnt = valid.sum(0).astype("float64")
    num = still.sum(0).astype("float64")
    out = num / np.maximum(cnt, 1.0)
    return np.where(cnt >= _min_cnt(w), out, np.nan)


def _traded_ratio(t: int, C: Ctx, w: int) -> np.ndarray:
    """过去 w 日中 A>0 的正常成交天数占比；分母为 A 有限的上市交易日。"""
    lo, A = C.sl(t, w, C.A)
    valid = np.isfinite(A) & (A >= 0.0)
    traded = valid & (A > 0.0)
    cnt = valid.sum(0).astype("float64")
    num = traded.sum(0).astype("float64")
    out = num / np.maximum(cnt, 1.0)
    return np.where(cnt >= _min_cnt(w), out, np.nan)


def _highlow_vol(t: int, C: Ctx, w: int) -> np.ndarray:
    """std((H-L)/C)，只统计有成交且 H>=L 的交易日。"""
    lo, H = C.sl(t, w, C.ADJ_H)
    L = C.ADJ_L[lo:int(t) + 1]
    Cc = C.C[lo:int(t) + 1]
    A = C.A[lo:int(t) + 1]
    ok = (np.isfinite(H) & np.isfinite(L) & np.isfinite(Cc) & np.isfinite(A)
          & (H >= L) & (Cc > 0.0) & (A > 0.0))
    val = (H - L) / np.where(ok, Cc, 1.0)
    out = C.std(val, ok, _min_cnt(w), ddof=1)
    return _fin(out)


# ================================================================== #
# 流动性冲击 / 成交额分位与趋势
# ================================================================== #
def _liquidity_shock(t: int, C: Ctx) -> np.ndarray:
    """log(A[t]/mean(A,60))，衡量当日相对自身的成交额冲击。"""
    m = _mean_pos_window(C, t, 60, C.A)
    cur = C.A[t]
    good = np.isfinite(cur) & (cur > 0.0) & np.isfinite(m) & (m > 0.0)
    return _log_positive(np.where(good, cur / np.where(good, m, 1.0), np.nan))


def _amt_percentile(t: int, C: Ctx, w: int) -> np.ndarray:
    """当日成交额在过去 w 日正成交额分布中的经验分位（含当日）。"""
    lo, A = C.sl(t, w, C.A)
    cur = C.A[t]
    ok = _positive(A)
    cnt = ok.sum(0).astype("float64")
    good_cur = np.isfinite(cur) & (cur > 0.0)
    le = (A <= cur[None, :]) & ok
    out = le.sum(0).astype("float64") / np.maximum(cnt, 1.0)
    valid = good_cur & (cnt >= _min_cnt(w))
    return np.where(valid, out, np.nan)


def _amt_trend_slope(t: int, C: Ctx, w: int) -> np.ndarray:
    """log(A) 对窗口内时间回归的斜率×100（%/交易日）。"""
    lo, A = C.sl(t, w, C.A)
    ok = _positive(A)
    logA = np.log(np.where(ok, A, 1.0))
    x = np.arange(logA.shape[0], dtype="float64")[:, None]
    slope = _ols_slope(x, logA, ok, _min_cnt(w))
    return _fin(slope * 100.0)


def _amt_accel(t: int, C: Ctx) -> np.ndarray:
    """成交额加速：(mean(A,5)/mean(A,20))/(mean(A,20)/mean(A,60))-1。"""
    m5 = _mean_pos_window(C, t, 5, C.A)
    m20 = _mean_pos_window(C, t, 20, C.A)
    m60 = _mean_pos_window(C, t, 60, C.A)
    good = (np.isfinite(m5) & np.isfinite(m20) & np.isfinite(m60)
            & (m5 > 0.0) & (m20 > 0.0) & (m60 > 0.0))
    out = (m5 * m60) / (m20 * m20) - 1.0
    return _fin(np.where(good, out, np.nan))


# ================================================================== #
# 注册表：38 个 lq2_* 因子
# ================================================================== #
FACTORS = {
    # --- 成交额水平 / 相对活跃度（12） ---
    "lq2_logamt20": dict(
        func=lambda t, md, C: _log_amt(t, C, 20),
        desc="log(20日均成交额)：流动性水平", direction=-1),
    "lq2_logamt60": dict(
        func=lambda t, md, C: _log_amt(t, C, 60),
        desc="log(60日均成交额)：流动性水平", direction=-1),
    "lq2_logamt120": dict(
        func=lambda t, md, C: _log_amt(t, C, 120),
        desc="log(120日均成交额)：流动性水平", direction=-1),
    "lq2_logamt250": dict(
        func=lambda t, md, C: _log_amt(t, C, 250),
        desc="log(250日均成交额)：流动性水平", direction=-1),
    "lq2_shareamt60": dict(
        func=lambda t, md, C: _share_amt(t, C, 60),
        desc="log(mean(A/C),60)：日均成交股数代理（C为复权价）", direction=-1),
    "lq2_amt_rel5": dict(
        func=lambda t, md, C: _amt_rel5(t, C),
        desc="当日成交额/60日均额-1：短期量能冲击", direction=-1),
    "lq2_amt_rel20": dict(
        func=lambda t, md, C: _amt_rel20(t, C),
        desc="20日均额/120日均额-1：中期相对活跃度", direction=-1),
    "lq2_amt_cv20": dict(
        func=lambda t, md, C: _amt_cv(t, C, 20),
        desc="20日成交额变异系数 std/mean", direction=-1),
    "lq2_amt_cv60": dict(
        func=lambda t, md, C: _amt_cv(t, C, 60),
        desc="60日成交额变异系数 std/mean", direction=-1),
    "lq2_amt_ar1_60": dict(
        func=lambda t, md, C: _amt_ar1(t, C, 60),
        desc="log成交额60日一阶自相关：活跃度持续性", direction=-1),
    "lq2_amt_hhi60": dict(
        func=lambda t, md, C: _amt_hhi(t, C, 60),
        desc="60日成交额份额Herfindahl指数：成交集中度", direction=1),
    "lq2_amt_entropy60": dict(
        func=lambda t, md, C: _amt_entropy(t, C, 60),
        desc="60日成交额份额Shannon熵：成交分散度", direction=-1),

    # --- Amihud / Amivest / Kyle / 价差（13） ---
    "lq2_amihud20b": dict(
        func=lambda t, md, C: _amihud(t, C, 20, median=True),
        desc="Amihud非流动性 median(|R|/A,20)×1e8", direction=1),
    "lq2_amihud120": dict(
        func=lambda t, md, C: _amihud(t, C, 120, median=False),
        desc="Amihud非流动性 mean(|R|/A,120)×1e8", direction=1),
    "lq2_amihud250": dict(
        func=lambda t, md, C: _amihud(t, C, 250, median=False),
        desc="Amihud非流动性 mean(|R|/A,250)×1e8", direction=1),
    "lq2_amihud_log60": dict(
        func=lambda t, md, C: _amihud_log(t, C, 60),
        desc="log(Amihud非流动性×1e8,60)", direction=1),
    "lq2_amihud_trend": dict(
        func=lambda t, md, C: _amihud_trend(t, C),
        desc="Amihud20/Amihud120-1：非流动性趋势", direction=1),
    "lq2_illiq_asym60": dict(
        func=lambda t, md, C: _illiq_asym(t, C, 60),
        desc="下跌日|R|/A均值 / 上涨日|R|/A均值（60日）", direction=1),
    "lq2_amivest60": dict(
        func=lambda t, md, C: _amivest(t, C, 60),
        desc="log(mean(A/|R|),60)：Amivest流动性", direction=-1),
    "lq2_kyle60": dict(
        func=lambda t, md, C: _kyle(t, C, 60),
        desc="无符号Kyle lambda：|R|对sqrt(A)的60日回归斜率", direction=1),
    "lq2_spread_cs60": dict(
        func=lambda t, md, C: _cs_spread(t, C, 60),
        desc="Corwin-Schultz高低价买卖价差估计（60日相邻日对均值）", direction=1),
    "lq2_spread_roll60": dict(
        func=lambda t, md, C: _roll_spread(t, C, 60),
        desc="Roll价差2*sqrt(-cov(ΔR_t,ΔR_{t-1}))（60日）", direction=1),
    "lq2_spread_hl20": dict(
        func=lambda t, md, C: _spread_hl(t, C, 20),
        desc="mean((复权H-复权L)/复权C,20)：高低价差", direction=1),
    "lq2_nt_log20": dict(
        func=lambda t, md, C: _nt_log(t, C, 20),
        desc="log(20日均成交笔数)：交易参与广度", direction=-1),
    "lq2_nt_log60": dict(
        func=lambda t, md, C: _nt_log(t, C, 60),
        desc="log(60日均成交笔数)：交易参与广度", direction=-1),

    # --- 单笔均额 / 成交结构（8） ---
    "lq2_tradesize20b": dict(
        func=lambda t, md, C: _trade_size_median(t, C, 20),
        desc="log(median(A/NT,20))：单笔均额中位数（机构参与度代理）", direction=-1),
    "lq2_tradesize60": dict(
        func=lambda t, md, C: _trade_size_mean(t, C, 60),
        desc="log(mean(A/NT,60))：单笔均额（机构参与度代理）", direction=-1),
    "lq2_tradesize_trend": dict(
        func=lambda t, md, C: _trade_size_trend(t, C),
        desc="log单笔均额20-log单笔均额120：大单化趋势", direction=1),
    "lq2_tradesize_skew60": dict(
        func=lambda t, md, C: _trade_size_skew(t, C, 60),
        desc="60日单笔均额A/NT偏度：偶发大单", direction=-1),
    "lq2_nt_per_amt60": dict(
        func=lambda t, md, C: _nt_per_amt(t, C, 60),
        desc="mean(NT/A,60)：每元成交额笔数（散户度代理）", direction=1),
    "lq2_still_days60": dict(
        func=lambda t, md, C: _still_days(t, C, 60),
        desc="过去60日A==0停牌天数占比（分母为上市交易日）", direction=-1),
    "lq2_still_days250": dict(
        func=lambda t, md, C: _still_days(t, C, 250),
        desc="过去250日A==0停牌天数占比（分母为上市交易日）", direction=-1),
    "lq2_traded_ratio60": dict(
        func=lambda t, md, C: _traded_ratio(t, C, 60),
        desc="过去60日A>0正常成交天数占比", direction=-1),

    # --- 高低价差波动 / 流动性冲击 / 成交额分位与趋势（5） ---
    "lq2_highlow_vol20": dict(
        func=lambda t, md, C: _highlow_vol(t, C, 20),
        desc="std((复权H-复权L)/复权C,20)：高低价差波动", direction=-1),
    "lq2_liquidity_shock": dict(
        func=lambda t, md, C: _liquidity_shock(t, C),
        desc="log(当日成交额/60日均额)：流动性冲击", direction=-1),
    "lq2_amt_percentile250": dict(
        func=lambda t, md, C: _amt_percentile(t, C, 250),
        desc="当日成交额在过去250日自身分布中的经验分位", direction=-1),
    "lq2_amt_trend_slope120": dict(
        func=lambda t, md, C: _amt_trend_slope(t, C, 120),
        desc="log成交额120日回归斜率×100（%/交易日）", direction=-1),
    "lq2_amt_accel": dict(
        func=lambda t, md, C: _amt_accel(t, C),
        desc="(均额5/均额20)/(均额20/均额60)-1：成交额加速", direction=-1),
}