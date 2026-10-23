# -*- coding: utf-8 -*-
"""相对市场因子（mk_*）：相对收益 / beta 调整 / 上下行捕获。

覆盖方向
--------
1. 相对收益      mk_rel_ret5 / mk_rel_ret20（短期相对反转）/ mk_rel_ret60 / mk_rel_ret120
2. 相对动量      mk_rel_mom12_1（252 日窗口、跳过最近 21 日）
3. 市场模型      mk_alpha60（年化 alpha）/ mk_resid_vol_ratio60（特质波动占比）
4. 相对强度分位  mk_rps60（当日横截面分位，规范 §5.10 明确要求）
5. 上下行捕获    mk_up_capture60 / mk_dn_capture60 / mk_capture_ratio60
6. beta / 相关   mk_beta_trend / mk_corr_chg / mk_lag_corr60
7. 相对风险      mk_rel_vol60 / mk_info_ratio60 / mk_rel_dd60 / mk_down_vol_share
8. 相对动能      mk_rel_accel / mk_rel_streak

实现约定
--------
- 只使用下标 <= t 的行；只 import numpy 与 .ctx
- 依赖 Ctx.mkt（等权市场收益，可能含 NaN）：所有窗口统计都与个股有效样本
  **配对**掩码（个股 R 与 mkt 同时有限才计入）
- 停牌日 R=0（数据层约定）计入收益类窗口；只有连续跑赢/跑输天数把停牌日跳过
- 出口统一过 _fin()：绝不返回 ±inf，无法计算处返回 NaN
- 市场模型 OLS / 上下行捕获 / 相对回撤的中间结果按 (t, w) 缓存在 Ctx 实例
  __dict__ 上（禁止模块级缓存；Ctx.truncated(t) 是全新对象，缓存天然隔离）
"""
from __future__ import annotations

import numpy as np

from .ctx import Ctx

_EPS = 1e-12
_SQRT252 = np.sqrt(252.0)


# ------------------------------------------------------------------ #
# 基础工具
# ------------------------------------------------------------------ #
def _fin(x) -> np.ndarray:
    """把 NaN / ±inf 统一收敛为 NaN，并确保 float64 输出。"""
    x = np.asarray(x, dtype="float64")
    return np.where(np.isfinite(x), x, np.nan)


def _min_cnt(w: int, frac: float = 0.6) -> int:
    """规范约定的窗口最少有效样本数：短窗 max(3, .6w)，长窗 max(20, .6w)。"""
    w = int(w)
    if w >= 120:
        return max(20, int(np.ceil(float(frac) * w)))
    return max(3, int(np.ceil(float(frac) * w)))


def _min_sub(w: int, frac: float = 0.2) -> int:
    """条件子样本（市场上涨日 / 下跌日）最少样本数，按窗口长度放宽。"""
    return max(3, int(np.ceil(float(frac) * int(w))))


def _cache(C: Ctx) -> dict:
    """每个 Ctx 实例独立缓存；禁止模块级缓存。"""
    return C.__dict__.setdefault("_mk_cache", {})


def _put(ck: dict, key, val, cap: int = 64) -> None:
    """写入实例缓存；条目数超过 cap 时按插入顺序淘汰最老的条目。"""
    ck[key] = val
    if len(ck) > int(cap):
        for k0 in list(ck.keys())[: len(ck) - int(cap)]:
            ck.pop(k0, None)


def _win(t: int, C: Ctx, w: int):
    """取 [t-w+1, t] 的个股收益窗口 W 与对齐的市场收益向量 M。"""
    t = int(t)
    lo, W = C.sl(t, int(w), C.R)
    M = np.asarray(C.mkt[lo:t + 1], dtype="float64")
    if M.shape[0] != W.shape[0]:                      # 理论不会发生，做防御
        n = min(W.shape[0], M.shape[0])
        W, M = W[:n], M[:n]
    return W, M


# ------------------------------------------------------------------ #
# 配对掩码市场模型 OLS（alpha / beta / 特质波动 / 跟踪误差 / 信息比率共享）
# ------------------------------------------------------------------ #
def _pair_ols(t: int, C: Ctx, w: int) -> dict:
    """按 (t, w) 缓存的配对掩码 OLS：R_i = alpha + beta * mkt + eps。

    用「掩码一阶/二阶矩」直接求方差协方差（sum, sum2, cross 三次归约），
    不构造中心化矩阵，250 日窗口也能压到 20ms 量级。

    返回 dict（全部为 (k,) 数组）：
      mu_y / mu_x  配对窗口内的个股 / 市场均值
      vy / vx / cxy 配对窗口方差协方差（ddof=1）
      beta / alpha 市场模型系数（alpha 为日频）
      var_rel      R_i - mkt 的窗口方差（跟踪误差平方，ddof=1）
      resid_var    OLS 残差方差 = vy - beta * cxy
      corr         个股与市场的窗口相关系数
      cnt          配对有效样本数
    """
    t = int(t)
    w = int(w)
    ck = _cache(C)
    key = ("ols", t, w)
    got = ck.get(key)
    if got is not None:
        return got

    k = C.k
    W, M = _win(t, C, w)
    if W.shape[0] < 2:
        nan = np.full(k, np.nan, dtype="float64")
        out = dict(mu_y=nan.copy(), mu_x=nan.copy(), vy=nan.copy(), vx=nan.copy(),
                   cxy=nan.copy(), beta=nan.copy(), alpha=nan.copy(),
                   var_rel=nan.copy(), resid_var=nan.copy(), corr=nan.copy(),
                   cnt=np.zeros(k, dtype="float64"))
        _put(ck, key, out)
        return out

    ok = np.isfinite(W) & np.isfinite(M)[:, None]
    cnt = ok.sum(0).astype("float64")
    Mf = np.where(np.isfinite(M), M, 0.0)             # 市场缺失日不参与（乘 0 保持有限）
    z = np.array(W, dtype="float64", copy=True)       # 掩码后的个股收益
    z[~ok] = 0.0
    sw = z.sum(0)
    sw2 = np.einsum("ij,ij->j", z, z)
    sm = np.einsum("i,ij->j", Mf, ok)
    sm2 = np.einsum("i,ij->j", Mf * Mf, ok)
    swm = np.einsum("ij,i->j", z, Mf)
    den = np.maximum(cnt, 1.0)
    df = np.maximum(cnt - 1.0, 1.0)
    mu_y = np.where(cnt > 0.0, sw / den, np.nan)
    mu_x = np.where(cnt > 0.0, sm / den, np.nan)
    vy = np.maximum((sw2 - sw * sw / den) / df, 0.0)
    vx = np.maximum((sm2 - sm * sm / den) / df, 0.0)
    cxy = (swm - sw * sm / den) / df

    good = (cnt >= float(_min_cnt(w))) & (vx > _EPS)
    beta = np.where(good, cxy / np.where(good, vx, 1.0), np.nan)
    alpha = np.where(good, mu_y - beta * mu_x, np.nan)
    beta_safe = np.where(good, beta, 0.0)
    var_rel = np.where(good, np.maximum(vy + vx - 2.0 * cxy, 0.0), np.nan)
    resid_var = np.where(good, np.maximum(vy - beta_safe * cxy, 0.0), np.nan)
    den_corr = np.sqrt(np.maximum(vx * vy, 0.0))
    corr_ok = good & (vy > _EPS) & (den_corr > _EPS)
    corr = np.where(corr_ok, cxy / np.where(corr_ok, den_corr, 1.0), np.nan)

    out = dict(mu_y=_fin(mu_y), mu_x=_fin(mu_x), vy=_fin(vy), vx=_fin(vx),
               cxy=_fin(cxy), beta=_fin(beta), alpha=_fin(alpha),
               var_rel=_fin(var_rel), resid_var=_fin(resid_var),
               corr=_fin(corr), cnt=cnt)
    _put(ck, key, out)
    return out


# ------------------------------------------------------------------ #
# 相对累计收益（复利）
# ------------------------------------------------------------------ #
def _rel_cum(t: int, C: Ctx, w: int, skip: int = 0) -> np.ndarray:
    """[t-skip-w+1, t-skip] 窗口：个股累计收益 - 同期市场累计收益（配对掩码、复利）。"""
    t = int(t)
    w = int(w)
    skip = int(skip)
    hi = t - skip
    lo = hi - w + 1
    if hi < 0 or lo < 0:
        return np.full(C.k, np.nan, dtype="float64")
    W = C.R[lo:hi + 1]
    M = np.asarray(C.mkt[lo:hi + 1], dtype="float64")
    ok = np.isfinite(W) & np.isfinite(M)[:, None]
    cnt = ok.sum(0).astype("float64")
    ci = np.prod(np.where(ok, 1.0 + W, 1.0), axis=0) - 1.0
    cm = np.prod(np.where(ok, 1.0 + M[:, None], 1.0), axis=0) - 1.0
    return _fin(np.where(cnt >= float(_min_cnt(w)), ci - cm, np.nan))


# ------------------------------------------------------------------ #
# 横截面分位（仅用当日一行）
# ------------------------------------------------------------------ #
def _cs_pct(x) -> np.ndarray:
    """当日横截面分位（0~1，并列取平均秩）；有效样本 < 3 返回全 NaN。"""
    v = np.asarray(x, dtype="float64")
    out = np.full(v.shape, np.nan, dtype="float64")
    ok = np.isfinite(v)
    n = int(ok.sum())
    if n < 3:
        return out
    idx = np.flatnonzero(ok)
    _, inv, cnts = np.unique(v[idx], return_inverse=True, return_counts=True)
    start = np.cumsum(cnts) - cnts
    avg_rank = start + (cnts - 1.0) / 2.0
    out[idx] = avg_rank[inv] / (n - 1.0)
    return out


# ------------------------------------------------------------------ #
# 上下行捕获
# ------------------------------------------------------------------ #
def _capture(t: int, C: Ctx, w: int) -> dict:
    """按 (t, w) 缓存的上下行捕获统计（配对掩码，涨跌日由 mkt 符号决定）。

    up_cap = 市场上涨日个股平均收益 / 市场上涨日市场平均收益
    dn_cap = 市场下跌日个股平均收益 / 市场下跌日市场平均收益（分母为负、比值>0）
    ratio  = up_cap / dn_cap
    """
    t = int(t)
    w = int(w)
    ck = _cache(C)
    key = ("cap", t, w)
    got = ck.get(key)
    if got is not None:
        return got

    k = C.k
    W, M = _win(t, C, w)
    ok = np.isfinite(W) & np.isfinite(M)[:, None]
    up = ok & (M[:, None] > 0.0)
    dn = ok & (M[:, None] < 0.0)
    cu = up.sum(0).astype("float64")
    cd = dn.sum(0).astype("float64")
    du = np.maximum(cu, 1.0)
    dd = np.maximum(cd, 1.0)
    y_up = np.where(up, W, 0.0).sum(0) / du
    m_up = np.where(up, M[:, None], 0.0).sum(0) / du
    y_dn = np.where(dn, W, 0.0).sum(0) / dd
    m_dn = np.where(dn, M[:, None], 0.0).sum(0) / dd
    need = float(_min_sub(w))
    g_up = (cu >= need) & (m_up > _EPS)
    g_dn = (cd >= need) & (m_dn < -_EPS)
    up_cap = np.where(g_up, y_up / np.where(g_up, m_up, 1.0), np.nan)
    dn_cap = np.where(g_dn, y_dn / np.where(g_dn, m_dn, 1.0), np.nan)
    r_ok = np.isfinite(up_cap) & np.isfinite(dn_cap) & (dn_cap > _EPS)
    ratio = np.where(r_ok, up_cap / np.where(r_ok, dn_cap, 1.0), np.nan)
    out = dict(up_cap=_fin(up_cap), dn_cap=_fin(dn_cap), ratio=_fin(ratio))
    _put(ck, key, out)
    return out


# ------------------------------------------------------------------ #
# 相对回撤（收益路径构造，回撤 <= 0）
# ------------------------------------------------------------------ #
def _rel_dd(t: int, C: Ctx, w: int) -> np.ndarray:
    """个股窗口最大回撤 - 市场窗口最大回撤（配对掩码，均按收益路径 cumprod 构造）。"""
    t = int(t)
    w = int(w)
    ck = _cache(C)
    key = ("dd", t, w)
    got = ck.get(key)
    if got is not None:
        return got
    W, M = _win(t, C, w)
    ok = np.isfinite(W) & np.isfinite(M)[:, None]
    cnt = ok.sum(0).astype("float64")
    g = np.where(ok, 1.0 + W, 1.0)
    gm = np.where(ok, 1.0 + M[:, None], 1.0)
    p = np.cumprod(g, axis=0)
    pm = np.cumprod(gm, axis=0)
    dd = (p / np.maximum.accumulate(p, axis=0) - 1.0).min(axis=0)
    ddm = (pm / np.maximum.accumulate(pm, axis=0) - 1.0).min(axis=0)
    out = _fin(np.where(cnt >= float(_min_cnt(w)), dd - ddm, np.nan))
    _put(ck, key, out)
    return out


# ------------------------------------------------------------------ #
# 滞后相关 / 相对波动占比 / 连续跑赢
# ------------------------------------------------------------------ #
def _lag_corr(t: int, C: Ctx, w: int) -> np.ndarray:
    """corr(R_i[j], mkt[j-1])：个股当日收益与前一交易日市场收益的配对相关。"""
    t = int(t)
    W, M = _win(t, C, w)
    if W.shape[0] < 2:
        return np.full(C.k, np.nan, dtype="float64")
    Y = W[1:]
    X = M[:-1]
    ok = np.isfinite(Y) & np.isfinite(X)[:, None]
    cnt = ok.sum(0).astype("float64")
    den = np.maximum(cnt, 1.0)
    my = np.where(ok, Y, 0.0).sum(0) / den
    mx = np.where(ok, X[:, None], 0.0).sum(0) / den
    yc = np.where(ok, Y - my, 0.0)
    xc = np.where(ok, X[:, None] - mx, 0.0)
    df = np.maximum(cnt - 1.0, 1.0)
    vy = (yc * yc).sum(0) / df
    vx = (xc * xc).sum(0) / df
    cxy = (yc * xc).sum(0) / df
    den_c = np.sqrt(np.maximum(vx * vy, 0.0))
    good = (cnt >= float(_min_cnt(w))) & (vx > _EPS) & (vy > _EPS) & (den_c > _EPS)
    out = np.where(good, cxy / np.where(good, den_c, 1.0), np.nan)
    return _fin(out)


def _down_vol_share(t: int, C: Ctx, w: int) -> np.ndarray:
    """市场下跌日的个股平方收益 / 窗口总平方收益（配对掩码）。"""
    W, M = _win(t, C, w)
    ok = np.isfinite(W) & np.isfinite(M)[:, None]
    cnt = ok.sum(0).astype("float64")
    sq = np.where(ok, W * W, 0.0)
    tot = sq.sum(0)
    dn = np.where(ok & (M[:, None] < 0.0), sq, 0.0).sum(0)
    good = (cnt >= float(_min_cnt(w))) & (tot > _EPS)
    return _fin(np.where(good, dn / np.where(good, tot, 1.0), np.nan))


def _rel_streak(t: int, C: Ctx, lookback: int = 60) -> np.ndarray:
    """连续跑赢/跑输市场的带符号天数（跑赢 +1 / 跑输 -1，停牌日跳过）。

    从 t 往回扫，方向反转即停；最长回看 lookback 个交易日，从未形成方向返回 NaN。
    """
    t = int(t)
    k = C.k
    sign = np.zeros(k, dtype="float64")
    count = np.zeros(k, dtype="float64")
    active = np.ones(k, dtype=bool)
    lo = max(t - int(lookback) + 1, 0)
    for j in range(t, lo - 1, -1):
        if not active.any():
            break
        Rj = C.R[j]
        Mj = C.mkt[j]
        Aj = C.A[j]
        valid = np.isfinite(Rj) & np.isfinite(Mj) & (Aj > 0.0)
        d = Rj - Mj
        s = np.where(d > 0.0, 1.0, np.where(d < 0.0, -1.0, 0.0))
        first = active & (count == 0.0) & valid & (s != 0.0)
        sign = np.where(first, s, sign)
        count = np.where(first, 1.0, count)
        nxt = active & (count > 0.0) & valid & ~first
        same = nxt & (s == sign)
        count = np.where(same, count + 1.0, count)
        active = active & ~(nxt & ~same)
    return _fin(np.where(count > 0.0, sign * count, np.nan))


# ------------------------------------------------------------------ #
# 因子入口
# ------------------------------------------------------------------ #
def _f_alpha60(t: int, C: Ctx) -> np.ndarray:
    """60 日市场模型 alpha 年化：alpha_daily * 252。"""
    return _fin(_pair_ols(t, C, 60)["alpha"] * 252.0)


def _f_resid_vol_ratio(t: int, C: Ctx) -> np.ndarray:
    """60 日特质波动率 / 总波动率 = sqrt(resid_var / vy) = sqrt(1 - R^2)。"""
    o = _pair_ols(t, C, 60)
    vy = o["vy"]
    rv = o["resid_var"]
    good = np.isfinite(vy) & np.isfinite(rv) & (vy > _EPS)
    r = np.sqrt(np.maximum(rv, 0.0) / np.where(good, vy, 1.0))
    return _fin(np.where(good, np.minimum(r, 1.0), np.nan))


def _f_rps60(t: int, C: Ctx) -> np.ndarray:
    """60 日相对强度的当日横截面分位（0~1）。"""
    return _cs_pct(_rel_cum(t, C, 60, 0))


def _f_up_capture(t: int, C: Ctx) -> np.ndarray:
    return _capture(t, C, 60)["up_cap"]


def _f_dn_capture(t: int, C: Ctx) -> np.ndarray:
    return _capture(t, C, 60)["dn_cap"]


def _f_capture_ratio(t: int, C: Ctx) -> np.ndarray:
    return _capture(t, C, 60)["ratio"]


def _f_beta_trend(t: int, C: Ctx) -> np.ndarray:
    b60 = _pair_ols(t, C, 60)["beta"]
    b250 = _pair_ols(t, C, 250)["beta"]
    good = np.isfinite(b60) & np.isfinite(b250) & (b250 > _EPS)
    return _fin(np.where(good, b60 / np.where(good, b250, 1.0) - 1.0, np.nan))


def _f_corr_chg(t: int, C: Ctx) -> np.ndarray:
    c60 = _pair_ols(t, C, 60)["corr"]
    c250 = _pair_ols(t, C, 250)["corr"]
    return _fin(c60 - c250)


def _f_rel_accel(t: int, C: Ctx) -> np.ndarray:
    return _fin(_rel_cum(t, C, 20, 0) - _rel_cum(t, C, 40, 20))


def _f_rel_vol60(t: int, C: Ctx) -> np.ndarray:
    vr = _pair_ols(t, C, 60)["var_rel"]
    return _fin(np.sqrt(np.maximum(vr, 0.0)))


def _f_info_ratio60(t: int, C: Ctx) -> np.ndarray:
    o = _pair_ols(t, C, 60)
    num = o["mu_y"] - o["mu_x"]
    sd = np.sqrt(np.maximum(o["var_rel"], 0.0))
    good = np.isfinite(num) & np.isfinite(sd) & (sd > _EPS)
    return _fin(np.where(good, num / np.where(good, sd, 1.0) * _SQRT252, np.nan))


# ------------------------------------------------------------------ #
FACTORS = {
    # --- 1. 相对收益 / 相对动量 ---
    "mk_rel_ret5": dict(func=lambda t, md, C: _rel_cum(t, C, 5, 0),
                        desc="过去5日个股累计收益-同期市场累计收益（短期相对反转）", direction=-1),
    "mk_rel_ret20": dict(func=lambda t, md, C: _rel_cum(t, C, 20, 0),
                         desc="过去20日个股累计收益-同期市场累计收益", direction=-1),
    "mk_rel_ret60": dict(func=lambda t, md, C: _rel_cum(t, C, 60, 0),
                         desc="过去60日相对强度：个股累计收益-同期市场累计收益", direction=1),
    "mk_rel_ret120": dict(func=lambda t, md, C: _rel_cum(t, C, 120, 0),
                          desc="过去120日相对强度：个股累计收益-同期市场累计收益", direction=1),
    "mk_rel_mom12_1": dict(func=lambda t, md, C: _rel_cum(t, C, 231, 21),
                           desc="12-1相对动量：252日窗口跳过最近21日的相对累计收益", direction=1),
    # --- 2. 市场模型 ---
    "mk_alpha60": dict(func=lambda t, md, C: _f_alpha60(t, C),
                       desc="60日市场模型alpha年化（配对OLS，alpha*252）", direction=1),
    "mk_resid_vol_ratio60": dict(func=lambda t, md, C: _f_resid_vol_ratio(t, C),
                                 desc="60日特质波动率/总波动率=sqrt(1-R^2)（配对OLS）", direction=-1),
    # --- 3. 相对强度分位 ---
    "mk_rps60": dict(func=lambda t, md, C: _f_rps60(t, C),
                     desc="60日相对强度的当日横截面分位(0~1，并列取平均秩)", direction=1),
    # --- 4. 上下行捕获 ---
    "mk_up_capture60": dict(func=lambda t, md, C: _f_up_capture(t, C),
                            desc="60日上行捕获率：市场上涨日个股平均收益/市场平均收益", direction=1),
    "mk_dn_capture60": dict(func=lambda t, md, C: _f_dn_capture(t, C),
                            desc="60日下跌捕获率：市场下跌日个股平均收益/市场平均收益（越小越好）", direction=-1),
    "mk_capture_ratio60": dict(func=lambda t, md, C: _f_capture_ratio(t, C),
                               desc="60日上行捕获率/下行捕获率", direction=1),
    # --- 5. beta / 相关性 ---
    "mk_beta_trend": dict(func=lambda t, md, C: _f_beta_trend(t, C),
                          desc="beta60/beta250-1：市场敏感度变化（配对OLS）", direction=-1),
    "mk_corr_chg": dict(func=lambda t, md, C: _f_corr_chg(t, C),
                        desc="corr60-corr250：与市场相关性变化（配对OLS）", direction=-1),
    "mk_lag_corr60": dict(func=lambda t, md, C: _lag_corr(t, C, 60),
                          desc="60日corr(个股当日收益, 市场前一日收益)：滞后反应度", direction=-1),
    # --- 6. 相对风险 / 相对动能 ---
    "mk_rel_accel": dict(func=lambda t, md, C: _f_rel_accel(t, C),
                         desc="近20日相对收益-前40日相对收益（相对动能加速度）", direction=-1),
    "mk_rel_vol60": dict(func=lambda t, md, C: _f_rel_vol60(t, C),
                         desc="60日相对收益波动率（跟踪误差，配对掩码，ddof=1）", direction=-1),
    "mk_info_ratio60": dict(func=lambda t, md, C: _f_info_ratio60(t, C),
                            desc="60日信息比率：mean(相对收益)/std(相对收益)*sqrt(252)", direction=1),
    "mk_rel_dd60": dict(func=lambda t, md, C: _rel_dd(t, C, 60),
                        desc="个股60日最大回撤-市场60日最大回撤（回撤<=0，越大越好）", direction=1),
    "mk_rel_streak": dict(func=lambda t, md, C: _rel_streak(t, C, 60),
                          desc="连续跑赢/跑输市场天数（跑赢+、跑输-，停牌跳过，回看60日）", direction=-1),
    "mk_down_vol_share": dict(func=lambda t, md, C: _down_vol_share(t, C, 60),
                              desc="市场下跌日个股平方收益占60日总平方收益比例", direction=-1),
}