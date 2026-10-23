# -*- coding: utf-8 -*-
"""风险扩展因子（rk2_*）：beta / 特质风险 / 尾部 / 区间波动估计。

本模块只使用 Ctx 中下标 <= t 的数据，不导入 pandas 或 core.*。
所有窗口统计均使用掩码版本；无法计算时返回 NaN，绝不返回 +/-inf。
"""
import numpy as np

from .ctx import Ctx

_EPS = 1e-12
_SQRT252 = np.sqrt(252.0)
_FOUR_LN2 = 4.0 * np.log(2.0)
_TWO_LN2_M1 = 2.0 * np.log(2.0) - 1.0


# ------------------------------------------------------------------ #
# 基础工具
# ------------------------------------------------------------------ #
def _fin(x):
    """把 NaN / +/-inf 统一收敛为 NaN，并确保 float64 输出。"""
    x = np.asarray(x, dtype=np.float64)
    return np.where(np.isfinite(x), x, np.nan)


def _cache(C: Ctx):
    """每个 Ctx 实例独立缓存；禁止模块级缓存。"""
    return C.__dict__.setdefault("_risk2_cache", {})


def _min_cnt(w: int) -> int:
    """规范约定的窗口最少有效样本数。"""
    w = int(w)
    if w >= 120:
        return max(20, int(np.ceil(0.6 * w)))
    return max(3, int(np.ceil(0.6 * w)))


def _sl(C: Ctx, t: int, w: int, X):
    """安全取窗口；非法 t 返回空窗口。"""
    t = int(t)
    X = np.asarray(X)
    if X.ndim != 2 or t < 0 or t >= X.shape[0]:
        return 0, X[0:0]
    return C.sl(t, w, X)


def _mean_var(X, ok, min_cnt):
    """掩码均值和样本方差（ddof=1）。"""
    cnt = ok.sum(0).astype(np.float64)
    den = np.maximum(cnt, 1.0)
    mu = np.where(ok, X, 0.0).sum(0) / den
    c = np.where(ok, X - mu, 0.0)
    var = (c * c).sum(0) / np.maximum(cnt - 1.0, 1.0)
    var = np.maximum(var, 0.0)
    good = cnt >= float(min_cnt)
    return np.where(good, mu, np.nan), np.where(good, var, np.nan), cnt


# ------------------------------------------------------------------ #
# 市场模型 OLS / 残差统计
# ------------------------------------------------------------------ #
def _pair_ols(W, M, min_cnt, subset=None, higher=True):
    """配对掩码 OLS：R_i = alpha + beta * mkt + eps。

    返回 beta、alpha、残差波动、残差偏度/峰度、R^2、相关系数等聚合量。
    不保存完整残差矩阵，避免缓存过大。
    """
    W = np.asarray(W, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64).reshape(-1)
    k = W.shape[1] if W.ndim == 2 else 0
    nan = np.full(k, np.nan, dtype=np.float64)
    zero = np.zeros(k, dtype=np.float64)
    if W.ndim != 2 or M.shape[0] != W.shape[0]:
        return dict(beta=nan, alpha=nan, ivol=nan, skew=nan, kurt=nan,
                    r2=nan, corr=nan, cnt=zero, mu_y=nan, mu_x=nan)

    ok = np.isfinite(W) & np.isfinite(M)[:, None]
    if subset is not None:
        ss = np.asarray(subset, dtype=bool).reshape(-1)
        if ss.shape[0] == W.shape[0]:
            ok &= ss[:, None]

    cnt = ok.sum(0).astype(np.float64)
    den = np.maximum(cnt, 1.0)
    mu_y = np.where(ok, W, 0.0).sum(0) / den
    mu_x = np.where(ok, M[:, None], 0.0).sum(0) / den
    yc = np.where(ok, W - mu_y, 0.0)
    xc = np.where(ok, M[:, None] - mu_x, 0.0)

    ss_y = (yc * yc).sum(0)
    ss_xy = (yc * xc).sum(0)
    df = np.maximum(cnt - 1.0, 1.0)
    vy = ss_y / df
    vx = (xc * xc).sum(0) / df
    cxy = ss_xy / df

    good = (cnt >= float(min_cnt)) & (vx > _EPS)
    beta = np.where(good, cxy / np.where(good, vx, 1.0), np.nan)
    alpha = np.where(good, mu_y - beta * mu_x, np.nan)
    beta_safe = np.where(good, beta, 0.0)
    n = np.maximum(cnt, 1.0)

    if higher:
        resid = yc - beta_safe[None, :] * xc
        m2 = (resid * resid).sum(0) / n
        m3 = (resid * resid * resid).sum(0) / n
        m4 = (resid * resid * resid * resid).sum(0) / n
        m2_ok = m2 > _EPS
        skew = np.where(good & m2_ok,
                        m3 / np.where(m2_ok, m2, 1.0) ** 1.5, np.nan)
        kurt = np.where(good & m2_ok,
                        m4 / np.where(m2_ok, m2, 1.0) ** 2 - 3.0, np.nan)
    else:
        # OLS 正交性：sum(eps^2)=sum(yc^2)-beta*sum(yc*xc)，无需构造残差矩阵。
        ss_res = ss_y - beta_safe * ss_xy
        m2 = np.maximum(ss_res, 0.0) / n
        skew = np.full(k, np.nan, dtype=np.float64)
        kurt = np.full(k, np.nan, dtype=np.float64)
    ivol = np.where(good, np.sqrt(np.maximum(m2, 0.0)) * _SQRT252, np.nan)

    den_corr = np.sqrt(np.maximum(vx * vy, 0.0))
    corr_ok = good & (vy > _EPS) & (den_corr > _EPS)
    corr = np.where(corr_ok, cxy / np.where(corr_ok, den_corr, 1.0), np.nan)
    r2 = corr * corr

    return dict(beta=_fin(beta), alpha=_fin(alpha), ivol=_fin(ivol),
                skew=_fin(skew), kurt=_fin(kurt), r2=_fin(r2),
                corr=_fin(corr), cnt=cnt, mu_y=mu_y, mu_x=mu_x)


def _slope_only(Y, X, min_cnt):
    """只计算配对 OLS 斜率与均值，供滚动波动率趋势因子使用。"""
    Y = np.asarray(Y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64).reshape(-1)
    k = Y.shape[1] if Y.ndim == 2 else 0
    nan = np.full(k, np.nan, dtype=np.float64)
    if Y.ndim != 2 or X.shape[0] != Y.shape[0]:
        return nan, nan.copy(), nan.copy(), nan.copy()
    ok = np.isfinite(Y) & np.isfinite(X)[:, None]
    cnt = ok.sum(0).astype(np.float64)
    den = np.maximum(cnt, 1.0)
    my = np.where(ok, Y, 0.0).sum(0) / den
    mx = np.where(ok, X[:, None], 0.0).sum(0) / den
    yc = np.where(ok, Y - my, 0.0)
    xc = np.where(ok, X[:, None] - mx, 0.0)
    df = np.maximum(cnt - 1.0, 1.0)
    vy = (yc * yc).sum(0) / df
    vx = (xc * xc).sum(0) / df
    cxy = (yc * xc).sum(0) / df
    good = (cnt >= float(min_cnt)) & (vx > _EPS)
    beta = np.where(good, cxy / np.where(good, vx, 1.0), np.nan)
    return beta, my, vy, vx

def _ols(t: int, C: Ctx, w: int):
    """按 (t, w) 缓存的窗口 OLS，供 beta/alpha/残差类因子共享。"""
    key = ("ols", int(t), int(w))
    cache = _cache(C)
    if key in cache:
        return cache[key]
    lo, W = _sl(C, t, w, C.R)
    if W.shape[0] == 0:
        out = _pair_ols(W, np.empty(0, dtype=np.float64), _min_cnt(w), higher=(int(w) == 60))
    else:
        M = C.mkt[lo:int(t) + 1]
        out = _pair_ols(W, M, _min_cnt(w), higher=(int(w) == 60))
    cache[key] = out
    return out


def _cond_beta(t: int, C: Ctx, w: int, up: bool):
    """仅用市场上涨/下跌日的配对掩码 OLS beta。"""
    key = ("beta_up" if up else "beta_dn", int(t), int(w))
    cache = _cache(C)
    if key in cache:
        return cache[key]
    lo, W = _sl(C, t, w, C.R)
    if W.shape[0] == 0:
        out = np.full(C.k, np.nan, dtype=np.float64)
    else:
        M = C.mkt[lo:int(t) + 1]
        subset = (M > 0.0) if up else (M < 0.0)
        min_sub = max(5, int(np.ceil(0.2 * int(w))))
        out = _pair_ols(W, M, min_sub, subset=subset, higher=False)["beta"]
    cache[key] = out
    return out


def _tracking_error(t: int, C: Ctx, w: int):
    """std(R - mkt)，配对掩码。"""
    lo, W = _sl(C, t, w, C.R)
    if W.shape[0] == 0:
        return np.full(C.k, np.nan, dtype=np.float64)
    M = C.mkt[lo:int(t) + 1]
    ok = np.isfinite(W) & np.isfinite(M)[:, None]
    D = np.where(ok, W - M[:, None], 0.0)
    cnt = ok.sum(0).astype(np.float64)
    mu = D.sum(0) / np.maximum(cnt, 1.0)
    c = np.where(ok, D - mu, 0.0)
    var = (c * c).sum(0) / np.maximum(cnt - 1.0, 1.0)
    out = np.sqrt(np.maximum(var, 0.0))
    return _fin(np.where(cnt >= _min_cnt(w), out, np.nan))


# ------------------------------------------------------------------ #
# 尾部 / 分位 / 区间风险
# ------------------------------------------------------------------ #
def _quantile(W, ok, q, min_cnt):
    """逐列掩码分位数（线性插值，与 numpy.quantile 默认口径一致）。"""
    S = np.sort(np.where(ok, W, np.inf), axis=0)
    n = ok.sum(0).astype(np.int64)
    n_safe = np.maximum(n, 1)
    pos = float(q) * (n_safe - 1).astype(np.float64)
    lo = np.floor(pos).astype(np.int64)
    hi = np.ceil(pos).astype(np.int64)
    lo = np.clip(lo, 0, n_safe - 1)
    hi = np.clip(hi, 0, n_safe - 1)
    cols = np.arange(W.shape[1], dtype=np.int64)
    a = S[lo, cols]
    b = S[hi, cols]
    with np.errstate(invalid="ignore", over="ignore"):
        out = a + (b - a) * (pos - lo)
    return np.where(n >= int(min_cnt), out, np.nan)


def _tail_mean(W, ok, q, min_cnt):
    """最差 q 比例样本的均值（经验 CVaR）。"""
    S = np.sort(np.where(ok, W, np.inf), axis=0)
    n = ok.sum(0).astype(np.int64)
    nt = np.maximum(1, np.ceil(float(q) * n).astype(np.int64))
    rows = np.arange(S.shape[0], dtype=np.int64)[:, None]
    mask = rows < nt[None, :]
    sums = np.where(mask, S, 0.0).sum(0)
    out = sums / np.maximum(nt, 1)
    return np.where(n >= int(min_cnt), out, np.nan)


def _semivar(t: int, C: Ctx, w: int, down: bool):
    """下行/上行半方差（按有效样本数平均后年化）。"""
    lo, W = _sl(C, t, w, C.R)
    ok = np.isfinite(W)
    cnt = ok.sum(0).astype(np.float64)
    if down:
        x = np.minimum(W, 0.0)
    else:
        x = np.maximum(W, 0.0)
    val = (np.where(ok, x, 0.0) ** 2).sum(0) / np.maximum(cnt, 1.0) * 252.0
    return _fin(np.where(cnt >= _min_cnt(w), val, np.nan))


def _drawdown_stats(t: int, C: Ctx, w: int):
    """窗口内回撤统计：Ulcer、underwater 天数占比、最大回撤速度。"""
    key = ("dd", int(t), int(w))
    cache = _cache(C)
    if key in cache:
        return cache[key]

    lo, P = _sl(C, t, w, C.C)
    k = P.shape[1] if P.ndim == 2 else C.k
    if P.shape[0] == 0:
        z = np.full(k, np.nan, dtype=np.float64)
        out = dict(ulcer=z, underwater=z.copy(), ddspeed=z.copy())
        cache[key] = out
        return out

    ok = np.isfinite(P) & (P > 0.0)
    cnt = ok.sum(0).astype(np.float64)
    run = np.maximum.accumulate(np.where(ok, P, -np.inf), axis=0)
    run_ok = np.isfinite(run) & (run > 0.0)
    denom = np.where(run_ok, run, 1.0)
    dd = np.where(ok & run_ok, 1.0 - P / denom, np.nan)
    good = cnt >= _min_cnt(w)

    valid = np.isfinite(dd)
    ulcer = np.sqrt(np.where(valid, dd * dd, 0.0).sum(0) / np.maximum(cnt, 1.0))
    underwater = np.where(valid & (dd > _EPS), 1.0, 0.0).sum(0) / np.maximum(cnt, 1.0)

    dd_f = np.where(valid, dd, -np.inf)
    depth = dd_f.max(0)
    trough = np.argmax(dd_f, axis=0)
    peak_mask = ok & run_ok & (P == run)
    peak = np.argmax(peak_mask, axis=0)
    days = np.maximum(trough - peak, 0)
    speed = np.where(depth > _EPS, depth / np.maximum(days, 1), 0.0)

    out = dict(ulcer=_fin(np.where(good, ulcer, np.nan)),
               underwater=_fin(np.where(good, underwater, np.nan)),
               ddspeed=_fin(np.where(good, speed, np.nan)))
    cache[key] = out
    return out


def _range_stats(t: int, C: Ctx, w: int):
    """Parkinson / Garman-Klass / Rogers-Satchell / Yang-Zhang 日频波动率。"""
    key = ("range", int(t), int(w))
    cache = _cache(C)
    if key in cache:
        return cache[key]

    lo, H = _sl(C, t, w, C.ADJ_H)
    _, L = _sl(C, t, w, C.ADJ_L)
    _, O = _sl(C, t, w, C.ADJ_O)
    _, Cl = _sl(C, t, w, C.C)
    k = H.shape[1] if H.ndim == 2 else C.k
    if H.shape[0] == 0:
        z = np.full(k, np.nan, dtype=np.float64)
        out = dict(park=z, gk=z.copy(), rs=z.copy(), yz=z.copy())
        cache[key] = out
        return out

    ok_hl = np.isfinite(H) & np.isfinite(L) & (H > 0.0) & (L > 0.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        lhl = np.log(H / L)
    park_term = lhl * lhl
    cnt_hl = ok_hl.sum(0).astype(np.float64)
    park_mean = np.where(ok_hl, park_term, 0.0).sum(0) / np.maximum(cnt_hl, 1.0)
    park = np.sqrt(np.maximum(park_mean, 0.0) / _FOUR_LN2)
    park_good = cnt_hl >= _min_cnt(w)

    # 60 日窗同时服务 GK/RS/YZ；120 日只有 Parkinson 因子，避免无谓计算。
    if int(w) == 60:
        ok_all = ok_hl & np.isfinite(O) & np.isfinite(Cl) & (O > 0.0) & (Cl > 0.0)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            lhc = np.log(H / Cl)
            lho = np.log(H / O)
            llc = np.log(L / Cl)
            llo = np.log(L / O)
            lco = np.log(Cl / O)
        cnt_all = ok_all.sum(0).astype(np.float64)
        gk_mean = np.where(ok_all, 0.5 * park_term - _TWO_LN2_M1 * lco * lco, 0.0).sum(0) \
            / np.maximum(cnt_all, 1.0)
        gk = np.where(gk_mean >= 0.0, np.sqrt(np.maximum(gk_mean, 0.0)), np.nan)
        gk_good = cnt_all >= _min_cnt(w)

        rs_term = lhc * lho + llc * llo
        rs_mean = np.where(ok_all, rs_term, 0.0).sum(0) / np.maximum(cnt_all, 1.0)
        rs = np.where(rs_mean >= 0.0, np.sqrt(np.maximum(rs_mean, 0.0)), np.nan)
        rs_good = cnt_all >= _min_cnt(w)

        m = H.shape[0]
        prev = np.full_like(Cl, np.nan, dtype=np.float64)
        if lo > 0:
            prev[0] = C.C[lo - 1]
        if m > 1:
            prev[1:] = Cl[:-1]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            on = np.log(O / prev)
        ok_on = np.isfinite(on) & np.isfinite(prev) & (prev > 0.0) & (O > 0.0)
        mu_on, var_on, cnt_on = _mean_var(on, ok_on, _min_cnt(w))
        mu_co, var_co, cnt_co = _mean_var(lco, ok_all, _min_cnt(w))
        n_eff = np.minimum(np.minimum(cnt_on, cnt_co), cnt_all)
        n_safe = np.maximum(n_eff, 2.0)
        kfac = 0.34 / (1.34 + (n_safe + 1.0) / (n_safe - 1.0))
        yz_var = var_on + kfac * var_co + (1.0 - kfac) * rs_mean
        yz_good = (cnt_on >= _min_cnt(w)) & (cnt_co >= _min_cnt(w)) \
            & (cnt_all >= _min_cnt(w)) & np.isfinite(yz_var) & (yz_var >= 0.0)
        yz = np.where(yz_good, np.sqrt(np.maximum(yz_var, 0.0)), np.nan)
    else:
        gk = np.full(k, np.nan, dtype=np.float64)
        rs = np.full(k, np.nan, dtype=np.float64)
        yz = np.full(k, np.nan, dtype=np.float64)
        gk_good = np.zeros(k, dtype=bool)
        rs_good = np.zeros(k, dtype=bool)

    out = dict(park=_fin(np.where(park_good, park, np.nan)),
               gk=_fin(np.where(gk_good, gk, np.nan)),
               rs=_fin(np.where(rs_good, rs, np.nan)),
               yz=_fin(yz))
    cache[key] = out
    return out


def _bpv_stats(t: int, C: Ctx, w: int):
    """双幂次变差和跳跃占比。"""
    key = ("bpv", int(t), int(w))
    cache = _cache(C)
    if key in cache:
        return cache[key]

    lo, W = _sl(C, t, w, C.R)
    ok = np.isfinite(W)
    cnt = ok.sum(0).astype(np.float64)
    a = np.where(ok, np.abs(W), 0.0)
    pair_ok = ok[1:] & ok[:-1]
    pair_sum = np.where(pair_ok, a[1:] * a[:-1], 0.0).sum(0)
    bpv = (np.pi / 2.0) * pair_sum
    rv = (np.where(ok, W, 0.0) ** 2).sum(0)
    pair_cnt = pair_ok.sum(0).astype(np.float64)
    minc = _min_cnt(w)
    good = (cnt >= minc) & (pair_cnt >= max(3, minc - 1))
    jump = np.where(good & (rv > _EPS),
                    (rv - bpv) / np.where(good & (rv > _EPS), rv, 1.0), np.nan)
    out = dict(bpv=_fin(np.where(good, bpv, np.nan)), jump=_fin(jump))
    cache[key] = out
    return out


# ------------------------------------------------------------------ #
# 滚动波动率 / 波动率趋势
# ------------------------------------------------------------------ #
def _rolling_vol_series(X, first_end, roll, min_cnt):
    """对 2D/1D 序列计算每个日期的 roll 日滚动波动率。

    first_end 是期望输出的第一个局部结束下标；返回从该下标到末尾的序列。
    仅使用每个结束日及其之前的数据。
    """
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] == 0:
        shape = (0,) if X.ndim == 1 else (0, X.shape[1])
        return np.full(shape, np.nan, dtype=np.float64)

    ok = np.isfinite(X)
    cnt_all = np.cumsum(ok, axis=0, dtype=np.float64)
    sum_all = np.cumsum(np.where(ok, X, 0.0), axis=0)
    sq_all = np.cumsum(np.where(ok, X * X, 0.0), axis=0)

    idx = np.arange(int(first_end), X.shape[0], dtype=np.int64)
    if idx.size == 0:
        shape = (0,) if X.ndim == 1 else (0, X.shape[1])
        return np.full(shape, np.nan, dtype=np.float64)

    wi = np.maximum(idx - int(roll) + 1, 0)

    def _sub(CUM):
        out = CUM[idx].astype(np.float64, copy=True)
        has = wi > 0
        if np.any(has):
            out[has] -= CUM[wi[has] - 1]
        return out

    cnt = _sub(cnt_all)
    s = _sub(sum_all)
    q = _sub(sq_all)
    den = np.maximum(cnt, 1.0)
    var = (q - s * s / den) / np.maximum(cnt - 1.0, 1.0)
    var = np.maximum(var, 0.0)
    out = np.sqrt(var)
    return np.where(cnt >= float(min_cnt), out, np.nan)


def _rolling_vol_pair(t: int, C: Ctx, span: int = 60, roll: int = 20,
                      with_market: bool = True):
    """过去 span 日窗口内，roll 日滚动波动率序列（个股与可选市场）。"""
    t = int(t)
    if C.R.shape[0] == 0 or t < 0 or t >= C.R.shape[0]:
        V = np.full((0, C.k), np.nan, dtype=np.float64)
        return V, (np.full(0, np.nan, dtype=np.float64) if with_market else None)

    start = max(0, t - int(span) + 1)
    ext_start = max(0, start - int(roll) + 1)
    X = C.R[ext_start:t + 1]
    min_roll = max(3, int(np.ceil(0.6 * int(roll))))
    first_end = start - ext_start
    V = _rolling_vol_series(X, first_end, int(roll), min_roll)
    if not with_market:
        return V, None
    M = C.mkt[ext_start:t + 1]
    Vm = _rolling_vol_series(M, first_end, int(roll), min_roll)
    return V, Vm


def _varbeta(t: int, C: Ctx):
    """20 日滚动波动率对市场同序列的 OLS 斜率。"""
    V, Vm = _rolling_vol_pair(t, C, 60, 20)
    m = V.shape[0]
    if m < 3:
        return np.full(C.k, np.nan, dtype=np.float64)
    minc = max(10, int(np.ceil(0.6 * m)))
    return _fin(_slope_only(V, Vm, minc)[0])


def _vol_slope(t: int, C: Ctx):
    """20 日滚动波动率对时间回归的斜率 / 均值。"""
    V, _ = _rolling_vol_pair(t, C, 60, 20, with_market=False)
    m = V.shape[0]
    if m < 3:
        return np.full(C.k, np.nan, dtype=np.float64)
    x = np.arange(m, dtype=np.float64)
    minc = max(10, int(np.ceil(0.6 * m)))
    beta, mu, _, _ = _slope_only(V, x, minc)
    good = np.abs(mu) > _EPS
    out = np.where(good, beta / np.where(good, mu, 1.0), np.nan)
    return _fin(out)


def _vol_std(t: int, C: Ctx, w: int):
    """窗口收益样本标准差（掩码）。"""
    lo, W = _sl(C, t, w, C.R)
    ok = np.isfinite(W)
    return _fin(C.std(W, ok, _min_cnt(w)))


def _ewma_vol(t: int, C: Ctx, w: int, lam: float = 0.94):
    """RiskMetrics EWMA 波动率：零均值加权二阶矩，年化。"""
    lo, W = _sl(C, t, w, C.R)
    k = W.shape[1] if W.ndim == 2 else C.k
    if W.shape[0] == 0:
        return np.full(k, np.nan, dtype=np.float64)
    m = W.shape[0]
    ages = np.arange(m - 1, -1, -1, dtype=np.float64)
    weights = float(lam) ** ages
    ok = np.isfinite(W)
    wv = weights[:, None] * ok
    den = wv.sum(0)
    x2 = np.where(ok, W * W, 0.0)
    var = (wv * x2).sum(0) / np.maximum(den, 1.0)
    cnt = ok.sum(0).astype(np.float64)
    out = np.sqrt(np.maximum(var, 0.0) * 252.0)
    return _fin(np.where(cnt >= _min_cnt(w), out, np.nan))


# ------------------------------------------------------------------ #
# 序列相关性 / 方差比 / 高阶矩
# ------------------------------------------------------------------ #
def _autocorr(t: int, C: Ctx, w: int):
    """窗口内日收益一阶自相关。"""
    lo, W = _sl(C, t, w, C.R)
    if W.shape[0] < 2:
        return np.full(C.k, np.nan, dtype=np.float64)
    x = W[:-1]
    y = W[1:]
    ok = np.isfinite(x) & np.isfinite(y)
    cnt = ok.sum(0).astype(np.float64)
    den = np.maximum(cnt, 1.0)
    mx = np.where(ok, x, 0.0).sum(0) / den
    my = np.where(ok, y, 0.0).sum(0) / den
    xc = np.where(ok, x - mx, 0.0)
    yc = np.where(ok, y - my, 0.0)
    df = np.maximum(cnt - 1.0, 1.0)
    vx = (xc * xc).sum(0) / df
    vy = (yc * yc).sum(0) / df
    cov = (xc * yc).sum(0) / df
    den_corr = np.sqrt(np.maximum(vx * vy, 0.0))
    good = (cnt >= _min_cnt(w)) & (vx > _EPS) & (vy > _EPS) & (den_corr > _EPS)
    corr = np.where(good, cov / np.where(good, den_corr, 1.0), np.nan)
    return _fin(corr)


def _varratio(t: int, C: Ctx, w: int = 20, q: int = 5):
    """重叠 q 日累计收益方差 / (q * 日收益方差) - 1。"""
    lo, W = _sl(C, t, w, C.R)
    k = W.shape[1] if W.ndim == 2 else C.k
    if W.shape[0] < int(q):
        return np.full(k, np.nan, dtype=np.float64)
    ok = np.isfinite(W)
    cnt = ok.sum(0).astype(np.float64)
    den = np.maximum(cnt, 1.0)
    mu = np.where(ok, W, 0.0).sum(0) / den
    c = np.where(ok, W - mu, 0.0)
    var1 = (c * c).sum(0) / np.maximum(cnt - 1.0, 1.0)

    nblock = W.shape[0] - int(q) + 1
    B = np.full((nblock, k), np.nan, dtype=np.float64)
    for i in range(nblock):
        block = W[i:i + int(q)]
        valid = np.isfinite(block).all(0)
        B[i] = np.where(valid, block.sum(0), np.nan)

    okb = np.isfinite(B)
    cntb = okb.sum(0).astype(np.float64)
    denb = np.maximum(cntb, 1.0)
    mub = np.where(okb, B, 0.0).sum(0) / denb
    cb = np.where(okb, B - mub, 0.0)
    varb = (cb * cb).sum(0) / np.maximum(cntb - 1.0, 1.0)
    need = max(3, int(np.ceil(0.6 * nblock)))
    good = (cnt >= _min_cnt(w)) & (cntb >= need) & (var1 > _EPS)
    out = varb / (float(q) * np.where(good, var1, 1.0)) - 1.0
    return _fin(np.where(good, out, np.nan))


def _ret_moments(t: int, C: Ctx, w: int):
    """窗口收益的偏度和超额峰度。"""
    lo, W = _sl(C, t, w, C.R)
    k = W.shape[1] if W.ndim == 2 else C.k
    if W.shape[0] == 0:
        z = np.full(k, np.nan, dtype=np.float64)
        return z, z.copy()
    ok = np.isfinite(W)
    cnt = ok.sum(0).astype(np.float64)
    den = np.maximum(cnt, 1.0)
    mu = np.where(ok, W, 0.0).sum(0) / den
    c = np.where(ok, W - mu, 0.0)
    n = np.maximum(cnt, 1.0)
    m2 = (c * c).sum(0) / n
    m3 = (c * c * c).sum(0) / n
    m4 = (c * c * c * c).sum(0) / n
    good = (cnt >= _min_cnt(w)) & (m2 > _EPS)
    skew = np.where(good, m3 / np.where(good, m2, 1.0) ** 1.5, np.nan)
    kurt = np.where(good, m4 / np.where(good, m2, 1.0) ** 2 - 3.0, np.nan)
    return _fin(skew), _fin(kurt)


def _maxret(t: int, C: Ctx, w: int):
    """窗口内最大单日收益。"""
    lo, W = _sl(C, t, w, C.R)
    k = W.shape[1] if W.ndim == 2 else C.k
    if W.shape[0] == 0:
        return np.full(k, np.nan, dtype=np.float64)
    ok = np.isfinite(W)
    cnt = ok.sum(0).astype(np.float64)
    mx = np.where(ok, W, -np.inf).max(0)
    return _fin(np.where(cnt >= _min_cnt(w), mx, np.nan))


# ------------------------------------------------------------------ #
# 各因子出口函数
# ------------------------------------------------------------------ #
def _f_beta(t, C, w):
    return _fin(_ols(t, C, w)["beta"])


def _f_alpha(t, C, w):
    return _fin(_ols(t, C, w)["alpha"])


def _f_ivol(t, C, w):
    return _fin(_ols(t, C, w)["ivol"])


def _f_idio_skew(t, C):
    return _fin(_ols(t, C, 60)["skew"])


def _f_idio_kurt(t, C):
    return _fin(_ols(t, C, 60)["kurt"])


def _f_r2(t, C, w):
    return _fin(_ols(t, C, w)["r2"])


def _f_mktcorr(t, C, w):
    return _fin(_ols(t, C, w)["corr"])


def _f_beta_stab(t, C):
    b60 = _ols(t, C, 60)["beta"]
    b120 = _ols(t, C, 120)["beta"]
    good = np.abs(b60) > _EPS
    return _fin(np.where(good, b120 / np.where(good, b60, 1.0), np.nan))


def _f_beta_up(t, C):
    return _fin(_cond_beta(t, C, 60, True))


def _f_beta_dn(t, C):
    return _fin(_cond_beta(t, C, 60, False))


def _f_trackerr(t, C):
    return _tracking_error(t, C, 60)


def _f_semidn(t, C):
    return _semivar(t, C, 120, True)


def _f_semiup(t, C):
    return _semivar(t, C, 120, False)


def _f_var5(t, C):
    lo, W = _sl(C, t, 60, C.R)
    ok = np.isfinite(W)
    return _fin(_quantile(W, ok, 0.05, _min_cnt(60)))


def _f_cvar5(t, C):
    lo, W = _sl(C, t, 60, C.R)
    ok = np.isfinite(W)
    return _fin(_tail_mean(W, ok, 0.05, _min_cnt(60)))


def _f_tailratio(t, C):
    lo, W = _sl(C, t, 60, C.R)
    ok = np.isfinite(W)
    a = np.abs(W)
    q95 = _quantile(a, ok, 0.95, _min_cnt(60))
    q05 = _quantile(a, ok, 0.05, _min_cnt(60))
    good = q05 > _EPS
    return _fin(np.where(good, q95 / np.where(good, q05, 1.0), np.nan))


def _f_ulcer(t, C, w):
    return _drawdown_stats(t, C, w)["ulcer"]


def _f_underwater(t, C):
    return _drawdown_stats(t, C, 60)["underwater"]


def _f_ddspeed(t, C):
    return _drawdown_stats(t, C, 60)["ddspeed"]


def _f_park(t, C, w):
    return _range_stats(t, C, w)["park"]


def _f_gk(t, C):
    return _range_stats(t, C, 60)["gk"]


def _f_rs(t, C):
    return _range_stats(t, C, 60)["rs"]


def _f_yz(t, C):
    return _range_stats(t, C, 60)["yz"]


def _f_bpv(t, C):
    return _bpv_stats(t, C, 60)["bpv"]


def _f_jump(t, C):
    return _bpv_stats(t, C, 60)["jump"]


def _f_volratio(t, C):
    v5 = _vol_std(t, C, 5)
    v60 = _vol_std(t, C, 60)
    good = v60 > _EPS
    return _fin(np.where(good, v5 / np.where(good, v60, 1.0), np.nan))


def _f_autocorr(t, C, w):
    return _autocorr(t, C, w)


def _f_varratio(t, C):
    return _varratio(t, C, 20, 5)


def _f_ret_skew(t, C):
    return _ret_moments(t, C, 120)[0]


def _f_ret_kurt(t, C):
    return _ret_moments(t, C, 120)[1]


# ------------------------------------------------------------------ #
FACTORS = {
    # beta / alpha / 特质风险
    "rk2_beta60": dict(func=lambda t, md, C: _f_beta(t, C, 60),
                       desc="过去60日对等权市场收益的掩码OLS市场模型beta", direction=-1),
    "rk2_beta120": dict(func=lambda t, md, C: _f_beta(t, C, 120),
                        desc="过去120日对等权市场收益的掩码OLS市场模型beta", direction=-1),
    "rk2_beta250": dict(func=lambda t, md, C: _f_beta(t, C, 250),
                        desc="过去250日对等权市场收益的掩码OLS市场模型beta", direction=-1),
    "rk2_beta_up60": dict(func=lambda t, md, C: _f_beta_up(t, C),
                          desc="过去60日仅用市场上涨日样本的掩码OLS beta", direction=-1),
    "rk2_beta_dn60": dict(func=lambda t, md, C: _f_beta_dn(t, C),
                          desc="过去60日仅用市场下跌日样本的掩码OLS beta", direction=-1),
    "rk2_beta_stab60": dict(func=lambda t, md, C: _f_beta_stab(t, C),
                            desc="120日beta/60日beta（beta稳定性）", direction=-1),
    "rk2_alpha60": dict(func=lambda t, md, C: _f_alpha(t, C, 60),
                        desc="过去60日市场模型截距年化（mean(resid)*252）", direction=1),
    "rk2_alpha120": dict(func=lambda t, md, C: _f_alpha(t, C, 120),
                         desc="过去120日市场模型截距年化（mean(resid)*252）", direction=1),
    "rk2_ivol120": dict(func=lambda t, md, C: _f_ivol(t, C, 120),
                        desc="过去120日市场模型残差标准差*sqrt(252)", direction=-1),
    "rk2_ivol250": dict(func=lambda t, md, C: _f_ivol(t, C, 250),
                        desc="过去250日市场模型残差标准差*sqrt(252)", direction=-1),
    "rk2_idio_skew60": dict(func=lambda t, md, C: _f_idio_skew(t, C),
                            desc="过去60日市场模型残差的偏度", direction=-1),
    "rk2_idio_kurt60": dict(func=lambda t, md, C: _f_idio_kurt(t, C),
                            desc="过去60日市场模型残差的超额峰度", direction=-1),
    "rk2_r2_60": dict(func=lambda t, md, C: _f_r2(t, C, 60),
                      desc="过去60日市场模型R^2（同步性）", direction=-1),
    "rk2_r2_250": dict(func=lambda t, md, C: _f_r2(t, C, 250),
                       desc="过去250日市场模型R^2（同步性）", direction=-1),
    "rk2_mktcorr60": dict(func=lambda t, md, C: _f_mktcorr(t, C, 60),
                          desc="过去60日个股收益与等权市场收益的相关系数", direction=-1),
    "rk2_mktcorr250": dict(func=lambda t, md, C: _f_mktcorr(t, C, 250),
                           desc="过去250日个股收益与等权市场收益的相关系数", direction=-1),
    "rk2_trackerr60": dict(func=lambda t, md, C: _f_trackerr(t, C),
                           desc="过去60日相对市场收益(R-mkt)的标准差", direction=-1),
    "rk2_varbeta60": dict(func=lambda t, md, C: _varbeta(t, C),
                          desc="过去60日内20日滚动波动率对市场同序列的OLS斜率", direction=-1),

    # 半方差 / 尾部风险
    "rk2_semidn120": dict(func=lambda t, md, C: _f_semidn(t, C),
                          desc="过去120日下行半方差*252（年化）", direction=-1),
    "rk2_semiup120": dict(func=lambda t, md, C: _f_semiup(t, C),
                          desc="过去120日上行半方差*252（年化）", direction=-1),
    "rk2_var5_60": dict(func=lambda t, md, C: _f_var5(t, C),
                        desc="过去60日收益5%分位（VaR，越负风险越大）", direction=1),
    "rk2_cvar5_60": dict(func=lambda t, md, C: _f_cvar5(t, C),
                         desc="过去60日最差5%日收益的均值（CVaR）", direction=1),
    "rk2_tailratio60": dict(func=lambda t, md, C: _f_tailratio(t, C),
                            desc="过去60日|收益|的P95/P5尾部比", direction=1),

    # 回撤 / 区间波动估计
    "rk2_ulcer60": dict(func=lambda t, md, C: _f_ulcer(t, C, 60),
                        desc="过去60日Ulcer指数 sqrt(mean(dd^2))，dd=1-C/cummax", direction=-1),
    "rk2_ulcer250": dict(func=lambda t, md, C: _f_ulcer(t, C, 250),
                         desc="过去250日Ulcer指数 sqrt(mean(dd^2))，dd=1-C/cummax", direction=-1),
    "rk2_underwater60": dict(func=lambda t, md, C: _f_underwater(t, C),
                             desc="过去60日低于窗口内历史峰值的天数占比", direction=-1),
    "rk2_ddspeed60": dict(func=lambda t, md, C: _f_ddspeed(t, C),
                          desc="过去60日最大回撤深度/到达谷底天数", direction=-1),
    "rk2_park60": dict(func=lambda t, md, C: _f_park(t, C, 60),
                       desc="过去60日Parkinson高低价波动率（日频）", direction=-1),
    "rk2_park120": dict(func=lambda t, md, C: _f_park(t, C, 120),
                        desc="过去120日Parkinson高低价波动率（日频）", direction=-1),
    "rk2_gk60": dict(func=lambda t, md, C: _f_gk(t, C),
                     desc="过去60日Garman-Klass波动率（日频）", direction=-1),
    "rk2_rs60": dict(func=lambda t, md, C: _f_rs(t, C),
                     desc="过去60日Rogers-Satchell波动率（日频）", direction=-1),
    "rk2_yz60": dict(func=lambda t, md, C: _f_yz(t, C),
                     desc="过去60日Yang-Zhang波动率（含隔夜跳空，日频）", direction=-1),
    "rk2_bpv60": dict(func=lambda t, md, C: _f_bpv(t, C),
                      desc="过去60日双幂次变差 pi/2*sum|r_t||r_{t-1}|", direction=-1),
    "rk2_jump60": dict(func=lambda t, md, C: _f_jump(t, C),
                       desc="过去60日跳跃占比(RV-BPV)/RV", direction=-1),
    "rk2_vol_ewma60": dict(func=lambda t, md, C: _ewma_vol(t, C, 60, 0.94),
                           desc="过去60日RiskMetrics EWMA波动率(lambda=0.94，年化)", direction=-1),
    "rk2_vol_slope60": dict(func=lambda t, md, C: _vol_slope(t, C),
                            desc="过去60日20日滚动波动率对时间的回归斜率/均值", direction=-1),
    "rk2_volratio5_60": dict(func=lambda t, md, C: _f_volratio(t, C),
                             desc="近5日收益波动/近60日收益波动", direction=-1),

    # 自相关 / 方差比 / 高阶矩
    "rk2_autocorr20": dict(func=lambda t, md, C: _f_autocorr(t, C, 20),
                           desc="过去20日日收益一阶自相关", direction=-1),
    "rk2_autocorr60": dict(func=lambda t, md, C: _f_autocorr(t, C, 60),
                           desc="过去60日日收益一阶自相关", direction=-1),
    "rk2_varratio5_20": dict(func=lambda t, md, C: _f_varratio(t, C),
                             desc="过去20日5日累计收益方差/(5*日收益方差)-1", direction=-1),
    "rk2_ret_skew120": dict(func=lambda t, md, C: _f_ret_skew(t, C),
                            desc="过去120日收益偏度", direction=-1),
    "rk2_ret_kurt120": dict(func=lambda t, md, C: _f_ret_kurt(t, C),
                            desc="过去120日收益超额峰度", direction=-1),
    "rk2_maxret_60": dict(func=lambda t, md, C: _maxret(t, C, 60),
                          desc="过去60日最大单日收益", direction=-1),
}






