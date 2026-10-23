# -*- coding: utf-8 -*-
"""趋势 / 技术指标类因子（trend2，前缀 tr2_，共 48 个）。

覆盖规范 §5.6 要求的全部 tr2_* 因子（40 个），并在同族内小幅超配 8 个
（ema_dev60 / ma_slope120 / ma_slope250 / bb_width60 / adx14 / breakout120 /
rsi_slope14 / ema_align），合计 48 个。

口径要点
--------
1. 价格一律复权口径（C.C / C.ADJ_H / C.ADJ_L），本模块只输出数值，不构成任何
   买卖信号。涨跌停、一字板判断所需的未复权价在本模块不涉及。
2. 因果性：只读下标 <= t 的行。所有窗口由 C.sl(t, w, X) 截取，窗口起点只由 t
   决定，因此 Ctx.truncated(t) 与全量 Ctx 的返回逐位一致。
3. 掩码统计：窗口内有效样本数 < max(3, ceil(0.6*w))（w>=120 时下限 20）返回 NaN；
   所有出口统一过 _fin()，绝不返回 ±inf；全 NaN 列、空窗口安静返回 NaN。
4. Wilder 平滑 / EMA 不使用全历史递推，而是用「窗口内指数权重」闭式实现：
       w_i = alpha * (1-alpha)^age_i,  age_i = t - i（0 表示当日）
       alpha = 1/n（Wilder）或 2/(span+1)（EMA）
   再用窗口内有效权重质量归一化（除权/停牌造成的缺口不会被当作 0 值）。
   预热窗口取 6*span+10 级别，截断误差量级 (1-alpha)^(6*span) ~ 0.25%，
   且只依赖 <= t 的数据；MACD / TRIX 的多层 EMA 用同一权重矩阵做线性组合，
   不做逐行 Python 递推。
5. 数据没有成交量与逐笔方向，本模块不需要成交量；技术指标只做数值输出。

关键实现细节
------------
* 均线偏离 tr2_madevW = C[t]/MA_w - 1（MA 为窗口 [t-w+1, t] 的掩码均值）。
* 均线斜率 tr2_ma_slopeW = (MA_w[t] - MA_w[t-5]) / MA_w[t]（5 日斜率除以均值）。
* EMA 偏离 tr2_ema_dev20 = C[t]/EMA20 - 1，EMA 用指数权重窗口闭式。
* RSI(n)：涨跌幅的 Wilder 平滑（alpha=1/n，按有效权重归一），
  avg_loss==0 且 avg_gain>0 记 100，两者都为 0 记 50。
* Stochastic：%K = (C - min(L,14))/(max(H,14)-min(L,14))*100，
  %D 为最近 3 个 %K 的等权均值（要求 3 个都有效），%J = 3K - 2D。
* Williams %R = -100*(max(H,14) - C)/(max(H,14) - min(L,14))。
* CCI20：典型价 TP=(H+L+C)/3，(TP-MA20)/(0.015*MAD20)，MAD 为平均绝对偏差。
* MACD：DIF = EMA12-EMA26，HIST = DIF - EMA9(DIF)，两者都除以 C[t] 标准化。
  DIF 序列在窗口内逐行用同一指数权重矩阵求出（每行独立预热），EMA9 再对
  DIF 序列取指数权重均值。
* TRIX12：(EMA3(C)/EMA3(C)[t-1] - 1)*10000，三层 EMA 的权重用同一权重矩阵的
  三次幂（A^3）直接合成，避免逐行递推。
* DMI/ADX：TR / +DM / -DM 用 Wilder 14 指数权重在最近 14 个终点上求值，
  +DI/-DI = 100*DM_avg/TR_avg，DX = 100*|+DI - -DI|/(+DI + -DI)，
  ADX = DX 的 Wilder 14 平滑；tr2_adx_slope14 = (ADX[t]-ADX[t-5])/5。
* Aroon(25)：窗口 25 行内极值位置（并列取最近一次），
  Up/Down = 100*(位置索引)/(24)，Osc = Up - Down。
* 唐奇安位置 = (C - min(L,w))/(max(H,w) - min(L,w))；突破 = C/max(H,w) - 1。
* 布林带 %B = (C-MA60)/(2*Std60)；带宽 = 4*Std_w/MA_w。
* Keltner 位置 = (C-EMA20)/(2*ATR20)，ATR 为 Wilder 平滑的真实波幅。
* 对数价格回归斜率（年化，x 为窗口内行号）与 R^2 用掩码最小二乘。
* Hurst60：R/S 代理，H = log(R/S)/log(n)，n 为窗口内有效收益个数，截断到 [0,1]。
* 净路径比 = |C[t]-C[t-w]| / sum(|dC|)，w=60。
* 均线剪刀差 = (MA5 - MA250)/MA250。

已知局限
--------
* 停牌日复权价被数据源前向填充（R=0），因此停牌不会在价格序列里形成缺口，
  但会稀释 Wilder/EMA 的有效信息密度；本模块按 ok=isfinite 处理，不额外剔除。
* 指数权重闭式是「窗口内」的近似 Wilder/EMA：与全历史递推相比有 ~0.25% 量级的
  截断误差（窗口 6*span+10），新股样本越短误差越大，因此另有有效样本数下限。
* Aroon / Stochastic / 唐奇安等区间位置因子对窗口内极值敏感，除权已由复权价处理，
  但停牌与涨跌停造成的一字板仍可能扭曲极值位置。
* Hurst 用经典 R/S 估计，小样本有已知向下偏差，仅作趋势持续性的代理。
* CCI 的平均绝对偏差与布林带的标准差都使用掩码样本，样本不足返回 NaN。
* 所有因子保持原量纲（%K/%R/RSI/ADX 为 0~100 或 -100~0，斜率为年化对数收益），
  不做横截面标准化；direction 只是先验方向。
"""
from __future__ import annotations

import numpy as np

from .ctx import Ctx

_YEAR = 252.0
_EPS = 1e-12


# ================================================================== #
# 基础工具：掩码统计 / 安全运算 / 指数权重
# ================================================================== #
def _fin(x) -> np.ndarray:
    """把 ±inf / NaN 统一收敛为 NaN。"""
    x = np.asarray(x, dtype="float64")
    return np.where(np.isfinite(x), x, np.nan)


def _nan(C: Ctx) -> np.ndarray:
    return np.full(C.k, np.nan, dtype="float64")


def _min_cnt(w: int, frac: float = 0.6) -> int:
    """窗口最低有效样本数：短窗 max(3, ceil(0.6w))，长窗(>=120) max(20, ceil(0.6w))。"""
    w = int(w)
    floor = 3 if w < 120 else 20
    return max(int(floor), int(np.ceil(frac * w)))


def _safe_div(num, den) -> np.ndarray:
    """安全除法：分母 <=0 / 非有限、或分子非有限时返回 NaN。"""
    num = np.asarray(num, dtype="float64")
    den = np.asarray(den, dtype="float64")
    good = np.isfinite(num) & np.isfinite(den) & (den > 0.0)
    out = np.full(num.shape, np.nan, dtype="float64")
    np.divide(num, np.where(good, den, 1.0), out=out, where=good)
    return out


def _win(t: int, C: Ctx, X: np.ndarray, w: int):
    """窗口 [t-w+1, t] 的矩阵、有效掩码、有效计数。"""
    lo, W = C.sl(t, w, X)
    ok = np.isfinite(W)
    return W, ok, ok.sum(0).astype("float64")


def _ma(t: int, C: Ctx, X: np.ndarray, w: int) -> np.ndarray:
    """窗口 [t-w+1, t] 的掩码均值（有效数不足返回 NaN）。"""
    W, ok, _ = _win(t, C, X, w)
    return C.mean(W, ok, _min_cnt(w))


def _hi_lo(t: int, C: Ctx, w: int):
    """窗口内复权最高价 / 最低价（掩码），返回 (hi, lo, good)。"""
    _, HW = C.sl(t, w, C.ADJ_H)
    _, LW = C.sl(t, w, C.ADJ_L)
    okh = np.isfinite(HW)
    okl = np.isfinite(LW)
    minc = _min_cnt(w)
    hi = np.where(okh, HW, -np.inf).max(0)
    lo = np.where(okl, LW, np.inf).min(0)
    good = (okh.sum(0) >= minc) & (okl.sum(0) >= minc)
    return np.where(good, hi, np.nan), np.where(good, lo, np.nan), good


def _weight_mat(alpha: float, ends, m: int) -> np.ndarray:
    """指数权重矩阵 G[a, i] = alpha*(1-alpha)^(ends[a]-i)（i <= ends[a]，否则 0）。"""
    idx = np.arange(m, dtype="float64")
    ends = np.asarray(ends, dtype="int64").ravel()
    G = np.zeros((ends.size, m), dtype="float64")
    for a in range(ends.size):
        j = int(ends[a])
        G[a, : j + 1] = alpha * (1.0 - alpha) ** (j - idx[: j + 1])
    return G


def _wsum(W: np.ndarray, ok: np.ndarray, G: np.ndarray):
    """G:(L,m) 指数权重，W:(m,k)。返回按有效权重质量归一化的加权平均 (L,k) 与质量。"""
    k = W.shape[1]
    den = np.repeat(G.sum(1)[:, None], k, axis=1)
    allok = ok.all(0)
    if not bool(allok.all()):
        sub = ~allok
        den[:, sub] = G @ ok.astype("float64")[:, sub]
    num = G @ np.where(ok, W, 0.0)
    good = den > 0.0
    return np.where(good, num / np.where(good, den, 1.0), np.nan), den


def _ewma(t: int, C: Ctx, X: np.ndarray, span: int, warm: int = 0, minc: int = 0) -> np.ndarray:
    """以 t 为终点的 EMA（窗口内指数权重闭式，按有效权重质量归一化）。"""
    if warm <= 0:
        warm = int(np.ceil(6.0 * span + 10.0))
    m = int(min(t + 1, max(warm, span + 2)))
    _, W = C.sl(t, m, X)
    ok = np.isfinite(W)
    alpha = 2.0 / (span + 1.0)
    G = _weight_mat(alpha, [W.shape[0] - 1], W.shape[0])
    val, den = _wsum(W, ok, G)
    cnt = ok.sum(0).astype("float64")
    if minc <= 0:
        minc = _min_cnt(span)
    good = (cnt >= minc) & (den[0] >= 0.5)
    return np.where(good, val[0], np.nan)


# ================================================================== #
# 1) 均线 / EMA 偏离与排列
# ================================================================== #
def _ma_dev(t: int, C: Ctx, w: int) -> np.ndarray:
    """C[t] / MA_w - 1。"""
    ma = _ma(t, C, C.C, w)
    return _safe_div(C.C[t] - ma, ma)


def _ma_slope(t: int, C: Ctx, w: int) -> np.ndarray:
    """(MA_w[t] - MA_w[t-5]) / MA_w[t]，即均线的 5 日斜率除以均值。"""
    if t < 5:
        return _nan(C)
    ma1 = _ma(t, C, C.C, w)
    ma0 = _ma(t - 5, C, C.C, w)
    return _safe_div(ma1 - ma0, ma1)


def _ma_align4(t: int, C: Ctx) -> np.ndarray:
    """(MA5-MA20)/MA20 + (MA20-MA60)/MA60 + (MA60-MA120)/MA120。"""
    ma5 = _ma(t, C, C.C, 5)
    ma20 = _ma(t, C, C.C, 20)
    ma60 = _ma(t, C, C.C, 60)
    ma120 = _ma(t, C, C.C, 120)
    v = _safe_div(ma5 - ma20, ma20) + _safe_div(ma20 - ma60, ma60) + _safe_div(ma60 - ma120, ma120)
    ok = np.isfinite(ma5) & np.isfinite(ma20) & np.isfinite(ma60) & np.isfinite(ma120)
    return np.where(ok, v, np.nan)


def _ema_dev(t: int, C: Ctx, span: int) -> np.ndarray:
    """C[t] / EMA_span - 1。"""
    e = _ewma(t, C, C.C, span)
    return _safe_div(C.C[t] - e, e)


def _ema_align(t: int, C: Ctx) -> np.ndarray:
    """(EMA12-EMA26)/EMA26 + (EMA26-EMA60)/EMA60，EMA 版多头排列强度。"""
    e12 = _ewma(t, C, C.C, 12)
    e26 = _ewma(t, C, C.C, 26)
    e60 = _ewma(t, C, C.C, 60)
    v = _safe_div(e12 - e26, e26) + _safe_div(e26 - e60, e60)
    ok = np.isfinite(e12) & np.isfinite(e26) & np.isfinite(e60)
    return np.where(ok, v, np.nan)


def _ma_gap(t: int, C: Ctx) -> np.ndarray:
    """(MA5 - MA250) / MA250，长短均线剪刀差。"""
    ma5 = _ma(t, C, C.C, 5)
    ma250 = _ma(t, C, C.C, 250)
    return _safe_div(ma5 - ma250, ma250)


# ================================================================== #
# 2) 摆动指标：RSI / Stochastic / Williams %R / CCI
# ================================================================== #
def _rsi(t: int, C: Ctx, n: int) -> np.ndarray:
    """RSI(n)：涨跌幅 Wilder 平滑（alpha=1/n），0~100。"""
    if t < 1:
        return _nan(C)
    m = int(min(t + 1, 6 * n + 20))
    _, P = C.sl(t, m, C.C)
    if P.shape[0] < n + 1:
        return _nan(C)
    ok = np.isfinite(P[1:]) & np.isfinite(P[:-1])
    d = np.where(ok, P[1:] - P[:-1], 0.0)
    gain = np.where(ok & (d > 0.0), d, 0.0)
    loss = np.where(ok & (d < 0.0), -d, 0.0)
    alpha = 1.0 / float(n)
    age = np.arange(ok.shape[0] - 1, -1, -1, dtype="float64")
    wgt = alpha * (1.0 - alpha) ** age
    okf = ok.astype("float64")
    den = wgt @ okf
    cnt = okf.sum(0)
    good = (cnt >= max(n, _min_cnt(n))) & (den >= 0.5)
    dnum = np.where(den > 0.0, den, 1.0)
    ag = (wgt @ gain) / dnum
    al = (wgt @ loss) / dnum
    out = np.where(good, 50.0, np.nan)
    out = np.where(good & (al <= 0.0) & (ag > 0.0), 100.0, out)
    both = good & (al > 0.0)
    rs = ag / np.where(both, al, 1.0)
    out = np.where(both, 100.0 - 100.0 / (1.0 + rs), out)
    return _fin(out)


def _stoch(t: int, C: Ctx, n: int = 14):
    """Stochastic %K / %D / %J（14 日）。%D 为最近 3 个 %K 的等权均值。"""
    z = _nan(C)
    m = int(min(t + 1, n + 2))
    if m < n + 2:
        return z, z, z
    _, HW = C.sl(t, m, C.ADJ_H)
    _, LW = C.sl(t, m, C.ADJ_L)
    _, CW = C.sl(t, m, C.C)
    minc = _min_cnt(n)
    ks = []
    for s in range(3):
        Hs = HW[s:s + n]
        Ls = LW[s:s + n]
        okh = np.isfinite(Hs)
        okl = np.isfinite(Ls)
        hi = np.where(okh, Hs, -np.inf).max(0)
        lo = np.where(okl, Ls, np.inf).min(0)
        c = CW[s + n - 1]
        rng = hi - lo
        good = ((okh.sum(0) >= minc) & (okl.sum(0) >= minc)
                & np.isfinite(c) & (rng > 0.0))
        ks.append(np.where(good, 100.0 * (c - lo) / np.where(good, rng, 1.0), np.nan))
    KK = np.vstack(ks)
    okk = np.isfinite(KK)
    K = KK[-1]
    D = C.mean(KK, okk, 3)
    J = 3.0 * K - 2.0 * D
    return _fin(K), _fin(D), _fin(J)


def _willr(t: int, C: Ctx, n: int = 14) -> np.ndarray:
    """Williams %R = -100*(max(H,n) - C)/(max(H,n) - min(L,n))，-100~0。"""
    hi, lo, good = _hi_lo(t, C, n)
    px = C.C[t]
    ok = good & np.isfinite(px) & (hi > lo)
    return np.where(ok, -100.0 * (hi - px) / np.where(ok, hi - lo, 1.0), np.nan)


def _cci(t: int, C: Ctx, n: int = 20) -> np.ndarray:
    """CCI20 = (TP - MA20(TP)) / (0.015 * MAD20)，TP=(H+L+C)/3。"""
    m = int(min(t + 1, n))
    if m < n:
        return _nan(C)
    _, HW = C.sl(t, m, C.ADJ_H)
    _, LW = C.sl(t, m, C.ADJ_L)
    _, CW = C.sl(t, m, C.C)
    ok = np.isfinite(HW) & np.isfinite(LW) & np.isfinite(CW)
    TP = np.where(ok, (HW + LW + CW) / 3.0, 0.0)
    minc = _min_cnt(n)
    ma = C.mean(TP, ok, minc)
    mad = C.mean(np.abs(TP - ma[None, :]), ok, minc)
    good = ok[-1] & np.isfinite(ma) & np.isfinite(mad) & (mad > 0.0)
    return np.where(good, (TP[-1] - ma) / (0.015 * np.where(good, mad, 1.0)), np.nan)


# ================================================================== #
# 3) MACD / TRIX
# ================================================================== #
def _macd(t: int, C: Ctx):
    """MACD：返回 (DIF/C[t], HIST/C[t])，DIF=EMA12-EMA26，HIST=DIF-EMA9(DIF)。"""
    cache = C.__dict__.setdefault("_tr2_cache", {})
    key = ("macd", int(t))
    got = cache.get(key)
    if got is None:
        got = _macd_raw(t, C)
        cache[key] = got
    return got


def _macd_raw(t: int, C: Ctx):
    z = _nan(C)
    m = int(min(t + 1, 6 * 26 + 10 + 6 * 9 + 10))
    if m < 30:
        return z, z
    _, P = C.sl(t, m, C.C)
    ok = np.isfinite(P)
    L = int(min(6 * 9 + 10, P.shape[0]))
    ends = np.arange(P.shape[0] - L, P.shape[0])
    e12, den12 = _wsum(P, ok, _weight_mat(2.0 / 13.0, ends, P.shape[0]))
    e26, den26 = _wsum(P, ok, _weight_mat(2.0 / 27.0, ends, P.shape[0]))
    dif = e12 - e26
    okd = np.isfinite(dif)
    a9 = 2.0 / 10.0
    age = np.arange(L - 1, -1, -1, dtype="float64")
    w9 = a9 * (1.0 - a9) ** age
    den9 = w9 @ okd.astype("float64")
    ema9 = (w9 @ np.where(okd, dif, 0.0)) / np.where(den9 > 0.0, den9, 1.0)
    dif_last = dif[-1]
    hist = dif_last - ema9
    px = C.C[t]
    good = (np.isfinite(px) & (px > 0.0) & (den12[-1] >= 0.5) & (den26[-1] >= 0.5)
            & np.isfinite(dif_last) & (den9 > 0.0))
    d = np.where(good, px, 1.0)
    return np.where(good, dif_last / d, np.nan), np.where(good, hist / d, np.nan)


def _trix(t: int, C: Ctx, span: int = 12) -> np.ndarray:
    """TRIX = (EMA3(C)[t]/EMA3(C)[t-1] - 1) * 10000，三层 EMA 权重由 A^3 合成。"""
    m = int(min(t + 1, int(2.5 * (6 * span + 10))))
    if m < 3 * span:
        return _nan(C)
    _, P = C.sl(t, m, C.C)
    ok = np.isfinite(P)
    alpha = 2.0 / (span + 1.0)
    idx = np.arange(P.shape[0], dtype="float64")
    A = np.tril(alpha * (1.0 - alpha) ** (idx[:, None] - idx[None, :]))
    A3 = A @ A @ A
    v1 = A3[-1]
    v0 = A3[-2]
    okf = ok.astype("float64")
    d1 = v1 @ okf
    d0 = v0 @ okf
    pf = np.where(ok, P, 0.0)
    good = (d1 >= 0.9) & (d0 >= 0.9)
    e1 = np.where(good, (v1 @ pf) / np.where(d1 > 0.0, d1, 1.0), np.nan)
    e0 = np.where(good, (v0 @ pf) / np.where(d0 > 0.0, d0, 1.0), np.nan)
    ok2 = good & (e0 > 0.0) & np.isfinite(e1)
    return _fin(np.where(ok2, (e1 / np.where(ok2, e0, 1.0) - 1.0) * 10000.0, np.nan))


# ================================================================== #
# 4) 趋势强度：Aroon / DMI / ADX / ATR
# ================================================================== #
def _aroon(t: int, C: Ctx, n: int = 25):
    """Aroon(n)：窗口 n 行内极值位置（并列取最近一次），返回 (Up, Down)。"""
    z = _nan(C)
    m = int(min(t + 1, n))
    if m < n:
        return z, z
    _, HW = C.sl(t, m, C.ADJ_H)
    _, LW = C.sl(t, m, C.ADJ_L)
    okh = np.isfinite(HW)
    okl = np.isfinite(LW)
    minc = _min_cnt(n)
    hf = np.where(okh, HW, -np.inf)
    lf = np.where(okl, LW, np.inf)
    idx_h = (n - 1) - np.argmax(hf[::-1], axis=0)
    idx_l = (n - 1) - np.argmin(lf[::-1], axis=0)
    good = (okh.sum(0) >= minc) & (okl.sum(0) >= minc)
    up = np.where(good, 100.0 * idx_h / (n - 1.0), np.nan)
    dn = np.where(good, 100.0 * idx_l / (n - 1.0), np.nan)
    return _fin(up), _fin(dn)


def _tr_dm(t: int, C: Ctx, w: int):
    """窗口内真实波幅 TR / +DM / -DM（对齐到 t 的 w-1 个日间观测）。

    返回 (tr, pdm, ndm, ok)：tr 已把无效位置填 0（可直接参与矩阵运算），
    ok 为有效掩码；pdm / ndm 在无效位置为 0。无效位置指当日或前一日
    复权高低收任一项缺失。
    """
    _, HW = C.sl(t, w, C.ADJ_H)
    _, LW = C.sl(t, w, C.ADJ_L)
    _, CW = C.sl(t, w, C.C)
    if HW.shape[0] < 2:
        z = np.zeros((0, C.k), dtype="float64")
        return z, z, z, np.zeros((0, C.k), dtype=bool)
    h, l, pc = HW[1:], LW[1:], CW[:-1]
    hp, lp = HW[:-1], LW[:-1]
    tr = np.maximum(np.maximum(h - l, np.abs(h - pc)), np.abs(l - pc))
    up = h - hp
    dn = lp - l
    ok = np.isfinite(tr + up + dn)
    pdm = np.where(ok & (up > dn) & (up > 0.0), up, 0.0)
    ndm = np.where(ok & (dn > up) & (dn > 0.0), dn, 0.0)
    return np.where(ok, tr, 0.0), pdm, ndm, ok


def _atr(t: int, C: Ctx, n: int = 14) -> np.ndarray:
    """ATR(n)：真实波幅的 Wilder n 日平滑（指数权重窗口闭式）。"""
    cache = C.__dict__.setdefault("_tr2_cache", {})
    key = ("atr", int(t), int(n))
    got = cache.get(key)
    if got is None:
        got = _atr_raw(t, C, n)
        cache[key] = got
    return got


def _atr_raw(t: int, C: Ctx, n: int) -> np.ndarray:
    m = int(min(t + 1, 6 * n + 20))
    trf, _, _, ok = _tr_dm(t, C, m)
    if trf.shape[0] < n:
        return _nan(C)
    G = _weight_mat(1.0 / float(n), [trf.shape[0] - 1], trf.shape[0])
    val, den = _wsum(trf, ok, G)
    cnt = ok.sum(0).astype("float64")
    good = (cnt >= max(n, _min_cnt(n))) & (den[0] >= 0.5)
    return np.where(good, val[0], np.nan)


def _dmi(t: int, C: Ctx, n: int = 14):
    """DMI/ADX：返回 dict(dip, dim, adx, adx5)（adx5 为 t-5 的 ADX）。"""
    cache = C.__dict__.setdefault("_tr2_cache", {})
    key = ("dmi", int(t), int(n))
    got = cache.get(key)
    if got is None:
        got = _dmi_raw(t, C, n)
        cache[key] = got
    return got


def _dmi_raw(t: int, C: Ctx, n: int):
    """Wilder 口径的 +DI / -DI / ADX。

    * TR / +DM / -DM 按日间观测对齐；
    * 先在第 j0 行（= 最后 E 行的首行，E = 6n+5）用窗口内指数权重
      alpha*(1-alpha)^age（alpha = 1/n）闭式求出 Wilder n 日平均，
      预热长度 6n 行，截断误差 (1-1/n)^(6n) ~ 0.25%；
    * 再从 j0 逐日递推（每步只做一次 k 维向量更新），得到最近 E 行的
      Wilder 平均；DI 是两个同掩码平滑量的比值，有效权重质量自动约掉；
    * DX = 100*|+DI - -DI|/(+DI + -DI)；ADX = 最近 6n 个 DX 的 Wilder 加权
      平均（同样按有效权重归一），长记忆截断误差 ~0.2%；
    * adx5 为截至 t-5 的同一口径 ADX，用于 5 日斜率。
    """
    z = _nan(C)
    E = int(6 * n + 5)                  # 需要的 DX 个数（6n 预热 + 5 日斜率偏移）
    W0 = int(6 * n)                     # 首个 DI 行的预热长度
    m = int(min(t + 1, E + W0))
    trf, pdm, ndm, ok = _tr_dm(t, C, m)
    R = trf.shape[0]
    if R < E + 1:
        return dict(dip=z, dim=z, adx=z, adx5=z)
    j0 = R - E
    alpha = 1.0 / float(n)
    age = np.arange(j0, -1, -1, dtype="float64")
    wgt = alpha * (1.0 - alpha) ** age
    okt = ok[: j0 + 1].astype("float64")
    den = wgt @ okt
    den = np.where(den > 0.0, den, 1.0)
    wtr = (wgt @ trf[: j0 + 1]) / den
    wdp = (wgt @ pdm[: j0 + 1]) / den
    wdn = (wgt @ ndm[: j0 + 1]) / den
    WTR = np.empty((E, C.k), dtype="float64")
    WDP = np.empty((E, C.k), dtype="float64")
    WDN = np.empty((E, C.k), dtype="float64")
    for i in range(j0, R):
        if i > j0:
            oki = ok[i]
            wtr = np.where(oki, wtr + alpha * (trf[i] - wtr), wtr)
            wdp = np.where(oki, wdp + alpha * (pdm[i] - wdp), wdp)
            wdn = np.where(oki, wdn + alpha * (ndm[i] - wdn), wdn)
        j = i - j0
        WTR[j] = wtr
        WDP[j] = wdp
        WDN[j] = wdn
    cnt = ok.sum(0).astype("float64")
    good = (WTR > 0.0) & (cnt[None, :] >= max(n + 1, int(0.6 * 6 * n)))
    dip = np.where(good, 100.0 * WDP / np.where(good, WTR, 1.0), np.nan)
    dim = np.where(good, 100.0 * WDN / np.where(good, WTR, 1.0), np.nan)
    s_ = dip + dim
    fin = np.isfinite(dip) & np.isfinite(dim)
    pos = fin & (s_ > 0.0)
    dx = np.where(pos, 100.0 * np.abs(dip - dim) / np.where(pos, s_, 1.0),
                  np.where(fin, 0.0, np.nan))
    b = 1.0 / float(n)
    age2 = np.arange(6 * n - 1, -1, -1, dtype="float64")
    wgt2 = b * (1.0 - b) ** age2
    dx1 = dx[-6 * n:]                   # 截至 t 的 6n 个 DX
    dx5 = dx[:6 * n]                    # 截至 t-5 的 6n 个 DX
    ok1 = np.isfinite(dx1).astype("float64")
    ok5 = np.isfinite(dx5).astype("float64")
    den1 = wgt2 @ ok1
    den5 = wgt2 @ ok5
    adx = np.where(den1 > 0.0,
                   (wgt2 @ np.where(np.isfinite(dx1), dx1, 0.0)) / np.where(den1 > 0.0, den1, 1.0),
                   np.nan)
    adx5 = np.where(den5 > 0.0,
                    (wgt2 @ np.where(np.isfinite(dx5), dx5, 0.0)) / np.where(den5 > 0.0, den5, 1.0),
                    np.nan)
    return dict(dip=dip[-1], dim=dim[-1], adx=adx, adx5=adx5)


# ================================================================== #
# 5) 通道 / 突破 / 波动带
# ================================================================== #
def _donch(t: int, C: Ctx, w: int) -> np.ndarray:
    """唐奇安通道位置 (C - min(L,w)) / (max(H,w) - min(L,w))。"""
    hi, lo, good = _hi_lo(t, C, w)
    px = C.C[t]
    ok = good & np.isfinite(px) & (hi > lo)
    return np.where(ok, (px - lo) / np.where(ok, hi - lo, 1.0), np.nan)


def _breakout(t: int, C: Ctx, w: int) -> np.ndarray:
    """C[t] / max(H, w) - 1（<=0，越接近 0 越贴近窗口最高价）。"""
    hi, _, good = _hi_lo(t, C, w)
    px = C.C[t]
    ok = good & np.isfinite(px) & (hi > 0.0)
    return np.where(ok, px / np.where(ok, hi, 1.0) - 1.0, np.nan)


def _bb_pctb(t: int, C: Ctx, w: int) -> np.ndarray:
    """布林带 %B = (C - MA_w) / (2 * Std_w)。"""
    W, ok, _ = _win(t, C, C.C, w)
    minc = _min_cnt(w)
    ma = C.mean(W, ok, minc)
    sd = C.std(W, ok, minc)
    px = C.C[t]
    good = np.isfinite(ma) & np.isfinite(sd) & (sd > 0.0) & np.isfinite(px)
    return np.where(good, (px - ma) / (2.0 * np.where(good, sd, 1.0)), np.nan)


def _bb_width(t: int, C: Ctx, w: int) -> np.ndarray:
    """布林带宽度 = 4 * Std_w / MA_w。"""
    W, ok, _ = _win(t, C, C.C, w)
    minc = _min_cnt(w)
    ma = C.mean(W, ok, minc)
    sd = C.std(W, ok, minc)
    return _safe_div(4.0 * sd, ma)


def _keltner(t: int, C: Ctx, span: int = 20, n: int = 20) -> np.ndarray:
    """Keltner 位置 = (C - EMA20) / (2 * ATR20)。"""
    e = _ewma(t, C, C.C, span)
    atr = _atr(t, C, n)
    px = C.C[t]
    den = 2.0 * atr
    ok = np.isfinite(e) & np.isfinite(den) & (den > 0.0) & np.isfinite(px)
    return np.where(ok, (px - e) / np.where(ok, den, 1.0), np.nan)


# ================================================================== #
# 6) 趋势形态统计：对数回归 / Hurst / 净路径比
# ================================================================== #
def _linfit(t: int, C: Ctx, w: int):
    """对数复权价对窗口内行号的掩码最小二乘，返回 (年化斜率, R^2)。"""
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
    slope = np.where(good, _YEAR * sxy / np.where(good, sxx, 1.0), np.nan)
    use = good & (syy > 0.0)
    r2 = np.where(use, (sxy * sxy) / np.where(use, sxx * syy, 1.0), np.nan)
    return _fin(slope), _fin(np.clip(r2, 0.0, 1.0))


def _hurst(t: int, C: Ctx, w: int = 60) -> np.ndarray:
    """Hurst 指数代理（R/S 法）：H = log(R/S)/log(n)，截断到 [0,1]。"""
    _, P = C.sl(t, w + 1, C.C)
    if P.shape[0] < 3:
        return _nan(C)
    ok = np.isfinite(P[1:]) & np.isfinite(P[:-1])
    cnt = ok.sum(0).astype("float64")
    d = np.where(ok, P[1:] - P[:-1], 0.0)
    mu = d.sum(0) / np.maximum(cnt, 1.0)
    dev = np.where(ok, d - mu, 0.0)
    cs = np.cumsum(dev, axis=0)
    rng = cs.max(0) - cs.min(0)
    sd = np.sqrt((dev ** 2).sum(0) / np.maximum(cnt - 1.0, 1.0))
    good = (cnt >= _min_cnt(w)) & (sd > 0.0) & (rng > 0.0)
    ratio = np.where(good, rng / np.where(good, sd, 1.0), 1.0)
    h = np.where(good, np.log(ratio) / np.log(np.maximum(cnt, 2.0)), np.nan)
    return _fin(np.clip(h, 0.0, 1.0))


def _net_path(t: int, C: Ctx, w: int = 60) -> np.ndarray:
    """净路径比 = |C[t]-C[t-w]| / sum(|dC|)，0~1（越接近 1 趋势越顺）。"""
    _, P = C.sl(t, w + 1, C.C)
    if P.shape[0] < 2:
        return _nan(C)
    ok = np.isfinite(P[1:]) & np.isfinite(P[:-1])
    cnt = ok.sum(0).astype("float64")
    path = np.where(ok, np.abs(P[1:] - P[:-1]), 0.0).sum(0)
    net = np.abs(P[-1] - P[0])
    good = (cnt >= _min_cnt(w)) & (path > 0.0) & np.isfinite(net)
    return np.where(good, net / np.where(good, path, 1.0), np.nan)


# ================================================================== #
# 组合型辅助：由上面基础量派生
# ================================================================== #
def _rsi_slope(t: int, C: Ctx, n: int = 14) -> np.ndarray:
    """RSI 的 5 日斜率：(RSI_n[t] - RSI_n[t-5]) / 5。"""
    if t < 5:
        return _nan(C)
    r1 = _rsi(t, C, n)
    r0 = _rsi(t - 5, C, n)
    return _fin((r1 - r0) / 5.0)


def _aroon_osc(t: int, C: Ctx, n: int = 25) -> np.ndarray:
    """Aroon 振荡指标 = AroonUp - AroonDown。"""
    up, dn = _aroon(t, C, n)
    return _fin(up - dn)


def _adx_slope(t: int, C: Ctx, n: int = 14) -> np.ndarray:
    """ADX 的 5 日斜率：(ADX[t] - ADX[t-5]) / 5。"""
    d = _dmi(t, C, n)
    return _fin((d["adx"] - d["adx5"]) / 5.0)


def _atr_ratio(t: int, C: Ctx, n: int = 14) -> np.ndarray:
    """ATR_n / C[t]。"""
    return _safe_div(_atr(t, C, n), C.C[t])


# ================================================================== #
# 注册表
# ================================================================== #
FACTORS = {
    # ---- 均线偏离 / 排列（8） ----
    "tr2_madev5": dict(func=lambda t, md, C: _ma_dev(t, C, 5),
                       desc="复权收盘 / MA5 - 1（5 日均线偏离）", direction=-1),
    "tr2_madev10": dict(func=lambda t, md, C: _ma_dev(t, C, 10),
                        desc="复权收盘 / MA10 - 1（10 日均线偏离）", direction=-1),
    "tr2_madev120": dict(func=lambda t, md, C: _ma_dev(t, C, 120),
                         desc="复权收盘 / MA120 - 1（120 日均线偏离）", direction=-1),
    "tr2_madev250": dict(func=lambda t, md, C: _ma_dev(t, C, 250),
                         desc="复权收盘 / MA250 - 1（250 日均线偏离）", direction=-1),
    "tr2_ma_slope20": dict(func=lambda t, md, C: _ma_slope(t, C, 20),
                           desc="MA20 的 5 日斜率除以均值 (MA20[t]-MA20[t-5])/MA20[t]",
                           direction=1),
    "tr2_ma_slope60": dict(func=lambda t, md, C: _ma_slope(t, C, 60),
                           desc="MA60 的 5 日斜率除以均值 (MA60[t]-MA60[t-5])/MA60[t]",
                           direction=1),
    "tr2_ma_slope120": dict(func=lambda t, md, C: _ma_slope(t, C, 120),
                            desc="MA120 的 5 日斜率除以均值 (MA120[t]-MA120[t-5])/MA120[t]",
                            direction=1),
    "tr2_ma_slope250": dict(func=lambda t, md, C: _ma_slope(t, C, 250),
                            desc="MA250 的 5 日斜率除以均值 (MA250[t]-MA250[t-5])/MA250[t]",
                            direction=1),
    "tr2_ma_align4": dict(func=lambda t, md, C: _ma_align4(t, C),
                          desc="多头排列强度 (MA5-MA20)/MA20+(MA20-MA60)/MA60+(MA60-MA120)/MA120",
                          direction=1),
    "tr2_ma_gap_ratio": dict(func=lambda t, md, C: _ma_gap(t, C),
                             desc="长短均线剪刀差 (MA5-MA250)/MA250", direction=1),
    # ---- EMA（3） ----
    "tr2_ema_dev20": dict(func=lambda t, md, C: _ema_dev(t, C, 20),
                          desc="复权收盘 / EMA20 - 1（EMA 用指数权重窗口闭式）",
                          direction=-1),
    "tr2_ema_dev60": dict(func=lambda t, md, C: _ema_dev(t, C, 60),
                          desc="复权收盘 / EMA60 - 1", direction=-1),
    "tr2_ema_align": dict(func=lambda t, md, C: _ema_align(t, C),
                          desc="EMA 多头排列强度 (EMA12-EMA26)/EMA26+(EMA26-EMA60)/EMA60",
                          direction=1),
    # ---- RSI（4） ----
    "tr2_rsi6": dict(func=lambda t, md, C: _rsi(t, C, 6),
                     desc="RSI6（涨跌幅 Wilder 平滑，0~100）", direction=-1),
    "tr2_rsi14b": dict(func=lambda t, md, C: _rsi(t, C, 14),
                       desc="RSI14（Wilder 平滑，0~100）", direction=-1),
    "tr2_rsi24": dict(func=lambda t, md, C: _rsi(t, C, 24),
                      desc="RSI24（Wilder 平滑，0~100）", direction=-1),
    "tr2_rsi_slope14": dict(func=lambda t, md, C: _rsi_slope(t, C, 14),
                            desc="RSI14 的 5 日变化除以 5（RSI 趋势，单位：点/日）",
                            direction=1),
    # ---- Stochastic / Williams / CCI（6） ----
    "tr2_stoch_k14": dict(func=lambda t, md, C: _stoch(t, C, 14)[0],
                          desc="随机指标 %K14 = (C-min(L,14))/(max(H,14)-min(L,14))*100",
                          direction=-1),
    "tr2_stoch_d14": dict(func=lambda t, md, C: _stoch(t, C, 14)[1],
                          desc="随机指标 %D14 = 最近 3 个 %K 的等权均值", direction=-1),
    "tr2_stoch_j14": dict(func=lambda t, md, C: _stoch(t, C, 14)[2],
                          desc="随机指标 %J14 = 3*%K - 2*%D", direction=-1),
    "tr2_willr14": dict(func=lambda t, md, C: _willr(t, C, 14),
                        desc="Williams %R14 = -100*(max(H,14)-C)/(max(H,14)-min(L,14))",
                        direction=-1),
    "tr2_cci20": dict(func=lambda t, md, C: _cci(t, C, 20),
                      desc="CCI20 = (典型价-MA20(典型价))/(0.015*平均绝对偏差)",
                      direction=-1),
    # ---- MACD / TRIX（3） ----
    "tr2_macd_dif": dict(func=lambda t, md, C: _macd(t, C)[0],
                         desc="MACD DIF = (EMA12-EMA26)/C[t]（除以价格标准化）",
                         direction=1),
    "tr2_macd_hist": dict(func=lambda t, md, C: _macd(t, C)[1],
                          desc="MACD HIST = (DIF-EMA9(DIF))/C[t]（除以价格标准化）",
                          direction=1),
    "tr2_trix12": dict(func=lambda t, md, C: _trix(t, C, 12),
                       desc="TRIX12 = 三重 EMA12 的单日变化率 * 10000", direction=1),
    # ---- Aroon（3） ----
    "tr2_aroon_up25": dict(func=lambda t, md, C: _aroon(t, C, 25)[0],
                           desc="Aroon Up(25) = 100*(距窗口最高价的位置)/(24)，0~100",
                           direction=1),
    "tr2_aroon_dn25": dict(func=lambda t, md, C: _aroon(t, C, 25)[1],
                           desc="Aroon Down(25) = 100*(距窗口最低价的位置)/(24)，0~100",
                           direction=-1),
    "tr2_aroon_osc25": dict(func=lambda t, md, C: _aroon_osc(t, C, 25),
                            desc="Aroon Oscillator(25) = AroonUp - AroonDown，-100~100",
                            direction=1),
    # ---- DMI / ADX（4） ----
    "tr2_di_plus14": dict(func=lambda t, md, C: _dmi(t, C, 14)["dip"],
                          desc="DMI +DI14（Wilder 14 平滑的 +DM/TR，0~100）", direction=1),
    "tr2_di_minus14": dict(func=lambda t, md, C: _dmi(t, C, 14)["dim"],
                           desc="DMI -DI14（Wilder 14 平滑的 -DM/TR，0~100）", direction=-1),
    "tr2_adx14": dict(func=lambda t, md, C: _dmi(t, C, 14)["adx"],
                      desc="ADX14（DX 的 Wilder 14 平滑，0~100，越高趋势越强）",
                      direction=1),
    "tr2_adx_slope14": dict(func=lambda t, md, C: _adx_slope(t, C, 14),
                            desc="ADX14 的 5 日斜率 (ADX[t]-ADX[t-5])/5（点/日）",
                            direction=1),
    # ---- 通道 / 突破（5） ----
    "tr2_donch_pos120": dict(func=lambda t, md, C: _donch(t, C, 120),
                             desc="唐奇安 120 日通道位置 (C-min(L,120))/(max(H,120)-min(L,120))",
                             direction=1),
    "tr2_donch_pos250": dict(func=lambda t, md, C: _donch(t, C, 250),
                             desc="唐奇安 250 日通道位置", direction=1),
    "tr2_breakout20": dict(func=lambda t, md, C: _breakout(t, C, 20),
                           desc="C/max(H,20)-1（距 20 日最高价的距离，<=0）", direction=1),
    "tr2_breakout60": dict(func=lambda t, md, C: _breakout(t, C, 60),
                           desc="C/max(H,60)-1（距 60 日最高价的距离，<=0）", direction=1),
    "tr2_breakout120": dict(func=lambda t, md, C: _breakout(t, C, 120),
                            desc="C/max(H,120)-1（距 120 日最高价的距离，<=0）", direction=1),
    # ---- 波动带（4） ----
    "tr2_bb_pctb60": dict(func=lambda t, md, C: _bb_pctb(t, C, 60),
                          desc="布林带 %B60 = (C-MA60)/(2*Std60)", direction=-1),
    "tr2_bb_width20": dict(func=lambda t, md, C: _bb_width(t, C, 20),
                           desc="布林带宽度 4*Std20/MA20", direction=-1),
    "tr2_bb_width60": dict(func=lambda t, md, C: _bb_width(t, C, 60),
                           desc="布林带宽度 4*Std60/MA60", direction=-1),
    "tr2_keltner_pos20": dict(func=lambda t, md, C: _keltner(t, C, 20, 20),
                              desc="Keltner 位置 = (C-EMA20)/(2*ATR20)", direction=-1),
    "tr2_atr14": dict(func=lambda t, md, C: _atr_ratio(t, C, 14),
                      desc="ATR14 / C[t]（Wilder 真实波幅占价格比）", direction=-1),
    # ---- 回归 / 形态统计（6） ----
    "tr2_linslope20": dict(func=lambda t, md, C: _linfit(t, C, 20)[0],
                           desc="对数复权价 20 日回归斜率（年化）", direction=1),
    "tr2_linslope250": dict(func=lambda t, md, C: _linfit(t, C, 250)[0],
                            desc="对数复权价 250 日回归斜率（年化）", direction=1),
    "tr2_lin_r2_20": dict(func=lambda t, md, C: _linfit(t, C, 20)[1],
                          desc="20 日对数价回归 R^2（趋势直线程度，0~1）", direction=1),
    "tr2_lin_r2_250": dict(func=lambda t, md, C: _linfit(t, C, 250)[1],
                           desc="250 日对数价回归 R^2（0~1）", direction=1),
    "tr2_hurst60": dict(func=lambda t, md, C: _hurst(t, C, 60),
                        desc="Hurst 指数代理（60 日 R/S，0~1，>0.5 趋势持续）", direction=1),
    "tr2_net_path_ratio60": dict(func=lambda t, md, C: _net_path(t, C, 60),
                                 desc="净路径比 |C[t]-C[t-60]|/sum(|dC|)，0~1", direction=1),
}