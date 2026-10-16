# -*- coding: utf-8 -*-
"""量价关系与资金流因子族（前缀 fl_）。

数据前提（重要）
----------------
本数据集**没有成交量**，只有成交额 `amt`（单位：元）与成交笔数 `num_trades`，
因此 OBV / MFI / CMF / VWAP 等经典「量价」指标一律**用成交额 amt 替代 volume**。
同一价格水平下成交量与成交额只差一个价格因子，指标形态与单调性不变，是数学上
合理的近似（对资金流类因子而言按金额加权反而更贴近真实资金含义）；只有当窗口内
价格水平本身大幅变动时，两种口径的权重才会有差异。下文每个受影响因子都会标注。

筹码分布代理
------------
用窗口内每日成交额作权重，把当日成交额按**复权** [low, high] 对数价格区间均匀铺到
价格桶上，得到成本分布，再派生：获利盘比例（成本低于现价者，现价之上即套牢盘）、
加权平均成本偏离、成本集中度。全部是「矩阵 -> 分桶 -> bincount/cumsum」的向量化写法，
没有逐股票 Python 循环；窗口内有效交易日不足则返回 NaN。

因果性
------
所有因子只读 Ctx 中下标 <= t 的行（窗口一律 C.sl 切片、统计量用掩码求和），
`Ctx.truncated(t)` 的结果与全量上下文完全一致；无法计算的位置返回 NaN，绝不返回 inf。
"""
from __future__ import annotations

import numpy as np

from .ctx import Ctx, cs_zscore

# --------------------------------------------------------------------------- #
# 小工具

def _safe_div(num, den):
    """den > 0 时才计算，否则 NaN（并屏蔽除零告警，杜绝 inf）。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / np.asarray(den, dtype="float64")
    return np.where(np.isfinite(den) & (den > 0), out, np.nan)


def _ok_window(C: Ctx, t: int, lo: int) -> np.ndarray:
    """当日「可交易」掩码：已上市 且 成交额 > 0（未上市 / 停牌日 amt = 0）。"""
    return C.listed[lo:t + 1] & (C.A[lo:t + 1] > 0)


def _cnt(arr_ok) -> np.ndarray:
    """窗口内有效样本数 (k,)。"""
    return arr_ok.sum(0).astype("float64")


# --------------------------------------------------------------------------- #
# 1) 量价相关性 corr(r, Δlog amt)
def _pvcorr(t: int, C: Ctx, w: int) -> np.ndarray:
    """过去 w 日 corr(复权收益 r, Δlog 成交额)：>0 = 放量上涨（量价配合）。"""
    lo, _ = C.sl(t, w, C.C)
    R = C.R[lo:t + 1]
    A = C.A[lo:t + 1]
    okA = _ok_window(C, t, lo)
    with np.errstate(divide="ignore", invalid="ignore"):
        dA = np.diff(np.log(np.where(okA, A, 1.0)), axis=0)
    ok = okA[1:] & okA[:-1] & np.isfinite(dA) & np.isfinite(R[1:])
    n = _cnt(ok)
    x = np.where(ok, R[1:], 0.0)
    y = np.where(ok, dA, 0.0)
    mx = x.sum(0) / np.maximum(n, 1.0)
    my = y.sum(0) / np.maximum(n, 1.0)
    xc = np.where(ok, x - mx, 0.0)
    yc = np.where(ok, y - my, 0.0)
    num = (xc * yc).sum(0)
    den = np.sqrt((xc ** 2).sum(0) * (yc ** 2).sum(0))
    c = np.clip(_safe_div(num, den), -1.0, 1.0)      # 数值兜底，绝不越界
    return np.where(n >= max(10.0, 0.6 * w), c, np.nan)


# --------------------------------------------------------------------------- #
# 2) 成交额加权动量
def _amtmom(t: int, C: Ctx, w: int) -> np.ndarray:
    """过去 w 日 Σ(amt·r) / Σamt：按当日成交额加权的收益（资金加权动量）。"""
    lo, _ = C.sl(t, w, C.C)
    R = C.R[lo:t + 1]
    A = C.A[lo:t + 1]
    ok = _ok_window(C, t, lo) & np.isfinite(R)
    aw = np.where(ok, A, 0.0)
    num = (aw * np.where(ok, R, 0.0)).sum(0)
    den = aw.sum(0)
    out = _safe_div(num, den)
    return np.where(_cnt(ok) >= max(5.0, 0.6 * w), out, np.nan)


# --------------------------------------------------------------------------- #
# 3) 量价背离
def _pvdiv_rank(t: int, C: Ctx, w: int) -> np.ndarray:
    """价格 w 日分位 - 成交额 w 日分位：价处高位而量处低位 => 背离（看跌）。"""
    lo, _ = C.sl(t, w, C.C)
    W = C.C[lo:t + 1]
    A = C.A[lo:t + 1]
    okw = _ok_window(C, t, lo) & np.isfinite(W)
    px, ax = C.C[t], C.A[t]
    cp, ca = _cnt(okw), _cnt(okw)
    lep = (okw & (W <= px[None, :])).sum(0).astype("float64")
    lea = (okw & (A <= ax[None, :])).sum(0).astype("float64")
    rp = _safe_div(lep - 1.0, cp - 1.0)              # 自身计入，减 1 后落在 [0, 1]
    ra = _safe_div(lea - 1.0, ca - 1.0)
    out = rp - ra
    good = (cp >= max(20.0, 0.6 * w)) & np.isfinite(px) & (px > 0) & np.isfinite(ax) \
        & np.isfinite(out)
    return np.where(good, out, np.nan)


def _pvdiv_z(t: int, C: Ctx, w: int = 20) -> np.ndarray:
    """量价背离（连续版）：z(ret_{w}d) - z(Δlog 成交额_{w}d)。

    价格涨而成交额缩 => 取值为正 => 背离（看跌）。用幅度加权而非纯符号，
    避免横截面上大量并列取值。
    """
    if t < 2 * w:
        return np.full(C.k, np.nan)
    lo = t - 2 * w + 1
    A = C.A[lo:t + 1]
    okA = _ok_window(C, t, lo)
    m_now = _safe_div(np.where(okA[w:], A[w:], 0.0).sum(0), _cnt(okA[w:]))
    m_pv = _safe_div(np.where(okA[:w], A[:w], 0.0).sum(0), _cnt(okA[:w]))
    with np.errstate(divide="ignore", invalid="ignore"):
        dlog = np.log(m_now / m_pv)
    dlog = np.where(np.isfinite(dlog), dlog, np.nan)
    ret = _safe_div(C.C[t], C.C[t - w]) - 1.0
    out = cs_zscore(ret) - cs_zscore(dlog)
    return np.where(np.isfinite(out), out, np.nan)


# --------------------------------------------------------------------------- #
# 4) OBV（用成交额替代成交量）
def _obv_lvl(t: int, C: Ctx, w: int = 20) -> np.ndarray:
    """归一化 OBV 水平 = Σ sign(r)·amt / Σamt（净流入比例；成交额替代成交量）。"""
    lo, _ = C.sl(t, w, C.C)
    R = C.R[lo:t + 1]
    A = C.A[lo:t + 1]
    ok = _ok_window(C, t, lo) & np.isfinite(R)
    sgn = np.sign(np.where(ok, R, 0.0))
    den = np.where(ok, A, 0.0).sum(0)
    out = _safe_div((sgn * np.where(ok, A, 0.0)).sum(0), den)
    return np.where(_cnt(ok) >= max(5.0, 0.6 * w), out, np.nan)


def _obv_slope(t: int, C: Ctx, w: int = 20) -> np.ndarray:
    """OBV 的 w 日斜率 / 窗口日均成交额（成交额替代成交量）：净流入的速度。"""
    lo, _ = C.sl(t, w, C.C)
    R = C.R[lo:t + 1]
    A = C.A[lo:t + 1]
    ok = _ok_window(C, t, lo) & np.isfinite(R)
    flow = np.sign(np.where(ok, R, 0.0)) * np.where(ok, A, 0.0)
    Y = np.cumsum(flow, axis=0)                      # 窗口内累计 OBV（起点不参与斜率）
    n = _cnt(ok)
    nr = t - lo + 1                                 # data start may shorten window
    x = (np.arange(nr, dtype="float64") - 0.5 * (nr - 1))[:, None]
    okf = ok.astype("float64")
    mx = (okf * x).sum(0) / np.maximum(n, 1.0)
    my = (okf * Y).sum(0) / np.maximum(n, 1.0)
    dx = np.where(ok, x - mx, 0.0)
    dy = np.where(ok, Y - my, 0.0)
    sxx = (dx ** 2).sum(0)
    sxy = (dx * dy).sum(0)
    scale = np.where(ok, A, 0.0).sum(0) / np.maximum(n, 1.0)
    out = _safe_div(sxy, sxx * scale)
    return np.where(_cnt(ok) >= max(8.0, 0.6 * w), out, np.nan)


# --------------------------------------------------------------------------- #
# 5) MFI（用成交额替代成交量）
def _mfi(t: int, C: Ctx, n: int) -> np.ndarray:
    """MFI = 100 - 100/(1 + 正资金流/负资金流)，资金流 = 典型价 × 成交额。

    典型价取复权 (high+low+close)/3，涨跌方向取复权典型价的日变化。
    本数据集无成交量，故**用成交额替代成交量**。
    """
    w = n + 1
    lo, _ = C.sl(t, w, C.C)
    tp = (C.ADJ_H[lo:t + 1] + C.ADJ_L[lo:t + 1] + C.C[lo:t + 1]) / 3.0
    A = C.A[lo:t + 1]
    base = _ok_window(C, t, lo) & np.isfinite(tp) & (tp > 0)
    d = np.diff(tp, axis=0)
    ok = base[1:] & base[:-1] & np.isfinite(d)
    mf = np.where(ok, tp[1:] * A[1:], 0.0)           # 金额口径资金流（恒为正）
    pos = np.where(ok & (d > 0), mf, 0.0).sum(0)
    neg = np.where(ok & (d < 0), mf, 0.0).sum(0)
    ratio = _safe_div(pos, neg)
    with np.errstate(divide="ignore", invalid="ignore"):
        mfi = 100.0 - 100.0 / (1.0 + ratio)
    mfi = np.where(np.isfinite(mfi), mfi, np.nan)
    mfi = np.where(np.isfinite(neg) & (neg <= 0) & (pos > 0), 100.0, mfi)   # 仅流入
    mfi = np.where(np.isfinite(neg) & (neg <= 0) & (pos <= 0), np.nan, mfi)
    mfi = np.clip(mfi, 0.0, 100.0)
    return np.where(_cnt(ok) >= max(5.0, 0.6 * n), mfi, np.nan)


# --------------------------------------------------------------------------- #
# 6) CMF（用成交额替代成交量）
def _cmf(t: int, C: Ctx, w: int = 20) -> np.ndarray:
    """蔡金资金流 CMF = Σ(MLF)/Σamt，MLF = ((C-L)-(H-C))/(H-L) × amt。"""
    lo, _ = C.sl(t, w, C.C)
    hi = C.ADJ_H[lo:t + 1]
    lo_ = C.ADJ_L[lo:t + 1]
    cl = C.C[lo:t + 1]
    A = C.A[lo:t + 1]
    rng = hi - lo_
    ok = _ok_window(C, t, lo) & np.isfinite(rng) & (rng > 0) & np.isfinite(cl)
    ml = np.clip(_safe_div((cl - lo_) - (hi - cl), rng), -1.0, 1.0)
    ml = np.where(ok, ml, 0.0)
    num = (ml * np.where(ok, A, 0.0)).sum(0)
    den = np.where(ok, A, 0.0).sum(0)                # 分子分母同取有效日，口径一致
    out = _safe_div(num, den)
    return np.where(_cnt(ok) >= max(8.0, 0.6 * w), out, np.nan)


# --------------------------------------------------------------------------- #
# 7) 筹码分布代理：成交额按当日复权 [low, high] 对数价格区间均匀分桶
_CHIP_W = 250                                        # 窗口：过去 250 个交易日
_CHIP_STEP = 0.05                                    # 桶宽 5%（对数价 => 约 5% 相对价距）
_CHIP_UM = 1.5                                       # 网格以现价为中心覆盖 ±150% 对数区间
_CHIP_NB = int(round(2 * _CHIP_UM / _CHIP_STEP))     # 60 个桶，现价恰落在桶边界上
_CHIP_CACHE = {}


def _chip_calc(C: Ctx, t: int, w: int):
    """向量化构造成本分布并派生 3 个指标，返回 (获利盘, 成本偏离, 成本集中度)。

    每个交易日的成交额按**复权** [low, high] 在对数价上均匀铺开：某个价格桶拿到的权重
    正比于「该日价格区间落在该桶内的长度」。两端部分重叠的桶单独 bincount（点权重），
    中间整桶用差分数组做区间加权，全部向量化，无逐股票 Python 循环。
    """
    lo, _ = C.sl(t, w, C.C)
    H = C.ADJ_H[lo:t + 1]
    L = C.ADJ_L[lo:t + 1]
    A = C.A[lo:t + 1]
    px = C.C[t]
    ok = _ok_window(C, t, lo) & (H > 0) & (L > 0) & np.isfinite(H) & np.isfinite(L) \
        & np.isfinite(px) & (px > 0)
    pxs = np.where(ok, px[None, :], 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        u1 = np.log(np.where(ok, np.maximum(H, L), 1.0) / pxs)
        u0 = np.log(np.where(ok, np.minimum(H, L), 1.0) / pxs)
    e0 = np.clip(u0, -_CHIP_UM, _CHIP_UM)            # 截断到网格内
    e1 = np.clip(u1, -_CHIP_UM, _CHIP_UM)
    span = e1 - e0
    deg = span <= 0.0                                # 退化（一字板/全部在网格外）：全额入单桶
    span_b = np.maximum(span / _CHIP_STEP, 1e-9)      # 区间长度（桶为单位，仅作分母）
    s = (e0 + _CHIP_UM) / _CHIP_STEP                 # 起点桶坐标 (桶为单位)
    e = (e1 + _CHIP_UM) / _CHIP_STEP                 # 终点桶坐标
    b0 = np.clip(np.floor(s), 0, _CHIP_NB - 1).astype(np.int64)
    b1 = np.clip(np.floor(e), 0, _CHIP_NB - 1).astype(np.int64)
    k = C.k
    col = np.arange(k, dtype=np.int64)[None, :]
    fl0 = b0 * k + col                               # 左边缘桶
    fl1 = b1 * k + col                               # 右边缘桶
    ist = np.minimum(b0 + 1, _CHIP_NB) * k + col     # 内部桶区间 [b0+1, b1)
    ien = np.maximum(b1, np.minimum(b0 + 1, _CHIP_NB)) * k + col
    ovl = np.minimum(e, b0 + 1.0) - s                # 左边缘重叠长度
    ovr = np.where(b1 > b0, e - b1, 0.0)             # 右边缘重叠长度
    wl = np.where(deg, 0.0, A * ovl / span_b)        # 都是「桶坐标」口径，单位一致
    wr = np.where(deg, 0.0, A * ovr / span_b)
    wm = np.where(deg, 0.0, A / span_b)              # 内部每个完整桶拿到 1/区间长度
    ml = (_CHIP_NB + 1) * k
    P = np.bincount(fl0[ok], weights=wl[ok], minlength=ml)   # 点权重：左边缘桶
    P += np.bincount(fl1[ok], weights=wr[ok], minlength=ml)  # 右边缘桶
    dm = ok & deg
    P += np.bincount(fl0[dm], weights=A[dm], minlength=ml)   # 退化日：全额入单桶
    Q = np.bincount(ist[ok], weights=wm[ok], minlength=ml)   # 内部桶：差分数组区间加权
    Q -= np.bincount(ien[ok], weights=wm[ok], minlength=ml)
    hist = P.reshape(_CHIP_NB + 1, k)[:_CHIP_NB] \
        + np.cumsum(Q.reshape(_CHIP_NB + 1, k), axis=0)[:_CHIP_NB]     # 成本分布 (NB, k)
    hist = np.maximum(hist, 0.0)
    total = hist.sum(0)
    good = (total > 0) & (_cnt(ok) >= max(60.0, 0.5 * w))
    hist = np.where(good[None, :], hist, 0.0)        # 无效列置零，杜绝 exp/除法溢出告警
    tot = np.where(good, total, 1.0)
    # (a) 获利盘比例：成本低于现价的成交额占比（现价之上即套牢盘 = 1 - 该值）
    j0 = int(round(_CHIP_UM / _CHIP_STEP))           # 现价 u=0 所在桶下标
    prof = _safe_div(hist[:j0].sum(0), tot)
    # (b) 加权平均成本偏离：对数价格重心 => 几何平均成本（|重心| <= UM）
    ctr = (-_CHIP_UM + (np.arange(_CHIP_NB, dtype="float64") + 0.5) * _CHIP_STEP)[:, None]
    mlog = (hist * ctr).sum(0) / tot
    dev = np.exp(-mlog) - 1.0
    # (c) 成本集中度：归一化 Herfindahl（0 = 完全分散，1 = 完全集中）
    z = hist / tot
    hhi = (z * z).sum(0)
    conc = (hhi - 1.0 / _CHIP_NB) / (1.0 - 1.0 / _CHIP_NB)
    return (np.where(good, np.clip(prof, 0.0, 1.0), np.nan),
            np.where(good, np.clip(dev, -0.95, 100.0), np.nan),
            np.where(good, np.clip(conc, 0.0, 1.0), np.nan))


def _chip(t: int, C: Ctx, w: int = _CHIP_W):
    """带缓存的筹码分布：3 个因子共用同一次分桶。

    缓存键含 C.n（截断上下文不会误命中全量上下文的结果），另有成交额指纹兜底。
    """
    key = (id(C.md), C.k, C.n, int(t), int(w))
    sig = float(np.nansum(C.A[t]))
    hit = _CHIP_CACHE.get(key)
    if hit is not None and hit[0] == sig:
        return hit[1]
    res = _chip_calc(C, t, w)
    if len(_CHIP_CACHE) > 32:
        _CHIP_CACHE.clear()
    _CHIP_CACHE[key] = (sig, res)
    return res


# --------------------------------------------------------------------------- #
FACTORS = {
    # --- 1. 量价相关性 ---
    "fl_pvcorr20": dict(func=lambda t, md, C: _pvcorr(t, C, 20),
                        desc="20日量价相关 corr(r, dlog成交额)，>0=放量上涨", direction=-1),
    "fl_pvcorr60": dict(func=lambda t, md, C: _pvcorr(t, C, 60),
                        desc="60日量价相关 corr(r, dlog成交额)", direction=-1),
    # --- 2. 成交额加权动量 ---
    "fl_amtmom20": dict(func=lambda t, md, C: _amtmom(t, C, 20),
                        desc="20日成交额加权收益 sum(amt*r)/sum(amt)", direction=-1),
    "fl_amtmom60": dict(func=lambda t, md, C: _amtmom(t, C, 60),
                        desc="60日成交额加权收益 sum(amt*r)/sum(amt)", direction=1),
    # --- 3. 量价背离 ---
    "fl_pvdiv60": dict(func=lambda t, md, C: _pvdiv_rank(t, C, 60),
                       desc="量价背离：60日价格分位-成交额分位（价高量缩）", direction=-1),
    "fl_pvdiv_z20": dict(func=lambda t, md, C: _pvdiv_z(t, C, 20),
                         desc="量价背离(连续)：z(ret20d)-z(dlog成交额20d)", direction=-1),
    # --- 4. OBV（成交额替代成交量） ---
    "fl_obvlvl20": dict(func=lambda t, md, C: _obv_lvl(t, C, 20),
                        desc="20日归一化OBV=sum(sign(r)*amt)/sum(amt)，成交额替代成交量",
                        direction=1),
    "fl_obvslp20": dict(func=lambda t, md, C: _obv_slope(t, C, 20),
                        desc="OBV 20日斜率/日均成交额，成交额替代成交量", direction=1),
    # --- 5/6. MFI / CMF（成交额替代成交量） ---
    "fl_mfi14": dict(func=lambda t, md, C: _mfi(t, C, 14),
                     desc="14日MFI=100-100/(1+正/负资金流)，资金流=典型价*成交额",
                     direction=-1),
    "fl_mfi28": dict(func=lambda t, md, C: _mfi(t, C, 28),
                     desc="28日MFI，成交额替代成交量", direction=-1),
    "fl_cmf20": dict(func=lambda t, md, C: _cmf(t, C, 20),
                     desc="20日蔡金资金流CMF=sum(MLF)/sum(amt)，成交额替代成交量",
                     direction=1),
    # --- 7. 筹码分布代理（250日成交额按复权价格区间分桶） ---
    "fl_chip_prof250": dict(func=lambda t, md, C: _chip(t, C)[0],
                            desc="获利盘比例：250日成本低于现价的成交额占比（筹码分布代理）",
                            direction=1),
    "fl_chip_dev250": dict(func=lambda t, md, C: _chip(t, C)[1],
                           desc="现价/250日加权平均成本-1（筹码分布代理）", direction=1),
    "fl_chip_conc250": dict(func=lambda t, md, C: _chip(t, C)[2],
                            desc="250日成本集中度（归一化Herfindahl，筹码分布代理）",
                            direction=1),
}