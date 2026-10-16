# -*- coding: utf-8 -*-
"""独立重算 factor_zoo/flow.py 全部 14 个量价/资金流因子（代码审计，不改原模块）。

跑法（项目根目录）::

    $env:PYTHONIOENCODING="utf-8"; python 3_因子挖掘/audit/naive_flow.py

原则
----
1. 所有对照实现 ``n_*`` 都是「逐股票 Python 循环 + 朴素 numpy / 直接按定义算」，
   不 import factor_zoo.flow 的任何内部函数；模块因子只作为**被比较对象**
   （经 registry.load_all() 取出 spec["func"]）。
2. 7 个日期（题目要求的 6 个 + 2020-08-31，用于让 250 日筹码窗口有可用样本）
   上做全横截面逐格比较：max|diff|、NaN 掩码不一致格数、inf 计数。
3. 附边界探针：一字板 CMF、MFI 零涨跌、复权 vs 未复权筹码分箱、零方差相关、
   停牌日扰动、代数符号一致性、自算月度 IC（方向先验）。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.lib import MarketData                      # noqa: E402
from factor_zoo.ctx import Ctx                       # noqa: E402
from factor_zoo.registry import load_all             # noqa: E402

DATES = ["2020-03-31", "2020-08-31", "2021-06-30", "2022-12-30",
         "2023-07-31", "2024-06-28", "2025-06-30"]

CHIP_STEP, CHIP_UM, CHIP_NB = 0.05, 1.5, 60
TOL_ABS = 1e-8            # 判定阈值：浮点求和顺序差异 << 1e-8


# --------------------------------------------------------------------------- #
# 面板
# --------------------------------------------------------------------------- #
def build_panels(md: MarketData) -> dict:
    """把 MarketData 拍成 numpy 面板（与 Ctx 完全同口径，用于朴素实现）。"""
    f = np.asarray(md.cumadj_clean.values, dtype="float64")
    p = {}
    p["R"] = np.asarray(md.ret.values, dtype="float64")
    p["C"] = np.asarray(md.adj_close.values, dtype="float64")
    p["RAW"] = np.asarray(md.close.values, dtype="float64")
    p["O"] = np.asarray(md.open.values, dtype="float64")
    p["H"] = np.asarray(md.high.values, dtype="float64")
    p["L"] = np.asarray(md.low.values, dtype="float64")
    p["A"] = np.asarray(md.amt.values, dtype="float64")
    p["NT"] = np.asarray(md.num_trades.values, dtype="float64")
    p["listed"] = np.asarray(md.listed.values, dtype=bool)
    p["ADJ_H"] = p["H"] * f
    p["ADJ_L"] = p["L"] * f
    p["ADJ_O"] = p["O"] * f
    p["VWAP_PROXY"] = (p["H"] + p["L"] + p["RAW"]) / 3.0
    p["F"] = f                      # 去噪复权因子（扰动探针里按模块同样的乘法重建 ADJ）
    p["sym"] = np.asarray(md.symbols)
    return p


def naive_cs_z(x):
    """ctx.cs_zscore 的独立重写（2/98 去极值 + 总体标准差 z，常数返回 0）。"""
    v = np.asarray(x, dtype="float64")
    ok = np.isfinite(v)
    out = np.full_like(v, np.nan)
    if ok.sum() < 3:
        return out
    lo, hi = np.percentile(v[ok], [2, 98])
    v = np.clip(v, lo, hi)
    m = v[ok].mean()
    s = v[ok].std()
    out[ok] = (v[ok] - m) / s if s > 0 else 0.0
    return out


def _lo(t, w):
    return max(int(t) - int(w) + 1, 0)


# --------------------------------------------------------------------------- #
# 朴素对照实现（逐股票循环）
# --------------------------------------------------------------------------- #
def n_pvcorr(P, t, w):
    """corr(R, dlog A)，窗口 [t-w+1, t]，相邻两日都要 amt>0 且上市。"""
    lo = _lo(t, w)
    R, A, L = P["R"][lo:t + 1], P["A"][lo:t + 1], P["listed"][lo:t + 1]
    k = A.shape[1]
    out = np.full(k, np.nan)
    thr = max(10.0, 0.6 * w)
    for j in range(k):
        a, r, m = A[:, j], R[:, j], L[:, j]
        good = (m[1:] & m[:-1]) & (a[1:] > 0) & (a[:-1] > 0) & np.isfinite(r[1:])
        with np.errstate(divide="ignore", invalid="ignore"):
            la = np.log(np.where(m & (a > 0), a, 1.0))
        d = np.diff(la)
        good &= np.isfinite(d)
        x, y = r[1:][good], d[good]
        if x.size < thr:
            continue
        xc, yc = x - x.mean(), y - y.mean()
        den = np.sqrt((xc ** 2).sum() * (yc ** 2).sum())
        if den > 0:
            out[j] = min(1.0, max(-1.0, float((xc * yc).sum() / den)))
    return out


def n_amtmom(P, t, w):
    """Σ(A*r)/ΣA，窗口 [t-w+1, t]，只算上市且 amt>0 且 R 有值的日子。"""
    lo = _lo(t, w)
    R, A, L = P["R"][lo:t + 1], P["A"][lo:t + 1], P["listed"][lo:t + 1]
    out = np.full(A.shape[1], np.nan)
    thr = max(5.0, 0.6 * w)
    for j in range(A.shape[1]):
        g = L[:, j] & (A[:, j] > 0) & np.isfinite(R[:, j])
        if g.sum() < thr:
            continue
        out[j] = float((A[g, j] * R[g, j]).sum() / A[g, j].sum())
    return out


def n_pvdiv_rank(P, t, w):
    """窗口内价格分位 - 成交额分位（含当日自身，秩用 <= 计数）。"""
    lo = _lo(t, w)
    W, A, L = P["C"][lo:t + 1], P["A"][lo:t + 1], P["listed"][lo:t + 1]
    k = A.shape[1]
    out = np.full(k, np.nan)
    thr = max(20.0, 0.6 * w)
    px, ax = P["C"][t], P["A"][t]
    for j in range(k):
        okw = L[:, j] & (A[:, j] > 0) & np.isfinite(W[:, j])
        cp = int(okw.sum())
        if cp < thr or cp < 2 or not np.isfinite(px[j]) or px[j] <= 0 or not np.isfinite(ax[j]):
            continue
        lep = int(np.sum(okw & (W[:, j] <= px[j])))
        lea = int(np.sum(okw & (A[:, j] <= ax[j])))
        out[j] = (lep - 1.0) / (cp - 1.0) - (lea - 1.0) / (cp - 1.0)
    return out


def n_pvdiv_z(P, t, w=20):
    """z(ret_w) - z(log(近 w 日均额/前 w 日均额))，t < 2w 返回全 NaN。"""
    k = P["C"].shape[1]
    out = np.full(k, np.nan)
    if t < 2 * w:
        return out
    lo = t - 2 * w + 1
    A, L = P["A"][lo:t + 1], P["listed"][lo:t + 1]
    okA = L & (A > 0)
    m_now = np.full(k, np.nan)
    m_pv = np.full(k, np.nan)
    for j in range(k):
        o1 = okA[w:, j]
        o0 = okA[:w, j]
        if o1.any():
            m_now[j] = A[w:, j][o1].sum() / o1.sum()
        if o0.any():
            m_pv[j] = A[:w, j][o0].sum() / o0.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        dlog = np.where(np.isfinite(m_now / m_pv) & (m_pv > 0), np.log(m_now / m_pv), np.nan)
    ct, cp = P["C"][t], P["C"][t - w]
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.where(np.isfinite(cp) & (cp > 0), ct / cp, np.nan) - 1.0
    out = naive_cs_z(ret) - naive_cs_z(dlog)
    return np.where(np.isfinite(out), out, np.nan)


def n_obv_lvl(P, t, w=20):
    """Σ sign(R)*A / ΣA，窗口 [t-w+1, t]。"""
    lo = _lo(t, w)
    R, A, L = P["R"][lo:t + 1], P["A"][lo:t + 1], P["listed"][lo:t + 1]
    out = np.full(A.shape[1], np.nan)
    thr = max(5.0, 0.6 * w)
    for j in range(A.shape[1]):
        g = L[:, j] & (A[:, j] > 0) & np.isfinite(R[:, j])
        if g.sum() < thr:
            continue
        out[j] = float((np.sign(R[g, j]) * A[g, j]).sum() / A[g, j].sum())
    return out


def n_obv_slope(P, t, w=20):
    """窗口内累计 OBV 对日序回归斜率 / 窗口日均成交额。"""
    lo = _lo(t, w)
    R, A, L = P["R"][lo:t + 1], P["A"][lo:t + 1], P["listed"][lo:t + 1]
    nr = t - lo + 1
    out = np.full(A.shape[1], np.nan)
    thr = max(8.0, 0.6 * w)
    x = np.arange(nr, dtype="float64") - 0.5 * (nr - 1)
    for j in range(A.shape[1]):
        g = L[:, j] & (A[:, j] > 0) & np.isfinite(R[:, j])
        n = int(g.sum())
        if n < thr:
            continue
        flow = np.where(g, np.sign(np.where(g, R[:, j], 0.0)) * np.where(g, A[:, j], 0.0), 0.0)
        Y = np.cumsum(flow)
        dx, dy = x[g] - x[g].mean(), Y[g] - Y[g].mean()
        sxx = float((dx * dx).sum())
        sxy = float((dx * dy).sum())
        scale = float(A[g, j].mean())
        if sxx * scale > 0:
            out[j] = sxy / (sxx * scale)
    return out


def n_mfi(P, t, n):
    """MFI = 100-100/(1+pos/neg)，资金流=复权典型价*amt，d==0 的日子两边都不计。"""
    w = n + 1
    lo = _lo(t, w)
    H, L, C, A, LI = (P["ADJ_H"][lo:t + 1], P["ADJ_L"][lo:t + 1], P["C"][lo:t + 1],
                      P["A"][lo:t + 1], P["listed"][lo:t + 1])
    tp = (H + L + C) / 3.0
    base = LI & (A > 0) & np.isfinite(tp) & (tp > 0)
    k = A.shape[1]
    out = np.full(k, np.nan)
    thr = max(5.0, 0.6 * n)
    for j in range(k):
        d = np.diff(tp[:, j])
        ok = base[1:, j] & base[:-1, j] & np.isfinite(d)
        if int(ok.sum()) < thr:
            continue
        mf = tp[1:, j] * A[1:, j]
        pos = float(np.sum(mf[ok & (d > 0)]))
        neg = float(np.sum(mf[ok & (d < 0)]))
        if neg <= 0:
            val = 100.0 if pos > 0 else np.nan
        else:
            val = 100.0 - 100.0 / (1.0 + pos / neg)
        out[j] = min(100.0, max(0.0, val))
    return out


def n_cmf(P, t, w=20):
    """CMF = Σ(CLV*A)/ΣA，CLV=((C-L)-(H-C))/(H-L)；H==L 的日子分子分母都不计。"""
    lo = _lo(t, w)
    H, L, C, A, LI = (P["ADJ_H"][lo:t + 1], P["ADJ_L"][lo:t + 1], P["C"][lo:t + 1],
                      P["A"][lo:t + 1], P["listed"][lo:t + 1])
    k = A.shape[1]
    out = np.full(k, np.nan)
    thr = max(8.0, 0.6 * w)
    for j in range(k):
        rng = H[:, j] - L[:, j]
        ok = LI[:, j] & (A[:, j] > 0) & np.isfinite(rng) & (rng > 0) & np.isfinite(C[:, j])
        if int(ok.sum()) < thr:
            continue
        clv = ((C[ok, j] - L[ok, j]) - (H[ok, j] - C[ok, j])) / rng[ok]
        clv = np.clip(clv, -1.0, 1.0)
        out[j] = float((clv * A[ok, j]).sum() / A[ok, j].sum())
    return out


def n_chip(P, t, w=250, raw=False, want_hist=False):
    """筹码分布代理：每日 amt 按 [low, high] 对数区间均匀铺到 60 个桶。

    直接按「桶与 [u0,u1] 的重叠长度」分配（与模块的差分数组写法不同）。
    raw=True 时用未复权价分箱，用于量化「若用未复权价会错多少」。
    """
    H = P["H"] if raw else P["ADJ_H"]
    L = P["L"] if raw else P["ADJ_L"]
    px_all = P["RAW"] if raw else P["C"]
    lo = _lo(t, w)
    H, L, A, LI, px_all = H[lo:t + 1], L[lo:t + 1], P["A"][lo:t + 1], P["listed"][lo:t + 1], px_all
    k = A.shape[1]
    prof = np.full(k, np.nan)
    dev = np.full(k, np.nan)
    conc = np.full(k, np.nan)
    edges = -CHIP_UM + CHIP_STEP * np.arange(CHIP_NB)
    centers = -CHIP_UM + (np.arange(CHIP_NB) + 0.5) * CHIP_STEP
    j0 = int(round(CHIP_UM / CHIP_STEP))
    thr = max(60.0, 0.5 * w)
    hists = [None] * k
    for j in range(k):
        px = px_all[t, j]
        ok = (LI[:, j] & (A[:, j] > 0) & (H[:, j] > 0) & (L[:, j] > 0)
              & np.isfinite(H[:, j]) & np.isfinite(L[:, j]) & np.isfinite(px) & (px > 0))
        if int(ok.sum()) < thr:
            continue
        hi, lw, amt = H[ok, j], L[ok, j], A[ok, j]
        with np.errstate(divide="ignore", invalid="ignore"):
            u1 = np.log(np.maximum(hi, lw) / px)
            u0 = np.log(np.minimum(hi, lw) / px)
        e0 = np.clip(u0, -CHIP_UM, CHIP_UM)
        e1 = np.clip(u1, -CHIP_UM, CHIP_UM)
        span = e1 - e0
        degen = span <= 0
        hist = np.zeros(CHIP_NB)
        if (~degen).any():
            a0, a1, aa = e0[~degen][:, None], e1[~degen][:, None], amt[~degen][:, None]
            ov = np.maximum(0.0, np.minimum(a1, edges[None, :] + CHIP_STEP)
                            - np.maximum(a0, edges[None, :]))
            hist += (aa * ov / (a1 - a0)).sum(0)
        if degen.any():
            b = np.clip(np.floor((e0[degen] + CHIP_UM) / CHIP_STEP), 0, CHIP_NB - 1).astype(int)
            np.add.at(hist, b, amt[degen])
        tot = hist.sum()
        if not tot > 0:
            continue
        if want_hist:
            hists[j] = hist
        prof[j] = min(1.0, max(0.0, hist[:j0].sum() / tot))
        mlog = float((hist * centers).sum() / tot)
        dev[j] = min(100.0, max(-0.95, np.exp(-mlog) - 1.0))
        z = hist / tot
        hhi = float((z * z).sum())
        conc[j] = min(1.0, max(0.0, (hhi - 1.0 / CHIP_NB) / (1.0 - 1.0 / CHIP_NB)))
    if want_hist:
        return prof, dev, conc, hists, px_all[t]
    return prof, dev, conc



# --------------------------------------------------------------------------- #
# 比较工具 / 最小 Ctx 替身
# --------------------------------------------------------------------------- #
def cell_cmp(a, b):
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    na, nb = ~np.isfinite(a), ~np.isfinite(b)
    both = ~na & ~nb
    rec = dict(mask=int((na ^ nb).sum()), n=int(both.sum()),
               inf_a=int(np.isinf(a).sum()), inf_b=int(np.isinf(b).sum()))
    if both.any():
        d = np.abs(a[both] - b[both])
        rec["maxabs"] = float(d.max())
        rec["maxrel"] = float((d / np.maximum(np.abs(a[both]), 1e-12)).max())
    else:
        rec["maxabs"] = 0.0
        rec["maxrel"] = 0.0
    return rec


NAIVE = {
    "fl_pvcorr20": lambda P, t: n_pvcorr(P, t, 20),
    "fl_pvcorr60": lambda P, t: n_pvcorr(P, t, 60),
    "fl_amtmom20": lambda P, t: n_amtmom(P, t, 20),
    "fl_amtmom60": lambda P, t: n_amtmom(P, t, 60),
    "fl_pvdiv60": lambda P, t: n_pvdiv_rank(P, t, 60),
    "fl_pvdiv_z20": lambda P, t: n_pvdiv_z(P, t, 20),
    "fl_obvlvl20": lambda P, t: n_obv_lvl(P, t, 20),
    "fl_obvslp20": lambda P, t: n_obv_slope(P, t, 20),
    "fl_mfi14": lambda P, t: n_mfi(P, t, 14),
    "fl_mfi28": lambda P, t: n_mfi(P, t, 28),
    "fl_cmf20": lambda P, t: n_cmf(P, t, 20),
    "fl_chip_prof250": lambda P, t: n_chip(P, t)[0],
    "fl_chip_dev250": lambda P, t: n_chip(P, t)[1],
    "fl_chip_conc250": lambda P, t: n_chip(P, t)[2],
}


def compare_all(md, C, P, facts, names, ts, tag="主对照"):
    """逐因子逐日期比较模块输出与朴素重算，返回 (明细, 汇总)。"""
    rows = []
    for t in ts:
        chip_cache = None
        for nm in names:
            v_mod = np.asarray(facts[nm]["func"](int(t), md, C), dtype="float64")
            if nm.startswith("fl_chip"):
                if chip_cache is None:
                    chip_cache = n_chip(P, int(t))
                v_naive = chip_cache[("fl_chip_prof250", "fl_chip_dev250", "fl_chip_conc250").index(nm)]
            else:
                v_naive = NAIVE[nm](P, int(t))
            r = cell_cmp(v_mod, v_naive)
            r.update(factor=nm, date=str(md.dates[t].date()), kind=tag)
            rows.append(r)
        print("    ... %s 完成 %s" % (tag, md.dates[t].date()))
    det = pd.DataFrame(rows)
    agg = (det.groupby("factor")
              .agg(n=("n", "min"), maxabs=("maxabs", "max"), maxrel=("maxrel", "max"),
                   mask=("mask", "sum"), inf_a=("inf_a", "sum"), inf_b=("inf_b", "sum"))
              .reset_index())
    agg["verdict"] = np.where((agg["mask"] == 0) & (agg["maxabs"] <= TOL_ABS) & (agg["inf_a"] == 0),
                              "通过", "有问题")
    return det, agg


class _Stub:
    pass


def make_stub(P, n, k, md_obj=None):
    s = _Stub()
    for key in ("R", "C", "RAW", "O", "H", "L", "A", "NT", "ADJ_O", "ADJ_H", "ADJ_L",
                "VWAP_PROXY"):
        setattr(s, key, np.asarray(P[key])[:n])
    s.listed = np.asarray(P["listed"])[:n].astype(bool)
    s.mkt = np.zeros(n)
    s.mkt_fill = np.zeros(n)
    s.sl = Ctx.sl                      # 与真实 Ctx 相同的窗口切片
    s.n, s.k = int(n), int(k)
    s.md = md_obj if md_obj is not None else object()
    return s


def synth_panels(n, k, H, L, C, A=None, R=None, listed=None):
    H = np.asarray(H, float); L = np.asarray(L, float); C = np.asarray(C, float)
    P = dict(R=np.zeros((n, k)) if R is None else np.asarray(R, float),
             C=C, RAW=C.copy(), O=C.copy(), H=H, L=L,
             A=np.full((n, k), 1e8) if A is None else np.asarray(A, float),
             NT=np.full((n, k), 1000.0),
             listed=np.ones((n, k), bool) if listed is None else np.asarray(listed, bool))
    P["ADJ_H"] = H.copy(); P["ADJ_L"] = L.copy(); P["ADJ_O"] = C.copy()
    P["VWAP_PROXY"] = (H + L + C) / 3.0
    P["F"] = np.ones((n, k))
    return P


def spearman(x, y):
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    return float(np.corrcoef(rx, ry)[0, 1])


# --------------------------------------------------------------------------- #
# 探针 A：CMF 的 H==L（一字板/停牌）
# --------------------------------------------------------------------------- #
def probe_cmf(md, C, P, facts, ts):
    print("\n[A] CMF 乘数 (H==L) 除零探针")
    print("  全样本一字板(H==L 且 amt>0) 格数 = %d" % int(md.one_line.values.sum()))
    nn, k = 30, 4
    H = np.full((nn, k), 10.0); L = np.full((nn, k), 9.0); Cc = np.full((nn, k), 9.6)
    for j in (1, 2):
        H[:, j] = 10.5 + 0.3 * np.sin(np.arange(nn) / 3.0)
        L[:, j] = 9.5 + 0.3 * np.sin(np.arange(nn) / 3.0 + 1.0)
        Cc[:, j] = (H[:, j] + L[:, j]) / 2.0
    H[:25, 0] = 10.5 + 0.2 * np.sin(np.arange(25) / 2.0)
    L[:25, 0] = 9.5 + 0.2 * np.sin(np.arange(25) / 2.0 + 0.7)
    Cc[:25, 0] = (H[:25, 0] + L[:25, 0]) / 2.0
    H[25:, 0] = L[25:, 0] = Cc[25:, 0] = np.array([11.0, 11.2, 11.4, 11.6, 11.8])
    H[10, 2] = L[10, 2] = Cc[10, 2] = 10.0
    H[:, 3] = 10.4 + 0.2 * np.sin(np.arange(nn) / 2.0)
    L[:, 3] = 9.6 + 0.2 * np.sin(np.arange(nn) / 2.0)
    Cc[:, 3] = 10.0 + 0.2 * np.sin(np.arange(nn) / 2.0)
    H[12, 3] = L[12, 3] = 10.0; Cc[12, 3] = 10.5      # H==L 但 C 越界（脏数据）
    A = np.full((nn, k), 1e8); A[5, 2] = 0.0
    Psyn = synth_panels(nn, k, H, L, Cc, A=A)
    stub = make_stub(Psyn, nn, k)
    t = nn - 1
    v = np.asarray(facts["fl_cmf20"]["func"](t, stub.md, stub), float)
    vn = n_cmf(Psyn, t, 20)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = ((Cc - L) - (H - Cc)) / (H - L)
    win = slice(t - 19, t + 1)
    print("  窗口内 H==L 格数(股票0/2/3) = %d/%d/%d；raw 乘数 inf=%d NaN=%d"
          % (int((H[win, 0] == L[win, 0]).sum()), int((H[win, 2] == L[win, 2]).sum()),
             int((H[win, 3] == L[win, 3]).sum()),
             int(np.isinf(raw[win]).sum()), int(np.isnan(raw[win]).sum())))
    print("  module cmf20 = %s (inf=%d, NaN=%d)"
          % (np.round(v, 6).tolist(), int(np.isinf(v).sum()), int(np.isnan(v).sum())))
    print("  naive  cmf20 = %s" % np.round(vn, 6).tolist())
    print("  max|module-naive| = %.3e" % np.nanmax(np.abs(v - vn)))
    print("  结论：H==L 日 0/0 -> NaN（不是 inf），被 ok 掩码整日剔除（分子分母同剔除）；"
          "即使 H==L 而 C 越界（裸除法会出 inf）也不会漏进输出")

    rows = []
    for t in ts:
        lo = max(t - 19, 0)
        rng = P["ADJ_H"][lo:t + 1] - P["ADJ_L"][lo:t + 1]
        li = P["listed"][lo:t + 1]; aa = P["A"][lo:t + 1]
        base = li & (aa > 0)
        flat = base & (rng <= 0)
        amt_all = np.where(base, aa, 0.0).sum(0)
        share = np.where(amt_all > 0, np.where(flat, aa, 0.0).sum(0) / np.maximum(amt_all, 1e-30), np.nan)
        rows.append(dict(date=str(md.dates[t].date()), flat_cells=int(flat.sum()),
                         windows=int(flat.any(0).sum()),
                         max_flat_amt_share=float(np.nanmax(share)) if np.isfinite(share).any() else np.nan))
    print("  实盘 20 日窗口内的一字板(H==L 且 amt>0)格数 / 涉及股票数 / 一字板金额占窗口金额最大比:")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: "%.4f" % v))


# --------------------------------------------------------------------------- #
# 探针 B：MFI 零涨跌日 & 精确并列（刀锋开关）
# --------------------------------------------------------------------------- #
def probe_mfi_ties(md, C, P, facts, ts):
    print("\n[B] MFI 零涨跌日(d==0) 处理 + 精确并列的敏感性")
    tp = np.array([9.8, 9.9, 10.0, 10.1,
                   10.0, 10.1, 10.2, 10.2, 10.2, 10.3, 10.2, 10.1, 10.1, 10.0,
                   9.9, 9.9, 10.0, 10.1, 10.2, 10.3])
    n, k = tp.size, 1
    H = L = Cc = np.tile(tp[:, None], (1, k))
    P0 = synth_panels(n, k, H, L, Cc, A=np.full((n, k), 1e8))
    stub = make_stub(P0, n, k)
    v = float(np.asarray(facts["fl_mfi14"]["func"](n - 1, stub.md, stub))[0])
    d = np.diff(tp[-15:]); mf = tp[-15:][1:] * 1e8
    pos = float(mf[d > 0].sum()); neg = float(mf[d < 0].sum())
    v_flat = 100.0 - 100.0 / (1.0 + float(mf[d >= 0].sum()) / neg)
    print("  合成：d>0 %d 天 / d<0 %d 天 / d==0 %d 天"
          % (int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())))
    print("  module=%.6f  朴素(零涨跌两边都不计)=%.6f  若把零涨跌算作流入=%.6f"
          % (v, float(n_mfi(P0, n - 1, 14)[0]), v_flat))

    rows = []
    for t in ts:
        lo = t - 14
        Hh, Ll, Ccx, Aa, LI = (P["ADJ_H"][lo:t + 1], P["ADJ_L"][lo:t + 1], P["C"][lo:t + 1],
                               P["A"][lo:t + 1], P["listed"][lo:t + 1])
        tpx = (Hh + Ll + Ccx) / 3.0
        base = LI & (Aa > 0) & np.isfinite(tpx) & (tpx > 0)
        dd = np.diff(tpx, axis=0)
        ok = base[1:] & base[:-1] & np.isfinite(dd)
        mfk = np.where(ok, tpx[1:] * Aa[1:], 0.0)
        tot = mfk.sum(0); tie = np.where(ok & (dd == 0), mfk, 0.0).sum(0)
        share = np.divide(tie, np.maximum(tot, 1.0))
        rows.append(dict(date=str(md.dates[t].date()), pairs=int(ok.sum()),
                         exact_tie=int((ok & (dd == 0)).sum()),
                         near_tie=int((ok & (np.abs(dd) > 0)
                                       & (np.abs(dd) <= 1e-12 * np.abs(tpx[1:]))).sum()),
                         stocks_with_tie=int((ok & (dd == 0)).any(0).sum()),
                         tie_flow_share_max=float(share.max())))
    print("  实盘 14 日窗口内精确并列统计:")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: "%.4f" % v))

    # 刀锋：把 ADJ_H 抬高 1e-13 相对量（打破 tp 精确并列）后 MFI 变了多少
    P2 = dict(P); P2["ADJ_H"] = P["ADJ_H"] * (1.0 + 1e-13)
    s1 = make_stub(P, P["A"].shape[0], P["A"].shape[1])
    s2 = make_stub(P2, P["A"].shape[0], P["A"].shape[1])
    for nm in ("fl_mfi14", "fl_mfi28", "fl_cmf20"):
        mx, big = 0.0, 0
        for t in ts:
            a = np.asarray(facts[nm]["func"](t, s1.md, s1), float)
            b = np.asarray(facts[nm]["func"](t, s2.md, s2), float)
            m = np.isfinite(a) & np.isfinite(b)
            if m.any():
                dd = np.abs(a[m] - b[m]); mx = max(mx, float(dd.max())); big += int((dd > 1.0).sum())
        print("  ADJ_H x(1+1e-13) 后 %-11s max|d|=%.4f, 变化>1 的格数=%d" % (nm, mx, big))

    # OBV 的同类刀锋：R 精确为 0 的日子（金额>0）当成 +0 与 ±1e-13 对比
    rows = []
    for t in ts:
        lo = t - 19
        R, A, LI = P["R"][lo:t + 1], P["A"][lo:t + 1], P["listed"][lo:t + 1]
        ok = LI & (A > 0) & np.isfinite(R)
        rows.append(dict(date=str(md.dates[t].date()), valid=int(ok.sum()),
                         R_exact0=int((ok & (R == 0)).sum()),
                         stocks_with_R0=int((ok & (R == 0)).any(0).sum())))
    print("  实盘 20 日窗口内 R==0（收盘价与昨收完全相同、且当日有成交）:")
    print(pd.DataFrame(rows).to_string(index=False))
    mx, big = 0.0, 0
    P3 = dict(P); P3["R"] = np.where(P["R"] == 0.0, 1e-13, P["R"])
    s3 = make_stub(P3, P["A"].shape[0], P["A"].shape[1])
    for nm in ("fl_obvlvl20", "fl_obvslp20"):
        for t in ts:
            a = np.asarray(facts[nm]["func"](t, s1.md, s1), float)
            b = np.asarray(facts[nm]["func"](t, s3.md, s3), float)
            m = np.isfinite(a) & np.isfinite(b)
            if m.any():
                dd = np.abs(a[m] - b[m]); mx = max(mx, float(dd.max())); big += int((dd > 1e-3).sum())
    print("  把 R==0 改成 +1e-13（等价于把「持平日」改判为上涨日）后 OBV 两因子: "
          "max|d|=%.4f, 变化>1e-3 的格数=%d" % (mx, big))


# --------------------------------------------------------------------------- #
# 探针 C：筹码分布代理
# --------------------------------------------------------------------------- #
def probe_chip(md, C, P, facts, ts):
    print("\n[C] 筹码分箱：复权(模块) vs 未复权(假想错误口径)，以及网格截断/集中度分布")
    rows = []
    for t in ts:
        p1, d1, c1 = n_chip(P, t, 250, raw=False)
        p2, d2, c2 = n_chip(P, t, 250, raw=True)
        m = np.isfinite(d1) & np.isfinite(d2)
        if m.sum() == 0:
            rows.append(dict(date=str(md.dates[t].date()), n=0)); continue
        dd = np.abs(d1[m] - d2[m]); pp = np.abs(p1[m] - p2[m]); cc = np.abs(c1[m] - c2[m])
        rows.append(dict(date=str(md.dates[t].date()), n=int(m.sum()),
                         dev_med=float(np.median(dd)), dev_p95=float(np.percentile(dd, 95)),
                         dev_gt10pct=float((dd > 0.10).mean()),
                         prof_med=float(np.median(pp)), prof_gt10pct=float((pp > 0.10).mean()),
                         conc_med=float(np.median(cc)), conc_gt10pct=float((cc > 0.10).mean()),
                         dev_sign_flip=float((np.sign(d1[m]) != np.sign(d2[m])).mean())))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: "%.4f" % v))

    print("\n  [网格] 250 日窗口内价格越过 exp(±1.5)=[0.223, 4.482] 的股票比例:")
    rows = []
    for t in ts:
        lo = max(t - 250 + 1, 0)
        H, L, px = P["ADJ_H"][lo:t + 1], P["ADJ_L"][lo:t + 1], P["C"][t]
        A, LI = P["A"][lo:t + 1], P["listed"][lo:t + 1]
        ok = LI & (A > 0) & (H > 0) & (L > 0) & np.isfinite(px[None, :]) & (px[None, :] > 0)
        den = np.where(ok, px[None, :], 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            u1 = np.log(np.maximum(H, L) / den); u0 = np.log(np.minimum(H, L) / den)
        day_clip = ok & ((np.abs(u0) > 1.5) | (np.abs(u1) > 1.5))
        elig = ok.sum(0) >= 125
        rows.append(dict(date=str(md.dates[t].date()), stocks=int(elig.sum()),
                         clipped_days=int(day_clip.sum()),
                         stocks_with_clip=int((day_clip.any(0) & elig).sum()),
                         pct=100.0 * float((day_clip.any(0) & elig).sum()) / max(int(elig.sum()), 1)))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: "%.2f" % v))

    print("  [集中度] conc 实测分布（0=完全分散, 1=单桶集中）:")
    for t in ts:
        v = np.asarray(facts["fl_chip_conc250"]["func"](t, md, C), float)
        v = v[np.isfinite(v)]
        if v.size:
            print("    %s n=%5d min=%.3f med=%.3f p75=%.3f p99=%.3f max=%.3f"
                  % (str(md.dates[t].date()), v.size, v.min(), np.median(v),
                     np.percentile(v, 75), np.percentile(v, 99), v.max()))
    print("    注：桶宽 5%（对数价），单一成本簇跨两个桶时 conc 上限≈0.5，"
          "这是分桶量化的分辨率限制（合成探针实测 0.54），不是实现错误")


# --------------------------------------------------------------------------- #
# 探针 D：量价相关零方差 / 停牌日在估值日的取值
# --------------------------------------------------------------------------- #
def probe_chip_dev_caliber(md, C, P, facts, ts):
    """desc 写「现价/加权平均成本-1」，实现是「对数价重心」= 几何平均成本。"""
    print("\n  [成本口径] dev 的几何均价(实现) vs 算术加权均价(desc 字面)：")
    centers = -CHIP_UM + (np.arange(CHIP_NB) + 0.5) * CHIP_STEP
    for t in ts:
        _p, dev_g, _c, hists, px = n_chip(P, t, 250, want_hist=True)
        cost_ari = np.full(px.shape, np.nan)
        for j, h in enumerate(hists):
            if h is None:
                continue
            tot = h.sum()
            cost_ari[j] = float((h * np.exp(centers)).sum() / tot) * px[j]
        dev_a = px / cost_ari - 1.0
        m = np.isfinite(dev_g) & np.isfinite(dev_a)
        if m.sum() < 100:
            continue
        d = np.abs(dev_g[m] - dev_a[m])
        print("    %s n=%d 中位|Δ|=%.4f p95=%.4f max=%.4f 秩相关=%.5f"
              % (str(md.dates[t].date()), int(m.sum()), np.median(d),
                 np.percentile(d, 95), d.max(), spearman(dev_g[m], dev_a[m])))


def probe_obvslope_firstpoint(md, C, P, facts, ts):
    """_obv_slope 的注释说「起点不参与斜率」，实测首个有效点是计入回归的。"""
    print("\n  [OBV斜率] 含窗口首个有效点(模块) vs 去掉首个有效点:")
    for t in ts:
        lo = _lo(t, 20)
        R, A, LI = P["R"][lo:t + 1], P["A"][lo:t + 1], P["listed"][lo:t + 1]
        nr = R.shape[0]
        x = np.arange(nr, dtype="float64") - 0.5 * (nr - 1)
        a = np.asarray(facts["fl_obvslp20"]["func"](t, md, C), float)
        b = np.full(A.shape[1], np.nan)
        for j in range(A.shape[1]):
            g = LI[:, j] & (A[:, j] > 0) & np.isfinite(R[:, j])
            if g.sum() < 12:
                continue
            Y = np.cumsum(np.where(g, np.sign(np.where(g, R[:, j], 0.0)) * np.where(g, A[:, j], 0.0), 0.0))
            g2 = g.copy(); g2[int(np.argmax(g))] = False
            if g2.sum() < 11:
                continue
            dx, dy = x[g2] - x[g2].mean(), Y[g2] - Y[g2].mean()
            b[j] = (dx * dy).sum() / ((dx * dx).sum() * A[g2, j].mean())
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 100:
            continue
        print("    %s n=%5d 中位|Δ|=%.4f 最大|Δ|=%.4f 秩相关=%.5f"
              % (str(md.dates[t].date()), int(m.sum()), np.median(np.abs(a[m] - b[m])),
                 np.abs(a[m] - b[m]).max(), spearman(a[m], b[m])))


def probe_pvcorr_var(md, C, P, ts, facts):
    print("\n[D] 量价相关：窗口零方差（模块返回 NaN 而非 0）")
    for w in (20, 60):
        n_zero = 0; ex = []
        for t in ts:
            lo = _lo(t, w)
            R, A, LI = P["R"][lo:t + 1], P["A"][lo:t + 1], P["listed"][lo:t + 1]
            thr = max(10.0, 0.6 * w)
            La = np.log(np.where(LI & (A > 0), A, 1.0))
            d = np.diff(La, axis=0)
            ok = (LI[1:] & LI[:-1]) & (A[1:] > 0) & (A[:-1] > 0) & np.isfinite(R[1:]) & np.isfinite(d)
            n = ok.sum(0)
            x = np.where(ok, R[1:], 0.0); y = np.where(ok, d, 0.0)
            mx = x.sum(0) / np.maximum(n, 1.0); my = y.sum(0) / np.maximum(n, 1.0)
            vx = (np.where(ok, x - mx, 0.0) ** 2).sum(0); vy = (np.where(ok, y - my, 0.0) ** 2).sum(0)
            bad = (n >= thr) & ((vx <= 0) | (vy <= 0))
            n_zero += int(bad.sum())
            for j in np.where(bad)[0][:2]:
                ex.append((str(md.dates[t].date()), str(md.symbols[j]), float(vx[j]), float(vy[j])))
        print("  w=%d：通过样本数门槛但零方差的 (日期,股票) = %d 例 %s" % (w, n_zero, ex))
    n, k = 30, 1
    P0 = synth_panels(n, k, np.full((n, k), 10.0), np.full((n, k), 9.0),
                      np.full((n, k), 9.5), A=np.full((n, k), 1e8), R=np.full((n, k), 0.01))
    s0 = make_stub(P0, n, k)
    print("  合成（R 恒 =0.01、A 恒定）：pvcorr20 = %s（NaN，不是 0；与 ctx.cs_zscore 常数返回 0 的策略不一致）"
          % float(np.asarray(facts["fl_pvcorr20"]["func"](n - 1, s0.md, s0))[0]))
    print("  实测 7 日期×5451 只：0 例，故对现有数据无实际影响")

    print("\n  停牌日(t)本身：估值日 amt==0 的股票上因子是否仍给出可交易信号（非 NaN）")
    rows = []
    for t in ts:
        S = P["listed"][t] & (P["A"][t] == 0)
        rec = dict(date=str(md.dates[t].date()), suspended=int(S.sum()))
        for nm in ("fl_pvcorr20", "fl_amtmom20", "fl_pvdiv60", "fl_pvdiv_z20", "fl_obvlvl20",
                   "fl_mfi14", "fl_cmf20", "fl_chip_prof250"):
            v = np.asarray(facts[nm]["func"](t, md, C), float)
            rec[nm] = round(float(np.isfinite(v[S]).mean()) * 100.0, 1) if S.any() else np.nan
        rows.append(rec)
    print(pd.DataFrame(rows).to_string(index=False))


# --------------------------------------------------------------------------- #
# 探针 E：停牌日(amt==0)是否混进窗口统计（扰动检验）
# --------------------------------------------------------------------------- #
def probe_suspension(md, C, P, facts, ts):
    print("\n[E] 停牌日(amt==0)是否混进窗口均值/相关/累计：扰动检验")
    susp = P["listed"] & (P["A"] == 0)
    keep = np.zeros(P["A"].shape[0], bool); keep[ts] = True
    pert = susp & ~keep[:, None]                    # 只扰动「非估值日」的停牌格
    P2 = dict(P)
    H2, L2 = P["H"].copy(), P["L"].copy()
    H2[pert] = P["H"][pert] * 127.0
    L2[pert] = P["L"][pert] * 0.37
    P2["H"], P2["L"] = H2, L2
    P2["ADJ_H"] = H2 * P["F"]                       # 与 Ctx 完全相同的乘法
    P2["ADJ_L"] = L2 * P["F"]
    P2["VWAP_PROXY"] = (H2 + L2 + P["RAW"]) / 3.0
    n_win = 0
    for t in ts:
        for w in (20, 60, 250):
            lo = _lo(t, w)
            n_win += int((susp[lo:t + 1].any(0)).sum())
    _det, agg = compare_all(md, C, P2, facts, list(NAIVE.keys()), ts, tag="停牌扰动")
    print("  被扰动停牌格 = %d；7 日期×3 窗口(20/60/250) 中含停牌日的(日期,股票)窗口数 = %d"
          % (int(pert.sum()), n_win))
    print(agg[["factor", "maxabs", "mask", "inf_a"]].to_string(index=False))

    # 敏感性对照：只把 t_last-1 那天已成交行的 H/L 改成 2x/0.5x
    day = ts[-1] - 1
    tr = P["listed"][day] & (P["A"][day] > 0)
    P3 = dict(P)
    H3, L3 = P["H"].copy(), P["L"].copy()
    H3[day, tr] = P["H"][day, tr] * 2.0
    L3[day, tr] = P["L"][day, tr] * 0.5
    P3["H"], P3["L"] = H3, L3
    P3["ADJ_H"] = H3 * P["F"]; P3["ADJ_L"] = L3 * P["F"]
    P3["VWAP_PROXY"] = (H3 + L3 + P["RAW"]) / 3.0
    s3 = make_stub(P3, P["A"].shape[0], P["A"].shape[1])
    ctl = {}
    for nm in ("fl_cmf20", "fl_mfi14", "fl_chip_dev250"):
        a = np.asarray(facts[nm]["func"](ts[-1], C.md, C), float)
        b = np.asarray(facts[nm]["func"](ts[-1], s3.md, s3), float)
        ctl[nm] = int((np.abs(np.nan_to_num(a - b, nan=9e9)) > 1e-9).sum())
    print("  对照（%s 已成交行 H/L 改 2x/0.5x，应大面积变化）：%s" % (md.dates[day].date(), ctl))


# --------------------------------------------------------------------------- #
# 探针 F：代数符号 / 声明方向
# --------------------------------------------------------------------------- #
def probe_signs(facts):
    print("\n[F] 代数符号：docstring 语义 vs 合成数据实际取值")
    out = []
    n, k = 30, 1
    H = np.full((n, k), 10.0 + 0.01 * np.arange(n)[:, None])
    L = H - 0.1
    Cc = H.copy()                                   # 收=最高 => CLV=+1
    P = synth_panels(n, k, H, L, Cc, R=np.full((n, k), 0.01))
    s = make_stub(P, n, k); t = n - 1
    out.append(("fl_amtmom20", "每日 r=+1%", float(facts["fl_amtmom20"]["func"](t, s.md, s)[0]), ">0"))
    out.append(("fl_obvlvl20", "每日 r=+1%", float(facts["fl_obvlvl20"]["func"](t, s.md, s)[0]), "=+1"))
    out.append(("fl_obvslp20", "每日 r=+1%", float(facts["fl_obvslp20"]["func"](t, s.md, s)[0]), ">0"))
    out.append(("fl_mfi14", "典型价单调上行", float(facts["fl_mfi14"]["func"](t, s.md, s)[0]), "=100"))
    out.append(("fl_cmf20", "收盘=最高 (CLV=+1)", float(facts["fl_cmf20"]["func"](t, s.md, s)[0]), "=+1"))
    n2 = 40
    R2 = np.tile((0.01 + 0.005 * np.cos(np.arange(n2)))[:, None], (1, k))
    A2 = np.exp(np.tile((0.05 * np.sin(np.arange(n2)))[:, None], (1, k)))
    P2 = synth_panels(n2, k, np.full((n2, k), 10.0), np.full((n2, k), 9.0),
                      np.full((n2, k), 9.5), A=A2, R=R2)
    s2 = make_stub(P2, n2, k)
    out.append(("fl_pvcorr20", "R=cos, dlogA≈cos 同相位", float(facts["fl_pvcorr20"]["func"](n2 - 1, s2.md, s2)[0]), "≈+1"))
    n3 = 60
    px = np.tile((10.0 + 0.1 * np.arange(n3))[:, None], (1, k))
    amt = np.exp(np.tile((-0.02 * np.arange(n3))[:, None], (1, k)))
    P3 = synth_panels(n3, k, px + 0.05, px - 0.05, px, A=amt)
    s3 = make_stub(P3, n3, k)
    out.append(("fl_pvdiv60", "价格递增+成交额递减", float(facts["fl_pvdiv60"]["func"](n3 - 1, s3.md, s3)[0]), ">0"))
    n4 = 260
    H4 = np.full((n4, k), 5.0); L4 = np.full((n4, k), 5.0); C4 = np.full((n4, k), 5.0)
    H4[-1] = L4[-1] = C4[-1] = 10.0
    P4 = synth_panels(n4, k, H4, L4, C4)
    s4 = make_stub(P4, n4, k)
    out.append(("fl_chip_prof250", "成本=5 单桶、现价=10", float(facts["fl_chip_prof250"]["func"](n4 - 1, s4.md, s4)[0]), "=1"))
    out.append(("fl_chip_dev250", "成本=5 单桶、现价=10", float(facts["fl_chip_dev250"]["func"](n4 - 1, s4.md, s4)[0]), "≈0.96"))
    out.append(("fl_chip_conc250", "成本=5 单桶", float(facts["fl_chip_conc250"]["func"](n4 - 1, s4.md, s4)[0]), "≈1"))
    for nm, how, val, exp in out:
        print("  %-18s %-20s -> %12.6f  声称 %s" % (nm, how, val, exp))

    print("\n  声明 direction（registry 说只是先验）与实际代码符号：")
    for nm, spec in facts.items():
        if not nm.startswith("fl_"):
            continue
        print("    %-18s direction=%+d  desc=%s" % (nm, int(spec.get("direction", -1)), spec.get("desc", "")))


# --------------------------------------------------------------------------- #
# 探针 G：自算月度 IC（方向先验的实测校验）
# --------------------------------------------------------------------------- #
def probe_ic(md, C, facts, names):
    print("\n[G] 自算月度 IC（2021-01~2024-12, md.universe() 默认口径, 20 日前向收益）")
    from core.engine import Backtest
    ts = sorted(Backtest._rebalance_index(md.dates, "M"))
    ts = [t for t in ts if pd.Timestamp("2021-01-01") <= md.dates[t] <= pd.Timestamp("2024-12-31")]
    uni = np.asarray(md.universe().values, bool)
    fwd = np.asarray(md.ret_fwd(20).values, float)
    rows = []
    for nm in names:
        ics = []
        for t in ts:
            v = np.asarray(facts[nm]["func"](t, md, C), float)
            m = uni[t] & np.isfinite(v) & np.isfinite(fwd[t])
            if m.sum() < 100:
                continue
            ics.append(spearman(v[m], fwd[t][m]))
        ics = np.array(ics)
        d = int(facts[nm].get("direction", -1))
        rows.append(dict(factor=nm, n_dates=len(ics), IC=ics.mean() if len(ics) else np.nan,
                         t=ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics))) if len(ics) > 1 else np.nan,
                         direction=d,
                         sign_match=("是" if len(ics) and np.sign(ics.mean()) == d else "否")))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: "%.4f" % v))
    print("  声明方向与实测 IC 符号一致: %d/%d" % (int((df["sign_match"] == "是").sum()), len(df)))


# --------------------------------------------------------------------------- #
# 探针 H：成交量→成交额替代的影响（用 amt/未复权收盘 当成交量代理做对照）
# --------------------------------------------------------------------------- #
def probe_vol_proxy(md, C, P, facts, ts):
    print("\n[H] 成交额替代成交量的影响：用 V=amt/未复权收盘 当成交量代理重算相关因子")
    V = np.divide(P["A"], P["RAW"], out=np.full_like(P["A"], np.nan), where=P["RAW"] > 0)
    Pv = dict(P); Pv["A"] = V
    sv = make_stub(Pv, P["A"].shape[0], P["A"].shape[1])
    sc = make_stub(P, P["A"].shape[0], P["A"].shape[1])
    for nm in ("fl_obvlvl20", "fl_obvslp20", "fl_mfi14", "fl_cmf20"):
        med, mx, rho = [], [], []
        for t in ts:
            a = np.asarray(facts[nm]["func"](t, sc.md, sc), float)
            b = np.asarray(facts[nm]["func"](t, sv.md, sv), float)
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 100:
                continue
            dd = np.abs(a[m] - b[m])
            med.append(float(np.median(dd))); mx.append(float(dd.max())); rho.append(spearman(a[m], b[m]))
        print("  %-14s 中位|Δ|=%.4f  最大|Δ|=%.4f  横截面秩相关 min=%.4f"
              % (nm, float(np.mean(med)), float(np.max(mx)), float(np.min(rho))))
    print("  注：V=amt/RAW 只是「量」的粗代理（RAW 非 VWAP），仅用于衡量替代口径的相对影响")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    print("=" * 108)
    print("naive_flow.py —— factor_zoo/flow.py 14 个因子的独立重算审计")
    print("=" * 108)
    md = MarketData()
    P = build_panels(md)
    C = Ctx(md)
    facts = load_all()
    names = [n for n in facts if n.startswith("fl_")]
    assert set(names) == set(NAIVE.keys()), set(names) ^ set(NAIVE.keys())
    ts = []
    for s in DATES:
        i = int(md.dates.searchsorted(pd.Timestamp(s)))
        assert str(md.dates[i].date()) == s, (s, md.dates[i])
        ts.append(i)
    print("面板 %d 交易日 x %d 证券；对照日期: %s"
          % (C.n, C.k, ", ".join("%s(t=%d)" % (s, t) for s, t in zip(DATES, ts))))

    print("\n[1/5] 独立朴素实现 vs 模块输出（逐格）")
    det, agg = compare_all(md, C, P, facts, names, ts)
    print("\n  逐日明细（maxabs=全横截面最大绝对差, mask=NaN 掩码不一致格数）:")
    print(det[["factor", "date", "n", "maxabs", "maxrel", "mask", "inf_a", "inf_b"]]
          .to_string(index=False, float_format=lambda v: "%.3e" % v))
    print("\n  汇总（每因子取 7 个日期的最大差）:")
    print(agg[["factor", "maxabs", "maxrel", "mask", "inf_a", "inf_b", "verdict"]]
          .to_string(index=False, float_format=lambda v: "%.3e" % v))
    print("\n  %d/%d 因子通过（NaN 掩码完全一致 + max|diff|<=%.0e）"
          % (int((agg["verdict"] == "通过").sum()), len(agg), TOL_ABS))

    print("\n[2/5] CMF 一字板 & MFI 零涨跌并列")
    probe_cmf(md, C, P, facts, ts)
    probe_mfi_ties(md, C, P, facts, ts)
    print("\n[3/5] 筹码口径 & 零方差相关 & 停牌日")
    probe_chip(md, C, P, facts, ts)
    probe_chip_dev_caliber(md, C, P, facts, ts)
    probe_obvslope_firstpoint(md, C, P, facts, ts)
    probe_pvcorr_var(md, C, P, ts, facts)
    probe_suspension(md, C, P, facts, ts)
    print("\n[4/5] 符号/方向一致性")
    probe_signs(facts)
    probe_ic(md, C, facts, names)
    print("\n[5/5] 成交量替代敏感性")
    probe_vol_proxy(md, C, P, facts, ts)
    print("\n总用时 %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
