# -*- coding: utf-8 -*-
"""筹码分布代理因子（前缀 ch_，家族：量价关系与资金流）。

没有逐笔成交与真实持仓数据，本模块用过去 N 日成交额按当日典型价
``typ=(ADJ_H+ADJ_L+C)/3`` 分箱，近似成本分布。价格箱始终是相对**当日收盘价**
的固定对数网格：64 个箱覆盖 ``[0.5, 2.0] * C[t]``，其中 1.0 恰好是一条箱边界。
实现上把全部窗口元素一次性 ``searchsorted`` 到全局相对价格边界，再用
``bincount`` 做矩阵化累计权重、``cumsum`` 得到箱权重与累计权重；没有逐股票
Python 循环。窗口/衰减结果按 ``(t, w, half_life)`` 缓存在 ``Ctx.__dict__`` 上，
使用有界 LRU 避免全量缓存占用过多内存；截断上下文是独立对象，缓存天然隔离。
"""
from __future__ import annotations

import numpy as np

from .ctx import Ctx

# --------------------------------------------------------------------------- #
# 固定相对对数价格网格
_NB = 64
_LOW = 0.5
_HIGH = 2.0
_LOG_LO = float(np.log(_LOW))
_LOG_HI = float(np.log(_HIGH))
_STEP = (_LOG_HI - _LOG_LO) / float(_NB)
_EDGES = np.exp(_LOG_LO + np.arange(_NB + 1, dtype="float64") * _STEP)
_CENTER_LOG = _LOG_LO + (np.arange(_NB, dtype="float64") + 0.5) * _STEP
_CUR_BIN = _NB // 2                       # 1.0 所在边界右侧的箱下标
_CACHE_MAX = 40


def _fin(x):
    """把非有限值统一变成 NaN，保证出口不出现 ±inf。"""
    x = np.asarray(x, dtype="float64")
    return np.where(np.isfinite(x), x, np.nan)


def _safe_div(num, den):
    """仅在分母有限且严格为正时相除，否则返回 NaN。"""
    den = np.asarray(den, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.asarray(num, dtype="float64") / den
    return np.where(np.isfinite(den) & (den > 0.0), out, np.nan)


def _min_cnt(w: int) -> float:
    """窗口最少有效样本数：短窗 60%，长窗（>=120）另加 20 日下限。"""
    w = int(w)
    if w >= 120:
        return float(max(20, int(np.ceil(0.6 * w))))
    return float(max(3, int(np.ceil(0.6 * w))))


def _empty_state(k: int) -> dict:
    """无有效窗口时的安静占位状态。"""
    return dict(
        share=np.zeros((_NB, k), dtype="float32"),
        total=np.full(k, np.nan, dtype="float64"),
        profit=np.full(k, np.nan, dtype="float64"),
        flat=np.full(k, np.nan, dtype="float64"),
        above=np.full(k, np.nan, dtype="float64"),
        mean_cost=np.full(k, np.nan, dtype="float64"),
        std_cost=np.full(k, np.nan, dtype="float64"),
        winner_mean=np.full(k, np.nan, dtype="float64"),
        conc=np.full(k, np.nan, dtype="float64"),
        skew=np.full(k, np.nan, dtype="float64"),
        kurt=np.full(k, np.nan, dtype="float64"),
        entropy=np.full(k, np.nan, dtype="float64"),
        good=np.zeros(k, dtype=bool),
        cnt=np.zeros(k, dtype="float64"),
    )


def _chip_state_calc(C: Ctx, t: int, w: int, half_life):
    """计算一个 (t, w, 衰减) 的筹码分布及全部可复用派生量。"""
    k = int(C.k)
    t = int(t)
    w = int(w)
    if t < 0 or t >= int(C.n) or t + 1 < w:
        return _empty_state(k)

    lo = t - w + 1
    H = C.ADJ_H[lo:t + 1]
    L = C.ADJ_L[lo:t + 1]
    CL = C.C[lo:t + 1]
    A = C.A[lo:t + 1]
    px = C.C[t]

    px_ok = np.isfinite(px) & (px > 0.0)
    ok = (C.listed[lo:t + 1] & np.isfinite(A) & (A > 0.0)
          & np.isfinite(H) & (H > 0.0)
          & np.isfinite(L) & (L > 0.0)
          & np.isfinite(CL) & (CL > 0.0)
          & px_ok[None, :])
    typ = (H + L + CL) / 3.0
    ok = ok & np.isfinite(typ) & (typ > 0.0)

    # 相对现价的对数网格；超出 [0.5, 2.0] 的典型价压到两端箱。
    denom = np.where(px_ok, px, 1.0)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = typ / denom
    rel = np.clip(rel, _LOW, _HIGH)
    rel = np.where(ok, rel, 1.0)

    aw = np.where(ok, A, 0.0)
    if half_life is not None and float(half_life) > 0.0:
        age = np.arange(A.shape[0] - 1, -1, -1, dtype="float64")
        decay = np.power(0.5, age / float(half_life))
        aw = aw * decay[:, None]

    # 矩阵化分箱：固定相对边界 -> 箱下标 -> 列内成交额累计 -> 箱权重/累计权重。
    idx = np.searchsorted(_EDGES, rel, side="right") - 1
    np.clip(idx, 0, _NB - 1, out=idx)
    col = np.arange(k, dtype=np.int64)[None, :]
    flat = idx * k + col
    hist = np.bincount(flat.ravel(), weights=aw.ravel(),
                       minlength=_NB * k).reshape(_NB, k)
    cum = np.cumsum(hist, axis=0)             # (NB, k)，cum[j]=箱 0..j 的累计权重
    total = cum[-1]

    cnt = ok.sum(0).astype("float64")
    good = (total > 0.0) & (cnt >= _min_cnt(w)) & px_ok
    tot = np.where(good, total, 1.0)

    # 归一化箱权重，用于分布形状指标。
    z = np.where(good[None, :], hist / tot[None, :], 0.0)

    # 获利盘/平价/套牢盘：1.0 是箱边界，严格小于 1.0 才计入获利盘。
    profit_w = cum[_CUR_BIN - 1] if _CUR_BIN > 0 else np.zeros(k, dtype="float64")
    flat_w = (aw * (rel == 1.0)).sum(0)
    above_w = np.maximum(total - profit_w - flat_w, 0.0)
    profit = np.where(good, _safe_div(profit_w, total), np.nan)
    flat = np.where(good, _safe_div(flat_w, total), np.nan)
    above = np.where(good, _safe_div(above_w, total), np.nan)

    # 精确成交额加权成本（典型价）与标准差；分布形状则用箱中心计算。
    typ_safe = np.where(ok, typ, 0.0)
    mean_cost = _safe_div((aw * typ_safe).sum(0), total)
    centered = np.where(ok, typ - mean_cost[None, :], 0.0)
    var_cost = _safe_div((aw * centered * centered).sum(0), total)
    std_cost = np.sqrt(np.maximum(var_cost, 0.0))
    std_cost = np.where(good & (std_cost > 0.0), std_cost, np.nan)

    # 获利筹码的平均典型价。
    win = ok & (rel < 1.0)
    win_w = (aw * win).sum(0)
    win_num = (aw * typ_safe * win).sum(0)
    winner_mean = np.where(good, _safe_div(win_num, win_w), np.nan)

    # 箱中心对数价格的分布矩。
    center = _CENTER_LOG[:, None]
    mean_log = (z * center).sum(0)
    d = center - mean_log[None, :]
    var_log = (z * d * d).sum(0)
    std_log = np.sqrt(np.maximum(var_log, 0.0))
    den_skew = np.power(np.maximum(var_log, 0.0), 1.5)
    den_kurt = np.square(np.maximum(var_log, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        skew = np.where(den_skew > 1e-18,
                        (z * d * d * d).sum(0) / den_skew, np.nan)
        kurt = np.where(den_kurt > 1e-24,
                        (z * d * d * d * d).sum(0) / den_kurt - 3.0, np.nan)

    # 集中度（归一化 HHI）与 Shannon 熵（除以 log(NB)）。
    hhi = (z * z).sum(0)
    conc = (hhi - 1.0 / _NB) / (1.0 - 1.0 / _NB)
    conc = np.clip(conc, 0.0, 1.0)
    ent = -np.sum(np.where(z > 0.0, z * np.log(np.maximum(z, 1e-300)), 0.0),
                  axis=0) / float(np.log(_NB))

    good_f = good
    return dict(
        share=z.astype(np.float32),
        total=total,
        profit=np.where(good_f, profit, np.nan),
        flat=np.where(good_f, flat, np.nan),
        above=np.where(good_f, above, np.nan),
        mean_cost=np.where(good_f, mean_cost, np.nan),
        std_cost=np.where(good_f, std_cost, np.nan),
        winner_mean=np.where(good_f, winner_mean, np.nan),
        conc=np.where(good_f, conc, np.nan),
        skew=np.where(good_f, skew, np.nan),
        kurt=np.where(good_f, kurt, np.nan),
        entropy=np.where(good_f, ent, np.nan),
        good=good_f,
        cnt=cnt,
    )


def _chip_state(C: Ctx, t: int, w: int, half_life=None) -> dict:
    """按 (t, w, half_life) 取筹码中间结果；缓存放在 Ctx 实例上。"""
    key = (int(t), int(w), int(half_life) if half_life is not None else 0)
    cache = C.__dict__.setdefault("_chip_state_cache", {})
    if key in cache:
        state = cache.pop(key)
        cache[key] = state                     # dict 插入序实现 LRU
        return state
    state = _chip_state_calc(C, t, w, half_life)
    if len(cache) >= _CACHE_MAX:
        cache.pop(next(iter(cache)))
    cache[key] = state
    return state


def _mean_cost(C: Ctx, t: int, w: int, half_life=None):
    """轻量计算成交额加权平均典型价，不构造完整箱分布。"""
    t = int(t)
    w = int(w)
    k = int(C.k)
    if t < 0 or t >= int(C.n) or t + 1 < w:
        return np.full(k, np.nan, dtype="float64")
    lo = t - w + 1
    H = C.ADJ_H[lo:t + 1]
    L = C.ADJ_L[lo:t + 1]
    CL = C.C[lo:t + 1]
    A = C.A[lo:t + 1]
    px = C.C[t]
    px_ok = np.isfinite(px) & (px > 0.0)
    ok = (C.listed[lo:t + 1] & np.isfinite(A) & (A > 0.0)
          & np.isfinite(H) & (H > 0.0)
          & np.isfinite(L) & (L > 0.0)
          & np.isfinite(CL) & (CL > 0.0)
          & px_ok[None, :])
    typ = (H + L + CL) / 3.0
    ok = ok & np.isfinite(typ) & (typ > 0.0)
    aw = np.where(ok, A, 0.0)
    if half_life is not None and float(half_life) > 0.0:
        age = np.arange(A.shape[0] - 1, -1, -1, dtype="float64")
        aw = aw * np.power(0.5, age / float(half_life))[:, None]
    total = aw.sum(0)
    mean = _safe_div((aw * np.where(ok, typ, 0.0)).sum(0), total)
    cnt = ok.sum(0).astype("float64")
    return np.where((total > 0.0) & (cnt >= _min_cnt(w)) & px_ok, mean, np.nan)


def _peak_dist(C: Ctx, t: int, w: int, side: str):
    """最近筹码峰距离：above 返回 log(C/峰价)<0，support 返回 >0。"""
    st = _chip_state(C, t, w)
    z = st["share"].astype("float64")
    good = st["good"]
    nb, k = z.shape

    # 现价恰好是 1.0 边界；上方峰不应把“平价筹码”误判为上方峰。
    if side == "above":
        z = z.copy()
        flat = st["flat"]
        flat = np.where(np.isfinite(flat), flat, 0.0)
        z[_CUR_BIN, :] = np.maximum(z[_CUR_BIN, :] - flat, 0.0)

    peak = np.zeros((nb, k), dtype=bool)
    if nb >= 3:
        peak[1:-1] = (z[1:-1] >= z[:-2]) & (z[1:-1] > z[2:])
        peak[0] = z[0] > z[1]
        peak[-1] = z[-1] > z[-2]

    idx = np.full(k, -1, dtype=np.int64)
    if side == "above":
        start = _CUR_BIN
        region = z[start:, :]
        for j in range(start, nb):
            take = good & peak[j] & (idx < 0)
            idx[take] = j
        if region.shape[0] > 0:
            has = good & (region.sum(0) > 0.0) & (idx < 0)
            best = start + np.argmax(region, axis=0)
            idx[has] = best[has]
        valid = good & (idx >= 0) & (region.sum(0) > 0.0)
    else:
        end = _CUR_BIN
        region = z[:end, :]
        for j in range(end - 1, -1, -1):
            take = good & peak[j] & (idx < 0)
            idx[take] = j
        if region.shape[0] > 0:
            has = good & (region.sum(0) > 0.0) & (idx < 0)
            best = np.argmax(region, axis=0)
            idx[has] = best[has]
        valid = good & (idx >= 0) & (region.sum(0) > 0.0)

    dist = np.where(valid, -_CENTER_LOG[np.clip(idx, 0, nb - 1)], np.nan)
    return _fin(dist)


def _above_amt60(t: int, C: Ctx):
    """60 日上方筹码占比 / 其过去 250 日均值（每 10 个交易日抽样近似）。

    当前占比使用完整 60 日窗口；历史 250 日均值对时间轴每 10 个交易日抽样，
    每个样本日仍以其自身收盘价为阈值、使用完整 60 日窗口计算上方成交额占比。
    这是为了把最重因子的单次调用控制在 0.5s 内而做的明确近似。
    """
    t = int(t)
    k = int(C.k)
    if t < 0 or t >= int(C.n) or t + 1 < 60:
        return np.full(k, np.nan, dtype="float64")

    cache = C.__dict__.setdefault("_chip_above_cache", {})
    if t in cache:
        return cache[t]

    st = _chip_state(C, t, 60)
    current = st["above"]
    j0 = max(0, t - 249)
    js = list(range(j0, t + 1, 10))
    if not js or js[-1] != t:
        js.append(t)

    i_start = max(0, j0 - 59)
    H = C.ADJ_H[i_start:t + 1]
    L = C.ADJ_L[i_start:t + 1]
    CL = C.C[i_start:t + 1]
    A = C.A[i_start:t + 1]
    px = C.C[i_start:t + 1]
    px_ok = np.isfinite(px) & (px > 0.0)
    ok = (C.listed[i_start:t + 1] & np.isfinite(A) & (A > 0.0)
          & np.isfinite(H) & (H > 0.0)
          & np.isfinite(L) & (L > 0.0)
          & np.isfinite(CL) & (CL > 0.0)
          & px_ok)
    typ = (H + L + CL) / 3.0
    ok = ok & np.isfinite(typ) & (typ > 0.0)

    ratios = []
    for j in js:
        i0 = max(0, j - 59) - i_start
        i1 = j - i_start + 1
        ti = typ[i0:i1]
        oi = ok[i0:i1]
        ai = np.where(oi, A[i0:i1], 0.0)
        cj = C.C[j]
        cj_ok = np.isfinite(cj) & (cj > 0.0)
        base = oi & cj_ok[None, :]
        hit = base & (ti > cj[None, :])
        num = np.einsum("ij,ij->j", ai, hit, optimize=True)
        den = np.einsum("ij,ij->j", ai, base, optimize=True)
        cnt = base.sum(0)
        ratios.append(np.where((cnt >= 36.0) & (den > 0.0),
                               num / np.maximum(den, 1.0), np.nan))

    R = np.vstack(ratios)
    min_cnt = max(3, int(np.ceil(0.6 * R.shape[0])))
    mean_above = C.mean(R, np.isfinite(R), min_cnt)
    out = _fin(np.where(current > 0.0, _safe_div(current, mean_above), np.nan))
    if len(cache) >= 32:
        cache.pop(next(iter(cache)))
    cache[t] = out
    return out


# --------------------------------------------------------------------------- #
# FACTORS：全部 ch_* 筹码分布因子
FACTORS = {
    # 60 日基础筹码分布
    "ch_profit60": dict(
        func=lambda t, md, C: _fin(_chip_state(C, t, 60)["profit"]),
        desc="60日获利盘比例：成本低于现价的成交额权重占比", direction=1),
    "ch_cost_dev60": dict(
        func=lambda t, md, C: _fin(_safe_div(C.C[t], _chip_state(C, t, 60)["mean_cost"]) - 1.0),
        desc="现价/60日成交额加权平均成本-1", direction=1),
    "ch_conc60": dict(
        func=lambda t, md, C: _fin(_chip_state(C, t, 60)["conc"]),
        desc="60日成本分布归一化Herfindahl集中度", direction=1),
    "ch_conc_std60": dict(
        func=lambda t, md, C: _fin(_safe_div(_chip_state(C, t, 60)["std_cost"], C.C[t])),
        desc="60日成本分布标准差/现价", direction=-1),
    "ch_skew60": dict(
        func=lambda t, md, C: _fin(_chip_state(C, t, 60)["skew"]),
        desc="60日成本分布偏度（箱中心对数价格）", direction=-1),
    "ch_kurt60": dict(
        func=lambda t, md, C: _fin(_chip_state(C, t, 60)["kurt"]),
        desc="60日成本分布超额峰度（箱中心对数价格）", direction=-1),
    "ch_price_vs_cost60": dict(
        func=lambda t, md, C: _fin(_safe_div(C.C[t] - _chip_state(C, t, 60)["mean_cost"],
                                            _chip_state(C, t, 60)["std_cost"])),
        desc="(现价-60日加权平均成本)/成本分布标准差", direction=1),
    "ch_winner_amt60": dict(
        func=lambda t, md, C: _fin(_safe_div(_chip_state(C, t, 60)["winner_mean"], C.C[t]) - 1.0),
        desc="60日获利筹码加权平均价/现价-1", direction=1),
    "ch_dist_entropy60": dict(
        func=lambda t, md, C: _fin(_chip_state(C, t, 60)["entropy"]),
        desc="60日成本分布Shannon熵/log(64)", direction=-1),
    "ch_peak_dist60": dict(
        func=lambda t, md, C: _peak_dist(C, t, 60, "above"),
        desc="现价上方最近筹码峰距离log(C/峰价)，负值越接近0越强", direction=-1),
    "ch_support_dist60": dict(
        func=lambda t, md, C: _peak_dist(C, t, 60, "below"),
        desc="现价下方最近筹码峰距离log(C/峰价)，正值", direction=1),
    "ch_cost_slope60": dict(
        func=lambda t, md, C: _fin(_safe_div(_mean_cost(C, t, 60),
                                            _mean_cost(C, t - 60, 60)) - 1.0),
        desc="60日加权平均成本相对前60日的变化率", direction=1),
    "ch_above_amt60": dict(
        func=lambda t, md, C: _above_amt60(t, C),
        desc="60日上方筹码占比/其过去250日均值", direction=-1),

    # 120 日基础/衰减筹码分布
    "ch_profit120": dict(
        func=lambda t, md, C: _fin(_chip_state(C, t, 120)["profit"]),
        desc="120日获利盘比例：成本低于现价的成交额权重占比", direction=1),
    "ch_trapped120": dict(
        func=lambda t, md, C: _fin(_chip_state(C, t, 120)["above"]),
        desc="120日套牢盘比例：成本高于现价的成交额权重占比", direction=-1),
    "ch_cost_dev120": dict(
        func=lambda t, md, C: _fin(_safe_div(C.C[t], _chip_state(C, t, 120)["mean_cost"]) - 1.0),
        desc="现价/120日成交额加权平均成本-1", direction=1),
    "ch_conc120": dict(
        func=lambda t, md, C: _fin(_chip_state(C, t, 120)["conc"]),
        desc="120日成本分布归一化Herfindahl集中度", direction=1),
    "ch_skew120": dict(
        func=lambda t, md, C: _fin(_chip_state(C, t, 120)["skew"]),
        desc="120日成本分布偏度（箱中心对数价格）", direction=-1),
    "ch_decay_profit120": dict(
        func=lambda t, md, C: _fin(_chip_state(C, t, 120, 40)["profit"]),
        desc="120日指数衰减(half-life=40日)获利盘比例", direction=1),
    "ch_decay_dev120": dict(
        func=lambda t, md, C: _fin(_safe_div(C.C[t], _chip_state(C, t, 120, 40)["mean_cost"]) - 1.0),
        desc="现价/120日指数衰减(half-life=40日)加权平均成本-1", direction=1),
    "ch_profit_chg20": dict(
        func=lambda t, md, C: _fin(_chip_state(C, t, 120)["profit"]
                                   - _chip_state(C, t - 20, 120)["profit"]),
        desc="120日获利盘比例相对20个交易日前的变化", direction=1),

    # 250 日长周期衰减筹码分布
    "ch_cost_dev250d": dict(
        func=lambda t, md, C: _fin(_safe_div(C.C[t], _chip_state(C, t, 250, 60)["mean_cost"]) - 1.0),
        desc="现价/250日指数衰减(half-life=60日)加权平均成本-1", direction=1),
}