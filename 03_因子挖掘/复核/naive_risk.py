# -*- coding: utf-8 -*-
"""factor_zoo/risk.py 独立复核脚本：朴素逐股重算 + 易错点探针 + 未来数据投毒。

运行（项目根目录下）：
    $env:PYTHONIOENCODING="utf-8"; python 3_因子挖掘/audit/naive_risk.py

设计原则
--------
1. 被审计模块只通过注册表 FACTORS[name]["func"] 调用，不借用其任何内部函数；
2. 朴素实现完全不 import factor_zoo.risk，逐股 for 循环 + 教科书公式手写；
3. 两边输入口径完全一致（md.ret / md.adj_close / md.ret_ok.mean(1)），
   因此差异只可能来自统计实现本身；
4. 另附：端点敏感性（证明本对比有能力发现 off-by-one）、未来数据投毒（含漏检对照）、
   门槛诊断、自由度/偏峰口径探针、合成序列语义探针。
"""
from __future__ import annotations

import copy
import gc
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.lib import MarketData               # noqa: E402
from factor_zoo.ctx import Ctx                # noqa: E402
from factor_zoo.risk import FACTORS as RISK   # noqa: E402  （被审计的参考实现）

AUDIT = os.path.dirname(os.path.abspath(__file__))
TARGET_DATES = ["2020-03-31", "2021-06-30", "2022-12-30",
                "2023-07-31", "2024-06-28", "2025-06-30"]
NAMES = list(RISK.keys())

# 每个因子的窗口 / 门槛参数（用于诊断打印）
PARAMS = {
    "rk_vol60":      dict(w=60,  floor=20, frac=0.6),
    "rk_vol120":     dict(w=120, floor=20, frac=0.6),
    "rk_vol250":     dict(w=250, floor=20, frac=0.6),
    "rk_ivol60":     dict(w=60,  floor=40, frac=0.6),
    "rk_dvol60":     dict(w=60,  floor=20, frac=0.6),
    "rk_udvol60":    dict(w=60,  floor=20, frac=0.6),
    "rk_maxdd60":    dict(w=60,  floor=20, frac=0.6),
    "rk_maxdd250":   dict(w=250, floor=20, frac=0.6),
    "rk_skew60":     dict(w=60,  floor=20, frac=0.6),
    "rk_skew250":    dict(w=250, floor=20, frac=0.6),
    "rk_kurt60":     dict(w=60,  floor=20, frac=0.6),
    "rk_kurt250":    dict(w=250, floor=20, frac=0.6),
    "rk_vr60250":    dict(w=250, floor=20, frac=0.6, w_short=60),
    "rk_vov120":     dict(w=120, floor=10, frac=0.6, sub=20, nseg=6),
}


# ===================================================================== #
# 0. 朴素实现（逐股循环，公式与 docstring 一一对应）
# ===================================================================== #
def mcnt(w, frac=0.6, floor=20):
    """有效样本门槛 max(floor, ceil(frac*w))（契约口径，两套实现共用）。"""
    return max(int(floor), int(math.ceil(float(frac) * int(w))))


def sl(t, w, X):
    lo = max(int(t) - int(w) + 1, 0)
    return X[lo:int(t) + 1], lo


def _std1(x):
    """样本标准差 ddof=1（pandas 默认口径）。"""
    return float(np.std(x, ddof=1)) if x.size > 1 else np.nan


def naive_panel(t, R, Cpx, mkt, vol_shift=0):
    """逐股朴素重算 14 个因子。

    vol_shift : 0 = 声明窗口 [t-w+1, t]；1 = 窗口整体前移一天 [t-w, t-1]（端点敏感性检验用）
    """
    k = R.shape[1]
    out = {nm: np.full(k, np.nan) for nm in NAMES}

    t60, t120, t250 = t - vol_shift, t - vol_shift, t - vol_shift
    W60, lo60 = sl(t60, 60, R)
    W120, lo120 = sl(t120, 120, R)
    W250, lo250 = sl(t250, 250, R)
    M60 = mkt[lo60:t60 + 1]
    mc60 = mcnt(60)
    mc120 = mcnt(120)
    mc250 = mcnt(250)
    mc_iv = mcnt(60, 0.6, 40)
    mc20 = mcnt(20, 0.6, 10)
    # vov：6 个不重叠 20 日段，端点 ti = t - 20j（j=0..5），覆盖 [t-119, t]
    seg_ends = [t - 20 * j for j in range(6)]
    vov_ok = all(ti >= 19 for ti in seg_ends)
    segW = [sl(ti, 20, R)[0] for ti in seg_ends]
    P60, _ = sl(t, 60, Cpx)
    P250, _ = sl(t, 250, Cpx)

    for j in range(k):
        # ---- 收益窗口（逐列）
        v60 = W60[:, j]; v60 = v60[np.isfinite(v60)]
        v120 = W120[:, j]; v120 = v120[np.isfinite(v120)]
        v250 = W250[:, j]; v250 = v250[np.isfinite(v250)]
        s60 = _std1(v60) if v60.size >= mc60 else np.nan
        s120 = _std1(v120) if v120.size >= mc120 else np.nan
        s250 = _std1(v250) if v250.size >= mc250 else np.nan
        out["rk_vol60"][j] = s60
        out["rk_vol120"][j] = s120
        out["rk_vol250"][j] = s250

        # ---- 下行 / 上行半偏差（ddof=0，除数 = 有效样本数 n）
        if v60.size >= mc60:
            dn2 = float(np.sum(np.where(v60 < 0.0, v60, 0.0) ** 2)) / v60.size
            up2 = float(np.sum(np.where(v60 > 0.0, v60, 0.0) ** 2)) / v60.size
            out["rk_dvol60"][j] = math.sqrt(dn2)
            if up2 > 0.0:
                out["rk_udvol60"][j] = math.sqrt(dn2 / up2)      # n 相消
            # up2 == 0（窗口内无正收益）→ NaN，与模块注释一致

        # ---- 区间最大回撤（显式滚动循环，完全不向量化）
        p = P60[:, j]
        valid = np.isfinite(p) & (p > 0.0)
        if int(valid.sum()) >= mc60:
            run = None
            worst = None
            for x in p[valid]:
                run = x if run is None else (x if x > run else run)
                d = 1.0 - x / run
                worst = d if worst is None else (d if d > worst else worst)
            out["rk_maxdd60"][j] = worst

        p = P250[:, j]
        valid = np.isfinite(p) & (p > 0.0)
        if int(valid.sum()) >= mc250:
            run = None
            worst = None
            for x in p[valid]:
                run = x if run is None else (x if x > run else run)
                d = 1.0 - x / run
                worst = d if worst is None else (d if d > worst else worst)
            out["rk_maxdd250"][j] = worst

        # ---- 偏度 / 超额峰度（总体中心矩 m_k = sum(x^k)/n）
        if v60.size >= mc60:
            mu = v60.mean()
            z = v60 - mu
            m2 = float(np.mean(z ** 2))
            if m2 > 0.0:
                out["rk_skew60"][j] = float(np.mean(z ** 3)) / m2 ** 1.5
                out["rk_kurt60"][j] = float(np.mean(z ** 4)) / m2 ** 2 - 3.0
        if v250.size >= mc250:
            mu = v250.mean()
            z = v250 - mu
            m2 = float(np.mean(z ** 2))
            if m2 > 0.0:
                out["rk_skew250"][j] = float(np.mean(z ** 3)) / m2 ** 1.5
                out["rk_kurt250"][j] = float(np.mean(z ** 4)) / m2 ** 2 - 3.0

        # ---- 特质波动率：市场模型 r_i = a + b*r_m + e，取残差标准差
        ok = np.isfinite(W60[:, j]) & np.isfinite(M60)
        x = W60[ok, j]
        m = M60[ok]
        n = x.size
        if n >= mc_iv:
            xc = x - x.mean()
            mcm = m - m.mean()
            v_m = float(np.sum(mcm ** 2)) / (n - 1.0)          # 市场方差（配对子样本）
            c_im = float(np.sum(xc * mcm)) / (n - 1.0)         # 协方差
            beta = (c_im / v_m) if v_m > 0.0 else 0.0
            res = xc - beta * mcm
            out["rk_ivol60"][j] = math.sqrt(float(np.sum(res ** 2)) / max(n - 1.0, 1.0))

        # ---- 波动率变化
        if np.isfinite(s60) and np.isfinite(s250) and s250 > 0.0:
            out["rk_vr60250"][j] = s60 / s250 - 1.0

        # ---- 波动率的波动：6 段 20 日波动率的标准差
        if vov_ok:
            sv = []
            for W in segW:
                u = W[:, j]; u = u[np.isfinite(u)]
                sv.append(_std1(u) if u.size >= mc20 else np.nan)
            sv = np.array(sv, dtype="float64")
            okv = np.isfinite(sv)
            if int(okv.sum()) >= max(2, 0.6 * 6):
                out["rk_vov120"][j] = _std1(sv[okv])
    return out

# ===================================================================== #
# 工具：日期对齐 / 差异统计 / 投毒
# ===================================================================== #
def near_date(dates, s):
    pos = int(dates.searchsorted(pd.Timestamp(s), side="right")) - 1
    return pos, dates[pos]


def mask_diff(a, b):
    """(联合有效格 max|diff|, 掩码不一致格数, mod-only 格数, naive-only 格数)"""
    fa, fb = np.isfinite(a), np.isfinite(b)
    both = fa & fb
    d = float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0
    return d, int((fa != fb).sum()), int((fa & ~fb).sum()), int((~fa & fb).sum())


def valid_cnt(t, w, X):
    lo = max(int(t) - int(w) + 1, 0)
    return np.isfinite(X[lo:int(t) + 1]).sum(0)


def poison(md, C, t, seed=12345):
    """把 t 行之后的所有可见数据换成随机噪声，用于检验因子是否偷看未来。"""
    rng = np.random.default_rng(seed)
    md2 = copy.copy(md)
    cache = dict(md._cache)
    for key in ("ret", "adj_close", "close", "cumadj_clean", "ret_ok", "listed",
                "open", "high", "low", "amt", "num_trades"):
        df = cache.get(key)
        if not isinstance(df, pd.DataFrame) or len(df) <= t + 1:
            continue
        d2 = df.copy()
        m = len(d2) - t - 1
        if pd.api.types.is_bool_dtype(d2.dtypes.iloc[0]):
            d2.iloc[t + 1:] = rng.random((m, d2.shape[1])) > 0.5
        else:
            d2.iloc[t + 1:] = rng.normal(0.0, 5.0, size=(m, d2.shape[1]))
        cache[key] = d2
    md2._cache = cache
    c2 = copy.copy(C)
    c2.md = md2
    for a in ("R", "C", "RAW", "ADJ_O", "ADJ_H", "ADJ_L", "A", "NT", "mkt", "mkt_fill"):
        X = getattr(C, a)
        Y = X.copy(order="K")      # 保持原 C/F 内存序，避免求和顺序差异掩盖真泄漏
        shape = (Y.shape[0] - t - 1,) if Y.ndim == 1 else (X.shape[0] - t - 1, X.shape[1])
        Y[t + 1:] = rng.normal(0.0, 5.0, size=shape)
        setattr(c2, a, Y)
    L = C.listed.copy(order="K")
    L[t + 1:] = rng.random((C.listed.shape[0] - t - 1, C.listed.shape[1])) > 0.5
    c2.listed = L
    return md2, c2

def naive_vol_only(t, X, w, shift=0, frac=0.6, floor=20):
    W, _ = sl(t - shift, w, X)
    mc = mcnt(w, frac, floor)
    out = np.full(X.shape[1], np.nan)
    for j in range(X.shape[1]):
        v = W[:, j]; v = v[np.isfinite(v)]
        if v.size >= mc:
            out[j] = np.std(v, ddof=1)
    return out


def naive_variants(t, R, CPX, MKT):
    """只重算口径变体：ivol(df=n-2)、skew/kurt 的偏差校正版。"""
    k = R.shape[1]
    out = {nm: np.full(k, np.nan) for nm in
           ("ivol_df2", "skew60_bc", "skew250_bc", "kurt60_bc", "kurt250_bc")}
    for w, tag in ((60, "60"), (250, "250")):
        lo = max(t - w + 1, 0)
        W = R[lo:t + 1]
        for j in range(k):
            v = W[:, j]; v = v[np.isfinite(v)]
            if v.size < mcnt(w) or v.size < 4:
                continue
            n = float(v.size)
            z = v - v.mean()
            m2 = float(np.mean(z ** 2))
            if m2 <= 0:
                continue
            g1 = float(np.mean(z ** 3)) / m2 ** 1.5
            g2 = float(np.mean(z ** 4)) / m2 ** 2 - 3.0
            out["skew" + tag + "_bc"][j] = g1 * math.sqrt(n * (n - 1.0)) / (n - 2.0)
            out["kurt" + tag + "_bc"][j] = ((n + 1.0) * g2 + 6.0) * (n - 1.0) / ((n - 2.0) * (n - 3.0))
    # ivol，残差自由度 n-2
    W, lo = sl(t, 60, R)
    M = MKT[lo:t + 1]
    for j in range(k):
        ok = np.isfinite(W[:, j]) & np.isfinite(M)
        x = W[ok, j]; m = M[ok]; n = x.size
        if n < mcnt(60, 0.6, 40) or n < 3:
            continue
        xc = x - x.mean(); mcm = m - m.mean()
        v_m = float(np.sum(mcm ** 2)) / (n - 1.0)
        beta = (float(np.sum(xc * mcm)) / (n - 1.0) / v_m) if v_m > 0 else 0.0
        res = xc - beta * mcm
        out["ivol_df2"][j] = math.sqrt(float(np.sum(res ** 2)) / (n - 2.0))
    return out


def naive_maxdd_full(t, CPX):
    """“截至 t”口径：cummax 不重置到窗口起点（全历史）。"""
    P = CPX[:t + 1]
    ok = np.isfinite(P) & (P > 0.0)
    run = np.maximum.accumulate(np.where(ok, P, -np.inf), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        dd = np.where(ok, 1.0 - P / np.where(run > 0.0, run, 1.0), -np.inf)
    return dd.max(0)


def synth_ctx(panel_R, mkt=None):
    """构造最小 Ctx（SimpleNamespace 版 md），供合成序列探针使用。"""
    n, kk = panel_R.shape
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    cols = ["S%03d" % i for i in range(kk)]
    Cpx = 100.0 * np.cumprod(1.0 + np.nan_to_num(panel_R), axis=0)
    if mkt is None:
        mkt = np.nanmean(panel_R, axis=1)

    def D(a):
        return pd.DataFrame(a, index=idx, columns=cols)

    md = SimpleNamespace(cumadj_clean=D(np.ones((n, kk))), ret=D(panel_R), adj_close=D(Cpx),
                         close=D(Cpx), open=D(Cpx), high=D(Cpx), low=D(Cpx),
                         amt=D(np.full((n, kk), 1e7)), num_trades=D(np.full((n, kk), 100.0)),
                         listed=D(np.ones((n, kk), bool)),
                         ret_ok=D(np.tile(mkt[:, None], (1, kk))))
    return md, Ctx(md)

# ===================================================================== #
# 主流程
# ===================================================================== #
def main():
    import time
    import warnings
    t0 = time.time()
    line = "=" * 100
    print(line)
    print("factor_zoo/risk.py 独立复核（朴素逐股重算 + 探针）")
    print(line)
    md = MarketData()
    C = Ctx(md)
    dates = md.dates
    R, CPX, MKT = C.R, C.C, C.mkt
    print("面板 %d 交易日 x %d 证券，%s ~ %s" % (C.n, C.k, dates[0].date(), dates[-1].date()))
    ts = []
    for s in TARGET_DATES:
        pos, d = near_date(dates, s)
        ts.append(pos)
        print("  目标 %-10s -> 实际交易日 %s (t=%d)" % (s, d.date(), pos))

    # ------------------------------------------------------------------ #
    print("\n" + "-" * 100)
    print("[1] 六个日期全横截面：模块 vs 朴素逐股重算（逐格比较，NaN 位置必须完全一致）")
    print("-" * 100)
    ref = {nm: [] for nm in NAMES}
    nai = {nm: [] for nm in NAMES}
    for t in ts:
        for nm in NAMES:
            ref[nm].append(np.asarray(RISK[nm]["func"](t, md, C), dtype="float64"))
        npv = naive_panel(t, R, CPX, MKT)
        for nm in NAMES:
            nai[nm].append(npv[nm])
        print("  完成 t=%4d (%s)，累计 %.1fs" % (t, dates[t].date(), time.time() - t0))

    rows, perdate = [], {}
    for nm in NAMES:
        A = np.vstack(ref[nm]); B = np.vstack(nai[nm])
        d, mm, aonly, bonly = mask_diff(A, B)
        rows.append(dict(factor=nm, max_abs_diff=d, mask_mismatch=mm, mod_only=aonly,
                         naive_only=bonly, inf_in_module=int(np.isinf(A).sum()),
                         cov_module=float(np.isfinite(A).mean()),
                         cov_naive=float(np.isfinite(B).mean())))
        perdate[nm] = [mask_diff(np.asarray(ref[nm][i]), np.asarray(nai[nm][i]))[:2]
                       for i in range(len(ts))]
    print("\n  %-13s %13s %9s %9s %9s %9s %9s" %
          ("factor", "max|diff|", "mask<->", "mod_only", "naive_only", "inf(mod)", "cov(mod)"))
    for r in rows:
        print("  %-13s %13.3e %9d %9d %9d %9d %9.4f" %
              (r["factor"], r["max_abs_diff"], r["mask_mismatch"], r["mod_only"],
               r["naive_only"], r["inf_in_module"], r["cov_module"]))
    pd.DataFrame(rows).to_csv(os.path.join(AUDIT, "risk_audit_compare.csv"),
                              index=False, encoding="utf-8-sig")
    print("\n  分日期 max|diff|（掩码不一致格数）")
    print("  %-13s" % "factor" + "".join("%20s" % str(dates[t].date()) for t in ts))
    for nm in NAMES:
        print("  %-13s" % nm + "".join("%20s" % ("%.2e (%d)" % perdate[nm][i])
                                       for i in range(len(ts))))

    # ------------------------------------------------------------------ #
    print("\n" + "-" * 100)
    print("[2] 端点敏感性对照：证明比较方法能发现窗口 off-by-one")
    print("-" * 100)
    A = np.vstack(ref["rk_vol60"])
    for tag, B in (("窗口前移一天 [t-w, t-1]", [naive_vol_only(t, R, 60, shift=1) for t in ts]),
                   ("窗口多取一天 [t-w, t]  ", [naive_vol_only(t, R, 61) for t in ts])):
        B = np.vstack(B)
        d, mm, _, _ = mask_diff(A, B)
        print("  %s：max|diff|=%.3e，掩码不一致=%d（若为 0 说明本对比没有检出力）" % (tag, d, mm))

    # ------------------------------------------------------------------ #
    print("\n" + "-" * 100)
    print("[3] 未来数据投毒：把 t 之后的所有可见数组换成随机噪声，因子值必须逐位不变")
    print("-" * 100)
    canary = lambda tt, m_, c_: np.nan_to_num(c_.R).sum(0)     # 确凿偷看未来的对照组
    for t in (ts[2], ts[4], ts[-1]):
        md2, C2 = poison(md, C, t)
        dR = float(np.nanmean(np.abs(C2.R[t + 1:] - C.R[t + 1:])))
        dcan = float(np.nanmax(np.abs(canary(t, md2, C2) - canary(t, md, C))))
        worst_d, worst_mm, worst_nm = 0.0, 0, "-"
        for nm in NAMES:
            a = np.asarray(RISK[nm]["func"](t, md2, C2), dtype="float64")
            b = np.asarray(RISK[nm]["func"](t, md, C), dtype="float64")
            d, mm, _, _ = mask_diff(a, b)
            if d > worst_d:
                worst_d, worst_nm = d, nm
            worst_mm = max(worst_mm, mm)
        print("  t=%4d (%s)：投毒强度 mean|ΔR|=%.3f，canary 差异=%.3e（必须>>0，否则测试无效）"
              % (t, dates[t].date(), dR, dcan))
        print("           14 个因子的最大数值差异 = %.3e（%s）；最大掩码不一致 = %d 格"
              % (worst_d, worst_nm, worst_mm))
        del md2, C2
        gc.collect()

    # ------------------------------------------------------------------ #
    print("\n" + "-" * 100)
    print("[4] 门槛诊断：窗口有效样本数 min_cnt 与“贴着门槛”的股票数（fin/bare）")
    print("-" * 100)
    print("  %-13s %5s %8s | %s" % ("factor", "w", "min_cnt",
                                    "  ".join("t%d:fin/bare" % i for i in range(len(ts)))))
    for nm in NAMES:
        p = PARAMS[nm]
        w = p["w"]
        mc = mcnt(w, p["frac"], p["floor"])
        cells = []
        mc_show = str(mc)
        for i, t in enumerate(ts):
            if nm == "rk_ivol60":
                cnt = (np.isfinite(R[max(t - w + 1, 0):t + 1]) &
                       np.isfinite(MKT[max(t - w + 1, 0):t + 1])[:, None]).sum(0)
                th = mc
            elif nm == "rk_vov120":
                segn = np.vstack([valid_cnt(t - 20 * j, 20, R) for j in range(6)])
                cnt = (segn >= mcnt(20, 0.6, 10)).sum(0)     # 有效段数
                th = int(max(2, 0.6 * 6))                    # >= 4 段
                mc_show = "4(段)"
            else:
                cnt = valid_cnt(t, w, R)
                th = mc
            fin = np.isfinite(np.asarray(ref[nm][i], dtype="float64"))
            cells.append("%d/%d" % (int((fin & (cnt >= th)).sum()),
                                    int((fin & (cnt <= th + 2)).sum())))
        print("  %-13s %5d %8s | %s" % (nm, w, mc_show, "  ".join("%12s" % c for c in cells)))

    print("\n  vov 的“0.6 门槛”口径（模块级约定写的是 0.6*w 有效日，vov 实际按段数计）：")
    for i, t in enumerate(ts):
        cnt120 = valid_cnt(t, 120, R)
        fin = np.isfinite(np.asarray(ref["rk_vov120"][i], dtype="float64"))
        if fin.any():
            c = cnt120[fin]
            print("    %s: 有效 vov 列 %d，其 120 日有效日数 min=%d / 中位=%.0f / "
                  "<0.6*120=72 的列数=%d（占 %.2f%%）"
                  % (dates[t].date(), int(fin.sum()), int(c.min()), float(np.median(c)),
                     int((c < 72).sum()), 100.0 * float((c < 72).mean())))

    # ------------------------------------------------------------------ #
    print("\n" + "-" * 100)
    print("[4b] NaN 政策核查：模块的 NaN 是否完全由声明的门槛解释")
    print("-" * 100)
    for i, t in enumerate(ts):
        c60 = valid_cnt(t, 60, R)
        c250 = valid_cnt(t, 250, R)
        civ = (np.isfinite(R[max(t - 59, 0):t + 1]) &
               np.isfinite(MKT[max(t - 59, 0):t + 1])[:, None]).sum(0)
        seg = np.vstack([valid_cnt(t - 20 * j, 20, R) for j in range(6)])
        # vov 还有一层日期级 early-return：t < 119 时 6 段里有一段不满 20 行，模块直接全 NaN
        exp_vov = ((seg >= mcnt(20, 0.6, 10)).sum(0) < max(2, 0.6 * 6)) | (t < 119)
        pairs = [("rk_vol60", c60 < mcnt(60)), ("rk_vol250", c250 < mcnt(250)),
                 ("rk_ivol60", civ < mcnt(60, 0.6, 40)), ("rk_vov120", exp_vov)]
        msg = []
        for nm, exp in pairs:
            v = np.asarray(ref[nm][i], float)
            msg.append("%s:未解释=%d" % (nm, int(((~np.isfinite(v)) != exp).sum())))
        print("  %s  %s" % (dates[t].date(), "  ".join(msg)))

    # ------------------------------------------------------------------ #
    print("\n" + "-" * 100)
    print("[5] 口径探针：残差自由度 / 偏峰总体-样本口径 / 回撤区间-全历史口径 / inf 扫描")
    print("-" * 100)
    vv = {nm: [] for nm in ("ivol_df2", "skew60_bc", "skew250_bc", "kurt60_bc", "kurt250_bc")}
    dd_full = []
    for t in ts:
        v = naive_variants(t, R, CPX, MKT)
        for nm in vv:
            vv[nm].append(v[nm])
        dd_full.append(naive_maxdd_full(t, CPX))
    iv2 = np.vstack(vv["ivol_df2"])
    iv1 = np.vstack(nai["rk_ivol60"])
    both = np.isfinite(iv1) & np.isfinite(iv2)
    pos = both & (iv1 > 0)
    print("  (a) ivol 残差自由度：模块(ddof=1, 即 n-1) vs n-2 版：max|diff|=%.3e，"
          "中位比值=%.6f，中位相对差=%.4f%%（理论 +0.86%% @ n=60）"
          % (float(np.max(np.abs(iv1[both] - iv2[both]))),
             float(np.median(iv2[pos] / iv1[pos])),
             100.0 * float(np.median(np.abs(iv2[pos] / iv1[pos] - 1.0)))))
    for nm, bk in (("rk_skew60", "skew60_bc"), ("rk_skew250", "skew250_bc"),
                   ("rk_kurt60", "kurt60_bc"), ("rk_kurt250", "kurt250_bc")):
        A = np.vstack(nai[nm]); B = np.vstack(vv[bk])
        o = np.isfinite(A) & np.isfinite(B)
        print("  (b) %s 总体口径 vs 样本(偏差校正)口径：max|diff|=%.3e，中位相对差=%.4f%%"
              % (nm, float(np.max(np.abs(A[o] - B[o]))),
                 100.0 * float(np.median(np.abs(B[o] - A[o]) / np.abs(A[o])))))
    A = np.vstack(ref["rk_maxdd60"]); B = np.vstack(dd_full)
    o = np.isfinite(A) & np.isfinite(B)
    print("  (c) 回撤：模块[区间内 cummax] vs [截至t 全历史 cummax]：max|diff|=%.3e，"
          "不一致列数=%d / %d（>0 说明模块确实是“区间内”口径）"
          % (float(np.max(np.abs(A[o] - B[o]))), int((np.abs(A[o] - B[o]) > 1e-12).sum()), int(o.sum())))
    mdd = np.vstack(ref["rk_maxdd60"]); m250 = np.vstack(ref["rk_maxdd250"])
    print("  (d) maxdd 值域：60日 min=%.6f max=%.6f，250日 min=%.6f max=%.6f，负值个数=%d"
          % (np.nanmin(mdd), np.nanmax(mdd), np.nanmin(m250), np.nanmax(m250),
             int((mdd[np.isfinite(mdd)] < 0).sum() + (m250[np.isfinite(m250)] < 0).sum())))
    ninf = sum(int(np.isinf(np.vstack(ref[nm])).sum()) for nm in NAMES)
    nnan = sum(int(np.isnan(np.vstack(ref[nm])).sum()) for nm in NAMES)
    print("  (e) inf 扫描：六个日期 x 14 因子 = %d 格，inf 格数=%d，NaN 格数=%d"
          % (len(ts) * C.k * len(NAMES), ninf, nnan))

    # ------------------------------------------------------------------ #
    print("\n" + "-" * 100)
    print("[6] 合成序列语义探针（模块在完全构造的数据上必须满足的代数性质）")
    print("-" * 100)
    rng = np.random.default_rng(20241016)
    n, kk = 300, 4
    R0 = rng.normal(0.0, 0.01, (n, kk))
    md0, C0 = synth_ctx(R0)
    t = n - 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        def mk(m_, c_):
            return {nm: np.asarray(RISK[nm]["func"](t, m_, c_), float) for nm in NAMES}

        F0 = mk(md0, C0)
        md1, C1 = synth_ctx(2.0 * R0)
        F1 = mk(md1, C1)
        md2, C2 = synth_ctx(-R0)
        F2 = mk(md2, C2)
        print("  (a) 收益整体 x2：vol60 比值=%.9f（应=2）；dvol60 比值=%.9f（应=2）；"
              "udvol60 max|差|=%.2e（应=0）；skew60 max|差|=%.2e（应=0）"
              % (float(np.mean(F1["rk_vol60"] / F0["rk_vol60"])),
                 float(np.mean(F1["rk_dvol60"] / F0["rk_dvol60"])),
                 float(np.nanmax(np.abs(F1["rk_udvol60"] - F0["rk_udvol60"]))),
                 float(np.nanmax(np.abs(F1["rk_skew60"] - F0["rk_skew60"])))))
        print("  (b) 收益取负：skew60 反号 max|g+g'|=%.2e；kurt60 max|差|=%.2e；vol60 max|差|=%.2e；"
              "udvol60xudvol60' 偏离 1 的最大值=%.2e"
              % (float(np.nanmax(np.abs(F2["rk_skew60"] + F0["rk_skew60"]))),
                 float(np.nanmax(np.abs(F2["rk_kurt60"] - F0["rk_kurt60"]))),
                 float(np.nanmax(np.abs(F2["rk_vol60"] - F0["rk_vol60"]))),
                 float(np.nanmax(np.abs(F2["rk_udvol60"] * F0["rk_udvol60"] - 1.0)))))
        m = rng.normal(0.0, 0.012, n)
        sig = np.array([0.0, 0.004, 0.012, 0.03])
        eps = np.column_stack([rng.normal(0.0, s, n) if s > 0 else np.zeros(n) for s in sig])
        md3, C3 = synth_ctx(m[:, None] + eps, mkt=m)
        F3 = mk(md3, C3)
        print("  (c) ivol 结构（r_i = 市场 + 特质噪声 sigma=[0, 0.004, 0.012, 0.03]）：")
        print("      ivol60 = %s" % np.array2string(F3["rk_ivol60"], precision=5))
        print("      vol60  = %s（市场自身波动=%.4f）"
              % (np.array2string(F3["rk_vol60"], precision=5), np.std(m, ddof=1)))
        Rd = np.zeros((n, 2)); Rd[-2, 0] = 0.10; Rd[-1, 0] = -0.10
        md4, C4 = synth_ctx(Rd)
        F4 = mk(md4, C4)
        print("  (d) maxdd：价格 100->110->99：模块 rk_maxdd60=%.9f，闭式 1-99/110=%.9f；"
              "全平价格列：maxdd=%.1f vol60=%.1f dvol60=%.1f udvol60=%s skew60=%s vr=%s vov120=%.1f"
              % (F4["rk_maxdd60"][0], 1.0 - 99.0 / 110.0, F4["rk_maxdd60"][1], F4["rk_vol60"][1],
                 F4["rk_dvol60"][1], F4["rk_udvol60"][1], F4["rk_skew60"][1],
                 F4["rk_vr60250"][1], F4["rk_vov120"][1]))
        Re = np.zeros((n, 1)); Re[n - 101, 0] = -0.50
        md5, C5 = synth_ctx(Re)
        F5 = mk(md5, C5)
        print("  (e) 区间口径：崩盘发生在 60 日窗口之外 -> 模块 maxdd60=%.3f（区间内），"
              "同点全历史 maxdd=%.3f（截至t 口径）"
              % (F5["rk_maxdd60"][0], float(naive_maxdd_full(t, C5.C)[0])))
        Rg = rng.normal(0.0, 0.01, (n, 4))
        Rg[:-36, 0] = np.nan; Rg[:-35, 1] = np.nan; Rg[:-150, 2] = np.nan; Rg[:-149, 3] = np.nan
        md6, C6 = synth_ctx(Rg)
        F6 = mk(md6, C6)
        n60 = np.isfinite(C6.R[-60:]).sum(0)
        n250 = np.isfinite(C6.R[-250:]).sum(0)
        e60, e250 = n60 < mcnt(60), n250 < mcnt(250)
        g60, g250 = np.isnan(F6["rk_vol60"]), np.isnan(F6["rk_vol250"])
        print("  (f) 门槛：各列 60 日窗有效点=%s -> 模块 vol60 NaN=%s，按 min_cnt(60)=%d 预期=%s，一致=%s"
              % (list(map(int, n60)), list(map(bool, g60)), mcnt(60), list(map(bool, e60)),
                 bool(np.array_equal(g60, e60))))
        print("      各列 250 日窗有效点=%s -> 模块 vol250 NaN=%s，按 min_cnt(250)=%d 预期=%s，一致=%s"
              % (list(map(int, n250)), list(map(bool, g250)), mcnt(250), list(map(bool, e250)),
                 bool(np.array_equal(g250, e250))))
        c6 = np.array([0.010, 0.012, 0.011, 0.013, 0.009, 0.0105])
        Rv = np.zeros((n, 1))
        for j, cj in enumerate(c6):
            end = n - 1 - 20 * j
            Rv[end - 19:end - 9, 0] = cj
            Rv[end - 9:end + 1, 0] = -cj
        md7, C7 = synth_ctx(Rv)
        F7 = mk(md7, C7)
        seg_vol = c6 * math.sqrt(20.0 / 19.0)
        print("  (g) vov：段内 +-c 交替（c=%s）-> 段波动=c*sqrt(20/19)，模块 rk_vov120=%.10f，"
              "闭式 std(段波动,ddof=1)=%.10f，差=%.2e"
              % (np.array2string(c6, precision=4), F7["rk_vov120"][0],
                 float(np.std(seg_vol, ddof=1)),
                 abs(F7["rk_vov120"][0] - float(np.std(seg_vol, ddof=1)))))
        Ri = rng.normal(0.0, 0.01, (n, 2))
        Ri[-3, 0] = 1e300
        Ri[-3, 1] = -0.999999
        mdi, Ci = synth_ctx(Ri)
        Fi = mk(mdi, Ci)
        ninf_s = sum(int(np.isinf(Fi[nm]).sum()) for nm in NAMES)
        print("  (h) 极端输入：1e300 收益（平方溢出）/ 价格近乎归零：14 个因子 inf 格数=%d；"
              "vol60(1e300)=%s（应 NaN），maxdd60(价格=0)=%.6f（应<1）"
              % (ninf_s, Fi["rk_vol60"][0], Fi["rk_maxdd60"][1]))

    # ------------------------------------------------------------------ #
    print("\n" + "-" * 100)
    print("[7] 真实数据诊断：复权口径 / 停牌(0 收益) / 无涨跌幅日 / 窗口污染")
    print("-" * 100)
    raw = md.close.values
    fadj = md.cumadj_clean.values
    adj_manual = raw * fadj
    d_adj = float(np.nanmax(np.abs(adj_manual - CPX)))
    rm = np.full_like(R, np.nan)
    lz = md.listed.values
    rr = adj_manual[1:] / adj_manual[:-1] - 1.0
    rm[1:] = np.where(lz[1:] & lz[:-1], rr, np.nan)
    d_ret = float(np.nanmax(np.abs(rm - R)))
    print("  复权口径：|close*cumadj_clean - C|max=%.2e；|pct_change(C) - R|max=%.2e" % (d_adj, d_ret))
    susp = md.suspended.values
    nl = md.no_limit.values
    Rf = np.isfinite(R)
    print("  停牌格=%d；其中 R 有效且恰为 0=%d（%.1f%%）；R 为 NaN 的停牌格=%d"
          % (int(susp.sum()), int((susp & Rf & (R == 0)).sum()),
             100.0 * float((R[susp & Rf] == 0).mean()), int((susp & ~Rf).sum())))
    aR = np.abs(R[Rf])
    print("  全样本 |R|：max=%.3f；|R|>0.5 的格=%d；|R|>0.2 的格=%d；no_limit 格=%d（其中 |R|>0.1 占 %.1f%%）"
          % (float(aR.max()), int((aR > 0.5).sum()), int((aR > 0.2).sum()), int(nl.sum()),
             100.0 * float((np.abs(R)[nl & Rf] > 0.1).mean())))
    uni = md.universe()
    for i, t in enumerate(ts):
        nopx = ~np.isfinite(CPX[t])
        if nopx.sum() == 0:
            continue
        pre = nopx & (np.isfinite(CPX[:t + 1]).sum(0) == 0)      # 真·未上市：窗口内无任何历史报价
        dele = nopx & ~pre                                       # t 日无报价但有历史（退市/长停）
        bad = {}
        for nm in NAMES:
            v = np.isfinite(np.asarray(ref[nm][i], float))
            if int(v[pre].sum()):
                bad[nm] = int(v[pre].sum())
        nzh = sum(1 for nm in NAMES
                  if int(np.isfinite(np.asarray(ref[nm][i], float))[dele].sum()))
        print("  %s：t 日无报价 %d 列 -> 真·未上市 %d 列，非 NaN 输出=%s（须为空）；"
              "另有历史但 t 日无报价 %d 列，其中 %d/14 个因子仍有值（退市股窗口内仍有数据，属预期）"
              % (dates[t].date(), int(nopx.sum()), int(pre.sum()), bad if bad else "无",
                 int(dele.sum()), nzh))
    print("\n  新股/无涨跌幅日对 rk_vol250 的污染（universe 默认要求上市满 250 日，"
          "恰好放行首日巨幅收益仍在 250 日窗内的股票）：")
    for i, t in enumerate(ts):
        c = nl[max(t - 249, 0):t + 1].sum(0)
        u = uni.values[t]
        v = np.asarray(ref["rk_vol250"][i], float)
        hit = u & (c > 0)
        oth = u & (c == 0) & np.isfinite(v)
        if u.sum() > 0 and np.isfinite(v[hit]).any():
            print("    %s 池内=%d，窗含 no_limit 日=%d：其 rk_vol250 中位=%.4f vs 池内其余=%.4f，"
                  "最大=%.4f（全池中位=%.4f）"
                  % (dates[t].date(), int(u.sum()), int(hit.sum()),
                     float(np.nanmedian(v[hit])), float(np.nanmedian(v[oth])),
                     float(np.nanmax(v[hit])), float(np.nanmedian(v[np.isfinite(v)]))))
        else:
            print("    %s 池内=%d，窗含 no_limit 日=%d" % (dates[t].date(), int(u.sum()), int(hit.sum())))
    print("\n  停牌(记 0 收益)对 rk_vol60 的影响，取 %s（池内股票按 60 日窗内停牌天数分组）：" % dates[ts[4]].date())
    tt = ts[4]
    s60 = susp[max(tt - 59, 0):tt + 1].sum(0)
    v60 = np.asarray(ref["rk_vol60"][4], float)
    u = uni.values[tt]
    pool = u & np.isfinite(v60)
    rank = np.full(C.k, np.nan)
    okr = np.isfinite(v60)
    rank[okr] = pd.Series(v60[okr]).rank(pct=True).values
    for a, b in ((0, 0), (1, 5), (6, 20), (21, 60)):
        sel = pool & (s60 >= a) & (s60 <= b)
        if sel.sum() > 0:
            print("    窗内停牌 %2d~%2d 天：池内 %4d 只，rk_vol60 中位=%.5f，"
                  "其 vol60 分位中位=%.1f%%（分位越低越像“低波”，越容易被买）"
                  % (a, b, int(sel.sum()), float(np.nanmedian(v60[sel])),
                     100.0 * float(np.nanmedian(rank[sel]))))
    print("  极端停牌（全市场，不限池；停牌日收益记 0）：")
    for i, t in enumerate(ts):
        sd = susp[max(t - 59, 0):t + 1].sum(0)
        v = np.asarray(ref["rk_vol60"][i], float)
        okv = np.isfinite(v)
        rp = np.full(C.k, np.nan)
        rp[okv] = pd.Series(v[okv]).rank(pct=True).values
        j = int(np.argmax(sd))
        sel30 = okv & (sd >= 30)
        q = float(np.median(rp[sel30])) if sel30.any() else float("nan")
        print("    %s 窗内停牌最多=%d 天（该股 rk_vol60=%.5f，vol60 分位=%.1f%%）；"
              "rk_vol60 恰为 0 的列=%d；停牌>=30 天的列=%d，其 vol60 分位中位=%.1f%%"
              % (dates[t].date(), int(sd[j]), float(v[j]), 100.0 * float(rp[j]),
                 int(((v == 0.0) & okv).sum()), int(sel30.sum()), 100.0 * q))

    # ------------------------------------------------------------------ #
    print("\n" + "-" * 100)
    print("[8] direction 先验与数值方向（因子值随“风险”增减的方向）")
    print("-" * 100)
    from scipy.stats import spearmanr
    base = np.asarray(ref["rk_vol60"][4], float)
    for nm in NAMES:
        v = np.asarray(ref[nm][4], float)
        o = np.isfinite(v) & np.isfinite(base)
        rho = float(spearmanr(v[o], base[o]).statistic)
        print("    %-13s direction=%+d  rho(因子, rk_vol60)=%+.3f  desc=%s"
              % (nm, RISK[nm]["direction"], rho, RISK[nm]["desc"]))

    # ------------------------------------------------------------------ #
    print("\n" + line)
    print("机器判定（判据：max|diff| <= 1e-9 且 NaN 掩码完全一致 且 无 inf）")
    print(line)
    bad = 0
    for r in rows:
        okk = (r["max_abs_diff"] <= 1e-9) and (r["mask_mismatch"] == 0) and (r["inf_in_module"] == 0)
        bad += 0 if okk else 1
        print("  %-13s max|diff|=%.3e  mask<->=%d  inf=%d  -> %s"
              % (r["factor"], r["max_abs_diff"], r["mask_mismatch"], r["inf_in_module"],
                 "通过" if okk else "不一致"))
    print("  %d/%d 因子与朴素重算逐格一致，总用时 %.1fs" % (len(NAMES) - bad, len(NAMES), time.time() - t0))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
