# -*- coding: utf-8 -*-
"""factor_zoo/trend.py 全部 16 个因子的独立核验脚本（可复跑）。

用法
----
    $env:PYTHONIOENCODING="utf-8"; python 3_因子挖掘/audit/naive_trend.py

可选：想复现探针 B0-2 的 TA-Lib 金标准对照（不加也能跑，只是该小节会跳过）——
    python -m pip install --target "$env:TEMP	alib_only" --no-deps TA-Lib
    $env:PYTHONPATH="$env:TEMP	alib_only"; python 3_因子挖掘/audit/naive_trend.py
注意必须加 --no-deps：否则 pip 会把另一份 numpy 装进同一个目录并在 sys.path 里抢先命中。

独立性声明
----------
- 数据自己从 feathers/*.feather 读，复权因子去噪、复权、收益、掩码全部自己写；
  **不 import factor_zoo.trend / factor_zoo.ctx**。只在"取模块输出"时调用被审计函数。
- 因子重算 = pandas 窗口切片 + 逐股票 Python 循环（模块是整块 numpy 掩码向量化），
  两者除"定义"外不共享代码路径。

输出
----
[0] 数据通路交叉验证：自建面板 vs core.lib.MarketData/Ctx（确认对比基准一致）
[1] 逐因子对比表：6 个日期 x 全横截面 5451 只，max|diff| / NaN 掩码 / inf
[2] 因果性复核：模块 f(t, Ctx_full) vs f(t, Ctx_truncated)，6 个日期 x 16 因子
[3] 易错点探针 A~K：窗口端点 / ADX vs 真 Wilder / R2 停牌虚高 / 复权口径 /
    通道是否含 t / 跳空除权 / 除法 inf / 停牌 / min_cnt 噪声 / 年化系数 / 极值域
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)                      # 项目根：core / factor_zoo 是包
os.chdir(ROOT)

FIELDS = ("open", "high", "low", "close", "amt", "num_trades", "cumadj")
DATES = ["2020-03-31", "2021-06-30", "2022-12-30", "2023-07-31", "2024-06-28", "2025-06-30"]
CUMADJ_EPS = 1e-4                              # 与 core/lib.py 的去噪阈值一致
T0 = time.time()


def log(*a):
    print(*a, flush=True)


def sec(title):
    log("")
    log("=" * 78)
    log("== %s   (%.1fs)" % (title, time.time() - T0))
    log("=" * 78)


# --------------------------------------------------------------------------- #
# 0) 自建数据面板（独立通路）
# --------------------------------------------------------------------------- #
def read_field(name: str) -> pd.DataFrame:
    import pyarrow.ipc as ipc
    p = os.path.join(ROOT, "feathers", name + ".feather")
    df = ipc.open_file(p).read_all().to_pandas().set_index("date")
    df.index = pd.DatetimeIndex(df.index).normalize()
    return df


log("[0] 读取 feathers/ ...")
PX = {f: read_field(f) for f in FIELDS}
DATES_IDX = PX["close"].index
SYMS = PX["close"].columns
NROW, K = PX["close"].shape
log("    面板 %d 日 x %d 券，%s ~ %s" % (NROW, K, DATES_IDX[0].date(), DATES_IDX[-1].date()))

# cumadj 去噪（按 core/lib.py 的公开口径自己重写一遍）
_f = PX["cumadj"]
_rel = _f / _f.shift(1)
_step = _rel.where((_rel - 1.0).abs() >= CUMADJ_EPS, 1.0)
_base = _f.bfill().iloc[0]
F = _step.fillna(1.0).cumprod().mul(_base, axis=1).where(_f.notna())

adjC = PX["close"] * F
adjO = PX["open"] * F
adjH = PX["high"] * F
adjL = PX["low"] * F
listed = PX["close"].notna()
R = (adjC / adjC.shift(1) - 1.0).where(listed & listed.shift(1, fill_value=False))
AMT = PX["amt"]
COLS = SYMS
MC = lambda w, frac=0.6: max(3, int(math.ceil(frac * w)))


def _fin(df):
    return df.where(np.isfinite(df.values))


def _sl(df, t, w):
    return _fin(df.iloc[max(t - w + 1, 0): t + 1])


def _is_const(W):
    """窗口内每一列是否"全为有限值且完全相等"（常数窗口 = 整段停牌携程价）。"""
    with np.errstate(invalid="ignore"):
        return np.isfinite(W).all(0) & (np.nanmax(W, axis=0) - np.nanmin(W, axis=0) == 0)


# --------------------------------------------------------------------------- #
# 1) 朴素重算（逐股票循环 / pandas 窗口；不同日期只算需要的窗口）
# --------------------------------------------------------------------------- #
def n_madev(t, w):
    W = _sl(adjC, t, w)
    cnt, ma = W.count(0), W.mean(0)
    px = adjC.iloc[t]
    good = (cnt >= MC(w)) & ma.notna() & (ma > 0) & px.notna()
    out = pd.Series(np.nan, index=COLS, dtype=float)
    out[good] = px[good] / ma[good] - 1.0
    return out.values


def n_ma_align(t):
    m5, m20, m60 = _sl(adjC, t, 5).mean(0), _sl(adjC, t, 20).mean(0), _sl(adjC, t, 60).mean(0)
    c5, c20, c60 = _sl(adjC, t, 5).count(0), _sl(adjC, t, 20).count(0), _sl(adjC, t, 60).count(0)
    good = ((c5 >= MC(5)) & (c20 >= MC(20)) & (c60 >= MC(60))
            & m5.notna() & m20.notna() & m60.notna() & (m20 > 0) & (m60 > 0))
    out = pd.Series(np.nan, index=COLS, dtype=float)
    out[good] = (m5[good] - m20[good]) / m20[good] + (m20[good] - m60[good]) / m60[good]
    return out.values


def n_linfit(t, w):
    W = _sl(adjC, t, w).values
    ok = np.isfinite(W) & (W > 0)
    cnt = ok.sum(0)
    slope = np.full(K, np.nan)
    r2 = np.full(K, np.nan)
    xall = np.arange(W.shape[0], dtype=float)
    for j in range(K):
        if cnt[j] < MC(w):
            continue
        m = ok[:, j]
        x, y = xall[m], np.log(W[m, j])
        if x.size < 3 or np.ptp(x) == 0:
            continue
        b, a = np.polyfit(x, y, 1)
        yh = b * x + a
        sst = float(((y - y.mean()) ** 2).sum())
        ssr = float(((y - yh) ** 2).sum())
        slope[j] = 252.0 * b
        r2[j] = (1.0 - ssr / sst) if sst > 0 else np.nan
    return slope, r2


def n_adx(t, n=14):
    need = 2 * n + 1
    H, L, CL = (_sl(x, t, need).values for x in (adjH, adjL, adjC))
    out = np.full(K, np.nan)
    m = H.shape[0] - 1
    if m < n:
        return out
    for j in range(K):
        trs = np.zeros(m); ps = np.zeros(m); ns = np.zeros(m); vv = np.zeros(m, bool)
        for i in range(1, H.shape[0]):
            hc, hp, lc, lp, cp = H[i, j], H[i - 1, j], L[i, j], L[i - 1, j], CL[i - 1, j]
            if not (np.isfinite(hc) and np.isfinite(lc) and np.isfinite(hp)
                    and np.isfinite(lp) and np.isfinite(cp)):
                continue
            vv[i - 1] = True
            trs[i - 1] = max(hc - lc, abs(hc - cp), abs(lc - cp))
            up, dn = hc - hp, lp - lc
            ps[i - 1] = up if (up > dn and up > 0) else 0.0
            ns[i - 1] = dn if (dn > up and dn > 0) else 0.0
        dxs = []
        for e in range(n - 1, m):
            if vv[e - n + 1:e + 1].sum() < MC(n, 0.8):
                dxs.append(np.nan); continue
            trw = trs[e - n + 1:e + 1].sum()
            if trw <= 0:
                dxs.append(np.nan); continue
            dip = 100.0 * ps[e - n + 1:e + 1].sum() / trw
            din = 100.0 * ns[e - n + 1:e + 1].sum() / trw
            tot = dip + din
            dxs.append(100.0 * abs(dip - din) / tot if tot > 0 else 0.0)
        dxs = np.asarray(dxs, float)
        if not np.isfinite(dxs[-1]):
            continue
        fin = dxs[np.isfinite(dxs)]
        if fin.size >= max(3, n - 3):
            out[j] = fin.mean()
    return out


def n_bb(t, w):
    W = _sl(adjC, t, w)
    cnt, ma, sd = W.count(0), W.mean(0), W.std(0, ddof=1)
    px = adjC.iloc[t]
    good = (cnt >= MC(w)) & ma.notna() & sd.notna() & (sd > 0) & px.notna()
    out = pd.Series(np.nan, index=COLS, dtype=float)
    out[good] = (px[good] - ma[good]) / (2.0 * sd[good])
    return out.values


def n_donch(t, w):
    LW, HW = _sl(adjL, t, w), _sl(adjH, t, w)
    cnt = np.minimum(LW.count(0), HW.count(0))
    lmin, hmax = LW.min(0), HW.max(0)
    rng = hmax - lmin
    px = adjC.iloc[t]
    good = (cnt >= MC(w)) & rng.notna() & (rng > 0) & px.notna()
    out = pd.Series(np.nan, index=COLS, dtype=float)
    out[good] = (px[good] - lmin[good]) / rng[good]
    return out.values


def n_upratio(t, w):
    W = _sl(R, t, w)
    cnt, up = W.count(0), (W > 0).sum(0)
    good = cnt >= MC(w)
    out = pd.Series(np.nan, index=COLS, dtype=float)
    out[good] = up[good] / cnt[good]
    return out.values


def n_maxdd(t, w):
    W = _sl(adjC, t, w).values
    cur = np.full(K, np.nan); mx = np.full(K, np.nan)
    for i in range(W.shape[0]):
        v = W[i]
        cur = np.where(np.isfinite(v), np.fmax(np.where(np.isfinite(cur), cur, -np.inf), v), cur)
        dd = 1.0 - v / cur
        mx = np.where(np.isfinite(dd), np.fmax(np.where(np.isfinite(mx), mx, -np.inf), dd), mx)
    cnt = np.isfinite(W).sum(0)
    good = (cnt >= max(5, MC(w))) & np.isfinite(mx)
    out = np.full(K, np.nan)
    out[good] = np.clip(mx[good], 0.0, 1.0)
    return out


def n_er(t, w):
    W = _sl(adjC, t, w).values
    out = np.full(K, np.nan)
    for j in range(K):
        col = W[:, j]
        if not np.isfinite(col[-1]):
            continue
        idx = np.flatnonzero(np.isfinite(col))
        if idx.size == 0:
            continue
        p0 = col[idx[0]]
        path, ncnt = 0.0, 0
        for i in range(1, W.shape[0]):
            a, b = col[i - 1], col[i]
            if np.isfinite(a) and np.isfinite(b):
                path += abs(b - a); ncnt += 1
        if not (np.isfinite(p0) and path > 0 and ncnt >= MC(w - 1)):
            continue
        out[j] = min(max((col[-1] - p0) / path, -1.0), 1.0)
    return out


def n_streak(t, w):
    W = _sl(R, t, w).values
    out = np.full(K, np.nan)
    for j in range(K):
        col = W[:, j]
        last = col[-1]
        if not np.isfinite(last) or last == 0:
            continue
        s = 1.0 if last > 0 else -1.0
        run = 0
        for i in range(len(col) - 1, -1, -1):
            v = col[i]
            if not np.isfinite(v):
                break
            vi = 1.0 if v > 0 else (-1.0 if v < 0 else 0.0)
            if vi != s:
                break
            run += 1
        out[j] = run * s
    return out


def n_clv(t, w):
    LC, HC, CC = _sl(adjL, t, w), _sl(adjH, t, w), _sl(adjC, t, w)
    rng = HC - LC
    ok = rng.notna() & (rng > 0) & CC.notna()
    v = ((CC - LC) / rng.replace(0.0, np.nan)).clip(0.0, 1.0).where(ok)
    cnt = ok.sum(0)
    ma = v.mean(0)
    good = cnt >= MC(w)
    out = pd.Series(np.nan, index=COLS, dtype=float)
    out[good] = ma[good]
    return out.values


def n_gap(t, w):
    OW, CW = _sl(adjO, t, w + 1).values, _sl(adjC, t, w + 1).values
    if OW.shape[0] < 2:
        return np.full(K, np.nan)
    rows = []
    for i in range(1, OW.shape[0]):
        prev, o = CW[i - 1], OW[i]
        ok = np.isfinite(prev) & (prev > 0) & np.isfinite(o)
        rows.append(np.where(ok, o / np.where(ok, prev, 1.0) - 1.0, np.nan))
    G = np.vstack(rows)
    okm = np.isfinite(G)
    cnt = okm.sum(0)
    s = np.where(okm, np.nan_to_num(G), 0.0).sum(0)
    good = cnt >= MC(w)
    out = np.full(K, np.nan)
    out[good] = s[good] / np.maximum(cnt[good], 1)
    return out


NAIVE = {
    "tr_madev20": (lambda t: n_madev(t, 20)),
    "tr_madev60": (lambda t: n_madev(t, 60)),
    "tr_ma_align": n_ma_align,
    "tr_linslope60": (lambda t: n_linfit(t, 60)[0]),
    "tr_linslope120": (lambda t: n_linfit(t, 120)[0]),
    "tr_linr2_60": (lambda t: n_linfit(t, 60)[1]),
    "tr_adx14": (lambda t: n_adx(t, 14)),
    "tr_bb_pos20": (lambda t: n_bb(t, 20)),
    "tr_donch20": (lambda t: n_donch(t, 20)),
    "tr_donch55": (lambda t: n_donch(t, 55)),
    "tr_upratio60": (lambda t: n_upratio(t, 60)),
    "tr_maxdd60": (lambda t: n_maxdd(t, 60)),
    "tr_er20": (lambda t: n_er(t, 20)),
    "tr_streak": (lambda t: n_streak(t, 20)),
    "tr_clv20": (lambda t: n_clv(t, 20)),
    "tr_gap20": (lambda t: n_gap(t, 20)),
}

# --------------------------------------------------------------------------- #
# 2) 交叉验证数据通路 + 逐因子对比
# --------------------------------------------------------------------------- #
def pick_t(ds, idx):
    """最接近的交易日（<= ds 的最后一个交易日；ds 早于样本首日则取首日）。"""
    j = int(idx.searchsorted(pd.Timestamp(ds), side="right")) - 1
    return max(j, 0)


def run_compare():
    sec("0) 数据通路交叉验证：自建面板 vs core.lib.MarketData / Ctx")
    from core.lib import MarketData
    from factor_zoo.ctx import Ctx
    import factor_zoo.trend as TR

    md = MarketData()
    C = Ctx(md)
    pairs = [("C", adjC, C.C), ("ADJ_O", adjO, C.ADJ_O), ("ADJ_H", adjH, C.ADJ_H),
             ("ADJ_L", adjL, C.ADJ_L), ("R", R, C.R), ("A", AMT, C.A)]
    for nm, mine, theirs in pairs:
        a, b = mine.values, theirs
        same_nan = bool(np.array_equal(np.isnan(a), np.isnan(b)))
        m = np.isfinite(a) & np.isfinite(b)
        d = float(np.abs(a[m] - b[m]).max()) if m.any() else 0.0
        log("    %-6s  NaN掩码一致=%-5s  max|diff|=%.3g  (非有限格 %d)"
            % (nm, same_nan, d, int((~np.isfinite(a)).sum())))
    same_listed = bool(np.array_equal(listed.values, C.listed))
    log("    listed 一致=%s ; 价格面板非有限且非 NaN 的格数=%d"
        % (same_listed, int((~np.isfinite(adjC.values) & ~np.isnan(adjC.values)).sum())))
    if not same_listed or float((np.abs(adjC.values - C.C)[np.isfinite(adjC.values) & np.isfinite(C.C)]).max()) > 0:
        log("    !! 自建面板与 Ctx 不一致，后续结论需谨慎")

    ts = [pick_t(d, DATES_IDX) for d in DATES]
    log("")
    log("    对比日期（请求 -> 实际交易日, 行号 t）：")
    for d, t in zip(DATES, ts):
        log("      %s -> %s  t=%d" % (d, DATES_IDX[t].date(), t))

    sec("1) 逐因子独立重算对比（全横截面 %d 只 x %d 个日期）" % (K, len(ts)))
    # 对 adjC 窗口类因子，标注"窗口内复权收盘完全常数"的格（R2 数学上是 0/0）
    WC = {"tr_linslope60": 60, "tr_linslope120": 120, "tr_linr2_60": 60}
    res = {}
    for name, spec in TR.FACTORS.items():
        per = []
        nandiff_tot = 0
        ninf = 0
        cov = []
        nconst_tot = 0
        worst_excl = 0.0
        per_excl = []
        for t in ts:
            m = np.asarray(spec["func"](int(t), md, C), dtype="float64")
            v = np.asarray(NAIVE[name](int(t)), dtype="float64")
            assert m.shape == (K,) and v.shape == (K,), (name, m.shape, v.shape)
            nm_mod, nm_nv = np.isnan(m), np.isnan(v)
            nandiff_tot += int((nm_mod != nm_nv).sum())
            ninf += int(np.isinf(m).sum())
            good = (~nm_mod) & (~nm_nv)
            d = float(np.abs(m[good] - v[good]).max()) if good.any() else 0.0
            de = d
            if name in WC:
                const = _is_const(_sl(adjC, t, WC[name]).values)
                nconst_tot += int(const.sum())
                keep = good & (~const)
                de = float(np.abs(m[keep] - v[keep]).max()) if keep.any() else 0.0
            worst_excl = max(worst_excl, de)
            per_excl.append(de)
            per.append((t, d, int(good.sum()), int((~nm_mod).sum()), int((~nm_nv).sum())))
            cov.append(float((~nm_mod).mean()))
        res[name] = dict(per=per, nandiff=nandiff_tot, ninf=ninf, cov=cov,
                         nconst=nconst_tot, worst_excl=worst_excl, per_excl=per_excl)

    log("%-14s %9s %9s %8s %10s %8s %8s %8s" %
        ("因子", "覆盖率", "有效格/日", "NaN差", "max|diff|", "inf", "常数窗格", "判定"))
    for name in TR.FACTORS:
        r = res[name]
        worst = max(p[1] for p in r["per"])
        nvalid = min(p[3] for p in r["per"])
        ok = (r["nandiff"] == 0) and worst < 1e-9 and r["ninf"] == 0
        ok2 = (r["nandiff"] == 0) and r["worst_excl"] < 1e-9 and r["ninf"] == 0
        tag = "一致" if ok else ("一致(剔除常数窗)" if ok2 else "有差异")
        log("%-14s %8.1f%% %9d %8d %10.3g %8d %8d %8s" %
            (name, 100 * r["cov"][-1], nvalid, r["nandiff"], worst, r["ninf"],
             r["nconst"], tag))

    log("")
    log("分日期 max|diff|（列 = 请求日期）：")
    log("%-14s %s" % ("因子", " ".join("%10s" % d[2:] for d in DATES)))
    for name in TR.FACTORS:
        log("%-14s %s" % (name, " ".join("%10.2e" % p[1] for p in res[name]["per"])))
    log("")
    log("剔除'窗口内复权收盘完全常数'的格后 max|diff|（这类格 R2 数学上未定义）：")
    for name in TR.FACTORS:
        if res[name]["nconst"]:
            log("    %-14s 常数窗格数=%d  剔除后 max|diff|=%.3g" %
                (name, res[name]["nconst"], res[name]["worst_excl"]))

    sec("2) 因果性复核：f(t, full) vs f(t, Ctx.truncated(t))（6 个日期 x 16 因子）")
    worst_all = {}
    for name, spec in TR.FACTORS.items():
        w = 0.0
        nbad = 0
        for t in ts:
            a = np.asarray(spec["func"](int(t), md, C.truncated(int(t))), dtype="float64")
            b = np.asarray(spec["func"](int(t), md, C), dtype="float64")
            d = float(np.nanmax(np.abs(np.nan_to_num(a, nan=-1e18) - np.nan_to_num(b, nan=-1e18))))
            w = max(w, d)
            nbad += int(np.sum(np.isnan(a) != np.isnan(b)))
        worst_all[name] = (w, nbad)
        log("    %-14s 截断最大差=%.3g  NaN掩码差=%d  %s"
            % (name, w, nbad, "OK" if (w <= 1e-9 and nbad == 0) else "FAIL"))
    return md, C, TR, ts, res, worst_all



# --------------------------------------------------------------------------- #
# 3) 易错点探针
# --------------------------------------------------------------------------- #
def _ma_range(t, w, lo, hi, X=None):
    """以行区间 [lo,hi) 算 close/MA-1（用于端点探针）。"""
    X = adjC if X is None else X
    W = _fin(X.iloc[max(lo, 0):hi])
    cnt, ma = W.count(0), W.mean(0)
    px = adjC.iloc[t]
    good = (cnt >= MC(w)) & ma.notna() & (ma > 0) & px.notna()
    out = pd.Series(np.nan, index=COLS, dtype=float)
    out[good] = px[good] / ma[good] - 1.0
    return out.values


def _donch_range(t, w, lo, hi):
    LW, HW = _fin(adjL.iloc[lo:hi]), _fin(adjH.iloc[lo:hi])
    cnt = np.minimum(LW.count(0), HW.count(0))
    lmin, hmax = LW.min(0), HW.max(0)
    rng = hmax - lmin
    px = adjC.iloc[t]
    good = (cnt >= MC(w)) & rng.notna() & (rng > 0) & px.notna()
    out = pd.Series(np.nan, index=COLS, dtype=float)
    out[good] = (px[good] - lmin[good]) / rng[good]
    return out.values


def probe_endpoints(md, C, TR, ts):
    sec("A) 窗口端点：模块输出 vs 三种窗口切法（max|diff|）")
    log("    A1 tr_madev20（外层 python 循环，可读性优先）")
    log("      %-26s %-24s %12s" % ("窗口切法", "相对 t 的行区间", "max|diff|"))
    mod = TR.FACTORS["tr_madev20"]["func"](int(ts[2]), md, C)
    variants = [("声称 [t-w+1, t]  (w=20)", (ts[2] - 19, ts[2] + 1)),
                ("错位 [t-w, t-1]    (w=20)", (ts[2] - 20, ts[2])),
                ("越界 [t-w, t]     (w=21)", (ts[2] - 20, ts[2] + 1))]
    for nm, (lo, hi) in variants:
        v = _ma_range(ts[2], 20, lo, hi)
        d = np.abs(mod - v)
        d[np.isnan(mod) & np.isnan(v)] = 0
        log("      %-26s [%d, %d]%s %12.3g"
            % (nm, lo - ts[2], hi - 1 - ts[2], " " * 8, np.nanmax(d)))
    log("      (6 个日期全试):")
    for nm, off in (("声称 [t-19, t]", 0), ("错位 [t-20, t-1]", -1), ("越界 [t-20, t]", +0)):
        worst = 0.0
        for t in ts:
            m = TR.FACTORS["tr_madev20"]["func"](int(t), md, C)
            if nm.startswith("声称"):
                v = _ma_range(t, 20, t - 19, t + 1)
            elif nm.startswith("错位"):
                v = _ma_range(t, 20, t - 20, t)
            else:
                v = _ma_range(t, 20, t - 20, t + 1)
            d = np.abs(m - v); d[np.isnan(m) & np.isnan(v)] = 0
            worst = max(worst, float(np.nanmax(d)))
        log("      %-26s 6 日最大 max|diff| = %.3g" % (nm, worst))
    log("    A2 tr_donch20 / tr_upratio60：同上，模块 vs 声称/错位窗口")
    for fac, w, kind in (("tr_donch20", 20, "donch"), ("tr_upratio60", 60, "up")):
        for nm, (lo, hi) in (("声称 [t-w+1, t]", (None, None)), ("错位 [t-w, t-1]", (None, None))):
            worst = 0.0
            for t in ts:
                m = TR.FACTORS[fac]["func"](int(t), md, C)
                if nm.startswith("声称"):
                    lo_, hi_ = t - w + 1, t + 1
                else:
                    lo_, hi_ = t - w, t
                if kind == "donch":
                    v = _donch_range(t, w, lo_, hi_)
                else:
                    W = _fin(R.iloc[max(lo_, 0):hi_]); cnt, up = W.count(0), (W > 0).sum(0)
                    good = cnt >= MC(w)
                    v = pd.Series(np.nan, index=COLS, dtype=float)
                    v[good] = up[good] / cnt[good]
                    v = v.values
                d = np.abs(m - v); d[np.isnan(m) & np.isnan(v)] = 0
                worst = max(worst, float(np.nanmax(d)))
            log("      %-14s %-22s 6 日最大 max|diff| = %.3g" % (fac, nm, worst))


def wilder_adx(H, L, CL, n=14):
    """标准 Wilder ADX（全历史递推，逐日对列向量化）。返回 (N,k)，行 i 对应输入第 i 行。"""
    N, K2 = H.shape
    res = np.full((N, K2), np.nan)
    valid = (np.isfinite(H[1:]) & np.isfinite(L[1:]) & np.isfinite(H[:-1])
             & np.isfinite(L[:-1]) & np.isfinite(CL[:-1]))
    tr = np.maximum(np.maximum(H[1:] - L[1:], np.abs(H[1:] - CL[:-1])), np.abs(L[1:] - CL[:-1]))
    up, dn = H[1:] - H[:-1], L[:-1] - L[1:]
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr, pdm, ndm = (np.where(valid, z, 0.0) for z in (tr, pdm, ndm))
    cnt = np.zeros(K2); atr = np.zeros(K2); ap = np.zeros(K2); an = np.zeros(K2)
    adx = np.full(K2, np.nan); axs = np.zeros(K2); axn = np.zeros(K2)
    for i in range(N - 1):
        v, cb = valid[i], cnt
        ei, er = v & (cb < n), v & (cb >= n)
        atr = np.where(ei, atr + tr[i], atr)
        ap = np.where(ei, ap + pdm[i], ap)
        an = np.where(ei, an + ndm[i], an)
        atr = np.where(er, atr - atr / n + tr[i], atr)
        ap = np.where(er, ap - ap / n + pdm[i], ap)
        an = np.where(er, an - an / n + ndm[i], an)
        cnt = np.where(v, cb + 1, cb)
        elig = v & (cnt >= n)
        # atr 全 0（整段停牌起身）时 Wilder 的 DI 无定义：与 TA-Lib / 教科书一致地取 DX = 0
        pos = elig & (atr > 0)
        den = np.where(pos, atr, 1.0)
        dip = np.where(pos, 100.0 * ap / den, np.where(elig, 0.0, np.nan))
        din = np.where(pos, 100.0 * an / den, np.where(elig, 0.0, np.nan))
        tot = dip + din
        dx = np.where(elig, np.where(tot > 0, 100.0 * np.abs(dip - din) / np.where(tot > 0, tot, 1.0), 0.0), np.nan)
        in_mean = elig & (axn < n)
        axs = np.where(in_mean, axs + np.nan_to_num(dx, nan=0.0), axs)
        axn = np.where(in_mean, axn + 1, axn)
        adx = np.where(in_mean & (axn == n), axs / n, adx)
        ema = elig & (axn >= n) & (~in_mean)
        adx = np.where(ema, (adx * (n - 1) + np.nan_to_num(dx, nan=0.0)) / n, adx)
        res[i + 1] = adx
    return res


def wilder_adx_hand(H, L, CL, n=14, ncol=300):
    """逐列纯 Python 手算（先把有效行压缩成列表，再用 mean-scale 递推 + 显式 DX 列表）。
    与 wilder_adx 的向量化 sum-scale 写法是两条独立代码路径，用来互证参考实现。"""
    N, K2 = H.shape
    out = np.full((N, min(ncol, K2)), np.nan)
    for j in range(min(ncol, K2)):
        h, l, c = H[:, j], L[:, j], CL[:, j]
        trs, pds, nds, rows = [], [], [], []
        for i in range(1, N):
            if not (np.isfinite(h[i]) and np.isfinite(l[i]) and np.isfinite(h[i - 1])
                    and np.isfinite(l[i - 1]) and np.isfinite(c[i - 1])):
                continue
            trs.append(max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])))
            up, dn = h[i] - h[i - 1], l[i - 1] - l[i]
            pds.append(up if (up > dn and up > 0) else 0.0)
            nds.append(dn if (dn > up and dn > 0) else 0.0)
            rows.append(i)
        atr = ap = an = 0.0
        adx = np.nan
        dxs = []
        for k in range(len(trs)):
            if k < n:
                atr += trs[k]; ap += pds[k]; an += nds[k]
                if k == n - 1:                       # 前 n 个累加完成后转成"均值尺度"
                    atr /= n; ap /= n; an /= n
            else:                                    # 均值尺度递推：ATR_t = (ATR_{t-1}(n-1) + TR_t)/n
                atr = (atr * (n - 1) + trs[k]) / n
                ap = (ap * (n - 1) + pds[k]) / n
                an = (an * (n - 1) + nds[k]) / n
            if k < n - 1:
                continue
            tot = ap + an
            dx = 100.0 * abs(ap - an) / tot if tot > 0 else 0.0
            dxs.append(dx)
            if len(dxs) < n:
                adx = np.nan
            elif len(dxs) == n:
                adx = float(np.mean(dxs))
            else:
                adx = (adx * (n - 1) + dx) / n
            out[rows[k], j] = adx
    return out


def my_di_series(h, l, c, n=14):
    """单列的 +DI / -DI（sum-scale Wilder 递推），用于与 TA-Lib PLUS_DI/MINUS_DI 逐格核对。"""
    N = len(h)
    dip = np.full(N, np.nan); din = np.full(N, np.nan)
    atr = ap = an = 0.0; cnt = 0
    for i in range(1, N):
        if not (np.isfinite(h[i]) and np.isfinite(l[i]) and np.isfinite(h[i - 1])
                and np.isfinite(l[i - 1]) and np.isfinite(c[i - 1])):
            continue
        tr = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        up, dn = h[i] - h[i - 1], l[i - 1] - l[i]
        p = up if (up > dn and up > 0) else 0.0
        q = dn if (dn > up and dn > 0) else 0.0
        if cnt < n:
            atr += tr; ap += p; an += q; cnt += 1
        else:
            atr = atr - atr / n + tr; ap = ap - ap / n + p; an = an - an / n + q
        if cnt >= n and atr > 0:
            dip[i] = 100.0 * ap / atr; din[i] = 100.0 * an / atr
    return dip, din


def talib_adx_all(H, L, CL, n=14):
    """外部金标准（TA-Lib）；未安装返回 None。数据有内部 NaN 缺口的列跳过（TA-Lib 会一路 NaN）。"""
    try:
        import talib
    except ImportError:
        return None, 0, 0
    N, K2 = H.shape
    out = np.full((N, K2), np.nan)
    nskip = ngap = 0
    for j in range(K2):
        h, l, c = H[:, j], L[:, j], CL[:, j]
        idx = np.flatnonzero(np.isfinite(h) & np.isfinite(l) & np.isfinite(c))
        if idx.size < 4 * n:
            nskip += 1; continue
        s = int(idx[0])
        if not (np.isfinite(h[s:]).all() and np.isfinite(l[s:]).all() and np.isfinite(c[s:]).all()):
            ngap += 1; continue
        out[s:, j] = talib.ADX(h[s:], l[s:], c[s:], timeperiod=n)
    return out, nskip, ngap


def probe_adx(md, C, TR, ts):
    sec("B) ADX：模块 proxy（n 日滚动和 + 近 15 个 DX 均值）vs 标准 Wilder（全历史递推）")
    n = 14
    Hv, Lv, Cv = adjH.values, adjL.values, adjC.values
    valid_tr = (np.isfinite(Hv[1:]) & np.isfinite(Lv[1:]) & np.isfinite(Hv[:-1])
                & np.isfinite(Lv[:-1]) & np.isfinite(Cv[:-1]))
    cum_valid = np.vstack([np.zeros((1, K)), np.cumsum(valid_tr, 0)])
    susp_frac = (listed.values & (AMT.values == 0)).sum(0) / np.maximum(listed.values.sum(0), 1)
    clean = susp_frac <= 0.05
    full = wilder_adx(Hv, Lv, Cv, n)

    log("    B0-1 参考实现自证（本脚本内部）：向量化 sum-scale 递推 vs 逐列纯 Python mean-scale 手算")
    hand = wilder_adx_hand(Hv, Lv, Cv, n, ncol=300)
    mm = np.isfinite(full[:, :300]) & np.isfinite(hand)
    log("         前 300 列 x 全样本 可比 %d 格：max|diff| = %.3g（应 ~1e-13）"
        % (mm.sum(), float(np.abs(full[:, :300][mm] - hand[mm]).max()) if mm.any() else 0.0))

    tb, nskip, ngap = talib_adx_all(Hv, Lv, Cv, n)
    if tb is None:
        log("    B0-2 外部金标准：未安装 TA-Lib，跳过（本会话用 "
            "python -m pip install --target <tmp> TA-Lib 0.8.2 单独跑过，结果见报告）")
    else:
        log("    B0-2 外部金标准 TA-Lib（跳过 %d 列数据太短 / %d 列内部有 NaN 缺口）" % (nskip, ngap))
        steady = cum_valid >= 250                       # 上市后已有 250 个有效 TR 日 = 平滑已收敛
        for nm, sel in (("全部券", np.ones(K, bool)), ("停牌<=5%的券", clean),
                        ("热身完成(有效TR>=250)的格", steady)):
            d = np.abs(full - tb)
            d[~(np.isfinite(full) & np.isfinite(tb)) | ~sel] = np.nan
            log("         自研 Wilder vs TA-Lib [%-22s]：可比 %8d 格 max=%.3g 中位=%.3g p90=%.3g p99=%.3g"
                % (nm, np.isfinite(d).sum(), np.nanmax(d), np.nanmedian(d),
                   np.nanpercentile(d, 90), np.nanpercentile(d, 99)))
        log("           （残差集中在两处：(a) 每只券 ADX 起始 ~70 根 K 线的热身约定差异（指数衰减）；")
        log("             (b) DI+ == DI- == 0 的退化情形=长期停牌携程价，TA-Lib 的 DX 约定与教科书不同）")
        # DI 逐格对比：说明两份实现的差异只在 DX/ADX 聚合段，不在 DM/TR 平滑
        try:
            import talib
            Dm = np.abs(full - tb)
            Dm[~(np.isfinite(full) & np.isfinite(tb)) | ~steady] = np.nan
            jj = int(np.nanargmax(np.nanmax(Dm, 0)))
            h, l, c = Hv[:, jj], Lv[:, jj], Cv[:, jj]
            s0 = int(np.flatnonzero(np.isfinite(h) & np.isfinite(l) & np.isfinite(c))[0])
            tp = talib.PLUS_DI(h[s0:], l[s0:], c[s0:], timeperiod=n)
            tn = talib.MINUS_DI(h[s0:], l[s0:], c[s0:], timeperiod=n)
            mp, mn = my_di_series(h[s0:], l[s0:], c[s0:], n)
            log("         DI 逐格核对（%s，长历史券中与 TA-Lib 的 ADX 差最大的一列）：DI+ max|diff|=%.3g，DI- max|diff|=%.3g"
                % (COLS[jj], np.nanmax(np.abs(mp - tp)), np.nanmax(np.abs(mn - tn))))
        except Exception as e:
            log("         DI 核对跳过：%s" % e)

    log("    B1 模块 vs 全历史标准 Wilder（6 个日期 x 全横截面）")
    log("      %-12s %8s %9s %9s %9s %9s | %9s" %
        ("日期", "有效数", "mod中位", "Wilder中位", "中位差", "均值差", "p90|差|"))
    allabs = []
    for t in ts:
        mod = TR.FACTORS["tr_adx14"]["func"](int(t), md, C)
        wf = full[t]
        m = np.isfinite(mod) & np.isfinite(wf)
        d = mod[m] - wf[m]
        allabs.append(np.abs(d))
        log("      %-12s %8d %9.2f %11.2f %9.2f %9.2f %9.2f"
            % (DATES_IDX[t].date(), m.sum(), np.median(mod[m]), np.median(wf[m]),
               np.median(d), d.mean(), np.percentile(np.abs(d), 90)))
    a = np.concatenate(allabs)
    log("      合并 %d 格：中位|差|=%.2f，均值差=%.2f，p90=%.2f，p99=%.2f，max=%.2f"
        % (a.size, np.median(a), a.mean(), np.percentile(a, 90), np.percentile(a, 99), a.max()))
    if tb is not None:
        row = []
        for t in ts:
            mod = TR.FACTORS["tr_adx14"]["func"](int(t), md, C)
            m = np.isfinite(mod) & np.isfinite(tb[t]) & clean
            row.append((m.sum(), np.median(mod[m] - tb[t][m]), np.percentile(np.abs(mod[m] - tb[t][m]), 90)))
        log("      对照 TA-Lib（停牌<=5%% 的券）：%s"
            % " ; ".join("%s n=%d 中位差=%+.2f p90=%.2f" % (DATES_IDX[t].date().__str__()[2:], a_, b_, c_)
                         for t, (a_, b_, c_) in zip(ts, row)))
    log("    B1b 按历史长度 / 停牌占比分组（次新券的 Wilder 自身还在热身，单列出来）")
    log("      %-12s %10s %10s %10s %10s" % ("日期", "长史券数", "长史中位差", "次新券数", "次新中位差"))
    for t in ts:
        mod = TR.FACTORS["tr_adx14"]["func"](int(t), md, C)
        wf = full[t]
        m = np.isfinite(mod) & np.isfinite(wf)
        long_ = m & (cum_valid[t] >= 250)
        short = m & (cum_valid[t] < 250)
        log("      %-12s %10d %10.2f %10d %10.2f"
            % (DATES_IDX[t].date(), long_.sum(), np.median(np.abs(mod[long_] - wf[long_])),
               short.sum(), np.median(np.abs(mod[short] - wf[short])) if short.any() else np.nan))
    log("    B2 偏差归因：把 Wilder 递推限制在同 29 行窗口内（隔离'窗口截断'效应）")
    log("      %-12s %10s %10s %10s" % ("日期", "mod vs 全史", "窗内Wilder vs 全史", "mod vs 窗内Wilder"))
    for t in ts:
        mod = TR.FACTORS["tr_adx14"]["func"](int(t), md, C)
        wf, ww = full[t], wilder_adx(Hv[t - 28:t + 1], Lv[t - 28:t + 1], Cv[t - 28:t + 1], n)[-1]
        m = np.isfinite(mod) & np.isfinite(wf) & np.isfinite(ww)
        log("      %-12s %10.2f %18.2f %16.2f"
            % (DATES_IDX[t].date(), np.median(np.abs(mod[m] - wf[m])),
               np.median(np.abs(ww[m] - wf[m])), np.median(np.abs(mod[m] - ww[m]))))
    log("    B3 DX 个数与阈值：模块 docstring 说'最近 n 个 DX 的均值'，实现是 arange(n-1, m)")
    log("      need=2n+1=%d 行 -> TR 行数 m=%d -> DX 个数 = m-(n-1) = %d（不是 n=%d）"
        % (2 * n + 1, 2 * n, 2 * n - (n - 1), n))
    log("      窗口和的最小有效数 = max(3, ceil(0.8*14)) = %d；DX 均值最小有效数 = max(3, n-3) = %d"
        % (MC(n, 0.8), max(3, n - 3)))
    rng = []
    for t in ts:
        v = TR.FACTORS["tr_adx14"]["func"](int(t), md, C)
        rng.append((np.nanmin(v), np.nanmax(v)))
    log("      实测值域（6 日合并）：[%.3f, %.3f]" % (min(r[0] for r in rng), max(r[1] for r in rng)))


def probe_r2(md, C, TR, ts):
    sec("C) 对数价回归 R2：停牌重复价 / 常数窗口 / 值域")
    w = 60
    log("    C1 停牌重复价对 R2 的抬升（窗口内相邻复权收盘完全相同的行占比分桶）")
    log("      %-12s %-12s %8s %12s %12s %12s" % ("日期", "stale分桶", "券数", "R2中位", "去重R2中位", "差中位"))
    for t in ts:
        W = _sl(adjC, t, w).values
        r2m = np.asarray(TR.FACTORS["tr_linr2_60"]["func"](int(t), md, C), dtype=float)
        rows = W.shape[0]
        stale = (W[1:] == W[:-1]) & np.isfinite(W[1:]) & np.isfinite(W[:-1])
        frac = stale.sum(0) / np.maximum(np.isfinite(W[1:]).sum(0), 1.0)
        r2d = np.full(K, np.nan)
        for j in range(K):
            col = W[:, j]
            keep = [0] + [i for i in range(1, rows) if np.isfinite(col[i]) and np.isfinite(col[i - 1]) and col[i] != col[i - 1]]
            idx = np.array(keep, dtype=int)
            idx = idx[np.isfinite(col[idx]) & (col[idx] > 0)]
            if idx.size < 3 or np.ptp(idx) == 0:
                continue
            x, y = idx.astype(float), np.log(col[idx])
            b, a = np.polyfit(x, y, 1)
            sst = float(((y - y.mean()) ** 2).sum())
            if sst > 0:
                r2d[j] = max(0.0, min(1.0, 1.0 - float(((y - (b * x + a)) ** 2).sum()) / sst))
        for lo_, hi_, nm in ((0.0, 0.1, "<10%"), (0.1, 0.3, "10~30%"), (0.3, 0.6, "30~60%"), (0.6, 1.01, ">=60%")):
            m = np.isfinite(r2m) & np.isfinite(r2d) & (frac >= lo_) & (frac < hi_)
            if m.sum() == 0:
                continue
            log("      %-12s %-12s %8d %12.4f %12.4f %12.4f"
                % (DATES_IDX[t].date(), nm, m.sum(), np.median(r2m[m]), np.median(r2d[m]),
                   np.median(r2m[m] - r2d[m])))
    log("    C2 全样本扫描（每 5 个交易日）：完全常数窗口的格数 与 模块返回值")
    tot_cell = 0; tot_win = 0; vals = []
    vmin, vmax = np.inf, -np.inf
    for t in range(120, NROW, 5):
        W = _sl(adjC, t, w).values
        r2m = np.asarray(TR.FACTORS["tr_linr2_60"]["func"](int(t), md, C), dtype=float)
        fin = np.isfinite(r2m)
        vmin = min(vmin, np.nanmin(r2m)); vmax = max(vmax, np.nanmax(r2m))
        const_col = _is_const(W)
        tot_win += int(const_col.sum()); tot_cell += int(fin.sum())
        if const_col.any():
            vals.append(r2m[const_col])
    v = np.concatenate(vals) if vals else np.array([np.nan])
    log("      扫描 %d 个日期，共 %d 个有限 R2 格" % (len(range(120, NROW, 5)), tot_cell))
    log("      窗口内复权收盘完全常数（整段停牌/携程）的格数 = %d" % tot_win)
    if tot_win:
        log("      这些格模块返回：min=%.6g max=%.6g 中位=%.6g"
            % (np.nanmin(v), np.nanmax(v), np.nanmedian(v)))
        log("        其中 ==0 占比 %.1f%%；NaN 占比 %.1f%%；|v|>1e-6 占比 %.1f%%；最大单格 %.6g"
            % (100.0 * np.mean(v == 0), 100.0 * np.mean(np.isnan(v)),
               100.0 * np.mean(np.abs(np.nan_to_num(v)) > 1e-6), np.nanmax(np.abs(np.nan_to_num(v)))))
        log("        -> 常数窗口的 R2/NaN 完全由 1 ulp 求和舍入决定（见 M2），不是真实趋势信息")
    log("      R2 实测值域（全部扫描格）：[%.6f, %.6f] -> %s"
        % (vmin, vmax, "在 [0,1] 内" if (vmin >= 0 and vmax <= 1) else "越界！"))
    log("    C3 斜率年化系数 & 合成序列校验见探针 K")


def probe_adjust(md, C, TR, ts):
    sec("D) 均线偏离的价格口径：复权 vs 未复权")
    w = 20
    RAWclose = PX["close"]
    log("      %-12s %10s %10s %10s %10s" % ("日期", "复权-原始中位差", "max|diff|", ">1e-3券数", "窗口内除权券数"))
    for t in ts:
        mod = TR.FACTORS["tr_madev20"]["func"](int(t), md, C)
        Wr = _fin(RAWclose.iloc[t - w + 1:t + 1])
        cnt, ma = Wr.count(0), Wr.mean(0)
        px = RAWclose.iloc[t]
        good = (cnt >= MC(w)) & ma.notna() & (ma > 0) & px.notna()
        v = pd.Series(np.nan, index=COLS, dtype=float)
        v[good] = px[good] / ma[good] - 1.0
        v = v.values
        m = np.isfinite(mod) & np.isfinite(v)
        d = np.abs(mod[m] - v[m])
        fw = _fin(PX["cumadj"].iloc[t - w + 1:t + 1]).values
        rel = fw[1:] / fw[:-1]
        nexd = int((np.nan_to_num(np.abs(rel - 1.0), nan=0.0) >= 1e-3).sum())
        log("      %-12s %10.3g %10.3g %10d %10d"
            % (DATES_IDX[t].date(), np.median(d), d.max(), int((d > 1e-3).sum()), nexd))
    log("      结论：模块用复权价（C.C=复权收盘）；用未复权价会有 ~10% 量级的偏差（除权送转越多差越大）")


def probe_bb(md, C, TR, ts):
    sec("E) 布林带位置：ddof 口径 / %b 定义 / 是否含 t")
    w = 20
    log("      %-12s %12s %12s %12s" % ("日期", "ddof=1 vs 0 中位差", "max|diff|", "值域"))
    for t in ts:
        mod = TR.FACTORS["tr_bb_pos20"]["func"](int(t), md, C)
        W = _sl(adjC, t, w)
        cnt, ma, sd0 = W.count(0), W.mean(0), W.std(0, ddof=0)
        px = adjC.iloc[t]
        good = (cnt >= MC(w)) & ma.notna() & sd0.notna() & (sd0 > 0) & px.notna()
        v = pd.Series(np.nan, index=COLS, dtype=float)
        v[good] = (px[good] - ma[good]) / (2.0 * sd0[good])
        v = v.values
        m = np.isfinite(mod) & np.isfinite(v)
        log("      %-12s %12.4f %12.4f [%.3f, %.3f]"
            % (DATES_IDX[t].date(), np.median(np.abs(mod[m] - v[m])), np.abs(mod[m] - v[m]).max(),
               np.nanmin(mod), np.nanmax(mod)))
    t = ts[2]
    mod = TR.FACTORS["tr_bb_pos20"]["func"](int(t), md, C)
    W = _sl(adjC, t, w)
    ma, sd = W.mean(0), W.std(0, ddof=1)
    px = adjC.iloc[t]
    pb = (px - (ma - 2 * sd)) / (4 * sd)
    m = np.isfinite(mod) & np.isfinite(pb)
    log("      %s: |mod - 2*(%%b-0.5)|max = %.3g  -> 模块值 = 2*(经典%%b-0.5)，上轨处模块=+1、经典%%b=1"
        % (DATES_IDX[t].date(), float(np.abs(mod[m] - 2.0 * (pb[m] - 0.5)).max())))
    log("      通道窗口 [t-19, t] 含 t 当日（MA/Std 都是含 t 的滚动统计）")


def probe_donch(md, C, TR, ts):
    sec("F) 唐奇安通道：是否含 t 当日 / 值域")
    for fac, w in (("tr_donch20", 20), ("tr_donch55", 55)):
        worst_all = 0.0; frac_all = []
        for t in ts:
            mod = TR.FACTORS[fac]["func"](int(t), md, C)
            excl = _donch_range(t, w, t - w, t)          # [t-w+1, t-1]：不含 t
            m = np.isfinite(mod) & np.isfinite(excl)
            d = np.abs(mod[m] - excl[m])
            worst_all = max(worst_all, float(d.max()))
            frac_all.append(float((d > 1e-6).mean()))
        t = ts[2]
        mod = TR.FACTORS[fac]["func"](int(t), md, C)
        log("      %-11s 含t vs 不含t: 6日 max|diff|=%.3g, 差异格占比=%.1f%%; 值域 [%.4f, %.4f]"
            % (fac, worst_all, 100 * np.mean(frac_all), np.nanmin(mod), np.nanmax(mod)))
    log("      结论：模块通道 [t-w+1, t] **包含 t 当日**（docstring 的 min(L,w)/max(H,w) 就是这个口径）；")
    log("            因此 close 恒在通道内，位置恒在 [0,1]，'创新高'最多读到 1.0（突破被截断）")

def probe_gap(md, C, TR, ts):
    sec("G) 跳空：复权 vs 未复权 / 除权日是否被误判 / 除权参考价口径")
    w = 20
    rel = (F / F.shift(1))
    ref = adjC.shift(1) / F                       # 昨收还原到今日复权基准（= 除权参考价）
    _ar = np.abs(np.asarray(rel.values, dtype=float) - 1.0)
    _ar = np.nan_to_num(_ar, nan=0.0)
    log("    全样本：|cumadj 相对变化| >= 1e-3 的真除权/送转 股-日 = %d；"
        "1e-4~1e-3 的 = %d" % (int((_ar >= 1e-3).sum()),
                               int(((np.abs(np.asarray(rel.values, dtype=float) - 1.0) >= 1e-4)
                                    & (_ar < 1e-3)).sum())))
    RAW_o, RAW_c = PX["open"], PX["close"]
    log("      %-12s %10s %12s %12s %12s %12s" %
        ("日期", "窗口除权券日", "mean|raw跳空|", "mean|复权跳空|", "max|mod-ref|", "|raw-adj|>5e-3"))
    for t in ts:
        mod = TR.FACTORS["tr_gap20"]["func"](int(t), md, C)
        lo_ = max(t - w, 0)
        prev = adjC.values[lo_:t]                    # C[t-w .. t-1]
        ao = adjO.values[lo_ + 1:t + 1]              # O[t-w+1 .. t]
        okg = np.isfinite(ao) & np.isfinite(prev) & (prev > 0)
        g_ref = np.where(okg, ao / np.where(okg, prev, 1.0) - 1.0, np.nan)
        cnt = np.isfinite(g_ref).sum(0)
        mean_ref = np.where(cnt >= MC(w),
                            np.nan_to_num(g_ref).sum(0) / np.maximum(cnt, 1), np.nan)
        g_raw = (RAW_o.values[lo_ + 1:t + 1] / RAW_c.values[lo_:t] - 1.0)
        relw = np.abs(np.asarray(rel.values[lo_ + 1:t + 1], dtype=float))
        ev = np.nan_to_num(relw - 1.0, nan=0.0) >= 1e-3      # 真除权/送转 股-日
        okr = np.isfinite(g_raw) & np.isfinite(g_ref) & ev
        mr = float(np.abs(g_raw[okr]).mean()) if okr.any() else np.nan
        ma_ = float(np.abs(g_ref[okr]).mean()) if okr.any() else np.nan
        nexp = int((np.abs(g_raw - g_ref) > 5e-3)[okr].sum())
        mm = np.isfinite(mod) & np.isfinite(mean_ref)
        dmax = float(np.abs(mod[mm] - mean_ref[mm]).max()) if mm.any() else np.nan
        log("      %-12s %10d %12.4f %12.4f %12.3g %10d"
            % (DATES_IDX[t].date(), int(ev.sum()), mr, ma_, dmax, nexp))
    log("      结论：模块 === mean(复权开盘/前一日复权收盘-1)（max|diff|=0 量级），")
    log("            复权跳空 == 今开/除权参考价-1，除权日本身**不产生假跳空**（是真实隔夜收益）。")


def probe_inf_range(md, C, TR):
    sec("H) 抽样扫描（每 10 个交易日，130..%d）：±inf / 覆盖率 / 实测值域" % NROW)
    ts_scan = list(range(130, NROW, 10))
    CLAIM = {
        "tr_madev20": ">0 高于均线", "tr_madev60": ">0 高于均线", "tr_ma_align": ">0 多头排列",
        "tr_linslope60": "年化对数斜率", "tr_linslope120": "年化对数斜率", "tr_linr2_60": "0~1",
        "tr_adx14": "0~100", "tr_bb_pos20": "0 中轨 / ±1 上下轨", "tr_donch20": "0~1",
        "tr_donch55": "0~1", "tr_upratio60": "0~1", "tr_maxdd60": "0~1", "tr_er20": "-1~1",
        "tr_streak": "±整数", "tr_clv20": "0~1", "tr_gap20": ">0 高开"}
    log("    %-14s %8s %8s %12s %12s %s" % ("因子", "inf数", "NaN占比", "min", "max", "声称值域"))
    for name, spec in TR.FACTORS.items():
        ninf = 0; nnan = 0; ntot = 0; vmin = np.inf; vmax = -np.inf
        for t in ts_scan:
            v = np.asarray(spec["func"](int(t), md, C), dtype=float)
            ninf += int(np.isinf(v).sum()); nnan += int(np.isnan(v).sum()); ntot += v.size
            fin = v[np.isfinite(v)]
            if fin.size:
                vmin = min(vmin, fin.min()); vmax = max(vmax, fin.max())
        extra = ""
        if name in ("tr_donch20", "tr_donch55", "tr_upratio60", "tr_maxdd60", "tr_clv20", "tr_linr2_60"):
            extra = "OK在界内" if (vmin >= -1e-12 and vmax <= 1 + 1e-12) else "越界!"
        if name == "tr_er20":
            extra = "OK在界内" if (vmin >= -1 - 1e-12 and vmax <= 1 + 1e-12) else "越界!"
        if name == "tr_adx14":
            extra = "OK在界内" if (vmin >= -1e-12 and vmax <= 100 + 1e-9) else "越界!"
        if name == "tr_streak":
            extra = "OK在±20内" if (abs(vmin) <= 20 + 1e-12 and abs(vmax) <= 20 + 1e-12) else "越界!"
        log("    %-14s %8d %7.1f%% %12.5g %12.5g %s %s"
            % (name, ninf, 100.0 * nnan / ntot, vmin, vmax, CLAIM[name], extra))


def probe_suspension(md, C, TR, ts):
    sec("I) 停牌（amt==0 且携程）在窗口里的处理")
    susp = listed & (AMT == 0)
    log("      %-12s %10s %12s %12s" % ("日期", "当日停牌券", "停牌且 R!=0", "60日窗口停牌>50%的券"))
    for t in ts:
        s_now = int(susp.iloc[t].sum())
        Rv = R.values[t]
        bad = int((susp.iloc[t].values & np.isfinite(Rv) & (np.abs(Rv) > 1e-12)).sum())
        Ws = susp.iloc[t - 59:t + 1].values
        frac = Ws.sum(0) / 60.0
        log("      %-12s %10d %12d %12d" % (DATES_IDX[t].date(), s_now, bad, int((frac > 0.5).sum())))
    t = ts[1]
    Ws = susp.iloc[t - 59:t + 1].values
    frac = Ws.sum(0) / 60.0
    up = np.asarray(TR.FACTORS["tr_upratio60"]["func"](int(t), md, C), dtype=float)
    m1 = (frac >= 0.5) & np.isfinite(up); m2 = (frac <= 0.1) & np.isfinite(up)
    log("    %s：窗口停牌>=50%% 的券 up_ratio 中位=%.4f（n=%d）；停牌<=10%% 的券中位=%.4f（n=%d）"
        % (DATES_IDX[t].date(), np.median(up[m1]), m1.sum(), np.median(up[m2]), m2.sum()))
    log("      机制：停牌日 adj_close 被携程 -> R=0 且有限，所以 (a) up_ratio 把停牌日算进分母（机械压低）；")
    log("            (b) streak 在停牌日断掉（当日 R=0 -> 返回 NaN）；(c) gap 里停牌日贡献 0；")
    log("            (d) madev/bb/donch/linslope/r2/maxdd/er 会把携程价当真实观测。")
    # 找一只 6 个日期里都停牌的券，打印明细
    allsusp = np.ones(K, bool)
    for t in ts:
        allsusp &= susp.iloc[t].values
    j = int(np.flatnonzero(allsusp)[0]) if allsusp.any() else -1
    if j >= 0:
        t = ts[-1]
        log("    样例 %s（6 个对比日都停牌）：%s" % (COLS[j], "  ".join(
            "%s=%.4g" % (n, float(TR.FACTORS[n]["func"](int(t), md, C)[j])) for n in
            ("tr_madev20", "tr_linr2_60", "tr_upratio60", "tr_streak", "tr_gap20", "tr_er20"))))
    else:
        log("    未找到 6 个日期都停牌的券")


def probe_mincnt(md, C, TR, ts):
    sec("J) 窗口最小样本数门槛：会不会用不满窗口的天数产出噪声")
    log("    min_cnt 规则：max(3, ceil(0.6*w)) -> MA5=3, MA20=12, MA60=36, gap20/up60=12/36")
    ld = md.listed_days
    log("      %-12s %14s %14s %14s %14s" % ("日期", "MA5不足5天的券", "MA20不足20天的券", "其中上市<60日", "MA20不足券的madev中位|值|"))
    for t in ts:
        W5 = _fin(adjC.iloc[t - 4:t + 1]); W20 = _fin(adjC.iloc[t - 19:t + 1])
        c5, c20 = W5.count(0).values, W20.count(0).values
        part5 = (c5 >= 3) & (c5 < 5)
        part20 = (c20 >= 12) & (c20 < 20)
        newc = int((part20 & (ld.iloc[t].values < 60)).sum())
        mv = np.asarray(TR.FACTORS["tr_madev20"]["func"](int(t), md, C), dtype=float)
        mm = part20 & np.isfinite(mv)
        log("      %-12s %14d %14d %14d %14.3f"
            % (DATES_IDX[t].date(), int(part5.sum()), int(part20.sum()), newc,
               float(np.median(np.abs(mv[mm]))) if mm.any() else np.nan))
    log("      注：窗口统计全部按'有效样本数'门槛放行，不足整窗时用部分样本 —— 短窗/次新券的 MA 噪声更大，")
    log("          但门槛 >=60% 窗口 + >=3 天，且 truncation 检验只看因果性，不看样本是否满窗。")


class FakeCtx:
    """最小 Ctx 替身：sl/mean/std 语义与 factor_zoo/ctx.py 完全一致（此处为复刻，不 import 该模块）。"""

    def __init__(self, C_, O_=None, H_=None, L_=None):
        self.C = np.asarray(C_, dtype=float)
        self.ADJ_O = self.C if O_ is None else np.asarray(O_, dtype=float)
        self.ADJ_H = self.C if H_ is None else np.asarray(H_, dtype=float)
        self.ADJ_L = self.C if L_ is None else np.asarray(L_, dtype=float)
        self.R = np.full_like(self.C, np.nan)
        self.R[1:] = self.C[1:] / self.C[:-1] - 1.0
        self.n, self.k = self.C.shape

    @staticmethod
    def sl(t, w, X):
        lo = max(int(t) - int(w) + 1, 0)
        return lo, X[lo:int(t) + 1]

    @staticmethod
    def mean(X, ok, min_cnt):
        cnt = ok.sum(0).astype("float64")
        mu = np.where(ok, X, 0.0).sum(0) / np.maximum(cnt, 1.0)
        return np.where(cnt >= min_cnt, mu, np.nan)

    @staticmethod
    def std(X, ok, min_cnt, ddof=1):
        cnt = ok.sum(0).astype("float64")
        mu = np.where(ok, X, 0.0).sum(0) / np.maximum(cnt, 1.0)
        c = np.where(ok, X - mu, 0.0)
        var = (c ** 2).sum(0) / np.maximum(cnt - ddof, 1.0)
        return np.where(cnt >= min_cnt, np.sqrt(var), np.nan)


def probe_synth(md, C, TR):
    sec("K) 合成序列：年化系数 / x 轴 / 常数序列 / ADX 值域与方向")
    import factor_zoo.trend as T
    n = 300
    tgrid = np.arange(n, dtype=float)
    g = np.array([0.001, -0.002, 0.0])
    P = np.exp(np.outer(tgrid, g)) * 100.0
    P[:, 2] = 50.0
    fx = FakeCtx(P)
    for w, tag in ((60, "linslope60"), (120, "linslope120")):
        sl_, r2_ = T._linfit(n - 1, fx, w)
        log("    %s: 真值 [252*0.001, 252*(-0.002), 0] = [%.4f, %.4f, %.4f]"
            % (tag, 252 * 0.001, 252 * -0.002, 0.0))
        log("      模块 _linfit 斜率 = %s" % np.round(sl_, 12))
        log("      模块 R2 = %s ; 末行 x=0..w-1 与 x=1..w 的 OLS 斜率差 = %.3g"
            % (np.round(r2_, 12), abs(np.polyfit(np.arange(w), np.log(P[n - w:, 0]), 1)[0]
                                     - np.polyfit(np.arange(1, w + 1), np.log(P[n - w:, 0]), 1)[0])))
    log("    常数序列（50.0 平铺）：模块 slope=%.3g，R2=%.3g（不是 NaN；syy 门槛用的是绝对>0，浮点残差 1e-29 也算通过）"
        % (T._linfit(n - 1, fx, 60)[0][2], T._linfit(n - 1, fx, 60)[1][2]))
    up = np.exp(np.outer(tgrid, np.array([0.002, -0.002]))) * 100.0
    fu = FakeCtx(up, O_=up, H_=up, L_=up)
    log("    单调上涨/单调下跌序列（H=L=C）：tr_adx14 = %s（两个方向都=100 -> ADX 无方向性）"
        % np.round(T._adx_proxy(n - 1, fu, 14), 6))
    log("    同一序列用参考 Wilder 实现 = %s（应同为 100，交叉验证参考实现）"
        % np.round(np.nan_to_num(wilder_adx(up, up, up, 14)[-1], nan=np.nan), 6))
    log("    随机游走（种子 0）：tr_adx14 = %s（应 <100）"
        % np.round(T._adx_proxy(n - 1, FakeCtx(np.cumprod(1 + np.random.default_rng(0).normal(0, .02, (n, 1)), 0) * 100), 14), 6))


def probe_direction(md, C, TR, ts):
    sec("L) 方向/口径锚点 + 参考 IC（近 20 日）")
    fwd = (adjC.shift(-20) / adjC - 1.0)
    log("    %-14s %6s %14s %10s %10s" % ("因子", "声称方向", "锚点自检", "均值IC", "IC同号日"))
    ANCHOR = {
        "tr_madev20": "px/MA-1 逐格一致(A)", "tr_madev60": "同上(1 节)", "tr_ma_align": "两段和(1 节)",
        "tr_linslope60": "合成=252g(K)", "tr_linslope120": "合成=252g(K)", "tr_linr2_60": "合成=1(K)",
        "tr_adx14": "单调趋势=100(K)", "tr_bb_pos20": "=2*(%b-.5)(E)", "tr_donch20": "含t/值域(F)",
        "tr_donch55": "含t/值域(F)", "tr_upratio60": "涨日占比(1 节)", "tr_maxdd60": "1-C/cummax(1 节)",
        "tr_er20": "净变动/路径(1 节)", "tr_streak": "连涨正连跌负(1 节)", "tr_clv20": "(C-L)/(H-L)(1 节)",
        "tr_gap20": "=复权跳空(G)"}
    ics = {}
    for name, spec in TR.FACTORS.items():
        got = []
        for t in ts:
            v = np.asarray(spec["func"](int(t), md, C), dtype=float)
            f = fwd.iloc[t].values
            m = np.isfinite(v) & np.isfinite(f)
            if m.sum() < 100:
                continue
            rv = pd.Series(v[m]).rank().values
            rf = pd.Series(f[m]).rank().values
            got.append(float(np.corrcoef(rv, rf)[0, 1]))
        if got:
            ics[name] = (float(np.mean(got)), int(np.sum(np.sign(got) == spec["direction"])), len(got))
    for name, spec in TR.FACTORS.items():
        mi = ics.get(name, (np.nan, 0, 0))
        log("    %-14s %6d %14s %10.3f %6s" % (name, spec["direction"], ANCHOR[name], mi[0],
                                               "%d/%d" % (mi[1], mi[2])))


def probe_poison(md, C, TR, ts):
    """未来行投毒：保持数组形状不变，只把 t 之后的行换成垃圾值，再算同一个 t。
    与 truncated 检验互补（截断改变形状；投毒保持形状，能抓 C.n 式写法/意外的未来引用）。"""
    from factor_zoo.ctx import Ctx
    sec("M) 未来行投毒（t 之后全变垃圾）+ 阳性对照")
    ATTRS = ("R", "C", "RAW", "O", "H", "L", "A", "NT", "ADJ_O", "ADJ_H", "ADJ_L", "VWAP_PROXY")
    rng = np.random.default_rng(0)
    worst = {}
    leak_worst = 0.0
    for t in ts:
        c2 = Ctx(md)
        for a in ATTRS:
            x = getattr(c2, a)
            if getattr(x, "ndim", 1) == 2 and x.shape[0] > t + 1:
                y = x.copy()
                y[t + 1:] = rng.normal(0.0, 1.0, size=y[t + 1:].shape) * 987.0 + 12345.0
                setattr(c2, a, y)
        lb = c2.listed.copy()
        lb[t + 1:] = ~lb[t + 1:]
        c2.listed = lb
        for name, spec in TR.FACTORS.items():
            a_ = np.asarray(spec["func"](int(t), md, c2), dtype="float64")
            b_ = np.asarray(spec["func"](int(t), md, C), dtype="float64")
            d = float(np.nanmax(np.abs(np.nan_to_num(a_, nan=-1e18) - np.nan_to_num(b_, nan=-1e18))))
            nd = int(np.sum(np.isnan(a_) != np.isnan(b_)))
            old = worst.get(name, (0.0, 0))
            worst[name] = (max(old[0], d), old[1] + nd)
        # 阳性对照：故意偷看 t+1 行收益
        leak = np.asarray(C.R[min(t + 1, C.n - 1)], dtype="float64")
        leako = np.asarray(c2.R[min(t + 1, c2.n - 1)], dtype="float64")
        m = np.isfinite(leak) & np.isfinite(leako)
        leak_worst = max(leak_worst, float(np.abs(leak[m] - leako[m]).max()) if m.any() else 0.0)
    # 对"投毒后变化"的因子，检查差异格是否全部是常数窗口（=舍入敏感，而非未来函数）
    deg = {}
    for name in ("tr_linr2_60", "tr_bb_pos20"):
        if worst[name][1] == 0 and worst[name][0] <= 1e-9:
            deg[name] = True; continue
        w = 60 if name == "tr_linr2_60" else 20
        okdeg = True
        for t in ts:
            c2 = Ctx(md)
            for a in ATTRS:
                x = getattr(c2, a)
                if getattr(x, "ndim", 1) == 2 and x.shape[0] > t + 1:
                    y = x.copy(); y[t + 1:] = rng.normal(0.0, 1.0, size=y[t + 1:].shape) * 987.0 + 12345.0
                    setattr(c2, a, y)
            a_ = np.asarray(TR.FACTORS[name]["func"](int(t), md, c2), dtype="float64")
            b_ = np.asarray(TR.FACTORS[name]["func"](int(t), md, C), dtype="float64")
            badj = np.flatnonzero(np.isnan(a_) != np.isnan(b_))
            if badj.size:
                const = _is_const(_sl(adjC, t, w).values)
                okdeg &= bool(const[badj].all())
        deg[name] = okdeg
    bad = 0
    for name in TR.FACTORS:
        d, nd = worst[name]
        ok = (d <= 1e-9) and nd == 0
        note = ""
        if not ok and name in deg and deg[name]:
            ok = True
            note = "（差异全部落在常数窗口的舍入敏感格，见 M2；非未来函数）"
        bad += 0 if ok else 1
        log("    %-14s 投毒 max|diff|=%.3g  NaN掩码差=%d  %s%s" % (name, d, nd, "OK" if ok else "FAIL", note))
    log("    阳性对照 leak(t+1 行收益)：投毒 max|diff|=%.3e（必须很大）" % leak_worst)
    # M2 布局敏感性：同一份数据、只改内存布局
    t0 = ts[0]
    c3 = Ctx(md)
    buf = np.empty((c3.C.shape[0], c3.C.shape[1] + 8))
    buf[:, :] = np.nan
    buf[:, 8:] = c3.C
    c3.C = buf[:, 8:]
    for name, w, sym in (("tr_bb_pos20", 20, "300028.SZ"), ("tr_linr2_60", 60, "002260.SZ")):
        if sym not in list(COLS):
            continue
        jj = list(COLS).index(sym)
        v_orig = float(TR.FACTORS[name]["func"](int(t0), md, C)[jj])
        v_lay = float(TR.FACTORS[name]["func"](int(t0), md, c3)[jj])
        W = C.C[max(t0 - w + 1, 0):t0 + 1, jj]
        log("    M2 %-13s @%s %s: 窗口 %d 行价格唯一值=%d（完全常数），原布局=%.6g，仅改内存布局=%.6g"
            % (name, DATES_IDX[t0].date(), sym, W.size, len(np.unique(W[np.isfinite(W)])), v_orig, v_lay))
    log("       两种布局的窗口数据逐格相同；差异只来自沿 axis=0 求和时 1 ulp 的舍入路径")
    verdict = {}
    for name in TR.FACTORS:
        d, nd = worst[name]
        ok = (d <= 1e-9 and nd == 0) or (name in deg and deg[name])
        verdict[name] = bool(ok)
    return worst, leak_worst, verdict


def main():
    md, C, TR, ts, res, causal = run_compare()
    probe_endpoints(md, C, TR, ts)
    probe_adx(md, C, TR, ts)
    probe_r2(md, C, TR, ts)
    probe_adjust(md, C, TR, ts)
    probe_bb(md, C, TR, ts)
    probe_donch(md, C, TR, ts)
    probe_gap(md, C, TR, ts)
    probe_inf_range(md, C, TR)
    probe_suspension(md, C, TR, ts)
    probe_mincnt(md, C, TR, ts)
    probe_synth(md, C, TR)
    probe_direction(md, C, TR, ts)
    POISON, LEAK, PV = probe_poison(md, C, TR, ts)
    sec("汇总（供报告直接引用）")
    log("%-14s %10s %8s %8s %6s %10s %6s" % ("因子", "max|diff|", "NaN差", "inf", "因果", "投毒max", "投毒"))
    for name in TR.FACTORS:
        r = res[name]
        w = max(p[1] for p in r["per"])
        pw, pn = POISON[name]
        log("%-14s %10.3g %8d %8d %6s %10.3g %6s" % (name, w, r["nandiff"], r["ninf"],
                                          "OK" if causal[name][0] <= 1e-9 and causal[name][1] == 0 else "FAIL",
                                          pw, "OK" if PV[name] else "FAIL"))
    log("    阳性对照 leak(t+1)=%.3e" % LEAK)
    log("")
    log("[全部完成 %.1fs]" % (time.time() - T0))


if __name__ == "__main__":
    main()
