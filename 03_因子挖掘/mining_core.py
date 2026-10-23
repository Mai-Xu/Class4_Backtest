# -*- coding: utf-8 -*-
"""因子挖掘共享库：网格 / 前视收益 / IC / 分层 / 相关性 / 聚类 / 成本。

约定（与回测口径一致）
----------------------
- 调仓网格：每月最后一个交易日收盘生成信号，**次日开盘成交**。
- 前视收益一律用**可成交口径**：`ADJ_O[t+1+h] / ADJ_O[t+1] - 1`
  （t 日收盘出信号 → t+1 日开盘买入 → 持有 h 个交易日后开盘卖出）。
- 股票池：非 ST 全市场（`MarketData.universe` 默认口径，动态、无未来信息）。
- 成本：佣金单边万一（1e-4）+ 印花税万五（卖出 5e-4）。
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from core.lib import MarketData  # noqa: F401  (re-export)

# ---------------------------------------------------------------- 口径常量
CASH = 1e9                       # 初始资金 10 亿
COMMISSION = 1e-4                # 佣金单边万一
STAMP = 5e-4                     # 印花税万五（卖出）
SELL_RATE = COMMISSION + STAMP   # 卖出合计 6e-4
ROUND_TRIP = COMMISSION + SELL_RATE

BT_START, BT_END = "2025-01-02", "2026-09-30"      # 用户指定回测区间
SIGNAL0 = "2024-12-31"                             # 建仓信号日（次日开盘建仓）
LIQ_SIGNAL = "2026-09-29"                          # 清仓信号日（2026-09-30 开盘清空）
GRID_START = "2021-01-01"                          # 面板/评价起点
IS_END = "2024-12-31"                              # 样本内截止（定方向/筛选只用它）
OOS_START = "2025-01-01"

HORIZONS = (1, 5, 10, 20)        # IC 衰减用的持有期（交易日）

UNI_KW = dict(min_listed_days=250, adv_window=20, min_adv=2e7,
              price_min=2.0, exclude_st=True, exclude_no_limit=True,
              exclude_one_line=True)


# ---------------------------------------------------------------- 基础工具
def monthly_grid(md, start=GRID_START, end=BT_END, rule="M"):
    """月频调仓网格在 md.dates 中的下标列表。"""
    from core.engine import Backtest
    idx = sorted(Backtest._rebalance_index(md.dates, rule))
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    return [int(t) for t in idx if lo <= md.dates[t] <= hi]


def universe_mask(md, ts, **kw):
    """(len(ts), k) 的可交易掩码：非 ST 全市场动态池。"""
    u = md.universe(**(kw or UNI_KW))
    return np.asarray(u.values, dtype=bool)[list(ts)]


def adj_open(md):
    """复权开盘价矩阵（成交价口径）。"""
    return (md.open * md.cumadj_clean).values.astype("float64")


def fwd_returns(md, ts, horizons=HORIZONS):
    """{h: (len(ts), k)} 可成交前视收益；不足 h+1 行的位置为 NaN。"""
    ao = adj_open(md)
    n = ao.shape[0]
    out = {}
    for h in horizons:
        arr = np.full((len(ts), ao.shape[1]), np.nan)
        for i, t in enumerate(ts):
            j0, j1 = int(t) + 1, int(t) + 1 + int(h)
            if j1 < n:
                with np.errstate(invalid="ignore", divide="ignore"):
                    arr[i] = ao[j1] / ao[j0] - 1.0
        out[int(h)] = np.where(np.isfinite(arr), arr, np.nan)
    return out


def rank_avg(x):
    """平均秩（处理并列），比 argsort 版本更稳。"""
    x = np.asarray(x, dtype="float64")
    uq, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    start = csum - cnt
    avg = (start + csum - 1) / 2.0
    return avg[inv]


def cs_rank_z(x):
    """横截面秩 -> 标准正态近似（0~1 分位映射，抗极值）。"""
    x = np.asarray(x, dtype="float64")
    ok = np.isfinite(x)
    out = np.full(x.shape, np.nan)
    if ok.sum() < 3:
        return out
    r = rank_avg(x[ok])
    out[ok] = (r / max(len(r) - 1.0, 1.0) - 0.5) * 3.4641016151377544   # ≈ 2*sqrt(3)
    return out


def spearman(a, b, mask=None, min_n=50):
    """秩相关（可带掩码），样本不足返回 NaN。"""
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    ok = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        ok &= np.asarray(mask, dtype=bool)
    if ok.sum() < min_n:
        return np.nan
    ra, rb = rank_avg(a[ok]), rank_avg(b[ok])
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def ic_series(f, y, mask, min_n=50):
    """逐期 RankIC 序列（长度 = 期数）。"""
    out = np.full(len(f), np.nan)
    for i in range(len(f)):
        out[i] = spearman(f[i], y[i], mask[i], min_n=min_n)
    return out


def ic_stats(ic, periods_per_year=12.0):
    """IC 序列 -> 统计量。"""
    v = np.asarray(ic, dtype="float64")
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 3:
        return dict(n=n, ic=np.nan, ic_std=np.nan, icir=np.nan, t=np.nan, pos=np.nan)
    mu, sd = float(v.mean()), float(v.std(ddof=1))
    return dict(n=n, ic=mu, ic_std=sd,
                icir=(mu / sd * np.sqrt(periods_per_year)) if sd > 0 else np.nan,
                t=(mu / sd * np.sqrt(n)) if sd > 0 else np.nan,
                pos=float((v > 0).mean()))


def decile_matrix(f, y, mask, q=10):
    """分层平均前视收益：(q, 期数)，NaN 表示该层该期无样本。"""
    out = np.full((q, len(f)), np.nan)
    for i in range(len(f)):
        ok = np.asarray(mask[i], bool) & np.isfinite(f[i]) & np.isfinite(y[i])
        n = int(ok.sum())
        if n < q * 5:
            continue
        v = f[i][ok]
        order = np.argsort(v, kind="stable")
        grp = (np.arange(n) * q) // n
        yy = y[i][ok][order]
        for g in range(q):
            sel = yy[grp == g]
            if len(sel):
                out[g, i] = float(sel.mean())
    return out


def weights_from_pick(pick, k, cap=None):
    """等权权重向量（可选单票上限）。"""
    w = np.zeros(k, dtype="float64")
    if len(pick) == 0:
        return w
    w[np.asarray(pick, dtype=int)] = 1.0 / len(pick)
    if cap:
        w = np.minimum(w, cap)
        s = w.sum()
        if s > 0:
            w = w / s
    return w


def turnover_cost(w_new, w_old, buy_rate=COMMISSION, sell_rate=SELL_RATE):
    """换手成本（占 NAV 比例）：先卖后买。"""
    d = np.asarray(w_new, dtype="float64") - np.asarray(w_old, dtype="float64")
    return float(buy_rate * d[d > 0].sum() + sell_rate * (-d[d < 0]).sum())


def decile_ls(f, y, mask, q=10, long_top=True):
    """多空（D10-D1）逐期毛收益与换手成本，返回 dict。"""
    dm = decile_matrix(f, y, mask, q=q)
    k = np.asarray(f).shape[1]
    prev = {0: np.zeros(k), q - 1: np.zeros(k)}
    gross, cost, to = [], [], []
    for i in range(len(f)):
        ok = np.asarray(mask[i], bool) & np.isfinite(f[i])
        if ok.sum() < q * 5:
            continue
        v = f[i][ok]
        idx = np.flatnonzero(ok)
        order = idx[np.argsort(v, kind="stable")]
        n = len(order)
        grp = (np.arange(n) * q) // n
        lo = order[grp == 0]
        hi = order[grp == q - 1]
        w_lo, w_hi = weights_from_pick(lo, k), weights_from_pick(hi, k)
        g = np.nan_to_num(dm[q - 1, i]) - np.nan_to_num(dm[0, i])
        c = (turnover_cost(w_hi, prev[q - 1]) + turnover_cost(w_lo, prev[0]))
        t = 0.5 * (np.abs(w_hi - prev[q - 1]).sum() + np.abs(w_lo - prev[0]).sum()) / 2.0
        prev[q - 1], prev[0] = w_hi, w_lo
        gross.append(g)
        cost.append(c)
        to.append(t)
    gross = np.asarray(gross)
    cost = np.asarray(cost)
    net = gross - cost
    years = max(len(gross) / 12.0, 1e-9)

    def ann(x):
        return float(np.prod(1.0 + x) ** (1.0 / years) - 1.0) if len(x) else np.nan

    return dict(gross=gross, cost=cost, net=net, turnover=np.asarray(to),
                gross_ann=ann(gross), net_ann=ann(net),
                gross_sharpe=float(gross.mean() / gross.std(ddof=1) * np.sqrt(12)) if gross.std(ddof=1) > 0 else np.nan,
                net_sharpe=float(net.mean() / net.std(ddof=1) * np.sqrt(12)) if net.std(ddof=1) > 0 else np.nan,
                cost_ann=float(cost.mean() * 12.0),
                turnover_ann=float(np.mean(to) * 12.0))


def monotonicity(dm):
    """分层单调性：层号与平均收益的 Spearman 相关。"""
    m = np.nanmean(dm, axis=1)
    ok = np.isfinite(m)
    if ok.sum() < 4:
        return np.nan
    return spearman(np.arange(len(m))[ok].astype("float64"), m[ok], min_n=4)


def avg_corr_matrix(panels_z, mask, sample_n=900, seed=7, min_n=100):
    """平均横截面相关矩阵（在股票子样本上算，省时间）。"""
    names = list(panels_z.keys())
    if not names:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    cols = rng.choice(mask.shape[1], size=min(sample_n, mask.shape[1]), replace=False)
    acc = np.zeros((len(names), len(names)), dtype="float64")
    cnt = 0
    for i in range(mask.shape[0]):
        base = mask[i][cols] & np.isfinite(panels_z[names[0]][i][cols])   # 占位，下面逐因子重建
        X = np.empty((len(names), len(cols)), dtype="float64")
        okall = mask[i][cols].copy()
        for a, nm in enumerate(names):
            v = np.asarray(panels_z[nm][i][cols], dtype="float64")
            X[a] = np.nan_to_num(v, nan=0.0)
            okall &= np.isfinite(v)
        if okall.sum() < min_n:
            continue
        Xs = X[:, okall]
        Xs = Xs - Xs.mean(1, keepdims=True)
        sd = Xs.std(1, keepdims=True)
        sd[sd <= 0] = np.nan
        Xs = Xs / sd
        Xs = np.nan_to_num(Xs, nan=0.0)
        C = (Xs @ Xs.T) / Xs.shape[1]
        acc += C
        cnt += 1
    if cnt == 0:
        return pd.DataFrame(np.nan, index=names, columns=names)
    return pd.DataFrame(acc / cnt, index=names, columns=names)


def cluster_by_corr(corr: pd.DataFrame, thresh=0.85):
    """按 |相关| 层次聚类，返回 {因子名: 簇号}。没有 scipy 时退化为贪心聚类。"""
    names = list(corr.index)
    n = len(names)
    if n == 0:
        return {}
    C = np.asarray(corr.values, dtype="float64")
    C = np.nan_to_num(C, nan=0.0)
    C = np.clip((C + C.T) / 2.0, -1, 1)
    dist = 1.0 - np.abs(C)
    np.fill_diagonal(dist, 0.0)
    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
        Z = linkage(squareform(dist, checks=False), method="average")
        lab = fcluster(Z, t=1.0 - thresh, criterion="distance")
        return {nm: int(lab[i]) for i, nm in enumerate(names)}
    except Exception:                                   # noqa: BLE001
        lab = np.zeros(n, dtype=int)
        cur = 0
        for i in range(n):
            if lab[i] != 0:
                continue
            cur += 1
            lab[i] = cur
            for j in range(i + 1, n):
                if lab[j] == 0 and abs(C[i, j]) >= thresh:
                    lab[j] = cur
        return {nm: int(lab[i]) for i, nm in enumerate(names)}


# ---------------------------------------------------------------- 报告小工具
def md_table(df: pd.DataFrame, floatfmt="%.4f"):
    d = df.copy()
    d.columns = [str(c) for c in d.columns]
    cols = list(d.columns)
    head = "| " + " | ".join([str(d.index.name or "")] + cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    rows = []
    for i in range(len(d)):
        vals = [str(d.index[i])]
        for c in cols:
            v = d[c].iloc[i]
            if isinstance(v, (bool, np.bool_)):
                vals.append("是" if v else "否")
            elif isinstance(v, (int, np.integer)):
                vals.append(format(int(v), ","))
            elif isinstance(v, (float, np.floating)):
                vals.append("—" if not np.isfinite(v) else floatfmt % v)
            else:
                vals.append(str(v))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([head, sep] + rows)


def pct(v, n=2):
    try:
        if v is None or not np.isfinite(v):
            return "—"
    except TypeError:
        return "—"
    return f"{100 * float(v):.{int(n)}f}%"
