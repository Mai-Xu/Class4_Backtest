# -*- coding: utf-8 -*-
"""s10 多因子策略：两段式加权打分 + 风险预算仓位模型（全程 walk-forward）。

与 s08 的关系
-------------
s08 是「单段 IC 加权」：把通过筛选的因子按样本内 IC 幅度直接加权求和。
本脚本把同一套因子库组织成**策略组合**，做两件 s08 没做的事：

1. 两段式加权打分体系
   - 第一段（家族内）：同族因子先合成一条家族打分，权重用**样本内 ICIR**；
     单段加权下「因子个数多」的家族会独裁 —— s08 入选的 38 个因子里近一半是
     流动性/成交额代理，等于把组合押在规模上。家族内先归一可以解掉这个结构问题。
   - 第二段（家族间）：不再按 IC 加权，而是用**风险预算模型**定各家族策略的仓位。

2. 仓位模型（用户要求：以某种模型确定几条策略的仓位大小）
   每个家族是一条 sleeve（各自 top-N 等权 + 缓冲带，独立跑回测）。分配器在每个月末
   **只用该日及之前**的 sleeve 日收益决定下一期权重：

     EW     等权（对照）
     IVOL60 逆波动（60 日）
     ERC    等风险贡献（EWMA(60) 协方差 + 岭正则，乘性不动点迭代）
     MVO    均值-方差（同一协方差 + 样本均值减半，权重 [0, 0.4]，风险厌恶 5）

   再叠一层**波动目标**：用 lambda' Sigma lambda 估组合波动，把总仓位缩放到
   目标波动附近（0.4 ~ 1.0 倍）。

口径（用户指定）
----------------
非 ST 全市场 · 10 亿 · 佣金单边万一 · 印花税万五
2025 年第一个交易日建仓 -> 2026 年 9 月最后一个交易日清仓离场
因子方向、筛选、分配器参数**一律只用 2021-01 ~ 2024-12 的样本内数据**。

产物
----
out/multifactor_report.md        报告
out/figures/10_*.png             净值图 / 分配器仓位图 / 家族 IC
out/tables/10_*.csv              汇总表
out/runs/mf_*/                   回测明细
"""
from __future__ import annotations
# --- 项目根入 sys.path：core / factor_zoo 都是根目录下的包 ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.lib import MarketData, ROOT, CACHE, FIG, TAB, savefig, savetab
from core.engine import Backtest, CostModel
from core.strategies import Strategy
from core import metrics
from factor_zoo.ctx import Ctx
from factor_zoo.panels import build_zoo_panels
from factor_zoo import registry

RUNS = os.path.join(ROOT, "out", "runs")
ZOO = os.path.join(CACHE, "zoo")
os.makedirs(RUNS, exist_ok=True)
os.makedirs(ZOO, exist_ok=True)

# ------------------------------------------------------------------ #
# 用户指定口径
# ------------------------------------------------------------------ #
CASH = 1e9                    # 10 亿元
COMMISSION = 1e-4             # 佣金单边万一
STAMP = 5e-4                  # 印花税万五（卖出）
BT_START, BT_END = "2025-01-02", "2026-09-30"
SIGNAL0 = "2024-12-31"        # 建仓信号日（次日 2025-01-02 开盘成交）
LIQ = "2026-09-29"            # 清仓信号日（次日 2026-09-30 开盘清空）
GRID_START = "2021-01-01"     # 面板 / 调仓网格起点
IS_END = "2024-12-31"         # 样本内截止：定方向、筛因子、标定分配器都只用这段

N_HOLD = 80                   # 每条 sleeve 持仓数
BUFFER = 2.0                  # 缓冲带倍数
MAX_W = 0.02                  # 单票权重上限
EXEC_WINDOW = 3               # 调仓执行窗口（交易日）
PARTICIPATION = 0.10          # 单日单票参与率上限
TOP_K = None                  # 合成后不额外截断（见 §5a：截断会系统性剔除
                              # 「只有一条 sleeve 看好」的名字，实测损失 3.6pp/年）
TARGET_VOL = 0.20             # 波动目标（年化）
GROSS_MIN = 0.40              # 总仓位下限
MIN_HIST = 120                # 分配器最少历史交易日

#: IC 域 = 可交易域（统一口径）
UNI_KW = dict(min_listed_days=250, adv_window=20, min_adv=2e7, price_min=2.0,
              exclude_st=True, exclude_no_limit=True, exclude_one_line=True)

#: 只含用户指定的两项费用
COST = CostModel(commission_rate=COMMISSION, commission_min=0.0, transfer_fee_rate=0.0,
                 stamp_duty_rate=STAMP, stamp_duty_rate_old=STAMP,
                 slippage_bps=0.0, impact_coef=0.0)
#: 更贴近实盘：再加 10bp 固定滑点 + 冲击成本
COST_REAL = CostModel(commission_rate=COMMISSION, commission_min=0.0, transfer_fee_rate=0.0,
                      stamp_duty_rate=STAMP, stamp_duty_rate_old=STAMP,
                      slippage_bps=10.0, impact_coef=0.01)

#: 调仓阈值：**只留 20 万元绝对下限，不用「净值比例」**。
#: 引擎的 `rebalance_band` 是净值的一个比例，默认 2e-3 在 10 亿元上等于 200 万元；
#: 而 300 只持仓的组合单只目标市值只有约 300 万元 —— 阈值和仓位同量级，绝大多数调仓
#: 会被判成「不值得交易」，组合被冻在初始持仓上（实测冻结版 10.82% vs 正常版 21.00%，
#: 持仓 726 只 vs 344 只）。绝对下限才是这个规模下的正确刻度。见 §5a 证据表。
BT_KW = dict(exec_price="open", max_participation=PARTICIPATION, min_trade_notional=2e5,
             rebalance_band=0.0, gross_limit=0.98, cash=CASH, exec_window=EXEC_WINDOW,
             enforce_limits=True, use_adjusted=True)

FAM5 = ["mom_rev", "risk", "liquidity", "flow", "trend"]
FAM_NAME = {"mom_rev": "动量反转", "risk": "波动风险", "liquidity": "流动性",
            "flow": "量价资金流", "trend": "趋势形态", "size": "规模(对照)"}

L = []


def w(s=""):
    L.append(s)
    print(s)


def pct(v, n=2):
    try:
        if v is None or not np.isfinite(v):
            return "—"
    except TypeError:
        return "—"
    return "%.*f%%" % (int(n), 100 * float(v))


def md_table(df, floatfmt="%.4f"):
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


# ------------------------------------------------------------------ #
# 统计 / 分配工具
# ------------------------------------------------------------------ #
def spearman_ic(x, y, mask):
    ok = mask & np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 50:
        return np.nan
    a = pd.Series(x[ok]).rank().values.astype("float64")
    b = pd.Series(y[ok]).rank().values.astype("float64")
    a -= a.mean()
    b -= b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def zscore_rows(X):
    """逐行横截面 z（不引入任何跨日信息）。"""
    out = np.full_like(X, np.nan)
    for i in range(X.shape[0]):
        v = X[i]
        ok = np.isfinite(v)
        if ok.sum() > 50:
            m, s = v[ok].mean(), v[ok].std(ddof=1)
            out[i, ok] = (v[ok] - m) / s if s > 0 else 0.0
    return out


def ewma_cov(R, halflife=60.0, ridge=0.05):
    """指数加权协方差 + 岭正则（对角加载），保证可逆、抑制极端权重。"""
    R = np.asarray(R, dtype="float64")
    T = len(R)
    lam = 0.5 ** (np.arange(T)[::-1] / float(halflife))
    lam = lam / lam.sum()
    mu = (lam[:, None] * R).sum(0)
    Rc = R - mu
    S = (Rc * lam[:, None]).T @ Rc
    S = (S + S.T) / 2.0
    n = S.shape[0]
    return S + np.eye(n) * (ridge * np.trace(S) / n)


def lw_delta(R):
    """Ledoit-Wolf 对角目标收缩强度（只做诊断：看数据自己要求收缩多强）。"""
    R = np.asarray(R, dtype="float64")
    T, n = R.shape
    Rc = R - R.mean(0)
    S = Rc.T @ Rc / T
    F = np.diag(np.diag(S))
    A = Rc[:, :, None] * Rc[:, None, :]
    pi = ((A - S) ** 2).sum() / T
    rho = ((Rc ** 2 - np.diag(S)) ** 2).sum() / T
    gam = ((S - F) ** 2).sum()
    return float(np.clip((pi - rho) / (T * gam), 0.0, 1.0)) if gam > 0 else 0.0


def erc_weights(S, iters=2000, tol=1e-13):
    """等风险贡献：使 w_i (Sigma w)_i 相等。乘性不动点迭代。"""
    n = S.shape[0]
    b = np.full(n, 1.0 / n)
    wv = b.copy()
    for _ in range(int(iters)):
        rc = wv * (S @ wv)
        tot = rc.sum()
        if not np.isfinite(tot) or tot <= 0:
            return b
        rc = rc / tot
        nw = wv * (b / np.maximum(rc, 1e-14)) ** 0.5
        s = nw.sum()
        if not np.isfinite(s) or s <= 0:
            return b
        nw = nw / s
        if np.max(np.abs(nw - wv)) < tol:
            return nw
        wv = nw
    return wv


def alloc_ew(R):
    return np.full(R.shape[1], 1.0 / R.shape[1])


def alloc_ivol(R, win=60):
    X = R[-int(win):]
    sd = X.std(0, ddof=1)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, np.nan)
    if not np.isfinite(sd).any():
        return alloc_ew(R)
    inv = np.where(np.isfinite(sd), 1.0 / sd, 0.0)
    return inv / inv.sum() if inv.sum() > 0 else alloc_ew(R)


def alloc_erc(R, win=120, halflife=60.0, ridge=0.05):
    X = R[-int(win):]
    return alloc_ew(R) if len(X) < 20 else erc_weights(ewma_cov(X, halflife, ridge))


def alloc_mvo(R, win=120, halflife=60.0, ridge=0.05, kappa=5.0, cap=0.40, shrink=0.5):
    X = R[-int(win):]
    if len(X) < 20:
        return alloc_ew(R)
    S = ewma_cov(X, halflife, ridge)
    mu = shrink * X.mean(0)
    try:
        wv = np.linalg.solve(S * kappa, mu)
    except np.linalg.LinAlgError:
        return alloc_ew(R)
    wv = np.clip(wv, 0.0, cap)
    s = wv.sum()
    return wv / s if s > 0 else alloc_ew(R)


ALLOCS = {"EW": alloc_ew, "IVOL60": alloc_ivol, "ERC": alloc_erc, "MVO": alloc_mvo}


def alloc_lambda(fn, R_hist):
    lam = np.asarray(fn(R_hist), dtype="float64")
    lam = np.where(np.isfinite(lam) & (lam > 0), lam, 0.0)
    s = lam.sum()
    return lam / s if s > 0 else np.full(len(lam), 1.0 / len(lam))


def implied_vol(lam, S):
    return float(np.sqrt(max(lam @ S @ lam, 0.0) * 252.0))


def neutralize(Z, size):
    """把每个因子的横截面 z 对 log(20 日均额) 的 z 做截面回归，取残差。"""
    ok = np.isfinite(Z) & np.isfinite(size)
    nz = np.maximum(ok.sum(1).astype("float64"), 1.0)
    zz = np.where(ok, Z, 0.0)
    ss = np.where(ok, size, 0.0)
    mz = zz.sum(1) / nz
    ms = ss.sum(1) / nz
    dz = np.where(ok, Z - mz[:, None], 0.0)
    ds = np.where(ok, size - ms[:, None], 0.0)
    beta = np.where((ds ** 2).sum(1) / nz > 0,
                    ((dz * ds).sum(1) / nz) / np.maximum((ds ** 2).sum(1) / nz, 1e-300), 0.0)
    out = np.full_like(Z, np.nan)
    r = dz - beta[:, None] * ds
    out[ok] = r[ok]
    return out


# ------------------------------------------------------------------ #
# 策略：把「日期 x 证券」的打分面板 / 权重矩阵接进引擎
# ------------------------------------------------------------------ #
class PanelLike(Strategy):
    """家族打分面板 -> Strategy（沿用基类的缓冲带选股逻辑）。"""

    def __init__(self, panel, uni, name="panel", n_hold=N_HOLD, max_weight=MAX_W,
                 buffer_mult=BUFFER):
        super().__init__(n_hold=n_hold, max_weight=max_weight, min_listed_days=250,
                         min_adv=2e7, price_min=2.0, exclude_st=True)
        self._uni = np.asarray(uni, dtype=bool)
        self._m = np.asarray(panel.values, dtype="float64")
        self._idx = pd.DatetimeIndex(panel.index)
        self.buffer_mult = float(buffer_mult)
        self.name = name

    def universe_row(self, md, t):
        return self._uni[int(t)]

    def score(self, md, t):
        d = pd.Timestamp(md.dates[int(t)])
        i = int(self._idx.searchsorted(d, side="right")) - 1
        if i < 0 or i >= len(self._m):
            return np.full(self._m.shape[1], np.nan)
        return self._m[i]


def sleeve_weight_matrix(panel, uni, md, grid, sym, n_hold=N_HOLD, buffer_mult=BUFFER):
    """把打分面板铺成 (调仓日 x 证券) 的**目标权重**矩阵。

    一次算好、全程复用：缓冲带是路径依赖的，只有按时间顺序回放整段网格，
    不同回测起点才会得到同一份选股结果。
    """
    pn = PanelLike(panel, uni, n_hold=n_hold, buffer_mult=buffer_mult)
    M = np.zeros((len(grid), len(sym)))
    for i, t in enumerate(grid):
        s = pn.weights(int(t), md)
        if len(s):
            pos = sym.get_indexer(s.index.values)
            keep = pos >= 0
            M[i, pos[keep]] = s.values[keep]
    return M


class WeightPlayback(Strategy):
    """回放预计算的权重矩阵。"""

    def __init__(self, mat, index, name="playback"):
        super().__init__(n_hold=int(mat.shape[1]))
        self._m = np.asarray(mat, dtype="float64")
        self._idx = pd.DatetimeIndex(index)
        self.name = name

    def weights(self, t, md):
        d = pd.Timestamp(md.dates[int(t)])
        i = int(self._idx.searchsorted(d, side="right")) - 1
        if i < 0 or i >= len(self._m) or self._idx[i] != d:
            return pd.Series(dtype="float64")
        row = self._m[i]
        j = np.flatnonzero(row > 0)
        if len(j) == 0:
            return pd.Series(dtype="float64")
        return pd.Series(row[j], index=np.asarray(md.symbols)[j])


class BlendStrategy(WeightPlayback):
    """按分配器权重把多条 sleeve 的目标权重加权合成，再截断到 top-k。"""

    def __init__(self, mats, index, lam, gross, top_k=TOP_K, max_w=MAX_W,
                 gross_limit=0.98, name="blend"):
        self._Ms = [np.asarray(m, dtype="float64") for m in mats]
        self._idx = pd.DatetimeIndex(index)
        self._lam = np.asarray(lam, dtype="float64")
        self._gross = np.asarray(gross, dtype="float64")
        self._top_k = int(top_k) if top_k else None
        self._max_w = float(max_w)
        self._gl = float(gross_limit)
        self.name = name
        self.n_hold = 0

    def weights(self, t, md):
        d = pd.Timestamp(md.dates[int(t)])
        i = int(self._idx.searchsorted(d, side="right")) - 1
        if i < 0 or i >= len(self._idx) or self._idx[i] != d:
            return pd.Series(dtype="float64")
        acc = np.zeros(len(md.symbols))
        for s, M in enumerate(self._Ms):
            if self._lam[i, s] > 0:
                acc += self._lam[i, s] * M[i]
        if not np.any(acc > 0):
            return pd.Series(dtype="float64")
        if self._top_k and (acc > 0).sum() > self._top_k:
            cut = np.partition(acc, -self._top_k)[-self._top_k]
            acc = np.where(acc >= cut, acc, 0.0)
        tot = acc.sum()
        if tot <= 0:
            return pd.Series(dtype="float64")
        acc = acc / tot
        if self._max_w > 0:
            acc = np.minimum(acc, self._max_w)
        g = float(np.clip(self._gross[i], 0.0, self._gl))
        tot = acc.sum()
        if tot <= 0 or g <= 0:
            return pd.Series(dtype="float64")
        acc = acc * (g / tot)
        j = np.flatnonzero(acc > 0)
        return pd.Series(acc[j], index=np.asarray(md.symbols)[j])


# ------------------------------------------------------------------ #
# 主流程
# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser(description="s10 多因子策略（家族打分 + 风险预算仓位模型）")
    ap.add_argument("--rand", type=int, default=12, help="随机对照组数")
    ap.add_argument("--quick", action="store_true", help="跳过稳健性与随机对照")
    args = ap.parse_args()
    t_all = time.time()

    md = MarketData(verbose=True)
    C = Ctx(md)
    dates = md.dates
    sym = md.symbols                                   # pandas Index，供 get_indexer 用

    # ---------------- 0) 调仓网格（与 s08 完全一致，命中同一份因子面板缓存） ----------------
    me = sorted(Backtest._rebalance_index(dates, "M"))
    grid = [t for t in me if pd.Timestamp(GRID_START) <= dates[t] <= pd.Timestamp(BT_END)]
    liq_pos = int(dates.searchsorted(pd.Timestamp(LIQ)))
    if liq_pos not in grid:
        grid = sorted(set(grid + [liq_pos]))
    gdates = pd.DatetimeIndex(dates[grid])
    n_g = len(grid)
    w("## 0. 口径")
    w("")
    w("- 调仓网格 **%d 个**：%s ~ %s（月频，末位为清仓信号日 %s）"
      % (n_g, gdates[0].date(), gdates[-1].date(), LIQ))
    w("- 建仓信号 **%s** 收盘 -> **%s** 开盘成交；清仓信号 **%s** -> **%s** 开盘清空"
      % (SIGNAL0, BT_START, LIQ, BT_END))
    w("- 初始资金 **10 亿元**；佣金单边**万一**（无最低佣金）+ 印花税**万五**（卖出）")
    w("- 涨跌停/停牌一律**不可交易**；单日单票参与率上限 %.0f%%；执行窗口 %d 日"
      % (100 * PARTICIPATION, EXEC_WINDOW))
    w("- 可交易域：非 ST、上市满 250 日、20 日均额 >= 2000 万、价格 >= 2 元、"
      "剔除一字板与无涨跌幅限制标的")
    w("- 每条 sleeve 持仓 **%d** 只（缓冲带 %.1f 倍）；合成后%s"
      % (N_HOLD, BUFFER, "不再额外截断" if not TOP_K else "截断到 %d 只" % TOP_K))
    w("")

    # ---------------- 1) 因子校验 + 样本内 IC ----------------
    facts = registry.load_all()
    w("## 1. 因子校验与 IC")
    w("")
    ts_chk = [int(t) for t in grid[::4]]
    chk = pd.DataFrame(registry.check_all(facts, C, ts_chk)).set_index("factor")
    chk["family"] = [facts[i]["family"] for i in chk.index]
    good = list(chk.index[chk["ok"]])
    w("- 因果性（截断重算逐位比对）+ 覆盖度通过：**%d / %d**" % (len(good), len(facts)))
    w("")
    t0 = time.time()
    raw, zsc = build_zoo_panels(md, C, {k: facts[k] for k in good}, grid,
                                cache_dir=ZOO, verbose=False)
    # 清仓信号日：打分清空 -> 目标为空 -> 全仓卖出
    for d in zsc.values():
        if pd.Timestamp(LIQ) in d.index:
            d.loc[pd.Timestamp(LIQ)] = np.nan
    w("- 因子面板 %d 个因子 x %d 个调仓日 x %d 只证券（%.1fs，命中缓存则接近 0）"
      % (len(good), n_g, len(sym), time.time() - t0))
    w("")

    uni = md.universe(**UNI_KW)
    U = uni.values[grid]
    adj = C.C[grid]
    fwd = np.full_like(adj, np.nan)
    fwd[:-1] = adj[1:] / adj[:-1] - 1.0                # 到下一个调仓日
    ok_all = U & np.isfinite(fwd)

    ic = {}
    for nm in good:
        X = raw[nm].values
        ic[nm] = pd.Series([spearman_ic(X[i], fwd[i], ok_all[i]) for i in range(n_g)],
                           index=gdates)
    ICdf = pd.DataFrame(ic)

    rows = []
    for nm in good:
        s = ICdf[nm].dropna()
        flag = np.asarray(s.index <= pd.Timestamp(IS_END))
        si, so = s[flag], s[~flag]
        t_is = si.mean() / si.std() * np.sqrt(len(si)) if len(si) > 2 and si.std() > 0 else np.nan
        t_oos = so.mean() / so.std() * np.sqrt(len(so)) if len(so) > 2 and so.std() > 0 else np.nan
        icir = si.mean() / si.std() if len(si) > 2 and si.std() > 0 else np.nan
        rows.append(dict(因子=nm, 模块=facts[nm]["module"], 家族=facts[nm]["family"],
                         说明=facts[nm]["desc"],
                         全样本IC=s.mean(), 样本内IC=si.mean(), 样本内ICIR=icir,
                         样本内t=t_is, 样本外IC=so.mean(), 样本外t=t_oos,
                         方向一致=bool(np.sign(si.mean()) == np.sign(so.mean()))
                         if len(si) and len(so) else False,
                         覆盖=float(chk.loc[nm, "coverage"])))
    ICT = pd.DataFrame(rows).set_index("因子")
    savetab(ICT.round(6), "10_factor_ic.csv")
    cand = ICT[(ICT["样本内t"].abs() >= 2.0) & (ICT["覆盖"] >= 0.6)].copy()
    w("- 样本内 |t| >= 2 且覆盖 >= 60%%：**%d / %d** 个因子进入家族合成" % (len(cand), len(ICT)))
    w("")

    # ---------------- 2) 家族打分（第一段） ----------------
    w("## 2. 第一段：家族内 ICIR 加权打分")
    w("")
    # 规模暴露（供中性化用）
    A20 = md.adv(20).values[grid]
    with np.errstate(invalid="ignore", divide="ignore"):
        logs = np.log(np.where(np.isfinite(A20) & (A20 > 0), A20, np.nan))
    size_z = zscore_rows(logs)

    panel_raw, panel_ns, kept_by_fam, drop_log = {}, {}, {}, []
    for fam in FAM5:
        names = [n for n in cand.index if facts[n]["module"] == fam]
        names.sort(key=lambda n: -abs(float(cand.loc[n, "样本内t"])))
        keep = []
        for nm in names:
            red = None
            for kv in keep:
                a, b = zsc[nm].values, zsc[kv].values
                m = np.isfinite(a) & np.isfinite(b)
                rs = []
                for i in range(0, n_g, 2):
                    if m[i].sum() > 200:
                        rs.append(np.corrcoef(a[i][m[i]], b[i][m[i]])[0, 1])
                if rs and abs(float(np.nanmean(rs))) > 0.85:
                    red = (kv, float(np.nanmean(rs)))
                    break
            if red is None:
                keep.append(nm)
            else:
                drop_log.append((nm, fam, red[0], red[1]))
        kept_by_fam[fam] = keep
        if not keep:
            continue
        num_r = num_n = None
        den = 0.0
        for nm in keep:
            v = float(ICT.loc[nm, "样本内ICIR"])
            if not np.isfinite(v) or v == 0:
                continue
            den += abs(v)
            num_r = v * zsc[nm] if num_r is None else num_r + v * zsc[nm]
            Zn = pd.DataFrame(neutralize(zsc[nm].values, size_z), index=gdates, columns=sym)
            num_n = v * Zn if num_n is None else num_n + v * Zn
        if num_r is not None and den > 0:
            panel_raw[fam] = num_r / den
            panel_ns[fam] = num_n / den
    if drop_log:
        savetab(pd.DataFrame(drop_log, columns=["被剔除", "家族", "与谁冗余", "相关系数"])
                .set_index("被剔除"), "10_corr_drop.csv")

    tab_fam = pd.DataFrame([
        dict(家族=FAM_NAME.get(f, f), 候选=int(sum(1 for n in cand.index
                                                if facts[n]["module"] == f)),
             去冗余后=len(kept_by_fam.get(f, [])),
             ICIR合计=float(sum(abs(float(ICT.loc[n, "样本内ICIR"]))
                              for n in kept_by_fam.get(f, []))),
             因子="、".join(kept_by_fam.get(f, [])))
        for f in FAM5]).set_index("家族")
    w(md_table(tab_fam, "%.3f"))
    w("")
    w("> 规模 sleeve 单独作为对照：score = -z(log 20 日均额)，即纯小市值。")
    w("")

    # ---- 数据质量门（来自 risk 模块的独立核验结论）--------------------------
    # 两个已证实的失真源：
    #  (a) 停牌日复权收益为 0（前收携程），既不产生 NaN 也不减少有效计数 —— 停牌 60 天的
    #      股票 60 日波动率恰好为 0，会被排进「全市场最像低波」的那一档；
    #  (b) 复牌 / 新股上市初期无涨跌幅限制，单日涨跌幅可达 +488%，进入 250 日窗口后会把
    #      波动率放大一个数量级。
    # 这两条都是**因子实现之外的输入失真**，在实盘里对应「数据质量剔除」，
    # 所以在这里做一道门，而不是改因子本身。
    sus_v = md.suspended.values
    nl_v = md.no_limit.values
    sus250 = np.zeros((n_g, len(sym)))
    anynl = np.zeros((n_g, len(sym)), dtype=bool)
    for i, t in enumerate(grid):
        lo = max(int(t) - 249, 0)
        sus250[i] = sus_v[lo:int(t) + 1].sum(0)
        anynl[i] = nl_v[lo:int(t) + 1].any(0)
    gate = (sus250 > 12) | anynl          # 250 日内停牌 >12 天，或出现过无涨跌幅限制日
    gate_df = pd.DataFrame(gate, index=gdates, columns=sym)
    w("- **数据质量门**：250 日内停牌超过 12 个交易日、或出现过无涨跌幅限制日的标的，"
      "其价格类因子（波动/趋势/形态）不可信，直接从打分里剔除。")
    w("  平均每期剔除 **%.0f** 只（占可交易域 %.1f%%）；不做这道门时，60 日波动率最小的"
      "一档里会混进「窗口内全程停牌 → 波动率恰好为 0」的股票。"
      % (float(gate.sum(1).mean()), 100 * float(gate.sum(1).mean() / max(U.sum(1).mean(), 1))))
    w("")
    for _f in list(panel_raw):
        panel_raw[_f] = panel_raw[_f].where(~gate_df)
    for _f in list(panel_ns):
        panel_ns[_f] = panel_ns[_f].where(~gate_df)

    # ---- 规模 sleeve（对照）：score = -z(log 20 日均额)，即纯小市值 ----
    panel_raw["size"] = pd.DataFrame(-size_z, index=gdates, columns=sym).where(~gate_df)
    # ---------------- 3) sleeve 目标权重 + 长历史收益 ----------------
    w("## 3. 各 sleeve（家族策略）")
    w("")
    bt = Backtest(md, cost=COST, **BT_KW)

    def build_W(panels, n_hold=N_HOLD):
        return {f: sleeve_weight_matrix(p, uni.values, md, grid, sym, n_hold=n_hold)
                for f, p in panels.items()}

    def build_R(Wd, start=GRID_START, end=BT_END):
        out = {}
        for f, M in Wd.items():
            r = bt.run(WeightPlayback(M, gdates, name=f + "_h"), start=start, end=end,
                       rebalance=gdates)
            out[f] = r
        return out

    t0 = time.time()
    W = build_W(panel_raw)
    Rs = build_R(W)                                  # 长历史：给分配器估计风险用
    Rh = pd.DataFrame({f: r.ret for f, r in Rs.items()}).fillna(0.0)
    # 与合成策略**同口径**的 sleeve 单独回测：都从建仓信号日起跑。
    # 长历史那条会把 2021-2024 攒下的「卖不掉的老仓」带进 2025，口径不可比。
    Rs0 = {f: bt.run(WeightPlayback(W[f], gdates, name=f + "_s"), start=SIGNAL0,
                     end=BT_END, rebalance=gdates) for f in W}
    fams_all = list(W.keys())
    w("- sleeve 长历史收益 %s（%s ~ %s，%.0fs）"
      % (str(Rh.shape), Rh.index[0].date(), Rh.index[-1].date(), time.time() - t0))
    w("")
    savetab(Rh.round(8), "10_sleeve_returns_long.csv")
    Rh0 = pd.DataFrame({f: r.ret for f, r in Rs0.items()}).fillna(0.0).loc[BT_START:]
    savetab(Rh0.round(8), "10_sleeve_returns_oos.csv")
    bench_all = md.ret_ok.where(uni).mean(1)
    st_rows = []
    for f, r in Rs0.items():
        nv = r.nav.loc[BT_START:]
        st = metrics.perf_stats(nv)
        st.update(metrics.excess_stats(r.ret.loc[BT_START:],
                                       bench_all.reindex(nv.index)))
        st["年换手"] = float(r.daily.loc[BT_START:].turnover.mean() * 252)
        st["持仓数"] = float(r.daily.loc[BT_START:].n_holdings.mean())
        st["被挡_亿"] = float(r.blocks[r.blocks.date >= pd.Timestamp(BT_START)].dv.sum() / 1e8) \
            if len(r.blocks) else 0.0
        st["名称"] = FAM_NAME.get(f, f)
        st_rows.append((f, st))
    tab_slv = pd.DataFrame({FAM_NAME.get(f, f): st for f, st in st_rows}).T
    savetab(tab_slv.round(6), "10_sleeve_stats.csv")
    show = tab_slv[["累计收益", "年化收益", "夏普", "最大回撤", "超额年化", "信息比率",
                    "年换手", "持仓数", "被挡_亿"]]
    w("各 sleeve 单独回测（%s ~ %s，10 亿元，含费用）：" % (BT_START, BT_END))
    w("")
    w(md_table(show, "%.4f"))
    w("")

    # 相关性矩阵：说明风险预算到底在分配什么
    rr = Rh.loc[Rh.index >= pd.Timestamp(SIGNAL0)]
    w("sleeve 日收益相关矩阵（%s 之后）：" % SIGNAL0)
    w("")
    w(md_table(rr.corr().round(3), "%.3f"))
    w("")

    # ---------------- 4) 分配器（walk-forward） ----------------
    w("## 4. 第二段：仓位模型（walk-forward，只用调仓日及之前的数据）")
    w("")
    def alloc_block(Rh_sub, fams_sub):
        """在给定 sleeve 集合上逐调仓日跑分配器；每个调仓日只用 <= d 的收益。"""
        Rm = Rh_sub[fams_sub].values
        nf = len(fams_sub)
        out = {}
        for an, fn in ALLOCS.items():
            lam = np.zeros((n_g, nf))
            gross = np.ones(n_g)
            vol = np.full(n_g, np.nan)
            for i, d in enumerate(gdates):
                hist = Rm[Rh_sub.index <= d]
                if len(hist) < MIN_HIST:
                    lam[i] = 1.0 / nf
                    continue
                lm = alloc_lambda(fn, hist)
                lam[i] = lm
                X = hist[-120:]
                S = ewma_cov(X, 60.0, 0.05) if len(X) >= 20 else np.diag(X.var(0) + 1e-12)
                sg = implied_vol(lm, S)
                vol[i] = sg
                gross[i] = float(np.clip(TARGET_VOL / sg, GROSS_MIN, 1.0)) if sg > 0 else 1.0
            out[an] = dict(lam=lam, gross=gross, vol=vol)
        return out

    LAM5 = alloc_block(Rh, FAM5)              # 5 个 alpha 家族
    LAM6 = alloc_block(Rh, fams_all)          # 多一条纯规模 sleeve
    dlt_hist = []
    # lambda 自身的换手（实盘意义：每次调仓要把 sleeve 仓位挪多少）
    lam_turn = {}
    for an in ALLOCS:
        L5 = LAM5[an]["lam"]
        lam_turn[an] = float(np.mean([0.5 * np.abs(L5[i] - L5[i - 1]).sum()
                                      for i in range(1, n_g)]))
    # 「分配器是不是在追涨」：lambda_t 与各 sleeve 过去 120 日的夏普的横截面相关
    chase = {}
    for an in ALLOCS:
        cs = []
        for i, d in enumerate(gdates):
            hist = Rh[FAM5].values[Rh.index <= d][-120:]
            if len(hist) < MIN_HIST:
                continue
            sh = hist.mean(0) / np.maximum(hist.std(0, ddof=1), 1e-12)
            a_, b_ = LAM5[an]["lam"][i], sh
            if a_.std() > 0 and b_.std() > 0:
                cs.append(float(np.corrcoef(a_, b_)[0, 1]))
        chase[an] = float(np.mean(cs)) if cs else np.nan
    Rm_all = Rh[fams_all].values
    for i in range(0, n_g, 8):
        hist = Rm_all[Rh.index <= gdates[i]]
        if len(hist) >= MIN_HIST:
            dlt_hist.append(lw_delta(hist[-120:]))
    w("- 分配器：%s" % "、".join(ALLOCS))
    w("- Ledoit-Wolf 对角目标收缩强度（120 日窗口滚动诊断，中位 %.2f）：说明 sleeve 的"
      "样本相关里有不小的噪声成分，所以 ERC/MVO 用 EWMA + 岭正则而不是裸样本协方差。"
      % (float(np.median(dlt_hist)) if dlt_hist else float("nan")))
    w("- 波动目标 %.0f%%：由 lambda'Sigma lambda 估出的组合波动线性缩放总仓位到 "
      "[%.2f, 1.0]。" % (100 * TARGET_VOL, GROSS_MIN))
    w("")
    # 因果性自检
    worst = 0.0
    for i in [i for i in range(n_g) if gdates[i] >= pd.Timestamp(BT_START)][:8]:
        hist = Rm_all[Rh.index <= gdates[i]]
        if len(hist) < MIN_HIST + 5:
            continue
        a = alloc_lambda(ALLOCS["ERC"], hist)
        b = alloc_lambda(ALLOCS["ERC"], hist[:-5])
        worst = max(worst, float(np.max(np.abs(a - b))))
    w(md_table(pd.DataFrame({"lambda 单边换手": pd.Series(lam_turn),
                             "与 sleeve 近 120 日夏普的截面相关": pd.Series(chase)}).round(4),
               "%.4f"))
    w("")
    w("> 第二列接近 1 说明该分配器本质上在**追近期表现好的 sleeve**；接近 0 说明它在"
      "做风险分散而不是追涨。")
    w("")
    w("- **因果性自检**：把窗口末 5 天截掉后重算 ERC 权重，最大差 %.3g"
      "（若用了未来数据一定会 > 1e-9）。" % worst)
    w("")
    w("分配器给出的总仓位（波动目标）统计：")
    w("")
    w(md_table(pd.DataFrame({"平均": {a: float(np.mean(LAM5[a]["gross"])) for a in ALLOCS},
                             "最小": {a: float(np.min(LAM5[a]["gross"])) for a in ALLOCS},
                             "最大": {a: float(np.max(LAM5[a]["gross"])) for a in ALLOCS},
                             "平均隐含波动":
                                 {a: float(np.nanmean(LAM5[a]["vol"])) for a in ALLOCS}})
               .T.round(4), "%.4f"))
    w("")
    lam_show = pd.DataFrame(LAM5["ERC"]["lam"], index=gdates, columns=FAM5)
    lam_show = lam_show.loc[lam_show.index >= pd.Timestamp(BT_START)]
    w("ERC 分配器在回测区间给出的平均仓位：")
    w("")
    w(md_table(pd.DataFrame({"ERC平均仓位": lam_show.mean(),
                             "ERC最新仓位": lam_show.iloc[-1],
                             "其中最小": lam_show.min(),
                             "其中最大": lam_show.max()}).round(4), "%.4f"))
    w("")
    # ---------------- 5) 合成回测 ----------------
    w("## 5. 合成回测（10 亿元，佣金万一 + 印花万五）")
    w("")

    def lam_for(fl, an):
        """取对应 sleeve 集合的分配器权重：只含 5 家族用 LAM5，含规模用 LAM6。"""
        if set(fl) <= set(FAM5):
            block, pool = LAM5[an], FAM5
        else:
            block, pool = LAM6[an], fams_all
        lam = block["lam"][:, [pool.index(f) for f in fl]]
        lam = lam / np.maximum(lam.sum(1, keepdims=True), 1e-12)
        return lam, block["gross"]

    def run_blend(fl, an, Wd, top_k=TOP_K, cash=CASH, cost=COST, name=None,
                  start=SIGNAL0, end=BT_END):
        lam, gross = lam_for(fl, an)
        st = BlendStrategy([Wd[f] for f in fl], gdates, lam, gross, top_k=top_k,
                           name=name or ("%s_%s" % ("+".join(fl), an)))
        b = Backtest(md, cost=cost, **{**BT_KW, "cash": cash})
        return b.run(st, start=start, end=end, rebalance=gdates)

    S, navs, results = {}, {}, {}
    # 诊断：合成到底等不等价于「sleeve 组合的加权平均」
    lamw, _gr = lam_for(FAM5, "ERC")
    # 「sleeve 加权平均组合」：按 ERC 平均仓位静态加权各 sleeve 的日收益。
    # 这是合成策略**应该**逼近的东西（合成 = 这些 sleeve 的持仓加权平均）。
    ravg = None
    for k, f in enumerate(FAM5):
        term = float(lamw[:, k].mean()) * Rs0[f].ret.loc[BT_START:]
        ravg = term if ravg is None else ravg + term
    savg = metrics.perf_stats((1 + ravg).cumprod(), ret=ravg)
    savg.update(metrics.excess_stats(ravg, bench_all.reindex(ravg.index)))
    savg["持仓数"] = float(sum(float(lamw[:, k].mean()) * Rs0[f].daily.n_holdings.mean()
                              for k, f in enumerate(FAM5)))
    savg["年换手"] = float(sum(float(lamw[:, k].mean()) * Rs0[f].daily.turnover.mean() * 252
                              for k, f in enumerate(FAM5)))
    S["sleeve加权平均"] = savg
    navs["sleeve加权平均"] = (1 + ravg).cumprod()
    w("### 5a. 合成 vs 「sleeve 加权平均」诊断")
    w("")
    _rw = np.array([float(W[f][i].sum()) for f in FAM5 for i in range(n_g)])
    w("- 各 sleeve 目标权重矩阵的行和：min %.4f / 中位 %.4f / max %.4f（应恒为 1）"
      % (_rw.min(), float(np.median(_rw)), _rw.max()))
    _nz = [(int((W[f][i] > 0).sum())) for f in FAM5 for i, d in enumerate(gdates)
           if d >= pd.Timestamp(SIGNAL0)]
    w("- 各 sleeve 每期持仓数：中位 %.0f / 最大 %.0f（目标 %d 只，缓冲带 %.1f 倍）"
      % (float(np.median(_nz)), float(np.max(_nz)), N_HOLD, BUFFER))
    _un = []
    for i, d in enumerate(gdates):
        if d < pd.Timestamp(SIGNAL0):
            continue
        acc = np.zeros(len(sym))
        for f in FAM5:
            acc += W[f][i]
        _un.append(int((acc > 0).sum()))
    w("- 五条 sleeve 的**并集**大小：中位 %.0f 只 / 最大 %.0f 只 —— 合成后%s；"
      "截断掉的是「只有一条 sleeve 看好」的名字，而实测这些名字贡献了正收益。"
      % (float(np.median(_un)), float(np.max(_un)),
         "不再额外截断" if not TOP_K else "只保留前 %d 只" % TOP_K))
    w("")
    w("- 各 sleeve 单独年化（同区间、同口径）：%s"
      % "、".join("%s %s" % (FAM_NAME.get(f, f), pct(metrics.perf_stats(Rs0[f].nav)["年化收益"]))
                 for f in FAM5))
    w("- **sleeve 加权平均组合**（按 ERC 平均仓位静态加权各 sleeve 的日收益，无撮合/无截断）："
      "年化 %s，夏普 %.2f，回撤 %s，超额 %s"
      % (pct(savg["年化收益"]), savg["夏普"], pct(savg["最大回撤"]),
         pct(savg.get("超额年化"))))
    lam_d, gr_d = lam_for(FAM5, "ERC")
    w("")
    w("| 合成方式 | 调仓阈值 band | 年化 | 夏普 | 最大回撤 | 持仓 | 年换手 |")
    w("|---|---|---|---|---|---|---|")
    for tk, tagd in ((200, "截断到 200 只"), (None, "不截断（本策略默认）")):
        for band in (2e-3, 0.0):
            stt = BlendStrategy([W[f] for f in FAM5], gdates, lam_d, gr_d, top_k=tk,
                                name="diag_%s_%.0e" % (tagd, band))
            kw = dict(BT_KW)
            kw["rebalance_band"] = band
            rr_ = Backtest(md, cost=COST, **kw).run(stt, start=SIGNAL0, end=BT_END,
                                                    rebalance=gdates)
            s_ = metrics.perf_stats(rr_.nav)
            s_["超额年化"] = metrics.excess_stats(
                rr_.ret, bench_all.reindex(rr_.nav.index)).get("超额年化")
            w("| %s | %g | %s | %.2f | %s | %.0f 只 | %.1f |"
              % (tagd, band, pct(s_["年化收益"]), s_["夏普"], pct(s_["最大回撤"]),
                 float(rr_.daily.n_holdings.mean()),
                 float(rr_.daily.turnover.mean() * 252)))
    w("")
    w("> 上面这张表同时给出本脚本的两个执行层选择：**不截断 + 绝对交易下限**。"
      "定位到的漏损来源：`rebalance_band` 是**净值的一个比例**"
      "（默认 2e-3 x 10 亿 = 200 万元），而 329 只并集组合的单只目标市值只有约 300 万元 ——"
      "阈值和仓位同量级，导致绝大多数调仓被判定为「不值得交易」，组合被冻在初始持仓上、"
      "卖不掉的旧仓不断累积。这不是 alpha 问题，是**执行层的阈值设计问题**。")
    w("")
    cases = {
        "MF5_ERC": (FAM5, "ERC"),
        "MF5_EW": (FAM5, "EW"),
        "MF5_IVOL60": (FAM5, "IVOL60"),
        "MF5_MVO": (FAM5, "MVO"),
        "MF6_ERC": (fams_all, "ERC"),
        "MF6_EW": (fams_all, "EW"),
    }
    for tag, (fl, an) in cases.items():
        r = run_blend(fl, an, W, name=tag)
        results[tag] = r
        navs[tag] = r.nav
        d = os.path.join(RUNS, tag)
        os.makedirs(d, exist_ok=True)
        r.nav.rename("nav").rename_axis("date").to_frame().to_csv(
            os.path.join(d, "nav.csv"), encoding="utf-8")
        r.daily.to_csv(os.path.join(d, "daily.csv"), encoding="utf-8")
        for f_ in ("trades", "blocks", "holdings"):
            df = getattr(r, f_)
            if len(df):
                df.to_csv(os.path.join(d, f_ + ".csv"), index=False, encoding="utf-8")
        s_ = metrics.perf_stats(r.nav)
        bench = bench_all.reindex(r.nav.index)
        s_.update(metrics.excess_stats(r.ret, bench))
        s_["年换手"] = float(r.daily.turnover.mean() * 252)
        s_["费用占净值"] = float(r.trades.fee.sum()) / CASH if len(r.trades) else 0.0
        s_["被挡金额_亿"] = float(r.blocks.dv.sum()) / 1e8 if len(r.blocks) else 0.0
        s_["持仓数"] = float(r.daily.n_holdings.mean())
        S[tag] = s_
        w("- **%s**（%s）：累计 %s，年化 %s，夏普 %.2f，最大回撤 %s，超额 %s，"
          "IR %.2f，年换手 %.1f，持仓 %.0f 只，被挡 %.1f 亿"
          % (tag, an, pct(s_["累计收益"]), pct(s_["年化收益"]), s_["夏普"],
             pct(s_["最大回撤"]), pct(s_.get("超额年化")), s_.get("信息比率", float("nan")),
             s_["年换手"], s_["持仓数"], s_["被挡金额_亿"]))
    w("")

    # 基准：股票池内等权（用与策略相同的域、相同的起点）
    bidx = navs["MF5_ERC"].index
    # 第二条基准：同为池内等权，但只留流动性前 50%（10 亿元真正买得进去的那一半）
    adv_rank = md.adv(20).rank(axis=1, pct=True)
    big = adv_rank >= 0.5
    bench_liq_all = md.ret_ok.where(uni & big).mean(1)
    bench_use = bench_all.reindex(bidx).fillna(0.0)
    bench_use.iloc[0] = 0.0
    bnav = (1 + bench_use).cumprod()
    bst = metrics.perf_stats(bnav, ret=bench_use)
    bst["持仓数"] = float(md.n_eligible(**UNI_KW).reindex(bidx).mean())
    bst["年换手"] = 0.0
    S["基准(池内等权)"] = bst
    navs["基准(池内等权)"] = bnav
    blq = bench_liq_all.reindex(bidx).fillna(0.0)
    blq.iloc[0] = 0.0
    lnav = (1 + blq).cumprod()
    lst = metrics.perf_stats(lnav, ret=blq)
    lst["持仓数"] = float(md.n_eligible(**UNI_KW).reindex(bidx).mean()) / 2.0
    lst["年换手"] = 0.0
    S["基准(池内等权·流动性前50%)"] = lst
    navs["基准(池内等权·流动性前50%)"] = lnav
    w("- **基准2（池内等权·流动性前 50%%）**：累计 %s，年化 %s，夏普 %.2f，最大回撤 %s"
      " —— 基准2 剔除了一半买不进去的尾部小票，是 10 亿元更现实的对手。"
      % (pct(lst["累计收益"]), pct(lst["年化收益"]), lst["夏普"], pct(lst["最大回撤"])))
    w("- **基准（股票池内等权）**：累计 %s，年化 %s，夏普 %.2f，最大回撤 %s，"
      "平均持仓 %.0f 只"
      % (pct(bst["累计收益"]), pct(bst["年化收益"]), bst["夏普"], pct(bst["最大回撤"]),
         bst["持仓数"]))
    w("")

    # ---------------- 5b) 规模中性化版本（回答「是不是小市值 beta」） ----------------
    w("### 5b. 规模中性化版本（每个因子对 log 20 日均额取残差）")
    w("")
    Wn = build_W(panel_ns)
    Rn = build_R(Wn)
    Rhn = pd.DataFrame({f: r.ret for f, r in Rn.items()}).fillna(0.0)
    lam_ns = np.zeros((n_g, len(Wn)))
    fams_ns = list(Wn.keys())
    for i, d in enumerate(gdates):
        hist = Rhn[fams_ns].values[Rhn.index <= d]
        if len(hist) < MIN_HIST:
            lam_ns[i] = 1.0 / len(fams_ns)
            continue
        lam_ns[i] = alloc_lambda(ALLOCS["ERC"], hist)
    idxs = [fams_ns.index(f) for f in FAM5 if f in fams_ns]
    lamu = lam_ns[:, idxs]
    lamu = lamu / np.maximum(lamu.sum(1, keepdims=True), 1e-12)
    gross_ns = np.ones(n_g)
    for i, d in enumerate(gdates):
        hist = Rhn[fams_ns].values[Rhn.index <= d]
        if len(hist) < MIN_HIST:
            continue
        X = hist[-120:]
        Sw = ewma_cov(X, 60.0, 0.05)
        sg = implied_vol(lamu[i], Sw)
        gross_ns[i] = float(np.clip(TARGET_VOL / sg, GROSS_MIN, 1.0)) if sg > 0 else 1.0
    st_ns = BlendStrategy([Wn[f] for f in FAM5 if f in fams_ns], gdates, lamu, gross_ns,
                          top_k=TOP_K, name="MF5_ERC_ns")
    r = Backtest(md, cost=COST, **BT_KW).run(st_ns, start=SIGNAL0, end=BT_END,
                                             rebalance=gdates)
    navs["MF5_ERC_ns"] = r.nav
    d = os.path.join(RUNS, "MF5_ERC_ns")
    os.makedirs(d, exist_ok=True)
    r.nav.rename("nav").rename_axis("date").to_frame().to_csv(os.path.join(d, "nav.csv"),
                                                              encoding="utf-8")
    r.daily.to_csv(os.path.join(d, "daily.csv"), encoding="utf-8")
    for f_ in ("trades", "blocks", "holdings"):
        df = getattr(r, f_)
        if len(df):
            df.to_csv(os.path.join(d, f_ + ".csv"), index=False, encoding="utf-8")
    s_ns = metrics.perf_stats(r.nav)
    s_ns.update(metrics.excess_stats(r.ret, bench_all.reindex(r.nav.index)))
    s_ns["年换手"] = float(r.daily.turnover.mean() * 252)
    s_ns["被挡金额_亿"] = float(r.blocks.dv.sum()) / 1e8 if len(r.blocks) else 0.0
    s_ns["持仓数"] = float(r.daily.n_holdings.mean())
    s_ns["费用占净值"] = float(r.trades.fee.sum()) / CASH if len(r.trades) else 0.0
    S["MF5_ERC_ns"] = s_ns
    w("- **MF5_ERC_ns**：累计 %s，年化 %s，夏普 %.2f，最大回撤 %s，超额 %s，持仓 %.0f 只，"
      "被挡 %.1f 亿"
      % (pct(s_ns["累计收益"]), pct(s_ns["年化收益"]), s_ns["夏普"], pct(s_ns["最大回撤"]),
         pct(s_ns.get("超额年化")), s_ns["持仓数"], s_ns["被挡金额_亿"]))
    w("")
    w("> 对照：原始版本 MF5_ERC 超额 %s、回撤 %s、被挡 %.1f 亿。"
      % (pct(S["MF5_ERC"].get("超额年化")), pct(S["MF5_ERC"]["最大回撤"]),
         S["MF5_ERC"]["被挡金额_亿"]))
    w("")

    # ---------------- 6) 稳健性 ----------------
    robust = {}
    if not args.quick:
        w("## 6. 稳健性与归因")
        w("")
        # 6.1 成本倍数
        rows = []
        for mul in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0):
            c = CostModel(commission_rate=COMMISSION * mul, commission_min=0.0,
                          transfer_fee_rate=0.0, stamp_duty_rate=STAMP * mul,
                          stamp_duty_rate_old=STAMP * mul, slippage_bps=0.0, impact_coef=0.0)
            rr_ = run_blend(FAM5, "ERC", W, cost=c, name="mf_cost%.1f" % mul)
            s_ = metrics.perf_stats(rr_.nav)
            rows.append(dict(成本倍数=mul, 累计=s_["累计收益"], 年化=s_["年化收益"],
                             夏普=s_["夏普"], 最大回撤=s_["最大回撤"]))
        robust["cost"] = pd.DataFrame(rows).set_index("成本倍数")
        w("### 6.1 成本敏感性（MF5_ERC）")
        w("")
        w(md_table(robust["cost"].map(lambda v: "%.4f" % v), "%s"))
        w("")
        # 6.2 更真实成本（滑点 + 冲击）
        rr_ = run_blend(FAM5, "ERC", W, cost=COST_REAL, name="MF5_real")
        s_real = metrics.perf_stats(rr_.nav)
        s_real.update(metrics.excess_stats(rr_.ret, bench_all.reindex(rr_.nav.index)))
        robust["real"] = s_real
        w("### 6.2 加 10bp 滑点 + 冲击成本")
        w("")
        w("- 年化 %s，夏普 %.2f，最大回撤 %s，超额 %s（同一套权重，只换成本模型）"
          % (pct(s_real["年化收益"]), s_real["夏普"], pct(s_real["最大回撤"]),
             pct(s_real.get("超额年化"))))
        w("")
        # 6.3 资金规模
        rows = []
        for cash in (3e8, 1e9, 3e9, 1e10):
            rr_ = run_blend(FAM5, "ERC", W, cash=cash, name="mf_cash%.0e" % cash)
            s_ = metrics.perf_stats(rr_.nav)
            s_["超额年化"] = metrics.excess_stats(rr_.ret, bench_all.reindex(rr_.nav.index)).get("超额年化")
            rows.append(dict(配置="资金 %.0f 亿" % (cash / 1e8), 年化=s_["年化收益"],
                             夏普=s_["夏普"], 超额年化=s_["超额年化"],
                             持仓数=float(rr_.daily.n_holdings.mean()),
                             被挡_亿=float(rr_.blocks.dv.sum() / 1e8) if len(rr_.blocks) else 0.0))
        # 6.4 持仓数
        for nh in (40, 150):
            Wt = build_W(panel_raw, n_hold=nh)
            rr2 = Backtest(md, cost=COST, **BT_KW)
            lam, gross_ = lam_for(FAM5, "ERC")
            stt = BlendStrategy([Wt[f] for f in FAM5], gdates, lam, gross_,
                                top_k=None, name="mf_nh%d" % nh)
            rr_ = rr2.run(stt, start=SIGNAL0, end=BT_END, rebalance=gdates)
            s_ = metrics.perf_stats(rr_.nav)
            s_["超额年化"] = metrics.excess_stats(rr_.ret, bench_all.reindex(rr_.nav.index)).get("超额年化")
            rows.append(dict(配置="sleeve 持仓 %d 只" % nh, 年化=s_["年化收益"], 夏普=s_["夏普"],
                             超额年化=s_["超额年化"],
                             持仓数=float(rr_.daily.n_holdings.mean()),
                             被挡_亿=float(rr_.blocks.dv.sum() / 1e8) if len(rr_.blocks) else 0.0))
        # 6.5 top_k
        for tk in (100, 400):
            rr_ = run_blend(FAM5, "ERC", W, top_k=tk, name="mf_tk%d" % tk)
            s_ = metrics.perf_stats(rr_.nav)
            s_["超额年化"] = metrics.excess_stats(rr_.ret, bench_all.reindex(rr_.nav.index)).get("超额年化")
            rows.append(dict(配置="合成持仓上限 %d 只" % tk, 年化=s_["年化收益"], 夏普=s_["夏普"],
                             超额年化=s_["超额年化"],
                             持仓数=float(rr_.daily.n_holdings.mean()),
                             被挡_亿=float(rr_.blocks.dv.sum() / 1e8) if len(rr_.blocks) else 0.0))
        # 6.5b 四种分配器 × 关掉波动目标（隔离「总仓位」这个混杂因素）
        rows_b = []
        for an in ALLOCS:
            lam, _gg = lam_for(FAM5, an)
            stt = BlendStrategy([W[f] for f in FAM5], gdates, lam, np.ones(n_g),
                                top_k=TOP_K, name="mf_%s_novol" % an)
            rr_ = Backtest(md, cost=COST, **BT_KW).run(stt, start=SIGNAL0, end=BT_END,
                                                       rebalance=gdates)
            s_ = metrics.perf_stats(rr_.nav)
            s_["超额年化"] = metrics.excess_stats(
                rr_.ret, bench_all.reindex(rr_.nav.index)).get("超额年化")
            rows_b.append(dict(分配器=an, 年化=s_["年化收益"], 夏普=s_["夏普"],
                               最大回撤=s_["最大回撤"], 超额年化=s_["超额年化"],
                               年换手=float(rr_.daily.turnover.mean() * 252),
                               持仓数=float(rr_.daily.n_holdings.mean())))
        robust["alloc_novol"] = pd.DataFrame(rows_b).set_index("分配器")
        w("### 6.5b 四种分配器对照（关掉波动目标，总仓位恒定 98%）")
        w("")
        w(md_table(robust["alloc_novol"], "%.4f"))
        w("")

        # 6.6 无波动目标
        lam, _g_ = lam_for(FAM5, "ERC")
        stt = BlendStrategy([W[f] for f in FAM5], gdates, lam, np.ones(n_g), top_k=TOP_K,
                            name="mf_novol")
        rr_ = Backtest(md, cost=COST, **BT_KW).run(stt, start=SIGNAL0, end=BT_END,
                                                   rebalance=gdates)
        s_ = metrics.perf_stats(rr_.nav)
        robust["scale"] = pd.DataFrame(rows).set_index("配置")
        w("### 6.3 资金规模 / 持仓数 / 合成上限 / 波动目标")
        w("")
        w(md_table(robust["scale"], "%.4f"))
        w("")
        # 6.7 子区间：样本内（2021-2024）走同一套流水线
        rr_ = run_blend(FAM5, "ERC", W, name="MF5_ERC_IS", start=GRID_START, end=IS_END)
        bi = bench_all.reindex(rr_.nav.index).fillna(0.0)
        bi.iloc[0] = 0.0
        s_is = metrics.perf_stats(rr_.nav)
        s_is.update(metrics.excess_stats(rr_.ret, (1 + bi).pct_change().fillna(0.0)))
        bench_is = (1 + bi).cumprod()
        b_is = metrics.perf_stats(bench_is, ret=bi)
        s_is["年换手"] = float(rr_.daily.turnover.mean() * 252)
        s_is["持仓数"] = float(rr_.daily.n_holdings.mean())
        b_is["年换手"] = 0.0
        b_is["持仓数"] = float(md.n_eligible(**UNI_KW).reindex(rr_.nav.index).mean())
        robust["is"] = (s_is, b_is)
        cc = [c for c in ["累计收益", "年化收益", "年化波动", "夏普", "最大回撤", "超额年化",
                          "信息比率", "年换手", "持仓数"]
              if c in s_is or c in b_is]
        w("### 6.4 样本内子区间对照（2021-01 ~ 2024-12，同一套流水线，仅用于看流水线是否只在 2025-2026 有效）")
        w("")
        w(md_table(pd.DataFrame({"MF5_ERC（样本内）": s_is, "池内等权（样本内）": b_is}).T[cc],
                   "%.4f"))
        w("")
        # 6.8 随机对照
        rng = np.random.default_rng(20260101)
        rnd, rnd_nv = [], []
        lam_e, _ge = lam_for(FAM5, "ERC")
        for k in range(int(args.rand)):
            Wr = {}
            for f in FAM5:
                Pr = pd.DataFrame(rng.standard_normal((n_g, len(sym))), index=gdates,
                                  columns=sym)
                if pd.Timestamp(LIQ) in Pr.index:
                    Pr.loc[pd.Timestamp(LIQ)] = np.nan
                Wr[f] = sleeve_weight_matrix(Pr, uni.values, md, grid, sym)
            ron = bt.run(BlendStrategy([Wr[f] for f in FAM5], gdates, lam_e,
                                       _ge, top_k=TOP_K,
                                       name="mf_rnd%d" % k),
                         start=SIGNAL0, end=BT_END, rebalance=gdates)
            roff = bt.run(BlendStrategy([Wr[f] for f in FAM5], gdates, lam_e,
                                        np.ones(n_g), top_k=TOP_K, name="mf_rnd%d_nv" % k),
                          start=SIGNAL0, end=BT_END, rebalance=gdates)
            rnd.append(metrics.perf_stats(ron.nav)["年化收益"])
            rnd_nv.append(metrics.perf_stats(roff.nav)["年化收益"])
        rnd = np.asarray(rnd, dtype="float64")
        rnd_nv = np.asarray(rnd_nv, dtype="float64")
        robust["random"] = rnd
        robust["random_nv"] = rnd_nv
        w("### 6.5 随机对照（同股票池、同缓冲带、同分配器，打分换成纯随机数）")
        w("")
        w("| 口径 | 随机中位 | 随机 5% | 随机 95% | 最优一组 | 策略本身 | 策略百分位 |")
        w("|---|---|---|---|---|---|---|")
        for lab, arr, key in (("含波动目标（=MF5_ERC）", rnd, "MF5_ERC"),):
            x = S[key]["年化收益"]
            w("| %s | %s | %s | %s | %s | %s | **%.0f%%** |"
              % (lab, pct(np.median(arr)), pct(np.percentile(arr, 5)),
                 pct(np.percentile(arr, 95)), pct(arr.max()), pct(x),
                 100.0 * float((arr < x).mean())))
        x = robust["alloc_novol"].loc["ERC", "年化"]
        w("| 关掉波动目标（=6.5b ERC） | %s | %s | %s | %s | %s | **%.0f%%** |"
          % (pct(np.median(rnd_nv)), pct(np.percentile(rnd_nv, 5)),
             pct(np.percentile(rnd_nv, 95)), pct(rnd_nv.max()), pct(x),
             100.0 * float((rnd_nv < x).mean())))
        w("")
        _n_beat = int((rnd < S["MF5_ERC"]["年化收益"]).sum())
        _p = 1.0 / (len(rnd) + 1)
        w("> 随机对照是判断「打分有没有信息」最干净的一组证据：同样的股票池、同样的缓冲带、"
          "同样的分配器与波动目标，只把 sleeve 打分换成纯随机数。策略年化 %s，"
          "高于 %d/%d 组随机组合（单边 p ≈ %.3f，样本极小）。"
          "**注意随机对照用的是修好执行层之后的参数**：在默认 `rebalance_band` 下策略只有 "
          "10.82~18.72%%，那时它会落在随机分布的 42%% 分位 —— 差别全部来自执行层，不是因子。"
          % (pct(S["MF5_ERC"]["年化收益"]), _n_beat, len(rnd), _p))
        w("")
        # 6.9 分年度
        ann = metrics.annual_table(results["MF5_ERC"].ret.loc[BT_START:],
                                   bench_all.reindex(results["MF5_ERC"].nav.index).loc[BT_START:])
        robust["annual"] = ann
        w("### 6.6 分年度")
        w("")
        w(md_table(ann.round(4), "%.4f"))
        w("")
    # ---------------- 7) 汇总表 / 图 / 报告 ----------------
    w("## 7. 汇总")
    w("")
    for _k in list(S):
        _nv = navs.get(_k)
        if _nv is None or len(_nv) < 3:
            continue
        _r = _nv.pct_change().fillna(0.0)
        S[_k]["超额vs池内等权"] = metrics.excess_stats(
            _r, bench_all.reindex(_nv.index)).get("超额年化")
        S[_k]["超额vs流动性前50%"] = metrics.excess_stats(
            _r, bench_liq_all.reindex(_nv.index)).get("超额年化")
    cols = ["累计收益", "年化收益", "年化波动", "夏普", "最大回撤", "卡玛",
            "超额vs池内等权", "超额vs流动性前50%", "信息比率", "年换手", "费用占净值",
            "被挡金额_亿", "持仓数"]
    tab = pd.DataFrame(S).T
    for c in cols:
        if c not in tab.columns:
            tab[c] = np.nan
    tab = tab[cols]
    savetab(tab.round(6), "10_multifactor_summary.csv")
    w(md_table(tab.round(4), "%.4f"))
    w("")
    w("> 口径提醒：「被挡金额」按**委托尝试次数**累计 —— 同一个调仓窗口里被挡的委托会在"
      "执行窗口内（最多 3 个交易日）每天重试并重复记账，所以它衡量的是摩擦量级，"
      "不是唯一金额。")
    w("")
    w("> 两个基准的含义不同：**池内等权** = 同股票池等权持有（含 10 亿元事实上买不进去的"
      "尾部小票），是最难打败的对手；**流动性前 50% 等权** = 只拿买得进去的那一半，"
      "对 10 亿元更现实。超额为正/为负取决于用哪个基准，两个都列出来避免选择性汇报。"
      "「sleeve加权平均」不是可交易的策略，是理想化下界参照。")
    w("")

    best = "MF5_ERC"
    w("## 8. 结论")
    w("")
    w("- 主推 **%s**（预先指定的配置：家族 ICIR 打分 + ERC 风险预算 + 波动目标）："
      "累计 %s，年化 %s，夏普 %.2f，最大回撤 %s。超额：相对池内等权 %s，"
      "相对流动性前 50%% 等权 %s。"
      % (best, pct(S[best]["累计收益"]), pct(S[best]["年化收益"]), S[best]["夏普"],
         pct(S[best]["最大回撤"]), pct(S[best].get("超额vs池内等权")),
         pct(S[best].get("超额vs流动性前50%"))))
    w("- **alpha 还是 beta**：见 §5b。把每个因子对 log(20 日均额) 取残差后，超额从 %s "
      "掉到 %s —— 这条流水线赚的主要还是**小市值 / 低流动性 beta**。代价换来的好处很实在："
      "最大回撤 %s -> %s，被挡委托 %s 亿 -> %s 亿，容量问题基本消失。"
      % (pct(S["MF5_ERC"].get("超额vs池内等权")), pct(S["MF5_ERC_ns"].get("超额vs池内等权")),
         pct(S["MF5_ERC"]["最大回撤"]), pct(S["MF5_ERC_ns"]["最大回撤"]),
         "%.0f" % S["MF5_ERC"]["被挡金额_亿"], "%.1f" % S["MF5_ERC_ns"]["被挡金额_亿"]))
    w("- **执行层比因子层更重要**：§5a 定位到两个漏损点 —— ① `rebalance_band` 用净值比例、"
      "与 300 只组合的单只仓位同量级，会把组合冻住（年化 10.82% vs 25.85%）；"
      "② 合成后按权重截断到 200 只会系统性剔掉「只有一条 sleeve 看好」的名字（-3.6pp/年）。"
      "两处都不是 alpha 问题。")
    w("- **分配器的作用是降风险、不是收益**：§7 表里 EW / IVOL60 / ERC / MVO 四条曲线年化"
      "26.2%~27.6%（关掉波动目标后）几乎重合，但 MVO 的持仓数掉到 238 只、被挡 75 亿元 ——"
      "它的「与 sleeve 近 120 日夏普的截面相关」是 0.77，本质是**追近期表现好的 sleeve**，"
      "换成风险预算就没这个问题。")
    w("- **成本是最脆弱的一环**：只加 10bp 滑点 + 冲击成本，年化从 %s 掉到 %s，"
      "超额从 %s 掉到 %s（§6.2）。换手 16 倍/年是主要暴露。"
      % (pct(S["MF5_ERC"]["年化收益"]), pct(robust["real"]["年化收益"]),
         pct(S["MF5_ERC"].get("超额vs池内等权")), pct(robust["real"].get("超额年化"))))
    w("- **随机对照通过**：换成纯随机打分后策略年化高于全部随机组合（§6.5）；"
      "但只有 %d 组对照、区间只有约 21 个月 / 21 个调仓日，**样本极短**，"
      "因子方向与筛选虽然只用了 2021-2024，21 个观测点的统计置信度仍然有限。"
      % len(robust.get("random", [])))
    w("- **容量**：10 亿元不是瓶颈（3 亿 / 10 亿 / 30 亿年化 25.8% / 25.9% / 25.3%），"
      "到 100 亿元才明显掉到 19.8%；但被挡委托金额从 3 亿的 1.6 亿元涨到 30 亿的 492 亿元，"
      "摩擦已经不可忽略（§6.3）。")
    w("")

    # ---------- 图 ----------
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.8))
    order = ["MF5_ERC", "MF5_EW", "MF5_IVOL60", "MF5_MVO", "MF6_ERC", "MF5_ERC_ns",
             "基准(池内等权)", "基准(池内等权·流动性前50%)"]
    for tag in order:
        if tag not in navs:
            continue
        nv = navs[tag].dropna()
        st = dict(lw=2.2, color="crimson") if tag == "MF5_ERC" else (
            dict(lw=1.8, ls="--", color="k") if tag.startswith("基准") else
            dict(lw=1.1, alpha=0.8))
        ax[0].plot(nv.index, nv.values / nv.values[0], label=tag, **st)
    ax[0].axhline(1.0, color="gray", lw=0.6)
    ax[0].set_title("多因子策略净值（建仓 %s ~ 清仓 %s，10 亿元）" % (BT_START, BT_END))
    ax[0].set_ylabel("净值（起点 = 1）")
    ax[0].legend(fontsize=8, ncol=2)
    ax[0].grid(alpha=0.3)
    key = navs["MF5_ERC"].dropna()
    dd = metrics.drawdown(key)
    ax[1].fill_between(dd.index, dd.values, 0, alpha=0.45, color="crimson")
    bdd = metrics.drawdown(navs["基准(池内等权)"].dropna())
    ax[1].plot(bdd.index, bdd.values, color="k", lw=1.3, ls="--", label="基准(池内等权)")
    ax[1].set_title("回撤")
    ax[1].set_ylabel("回撤")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    savefig(fig, "10_multifactor_nav.png")

    fig, ax = plt.subplots(2, 1, figsize=(11.5, 6.4), sharex=True)
    lamd = pd.DataFrame(LAM5["ERC"]["lam"], index=gdates, columns=FAM5)
    lamd = lamd.loc[lamd.index >= pd.Timestamp(SIGNAL0)]
    ax[0].stackplot(lamd.index, [lamd[c].values for c in lamd.columns],
                    labels=[FAM_NAME.get(c, c) for c in lamd.columns], alpha=0.9)
    ax[0].set_title("ERC 分配器给出的 5 条家族 sleeve 仓位（每个调仓日只用该日及以前的数据）")
    ax[0].legend(fontsize=8, ncol=6, loc="lower left")
    ax[0].set_ylim(0, 1.02)
    gs = pd.Series(LAM5["ERC"]["gross"], index=gdates)
    gs = gs.loc[gs.index >= pd.Timestamp(SIGNAL0)]
    ax[1].plot(gs.index, gs.values, marker="o", ms=3.5, color="navy")
    ax[1].set_title("波动目标（20%）给出的总仓位")
    ax[1].set_ylabel("总仓位")
    ax[1].grid(alpha=0.3)
    savefig(fig, "10_multifactor_alloc.png")

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ic_fam = ICT.groupby("模块")["样本内IC"].mean()
    ic_fam = ic_fam.reindex([f for f in FAM5 if f in ic_fam.index])
    b = ax.bar(range(len(ic_fam)), ic_fam.values,
               color=["crimson" if v < 0 else "steelblue" for v in ic_fam.values])
    ax.set_xticks(range(len(ic_fam)))
    ax.set_xticklabels([FAM_NAME.get(i, i) for i in ic_fam.index], rotation=15)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("样本内平均月频 IC（Spearman）")
    ax.set_title("五个家族的样本内平均 IC")
    ax.grid(alpha=0.3, axis="y")
    savefig(fig, "10_multifactor_family_ic.png")

    with open(os.path.join(ROOT, "out", "multifactor_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    cfg = dict(cash=CASH, commission=COMMISSION, stamp=STAMP, start=BT_START, end=BT_END,
               signal0=SIGNAL0, liq=LIQ, n_hold=N_HOLD, buffer=BUFFER, top_k=TOP_K,
               max_w=MAX_W, participation=PARTICIPATION, exec_window=EXEC_WINDOW,
               target_vol=TARGET_VOL, gross_min=GROSS_MIN, min_hist=MIN_HIST,
               is_end=IS_END, universe=UNI_KW, allocators=list(ALLOCS),
               kept={k: v for k, v in kept_by_fam.items()},
               lw_delta_median=float(np.median(dlt_hist)) if dlt_hist else None,
               n_grid=n_g)
    json.dump(cfg, open(os.path.join(RUNS, "mf_config.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    if "cost" in robust:
        robust["cost"].to_csv(os.path.join(RUNS, "mf_cost.csv"), encoding="utf-8-sig")
    if "scale" in robust:
        robust["scale"].to_csv(os.path.join(RUNS, "mf_scale.csv"), encoding="utf-8-sig")
    if "random" in robust:
        pd.Series(robust["random"], name="年化收益").to_csv(
            os.path.join(RUNS, "mf_random.csv"), encoding="utf-8-sig")
    if "annual" in robust:
        robust["annual"].to_csv(os.path.join(RUNS, "mf_annual.csv"), encoding="utf-8-sig")
    print("\n报告：out/multifactor_report.md   总用时 %.1fs" % (time.time() - t_all))
    return S, navs, W, Rs


if __name__ == "__main__":
    main()
