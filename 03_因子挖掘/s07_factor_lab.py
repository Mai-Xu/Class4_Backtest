# -*- coding: utf-8 -*-
"""s07 因子挖矿实验室：把「挖因子」的全过程渲染成一张自包含的交互看板。

六道工序，对应看板的六幕：

    矿区   股票池规模、调仓月数、因子族构成
    原矿   13 个因子的定义卡（族 / 窗口 / 覆盖率 / IC 迷你走势）
    淘洗   逐月 IC 热力图、IC 衰减光谱、IC 分布
    闯关   方向一致性 -> |t| >= 1 的筛选漏斗（Sankey）、样本内/外四象限
    体检   因子相关性聚类热力图 + 因子星图 + 冗余对清单
    熔炼   单因子赛马、十分层回测、留一法边际贡献、合成净值与回撤
    成色   参数敏感性、随机对照、成本与容量

现场补算（其余全部复用 s04 / s05 的产物）：
    因子相关性、IC 衰减光谱、十分层回测（10 组真回测）、留一法边际贡献、覆盖率

产物
    out/factor_lab.html         自包含交互看板（plotly.js 内联，可离线打开）
    out/tables/07_*.csv         看板用到的全部中间表
    out/factor_lab_report.md    文字版小结

运行
    python s07_factor_lab.py          # 约 60 秒
    python s07_factor_lab.py --cdn    # 不内联 plotly.js（看板小 5 MB，需联网）
"""
from __future__ import annotations
# --- 项目根入 sys.path：core / factor_zoo 都是根目录下的包 ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- 产物归档到本工作的目录：out/03_因子挖掘/ ---
_os.environ.setdefault("CLASS4_TRACK", "03_因子挖掘")


import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline as po
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from core.lib import MarketData, ROOT, CACHE, TAB, savetab, RUNS, OUT
from core.engine import Backtest
from core.strategies import PanelStrategy, build_factor_panels
from core.factors import FACTORS, DESC
from core import metrics

FUC = os.path.join(CACHE, "factors")
#: s07 复用「工作②」的回测产物（vv_* 以及 04_/05_ 表），不重复跑回测。
#: 这是全项目唯一一处**跨工作读取**，所以路径显式指向 out/02_回测系统/，
#: 不跟着本工作的产物目录（out/03_因子挖掘/）走。
SRC = os.path.join(ROOT, "out", "02_回测系统")
SRC_RUNS = os.path.join(SRC, "runs")
SRC_TAB = os.path.join(SRC, "tables")

OUT_HTML = os.path.join(OUT, "factor_lab.html")
OUT_MD = os.path.join(OUT, "factor_lab_report.md")

# 因子族：与 s04 报告里的口径一致
VOL_FAMILY = ("ivol", "beta", "dvol", "maxdd", "skew", "kurt", "vol_trend", "vol_of_vol")
AMT_FAMILY = ("amt_trend", "amt_cv", "amihud", "trade_size", "pv_corr")
FAMILY = {f: "波动率族" for f in VOL_FAMILY}
FAMILY.update({f: "成交量族" for f in AMT_FAMILY})

# 因子窗口（取自 factors.py 的默认参数）：只用于卡片上的「配方」一行
WINDOW = {
    "ivol": "60 日回归残差", "beta": "60 日回归", "dvol": "60 日半偏差",
    "maxdd": "60 日区间", "skew": "60 日三阶矩", "kurt": "60 日四阶矩",
    "vol_trend": "20 / 60 日", "vol_of_vol": "120 日切 4 段",
    "amt_trend": "5 / 60 日", "amt_cv": "60 日", "amihud": "20 日",
    "trade_size": "20 日", "pv_corr": "60 日",
}
# 一句话「配方」：卡片上给读者看的公式直觉，不是严格定义
FORMULA = {
    "ivol": "sigma( r - (a + b*r_mkt) )",
    "beta": "r 对等权市场收益回归的斜率",
    "dvol": "只统计负收益的标准差",
    "maxdd": "1 - P_t / cummax(P)",
    "skew": "E[(r-mu)^3] / sigma^3",
    "kurt": "E[(r-mu)^4] / sigma^4 - 3",
    "vol_trend": "sigma_20 / sigma_60 - 1",
    "vol_of_vol": "std(分段波动率)",
    "amt_trend": "mean(amt)_5 / mean(amt)_60 - 1",
    "amt_cv": "std(amt) / mean(amt)",
    "amihud": "mean( |r| / amt ) x 1e8",
    "trade_size": "mean( log(amt / num_trades) )",
    "pv_corr": "corr( r, dlog(amt) )",
}

HORIZONS = (1, 3, 5, 10, 21, 42, 63)   # IC 衰减光谱的横轴（交易日）
FWD = 21                               # 主口径前瞻窗口
N_DEC = 10                             # 分层回测层数
N_RANDOM = 200                         # 随机对照组数（毛收益口径）

HG = dict(exec_price="open", max_participation=0.10, min_trade_notional=5e3,
          rebalance_band=2e-3, gross_limit=0.98, cash=1e8, exec_window=3)

PAL = dict(bg="#0a0e14", panel="#111721", line="#1f2733", grid="#232c3a",
           text="#e7edf5", mut="#8494a8", dim="#5b6a7d",
           vol="#38bdf8", amt="#f0b429", comp="#f472b6",
           up="#2fbf71", dn="#e5534b", gold="#e3b341", bench="#7d8b9e")

FONT = ('"Microsoft YaHei UI","PingFang SC","Hiragino Sans GB","Noto Sans SC",'
        'system-ui,sans-serif')


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def spearman(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    """横截面 Spearman 秩相关（与 s04 的 IC 口径完全一致）。"""
    ok = mask & np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 50:
        return np.nan
    a = pd.Series(x[ok]).rank().values
    b = pd.Series(y[ok]).rank().values
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def _masked_mean(x: np.ndarray, mask: np.ndarray) -> float:
    """掩码均值：没有有效样本时返回 NaN，不去触发 all-NaN 告警。"""
    ok = mask & np.isfinite(x)
    return float(x[ok].mean()) if ok.sum() else np.nan


def ic_series(panel: np.ndarray, fwd: np.ndarray, uni: np.ndarray) -> np.ndarray:
    """一个因子在全部调仓日上的 IC 序列。"""
    return np.array([spearman(panel[i], fwd[i], uni[i]) for i in range(len(panel))])


def fmt_pct(v, nd=2):
    return "—" if v is None or not np.isfinite(v) else "%.*f%%" % (nd, 100.0 * float(v))


def fmt_num(v, nd=2):
    return "—" if v is None or not np.isfinite(v) else "%.*f" % (nd, float(v))


def md_table(df: pd.DataFrame) -> str:
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
            elif isinstance(v, (float, np.floating)):
                vals.append("—" if not np.isfinite(v) else "%.4g" % v)
            else:
                vals.append(str(v))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([head, sep] + rows)


def style(fig, height=420, title=None, legend=True, legend_pos="top", margin=None, **kw):
    """统一深色皮肤：标题靠左、图例靠右；太挤的图把图例放到底部。"""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=PAL["text"]),
        height=height, margin=margin or dict(l=64, r=28, t=54 if title else 18, b=46),
        hoverlabel=dict(bgcolor="#151c28", bordercolor=PAL["line"], font_size=12),
        showlegend=legend, **kw)
    if legend:
        if legend_pos == "bottom":
            fig.update_layout(legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h",
                                          yanchor="top", y=-0.15, x=0, font=dict(size=11)))
        else:
            fig.update_layout(legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h",
                                          yanchor="bottom", y=1.03, x=1.0, xanchor="right",
                                          font=dict(size=11)))
    if title:
        fig.update_layout(title=dict(text=title, x=0.008, xanchor="left",
                                     font=dict(size=13, color=PAL["mut"])))
    fig.update_xaxes(gridcolor=PAL["grid"], zeroline=False, linecolor=PAL["line"],
                     tickfont=dict(size=11), title_font=dict(size=11, color=PAL["mut"]))
    fig.update_yaxes(gridcolor=PAL["grid"], zeroline=False, linecolor=PAL["line"],
                     tickfont=dict(size=11), title_font=dict(size=11, color=PAL["mut"]))
    return fig


def sparkline(values, color="#38bdf8", w=132, h=30):
    """因子卡上的迷你 IC 走势（纯 SVG）。"""
    v = np.asarray(values, dtype="float64")
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return ""
    x = np.linspace(0.0, w, len(v))
    lo, hi = float(v.min()), float(v.max())
    span = (hi - lo) or 1e-9
    y = h - (v - lo) / span * h
    pts = " ".join("%.1f,%.1f" % (a, b) for a, b in zip(x, y))
    zy = h - (0.0 - lo) / span * h
    return ('<svg class="spark" viewBox="0 0 %d %d" preserveAspectRatio="none">'
            '<line x1="0" y1="%.1f" x2="%d" y2="%.1f" stroke="%s" stroke-width="1" '
            'stroke-dasharray="2 2"/><polyline points="%s" fill="none" stroke="%s" '
            'stroke-width="1.5" stroke-linejoin="round"/></svg>'
            % (w, h, zy, w, zy, PAL["line"], pts, color))


# --------------------------------------------------------------------------- #
# 二、读入 s04 / s05 的产物
# --------------------------------------------------------------------------- #
def load_artifacts() -> dict:
    cfg = json.load(open(os.path.join(SRC_RUNS, "vv_config.json"), encoding="utf-8"))
    art = {"cfg": cfg}
    art["icstat"] = pd.read_csv(os.path.join(SRC_TAB, "04_factor_ic.csv"),
                                encoding="utf-8", index_col=0)
    ic = pd.read_csv(os.path.join(SRC_RUNS, "vv_ic.csv"), encoding="utf-8",
                     index_col=0, parse_dates=True)
    art["ic"] = ic[list(FACTORS)]
    art["single"] = pd.read_csv(os.path.join(SRC_TAB, "04_vol_strategy_summary.csv"),
                                encoding="utf-8", index_col=0)
    art["sens"] = pd.read_csv(os.path.join(SRC_TAB, "05_sensitivity.csv"),
                              encoding="utf-8", index_col=0)
    art["sub"] = pd.read_csv(os.path.join(SRC_TAB, "05_subperiod.csv"),
                             encoding="utf-8", index_col=0)
    art["comp_daily"] = pd.read_csv(os.path.join(SRC_RUNS, "vv_comp", "daily.csv"),
                                    encoding="utf-8", index_col=0, parse_dates=True)
    art["comp_nav"] = art["comp_daily"]["nav"]
    art["factor_nav"] = {
        nm: pd.read_csv(os.path.join(SRC_RUNS, "vv_" + nm, "nav.csv"), encoding="utf-8",
                        index_col=0, parse_dates=True)["nav"] for nm in FACTORS}
    return art


# --------------------------------------------------------------------------- #
# 三、现场补算：因子相关性 / 衰减 / 分层 / 边际贡献
# --------------------------------------------------------------------------- #
def factor_corr(zsc: dict, min_n: int = 50) -> pd.DataFrame:
    """逐日横截面 Spearman 相关，再按日平均 —— 衡量两个因子在**同一批股票上**
    是不是在说同一件事（比把整块面板堆起来算 corr 更贴近实盘用法）。"""
    names = list(FACTORS)
    k = len(names)
    n_date = len(zsc[names[0]])
    S = np.zeros((k, k))
    W = np.zeros((k, k))
    for r in range(n_date):
        A = np.vstack([np.asarray(zsc[nm].values[r], dtype="float64") for nm in names])
        R = pd.DataFrame(A).rank(axis=1, na_option="keep").values
        ok = np.isfinite(R)
        for i in range(k):
            for j in range(i, k):
                m = ok[i] & ok[j]
                if m.sum() < min_n:
                    continue
                a = R[i][m] - R[i][m].mean()
                b = R[j][m] - R[j][m].mean()
                d = np.sqrt((a * a).sum() * (b * b).sum())
                c = float((a * b).sum() / d) if d > 0 else np.nan
                if np.isfinite(c):
                    S[i, j] += c
                    W[i, j] += 1.0
    C = np.where(W > 0, S / np.maximum(W, 1e-9), np.nan)
    C = np.triu(C) + np.triu(C, 1).T
    np.fill_diagonal(C, 1.0)
    return pd.DataFrame(C, index=names, columns=names)


def cluster_order(C: pd.DataFrame):
    """按 1 - |corr| 做平均连接聚类，返回叶子顺序（用于相关性热力图排序）。"""
    D = 1.0 - np.abs(C.values)
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0
    Z = linkage(squareform(D, checks=False), method="average")
    from scipy.cluster.hierarchy import leaves_list
    return list(C.index[leaves_list(Z)]), Z


def correlate_pairs(C: pd.DataFrame, thr: float = 0.4):
    names = list(C.index)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            v = float(C.iloc[i, j])
            if np.isfinite(v) and abs(v) >= thr:
                rows.append(dict(因子A=names[i], 因子B=names[j], 相关系数=v))
    return pd.DataFrame(rows).sort_values("相关系数", key=lambda s: -s.abs()).reset_index(drop=True)


def ic_decay(zsc: dict, fwd_h: dict, uni: np.ndarray, dates) -> pd.DataFrame:
    """IC 衰减光谱：同一个因子在不同前瞻窗口下的平均 IC。"""
    out = {}
    for h, fwd in fwd_h.items():
        vals = {}
        for nm in FACTORS:
            x = np.asarray(zsc[nm].values, dtype="float64")
            v = np.array([spearman(x[i], fwd[i], uni[i]) for i in range(len(x))])
            vals[nm] = np.nanmean(v)
        out[h] = vals
    return pd.DataFrame(out)


def decay_kind(row: pd.Series) -> str:
    """按 IC 衰减光谱给因子分型：

        快衰减   |IC| 在 5 日内见顶（信号周内就被市场吃掉，必须高频调仓）
        慢因子   |IC| 一路走强、63 日仍在放大（拿得越久越兑现，月频都嫌快）
        中期因子 峰值落在 10~21 日附近
    """
    a = np.abs(row.values.astype(float))
    hz = list(row.index)
    peak = hz[int(np.nanargmax(a))]
    if peak <= 5:
        return "快衰减"
    if peak >= 42 and a[-1] >= a[0] * 1.15:
        return "慢因子"
    return "中期因子"


def coverage_table(raw: dict, mask: np.ndarray) -> pd.Series:
    """覆盖率：**在可投资域内**，调仓日上有多大比例真的算得出这个因子。"""
    rows = {}
    for nm in FACTORS:
        v = np.asarray(raw[nm].values, dtype="float64")
        ok = np.isfinite(v) & mask
        n = mask.sum(1)
        rows[nm] = float(np.nanmean(np.where(n > 0, ok.sum(1) / np.maximum(n, 1), np.nan)))
    return pd.Series(rows, name="覆盖率")


def signal_autocorr(zsc: dict, mask: np.ndarray) -> pd.Series:
    """信号黏性：相邻两个调仓日、同一个因子的横截面秩相关。
    黏性低 = 排名换得快 = 换手高，卡片上的「年换手」就是从这儿来的。"""
    rows = {}
    for nm in FACTORS:
        X = np.asarray(zsc[nm].values, dtype="float64")
        v = [spearman(X[i], X[i + 1], mask[i] & mask[i + 1]) for i in range(len(X) - 1)]
        rows[nm] = float(np.nanmean(v))
    return pd.Series(rows, name="信号黏性")


def run_deciles(md, comp: pd.DataFrame, cfg: dict, verbose=True) -> dict:
    """十分层回测：把合成打分切成 10 层，每层整层等权买入，走**真引擎**
    （含涨跌停、参与率、成本），看年化与夏普是不是随层号单调。"""
    R = comp.rank(axis=1, pct=True)
    navs, rows = {}, []
    # 分层组合每层约 450 只、单只权重 ~0.2%，而基准配置的调仓带是 2e-3 —— 两者同量级，
    # 会让「已经跌出这一层」的老仓位因为 delta 不够大而永远卖不掉（持仓数滚到几千只）。
    # 所以分层测试单独用 1e-5 的带，其余约束（涨跌停、参与率、成本、T+1）与基准一致。
    HG_DEC = dict(HG, rebalance_band=1e-5)
    for d in range(N_DEC):
        lo, hi = d / N_DEC, (d + 1) / N_DEC
        panel = comp.where((R > lo) & (R <= hi))
        st = PanelStrategy(panel, name="D%d" % (d + 1), n_hold=10 ** 6,
                           buffer_mult=0.0, max_weight=1.0)
        res = Backtest(md, **HG_DEC).run(st, start=cfg["start"], end=cfg["end"], rebalance="M")
        navs["D%d" % (d + 1)] = res.nav
        st_ = metrics.perf_stats(res.nav)
        rows.append(dict(层="D%d" % (d + 1), 年化=st_["年化收益"], 夏普=st_["夏普"],
                         最大回撤=st_["最大回撤"], 年换手=float(res.daily.turnover.mean() * 252),
                         持仓数=float(res.daily.n_holdings.median()),
                         成本占净值=float(res.trades.fee.sum() / cfg["cash"]) if len(res.trades) else 0.0))
        if verbose:
            print("   %s 年化 %6.2f%%  夏普 %.2f  持仓 %.0f 只"
                  % ("D%d" % (d + 1), 100 * st_["年化收益"], st_["夏普"],
                     float(res.daily.n_holdings.median())))
    return {"nav": pd.DataFrame(navs), "table": pd.DataFrame(rows).set_index("层")}


def decile_gross(comp: pd.DataFrame, fwd: np.ndarray, uni: np.ndarray,
                 dates) -> pd.Series:
    """毛收益口径的分层均值（不含成本与撮合约束），作为引擎结果的对照。"""
    R = comp.rank(axis=1, pct=True).values
    rows = []
    for i in range(len(dates)):
        ok = uni[i]
        v = []
        for d in range(N_DEC):
            lo, hi = d / N_DEC, (d + 1) / N_DEC
            m = ok & (R[i] > lo) & (R[i] <= hi) & np.isfinite(fwd[i])
            v.append(float(np.nanmean(fwd[i][m])) if m.sum() > 10 else np.nan)
        rows.append(v)
    mean_r = pd.DataFrame(rows, index=dates,
                          columns=["D%d" % (d + 1) for d in range(N_DEC)]).mean()
    n_per_year = 252.0 / FWD
    return (1.0 + mean_r) ** n_per_year - 1.0


def ic_stats_of(panel, fwd, uni, dates, is_end) -> dict:
    s = pd.Series(ic_series(np.asarray(panel, dtype="float64"), fwd, uni),
                  index=dates).dropna()
    if len(s) < 3:
        return dict(IC=np.nan, ICIR=np.nan, 胜率=np.nan, 样本内IC=np.nan, 样本外IC=np.nan)
    si = s[s.index <= pd.Timestamp(is_end)]
    so = s[s.index > pd.Timestamp(is_end)]
    return dict(IC=float(s.mean()), ICIR=float(s.mean() / s.std()),
                胜率=float((s > 0).mean()), 样本内IC=float(si.mean()),
                样本外IC=float(so.mean()), 自相关=float(s.autocorr(1)))


def run_loo(zsc: dict, sel: list, signs: dict, fwd: np.ndarray, uni: np.ndarray,
            dates, is_end) -> pd.DataFrame:
    """留一法边际贡献：去掉某个因子后，合成打分的 IC 变化了多少。"""
    full = sum(zsc[nm].values * signs[nm] for nm in sel) / len(sel)
    base = ic_stats_of(full, fwd, uni, dates, is_end)
    rows = []
    for nm in sel:
        rest = [f for f in sel if f != nm]
        panel = sum(zsc[f].values * signs[f] for f in rest) / len(rest)
        st = ic_stats_of(panel, fwd, uni, dates, is_end)
        rows.append(dict(因子=nm, 去掉后IC=st["IC"], 去掉后样本外IC=st["样本外IC"],
                         ΔIC=base["IC"] - st["IC"],
                         Δ样本外IC=base["样本外IC"] - st["样本外IC"]))
    return pd.DataFrame(rows).set_index("因子"), base


def random_control(ok: np.ndarray, fwd: np.ndarray, n_draw=N_RANDOM,
                   top_n: int = 50, seed: int = 20211016) -> np.ndarray:
    """随机对照（毛收益口径）：每期在股票池里随机抽 top_n 只等权持有 FWD 个交易日。
    它衡量的是「同池同规则下纯抽签能拿到多少」。"""
    rng = np.random.default_rng(seed)
    n_period = len(ok)
    ann = []
    for _ in range(n_draw):
        tot = 1.0
        for i in range(n_period):
            idx = np.flatnonzero(ok[i])
            if len(idx) == 0:
                continue
            pick = rng.choice(idx, size=min(top_n, len(idx)), replace=False)
            r = float(np.nanmean(fwd[i][pick]))
            if np.isfinite(r):
                tot *= (1.0 + r)
        ann.append(tot ** (12.0 / n_period) - 1.0)
    return np.array(ann)


def topn_gross(score: np.ndarray, ok: np.ndarray, fwd: np.ndarray,
               top_n: int = 50) -> float:
    """给定打分，每期取池内前 top_n 只等权持有的毛收益年化（与随机对照同口径）。"""
    tot = 1.0
    for i in range(len(ok)):
        m = ok[i] & np.isfinite(score[i]) & np.isfinite(fwd[i])
        idx = np.flatnonzero(m)
        if len(idx) == 0:
            continue
        k = min(top_n, len(idx))
        pick = idx[np.argsort(-score[i][idx], kind="stable")[:k]]
        tot *= (1.0 + float(np.nanmean(fwd[i][pick])))
    return tot ** (12.0 / len(ok)) - 1.0

def analyze(md, ts, raw, zsc, art, verbose=True) -> dict:
    """把看板要用的所有数字一次算完。"""
    t0 = time.time()
    cfg = art["cfg"]
    is_end = cfg["is_end"]
    dates = md.dates[list(ts)]
    adj = md.adj_close
    uni_all = md.universe()
    uni = uni_all.values[list(ts)]

    fwd_h = {h: (adj.shift(-h) / adj - 1.0).values[list(ts)] for h in HORIZONS}
    fwd = fwd_h[FWD]
    investable = uni & np.isfinite(fwd)

    sel = list(cfg["sel_factors"])
    signs = {}
    for nm in FACTORS:
        v = float(art["icstat"].loc[nm, "方向"])
        signs[nm] = v if np.isfinite(v) else 1.0
    comp = sum(zsc[nm] * signs[nm] for nm in sel) / len(sel)
    comp.name = "comp"
    if verbose:
        print("   合成面板 %s（%d 个因子）" % (comp.shape, len(sel)))

    # ---- 因子体检：IC 统计 + 覆盖率 + 衰减类型 ----
    stats = {}
    for nm in FACTORS:
        s = art["ic"][nm].dropna()
        si = s[s.index <= pd.Timestamp(is_end)]
        so = s[s.index > pd.Timestamp(is_end)]
        stats[nm] = dict(
            族=FAMILY[nm], 含义=DESC[nm], 窗口=WINDOW[nm], 配方=FORMULA[nm],
            IC=float(s.mean()), ICIR=float(s.mean() / s.std()) if s.std() > 0 else np.nan,
            胜率=float((s > 0).mean()),
            样本内IC=float(si.mean()), 样本外IC=float(so.mean()),
            方向=signs[nm], 方向一致=bool(np.sign(si.mean()) == np.sign(so.mean())),
            IC自相关=float(s.autocorr(1)))
    STAT = pd.DataFrame(stats).T
    STAT["单因子年化"] = [float(art["single"].loc[nm, "年化收益"]) for nm in STAT.index]
    STAT["单因子夏普"] = [float(art["single"].loc[nm, "夏普"]) for nm in STAT.index]
    STAT["单因子换手"] = [float(art["single"].loc[nm, "年换手"]) for nm in STAT.index]

    cov = coverage_table(raw, investable)
    sac = signal_autocorr(zsc, investable)
    STAT["覆盖率"] = [float(cov[nm]) for nm in STAT.index]
    STAT["信号黏性"] = [float(sac[nm]) for nm in STAT.index]

    if verbose:
        print("   因子相关性 %.1fs" % (time.time() - t0))
    C = factor_corr(zsc)
    order, _ = cluster_order(C)
    pairs = correlate_pairs(C, thr=0.35)

    if verbose:
        print("   IC 衰减光谱 %.1fs" % (time.time() - t0))
    DEC = ic_decay(zsc, fwd_h, investable, dates)
    STAT["衰减类型"] = [decay_kind(DEC.loc[nm]) for nm in STAT.index]
    STAT["IC放大"] = [float(abs(DEC.loc[nm, 63]) / max(abs(DEC.loc[nm, 1]), 1e-6))
                   for nm in STAT.index]

    if verbose:
        print("   十分层回测（10 组真回测）...")
    dec = run_deciles(md, comp, cfg, verbose=verbose)
    dec["gross"] = decile_gross(comp, fwd, investable, dates)
    ls = dec["nav"]["D10"] / dec["nav"]["D1"]
    spread = pd.DataFrame({"D10/D1": ls})

    if verbose:
        print("   留一法边际贡献 %.1fs" % (time.time() - t0))
    loo, comp_stats = run_loo(zsc, sel, signs, fwd, investable, dates, is_end)

    # ---- 毛收益口径：因子 top50 vs 随机抽签 vs 全市场 ----
    comp_vals = comp.values
    if verbose:
        print("   随机对照（%d 组，毛收益口径）..." % N_RANDOM)
    rand = random_control(investable, fwd)
    mkt_period = np.array([_masked_mean(fwd[i], investable[i]) for i in range(len(dates))])
    mkt_period = mkt_period[np.isfinite(mkt_period)]
    gross_mkt = float(np.prod(1.0 + mkt_period) ** (12.0 / len(mkt_period)) - 1.0)
    gross_comp = topn_gross(comp_vals, investable, fwd, top_n=int(cfg["n_hold"]))

    A = dict(cfg=cfg, dates=dates, ts=ts, uni=uni, investable=investable, fwd=fwd,
             fwd_h=fwd_h, sel=sel, signs=signs, comp=comp, comp_vals=comp_vals,
             STAT=STAT, corr=C, order=order, pairs=pairs, decay=DEC, cover=cov,
             decile=dec, spread=spread, loo=loo, comp_stats=comp_stats, rand=rand,
             gross=dict(市场=float(np.mean(mkt_period)), market_ann=gross_mkt,
                        comp_ann=gross_comp),
             n_period=len(dates))
    if verbose:
        print("   分析完成 %.1fs" % (time.time() - t0))
    return A

# --------------------------------------------------------------------------- #
# 四、图形
# --------------------------------------------------------------------------- #
DIV_IC = [[0.0, "#2f6fd0"], [0.35, "#16283f"], [0.5, "#0d131c"],
          [0.65, "#3f2326"], [1.0, "#e5534b"]]
FAMS = {"波动率族": PAL["vol"], "成交量族": PAL["amt"]}


def weekly(s: pd.Series) -> pd.Series:
    """净曲线抽稀到周频：图形一样，HTML 体积小一半。"""
    w = s.resample("W-FRI").last().dropna()
    return pd.concat([s.iloc[[0]], w[w.index > s.index[0]]])


def fig_universe(md, cfg):
    d = md.dates
    n_all = np.isfinite(md.close.values).sum(1)
    n_uni = md.universe().sum(1).values
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d, y=n_all, name="全部有报价证券", mode="lines",
                             line=dict(color=PAL["dim"], width=1.2),
                             hovertemplate="%{x|%Y-%m-%d}<br>%{y} 只<extra></extra>"))
    fig.add_trace(go.Scatter(x=d, y=n_uni, name="可交易股票池", mode="lines",
                             line=dict(color=PAL["gold"], width=1.8),
                             fill="tozeroy", fillcolor="rgba(227,179,65,0.08)",
                             hovertemplate="%{x|%Y-%m-%d}<br>%{y} 只<extra></extra>"))
    fig.add_vrect(x0=cfg["start"], x1=cfg["end"], fillcolor="rgba(56,189,248,0.07)",
                  line_width=0, layer="below")
    fig.add_annotation(x=cfg["start"], y=0.04, yref="paper", xanchor="left", xshift=8,
                       text="回测区间（右侧阴影）", showarrow=False,
                       font=dict(size=11, color=PAL["mut"]))
    fig.update_yaxes(title="证券数")
    return style(fig, height=330, title="矿区：股票池随时间的宽度（含上市满 250 日、20 日均额 ≥ 2000 万、股价 ≥ 2 元的过滤）")


def fig_ic_heatmap(ic: pd.DataFrame, is_end):
    fig = go.Figure(go.Heatmap(
        z=ic.values.T, x=ic.index, y=list(ic.columns), colorscale=DIV_IC, zmid=0.0,
        zmin=-0.15, zmax=0.15,
        colorbar=dict(title=dict(text="IC", font=dict(size=11)), thickness=12,
                      len=0.85, tickfont=dict(size=10)),
        hovertemplate="%{y}　%{x|%Y-%m}<br>IC %{z:.3f}<extra></extra>"))
    fig.add_vline(x=pd.Timestamp(is_end), line=dict(color=PAL["gold"], width=1.4, dash="dash"))
    fig.add_annotation(x=pd.Timestamp("2022-06-30"), y=1.06, yref="paper", showarrow=False,
                       text="样本内（定方向）", font=dict(size=11, color=PAL["mut"]))
    fig.add_annotation(x=pd.Timestamp("2025-02-28"), y=1.06, yref="paper", showarrow=False,
                       text="样本外（检验）", font=dict(size=11, color=PAL["mut"]))
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=11))
    return style(fig, height=430, legend=False,
                 title="淘洗：逐月 IC 热力图（红 = 因子值大的股票随后涨得多，蓝 = 反过来）")


def fig_ic_decay(dec: pd.DataFrame, stat: pd.DataFrame):
    """IC 衰减光谱：横轴是前瞻窗口，看每个因子的信号是「快」还是「慢」。"""
    cols = list(dec.columns)
    hz = ["%s" % c for c in cols]          # 类别轴：用字符串，避免数值把轴拉长
    rank = stat["IC"].abs().sort_values(ascending=False).index[:4]
    fig = go.Figure()
    for nm in dec.index:
        v = dec.loc[nm].values.astype(float)
        hi = nm in list(rank)
        fig.add_trace(go.Scatter(
            x=hz, y=v, name=nm, mode="lines+markers",
            line=dict(color=FAMS[stat.loc[nm, "族"]], width=2.6 if hi else 1.0),
            marker=dict(size=6 if hi else 3.5), opacity=1.0 if hi else 0.26,
            hovertemplate=nm + "　" + str(stat.loc[nm, "含义"]) + "<br>%{x} 日 IC %{y:.4f}<extra></extra>"))
    # 末端直接标名字。注意不能用 add_annotation：类别轴上的 annotation 会把
    # 坐标轴自动拉长（图形被挤到左边），改成一条 cliponaxis=False 的文字 trace。
    ends = sorted((float(dec.loc[nm, cols[-1]]), nm) for nm in rank)
    ys = []
    for y0, _nm in ends:                     # 末端挨得太近的错开一点
        ys.append(y0 if (not ys or y0 - ys[-1] > 0.014) else ys[-1] + 0.014)
    fig.add_trace(go.Scatter(x=[hz[-1]] * len(ends), y=ys, mode="text",
                             text=["   " + nm for _, nm in ends],
                             textposition="middle right", textfont=dict(size=11),
                             showlegend=False, hoverinfo="skip", cliponaxis=False))
    fig.update_xaxes(title="前瞻窗口（交易日）", type="category",
                     categoryorder="array", categoryarray=hz)
    fig.update_yaxes(title="平均 IC")
    fig.add_hline(y=0, line=dict(color=PAL["line"], width=1))
    return style(fig, height=470, legend=False,
                 margin=dict(l=64, r=86, t=54, b=58),
                 title="淘洗：IC 衰减光谱（蓝 = 波动率族，金 = 成交量族；粗线 = IC 最强的 4 个）"
                       "—— 随窗口走强 = 慢因子，峰在周内 = 快因子")


def fig_ic_box(ic: pd.DataFrame):
    fig = go.Figure()
    order = ic.mean().abs().sort_values(ascending=False).index
    for nm in order:
        v = ic[nm].dropna().values
        fig.add_trace(go.Box(y=v, name=nm, boxpoints=False, width=0.55,
                             line=dict(color=FAMS[FAMILY[nm]], width=1.4),
                             fillcolor="rgba(56,189,248,0.16)" if FAMILY[nm] == "波动率族"
                             else "rgba(240,180,41,0.16)",
                             hovertemplate=nm + "<br>IC %{y:.3f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=PAL["grid"], width=1))
    fig.update_yaxes(title="逐月 IC")
    return style(fig, height=380, legend=False,
                 title="淘洗：IC 的分布 —— 箱体越窄越稳；箱子整体偏在一侧才是有方向的因子")


def fig_tstat(ist: pd.DataFrame):
    rows = [nm for nm in FACTORS]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=rows, x=ist.loc[rows, "样本内t"], name="样本内 t（21-23）",
                         orientation="h", marker_color=PAL["vol"],
                         hovertemplate="%{y} 样本内 t %{x:.2f}<extra></extra>"))
    fig.add_trace(go.Bar(y=rows, x=ist.loc[rows, "样本外t"], name="样本外 t（24-26）",
                         orientation="h", marker_color=PAL["amt"],
                         hovertemplate="%{y} 样本外 t %{x:.2f}<extra></extra>"))
    for v in (1.0, -1.0):
        fig.add_vline(x=v, line=dict(color=PAL["gold"], width=1.2, dash="dot"))
    fig.add_vline(x=0, line=dict(color=PAL["line"], width=1))
    fig.update_xaxes(title="IC 均值的 t 值")
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=11))
    fig.update_layout(barmode="group")
    return style(fig, height=430,
                 title="闯关：门坎是 |样本内 t| ≥ 1（虚线之外才算数）")


def fig_quadrant(stat: pd.DataFrame):
    fig = go.Figure()
    lim = float(np.nanmax(np.abs(np.r_[stat["样本内IC"], stat["样本外IC"]]))) * 1.25
    fig.add_shape(type="rect", x0=0, y0=0, x1=lim, y1=lim,
                  fillcolor="rgba(47,191,113,0.10)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=-lim, y0=-lim, x1=0, y1=0,
                  fillcolor="rgba(47,191,113,0.10)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=-lim, y0=0, x1=0, y1=lim,
                  fillcolor="rgba(229,83,75,0.10)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=0, y0=-lim, x1=lim, y1=0,
                  fillcolor="rgba(229,83,75,0.10)", line_width=0, layer="below")
    for nm in stat.index:
        fig.add_trace(go.Scatter(
            x=[stat.loc[nm, "样本内IC"]], y=[stat.loc[nm, "样本外IC"]], mode="markers+text",
            text=[nm], textposition="top center", textfont=dict(size=10.5),
            marker=dict(size=8 + 90 * abs(stat.loc[nm, "IC"]), color=FAMS[stat.loc[nm, "族"]],
                        line=dict(color=PAL["bg"], width=1), opacity=0.9),
            name=nm, showlegend=False,
            hovertemplate=(nm + " " + str(stat.loc[nm, "含义"]) +
                           "<br>样本内 IC %{x:.4f}<br>样本外 IC %{y:.4f}<extra></extra>")))
    fig.add_hline(y=0, line=dict(color=PAL["dim"], width=1))
    fig.add_vline(x=0, line=dict(color=PAL["dim"], width=1))
    fig.update_xaxes(title="样本内 IC（2021-2023）", range=[-lim, lim])
    fig.update_yaxes(title="样本外 IC（2024-2026）", range=[-lim, lim])
    return style(fig, height=470,
                 title="闯关：样本内 × 样本外四象限 —— 绿区（右上 / 左下）= 两段都认账，红区 = 样本内才灵光")


def fig_sankey(stat: pd.DataFrame, ist: pd.DataFrame, sel: list):
    names = list(FACTORS)
    n = len(names)
    fail_dir = [nm for nm in names if not bool(stat.loc[nm, "方向一致"])]
    fail_t = [nm for nm in names if bool(stat.loc[nm, "方向一致"])
              and abs(float(ist.loc[nm, "样本内t"])) < 1.0]
    pas = [nm for nm in names if nm not in fail_dir + fail_t]
    labels = names + ["方向不一致\n（样本外翻脸）", "样本内 |t| 不足 1\n（像噪音）",
                      "通过 %d 个" % len(pas), "等权 z-score 合成\n（前 50 只）"]
    idx = {nm: i for i, nm in enumerate(names)}
    g_dir, g_t, g_pass, g_comp = n, n + 1, n + 2, n + 3
    src, dst, val, lcol = [], [], [], []
    for nm in names:
        tgt = g_dir if nm in fail_dir else (g_t if nm in fail_t else g_pass)
        src.append(idx[nm]); dst.append(tgt); val.append(1)
        lcol.append("rgba(229,83,75,0.35)" if tgt in (g_dir, g_t)
                    else "rgba(227,179,65,0.35)")
    for _ in pas:
        src.append(g_pass); dst.append(g_comp); val.append(1)
        lcol.append("rgba(244,114,182,0.35)")
    node_col = [FAMS[FAMILY[nm]] for nm in names] + [PAL["dn"], PAL["dn"], PAL["gold"], PAL["comp"]]
    node_x = [0.012] * n + [0.52, 0.52, 0.52, 0.99]
    node_y = list(np.linspace(0.03, 0.97, n)) + [0.16, 0.5, 0.86, 0.5]
    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        node=dict(label=labels, x=node_x, y=node_y, color=node_col, pad=16, thickness=14,
                  line=dict(color="rgba(0,0,0,0)", width=0),
                  hovertemplate="%{label}<extra></extra>"),
        link=dict(source=src, target=dst, value=val, color=lcol,
                  hovertemplate="%{source.label} → %{target.label}<extra></extra>")))
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(family=FONT, size=12, color=PAL["text"]),
                      title=dict(text="闯关：13 → 10 的筛选漏斗（先看方向是否翻脸，再看 |t| 够不够格）",
                                 x=0.008, xanchor="left", font=dict(size=13, color=PAL["mut"])),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig

def fig_corr_heatmap(C: pd.DataFrame, order: list):
    M = C.loc[order, order]
    fig = go.Figure(go.Heatmap(
        z=M.values, x=list(order), y=list(order), colorscale=DIV_IC, zmid=0.0,
        zmin=-1, zmax=1, text=np.round(M.values, 2), texttemplate="%{text}",
        textfont=dict(size=10), colorbar=dict(thickness=12, len=0.85,
                                              tickfont=dict(size=10)),
        hovertemplate="%{y} ~ %{x}<br>ρ = %{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=10))
    return style(fig, height=560, legend=False, margin=dict(l=64, r=28, t=54, b=86),
                 title="体检：因子相关性（横截面 Spearman 相关，按 1-|ρ| 平均连接聚类排序）")


def mds2(C: pd.DataFrame):
    """经典多维标度：把 13 个因子按 1-ρ 当成距离摊到二维平面上。"""
    D = 1.0 - C.values.astype(float)
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D @ J
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1][:2]
    w = np.clip(w[idx], 1e-12, None)
    return V[:, idx] * np.sqrt(w)


def fig_star(C: pd.DataFrame, stat: pd.DataFrame):
    XY = mds2(C)
    names = list(C.index)
    pos = {nm: (float(XY[i, 0]), float(XY[i, 1])) for i, nm in enumerate(names)}
    fig = go.Figure()
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            r = float(C.loc[a, b])
            if not np.isfinite(r) or abs(r) < 0.35:
                continue
            fig.add_trace(go.Scatter(
                x=[pos[a][0], pos[b][0]], y=[pos[a][1], pos[b][1]], mode="lines",
                line=dict(color="rgba(229,83,75,0.55)" if r > 0 else "rgba(47,111,208,0.55)",
                          width=0.8 + 4.0 * abs(r)), showlegend=False, hoverinfo="skip"))
    for nm in names:
        s = 10 + 110 * abs(float(stat.loc[nm, "IC"]))
        fig.add_trace(go.Scatter(
            x=[pos[nm][0]], y=[pos[nm][1]], mode="markers+text", text=[nm],
            textposition="top center", textfont=dict(size=11),
            marker=dict(size=s, color=FAMS[stat.loc[nm, "族"]],
                        line=dict(color=PAL["bg"], width=1.2)),
            name=stat.loc[nm, "族"], legendgroup=stat.loc[nm, "族"],
            showlegend=nm in ("ivol", "amihud"),
            hovertemplate=(nm + " " + str(stat.loc[nm, "含义"]) +
                           "<br>IC %{customdata[0]:.4f}<br>ICIR %{customdata[1]:.2f}<extra></extra>"),
            customdata=[[float(stat.loc[nm, "IC"]), float(stat.loc[nm, "ICIR"])]]))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=1.02, x=0, title=None))
    return style(fig, height=520,
                 title="体检：因子星图（把 1-相关系数 当距离摊到平面上；连线越粗越像，点越大 |IC| 越强）")


def fig_race(art, bench_nav, stat):
    fig = go.Figure()
    for nm in FACTORS:
        s = weekly(art["factor_nav"][nm]) / 1e8
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, name=nm, mode="lines", legendgroup=stat.loc[nm, "族"],
            showlegend=False, line=dict(color=FAMS[stat.loc[nm, "族"]], width=1.0),
            opacity=0.5, hovertemplate=nm + " 净值 %{y:.3f}<extra></extra>"))
    for fam, col in FAMS.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                 name="%s（单因子）" % fam,
                                 line=dict(color=col, width=1.6), hoverinfo="skip"))
    for s, nm, col, w, dash, op in ((bench_nav, "等权全市场", PAL["bench"], 1.4, "dash", 0.9),
                                    (art["comp_nav"], "合成策略（前 50 只）", PAL["comp"], 2.6, "solid", 1.0)):
        x = weekly(s) / 1e8
        fig.add_trace(go.Scatter(x=x.index, y=x.values, name=nm, mode="lines",
                                 line=dict(color=col, width=w, dash=dash), opacity=op,
                                 hovertemplate=nm + " 净值 %{y:.3f}<extra></extra>"))
    fig.update_yaxes(title="净值（起点 = 1）")
    return style(fig, height=490, legend_pos="bottom",
                 title="熔炼：13 条单因子净值 + 合成策略（细线 = 单因子前 50 只等权，含全部成本与涨跌停约束）")


def fig_bubble(stat: pd.DataFrame):
    fig = go.Figure()
    for nm in stat.index:
        fig.add_trace(go.Scatter(
            x=[float(stat.loc[nm, "单因子换手"])], y=[float(stat.loc[nm, "单因子夏普"])],
            mode="markers+text", text=[nm], textposition="top center",
            textfont=dict(size=10.5),
            marker=dict(size=8 + 120 * abs(float(stat.loc[nm, "IC"])),
                        color=FAMS[stat.loc[nm, "族"]], opacity=0.85,
                        line=dict(color=PAL["bg"], width=1)),
            showlegend=False,
            hovertemplate=(nm + " " + str(stat.loc[nm, "含义"]) +
                           "<br>年换手 %{x:.1f}x<br>夏普 %{y:.2f}"
                           "<br>IC %{customdata:.4f}<extra></extra>"),
            customdata=[float(stat.loc[nm, "IC"])]))
    fig.add_hline(y=0, line=dict(color=PAL["line"], width=1))
    fig.update_xaxes(title="年换手（倍）")
    fig.update_yaxes(title="单因子组合夏普")
    return style(fig, height=470,
                 title="熔炼：单因子组合的「夏普 × 换手」（气泡 = |IC|）—— 右上角才是又能赚又便宜的好矿")


def fig_decile(dec: dict):
    tab = dec["table"]
    layers = list(tab.index)
    amt = tab["年化"].values * 100
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=layers, y=amt, name="分层年化收益（净，含成本）",
        marker=dict(color=amt, colorscale=[[0, "#2f6fd0"], [1, "#e3b341"]],
                    line=dict(color=PAL["bg"], width=1)),
        text=["%.1f%%" % v for v in amt], textposition="outside",
        textfont=dict(size=10.5),
        hovertemplate="%{x} 年化 %{y:.2f}%<br>年换手 %{customdata[0]:.1f}x<extra></extra>",
        customdata=tab[["年换手"]].values))
    fig.add_trace(go.Scatter(
        x=layers, y=tab["夏普"].values, name="夏普（右轴）", mode="lines+markers",
        line=dict(color=PAL["comp"], width=2), marker=dict(size=7), yaxis="y2",
        hovertemplate="%{x} 夏普 %{y:.2f}<extra></extra>"))
    fig.update_layout(yaxis2=dict(title="夏普", overlaying="y", side="right",
                                  showgrid=False, tickfont=dict(size=11),
                                  range=[0, float(tab["夏普"].max()) * 1.25]))
    fig.update_yaxes(title="年化收益 %")
    fig.update_xaxes(title="按合成打分切的十分层（D10 = 打分最高的十分之一）")
    return style(fig, height=470,
                 title="熔炼：十分层回测（真引擎、含成本）—— 单调就是信号，不单调就是巧合")


def fig_decile_nav(dec: dict, spread: pd.DataFrame):
    NAV = dec["nav"]
    fig = go.Figure()
    cols = list(NAV.columns)
    for i, c in enumerate(cols):
        s = weekly(NAV[c]) / 1e8
        thick = c in ("D1", "D5", "D10")
        t = i / max(len(cols) - 1, 1)
        col = "rgb(%d,%d,%d)" % (int(47 + t * (227 - 47)), int(111 + t * (179 - 111)),
                                 int(208 + t * (65 - 208)))
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=c, mode="lines",
                                 line=dict(color=col, width=2.4 if thick else 1.0),
                                 opacity=1.0 if thick else 0.5,
                                 hovertemplate=c + " 净值 %{y:.3f}<extra></extra>"))
    fig.update_yaxes(title="净值（起点 = 1）")
    return style(fig, height=430, legend_pos="bottom",
                 title="熔炼：十分层净值（蓝 = D1 打分最低，金 = D10 打分最高）")


def fig_spread(spread: pd.DataFrame):
    s = weekly(spread["D10/D1"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, name="D10 / D1（多空价差）",
                             mode="lines", line=dict(color=PAL["gold"], width=2),
                             fill="tozeroy", fillcolor="rgba(227,179,65,0.10)",
                             hovertemplate="%{x|%Y-%m}<br>价差 %{y:.3f}<extra></extra>"))
    mdd = float((s / s.cummax() - 1.0).min())
    fig.add_annotation(x=0.99, y=0.06, xref="paper", yref="paper", xanchor="right",
                       text="价差期间最深回撤 %.1f%%" % (100 * mdd), showarrow=False,
                       font=dict(size=12, color=PAL["mut"]))
    fig.update_yaxes(title="净值  D10 / D1")
    return style(fig, height=420, legend=False,
                 title="熔炼：多空价差（D10 ÷ D1）—— 不做空也能看的信号「纯度」指标")


def fig_loo(loo: pd.DataFrame, base: dict):
    d = loo.sort_values("ΔIC")
    fig = go.Figure()
    fig.add_trace(go.Bar(y=list(d.index), x=d["ΔIC"], orientation="h",
                         name="去掉后全样本 IC 掉多少", marker_color=PAL["gold"],
                         hovertemplate="%{y}：去掉后 IC 少 %{x:.4f}<extra></extra>"))
    fig.add_trace(go.Bar(y=list(d.index), x=d["Δ样本外IC"], orientation="h",
                         name="去掉后样本外 IC 掉多少", marker_color=PAL["vol"],
                         hovertemplate="%{y}：去掉后样本外 IC 少 %{x:.4f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=PAL["line"], width=1))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="ΔIC = 合成 IC − 去掉该因子后的合成 IC")
    fig.update_yaxes(tickfont=dict(size=11))
    return style(fig, height=440,
                 title="熔炼：留一法边际贡献（正 = 这个因子在给组合加分，负 = 它在拖后腿）")

def fig_comp_nav(art, bench_nav):
    from plotly.subplots import make_subplots
    nav = art["comp_nav"]
    bench = bench_nav.reindex(nav.index).ffill()
    w_nav = weekly(nav / nav.iloc[0])
    w_bench = weekly(bench / bench.dropna().iloc[0])
    dd = (nav / nav.cummax() - 1.0)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.06)
    fig.add_trace(go.Scatter(x=w_bench.index, y=w_bench.values, name="等权全市场",
                             line=dict(color=PAL["bench"], width=1.4, dash="dash"),
                             hovertemplate="%{x|%Y-%m}<br>基准 %{y:.3f}<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=w_nav.index, y=w_nav.values, name="合成策略（前 50 只）",
                             line=dict(color=PAL["comp"], width=2.4),
                             fill="tozeroy", fillcolor="rgba(244,114,182,0.07)",
                             hovertemplate="%{x|%Y-%m}<br>策略 %{y:.3f}<extra></extra>"),
                  row=1, col=1)
    w_dd = weekly(dd)
    fig.add_trace(go.Scatter(x=w_dd.index, y=w_dd.values, name="回撤", showlegend=False,
                             line=dict(color=PAL["dn"], width=1.3),
                             fill="tozeroy", fillcolor="rgba(229,83,75,0.18)",
                             hovertemplate="%{x|%Y-%m}<br>回撤 %{y:.2%}<extra></extra>"),
                  row=2, col=1)
    fig.update_yaxes(tickformat=".0%", row=2, col=1, title="回撤")
    fig.update_yaxes(title="净值（起点 = 1）", row=1, col=1)
    return style(fig, height=580, legend_pos="bottom",
                 title="熔炼：合成策略 vs 等权全市场（真引擎结果）")


def fig_annual(nav, bench_nav):
    ret = nav.pct_change().fillna(0.0)
    bret = bench_nav.reindex(ret.index).ffill().pct_change().fillna(0.0)
    tab = metrics.annual_table(ret, bret)
    yrs = [str(i) for i in tab.index]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=yrs, y=tab["收益"].values * 100, name="合成策略",
                         marker_color=PAL["comp"], text=["%.1f%%" % (100 * v) for v in tab["收益"]],
                         textposition="outside", textfont=dict(size=10.5),
                         hovertemplate="%{x} 策略 %{y:.2f}%<extra></extra>"))
    fig.add_trace(go.Bar(x=yrs, y=tab["基准"].values * 100, name="等权全市场",
                         marker_color=PAL["bench"],
                         hovertemplate="%{x} 基准 %{y:.2f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=yrs, y=tab["超额"].values * 100, name="超额",
                             mode="lines+markers", line=dict(color=PAL["gold"], width=2),
                             marker=dict(size=7),
                             hovertemplate="%{x} 超额 %{y:.2f}%<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=PAL["line"], width=1))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="收益 %")
    return style(fig, height=400, title="熔炼：分年度收益与超额")


def fig_tornado(sens: pd.DataFrame):
    base = float(sens.loc["基准配置", "年化收益"])
    d = sens.drop(index=["基准配置"])["年化收益"].dropna()
    d = ((d - base) * 100).sort_values()
    cols = [PAL["up"] if v >= 0 else PAL["dn"] for v in d.values]
    fig = go.Figure(go.Bar(
        y=list(d.index), x=d.values, orientation="h", marker_color=cols,
        text=["%+.2f pp" % v for v in d.values], textposition="outside",
        textfont=dict(size=10.5),
        hovertemplate="%{y}<br>年化相对基准 %{x:+.2f} pp<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=PAL["gold"], width=1.4))
    mx = float(np.nanmax(np.abs(d.values))) * 1.42
    fig.update_xaxes(title="年化收益相对基准配置的变化（百分点）", range=[-mx, mx])
    fig.update_yaxes(tickfont=dict(size=11))
    return style(fig, height=520, legend=False,
                 title="成色：参数敏感性（基准配置 = 月频 / 前 50 只 / 缓冲 2x / 参与率 10% / 成本 1x）")


def fig_random(rand, comp_ann, mkt_ann):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=rand * 100, nbinsx=26, name="随机抽 50 只 × 200 组",
                               marker=dict(color="rgba(132,148,168,0.55)",
                                           line=dict(color="#5b6a7d", width=0.6)),
                               hovertemplate="年化 %{x:.1f}%　%{y} 组<extra></extra>"))
    fig.add_vline(x=comp_ann * 100, line=dict(color=PAL["gold"], width=2.4))
    fig.add_vline(x=mkt_ann * 100, line=dict(color=PAL["bench"], width=1.6, dash="dash"))
    top = float((rand < comp_ann).mean())
    fig.add_annotation(x=comp_ann * 100, y=0.6, yref="paper", yanchor="bottom",
                       xanchor="right", text=" 合成 top50：%.1f%%　" % (100 * comp_ann),
                       showarrow=False, font=dict(size=12, color=PAL["gold"]))
    fig.add_annotation(x=mkt_ann * 100, y=0.76, yref="paper", yanchor="bottom", xanchor="left",
                       text=" 全市场等权 %.1f%%" % (100 * mkt_ann), showarrow=False,
                       font=dict(size=11, color=PAL["bench"]))
    fig.update_xaxes(title="毛收益口径年化 %")
    fig.update_yaxes(title="组数")
    return style(fig, height=430, legend=False,
                 title="成色：随机对照（同股票池、同调仓频率，随机抽 50 只等权持有）—— 策略高于 %.0f%% 的随机组"
                       % (100 * top))

def fig_quality(stat: pd.DataFrame):
    """信号黏性（条长）与覆盖率（条尾文字）：一个说「排名换得快不快」，
    一个说「这个因子在池子里到底算不算得出来」。"""
    d = stat[["覆盖率", "信号黏性"]].astype(float).sort_values("信号黏性")
    fig = go.Figure(go.Bar(
        x=d["信号黏性"].values, y=list(d.index), orientation="h",
        marker=dict(color=d["信号黏性"].values,
                    colorscale=[[0, "#2f6fd0"], [1, "#e3b341"]],
                    line=dict(color=PAL["bg"], width=1)),
        text=["覆盖率 %.1f%%" % (100 * v) for v in d["覆盖率"]],
        textposition="outside", textfont=dict(size=10.5, color=PAL["mut"]),
        hovertemplate="%{y}<br>信号黏性 %{x:.2f}<br>覆盖率 %{customdata:.2%}<extra></extra>",
        customdata=d["覆盖率"].values))
    fig.add_vline(x=float(d["信号黏性"].mean()),
                  line=dict(color=PAL["gold"], width=1.2, dash="dot"))
    fig.update_xaxes(range=[0, 1.2],
                     title="信号黏性 = 相邻两个调仓日、同一因子的横截面秩相关（越高越黏，换手越低）")
    fig.update_yaxes(tickfont=dict(size=11))
    return style(fig, height=460, legend=False,
                 title="原矿：信号黏性（条长）与覆盖率（条尾）—— 黏性低 = 排名换得快 = 换手高")

# --------------------------------------------------------------------------- #
# 五、HTML
# --------------------------------------------------------------------------- #
CSS = """
:root{--bg:#0a0e14;--panel:#111721;--panel2:#141b26;--line:#1f2733;--text:#e7edf5;
 --mut:#8494a8;--vol:#38bdf8;--amt:#f0b429;--comp:#f472b6;--gold:#e3b341;--up:#2fbf71;--dn:#e5534b}
*{box-sizing:border-box}
body{margin:0;background:#0a0e14;background-image:radial-gradient(1100px 520px at 15% -8%,#17222f 0%,rgba(10,14,20,0) 62%),
 radial-gradient(900px 480px at 88% 4%,#1c1a24 0%,rgba(10,14,20,0) 60%) fixed;
 color:var(--text);font-family:"Microsoft YaHei UI","PingFang SC","Hiragino Sans GB",
 "Noto Sans SC",system-ui,sans-serif;line-height:1.72;-webkit-font-smoothing:antialiased}
.wrap{max-width:1320px;margin:0 auto;padding:0 24px}
header{padding:56px 0 22px;border-bottom:1px solid var(--line)}
.kicker{letter-spacing:.34em;font-size:11px;color:var(--gold);text-transform:uppercase}
h1{font-size:clamp(28px,4.4vw,44px);margin:12px 0 8px;font-weight:800;letter-spacing:.01em}
h1 small{font-size:.42em;color:var(--mut);font-weight:400;letter-spacing:.06em;margin-left:8px}
.lead{color:#bcc9d9;max-width:960px;margin:6px 0 20px}
.lead b{color:var(--text)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(148px,1fr));gap:11px;margin:18px 0 6px}
.kpi{background:linear-gradient(180deg,#141b26,#0e141d);border:1px solid var(--line);
 border-radius:12px;padding:12px 14px}
.kpi .v{font-size:21px;font-weight:700;letter-spacing:.01em}
.kpi .k{font-size:11.5px;color:var(--mut);margin-top:2px}
nav{position:sticky;top:0;z-index:30;background:rgba(10,14,20,.88);
 backdrop-filter:blur(12px);border-bottom:1px solid var(--line)}
nav .wrap{display:flex;gap:4px;overflow-x:auto;padding:9px 24px;scrollbar-width:none}
nav a{color:var(--mut);text-decoration:none;font-size:12.5px;padding:5px 12px;border-radius:999px;white-space:nowrap}
nav a:hover{color:#fff;background:#1a2230}
.stage{padding:38px 0 6px;border-bottom:1px solid rgba(31,39,51,.55)}
.shead{display:flex;align-items:center;gap:12px;margin-bottom:4px}
.badge{width:30px;height:30px;flex:0 0 30px;border-radius:9px;display:flex;align-items:center;
 justify-content:center;font-size:13px;font-weight:700;color:#0a0e14;
 background:linear-gradient(135deg,var(--gold),#c9922a)}
h2{font-size:clamp(19px,2.4vw,25px);margin:0;font-weight:700}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;
 padding:6px 10px 2px;margin:14px 0}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.grid2>.card{margin:14px 0}
@media(max-width:1000px){.grid2{grid-template-columns:1fr}}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:13px;margin:16px 0}
.fcard{background:linear-gradient(180deg,var(--panel2),#0e141d);border:1px solid var(--line);
 border-radius:14px;padding:15px 16px 13px;position:relative}
.fcard.sel{border-color:#3a4a33;box-shadow:inset 0 0 0 1px rgba(47,191,113,.12)}
.fcard.out{opacity:.62}
.frow{display:flex;align-items:center;justify-content:space-between;gap:8px}
.nm{font-size:17px;font-weight:700}
.tag{font-size:10.5px;padding:2px 9px;border-radius:999px;border:1px solid}
.t-vol{color:var(--vol);border-color:rgba(56,189,248,.4);background:rgba(56,189,248,.08)}
.t-amt{color:var(--amt);border-color:rgba(240,180,41,.4);background:rgba(240,180,41,.08)}
.t-ok{color:var(--up);border-color:rgba(47,191,113,.45);background:rgba(47,191,113,.08)}
.t-no{color:var(--mut);border-color:var(--line);background:rgba(132,148,168,.07)}
.desc{color:#b9c6d6;font-size:13px;margin:6px 0 2px}
.spark{width:100%;height:32px;display:block;margin:6px 0 2px}
.fmeta{display:flex;gap:12px;flex-wrap:wrap;font-size:11.5px;color:var(--mut);margin-top:5px}
.fmeta b{color:var(--text);font-weight:600}
.pos{color:var(--dn)}.neg{color:var(--vol)}
.formula{font-family:ui-monospace,Consolas,"Courier New",monospace;font-size:11.5px;color:#9fb3c8;
 background:#0b1017;border:1px solid var(--line);border-radius:8px;padding:6px 9px;margin-top:9px;
 overflow-x:auto;white-space:nowrap}
.tw{overflow-x:auto;margin:14px 0;border:1px solid var(--line);border-radius:12px;background:var(--panel)}
table.data{border-collapse:collapse;width:100%;font-size:12.5px}
table.data th,table.data td{padding:8px 12px;border-bottom:1px solid rgba(31,39,51,.7);
 text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}
table.data th:first-child,table.data td:first-child{text-align:left;position:sticky;left:0;
 background:var(--panel)}
table.data thead th{color:var(--mut);font-weight:500;background:#131a24}
table.data tr:hover td{background:#161e29}
.note{color:var(--mut);font-size:12.8px;margin:10px 0 0}
ul.clean{padding-left:20px;margin:10px 0}
ul.clean li{margin:7px 0;color:#c3cfdd}
ul.clean li b{color:var(--text)}
footer{margin-top:44px;padding:26px 0 70px;color:var(--mut);font-size:12.5px}
footer h3{color:var(--text);font-size:15px;margin:18px 0 6px}
code{font-family:ui-monospace,Consolas,monospace;font-size:12px;background:#0b1017;
 border:1px solid var(--line);border-radius:6px;padding:1px 6px;color:#9fb3c8}
"""

CFG = dict(displayModeBar=False, responsive=True, displaylogo=False, scrollZoom=False)


def fig_html(fig, div_id):
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id=div_id,
                       config=CFG, default_width="100%")


def table_html(df: pd.DataFrame) -> str:
    head = "".join("<th>%s</th>" % c for c in [df.index.name or ""] + [str(c) for c in df.columns])
    body = []
    for i, r in df.iterrows():
        cells = "".join("<td>%s</td>" % r[c] for c in df.columns)
        body.append("<tr><td>%s</td>%s</tr>" % (i, cells))
    return ('<div class="tw"><table class="data"><thead><tr>%s</tr></thead>'
            '<tbody>%s</tbody></table></div>' % (head, "".join(body)))


def kpi(v, k):
    return '<div class="kpi"><div class="v">%s</div><div class="k">%s</div></div>' % (v, k)


def sign_span(v, nd=4, invert=False):
    cls = "pos" if (v > 0) != invert else "neg"
    return '<b class="%s">%s</b>' % (cls, ("%+.*f" % (nd, v)) if nd else "%+d" % v)


def card_html(nm, stat, ic_series, sel, ist):
    s = stat.loc[nm]
    col = FAMS[s["族"]]
    tagcls = "t-vol" if s["族"] == "波动率族" else "t-amt"
    if nm in sel:
        state = '<span class="tag t-ok">入闱</span>'
        card_cls = "fcard sel"
    else:
        reason = "样本外方向翻脸" if not bool(s["方向一致"]) else "|样本内 t| &lt; 1"
        state = '<span class="tag t-no">落选 · %s</span>' % reason
        card_cls = "fcard out"
    spark = sparkline(ic_series.rolling(6, min_periods=2).mean().values, color=col)
    hi = "买高值" if float(s["方向"]) > 0 else "买低值"
    return ('''<div class="%s">
  <div class="frow"><span class="nm">%s</span><span class="tag %s">%s</span></div>
  <div class="desc">%s</div>
  %s
  <div class="fmeta"><span>IC %s</span><span>ICIR %.2f</span><span>IC 胜率 %.0f%%</span></div>
  <div class="fmeta"><span>样本内 t %.2f</span><span>样本外 t %.2f</span><span>覆盖率 %.0f%%</span></div>
  <div class="formula">%s</div>
  <div class="fmeta"><span>方向 %s</span><span>%s</span><span>单因子夏普 %.2f</span><span>年换手 %.1fx</span></div>
  <div style="margin-top:8px">%s</div>
</div>''' % (card_cls, nm, tagcls, s["族"], s["含义"], spark,
             sign_span(float(s["IC"]), 4), float(s["ICIR"]), 100 * float(s["胜率"]),
             float(ist.loc[nm, "样本内t"]), float(ist.loc[nm, "样本外t"]),
             100 * float(s["覆盖率"]), s["配方"], hi, s["衰减类型"],
             float(s["单因子夏普"]), float(s["单因子换手"]), state))
def stage(num, name, sub, body, note=""):
    tail = '<p class="note">%s</p>' % note if note else ""
    return ('<section class="stage" id="s%d"><div class="shead"><span class="badge">%d</span>'
            '<h2>%s</h2></div><p class="lead">%s</p>%s%s</section>'
            % (num, num, name, sub, body, tail))


def grid2(a, b):
    return '<div class="grid2"><div class="card">%s</div><div class="card">%s</div></div>' % (a, b)


def card(a):
    return '<div class="card">%s</div>' % a


def build_html(A: dict, art: dict, figs: dict, cdn: bool = False) -> str:
    cfg = A["cfg"]
    stat = A["STAT"]
    ist = art["icstat"]
    sel = list(A["sel"])
    ic = art["ic"]
    dec, tab, loo = A["decile"], A["decile"]["table"], A["loo"]
    sens, sub = art["sens"], art["sub"]
    comp = art["single"].loc["vv_comp"]
    bench = art["single"].loc["基准(等权全市场)"]
    n_period = A["n_period"]
    cnt = A["uni"].sum(1)
    pool, pool_lo, pool_hi = int(np.nanmean(cnt)), int(cnt.min()), int(cnt.max())
    n_ok = int(stat["方向一致"].sum())
    best = stat["IC"].astype(float)
    top3 = list(best.abs().sort_values(ascending=False).index[:3])
    p0 = A["pairs"].iloc[0]
    d1, d10 = float(tab.loc["D1", "年化"]), float(tab.loc["D10", "年化"])
    s1s, s10s = float(tab.loc["D1", "夏普"]), float(tab.loc["D10", "夏普"])
    sp = A["spread"]["D10/D1"]
    sp_end = float(sp.iloc[-1])
    sp_dd = float((sp / sp.cummax() - 1.0).min())
    rand_top = float((A["rand"] < A["gross"]["comp_ann"]).mean())
    looS = loo["ΔIC"].sort_values(ascending=False)
    base_ann = float(sens.loc["基准配置", "年化收益"])
    sv = (sens["年化收益"].astype(float) - base_ann) * 100
    sv = sv[sv.index != "基准配置"].dropna()
    knob = sv.abs().sort_values(ascending=False)
    cost3 = float(sens.loc["成本x3", "年化收益"]) if "成本x3" in sens.index else np.nan
    slow = list(stat.index[stat["衰减类型"] == "慢因子"])
    fast = list(stat.index[stat["衰减类型"] == "快衰减"])
    drop = [nm for nm in FACTORS if nm not in sel]

    kpis = "".join([
        kpi("%d → %d" % (len(FACTORS), len(sel)), "因子池 → 入闱"),
        kpi("%d" % n_period, "调仓月（%s 起）" % cfg["start"][:4]),
        kpi(format(pool, ","), "池内平均证券数"),
        kpi(fmt_pct(comp["年化收益"]), "合成策略年化"),
        kpi("%.2f" % comp["夏普"], "夏普"),
        kpi(fmt_pct(comp["最大回撤"]), "最大回撤"),
        kpi(fmt_pct(comp["超额年化"]), "超额年化"),
        kpi("%.1fx" % comp["年换手"], "年换手"),
        kpi("%s → %s" % (fmt_pct(d1, 1), fmt_pct(d10, 1)), "分层 D1 → D10"),
        kpi("%.2f×" % sp_end, "多空价差 D10/D1"),
    ])
    nav = "".join('<a href="#s%d">%d %s</a>' % (i, i, t) for i, t in enumerate(
        ["矿区", "原矿", "淘洗", "闯关", "体检", "熔炼", "成色", "尾声"]))

    # ---- 表格 ----
    t_ic = pd.DataFrame({
        "族": stat["族"], "含义": stat["含义"], "窗口": stat["窗口"],
        "IC": best.map(lambda v: "%+.4f" % v),
        "ICIR": stat["ICIR"].astype(float).map(lambda v: "%.2f" % v),
        "IC 胜率": stat["胜率"].astype(float).map(lambda v: "%.0f%%" % (100 * v)),
        "样本内 t": ist.loc[list(stat.index), "样本内t"].astype(float).map(lambda v: "%.2f" % v),
        "样本外 t": ist.loc[list(stat.index), "样本外t"].astype(float).map(lambda v: "%.2f" % v),
        "方向": stat["方向"].astype(float).map(lambda v: "买高值" if v > 0 else "买低值"),
        "样本外一致": stat["方向一致"].map(lambda v: "是" if v else "否"),
        "衰减": stat["衰减类型"], "覆盖率": stat["覆盖率"].astype(float).map(lambda v: "%.0f%%" % (100 * v)),
        "单因子夏普": stat["单因子夏普"].astype(float).map(lambda v: "%.2f" % v),
        "年换手": stat["单因子换手"].astype(float).map(lambda v: "%.1fx" % v),
    })
    t_ic = t_ic.loc[list(best.abs().sort_values(ascending=False).index)]
    t_ic.index.name = "因子"
    t_pair = pd.DataFrame({
        "因子 A": A["pairs"]["因子A"], "因子 B": A["pairs"]["因子B"],
        "相关系数 ρ": A["pairs"]["相关系数"].map(lambda v: "%+.2f" % v),
    }).head(10)
    t_pair.index = range(1, len(t_pair) + 1)
    t_pair.index.name = "#"
    t_dec = pd.DataFrame({
        "年化收益": tab["年化"].map(lambda v: "%.2f%%" % (100 * v)),
        "夏普": tab["夏普"].map(lambda v: "%.2f" % v),
        "最大回撤": tab["最大回撤"].map(lambda v: "%.1f%%" % (100 * v)),
        "年换手": tab["年换手"].map(lambda v: "%.1fx" % v),
        "持仓（中位）": tab["持仓数"].map(lambda v: "%.0f 只" % v),
        "毛收益年化": dec["gross"].map(lambda v: "%.2f%%" % (100 * v)),
    })
    t_dec.index.name = "分层"
    t_loo = pd.DataFrame({
        "去掉后全样本 IC": loo["去掉后IC"].map(lambda v: "%+.4f" % v),
        "去掉后样本外 IC": loo["去掉后样本外IC"].map(lambda v: "%+.4f" % v),
        "ΔIC": loo["ΔIC"].map(lambda v: "%+.4f" % v),
        "Δ样本外 IC": loo["Δ样本外IC"].map(lambda v: "%+.4f" % v),
    })
    t_loo.index.name = "因子"
    t_sub = pd.DataFrame({
        "年化收益": sub["年化收益"].map(lambda v: "%.2f%%" % (100 * v)),
        "夏普": sub["夏普"].map(lambda v: "%.2f" % v),
        "最大回撤": sub["最大回撤"].map(lambda v: "%.2f%%" % (100 * v)),
        "超额年化": sub["超额年化"].map(lambda v: "%.2f%%" % (100 * v)),
        "信息比率": sub["信息比率"].map(lambda v: "%.2f" % v),
    })
    t_sub.index.name = "区间"
    t_map = pd.DataFrame({
        "工序": ["矿区", "原矿", "淘洗", "闯关", "体检", "熔炼", "成色"],
        "在看什么": ["股票池的宽度与稳定性", "因子怎么定义、覆盖多少股票",
                 "IC 的时序、分布与衰减", "两道门：方向一致 + |t| ≥ 1",
                 "因子之间像不像、冗余在哪", "分层单调性、边际贡献、净曲线",
                 "参数、成本、容量、随机对照"],
        "代码": ["lib.py · 1_数据真实性验证/", "factors.py", "s04_vol_volume.py",
               "s04_vol_volume.py", "s07_factor_lab.py", "s04 + s07", "s05_robustness.py"],
    }).set_index("工序")

    cards = "".join(card_html(nm, stat, ic[nm].dropna(), sel, ist)
                    for nm in best.abs().sort_values(ascending=False).index)

    # ---- 各幕文字 ----
    lead0 = (f"回测区间 <b>{cfg['start']} ~ {cfg['end']}</b>，共 {n_period} 个调仓月、"
             f"{len(FACTORS)} 个候选因子。可交易股票池在 {pool_lo:,} ~ {pool_hi:,} 只之间浮动，"
             f"平均 {pool:,} 只 —— 池子的宽度决定了每个横截面统计量有多少个样本，"
             f"也决定了 IC 的噪音地板在哪里。")
    lead1 = (f"13 个因子分两族：<b>波动率族</b> 8 个、<b>成交量族</b> 5 个。卡片按 |IC| 从大到小排；"
             f"斜线是 6 个月滚动平均 IC —— <b>越平越稳</b>，一路往下走说明它只在某一段时间里好使。"
             f"入闱与否由第 3 幕的两道门决定，这里先只把矿石摆出来。")
    lead2 = (f"IC = 调仓日因子值与随后 21 个交易日复权收益的横截面 Spearman 秩相关。"
             f"全样本 IC 最强的是 <b>{top3[0]}</b>（{best[top3[0]]:+.4f}）、"
             f"<b>{top3[1]}</b>（{best[top3[1]]:+.4f}）、<b>{top3[2]}</b>（{best[top3[2]]:+.4f}）—— "
             f"量级都落在 ±0.15 以内。这是低频截面因子的正常体重：谁要是拿出 IC 0.5 的因子，"
             f"多半是样本内挖出来的幻觉。<br>右侧的<b>衰减光谱</b>是这次挖矿最有用的一张图："
             f"把前瞻窗口从 1 天拉到 63 天，<b>{len(slow)}/{len(FACTORS)}</b> 个因子的 |IC| "
             f"到 63 日还在变大 —— 它们不是「周内就被吃掉」的快因子，而是需要拿几周才兑现的"
             f"<b>慢因子</b>（{'、'.join(slow[:4])} 最典型），只有 beta 的峰值落在 21 日附近。"
             f"这条信息直接影响调仓频率：对慢因子来说，月频已经偏快。")
    lead3 = (f"两道门，一道比一道狠：<br>① <b>方向一致性</b> —— 用 2021-2023 定方向、"
             f"2024-2026 检验，13 个里 <b>{n_ok} 个</b>没翻脸；<br>"
             f"② <b>|样本内 t| ≥ 1</b> —— 剩下的再砍掉 t 值不够格的。"
             f"落选：<b>{'、'.join(drop)}</b>。"
             f"其中 amt_cv 是样本外直接翻符号（样本内 +0.0005、样本外 −0.0483），"
             f"beta 与 maxdd 的 t 值只有 0.49 / −0.22，等于噪音。<br>"
             f"入闱的 10 个等权 z-score 合成 —— 注意这里<b>故意不做相关性去重</b>，"
             f"理由在第 5 幕的留一法里。")
    lead4 = (f"13 个因子两两的横截面秩相关（逐日算、再按日平均），按 1−|ρ| 聚类排序。"
             f"最像的一对是 <b>{p0['因子A']} ~ {p0['因子B']}</b>（ρ = {p0['相关系数']:+.2f}）："
             f"一个说「波动大」、一个说「跌得狠」，本质是同一座山。"
             f"成交量族则分成三小簇：量能（amt_trend / vol_trend）、"
             f"流动性（amihud / trade_size）、形态（skew / pv_corr）。<br>"
             f"但<b>相关 ≠ 冗余</b>：留一法会告诉你，同一簇里的因子去掉任何一个，"
             f"合成 IC 都会掉 —— 因为它们在横截面上挑的是不同的一批股票。")
    lead5 = (f"把合成打分切成十分层，每层整层等权买入，走<b>真引擎</b>（涨跌停不可交易、参与率上限 10%、"
             f"含全部成本）。年化从 D1 的 <b>{d1:.1%}</b> 单调升到 D10 的 <b>{d10:.1%}</b>，"
             f"夏普 {s1s:.2f} → {s10s:.2f} —— 单调意味着打分是<b>连续有效</b>的，"
             f"不是只有两头极端才有用；而且 D1 的年化是<b>负的</b>，"
             f"说明这个打分是「两端都分得开」，而不是靠一层层摊薄。<br>"
             f"多空价差 D10/D1 五年做到 <b>{sp_end:.2f}×</b>"
             f"（期间最深回撤 {sp_dd:.1%}），这是不做空也能看的「信号纯度」。<br>"
             f"留一法：去掉 <b>{looS.index[0]}</b> 合成 IC 掉 {looS.iloc[0]:+.4f}，"
             f"去掉 <b>{looS.index[1]}</b> 掉 {looS.iloc[1]:+.4f}；"
             f"而 {looS.index[-1]} 是 −{abs(looS.iloc[-1]):.4f}（去掉它反而更好，"
             f"但等权合成下它是被 ivol 簇带进来的）。")
    lead6 = (f"把 5 个旋钮各拧一下：最脆弱的不是持仓数也不是缓冲带，而是 <b>{knob.index[0]}</b>"
             f"（{knob.iloc[0]:+.2f} pp）。成本 ×3 时年化掉到 {cost3:.2%}，"
             f"说明这条策略的收益有一大块是<b>挣在成本线上</b>的："
             f"年换手 {comp['年换手']:.1f}×、成本占净值 {comp['成本占净值']:.2%}。<br>"
             f"随机对照（同池、同调仓频率、随机抽 50 只等权、毛收益口径、200 组）："
             f"中位数 {np.median(A['rand']):.1%}，"
             f"策略 top50 毛收益 <b>{A['gross']['comp_ann']:.1%}</b>，"
             f"高于 <b>{100 * rand_top:.0f}%</b> 的随机组 —— "
             f"超额不是随机抽签能抽出来的。")
    lead7 = ("把六道工序连起来看，这次挖矿的结论是：<ul class='clean'>"
             f"<li><b>信号是真的，但很瘦。</b>分层单调（D1 {d1:.1%} → D10 {d10:.1%}）、"
             f"样本外 {n_ok}/{len(FACTORS)} 方向不翻脸、毛收益高于 {100 * rand_top:.0f}% 的随机组；"
             f"但 IC 只有 0.1 量级，全部优势都要靠横截面铺得宽（每期 50 只）才攒得出来。</li>"
             f"<li><b>成本决定了它能不能活。</b>年换手 {comp['年换手']:.1f}×，"
             f"成本占净值 {comp['成本占净值']:.2%}；成本 ×3 后年化只剩 {cost3:.2%}。"
             f"换句话说：这条策略赚的钱里，有相当一部分是在跟交易成本抢。</li>"
             f"<li><b>慢因子是主力。</b>{'、'.join(slow[:4])} 在 63 日窗口上仍然走强，"
             f"说明月频调仓对它们来说甚至偏快 —— 降频、加大缓冲带、对慢因子单独用长周期，"
             f"都有空间。</li>"
             f"<li><b>容量是硬顶。</b>参与率压到 1% 时被挡金额 "
             f"{float(sens.loc['参与率1%', '被挡金额_亿']):.0f} 亿、持仓数中位数飙到 "
             f"{float(sens.loc['参与率1%', '持仓数']):.0f} 只 —— 这条策略装不下大钱。</li></ul>")

    nxt = ("<h3>下一层矿脉（这一版没做的）</h3><ul class='clean'>"
           "<li><b>正交化</b>：把 ivol 对 dvol / maxdd / vol_of_vol 回归取残差，"
           "看看波动率簇里还剩多少独立信息（当前 ρ 最高 0.89）。</li>"
           "<li><b>不等权合成</b>：按 ICIR 或半衰期加权，而不是 10 个因子 1/10 等权；"
           "结合衰减光谱给快因子短窗、慢因子长窗。</li>"
           "<li><b>换手惩罚</b>：把目标权重写成「信号 − λ×换手」的优化问题，"
           "而不是现在的缓冲带硬规则。</li>"
           "<li><b>中性化</b>：当前数据里没有行业/市值字段，"
           "加上外部数据后可以压掉板块和市值的隐性暴露。</li>"
           "<li><b>调仓频率匹配</b>：慢因子月频、快因子周频，两个频段分开做，"
           "而不是全池统一月频。</li></ul>")

    gs = lambda a, b: grid2(a, b)
    body = "".join([
        stage(0, "矿区：这块地有多宽", lead0, card(figs["universe"]),
              "股票池是每天动态算的：上市满 250 日、20 日均额 ≥ 2000 万、股价 ≥ 2 元，"
              "再剔除次新、一字板与停牌。池子越宽，横截面 IC 的样本越多、噪音越小。"),
        stage(1, "原矿：13 个候选因子", lead1,
              '<div class="cards">%s</div>%s' % (cards, card(figs["quality"])),
              "覆盖率 = 调仓日上、可投资域里真正算得出该因子值的比例（成交量族依赖 "
              "num_trades，这份数据在少数交易日有缺口，所以 trade_size / amihud 略低）。"
              "信号黏性 = 相邻两个调仓日、同一因子排名的秩相关：黏性低的因子（vol_trend、"
              "amt_trend）排名换得快、年换手自然高，黏性高的（trade_size、maxdd）可以少动。"),
        stage(2, "淘洗：把噪音冲掉", lead2,
              card(figs["ic_heat"]) + gs(figs["ic_decay"], figs["ic_box"]),
              "怎么看这两张图：箱体窄 = IC 稳定；箱体整体偏在一侧 = 有方向；"
              "红色格子（正 IC）代表「因子值大 → 随后涨得多」，蓝色反之。"),
        stage(3, "闯关：两道门与入闱名单", lead3,
              card(figs["sankey"]) + gs(figs["quad"], figs["tstat"]),
              "门坎是在样本内定的（用 2021-2023 定方向、定门坎），样本外只用来检验 —— "
              "这也是为什么第 3 幕的样本外数字才是真数字。"),
        stage(4, "体检：谁跟谁是一家人", lead4,
              gs(figs["corr"], figs["star"]) + table_html(t_pair),
              "相关系数高只说明两个因子「排序相似」，不说明它们在组合里的边际贡献相同；"
              "判断冗余要看留一法（第 5 幕），而不是只看热力图。"),
        stage(5, "熔炼：从排名到净值", lead5,
              card(figs["decile"]) + gs(figs["decile_nav"], figs["spread"]) +
              card(figs["race"]) + card(figs["comp_nav"]) +
              gs(figs["loo"], figs["bubble"]) + card(figs["annual"]) +
              table_html(t_dec) + table_html(t_loo),
              "分层是「整层等权买入」：每层约 450 只、单只权重只有 0.2%。"
              "为了让「跌出这一层」的老仓位真的被卖掉，分层测试把调仓带缩到 1e-5 "
              "（基准的 2e-3 恰好等于单只权重，会让落选的仓位永远留在账上，持仓数滚到几千只）；"
              "涨跌停、参与率、成本、T+1 等约束与基准完全一致。"
              "而实盘策略（合成 top50）持仓集中得多、成本也更重 —— 两张表的差距就是集中度的代价。"),
        stage(6, "成色：压一压它会不会碎", lead6,
              gs(figs["tornado"], figs["random"]) + table_html(t_sub),
              "敏感性表在 out/tables/05_sensitivity.csv，这里只画相对基准配置的变化；"
              "随机对照是毛收益口径（不含成本与撮合约束），与单因子毛收益可比、"
              "与含成本的净值不可直接比。"),
        stage(7, "尾声：六道工序的账", lead7,
              table_html(t_map) + nxt, ""),
    ])

    js = "" if cdn else "<script>%s</script>" % po.get_plotlyjs()
    if cdn:
        js = '<script src="https://cdn.plot.ly/plotly-3.3.1.min.js" charset="utf-8"></script>'
    head = ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>因子挖矿实验室 · Factor Mining Lab</title>' + js +
            '<style>' + CSS + '</style></head><body>')
    top = ('<header><div class="wrap"><div class="kicker">factor mining lab · '
           'class4 backtest</div><h1>因子挖矿实验室<small>13 个因子的淘洗 · 闯关 · 熔炼 · 成色</small></h1>'
           '<p class="lead">这个看板把「挖因子」的全过程摊开：'
           '<b>矿区</b>看股票池有多宽，<b>原矿</b>看 13 个候选因子的配方，'
           '<b>淘洗</b>看 IC 的时序与衰减，<b>闯关</b>用样本外方向与 t 值筛人，'
           '<b>体检</b>看谁跟谁冗余，<b>熔炼</b>把排名变成净值，'
           '<b>成色</b>再拿参数、成本、容量和随机对照压一遍。'
           '所有数字都由 <code>s07_factor_lab.py</code> 现算或直接取自 s04 / s05 的产物，'
           '回测一律走自研引擎（涨跌停不可交易、T+1、参与率上限、真实成本）。</p>'
           '<div class="kpis">' + kpis + '</div></div></header>'
           '<nav><div class="wrap">' + nav + '</div></nav><div class="wrap">')
    foot = ('</div><footer><div class="wrap"><h3>口径</h3>'
            '<p>信号在 t 日收盘产生，t+1 起 3 个交易日内按目标权重在<b>开盘价</b>成交；'
            '涨跌停 / 停牌 / 一字板按方向性规则拒单，单日单票 ≤ 当日成交额 10%，'
            'T+1、佣金 / 过户费 / 印花税 / 滑点 / 冲击成本全部计入。'
            '初始资金 1 亿元，月频调仓，持仓 50 只，缓冲带 2×。'
            '因子只用 t 日及之前的数据，逐日现算，结构上无未来函数。</p>'
            '<h3>复算</h3>'
            '<p><code>python s04_vol_volume.py</code> → <code>python s05_robustness.py</code> '
            '→ <code>python s07_factor_lab.py</code>。'
            '看板用到的中间表全部落在 <code>out/tables/07_*.csv</code>，'
            '文字版小结在 <code>out/factor_lab_report.md</code>。</p>'
            '<h3>数据</h3><p>feathers/ 的 7 个日频矩阵：1643 个交易日 × 5451 只证券的'
            'open / high / low / close / amt / num_trades / cumadj。'
            '数据真实性与外部比对见 1_数据真实性验证/out/data_audit.md。</p>'
            '</div></footer></body></html>')
    return head + top + body + foot

# --------------------------------------------------------------------------- #
# 六、文字版小结
# --------------------------------------------------------------------------- #
def write_md(A: dict, art: dict, path: str) -> None:
    cfg = A["cfg"]
    stat, tab, loo, dec = A["STAT"], A["decile"]["table"], A["loo"], A["decile"]
    ist = art["icstat"]
    comp = art["single"].loc["vv_comp"]
    n_ok = int(stat["方向一致"].sum())
    d1, d10 = float(tab.loc["D1", "年化"]), float(tab.loc["D10", "年化"])
    sp = A["spread"]["D10/D1"]
    looS = loo["ΔIC"].sort_values(ascending=False)
    rand_top = float((A["rand"] < A["gross"]["comp_ann"]).mean())
    knob = ((art["sens"]["年化收益"].astype(float) - float(art["sens"].loc["基准配置", "年化收益"]))
            * 100).drop(labels=["基准配置"]).dropna()
    L = []
    L.append("# 因子挖矿实验室 · 文字版小结\n")
    L.append("> 由 `s07_factor_lab.py` 生成（%s ~ %s，%d 个调仓月、%d 个因子）。"
             "完整交互看板见 `out/factor_lab.html`。\n"
             % (cfg["start"], cfg["end"], A["n_period"], len(FACTORS)))
    L.append("## 0. 一句话\n")
    L.append("- **合成策略**（10 因子等权 z-score → 前 50 只等权，含全部成本）："
             "年化 %.2f%%、夏普 %.2f、最大回撤 %.2f%%、超额年化 %.2f%%、年换手 %.1f×。"
             % (100 * comp["年化收益"], comp["夏普"], 100 * comp["最大回撤"],
                100 * comp["超额年化"], comp["年换手"]))
    L.append("- **信号是真的**：十分层年化从 D1 %.2f%% 单调升到 D10 %.2f%%；"
             "多空价差 %.2f×；样本外 %d/%d 个因子方向不翻脸；"
             "毛收益口径下 top50 高于 %.0f%% 的随机组。"
             % (100 * d1, 100 * d10, float(sp.iloc[-1]), n_ok, len(FACTORS), 100 * rand_top))
    L.append("- **成本是命门**：成本 ×3 时年化只剩 %.2f%%。" % (100 * float(art["sens"].loc["成本x3", "年化收益"])) if "成本x3" in art["sens"].index else "")
    L.append("- **最脆弱 / 最不脆弱的旋钮**：%s %+.2f pp；持仓数从 30 加到 120 反而改善。"
             % (knob.abs().idxmax(), knob.loc[knob.abs().idxmax()]))
    L.append("")
    L.append("## 1. 因子体检表（按 |IC| 排序）\n")
    t = pd.DataFrame({
        "族": stat["族"], "含义": stat["含义"], "IC": stat["IC"].astype(float).round(4),
        "ICIR": stat["ICIR"].astype(float).round(2),
        "IC胜率": stat["胜率"].astype(float).round(3),
        "样本内t": ist.loc[list(stat.index), "样本内t"].astype(float).round(2),
        "样本外t": ist.loc[list(stat.index), "样本外t"].astype(float).round(2),
        "样本外一致": stat["方向一致"].map(lambda v: "是" if v else "否"),
        "覆盖率": stat["覆盖率"].astype(float).round(3),
        "衰减类型": stat["衰减类型"], "单因子夏普": stat["单因子夏普"].astype(float).round(2),
        "入闱": [nm in A["sel"] for nm in stat.index],
    })
    t = t.loc[stat["IC"].astype(float).abs().sort_values(ascending=False).index]
    t.index.name = "因子"
    L.append(md_table(t))
    L.append("")
    L.append("## 2. IC 衰减光谱（不同前瞻窗口的平均 IC）\n")
    L.append(md_table(A["decay"].round(4).rename(columns=lambda h: "%d 日" % h)))
    L.append("")
    L.append("## 3. 十分层回测（真引擎、含成本）\n")
    L.append(md_table(pd.DataFrame({
        "年化收益": tab["年化"].round(4), "夏普": tab["夏普"].round(3),
        "最大回撤": tab["最大回撤"].round(4), "年换手": tab["年换手"].round(2),
        "持仓数": tab["持仓数"].round(0),
        "毛收益年化": dec["gross"].round(4)})))
    L.append("")
    L.append("## 4. 留一法边际贡献\n")
    L.append(md_table(loo.round(4)))
    L.append("")
    L.append("## 5. 因子相关性最高的 10 对\n")
    L.append(md_table(A["pairs"].head(10).set_index(pd.Index(
        ["%s ~ %s" % (a, b) for a, b in zip(A["pairs"]["因子A"].head(10),
                                           A["pairs"]["因子B"].head(10))], name="因子对"))
        [["相关系数"]].round(3)))
    L.append("")
    L.append("## 6. 随机对照与敏感性\n")
    L.append("- 随机对照（同池同频、随机抽 50 只、毛收益口径、%d 组）：中位数 %.2f%%，"
             "5%% 分位 %.2f%%，95%% 分位 %.2f%%；合成 top50 毛收益 %.2f%%，高于 %.0f%% 的随机组。"
             % (len(A["rand"]), 100 * np.median(A["rand"]), 100 * np.percentile(A["rand"], 5),
                100 * np.percentile(A["rand"], 95), 100 * A["gross"]["comp_ann"], 100 * rand_top))
    L.append("- 参数敏感性（相对基准配置的年化变化，百分点）：")
    L.append(md_table(knob.sort_values().round(2).to_frame("Δ年化 pp")))
    L.append("")
    L.append("## 7. 口径\n")
    L.append("- 信号 t 日收盘产生，t+1 起 3 个交易日内按开盘价成交；涨跌停 / 停牌按方向性规则拒单；"
             "单日单票 ≤ 成交额 10%%，T+1，含佣金 / 过户费 / 印花税 / 滑点 / 冲击成本。")
    L.append("- 初始资金 1 亿元，月频调仓，持仓 50 只，缓冲带 2×；因子只用 t 日及之前数据。")
    L.append("- 中间表：`out/tables/07_*.csv`。")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join([x for x in L if x is not None]) + "\n")


# --------------------------------------------------------------------------- #
# 七、主流程
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="因子挖矿实验室：把挖因子的过程可视化")
    ap.add_argument("--cdn", action="store_true",
                    help="plotly.js 走 CDN，看板体积小约 5 MB（打开时需联网）")
    ap.add_argument("--open", action="store_true", help="生成后用默认浏览器打开看板")
    args = ap.parse_args(argv)
    t0 = time.time()

    print("[1/4] 读数据与 s04 / s05 的产物 ...")
    md = MarketData(verbose=False)
    art = load_artifacts()
    cfg = art["cfg"]

    print("[2/4] 铺因子面板 ...")
    ts = sorted(Backtest._rebalance_index(md.dates, cfg["rebal"]))
    ts = [t for t in ts if md.dates[t] >= pd.Timestamp(cfg["start"])]
    raw, zsc = build_factor_panels(md, FACTORS, ts, cache_dir=FUC, verbose=False)

    print("[3/4] 补算：相关性 / IC 衰减 / 十分层回测 / 留一法 / 随机对照 ...")
    A = analyze(md, ts, raw, zsc, art)
    bench_nav = (1.0 + md.equal_weight_ret()).cumprod()
    bench_nav = bench_nav.loc[cfg["start"]:cfg["end"]] * 1e8

    print("[4/4] 出图与拼装看板 ...")
    stat = A["STAT"]
    figs = dict(
        universe=fig_html(fig_universe(md, cfg), "f_universe"),
        quality=fig_html(fig_quality(stat), "f_quality"),
        ic_heat=fig_html(fig_ic_heatmap(art["ic"], cfg["is_end"]), "f_icheat"),
        ic_decay=fig_html(fig_ic_decay(A["decay"], stat), "f_decay"),
        ic_box=fig_html(fig_ic_box(art["ic"]), "f_icbox"),
        sankey=fig_html(fig_sankey(stat, art["icstat"], A["sel"]), "f_sankey"),
        quad=fig_html(fig_quadrant(stat), "f_quad"),
        tstat=fig_html(fig_tstat(art["icstat"]), "f_tstat"),
        corr=fig_html(fig_corr_heatmap(A["corr"], A["order"]), "f_corr"),
        star=fig_html(fig_star(A["corr"], stat), "f_star"),
        race=fig_html(fig_race(art, bench_nav, stat), "f_race"),
        bubble=fig_html(fig_bubble(stat), "f_bubble"),
        decile=fig_html(fig_decile(A["decile"]), "f_decile"),
        decile_nav=fig_html(fig_decile_nav(A["decile"], A["spread"]), "f_dnav"),
        spread=fig_html(fig_spread(A["spread"]), "f_spread"),
        loo=fig_html(fig_loo(A["loo"], A["comp_stats"]), "f_loo"),
        comp_nav=fig_html(fig_comp_nav(art, bench_nav), "f_cnav"),
        annual=fig_html(fig_annual(art["comp_nav"], bench_nav), "f_annual"),
        tornado=fig_html(fig_tornado(art["sens"]), "f_tornado"),
        random=fig_html(fig_random(A["rand"], A["gross"]["comp_ann"],
                                   A["gross"]["market_ann"]), "f_random"),
    )
    html = build_html(A, art, figs, cdn=args.cdn)
    with open(OUT_HTML, "w", encoding="utf-8") as fh:
        fh.write(html)

    # ---- 中间表落盘 ----
    savetab(stat, "07_factor_stats.csv")
    savetab(A["corr"].round(4), "07_factor_corr.csv")
    savetab(A["pairs"].round(4), "07_factor_pairs.csv")
    savetab(A["decay"].round(4), "07_ic_decay.csv")
    savetab(A["decile"]["table"].round(4), "07_decile.csv")
    savetab(A["decile"]["gross"].round(4).to_frame("毛收益年化"), "07_decile_gross.csv")
    savetab(A["spread"].round(4), "07_decile_spread.csv")
    savetab(A["loo"].round(4), "07_loo.csv")
    rand = A["rand"]
    savetab(pd.Series({"组数": len(rand), "中位数": float(np.median(rand)),
                       "5% 分位": float(np.percentile(rand, 5)),
                       "95% 分位": float(np.percentile(rand, 95)),
                       "最好一组": float(rand.max()), "最差一组": float(rand.min()),
                       "合成 top50 毛收益": A["gross"]["comp_ann"],
                       "全市场等权毛收益": A["gross"]["market_ann"]}).to_frame("年化"),
            "07_random_control.csv")
    write_md(A, art, OUT_MD)

    tab = A["decile"]["table"]
    comp = art["single"].loc["vv_comp"]
    print("\n看板：%s（%.1f MB）" % (OUT_HTML, os.path.getsize(OUT_HTML) / 1e6))
    print("小结：%s" % OUT_MD)
    print("中间表：out/tables/07_*.csv")
    print("—— 合成策略：年化 %.2f%%、夏普 %.2f、回撤 %.2f%%、年换手 %.1fx"
          % (100 * comp["年化收益"], comp["夏普"], 100 * comp["最大回撤"], comp["年换手"]))
    print("—— 分层：D1 %.2f%% → D10 %.2f%%（夏普 %.2f → %.2f）"
          % (100 * tab.loc["D1", "年化"], 100 * tab.loc["D10", "年化"],
             tab.loc["D1", "夏普"], tab.loc["D10", "夏普"]))
    print("—— 入闱 %d/%d，总用时 %.0fs" % (len(A["sel"]), len(FACTORS), time.time() - t0))
    if args.open:
        import webbrowser
        webbrowser.open("file:///" + OUT_HTML.replace("\\", "/"))
    return A, art


if __name__ == "__main__":
    main()
