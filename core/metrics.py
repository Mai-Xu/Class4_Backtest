# -*- coding: utf-8 -*-
"""绩效与执行质量指标。"""
from __future__ import annotations

import numpy as np
import pandas as pd

T252 = 252.0


def drawdown(nav: pd.Series) -> pd.Series:
    return nav / nav.cummax() - 1.0


def max_drawdown_detail(nav: pd.Series) -> dict:
    dd = drawdown(nav)
    if len(dd) == 0:
        return dict(mdd=0.0, peak=None, trough=None, recover=None, days_to_trough=0, days_to_recover=None)
    trough = dd.idxmin()
    mdd = float(dd.min())
    peak = nav.loc[:trough].idxmax()
    after = nav.loc[trough:]
    rec = after[after >= nav.loc[peak]]
    recover = rec.index[0] if len(rec) else None
    return dict(mdd=mdd, peak=peak, trough=trough, recover=recover,
                days_to_trough=int(nav.loc[peak:trough].shape[0]),
                days_to_recover=(None if recover is None else int(nav.loc[trough:recover].shape[0])))


def perf_stats(nav: pd.Series, ret: pd.Series | None = None, rf: float = 0.0,
               freq: float = T252) -> dict:
    """净值曲线的标准绩效指标。ret 缺省由 nav 计算。"""
    nav = nav.dropna()
    if ret is None:
        ret = nav.pct_change().fillna(0.0)
    ret = ret.reindex(nav.index).fillna(0.0)
    n = len(nav)
    if n < 2:
        return {}
    years = n / freq
    total = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    cagr = float((nav.iloc[-1] / nav.iloc[0]) ** (1.0 / years) - 1.0) if nav.iloc[0] > 0 else np.nan
    vol = float(ret.std(ddof=1) * np.sqrt(freq))
    downside = ret[ret < 0].std(ddof=1) * np.sqrt(freq) if (ret < 0).any() else 0.0
    sharpe = (cagr - rf) / vol if vol > 0 else np.nan
    sortino = (cagr - rf) / downside if downside > 0 else np.nan
    d = max_drawdown_detail(nav)
    mdd = d["mdd"]
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    return dict(累计收益=total, 年化收益=cagr, 年化波动=vol, 夏普=sharpe, 索提诺=sortino,
                最大回撤=mdd, 卡玛=calmar,
                回撤峰=str(d["peak"])[:10] if d["peak"] is not None else "",
                回撤谷=str(d["trough"])[:10] if d["trough"] is not None else "",
                修复日=str(d["recover"])[:10] if d["recover"] is not None else "未修复",
                回撤天数=d["days_to_trough"], 修复天数=d["days_to_recover"],
                日胜率=float((ret > 0).mean()), 日波动=float(ret.std(ddof=1)),
                偏度=float(ret.skew()), 峰度=float(ret.kurtosis()),
                最好一日=float(ret.max()), 最差一日=float(ret.min()),
                年数=float(years))


def perf_table(stats: dict, pct_keys=None) -> pd.DataFrame:
    """把指标字典整理成两列（指标 / 数值）便于写入报告。"""
    pct_keys = pct_keys or {"累计收益", "年化收益", "年化波动", "最大回撤", "日胜率",
                            "日波动", "最好一日", "最差一日"}
    rows = []
    for k, v in stats.items():
        if isinstance(v, float) and np.isfinite(v):
            v2 = ("%.2f%%" % (100 * v)) if k in pct_keys else ("%.3f" % v)
        else:
            v2 = str(v)
        rows.append((k, v2))
    return pd.DataFrame(rows, columns=["指标", "数值"]).set_index("指标")


def excess_stats(ret: pd.Series, bench_ret: pd.Series, rf: float = 0.0) -> dict:
    """相对基准的表现。"""
    a, b = ret.align(bench_ret, join="inner")
    if len(a) < 2:
        return {}
    act = (a - b).dropna()
    nav_a = (1 + a).cumprod()
    nav_b = (1 + b).cumprod()
    years = len(a) / T252
    cagr_a = nav_a.iloc[-1] ** (1 / years) - 1
    cagr_b = nav_b.iloc[-1] ** (1 / years) - 1
    te = float(act.std(ddof=1) * np.sqrt(T252))
    return dict(基准累计=float(nav_b.iloc[-1] - 1), 基准年化=float(cagr_b),
                超额年化=float(cagr_a - cagr_b),
                跟踪误差=te,
                信息比率=float((cagr_a - cagr_b) / te) if te > 0 else np.nan,
                月胜率=float((a.resample("ME").apply(lambda x: (1 + x).prod() - 1)
                              > b.resample("ME").apply(lambda x: (1 + x).prod() - 1)).mean()))


def annual_table(ret: pd.Series, bench_ret: pd.Series | None = None) -> pd.DataFrame:
    """分年度收益、波动与超额。"""
    g = ret.groupby(ret.index.year)
    tab = pd.DataFrame({
        "收益": g.apply(lambda x: (1 + x).prod() - 1),
        "波动": g.apply(lambda x: x.std(ddof=1) * np.sqrt(T252)),
        "交易日": g.size(),
    })
    if bench_ret is not None:
        bb = bench_ret.reindex(ret.index).groupby(ret.index.year).apply(lambda x: (1 + x).prod() - 1)
        tab["基准"] = bb
        tab["超额"] = tab["收益"] - tab["基准"]
    return tab


def execution_report(result, initial_cash: float | None = None) -> pd.DataFrame:
    """执行质量：被挡下的交易、参与率、成本。"""
    tr, bl = result.trades, result.blocks
    rows = []
    rows.append(("成交笔数", f"{len(tr):,}" if len(tr) else "0"))
    rows.append(("成交金额（万元）", f"{tr['dv'].sum() / 1e4:,.0f}" if len(tr) else "0"))
    if len(tr):
        rows.append(("单笔成交金额中位数（万元）", f"{tr['dv'].median() / 1e4:,.1f}"))
        rows.append(("平均滑点（bp）", f"{tr['slip_bps'].mean():.2f}"))
        rows.append(("总费用（万元）", f"{tr['fee'].sum() / 1e4:,.1f}"))
    if len(bl):
        g = bl.groupby("reason")["dv"].agg(["size", "sum"])
        g = g.sort_values("sum", ascending=False)
        for reason, r in g.iterrows():
            rows.append((f"被挡：{reason}（笔 / 万元）",
                         f"{int(r['size']):,} / {r['sum'] / 1e4:,.0f}"))
    else:
        rows.append(("被挡：无", "0 / 0"))
    return pd.DataFrame(rows, columns=["项目", "数值"]).set_index("项目")
