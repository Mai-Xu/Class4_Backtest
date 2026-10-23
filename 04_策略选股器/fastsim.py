# -*- coding: utf-8 -*-
"""组合策略选股器 · 快速回测口径（交互级，纯 numpy，不依赖 core.lib）。

为什么要单独写一套口径
----------------------
`core.engine` 是**逐笔撮合**的实盘级引擎：T+1、涨跌停/一字板挡单、参与率上限、
执行窗口内逐日补单，一次 1700 只、21 个月的回测要 80 秒左右 —— 交互式 WebUI 等不起。
本模块用向量化口径复刻同一套**记账规则**，把单次回测压到 10 毫秒级：

| 项目 | 与 core.engine 是否一致 |
| --- | --- |
| 复权份额记账（units × 复权价，不用「原始股数 × 复权价」） | 一致 |
| 信号 t 日收盘产生 → t+1 日**开盘**成交（复权开盘价） | 一致 |
| 目标总仓位 98%（gross_limit，留 2% 现金缓冲） | 一致 |
| 调仓死区 band = max(min(2e5, 0.2×cash/N), min(band_frac/N, 2e-3)×nav) | 一致 |
| 成本：佣金单边万一 + 印花税万五（卖出）、先卖后买、现金不够按比例缩 | 一致 |
| T+1 / 涨跌停 / 停牌 / 参与率上限 / 执行窗口逐日补单 | **不模拟**（偏差见 verify_vs_engine.py） |
| 末日清仓 | 引擎受死区影响会留残仓，这里**强制清空** |

实测偏差（`verify_vs_engine.py`，同一分数面板、同一参数）见 `out/04_策略选股器/tables/`。
"""
from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------ 口径常量
CASH0 = 1e9                 # 初始资金 10 亿元
COMMISSION = 1e-4           # 佣金单边万一
STAMP = 5e-4                # 印花税万五（卖出）
FEE_BUY = COMMISSION
FEE_SELL = COMMISSION + STAMP
GROSS_LIMIT = 0.98          # 目标总仓位（与引擎一致）
BAND_FRAC = 0.25            # 调仓死区（单票，相对 1/N 的倍数）
MIN_NOTIONAL_CAP = 2e5      # 单票最小成交额上限
T252 = 252.0


# ------------------------------------------------------------------ 横截面工具
def rank_z(x, ok=None):
    """横截面秩 z：0~1 分位线性映射到 ±√3（≈ ±1.732），NaN/掩码外保持 NaN。

    与 mining_core.cs_rank_z 同口径（平均秩），抗极值、不假设分布。
    """
    x = np.asarray(x, dtype="float64")
    out = np.full(x.shape, np.nan)
    m = np.isfinite(x)
    if ok is not None:
        m &= np.asarray(ok, dtype=bool)
    if int(m.sum()) < 3:
        return out
    v = x[m]
    # 平均秩（处理并列）
    order = np.argsort(v, kind="stable")
    r = np.empty(v.size, dtype="float64")
    r[order] = np.arange(v.size, dtype="float64")
    uq, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    avg = (csum - cnt + csum - 1) / 2.0
    r = avg[inv]
    out[m] = (r / max(v.size - 1.0, 1.0) - 0.5) * 3.4641016151377544   # ≈ 2√3
    return out


def composite_score(z_stack, weights, signs):
    """多因子合成分数：Σ (w_i/Σw) · s_i · z_i。

    z_stack : (k, n) 当日横截面秩 z（池外/缺失已置 0）
    weights : (k,) 均为正数；按和归一
    signs   : (k,) 因子方向（+1/-1，用样本内 IC 符号确定）
    """
    Z = np.asarray(z_stack, dtype="float64")
    w = np.asarray(weights, dtype="float64").reshape(-1)
    s = np.asarray(signs, dtype="float64").reshape(-1)
    tot = float(np.abs(w).sum())
    if tot <= 0:
        raise ValueError("权重之和必须为正")
    cw = (w / tot) * s
    return np.tensordot(cw, Z, axes=(0, 0))


def select_weights(score, ok, n_hold, prev=None, buffer=1.0, max_weight=0.02,
                   gross=GROSS_LIMIT, min_dispersion=1e-12):
    """取分数前 n_hold 名等权（带缓冲带）。返回 (权重向量, 入选下标)。

    缓冲带：老持仓只有跌出前 n_hold×buffer 名才被踢出（降低换手），
    与本项目 s23 的 ScoreStrategy 完全同规则。

    **横截面无差异**（例如日历因子在非事件月恒为 0）：当日横截面上没有任何可比信息，
    此时不按「代码顺序」乱选，而是返回上一期持仓（无 prev 则返回空仓）。
    """
    s = np.asarray(score, dtype="float64")
    ok = np.asarray(ok, dtype=bool)
    cand = np.flatnonzero(ok & np.isfinite(s))
    w = np.zeros(s.shape, dtype="float64")
    if cand.size == 0:
        return w, np.empty(0, dtype="int64")
    v = s[cand]
    if float(v.max() - v.min()) <= min_dispersion:
        if prev is None or len(prev) == 0:
            return w, np.empty(0, dtype="int64")
        keep = np.unique(np.asarray(prev, dtype="int64"))
        keep = keep[ok[keep] & np.isfinite(s[keep])]
        if keep.size == 0:
            return w, np.empty(0, dtype="int64")
        w[keep] = 1.0 / keep.size
        if max_weight and max_weight > 0:
            w = np.minimum(w, max_weight)
        w = w / float(w.sum()) * float(gross)
        return w, keep
    n = int(min(n_hold, cand.size))
    order = cand[np.argsort(-s[cand], kind="stable")]
    rank = np.full(s.shape, 10 ** 9, dtype="int64")
    rank[order] = np.arange(order.size)
    if prev is None or len(prev) == 0 or not buffer or buffer <= 1.0:
        pick = order[:n]
    else:
        prev = np.asarray(prev, dtype="int64")
        exit_rank = int(min(max(round(n * float(buffer)), n + 1), order.size))
        keep = [int(j) for j in prev if ok[j] and rank[j] < exit_rank]
        pick_set = set(keep)
        for j in order:
            if len(pick_set) >= n:
                break
            pick_set.add(int(j))
        pick = np.array(sorted(pick_set), dtype="int64")
    w[pick] = 1.0 / len(pick)
    if max_weight and max_weight > 0:
        w = np.minimum(w, max_weight)
    tot = float(w.sum())
    if tot > 0:
        w = w / tot * float(gross)          # 与引擎 _normalize 的 gross_limit 一致
    return w, pick


def build_plan(scores, sig_dates, trade_dates, uni, n_hold, buffer=1.0,
               max_weight=0.02, gross=GROSS_LIMIT, liquidate=True,
               liquidate_signal=None):
    """把每个信号日的打分化成「执行日 -> 目标权重」的调仓计划。

    scores : (n_sig, n_sym) 合成分数
    uni    : (n_sig, n_sym) bool 股票池掩码（信号日当日口径）
    """
    dates = np.asarray([np.datetime64(str(d)[:10]) for d in sig_dates])
    tdates = np.asarray([np.datetime64(str(d)[:10]) for d in trade_dates])
    plan, log, prev = {}, [], None
    for i, sd in enumerate(dates):
        d = int(np.searchsorted(tdates, sd, side="right"))   # 信号日之后第一个交易日开盘执行
        if d >= len(tdates):
            continue
        cand = np.flatnonzero(np.asarray(uni[i], dtype=bool) & np.isfinite(scores[i]))
        flat = (cand.size == 0) or float(np.ptp(np.asarray(scores[i])[cand])) <= 0.0
        if flat:
            # 横截面无差异（因子当期为常数，如日历因子在非事件月）：不调仓，不下单
            log.append(dict(signal=str(pd_ts(sig_dates[i])), exec=str(pd_ts(trade_dates[d])),
                            n_hold=None, note=("当期横截面无差异 · 不调仓"
                                               if (prev is not None and len(prev))
                                               else "当期横截面无差异 · 空仓等待")))
            continue
        w, pick = select_weights(scores[i], uni[i], n_hold, prev=prev,
                                 buffer=buffer, max_weight=max_weight, gross=gross)
        plan[d] = w
        prev = pick
        log.append(dict(signal=str(pd_ts(sig_dates[i])), exec=str(pd_ts(trade_dates[d])),
                        n_hold=int(len(pick))))
    if liquidate:
        if liquidate_signal is None:
            last = dates[-1]
        else:
            last = np.datetime64(str(liquidate_signal)[:10])
        d = int(np.searchsorted(tdates, last, side="right"))
        if d < len(tdates):
            plan[d] = np.zeros(uni.shape[1], dtype="float64")
            log.append(dict(signal=str(liquidate_signal), exec=str(pd_ts(trade_dates[d])),
                            n_hold=0))
    return plan, log


def pd_ts(x):
    return np.datetime_as_string(np.datetime64(str(x)[:10]))


# ------------------------------------------------------------------ 回测
def simulate(plan, adj_open, adj_close, cash0=CASH0, fee_buy=FEE_BUY, fee_sell=FEE_SELL,
             band_frac=BAND_FRAC, n_hold=100, min_notional_cap=MIN_NOTIONAL_CAP):
    """按调仓计划逐日推进，返回净值/换手/持仓序列。

    plan : {执行日下标: 目标权重向量}；空权重向量 = 清仓。
    adj_open/adj_close : (n_day, n_sym) 复权价，行 = 交易日。
    """
    n_day, n_sym = adj_open.shape
    units = np.zeros(n_sym, dtype="float64")
    cash = float(cash0)
    nav = np.empty(n_day, dtype="float64")
    traded = np.zeros(n_day, dtype="float64")
    n_pos = np.zeros(n_day, dtype="int64")
    band = max(min(band_frac / max(int(n_hold), 1), 2e-3) * 1.0, 0.0)
    min_notional = min(min_notional_cap, 0.2 * cash0 / max(int(n_hold), 1))

    for d in range(n_day):
        w = plan.get(d)
        if w is not None:
            px = adj_open[d]
            ok = np.isfinite(px) & (px > 0)
            val = np.where(ok, units * px, 0.0)
            nav_pre = cash + float(val.sum())
            if nav_pre > 0:
                force = float(np.abs(w).sum()) <= 0.0        # 清仓信号：不受死区限制
                thr = 0.0 if force else max(min_notional, band * nav_pre)
                delta = np.where(ok, np.asarray(w, dtype="float64") * nav_pre - val, 0.0)
                delta = np.where(np.abs(delta) > thr, delta, 0.0)
                sell_m = delta < 0
                if sell_m.any():
                    sv = -delta[sell_m]
                    cash += float(sv.sum()) * (1.0 - fee_sell)
                    units[sell_m] -= sv / px[sell_m]
                    traded[d] += float(sv.sum())
                buy_m = delta > 0
                if buy_m.any():
                    bv = delta[buy_m]
                    want = float(bv.sum())
                    room = cash / (1.0 + fee_buy)
                    scale = 1.0 if want <= room else (room / want if want > 0 else 0.0)
                    spend = want * scale
                    cash -= spend * (1.0 + fee_buy)
                    units[buy_m] += (bv * scale) / px[buy_m]
                    traded[d] += spend
        cl = adj_close[d]
        mv = float(np.nansum(np.where(np.isfinite(cl), units * cl, 0.0)))
        nav[d] = cash + mv
        if nav[d] > 0:
            traded[d] = traded[d] / nav[d]                  # 存成换手率（与引擎 daily.turnover 同口径）
        n_pos[d] = int((np.where(np.isfinite(cl), units * cl, 0.0) > 1.0).sum())

    ret = np.zeros(n_day, dtype="float64")
    ret[1:] = nav[1:] / nav[:-1] - 1.0
    return dict(nav=nav, ret=ret, turnover=traded, n_holdings=n_pos)


# ------------------------------------------------------------------ 绩效
def nav_ratio(nav, cash0=CASH0):
    """净值曲线（起点 = 1.0）：补上建仓前基准点，并把绝对金额换算成净值。

    项目里所有净值序列都是「425 个交易日 = 1.6865 年」的口径（含 2024-12-31 这个起点，
    当时组合还是现金），补上它年化才与 02/03 的报告逐位对齐。
    """
    nav = np.asarray(nav, dtype="float64")
    return np.concatenate([[1.0], nav / float(cash0)])


def drawdown(nav):
    nav = np.asarray(nav, dtype="float64")
    peak = np.maximum.accumulate(nav)
    return nav / peak - 1.0


def perf_stats(nav, ret=None):
    """与 core.metrics.perf_stats 同口径（夏普 = 年化收益 / 年化波动）。"""
    nav = np.asarray(nav, dtype="float64")
    if ret is None:
        ret = np.zeros_like(nav)
        ret[1:] = nav[1:] / nav[:-1] - 1.0
    n = len(nav)
    if n < 2 or nav[0] <= 0:
        return {}
    years = n / T252
    total = float(nav[-1] / nav[0] - 1.0)
    cagr = float((nav[-1] / nav[0]) ** (1.0 / years) - 1.0)
    vol = float(np.std(ret, ddof=1) * np.sqrt(T252))
    dd = drawdown(nav)
    mdd = float(dd.min())
    neg = ret[ret < 0]
    downside = float(np.std(neg, ddof=1) * np.sqrt(T252)) if neg.size > 1 else 0.0
    return dict(累计收益=total, 年化收益=cagr, 年化波动=vol,
                夏普=(cagr / vol if vol > 0 else float("nan")),
                索提诺=(cagr / downside if downside > 0 else float("nan")),
                最大回撤=mdd, 卡玛=(cagr / abs(mdd) if mdd < 0 else float("nan")),
                日胜率=float((ret > 0).mean()),
                最好一日=float(ret.max()), 最差一日=float(ret.min()),
                年数=float(years))


def tracking_stats(ret, bench_ret):
    """相对基准的超额年化 / 跟踪误差 / 信息比率（按年度化收益差算超额）。"""
    r = np.asarray(ret, dtype="float64")
    b = np.asarray(bench_ret, dtype="float64")
    n = min(len(r), len(b))
    r, b = r[-n:], b[-n:] if n else (r, b)
    if n < 2:
        return dict(超额年化=float("nan"), 跟踪误差=float("nan"), 信息比率=float("nan"))
    years = n / T252
    ca = float(np.prod(1.0 + r) ** (1.0 / years) - 1.0)
    cb = float(np.prod(1.0 + b) ** (1.0 / years) - 1.0)
    te = float(np.std(r - b, ddof=1) * np.sqrt(T252))
    return dict(超额年化=ca - cb, 跟踪误差=te,
                信息比率=((ca - cb) / te if te > 0 else float("nan")))


def turnover_ann(turnover, n_days=None):
    """年化换手：Σ(单日单边成交额/净值) ÷ 年数（与 02/03 报告的 annual_turnover 同口径）。"""
    t = np.asarray(turnover, dtype="float64")
    n = len(t) if n_days is None else int(n_days)
    return float(t.sum() / (n / T252)) if n > 0 else float("nan")


def aggregate_returns(ret, dates, freq="M"):
    """按月/按年聚合收益（几何连乘）。返回 [(期间, 收益), ...]。"""
    import pandas as pd
    s = pd.Series(np.asarray(ret, dtype="float64"), index=pd.DatetimeIndex(dates))
    freq = {"ME": "M", "YE": "Y", "QE": "Q"}.get(str(freq).upper(), freq)
    g = s.groupby(s.index.to_period(freq))
    return [(str(k), float((1.0 + v).prod() - 1.0)) for k, v in g]
