# -*- coding: utf-8 -*-
"""策略选股器 · 预计算 UI 数据（离线跑一次，约 1~3 分钟）。

产物（`out/cache/selector/`，可再生、不入库）
------------------------------------------------
| 文件 | 内容 |
| --- | --- |
| `meta.json` | 口径常量 / 信号日 / 交易日 / 股票池规模 / 7 大家族说明 |
| `factor_meta.json` | 390 个因子：家族 / 口径 / 方向 / 样本内 IC·ICIR / 分层多空 / 样本内策略绩效（截止 2024 年底） |
| `z/<因子>.npy` | 该因子在股票池内的**当日横截面秩 z**（信号日 × 证券，float32，池外/缺失=0） |
| `universe.npy` | 股票池掩码（信号日 × 证券，bool） |
| `prices_oos.npz` | 回测区间复权开/收盘价（交易日 × 证券，float32） |
| `benchmarks.json` | 全市场等权 / 最佳线性打分（MF1700_raw）/ 最佳树模型打分（ML1700）日净值 |

为什么这么存：WebUI 每次计算只需要**被选中那几个因子**的 z 面板（每个 ~440KB），
读取是毫秒级；390 个全量面板（约 1.8GB）留在 `out/cache/zoo2/` 不动。

历史表现口径：每月末打分前 1700 只等权（项目主推的宽持仓用法），2021-01-29 收盘建仓、
2024-12-31 收盘计值，期间不调仓费率以外不加任何约束（与 WebUI 同一套 `fastsim` 口径）。
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "03_因子挖掘"))
os.environ.setdefault("CLASS4_TRACK", "04_策略选股器")

import fastsim as fs                                     # noqa: E402
import mining_core as mc                                 # noqa: E402
from core.lib import MarketData, CACHE                   # noqa: E402
from factor_zoo import api as zoo_api                    # noqa: E402

SEL = os.path.join(CACHE, "selector")
ZDIR = os.path.join(SEL, "z")

IS_N_HOLD = 1700          # 因子历史表现展示口径：每月打分前 1700 只等权
IS_BASE = "2021-01-29"    # 面板第一个信号日（= 样本内建仓信号日）

FAMILY_DESC = {
    "动量与反转": "过去涨得多/跌得多的股票，接下来是延续还是回摆。含短期反转、3-1/6-1/12-1 动量、52 周高点、彩票效应、日内与隔夜拆解。",
    "波动率与风险": "波动、beta、特质风险、偏度峰度、VaR/CVaR、区间波动率、跳跃占比。低波动本身在 A 股长期是一类稳定的风险溢价。",
    "流动性与交易活跃度": "成交额水平/趋势/波动、Amihud 非流动性、Amivest、Kyle λ、买卖价差代理、单笔均额、停牌天数。",
    "量价关系与资金流": "量价相关性、成交额加权收益、OBV/MFI/CMF/PVT、VWAP 偏离、量价背离、大单代理，以及筹码分布代理（获利盘/套牢盘/成本乖离/集中度）。",
    "趋势与形态": "均线偏离与排列、RSI/KDJ/CCI、MACD/TRIX、Aroon、DMI/ADX、唐奇安与布林通道、线性回归斜率与 R²、K 线形态、路径熵。",
    "日历与季节性": "月末月初、季末年末、长假前后、周内效应、月份效应与季节性动量——全部写成「日历状态 × 个股状态」以保证横截面有差异。",
    "交互与合成": "把两个家族的信息乘起来：反转×非流动性、反转×低波、动量×低活跃、彩票×非流动性、偏度×波动、筹码×反转、多口径反转共识等。",
}

# ---------------------------------------------------------------- 子进程全局
_G: dict = {}


def _init_worker(cfg: dict):
    import fastsim as _fs
    _G["fs"] = _fs
    _G["cfg"] = cfg
    z = np.load(cfg["is_prices"])
    _G["adj_open"], _G["adj_close"] = z["adj_open"], z["adj_close"]


def _work(job):
    nm, path, direction = job
    fs_, cfg = _G["fs"], _G["cfg"]
    P = pd.read_parquet(path)
    Pm = P.values.astype("float64")
    row_of = {str(d.date()): i for i, d in enumerate(P.index)}   # key = 'YYYY-MM-DD'
    sign = 1.0 if (direction is None or not np.isfinite(direction) or direction >= 0) else -1.0

    # ---------- 样本内历史表现：秩 z → 猜分 → 前 1700 只等权 ----------
    Z_is = np.zeros((len(cfg["is_sig"]), Pm.shape[1]), dtype="float64")
    for i, d in enumerate(cfg["is_sig"]):
        Z_is[i] = fs_.rank_z(Pm[row_of[d]], cfg["uni_is"][i])
    score_is = sign * np.nan_to_num(Z_is, nan=0.0)
    plan_is, _ = fs_.build_plan(score_is, cfg["is_sig"], cfg["is_trade_dates"], cfg["uni_is"],
                                IS_N_HOLD, liquidate=False)
    res = fs_.simulate(plan_is, _G["adj_open"], _G["adj_close"], n_hold=IS_N_HOLD)
    nav = fs_.nav_ratio(res["nav"])            # 起点 = 1.0（2021-01-29 收盘时还是现金）
    st = fs_.perf_stats(nav)
    ret = nav[1:] / nav[:-1] - 1.0
    mret = fs_.aggregate_returns(ret, pd.DatetimeIndex(cfg["is_trade_dates"]), freq="ME")

    # ---------- 样本外：只存 z 面板 ----------
    Z_oos = np.zeros((len(cfg["sig_oos"]), Pm.shape[1]), dtype="float32")
    for i, d in enumerate(cfg["sig_oos"]):
        Z_oos[i] = fs_.rank_z(Pm[row_of[d]], cfg["uni_oos"][i]).astype("float32")
    np.save(os.path.join(cfg["zdir"], nm + ".npy"), np.nan_to_num(Z_oos, nan=0.0))

    return dict(factor=nm, sign=sign,
                is_stats=dict(累计收益=st["累计收益"], 年化收益=st["年化收益"],
                              年化波动=st["年化波动"], 夏普=st["夏普"],
                              最大回撤=st["最大回撤"], 卡玛=st["卡玛"], 日胜率=st["日胜率"]),
                is_monthly=[[p, v] for p, v in mret])


# ---------------------------------------------------------------- 主流程
def main():
    t0 = time.time()
    os.makedirs(ZDIR, exist_ok=True)
    md = MarketData(verbose=False)
    fd = zoo_api.load_factor_dict()
    factors = list(fd.index)
    n_f = len(factors)

    # ---------------- 交易日历 / 网格 ----------------
    sample = pd.read_parquet(zoo_api.panel_path(factors[0]))
    grid = pd.DatetimeIndex(sample.index)
    del sample
    dates = md.dates
    sig_oos = [d for d in grid if pd.Timestamp(mc.SIGNAL0) <= d <= pd.Timestamp("2026-08-31")]
    sig_is = [d for d in grid if d < pd.Timestamp(mc.SIGNAL0)]
    sig_oos_s = [str(d.date()) for d in sig_oos]
    sig_is_s = [str(d.date()) for d in sig_is]

    i_lo = int(dates.searchsorted(pd.Timestamp(mc.BT_START)))
    i_hi = int(dates.searchsorted(pd.Timestamp(mc.BT_END)))
    i_is_lo = int(dates.searchsorted(pd.Timestamp(IS_BASE))) + 1     # 2021-01-29 之后第一个交易日
    i_is_hi = int(dates.searchsorted(pd.Timestamp(mc.IS_END)))
    trade_dates = [str(d.date()) for d in dates[i_lo:i_hi + 1]]
    is_trade_dates = [str(d.date()) for d in dates[i_is_lo:i_is_hi + 1]]

    adj_open = (md.open * md.cumadj_clean).values[i_lo:i_hi + 1].astype("float32")
    adj_close = md.adj_close.values[i_lo:i_hi + 1].astype("float32")
    is_open = (md.open * md.cumadj_clean).values[i_is_lo:i_is_hi + 1].astype("float32")
    is_close = md.adj_close.values[i_is_lo:i_is_hi + 1].astype("float32")
    print("回测区间 %s ~ %s：%d 个交易日 / %d 个调仓信号"
          % (trade_dates[0], trade_dates[-1], len(trade_dates), len(sig_oos)))
    print("样本内     %s ~ %s：%d 个交易日 / %d 个调仓信号"
          % (is_trade_dates[0], is_trade_dates[-1], len(is_trade_dates), len(sig_is)))

    # ---------------- 股票池 ----------------
    uni = md.universe(**mc.UNI_KW)
    uni_arr = np.asarray(uni.values, dtype=bool)
    uni_oos = uni_arr[[int(dates.searchsorted(d)) for d in sig_oos]]
    uni_is = uni_arr[[int(dates.searchsorted(d)) for d in sig_is]]
    np.save(os.path.join(SEL, "universe.npy"), uni_oos)
    print("股票池规模：样本外 %d ~ %d 只；样本内 %d ~ %d 只"
          % (uni_oos.sum(1).min(), uni_oos.sum(1).max(), uni_is.sum(1).min(), uni_is.sum(1).max()))

    # ---------------- 并行：z 面板 + 样本内历史表现 ----------------
    is_prices_p = os.path.join(SEL, "_is_prices.npz")
    np.savez(is_prices_p, adj_open=is_open, adj_close=is_close)
    cfg = dict(is_sig=sig_is_s, sig_oos=sig_oos_s, is_trade_dates=is_trade_dates,
               uni_is=uni_is, uni_oos=uni_oos, zdir=ZDIR, is_prices=is_prices_p)
    jobs = [(nm, zoo_api.panel_path(nm), float(fd.loc[nm, "direction_is"])) for nm in factors]

    results = {}
    workers = int(os.environ.get("SELECTOR_WORKERS", max(1, min(8, (os.cpu_count() or 4) - 1))))
    print("\n计算 %d 个因子（横截面秩 z + 样本内历史表现，%d 进程）..." % (n_f, workers))
    if workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers, initializer=_init_worker, initargs=(cfg,)) as pool:
            for k, r in enumerate(pool.imap_unordered(_work, jobs, chunksize=2), 1):
                results[r["factor"]] = r
                if k % 30 == 0 or k == n_f:
                    print("  已算 %d/%d（%.0fs）" % (k, n_f, time.time() - t0))
    else:
        _init_worker(cfg)
        for k, j in enumerate(jobs, 1):
            r = _work(j)
            results[r["factor"]] = r
            if k % 30 == 0 or k == n_f:
                print("  已算 %d/%d（%.0fs）" % (k, n_f, time.time() - t0))
    try:
        os.remove(is_prices_p)
    except OSError:
        pass

    # ---------------- 因子字典 ----------------
    fmeta = {}
    for nm in factors:
        row = fd.loc[nm]
        r = results.get(nm, {})
        fmeta[nm] = dict(
            family=str(row["family"]), module=str(row["module"]), desc=str(row["desc"]),
            direction=float(r.get("sign", 1.0)),
            ic20=_f(row["ic20"]), icir=_f(row["icir"]), ls_net_ann=_f(row.get("ls_net_ann")),
            mono=_f(row.get("mono")), stability=_f(row.get("stability")),
            coverage=_f(row.get("coverage_is")), oos_ic20=_f(row.get("oos_ic20")),
            cluster=int(row["cluster"]), is_rep=bool(row["is_rep"]),
            is_stats={k: _f(v) for k, v in (r.get("is_stats") or {}).items()},
            is_monthly=[[p, _f(v)] for p, v in (r.get("is_monthly") or [])],
        )
    with open(os.path.join(SEL, "factor_meta.json"), "w", encoding="utf-8") as f:
        json.dump(fmeta, f, ensure_ascii=False, separators=(",", ":"))

    # ---------------- 基准曲线 ----------------
    bench = dict(dates=[str(d.date()) for d in dates[i_lo:i_hi + 1]],
                 base_date=str(pd.Timestamp(mc.SIGNAL0).date()))
    ew = md.ret.where(uni).mean(axis=1).reindex(dates[i_lo:i_hi + 1]).fillna(0.0).values
    bench["market_ew"] = dict(
        label="全市场等权（池内等权）",
        note="非 ST 全市场动态池，日频等权再平衡；与项目 02/03 报告的基准同口径",
        nav=_pack(np.concatenate([[1.0], np.cumprod(1.0 + ew)])))
    runs03 = os.path.join(ROOT, "out", "03_因子挖掘", "runs")
    for key, path, label, note in [
        ("mf1700", os.path.join(runs03, "23_raw", "MF1700_raw", "nav.csv"),
         "最佳线性打分 · 165 因子 ICIR 加权",
         "MF1700_raw：家族内→家族间两级 ICIR 加权，打分前 1700 只等权；引擎口径 22.51%/年、夏普 1.21"),
        ("ml1700", os.path.join(runs03, "26_ml", "ML1700", "nav.csv"),
         "最佳打分模型 · 树模型 ML1700",
         "梯度提升树打分，只用 2024-11-29 前的样本训练；引擎口径 29.38%/年、夏普 1.34"),
    ]:
        s = None
        if os.path.exists(path):
            s = pd.read_csv(path, index_col=0, parse_dates=True)["nav"]
            s = s.reindex(pd.DatetimeIndex([pd.Timestamp(mc.SIGNAL0)]).append(dates[i_lo:i_hi + 1]))
        if s is not None and np.isfinite(s.values).all():
            bench[key] = dict(label=label, note=note, nav=_pack(s.values / s.values[0]))
        else:
            print("  警告：基准 %s 读取失败，跳过" % key)
    with open(os.path.join(SEL, "benchmarks.json"), "w", encoding="utf-8") as f:
        json.dump(bench, f, ensure_ascii=False, separators=(",", ":"))

    # ---------------- 价格 + meta ----------------
    np.savez_compressed(os.path.join(SEL, "prices_oos.npz"),
                        adj_open=adj_open, adj_close=adj_close)
    meta = dict(
        built_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        factors=factors,
        symbols=[str(s) for s in md.symbols],
        trade_dates=trade_dates,
        base_date=str(pd.Timestamp(mc.SIGNAL0).date()),
        signal_dates=sig_oos_s,
        liq_signal=str(pd.Timestamp(mc.LIQ_SIGNAL).date()),
        is_signals=sig_is_s,
        is_n_hold=IS_N_HOLD,
        cash=fs.CASH0, commission=fs.COMMISSION, stamp=fs.STAMP, gross_limit=fs.GROSS_LIMIT,
        universe_size=[int(x) for x in uni_oos.sum(axis=1)],
        families={k: dict(desc=v, n=int((fd["family"] == k).sum()),
                          rep=int(((fd["family"] == k) & fd["is_rep"].astype(bool)).sum()))
                  for k, v in FAMILY_DESC.items()},
    )
    with open(os.path.join(SEL, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print("\n产物目录：%s" % SEL)
    print("  z 面板 %d 个 / 因子字典 %d 条 / 信号 %d 个 / 交易日 %d"
          % (len(os.listdir(ZDIR)), len(fmeta), len(sig_oos), len(trade_dates)))
    print("用时 %.0fs" % (time.time() - t0))


def _f(v):
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(x) else x


def _pack(a, nd=6):
    return [round(float(x), nd) for x in np.asarray(a, dtype="float64")]


if __name__ == "__main__":
    main()
