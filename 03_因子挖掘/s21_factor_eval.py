# -*- coding: utf-8 -*-
"""s21 因子评价主线：铺面板 -> 覆盖率/因果性 -> IC 家族 -> 分层多空 -> 相关性去冗余。

口径
----
- 网格：2021-01 ~ 2026-09 每月最后一个交易日（信号日），次日开盘成交。
- 股票池：非 ST 全市场（动态池，`MarketData.universe` 默认口径）。
- 前视收益：`ADJ_O[t+1+h]/ADJ_O[t+1]-1`（可成交口径，不是收盘价对收盘价）。
- 样本内（定方向/筛选）只到 2024-12-31；2025-01 之后完全不参与选择，只做样本外检验。

产物
----
out/03_因子挖掘/tables/21_*.csv    因子统计 / IC 衰减 / 分层 / 聚类 / 暴露
out/cache/zoo2/*.parquet           因子面板缓存（供下游打分程序复用）
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.environ.setdefault("CLASS4_TRACK", "03_因子挖掘")

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from core.lib import MarketData, CACHE, TAB, RUNS, savetab
from core.engine import Backtest
from factor_zoo import registry
from factor_zoo.ctx import Ctx
from factor_zoo.panels import build_panels_parallel
import mining_core as mc

PANEL_DIR = os.path.join(CACHE, "zoo2")


def _zpanel(P):
    return np.vstack([mc.cs_rank_z(P[i]) for i in range(P.shape[0])])


def _aux_exposures(md, C, ts):
    """规模/波动/前期收益三个暴露代理，用于识别因子在赌什么 beta。"""
    adv = np.log(np.maximum(md.adv(20).values.astype("float64"), 1.0))[list(ts)]
    R = C.R
    vol = np.full_like(adv, np.nan)
    ret20 = np.full_like(adv, np.nan)
    for i, t in enumerate(ts):
        lo = max(int(t) - 59, 0)
        W = R[lo:int(t) + 1]
        ok = np.isfinite(W)
        vol[i] = C.std(W, ok, 20) * np.sqrt(252.0)
        if int(t) >= 20:
            a, b = C.C[int(t) - 20], C.C[int(t)]
            with np.errstate(invalid="ignore", divide="ignore"):
                ret20[i] = np.where((a > 0) & np.isfinite(a) & np.isfinite(b), b / a - 1.0, np.nan)
    return dict(log_adv=adv, vol60=vol, ret20=ret20)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="只算前 N 个因子（调试）")
    ap.add_argument("--skip-corr", action="store_true")
    ap.add_argument("--force", action="store_true", help="忽略面板缓存重算")
    a = ap.parse_args(argv)

    t_all = time.time()
    md = MarketData(verbose=True)
    C = Ctx(md)
    ts = mc.monthly_grid(md, mc.GRID_START, mc.BT_END)
    gdates = md.dates[ts]
    print("调仓网格 %d 个：%s ~ %s" % (len(ts), gdates[0].date(), gdates[-1].date()))

    facts = registry.load_all(strict=False)
    if a.limit:
        facts = dict(list(facts.items())[: a.limit])
    print("注册因子 %d 个" % len(facts))
    fam = pd.Series({k: v["family"] for k, v in facts.items()}, name="family")
    print(fam.value_counts().to_string())

    # ---------------- 1) 面板 ----------------
    print("\n[1/5] 铺因子面板（workers=%d）..." % a.workers)
    t0 = time.time()
    panels = build_panels_parallel(md, facts, ts, cache_dir=PANEL_DIR,
                                   workers=a.workers, verbose=True, force=a.force)
    print("  面板完成 %.1fs，共 %d 个" % (time.time() - t0, len(panels)))

    uni = mc.universe_mask(md, ts)
    fwd = mc.fwd_returns(md, ts, mc.HORIZONS)
    is_rows = np.array([md.dates[t] <= pd.Timestamp(mc.IS_END) for t in ts])
    oos_rows = ~is_rows
    print("  样本内 %d 期 / 样本外 %d 期；池内平均 %.0f 只"
          % (is_rows.sum(), oos_rows.sum(), uni.sum(1).mean()))

    # ---------------- 2) 逐因子统计 ----------------
    print("\n[2/5] IC / 衰减 / 分层 / 稳定性 ...")
    aux = _aux_exposures(md, C, ts)
    Z, rows, decay_rows, decile_rows, expo_rows = {}, [], [], [], []
    for n, (nm, spec) in enumerate(facts.items(), 1):
        P = np.asarray(panels[nm].values, dtype="float64")
        Z[nm] = _zpanel(P)
        cov = (np.isfinite(P) & uni).sum(1) / np.maximum(uni.sum(1), 1)
        cov_is = float(np.nanmean(cov[is_rows])) if is_rows.any() else np.nan
        ic = {h: mc.ic_series(P, fwd[h], uni) for h in mc.HORIZONS}
        s20 = mc.ic_stats(ic[20][is_rows])
        s20o = mc.ic_stats(ic[20][oos_rows])
        for h in mc.HORIZONS:
            st = mc.ic_stats(ic[h][is_rows])
            decay_rows.append(dict(factor=nm, horizon=h, ic=st["ic"], icir=st["icir"],
                                   t=st["t"], pos=st["pos"], n=st["n"]))
        ls = mc.decile_ls(P, fwd[20], uni, q=10)
        dm = mc.decile_matrix(P, fwd[20], uni, q=10)
        mono = mc.monotonicity(dm)
        # 因子自身稳定性：相邻期秩相关的均值（样本内）
        stab = []
        for i in range(1, len(ts)):
            if not is_rows[i]:
                continue
            stab.append(mc.spearman(Z[nm][i], Z[nm][i - 1], uni[i] & uni[i - 1], min_n=50))
        stab = float(np.nanmean(stab)) if len(stab) else np.nan
        ex = {k: float(np.nanmean([mc.spearman(P[i], v[i], uni[i], min_n=50)
                                   for i in range(len(ts)) if is_rows[i]]))
              for k, v in aux.items()}
        expo_rows.append(dict(factor=nm, **{"corr_" + k: v for k, v in ex.items()}))
        rows.append(dict(
            factor=nm, family=spec["family"], module=spec["module"],
            desc=spec.get("desc", ""), direction_prior=int(spec.get("direction", -1)),
            coverage_is=cov_is,
            ic1=s20["ic"] * 0 + mc.ic_stats(ic[1][is_rows])["ic"],
            ic5=mc.ic_stats(ic[5][is_rows])["ic"],
            ic10=mc.ic_stats(ic[10][is_rows])["ic"],
            ic20=s20["ic"], ic_std=s20["ic_std"], icir=s20["icir"], t=s20["t"],
            pos=s20["pos"], n_is=s20["n"],
            oos_ic20=s20o["ic"], oos_icir=s20o["icir"], oos_t=s20o["t"], n_oos=s20o["n"],
            ls_gross_ann=ls["gross_ann"], ls_net_ann=ls["net_ann"],
            ls_gross_sharpe=ls["gross_sharpe"], ls_net_sharpe=ls["net_sharpe"],
            ls_turnover_ann=ls["turnover_ann"], ls_cost_ann=ls["cost_ann"],
            mono=mono, stability=stab,
        ))
        decile_rows.append(dict(factor=nm, **{"D%d" % (i + 1): float(np.nanmean(dm[i])) for i in range(10)},
                                spread=float(np.nanmean(dm[9] - dm[0])), mono=mono))
        if n % 20 == 0 or n == len(facts):
            print("    %d/%d  %s" % (n, len(facts), nm))

    stats = pd.DataFrame(rows)
    stats["direction_is"] = np.sign(stats["ic20"]).replace(0, np.nan)
    stats["ic20_adj"] = stats["ic20"] * stats["direction_is"]
    stats["icir_adj"] = stats["icir"] * stats["direction_is"]
    stats["valid"] = (stats["coverage_is"] >= 0.5) & stats["ic20"].notna()
    savetab(stats, "21_factor_stats.csv", index=False)
    savetab(pd.DataFrame(decay_rows), "21_ic_decay.csv", index=False)
    savetab(pd.DataFrame(decile_rows), "21_decile.csv", index=False)
    savetab(pd.DataFrame(expo_rows), "21_exposure.csv", index=False)

    fam_tab = (stats.groupby("family")
               .agg(n=("factor", "size"), ic20_med=("ic20", "median"),
                    icir_med=("icir", "median"),
                    n_abs_ic_gt_2pct=("ic20", lambda s: int((s.abs() > 0.02).sum())),
                    net_ann_med=("ls_net_ann", "median"))
               .sort_values("icir_med", key=lambda s: -s.abs()))
    savetab(fam_tab, "21_family.csv")

    # ---------------- 3) 相关性 + 聚类 ----------------
    if not a.skip_corr:
        print("\n[3/5] 平均横截面相关矩阵 + 冗余聚类 ...")
        t0 = time.time()
        corr = mc.avg_corr_matrix(Z, uni, sample_n=900, seed=7)
        savetab(corr, "21_factor_corr.csv")
        cl = mc.cluster_by_corr(corr, thresh=0.85)
        stats["cluster"] = stats["factor"].map(cl).fillna(-1).astype(int)
        # 每簇代表：|ICIR| 最大者
        rep = {}
        for c, g in stats.groupby("cluster"):
            rep[c] = g.loc[g["icir_adj"].abs().idxmax(), "factor"]
        stats["cluster_rep"] = stats["cluster"].map(rep)
        stats["is_rep"] = stats["factor"] == stats["cluster_rep"]
        savetab(stats.sort_values(["cluster", "icir_adj"], key=lambda s: s.abs() if s.name == "icir_adj" else s),
                "21_factor_stats.csv", index=False)
        print("  相关矩阵 %.1fs，簇数 %d（冗余因子 %d 个）"
              % (time.time() - t0, stats["cluster"].nunique(),
                 int((~stats["is_rep"]).sum())))
    else:
        stats["cluster"] = -1
        stats["is_rep"] = True

    # ---------------- 4) 元数据 ----------------
    meta = dict(grid_dates=[str(d.date()) for d in gdates],
                n_factors=int(len(stats)), n_is=int(is_rows.sum()), n_oos=int(oos_rows.sum()),
                uni_kw={k: (None if v is None else v) for k, v in mc.UNI_KW.items()},
                cost=dict(commission=mc.COMMISSION, stamp=mc.STAMP),
                horizons=list(mc.HORIZONS), is_end=mc.IS_END)
    with open(os.path.join(CACHE, "zoo2", "panel_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    # ---------------- 5) 小结 ----------------
    ok = stats[stats["valid"]]
    print("\n[4/5] 小结：有效因子 %d/%d" % (len(ok), len(stats)))
    print("  |IC20| >= 0.02 : %d" % int((ok["ic20"].abs() >= 0.02).sum()))
    print("  |ICIR| >= 0.5  : %d" % int((ok["icir"].abs() >= 0.5).sum()))
    print("  净多空年化 > 0  : %d" % int((ok["ls_net_ann"] > 0).sum()))
    print("  样本外同向(|IC|>0.02): %d"
          % int(((ok["oos_ic20"].abs() >= 0.02) & (np.sign(ok["oos_ic20"]) == ok["direction_is"])).sum()))
    print("\n[5/5] 完成，用时 %.1fs" % (time.time() - t_all))
    print("  表：%s" % TAB)
    print("  面板缓存：%s" % PANEL_DIR)
    return stats


if __name__ == "__main__":
    main()
