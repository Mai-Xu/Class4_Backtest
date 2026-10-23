# -*- coding: utf-8 -*-
"""v2 复核 · 附加验证6：C.mkt 的时点一致性（数据层 cap_tolerance 是否偷看未来）。

core.lib 的 cap_tolerance() 用**全样本**中位价（close.where(listed).median(0)）构造
涨跌停容差，进而影响 no_limit -> ret_ok -> C.mkt（等权市场收益）。
复核：用「只用 <= t 的扩展中位数」重建容差与 mkt，比较两者差异，
并检查它对一个 mkt 重度因子（mk_rel_ret20）样本内 IC20 的影响。
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from v2_lib import load_ref, monthly_grid, universe_mask, fwd_open_ret, spearman  # noqa: E402
from v2_lib import ref_mk_rel_ret20                                            # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
TAB = os.path.join(ROOT, "out", "03_因子挖掘", "tables")
CACHE = os.path.join(ROOT, "out", "cache")
HALF_TICK = 0.005


def main():
    t_all = time.time()
    r, md = load_ref()
    LR = np.asarray(md.limit_ratio.values, dtype="float64")
    NS = np.asarray(md._new_stock.values, dtype=bool)
    close = np.asarray(md.close.values, dtype="float64")
    listed = r.LISTED
    traded = listed & np.isfinite(r.A) & (r.A > 0.0)
    # 全样本口径（core.lib 现状）
    tol_full = np.asarray(md.cap_tolerance().values, dtype="float64")[None, :]
    # 时点口径：<= t 的扩展中位数
    med_pit = pd.DataFrame(np.where(listed, close, np.nan)).expanding(min_periods=1).median().values
    tol_pit = 0.002 + HALF_TICK / med_pit

    over_full = np.abs(r.R) > (LR + tol_full)
    over_pit = np.abs(r.R) > (LR + tol_pit)
    nl_full = listed & (NS | over_full)
    nl_pit = listed & (NS | over_pit)
    n_over_diff = int((over_full != over_pit).sum())
    n_nl_diff = int((nl_full != nl_pit).sum())

    keep = traded & np.roll(traded, 1, axis=0)
    keep[0] = False
    ret_ok_full = np.where(keep & ~nl_full, r.R, np.nan)
    ret_ok_pit = np.where(keep & ~nl_pit, r.R, np.nan)
    mkt_full = np.nanmean(np.where(np.isfinite(ret_ok_full), ret_ok_full, np.nan), axis=1)
    mkt_pit = np.nanmean(np.where(np.isfinite(ret_ok_pit), ret_ok_pit, np.nan), axis=1)
    d = np.abs(mkt_full - mkt_pit)
    print("no_limit 判定位 (行,列) 差异：%d 格；over 掩码差异 %d 格" % (n_nl_diff, n_over_diff))
    print("mkt 差异：max=%.3e  均值=%.3e  非零日数=%d/%d"
          % (float(np.nanmax(d)), float(np.nanmean(d)), int((d > 0).sum()), d.size))
    ref_pipe = np.asarray(md.ret_ok.mean(1).values, dtype="float64")
    print("复核重算的 mkt(全样本容差) vs Ctx.mkt：max=%.3e"
          % float(np.nanmax(np.abs(ref_pipe - mkt_full))))

    # 对 mk_rel_ret20 的影响
    grid = monthly_grid(r.dates, "2021-01-01", "2026-09-30")
    uni = universe_mask(md, grid)
    fwd20 = fwd_open_ret(r, grid, 20)
    is_rows = np.array([r.dates[t] <= pd.Timestamp("2024-12-31") for t in grid])
    man = pd.read_csv(os.path.join(TAB, "panel_manifest.csv"))
    man = {row["factor"]: os.path.join(CACHE, row["file"]) for _, row in man.iterrows()}
    sock = r.MKT.copy()
    out = {}
    for tag, mkt in (("pipeline", sock), ("pit", mkt_pit)):
        r.MKT = mkt
        P = np.vstack([ref_mk_rel_ret20(r, int(t), np.arange(r.k)) for t in grid])
        ic = np.array([spearman(P[i], fwd20[i], uni[i], min_n=50) for i in range(len(grid))])
        out[tag] = float(np.nanmean(ic[is_rows]))
    r.MKT = sock
    print("mk_rel_ret20 样本内 IC20：管线 mkt=%.6f  时点 mkt=%.6f  Δ=%.3e"
          % (out["pipeline"], out["pit"], out["pit"] - out["pipeline"]))
    with open(os.path.join(HERE, "v2_t6_mkt_pit.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(n_no_limit_diff=n_nl_diff, n_over_diff=n_over_diff,
                       mkt_max_abs_diff=float(np.nanmax(d)),
                       mkt_mean_abs_diff=float(np.nanmean(d)),
                       mkt_days_diff=int((d > 0).sum()), n_days=int(d.size),
                       ic20_pipeline=out["pipeline"], ic20_pit=out["pit"],
                       ic20_diff=out["pit"] - out["pipeline"],
                       seconds=time.time() - t_all), fh, ensure_ascii=False, indent=1)
    print("总耗时 %.1fs → v2_t6_mkt_pit.json" % (time.time() - t_all))


if __name__ == "__main__":
    main()