# -*- coding: utf-8 -*-
"""v2 复核 · 验证4：IC 复核（独立 Spearman + 可成交前视收益）。

抽 5 个因子（lq_amt_log20 / rev_max1d20 / ch_profit60 / tr2_di_plus14 / ix_consensus_rev），
用复核方自己实现的 Spearman 秩相关与 ADJ_O[t+21]/ADJ_O[t+1]-1 前视收益，
在 2021-01~2024-12（样本内）重算平均 IC20，与
out/03_因子挖掘/tables/21_factor_stats.csv 的 ic20 比对。
因子取值直接用 out/cache/zoo2 面板（主线真实消费的模块输出，float32）。
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
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from v2_lib import load_ref, monthly_grid, universe_mask, fwd_open_ret, spearman  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
TAB = os.path.join(ROOT, "out", "03_因子挖掘", "tables")
CACHE = os.path.join(ROOT, "out", "cache")

FACTORS = ["lq_amt_log20", "rev_max1d20", "ch_profit60", "tr2_di_plus14", "ix_consensus_rev"]


def main():
    t_all = time.time()
    r, md = load_ref()
    grid = monthly_grid(r.dates, "2021-01-01", "2026-09-30")
    print("月频网格 %d 个日期" % len(grid))
    uni = universe_mask(md, grid)
    fwd20 = fwd_open_ret(r, grid, 20)
    is_rows = np.array([r.dates[t] <= pd.Timestamp("2024-12-31") for t in grid])
    print("样本内 %d 期 / 样本外 %d 期" % (is_rows.sum(), (~is_rows).sum()))

    man = pd.read_csv(os.path.join(TAB, "panel_manifest.csv"))
    man = {row["factor"]: os.path.join(CACHE, row["file"]) for _, row in man.iterrows()}
    stats = pd.read_csv(os.path.join(TAB, "21_factor_stats.csv")).set_index("factor")

    rows = []
    for nm in FACTORS:
        if nm not in man:
            raise KeyError("面板缺失：%s" % nm)
        df = pd.read_parquet(man[nm])
        if not np.array_equal(np.asarray(df.index.values), np.asarray(r.dates[grid].values)):
            raise RuntimeError("面板日期与复核网格不一致：%s" % nm)
        P = df.values.astype("float64")
        ic = np.array([spearman(P[i], fwd20[i], uni[i], min_n=50) for i in range(len(grid))])
        ic_is = ic[is_rows]
        ic20 = float(np.nanmean(ic_is))
        n_is = int(np.isfinite(ic_is).sum())
        tab_ic = float(stats.loc[nm, "ic20"])
        tab_n = int(stats.loc[nm, "n_is"])
        rec = dict(factor=nm, module=str(stats.loc[nm, "module"]),
                   ic20_mine=ic20, ic20_table=tab_ic, diff=ic20 - tab_ic,
                   n_is_mine=n_is, n_is_table=tab_n,
                   ic_std_mine=float(np.nanstd(ic_is, ddof=1)),
                   ic_std_table=float(stats.loc[nm, "ic_std"]),
                   icir_mine=float(ic20 / np.nanstd(ic_is, ddof=1) * np.sqrt(12.0)),
                   icir_table=float(stats.loc[nm, "icir"]),
                   pos_mine=float((ic_is[np.isfinite(ic_is)] > 0).mean()),
                   oos_ic20_mine=float(np.nanmean(ic[~is_rows])),
                   oos_ic20_table=float(stats.loc[nm, "oos_ic20"]))
        rows.append(rec)
        print("%-20s ic20 复核=%+.6f 表格=%+.6f Δ=%+.3e | ICIR 复核%.3f 表格%.3f | 期数 %d/%d"
              % (nm, rec["ic20_mine"], rec["ic20_table"], rec["diff"],
                 rec["icir_mine"], rec["icir_table"], n_is, tab_n))

    worst = max(abs(x["diff"]) for x in rows)
    print("\n5 个因子 ic20 最大绝对差 = %.3e" % worst)
    with open(os.path.join(HERE, "v2_t4_ic.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(factors=FACTORS, rows=rows, seconds=time.time() - t_all), fh,
                  ensure_ascii=False, indent=1, default=str)
    print("总耗时 %.1fs → v2_t4_ic.json" % (time.time() - t_all))


if __name__ == "__main__":
    main()