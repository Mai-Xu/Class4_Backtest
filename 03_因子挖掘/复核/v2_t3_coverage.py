# -*- coding: utf-8 -*-
"""v2 复核 · 验证3：覆盖率与常数检查（2025-2026 月频调仓日）。

- 已注册的 9 个模块：直接用 out/cache/zoo2 面板（= 主线真实消费的模块输出），
  ① 复现 s20 的 12 个抽样日覆盖率，与 out/03_因子挖掘/tables/20_zoo_check.csv 比对；
  ② 在 2025-01~2026-09 的 21 个月频调仓日上检查横截面标准差 > 0（全市场 + 池内）。
- flow2（36 个 fl2_* 因子）：不在 registry.CATEGORIES 内，没有面板/表行，
  改为直接调用模块函数重算同样三个指标（并明确标注"主线未注册"）。
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

from v2_lib import load_ref, monthly_grid, universe_mask     # noqa: E402
from v2_modules import load_modules                          # noqa: E402
from factor_zoo.ctx import Ctx                               # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
TAB = os.path.join(ROOT, "out", "03_因子挖掘", "tables")
CACHE = os.path.join(ROOT, "out", "cache")

COV = {
    "reversal": ["rev_ret1", "rev_ret20g", "rev_amtmom20", "rev_intra20", "rev_high5_60"],
    "risk2": ["rk2_beta250", "rk2_ivol120", "rk2_semidn120", "rk2_tailratio60", "rk2_jump60"],
    "liquidity2": ["lq2_logamt250", "lq2_amt_cv60", "lq2_amivest60", "lq2_tradesize60",
                   "lq2_still_days250"],
    "flow2": ["fl2_amtmom10", "fl2_sign_amt_ratio20", "fl2_force13", "fl2_pvt_slope60",
              "fl2_emv20"],
    "chip": ["ch_profit120", "ch_conc120", "ch_skew60", "ch_support_dist60", "ch_cost_dev250d"],
    "trend2": ["tr2_madev5", "tr2_aroon_up25", "tr2_stoch_k14", "tr2_trix12", "tr2_donch_pos250"],
    "pattern": ["pt_upper_shadow1", "pt_close_vs_open20", "pt_nr7_count20",
                "pt_eod_reversal20", "pt_path_entropy20"],
    "calendar": ["cal_tom_x_amt", "cal_quarter_x_rev20", "cal_pre_holiday_x_ret5",
                 "cal_month_x_mom60", "cal_post_holiday_x_mom20"],
    "interaction": ["ix_rev_x_lowvol", "ix_max_x_illiq", "ix_rev_x_size",
                    "ix_nearhigh_x_amt", "ix_mom_rank_x_vol_rank"],
    "marketrel": ["mk_rel_ret5", "mk_resid_vol_ratio60", "mk_rps60", "mk_info_ratio60",
                  "mk_down_vol_share"],
}


def main():
    t_all = time.time()
    r, md = load_ref()
    facts = load_modules()
    grid = monthly_grid(r.dates, "2021-01-01", "2026-09-30")
    print("月频网格 %d 个日期（%s ~ %s）" % (len(grid), r.dates[grid[0]].date(), r.dates[grid[-1]].date()))
    step = max(1, len(grid) // 12)
    ts12 = grid[::step][:12]
    print("复现 s20 的 12 个抽样日（grid[::%d][:12]）：%s"
          % (step, ", ".join(str(r.dates[t].date()) for t in ts12)))
    ts25 = [t for t in grid if r.dates[t] >= pd.Timestamp("2025-01-01")]
    print("2025-2026 月频调仓日 %d 个：%s ~ %s"
          % (len(ts25), r.dates[ts25[0]].date(), r.dates[ts25[-1]].date()))

    man = pd.read_csv(os.path.join(TAB, "panel_manifest.csv"))
    man = {row["factor"]: os.path.join(CACHE, row["file"]) for _, row in man.iterrows()}
    stats20 = pd.read_csv(os.path.join(TAB, "20_zoo_check.csv")).set_index("factor")
    uni25 = universe_mask(md, ts25)
    C = Ctx(md)

    rows = []
    for mod, names in COV.items():
        for nm in names:
            if nm not in facts[mod]:
                raise KeyError("模块 %s 缺因子 %s" % (mod, nm))
            need = pd.DatetimeIndex([r.dates[t] for t in ts12] + [r.dates[t] for t in ts25])
            if nm in man:                       # 已注册模块：用缓存面板（主线真实产物）
                df = pd.read_parquet(man[nm])
                sub = df.reindex(need)
                ts12_pos = list(range(len(ts12)))
                ts25_pos = list(range(len(ts12), len(ts12) + len(ts25)))
                V = sub.values.astype("float64")
                src = "panel"
            else:                               # flow2：直接调用模块函数
                f = facts[mod][nm]["func"]
                V = np.vstack([np.asarray(f(int(t), md, C), dtype="float64")
                               for t in list(ts12) + list(ts25)])
                ts12_pos = list(range(len(ts12)))
                ts25_pos = list(range(len(ts12), len(ts12) + len(ts25)))
                src = "direct"
            V12 = V[ts12_pos]
            V25 = V[ts25_pos]
            cov12 = float(np.isfinite(V12).mean())
            csv_cov = float(stats20.loc[nm, "coverage"]) if nm in stats20.index else float("nan")
            fin25 = np.isfinite(V25)
            cov25 = float(fin25.mean())
            sd_all = np.nanstd(V25, axis=1)
            sd_uni = np.nanstd(np.where(uni25, V25, np.nan), axis=1)
            cov25_uni = float((fin25 & uni25).sum() / max(int(uni25.sum()), 1))
            rec = dict(module=mod, factor=nm, source=src,
                       cov12_mine=cov12, cov12_csv=csv_cov, cov12_diff=cov12 - csv_cov,
                       cov2526_all=cov25, cov2526_uni=cov25_uni,
                       n_days=len(ts25),
                       n_sd_zero_all=int((~np.isfinite(sd_all) | (sd_all <= 0)).sum()),
                       n_sd_zero_uni=int((~np.isfinite(sd_uni) | (sd_uni <= 0)).sum()),
                       min_sd_all=float(np.nanmin(sd_all)), min_sd_uni=float(np.nanmin(sd_uni)))
            rows.append(rec)
            print("%-12s %-26s %-6s | cov12=%.4f vs csv %.4f (Δ%+.2e) | cov25全市场=%.4f 池内=%.4f"
                  " | sd≤0 日数 全%2d/池内%2d | min_sd=%.3g"
                  % (mod, nm, src, cov12, csv_cov, rec["cov12_diff"], cov25, cov25_uni,
                     rec["n_sd_zero_all"], rec["n_sd_zero_uni"], rec["min_sd_all"]))

    reg = [x for x in rows if x["source"] == "panel"]
    d = np.array([abs(x["cov12_diff"]) for x in reg])
    print("\n已注册 45 个因子 cov12 与 20_zoo_check.csv 的 |差|：max=%.3e 中位=%.3e（float32 面板舍入）"
          % (d.max(), float(np.median(d))))
    n_sd0 = sum(x["n_sd_zero_all"] for x in reg)
    print("2025-2026 × %d 日 × 45 因子：全市场标准差 ≤ 0 或 NaN 的 (因子,日期) 格数 = %d" % (len(ts25), n_sd0))
    fl2 = [x for x in rows if x["source"] == "direct"]
    print("flow2 5 个因子（直接调用，主线未注册）：cov25 全市场范围 %.4f ~ %.4f，sd≤0 格数 = %d"
          % (min(x["cov2526_all"] for x in fl2), max(x["cov2526_all"] for x in fl2),
             sum(x["n_sd_zero_all"] for x in fl2)))

    with open(os.path.join(HERE, "v2_t3_coverage.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(dates12=[str(r.dates[t].date()) for t in ts12],
                       dates2526=[str(r.dates[t].date()) for t in ts25],
                       rows=rows, seconds=time.time() - t_all), fh,
                  ensure_ascii=False, indent=1, default=str)
    print("总耗时 %.1fs → v2_t3_coverage.json" % (time.time() - t_all))


if __name__ == "__main__":
    main()