# -*- coding: utf-8 -*-
"""v2 复核 · 验证2：数值正确性抽检（每模块 2 个因子，独立公式重算）。

参考公式在 v2_lib.py（复核方独立实现）；被测值 = 直接调用模块公开的
FACTORS[name]["func"](t, md, C)（黑盒，仅取输出）。日期固定在 2023-07-31
（月频网格第 31 期，周一，日号 31——日历类因子状态均非平凡）。
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from v2_lib import load_ref, monthly_grid, REF_FACTORS          # noqa: E402
from v2_modules import load_modules                             # noqa: E402
from factor_zoo.ctx import Ctx                                  # noqa: E402

#: (模块, 因子, 比较列数：all=全横截面；整数=该数量的上市个股随机子样本)
NUM = [
    ("reversal", "rev_ret10", "all"),
    ("reversal", "rev_max1d20", "all"),
    ("risk2", "rk2_park60", "all"),
    ("risk2", "rk2_var5_60", "all"),
    ("liquidity2", "lq2_logamt60", "all"),
    ("liquidity2", "lq2_amihud120", "all"),
    ("flow2", "fl2_cmf60", "all"),
    ("flow2", "fl2_vwapdev20", "all"),
    ("chip", "ch_profit60", "all"),
    ("chip", "ch_cost_dev120", "all"),
    ("trend2", "tr2_rsi14b", 800),
    ("trend2", "tr2_di_plus14", 800),
    ("pattern", "pt_body1", "all"),
    ("pattern", "pt_up_days20", "all"),
    ("calendar", "cal_tom_x_rev5", "all"),
    ("calendar", "cal_seasonal_mom", "all"),
    ("interaction", "ix_consensus_rev", "all"),
    ("interaction", "ix_rev_x_illiq", "all"),
    ("marketrel", "mk_rel_ret20", "all"),
    ("marketrel", "mk_rel_mom12_1", "all"),
]


def pick_cols(n_listed, spec, seed=7):
    idx = np.flatnonzero(n_listed)
    if spec == "all":
        return idx
    rng = np.random.default_rng(seed)
    n = min(int(spec), idx.size)
    return np.sort(rng.choice(idx, size=n, replace=False))


def main():
    t_all = time.time()
    r, md = load_ref()
    facts = load_modules()
    grid = monthly_grid(r.dates, "2021-01-01", "2026-09-30")
    t = grid[30]
    print("数值抽检日期 = %s（t=%d）" % (r.dates[t].date(), t))
    C = Ctx(md)

    rows = []
    for mod, nm, spec in NUM:
        if nm not in REF_FACTORS:
            raise KeyError("参考实现缺失：%s" % nm)
        if nm not in facts[mod]:
            raise KeyError("模块 %s 无因子 %s" % (mod, nm))
        ref_fn, ref_desc = REF_FACTORS[nm]
        cols = pick_cols(r.LISTED[t], spec)
        got = np.asarray(facts[mod][nm]["func"](int(t), md, C), dtype="float64")[cols]
        exp = np.asarray(ref_fn(r, t, cols), dtype="float64")
        fg, fe = np.isfinite(got), np.isfinite(exp)
        mask_mis = int((fg != fe).sum())
        both = fg & fe
        d = np.abs(got[both] - exp[both]) if both.any() else np.array([0.0])
        big = np.abs(exp[both]) > 1e-8 if both.any() else np.array([], dtype=bool)
        rel = (d / np.maximum(np.abs(exp[both]), 1e-300))[big] if both.any() and big.any() else np.array([0.0])
        rec = dict(module=mod, factor=nm, ref=ref_desc, n_cols=int(cols.size),
                   n_both_finite=int(both.sum()), n_both_nan=int((~fg & ~fe).sum()),
                   n_nan_mask_mismatch=mask_mis,
                   max_abs=float(d.max()), mean_abs=float(d.mean()),
                   max_rel=float(rel.max()), mean_rel=float(rel.mean()),
                   verdict="OK" if (mask_mis == 0 and (float(rel.max()) < 1e-9 or float(d.max()) < 1e-9)) else "CHECK")
        rows.append(rec)
        print("%-12s %-24s 比较%5d 列 | 双有限%5d 掩码不一致%3d | max|Δ|=%.3e max_rel=%.3e | %s"
              % (mod, nm, cols.size, rec["n_both_finite"], mask_mis, rec["max_abs"],
                 rec["max_rel"], rec["verdict"]))

    n_ok = sum(1 for x in rows if x["verdict"] == "OK")
    print("\n数值一致（判定 OK）：%d/%d" % (n_ok, len(rows)))
    with open(os.path.join(HERE, "v2_t2_numeric.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(date=str(r.dates[t].date()), t=int(t), rows=rows,
                       seconds=time.time() - t_all), fh, ensure_ascii=False, indent=1)
    print("总耗时 %.1fs → v2_t2_numeric.json" % (time.time() - t_all))


if __name__ == "__main__":
    main()