# -*- coding: utf-8 -*-
"""v2 复核 · 验证1：因果性投毒测试（独立实现，不使用 Ctx.truncated）。

对 10 个模块各抽 5 个因子 × 3 个日期，构造「t+1 行及以后全部改成 NaN / 随机值」
的 md 副本 + 全新 Ctx，比较 f(t) 与干净上下文是否逐位相同。
三种投毒模式：
  copy —— 数组照抄（b<=t 逐位相同、>t 也相同），用于测「浮点噪声下限」；
  nan  —— >t 行全部 NaN；
  rand —— >t 行用该列 <=t 均值/标准差生成的随机值。
另加两个「故意偷看未来」的控制因子（t+1 收盘价、全样本均值），证明本测试有检出力。
"""
from __future__ import annotations

import gc
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

from v2_lib import load_ref, monthly_grid                      # noqa: E402
from v2_modules import load_modules, poison_md, compare        # noqa: E402
from factor_zoo.ctx import Ctx                                 # noqa: E402  共享上下文（非被测模块）

POISON = {
    "reversal": ["rev_ret10", "rev_max1d20", "rev_ret5_rank60", "rev_upstreak", "rev_ovn_last"],
    "risk2": ["rk2_beta60", "rk2_park60", "rk2_var5_60", "rk2_ulcer60", "rk2_vol_ewma60"],
    "liquidity2": ["lq2_logamt60", "lq2_amihud120", "lq2_spread_roll60",
                   "lq2_still_days60", "lq2_amt_percentile250"],
    "flow2": ["fl2_pvcorr10", "fl2_cmf60", "fl2_vwapdev20", "fl2_obv_z250", "fl2_mfi60"],
    "chip": ["ch_profit60", "ch_cost_dev120", "ch_conc60", "ch_decay_profit120", "ch_above_amt60"],
    "trend2": ["tr2_di_plus14", "tr2_rsi14b", "tr2_macd_dif", "tr2_linslope20", "tr2_hurst60"],
    "pattern": ["pt_body1", "pt_shape20", "pt_gap_up20", "pt_run_up", "pt_doji5"],
    "calendar": ["cal_tom_x_rev5", "cal_seasonal_mom", "cal_weekday_x_rev5",
                 "cal_dom_x_rev5", "cal_seasonal_mom_max"],
    "interaction": ["ix_consensus_rev", "ix_rev_x_illiq", "ix_mom_x_lowvol",
                    "ix_rev_sq", "ix_consensus_mom"],
    "marketrel": ["mk_rel_ret20", "mk_rel_mom12_1", "mk_alpha60",
                  "mk_up_capture60", "mk_rel_streak"],
}


def _ctrl_peek_next(t, md, C):
    j = int(t) + 1
    if j >= C.n:
        return np.full(C.k, np.nan)
    return np.asarray(C.C[j], dtype="float64")


def _ctrl_full_mean(t, md, C):
    ok = np.isfinite(C.R)
    cnt = ok.sum(0)
    s = np.where(ok, C.R, 0.0).sum(0)
    return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)


CTRLS = {"ctrl_peek_next_row": _ctrl_peek_next, "ctrl_full_sample_mean": _ctrl_full_mean}


def main():
    t_all = time.time()
    r, md = load_ref()
    facts = load_modules()
    grid = monthly_grid(r.dates, "2021-01-01", "2026-09-30")
    print("月频网格 %d 个日期" % len(grid))
    ts = [grid[5], grid[30], grid[60]]
    print("投毒测试日期：%s" % ", ".join(str(r.dates[t].date()) for t in ts))

    C_clean = Ctx(md)
    clean = {}
    for mod, names in POISON.items():
        for nm in names:
            f = facts[mod][nm]["func"]
            for t in ts:
                clean[(mod, nm, t)] = np.asarray(f(int(t), md, C_clean), dtype="float64")
    for cnm, cf in CTRLS.items():
        for t in ts:
            clean[("__ctrl__", cnm, t)] = np.asarray(cf(int(t), md, C_clean), dtype="float64")
    print("干净上下文值完成（%d 个因子×日期）" % len(clean))

    rows = []
    for mode in ("copy", "nan", "rand"):
        for t in ts:
            t1 = time.time()
            mdp = poison_md(md, t, mode)
            C_p = Ctx(mdp)
            print("  投毒上下文 [%s @ %s] 构造完成 %.1fs"
                  % (mode, r.dates[t].date(), time.time() - t1))
            for mod, names in POISON.items():
                for nm in names:
                    f = facts[mod][nm]["func"]
                    try:
                        v = np.asarray(f(int(t), mdp, C_p), dtype="float64")
                        rec = compare(clean[(mod, nm, t)], v)
                        rec["err"] = None
                    except Exception as e:                       # noqa: BLE001
                        rec = dict(identical=False, n_mismatch=-1, max_abs=float("nan"),
                                   max_rel=float("nan"), shape_ok=False,
                                   err="%s: %s" % (type(e).__name__, e))
                    rows.append(dict(module=mod, factor=nm, date=str(r.dates[t].date()),
                                     mode=mode, **rec))
            for cnm, cf in CTRLS.items():
                try:
                    v = np.asarray(cf(int(t), mdp, C_p), dtype="float64")
                    rec = compare(clean[("__ctrl__", cnm, t)], v)
                    rec["err"] = None
                except Exception as e:                           # noqa: BLE001
                    rec = dict(identical=False, n_mismatch=-1, max_abs=float("nan"),
                               max_rel=float("nan"), shape_ok=False, err=str(e))
                rows.append(dict(module="__ctrl__", factor=cnm, date=str(r.dates[t].date()),
                                 mode=mode, **rec))
            del C_p, mdp
            gc.collect()

    # ---------------- 汇总（以 copy 模式为浮点噪声下限） ----------------
    noise = {}
    for x in rows:
        if x["mode"] == "copy":
            noise[(x["module"], x["factor"], x["date"])] = x["max_abs"]
    for x in rows:
        nf = noise.get((x["module"], x["factor"], x["date"]), 0.0)
        x["noise_floor"] = nf
        lim = max(10.0 * float(nf), 1e-12)
        x["pass"] = bool(x["n_nan_mask_mismatch"] == 0 and
                         (x["max_abs"] <= 1e-9 or x["max_abs"] <= lim))

    def agg(sub):
        return dict(n=int(len(sub)),
                    n_identical=int(sum(1 for x in sub if x["identical"])),
                    n_pass=int(sum(1 for x in sub if x["pass"])),
                    n_nan_mask_mismatch=int(sum(x["n_nan_mask_mismatch"] for x in sub
                                                if x["n_nan_mask_mismatch"] >= 0)),
                    max_abs=float(np.nanmax([x["max_abs"] for x in sub])) if sub else float("nan"),
                    max_rel=float(np.nanmax([x["max_rel"] for x in sub])) if sub else float("nan"))

    summary = {}
    print("\n模块汇总（nan/rand 两种投毒；copy 模式仅用于噪声下限）")
    print("%-12s %4s %10s %8s %8s %12s %12s" %
          ("module", "n", "逐位相同", "判定通过", "NaN掩码错", "max|Δ|", "max_rel"))
    for mod in list(POISON) + ["__ctrl__"]:
        sub = [x for x in rows if x["module"] == mod and x["mode"] in ("nan", "rand")]
        summary[mod] = agg(sub)
        print("%-12s %4d %10d %8d %8d %12.3e %12.3e"
              % (mod, summary[mod]["n"], summary[mod]["n_identical"], summary[mod]["n_pass"],
                 summary[mod]["n_nan_mask_mismatch"], summary[mod]["max_abs"],
                 summary[mod]["max_rel"]))

    print("\n噪声下限（copy 模式，逐模块最大绝对差）：")
    for mod in list(POISON) + ["__ctrl__"]:
        sub = [x for x in rows if x["module"] == mod and x["mode"] == "copy"]
        print("  %-12s %.3e" % (mod, float(np.nanmax([x["max_abs"] for x in sub])) if sub else 0.0))

    n_fail = int(sum(1 for x in rows
                     if x["module"] != "__ctrl__" and x["mode"] in ("nan", "rand") and not x["pass"]))
    print("\n非控制组未通过的 (因子,日期,模式) 组合数 = %d" % n_fail)
    for x in rows:
        if x["module"] != "__ctrl__" and x["mode"] in ("nan", "rand") and not x["pass"]:
            print("  %-12s %-24s %s %-4s max|Δ|=%.3e 噪声下限=%.3e NaN掩码错=%d err=%s"
                  % (x["module"], x["factor"], x["date"], x["mode"], x["max_abs"],
                     x["noise_floor"], x["n_nan_mask_mismatch"], x["err"]))

    out = dict(dates=[str(r.dates[t].date()) for t in ts], modes=["copy", "nan", "rand"],
               poison_factors=POISON, summary=summary, rows=rows,
               n_fail=int(n_fail), seconds=time.time() - t_all)
    with open(os.path.join(HERE, "v2_t1_poison.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
    print("\n总耗时 %.1fs → v2_t1_poison.json" % (time.time() - t_all))


if __name__ == "__main__":
    main()