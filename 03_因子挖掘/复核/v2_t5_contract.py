# -*- coding: utf-8 -*-
"""v2 复核 · 附加验证：320 个新增因子的全量契约体检 + 全量投毒因果性。

① 契约：返回 shape=(5451,)、无 ±inf、不抛异常、desc/direction 合法、前缀与模块一致、
   跨模块无重名；在 2 个月频调仓日上检查横截面标准差。
② 因果性：在 1 个日期上对**全部 320 个因子**做 copy/nan/rand 三种上下文重算，
   与干净上下文逐位比较；copy 模式给出浮点噪声下限（数组来源不同导致的 1ulp 级差异）。
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

from v2_lib import load_ref, monthly_grid                      # noqa: E402
from v2_modules import load_modules, poison_md, compare        # noqa: E402
from factor_zoo.ctx import Ctx                                 # noqa: E402

PREFIX = {"reversal": "rev_", "risk2": "rk2_", "liquidity2": "lq2_", "flow2": "fl2_",
          "chip": "ch_", "trend2": "tr2_", "pattern": "pt_", "calendar": "cal_",
          "interaction": "ix_", "marketrel": "mk_"}


def main():
    t_all = time.time()
    r, md = load_ref()
    facts = load_modules()
    grid = monthly_grid(r.dates, "2021-01-01", "2026-09-30")
    t_poison = grid[30]
    t_second = grid[65]
    print("投毒日期 %s；第二个契约日期 %s；模块 %d，因子合计 %d"
          % (r.dates[t_poison].date(), r.dates[t_second].date(), len(facts),
             sum(len(v) for v in facts.values())))

    # ---------------- 元数据检查 ----------------
    meta = []
    seen = {}
    for mod, ff in facts.items():
        for nm, spec in ff.items():
            errs = []
            if not nm.startswith(PREFIX[mod]):
                errs.append("前缀不符（应 %s）" % PREFIX[mod])
            if nm in seen:
                errs.append("与 %s 重名" % seen[nm])
            seen[nm] = mod
            if int(spec.get("direction", 0)) not in (-1, 1):
                errs.append("direction=%r 非法" % spec.get("direction"))
            if not str(spec.get("desc", "")).strip():
                errs.append("desc 为空")
            if not callable(spec.get("func")):
                errs.append("func 不可调用")
            meta.append(dict(module=mod, factor=nm, direction=int(spec.get("direction", 0)),
                             desc=str(spec.get("desc", "")), meta_errors=errs))
    n_meta_err = sum(1 for x in meta if x["meta_errors"])
    print("元数据检查：%d 个因子，异常 %d 个" % (len(meta), n_meta_err))
    for x in meta:
        if x["meta_errors"]:
            print("  [META] %-14s %-26s %s" % (x["module"], x["factor"], "; ".join(x["meta_errors"])))

    # ---------------- 契约 + 常数性 ----------------
    C = Ctx(md)
    clean = {}          # (mod, nm) -> {(t): value}
    contract = []
    for mod, ff in facts.items():
        for nm, spec in ff.items():
            f = spec["func"]
            rec = dict(module=mod, factor=nm, shape_bad=[], exceptions=[], n_inf=0,
                       sd_zero_dates=[])
            for t in (t_poison, t_second):
                try:
                    v = np.asarray(f(int(t), md, C), dtype="float64")
                except Exception as e:                           # noqa: BLE001
                    rec["exceptions"].append("%s: %s" % (r.dates[t].date(), e))
                    continue
                if v.shape != (r.k,):
                    rec["shape_bad"].append("%s: %s" % (r.dates[t].date(), v.shape))
                    continue
                rec["n_inf"] += int(np.isinf(v).sum())
                clean[(mod, nm, int(t))] = v
                sd = np.nanstd(v)
                if not (np.isfinite(sd) and sd > 0):
                    rec["sd_zero_dates"].append(str(r.dates[t].date()))
            contract.append(rec)
    n_bad = sum(1 for x in contract if x["shape_bad"] or x["exceptions"] or x["n_inf"])
    n_sd0 = sum(1 for x in contract if x["sd_zero_dates"])
    print("契约检查（2 个日期）：形状/异常/inf 违例 %d 个因子；横截面标准差为 0 的因子 %d 个"
          % (n_bad, n_sd0))
    for x in contract:
        if x["shape_bad"] or x["exceptions"] or x["n_inf"]:
            print("  [CONTRACT] %-14s %-26s shape=%s exc=%s inf=%d"
                  % (x["module"], x["factor"], x["shape_bad"], x["exceptions"], x["n_inf"]))
    for x in contract:
        if x["sd_zero_dates"]:
            print("  [常数] %-14s %-26s %s" % (x["module"], x["factor"], x["sd_zero_dates"]))

    # ---------------- 全量投毒 ----------------
    rows = []
    for mode in ("copy", "nan", "rand"):
        t1 = time.time()
        mdp = poison_md(md, t_poison, mode)
        C_p = Ctx(mdp)
        print("  投毒上下文 [%s @ %s] 构造完成 %.1fs" % (mode, r.dates[t_poison].date(), time.time() - t1))
        for mod, ff in facts.items():
            for nm, spec in ff.items():
                try:
                    v = np.asarray(spec["func"](int(t_poison), mdp, C_p), dtype="float64")
                    rec = compare(clean[(mod, nm, int(t_poison))], v)
                    rec["err"] = None
                except Exception as e:                           # noqa: BLE001
                    rec = dict(identical=False, n_mismatch=-1, max_abs=float("nan"),
                               max_rel=float("nan"), n_nan_mask_mismatch=-1,
                               err="%s: %s" % (type(e).__name__, e))
                rows.append(dict(module=mod, factor=nm, mode=mode, **rec))
        del C_p, mdp

    noise = {(x["module"], x["factor"]): x["max_abs"] for x in rows if x["mode"] == "copy"}
    n_fail = 0
    n_identical = 0
    worst = []
    for x in rows:
        if x["mode"] == "copy":
            continue
        nf = float(noise.get((x["module"], x["factor"]), 0.0))
        lim = max(10.0 * nf, 1e-12)
        ok = bool(x["n_nan_mask_mismatch"] == 0 and (x["max_abs"] <= 1e-9 or x["max_abs"] <= lim))
        if not ok:
            n_fail += 1
            worst.append((x, nf))
        if x["identical"]:
            n_identical += 1
    n_pair = int(sum(1 for x in rows if x["mode"] != "copy"))
    print("\n全量投毒：%d 个因子 × 2 种投毒 = %d 次重算，逐位相同 %d 次，"
          "超出噪声下限的违例 %d 次" % (len(seen), n_pair, n_identical, n_fail))
    for x, nf in sorted(worst, key=lambda z: -z[0]["max_abs"])[:20]:
        print("  [POISON] %-14s %-26s %-4s max|Δ|=%.3e 噪声下限=%.3e mask错=%d"
              % (x["module"], x["factor"], x["mode"], x["max_abs"], nf,
                 x["n_nan_mask_mismatch"]))
    mx_rel = max((x["max_rel"] for x in rows if x["mode"] != "copy" and np.isfinite(x["max_rel"])),
                 default=0.0)
    mx_abs = max((x["max_abs"] for x in rows if x["mode"] != "copy" and np.isfinite(x["max_abs"])),
                 default=0.0)
    print("非 copy 模式：max|Δ|=%.3e，max_rel=%.3e（含 1ulp 级浮点噪声）" % (mx_abs, mx_rel))

    with open(os.path.join(HERE, "v2_t5_contract.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(n_factors=len(seen), meta=meta, contract=contract,
                       poison_rows=rows, n_poison_fail=int(n_fail),
                       n_poison_identical=int(n_identical), seconds=time.time() - t_all),
                  fh, ensure_ascii=False, indent=1, default=str)
    print("总耗时 %.1fs → v2_t5_contract.json" % (time.time() - t_all))


if __name__ == "__main__":
    main()