# -*- coding: utf-8 -*-
"""V2 audit: multi-seed budget comparison, multiple-testing, failure taxonomy, cost sensitivity.

This does NOT select a new candidate on OOS. It only audits the already-frozen
candidate and compares the search arms on train/validation (development data).
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
import evo_core as e

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = e.ROOT
sys.path.insert(0, str(ROOT / "04_策略选股器"))
import fastsim


def bootstrap_ci(x, block=3, n_boot=3000, seed=7):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 5: return (np.nan, np.nan)
    rng = np.random.default_rng(seed); n = len(x); vals = []
    for _ in range(n_boot):
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        idx = np.concatenate([(s + np.arange(block)) % n for s in starts])[:n]
        vals.append(float(x[idx].mean()))
    return (float(np.quantile(vals, .025)), float(np.quantile(vals, .975)))


def best_valid(records, data, dirs, top_n=8):
    top = sorted(records, key=lambda r: r["fitness"], reverse=True)[:top_n]
    out = [e.validation_record(r, data, dirs) for r in top]
    return max(out, key=lambda x: x["combined"]), out


def pvals_for(rec, data, dirs, n_trials):
    vidx = e.idx_for(data["panel_dates"], *e.VALID)
    st, turn, ic = e.stats_for(vidx, rec["genome"], data, dirs)
    ic = np.asarray(ic, float); ic = ic[np.isfinite(ic)]
    t, p = ttest_1samp(ic, 0.0)
    lo, hi = bootstrap_ci(ic)
    return dict(n_valid=len(ic), mean_ic=float(ic.mean()), t=float(t), p_nominal=float(p),
                p_bonferroni=float(min(1.0, p * n_trials)), ci_lo=lo, ci_hi=hi)


def failure_reason(rec, val):
    tr = rec; va = val
    if tr["ic"] <= 0.02: return "train_ic<=0.02"
    if tr["t"] <= 2.0: return "train_t<=2"
    if va["ic"] <= 0.02: return "valid_ic<=0.02"
    if va["t"] <= 1.5: return "valid_t<=1.5"
    ret = va["icir"] / tr["icir"] if tr["icir"] > 0 else -1
    if ret < 0.45: return "retention<0.45"
    if rec["n_active"] > 16: return "too_complex"
    return "pass"


def main():
    t0 = time.time(); data = e.build_data(); train_idx = e.idx_for(data["panel_dates"], *e.TRAIN)
    dirs, stats, pool = e.directions_on_train(data, train_idx)
    seeds = [20261030, 20261031, 20261032, 20261033, 20261034]
    rows = []
    # One representative seed is also used for the full failure taxonomy.
    full_ev = None
    for si, seed in enumerate(seeds):
        print(f"[seed {seed}] evolution + random budget ...")
        ev, base = e.run_evolution(data, dirs, stats, pool, seed=seed)
        rn = e.random_budget(pool, data["families"], data, dirs, len(ev), seed=seed + 100)
        bev, ev_val = best_valid(ev, data, dirs)
        brv, rn_val = best_valid(rn, data, dirs)
        pev = pvals_for(bev["train"], data, dirs, n_trials=len(ev))
        prn = pvals_for(brv["train"], data, dirs, n_trials=len(rn))
        rows.append(dict(seed=seed,
            evolved_train_best=max(x["fitness"] for x in ev), random_train_best=max(x["fitness"] for x in rn),
            evolved_valid_icir=bev["valid"]["icir"], random_valid_icir=brv["valid"]["icir"],
            evolved_valid_ic=bev["valid"]["ic"], random_valid_ic=brv["valid"]["ic"],
            evolved_combined=bev["combined"], random_combined=brv["combined"],
            evolved_n=bev["train"]["n_active"], random_n=brv["train"]["n_active"],
            evolved_t=pev["t"], evolved_p_adj=pev["p_bonferroni"], evolved_ci_lo=pev["ci_lo"], evolved_ci_hi=pev["ci_hi"],
            random_t=prn["t"], random_p_adj=prn["p_bonferroni"], random_ci_lo=prn["ci_lo"], random_ci_hi=prn["ci_hi"]))
        if si == 0: full_ev = ev
    df = pd.DataFrame(rows); df.to_csv(e.OUT / "v2_multiseed.csv", index=False, encoding="utf-8-sig")
    # Full failure taxonomy on the representative seed (development validation only).
    tax = []
    for rec in full_ev:
        val = e.validation_record(rec, data, dirs)
        tax.append(dict(generation=rec.get("generation"), n_active=rec["n_active"],
                        train_ic=rec["ic"], train_icir=rec["icir"], train_t=rec["t"],
                        valid_ic=val["valid"]["ic"], valid_icir=val["valid"]["icir"], valid_t=val["valid"]["t"],
                        reason=failure_reason(rec, val["valid"])))
    tdf = pd.DataFrame(tax); tdf.to_csv(e.OUT / "v2_failure_taxonomy.csv", index=False, encoding="utf-8-sig")
    reason_counts = tdf["reason"].value_counts().rename_axis("reason").reset_index(name="count")
    # Cost sensitivity of the already-frozen main candidate; no re-selection on OOS.
    freeze = json.loads((e.OUT / "freeze.json").read_text(encoding="utf-8"))
    main_g = freeze["selection"]["main"]["genome"]
    meta = json.loads((ROOT / "out/cache/selector/meta.json").read_text(encoding="utf-8"))
    sig = [pd.Timestamp(d) for d in meta["signal_dates"]]
    sidx = np.array([data["panel_dates"].get_loc(d) for d in sig], dtype=int)
    scores = e.score_genome(main_g, sidx, data, dirs); uni = data["uni"][sidx]
    plan, _ = fastsim.build_plan(scores, meta["signal_dates"], meta["trade_dates"], uni,
                                 n_hold=meta["is_n_hold"], buffer=1.0, liquidate_signal=meta["liq_signal"])
    px = np.load(ROOT / "out/cache/selector/prices_oos.npz")
    cost_rows = []
    for slip in [0.0, 0.001, 0.002, 0.003]:
        sim = fastsim.simulate(plan, px["adj_open"], px["adj_close"], n_hold=meta["is_n_hold"],
                               fee_buy=fastsim.FEE_BUY + slip, fee_sell=fastsim.FEE_SELL + slip)
        nav = fastsim.nav_ratio(sim["nav"]); st = fastsim.perf_stats(nav)
        cost_rows.append(dict(slippage=slip, cagr=st["年化收益"], sharpe=st["夏普"], mdd=st["最大回撤"],
                              ann_turnover=fastsim.turnover_ann(sim["turnover"])))
    cdf = pd.DataFrame(cost_rows); cdf.to_csv(e.OUT / "v2_cost_sensitivity.csv", index=False, encoding="utf-8-sig")
    # Report
    def tb(d): return "\n".join(["| "+" | ".join(d.columns)+" |", "| "+" | ".join(["---"]*len(d.columns))+" |"] +
                                ["| "+" | ".join(str(v) for v in r)+" |" for r in d.itertuples(index=False)])
    report = f"""# V2 改进审计

这是对已有实验的**开发集审计**，不是新的 OOS 选股。2025–2026 只用于冻结主候选的成本稳健性，不再重新挑冠军。

## 1. 多随机种子：进化 vs 同预算随机

{rows and tb(df)}

- 5 个种子，每个种子进化与随机各 59 个候选；只比较 train/validation。
- 关键看 **evolved_valid_icir − random_valid_icir** 是否稳定为正，而不是单次最优。
- `p_bonferroni` 把 59 次搜索预算计入多重检验；`ci` 是验证期月度 RankIC 的 3-block bootstrap 95% 区间。

## 2. 失败原因分类（种子 20261030）

{tb(reason_counts)}

## 3. 冻结主候选的成本敏感性（OOS，仅稳健性）

{tb(cdf)}

## 4. 改进结论

- V1 的主要问题是“单种子 + 只比一次随机搜索 + 只看最终冠军”。
- V2 至少补上了：多随机种子、同预算随机对照、多重检验校正、bootstrap 区间、失败原因台账、成本敏感性。
- 仍然没有解决的问题：公式级进化、行业/市值中性化、真实逐笔撮合、未来独立数据。当前结果只能作为课程研究实验。
"""
    (e.OUT / "v2_audit_report.md").write_text(report, encoding="utf-8")
    print(report.encode("gbk", errors="replace").decode("gbk")); print("elapsed", time.time()-t0)

if __name__ == "__main__": main()
