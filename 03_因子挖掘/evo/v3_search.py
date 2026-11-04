# -*- coding: utf-8 -*-
"""V3: nested development split + gap penalty + budget-matched random control.

Inner train: 2021-01..2022-06
Inner holdout: 2022-07..2022-12
Outer validation: 2023-01..2024-12
OOS is NOT touched in this script. V3 is a new research version for future data.
"""
from __future__ import annotations
import sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import evo_core as e

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

INNER_TRAIN = ("2021-01-01", "2022-06-30")
INNER_HOLD = ("2022-07-01", "2022-12-31")


def fit_v3(genome, itrain, ihold, data, dirs):
    tr = e.evaluate(genome, itrain, data, dirs)
    ho = e.evaluate(genome, ihold, data, dirs)
    ret = ho["icir"] / tr["icir"] if tr["icir"] > 0 else -1.0
    gap = max(0.0, tr["icir"] - ho["icir"]) / max(abs(tr["icir"]), 1e-9)
    score = (0.55 * tr["icir"] + 0.35 * ho["icir"] + 0.10 * max(0.0, min(1.0, ret))
             - 0.04 * max(0, len(genome) - 8)
             - 0.6 * max(0.0, (ho["turnover_top1700"] if np.isfinite(ho["turnover_top1700"]) else 0.0) - 0.65)
             - 0.25 * min(1.0, gap))
    return score, tr, ho


def run_arm_v3(data, dirs, stats, pool, seed, budget=59):
    rng = np.random.default_rng(seed); fam = data["families"]
    itrain = e.idx_for(data["panel_dates"], *INNER_TRAIN)
    ihold = e.idx_for(data["panel_dates"], *INNER_HOLD)
    ranked = sorted(pool, key=lambda nm: abs(stats[nm]["icir"]), reverse=True)
    base, cnt = {}, Counter()
    for nm in ranked:
        f = fam.get(nm, "?")
        if cnt[f] >= 2: continue
        base[nm] = abs(stats[nm]["icir"]); cnt[f] += 1
        if len(base) >= 8: break
    pop = [e.normalized(base, fam, pool)]
    while len(pop) < 24:
        g = e.random_genome(rng, pool, fam)
        if g not in pop: pop.append(g)
    recs = {}; fit = []
    for gen in range(3):
        if gen == 0: batch = pop
        else:
            elites = [pop[i] for i in np.argsort(fit)[-6:]]
            batch = list(elites)
            while len(batch) < 24:
                p1 = e.tournament(pop, fit, rng); p2 = e.tournament(pop, fit, rng)
                child = e.mutate(p1, rng, pool, fam)
                if rng.random() < 0.25: child = e.mutate(p2, rng, pool, fam)
                child = e.normalized(child, fam, pool)
                if child not in batch: batch.append(child)
        nf = []
        for g in batch:
            key = tuple(sorted(g.items()))
            if key not in recs:
                s, tr, ho = fit_v3(g, itrain, ihold, data, dirs)
                recs[key] = dict(genome=g, generation=gen, fitness=float(s), inner_train=tr, inner_hold=ho)
            nf.append(recs[key]["fitness"])
        pop, fit = batch, nf
    return sorted(recs.values(), key=lambda r: r["fitness"], reverse=True)[:budget]


def random_arm_v3(data, dirs, pool, seed, n):
    rng = np.random.default_rng(seed); itrain = e.idx_for(data["panel_dates"], *INNER_TRAIN)
    ihold = e.idx_for(data["panel_dates"], *INNER_HOLD); recs = []
    for _ in range(n):
        g = e.random_genome(rng, pool, data["families"])
        s, tr, ho = fit_v3(g, itrain, ihold, data, dirs)
        recs.append(dict(genome=g, generation=-1, fitness=float(s), inner_train=tr, inner_hold=ho))
    return sorted(recs, key=lambda r: r["fitness"], reverse=True)


def main():
    t0 = time.time(); data = e.build_data(); itrain = e.idx_for(data["panel_dates"], *INNER_TRAIN)
    dirs, stats, pool = e.directions_on_train(data, itrain)
    seeds = [20261030, 20261031, 20261032, 20261033, 20261034]
    rows = []; best_all = None
    for seed in seeds:
        print("V3 seed", seed)
        ev = run_arm_v3(data, dirs, stats, pool, seed, 59)
        rn = random_arm_v3(data, dirs, pool, seed + 200, len(ev))
        def flat(r):
            x = dict(r["inner_train"]); x["genome"] = r["genome"]
            x["generation"] = r["generation"]; x["fitness"] = r["fitness"]
            return x
        ev_val = [e.validation_record(flat(r), data, dirs) for r in ev[:8]]
        rn_val = [e.validation_record(flat(r), data, dirs) for r in rn[:8]]
        bev = max(ev_val, key=lambda x: x["combined"]); brv = max(rn_val, key=lambda x: x["combined"])
        rows.append(dict(seed=seed,
            evolved_inner_fit=ev[0]["fitness"], random_inner_fit=rn[0]["fitness"],
            evolved_outer_icir=bev["valid"]["icir"], random_outer_icir=brv["valid"]["icir"],
            evolved_outer_ic=bev["valid"]["ic"], random_outer_ic=brv["valid"]["ic"],
            evolved_combined=bev["combined"], random_combined=brv["combined"],
            evolved_n=len(bev["genome"]), random_n=len(brv["genome"])))
        if best_all is None or bev["combined"] > best_all["combined"]:
            best_all = bev
    df = pd.DataFrame(rows); df.to_csv(e.OUT / "v3_multiseed.csv", index=False, encoding="utf-8-sig")
    wins = int((df["evolved_combined"] > df["random_combined"]).sum())
    mean_diff = float((df["evolved_combined"] - df["random_combined"]).mean())
    freeze = dict(version="v3", created_at=pd.Timestamp.now().isoformat(),
                  inner_train=list(INNER_TRAIN), inner_holdout=list(INNER_HOLD), outer_validation=list(e.VALID),
                  oos_untouched=True, seed=20261030,
                  candidate=dict(genome=best_all["genome"], inner_train=best_all["train"], outer_validation=best_all["valid"],
                                 combined=best_all["combined"]),
                  note="V3 is a protocol for future data; 2025-2026 was already consumed by V1 and is not used here.")
    e.save_json(e.OUT / "v3_freeze_future.json", freeze)
    def tb(d): return "\n".join(["| "+" | ".join(d.columns)+" |", "| "+" | ".join(["---"]*len(d.columns))+" |"] +
                                ["| "+" | ".join(str(v) for v in r)+" |" for r in d.itertuples(index=False)])
    report = f"""# V3：把“验证”放回搜索内部，但把 OOS 留到未来

## 改了什么

- V2：进化只看训练集，验证集只在短名单阶段看一次；5 个种子中随机搜索经常赢。
- V3：训练内部再切出 2022H2 作为 **inner holdout**，适应度加入：
  - inner holdout ICIR；
  - inner train → inner holdout 的 retention；
  - 过大的训练/留出 gap 惩罚；
  - 复杂度与换手惩罚。
- 2023–2024 仍是 outer validation；2025–2026 **不在本脚本中使用**，因为上一轮已经消费过，不能再次冒充盲测。

## 五种子的 V3 结果

{tb(df)}

- 进化在 {wins}/5 个种子上 combined 高于同预算随机；
- 平均 combined 差值：{mean_diff:+.3f}。
- 如果这个差值仍然很小或为负，说明当前“进化”仍只是带约束的随机搜索，不能包装成 RSI。

## 解读

1. V3 的主要改进是**搜索内部有 holdout**，不是换一个更复杂的模型。
2. 如果 V3 仍不能稳定赢随机：下一步应改因子表示/公式生成，而不是继续调权重。
3. `v3_freeze_future.json` 只供未来数据预注册；不能拿 2025–2026 再做第二次选择。
"""
    (e.OUT / "v3_report.md").write_text(report, encoding="utf-8")
    print(report.encode("gbk", errors="replace").decode("gbk")); print("elapsed", time.time()-t0)

if __name__ == "__main__": main()
