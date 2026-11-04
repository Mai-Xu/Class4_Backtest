# -*- coding: utf-8 -*-
"""Turn V2 accept/reject evidence into a policy memory for the next round."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]; V2=ROOT/"out/03_因子挖掘/evo_formula/v2"
trials=pd.read_csv(V2/"walkforward_trials.csv"); tax=pd.read_csv(V2/"failure_taxonomy.csv")
ev=trials[trials.arm=="evolved"].copy(); ev["op"]=ev["op"].fillna("seed")
# operator-level success proxy: fraction in top quartile by fitness
q=ev.fitness.quantile(.75); g=ev.groupby("op").agg(n=("fitness","size"),fitness_mean=("fitness","mean"),fitness_median=("fitness","median"),top_quartile_rate=("fitness",lambda s:(s>=q).mean())).reset_index()
g["recommended_weight"]=(g.fitness_mean-g.fitness_mean.min()+1e-6); g["recommended_weight"]/=g.recommended_weight.sum()
reasons=tax.reason.value_counts().to_dict()
mem=dict(version="formula_policy_memory_v2",source="walkforward_trials.csv",top_quartile_threshold=float(q),
         operator_stats=g.to_dict(orient="records"),failure_counts=reasons,
         next_round_directives=["prefer operators with positive top-quartile_rate, but keep >=25% exploration",
             "penalise formulas with >5 nodes and holdout turnover >0.65",
             "require direction consistency >=0.80 across rolling folds",
             "run matched random arm with exactly the same candidate budget",
             "never use 2025-2026 as a new blind OOS; wait for a genuinely future window"])
(V2/"policy_memory.json").write_text(json.dumps(mem,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(mem,ensure_ascii=False,indent=2))
