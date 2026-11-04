# -*- coding: utf-8 -*-
"""Train/validate the evolutionary factor-combination search; write freeze."""
from __future__ import annotations
import hashlib, json, sys, time
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import pandas as pd
import evo_core as e


def sha256_file(path, limit=None):
    h = hashlib.sha256(); left = limit
    with open(path, "rb") as f:
        while True:
            n = 1 << 20 if left is None else min(1 << 20, left)
            if n <= 0: break
            b = f.read(n)
            if not b: break
            h.update(b)
            if left is not None: left -= len(b)
    return h.hexdigest()


def record_row(role, r):
    tr, va = r["train"], r["valid"]
    return dict(role=role, generation=r.get("generation"), n_active=tr.get("n_active"),
                train_ic=tr.get("ic"), train_icir=tr.get("icir"), train_t=tr.get("t"),
                train_fold_min=tr.get("fold_min_icir"), train_turn=tr.get("turnover_top1700"),
                valid_ic=va.get("ic"), valid_icir=va.get("icir"), valid_t=va.get("t"),
                valid_fold_min=va.get("fold_min_icir"), valid_turn=va.get("turnover_top1700"),
                combined=r.get("combined"), retention=r.get("retention"), gate_pass=r.get("gate_pass"),
                factors=";".join(f"{k}:{v:.2f}" for k, v in sorted(r["genome"].items())))


def main():
    t0 = time.time(); data = e.build_data()
    train_idx = e.idx_for(data["panel_dates"], *e.TRAIN)
    dirs, stats, pool = e.directions_on_train(data, train_idx)
    print(f"train months={len(train_idx)}, factors={len(data['names'])}, usable pool={len(pool)}")
    evolved, base = e.run_evolution(data, dirs, stats, pool)
    bst = e.evaluate(base, train_idx, data, dirs)
    base_rec = dict(genome=base, generation=-2, fitness=e.train_fitness(bst, bst["turnover_top1700"], len(base)))
    base_rec.update(bst)
    rand = e.random_budget(pool, data["families"], data, dirs, len(evolved))
    ev_top = sorted(evolved, key=lambda r: r["fitness"], reverse=True)[:8]
    rn_top = sorted(rand, key=lambda r: r["fitness"], reverse=True)[:8]
    val_records = []
    for r in ev_top:
        x = e.validation_record(r, data, dirs); x["role"] = "evolved"; val_records.append(x)
    for r in rn_top:
        x = e.validation_record(r, data, dirs); x["role"] = "random"; val_records.append(x)
    base_val = e.validation_record(base_rec, data, dirs); base_val["role"] = "train_icir8_baseline"; val_records.append(base_val)
    best_ev = max([x for x in val_records if x["role"] == "evolved"], key=lambda x: x["combined"])
    flip = {k: -v for k, v in dirs.items()}
    ctrl_src = dict(best_ev["train"]); ctrl_src["genome"] = best_ev["genome"]
    ctrl = e.validation_record(ctrl_src, data, flip)
    ctrl["role"] = "sign_flip_control"; val_records.append(ctrl)
    rerun = e.validation_record(ctrl_src, data, dirs)
    rerun_equal = (abs(rerun["valid"]["ic"] - best_ev["valid"]["ic"]) < 1e-12 and
                   abs(rerun["valid"]["icir"] - best_ev["valid"]["icir"]) < 1e-12)
    passed = [x for x in val_records if x["gate_pass"] and x["role"] in ("evolved", "random", "train_icir8_baseline")]
    main_rec = max(passed, key=lambda x: x["combined"]) if passed else max(val_records, key=lambda x: x["combined"])
    simple_pool = [x for x in passed if x["train"]["n_active"] <= 8]
    simple = max(simple_pool, key=lambda x: x["combined"]) if simple_pool else None
    rows = [record_row(x["role"], x) for x in sorted(val_records, key=lambda z: z["combined"], reverse=True)]
    pd.DataFrame(rows).to_csv(e.OUT / "validation_table.csv", index=False, encoding="utf-8-sig")
    rand_val = [x for x in val_records if x["role"] == "random"]
    rb = max(rand_val, key=lambda x: x["combined"]) if rand_val else None
    freeze = dict(
        created_at=pd.Timestamp.now().isoformat(), seed=20261030,
        train=list(e.TRAIN), validation=list(e.VALID), oos_untouched=True,
        n_factors=len(data["names"]), n_pool=len(pool), n_evolved_evaluations=len(evolved),
        n_random_evaluations=len(rand), n_validation_looks=len(val_records),
        selection=dict(
            main=dict(role=main_rec["role"], genome=main_rec["genome"], train=main_rec["train"], valid=main_rec["valid"], combined=main_rec["combined"]),
            simple=dict(role=simple["role"], genome=simple["genome"], train=simple["train"], valid=simple["valid"], combined=simple["combined"]) if simple else None,
            train_icir8_baseline=dict(genome=base, train=base_val["train"], valid=base_val["valid"]),
            random_best=dict(role="random", genome=rb["genome"], train=rb["train"], valid=rb["valid"], combined=rb["combined"]) if rb else None,
        ),
        control=dict(sign_flip_valid_ic=ctrl["valid"]["ic"], sign_flip_valid_icir=ctrl["valid"]["icir"], rerun_equal=bool(rerun_equal)),
        data=dict(z_sha256=sha256_file(e.CACHE/'rankz.npy', limit=8*1024*1024), z_bytes=(e.CACHE/'rankz.npy').stat().st_size,
                  y_sha256=sha256_file(e.CACHE/'y20.npy'), universe_sha256=sha256_file(e.CACHE/'universe.npy')),
        known_limits=["factor panels are precomputed with the course factor library; evolving only subset/weights, not formulas",
            "validation was looked at once for a shortlist of 18 candidate combinations",
            "2025-2026 OOS is only 21 months and cannot support precise annualized claims",
            "fastsim backtest is a vectorized proxy; the course engine is the execution-grade check",
            "family cap and complexity/turnover penalties are researcher choices"],
    )
    e.save_json(e.OUT / "freeze.json", freeze)
    top = pd.DataFrame(rows).head(12)
    def md_table(df):
        return "\n".join(["| "+" | ".join(df.columns)+" |", "| "+" | ".join(["---"]*len(df.columns))+" |"] +
                         ["| "+" | ".join(str(v) for v in row)+" |" for row in df.itertuples(index=False)])
    txt = f"""# Class4 自进化因子组合实验（训练/验证）

- 训练：{e.TRAIN[0]} ~ {e.TRAIN[1]}（{len(train_idx)} 个月）
- 验证：{e.VALID[0]} ~ {e.VALID[1]}（单次短名单检验）
- 2025-2026：**尚未读取，freeze.json 写入后才允许 OOS**
- 因子宇宙：{len(data['names'])} 个聚类代表；训练可用池：{len(pool)}
- 搜索预算：进化 {len(evolved)} 个候选；匹配随机对照 {len(rand)} 个
- 验证短名单：{len(val_records)} 个（不是全量重选）
- 负对照复现：{rerun_equal}

## 验证短名单（按 combined 排序）

{md_table(top)}

## 冻结选择

- 主候选：`{main_rec['role']}`，{main_rec['train']['n_active']} 因子，train ICIR={main_rec['train']['icir']:.3f}，valid ICIR={main_rec['valid']['icir']:.3f}
- 简形式：{simple['role'] + ' / ' + str(simple['train']['n_active']) + ' factors' if simple else '无（无 ≤8 因子候选通过闸门）'}
- 匹配随机最佳：{freeze['selection']['random_best']['valid']['icir']:.3f} valid ICIR
- 符号翻转对照：valid IC={ctrl['valid']['ic']:.4f}，ICIR={ctrl['valid']['icir']:.3f}

## 选择逻辑（事前）

1. train 只用于方向、权重和进化适应度；2023-2024 仅对短名单看一次。
2. 通过闸门：train IC>0.02、valid IC>0.02、train t>2、valid t>1.5。
3. v2 combined = 0.25·train ICIR + 0.55·valid ICIR + 2·retention − 复杂度/换手惩罚；retention=valid ICIR / train ICIR，并要求 ≥0.45。
4. 与匹配预算随机搜索比较；若进化优势接近噪声，则不以“进化”叙事替代预算解释。
5. 写入 freeze 后，才可执行 `run_oos.py`；OOS 结果不回改选择。
"""
    (e.OUT / "train_report.md").write_text(txt, encoding="utf-8")
    print(txt.encode("gbk", errors="replace").decode("gbk")); print("elapsed", time.time()-t0)

if __name__ == "__main__": main()

