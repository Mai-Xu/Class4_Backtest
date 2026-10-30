# -*- coding: utf-8 -*-
"""Evolve cross-sectional formula trees on Class4 data (train -> one validation)."""
from __future__ import annotations
import hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import formula_core as fc

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass


def top_turnover(score, uni, n_hold=1700):
    prev=None; vals=[]
    for i in range(score.shape[0]):
        cand=np.flatnonzero(uni[i] & np.isfinite(score[i]))
        if len(cand)<n_hold: continue
        cur=set(cand[np.argsort(score[i,cand])[-n_hold:]].tolist())
        if prev is not None: vals.append(1-len(cur&prev)/float(n_hold))
        prev=cur
    return float(np.mean(vals)) if vals else np.nan


def score_tree(tree, idx, data):
    s=fc.eval_tree(tree,data)[idx]
    return np.where(data["uni"][idx],s,np.nan)


def ic_stats(tree, idx, data, force_direction=None):
    s=score_tree(tree,idx,data); y=data["y20"][idx]; u=data["uni"][idx]
    ic=fc.mc.ic_series(s,y,u); st=fc.mc.ic_stats(ic,12.0)
    d=force_direction if force_direction is not None else (1.0 if st["ic"]>=0 else -1.0)
    st["direction"]=float(d); st["icir_adj"]=float(st["icir"]*d); st["t_adj"]=float(st["t"]*d)
    # 4 contiguous folds, direction fixed from the parent segment.
    vals=[]
    for part in np.array_split(np.arange(len(idx)),4):
        sub=idx[part]
        if len(sub)<3: continue
        ss=score_tree(tree,sub,data); ii=fc.mc.ic_series(ss,data["y20"][sub],data["uni"][sub])
        vals.append(fc.mc.ic_stats(ii)["icir"]*d)
    st["fold_min_icir_adj"]=float(np.nanmin(vals)) if vals and np.isfinite(vals).any() else np.nan
    st["turnover_top1700"]=top_turnover(s,u)
    st["nodes"]=fc.tree_nodes(tree)
    return st


def fitness(st):
    return float(0.80*st["icir_adj"] + 0.20*min(0.0,st["fold_min_icir_adj"])
                 -0.04*max(0,st["nodes"]-5)
                 -0.50*max(0.0,(st["turnover_top1700"] if np.isfinite(st["turnover_top1700"]) else 0.0)-0.65))


def cap_tree(t, features, rng):
    tries=0
    while fc.tree_nodes(t)>fc.MAX_NODES and tries<20:
        # Replace an oversized child by one of its immediate children or a leaf.
        if t[0]=="feat": break
        children=[x for x in t[1:] if isinstance(x,tuple)]
        t = children[rng.integers(len(children))] if children else ("feat",str(rng.choice(features)))
        tries+=1
    if fc.tree_nodes(t)>fc.MAX_NODES:
        t=("feat",str(rng.choice(features)))
    return t


def mutate(parent, rng, features):
    op=str(rng.choice(["wrap","binary","cross"]))
    if op=="wrap":
        child=(str(rng.choice(["neg","abs","square","rank","z"])),parent)
    elif op=="binary":
        child=(str(rng.choice(["add","sub","mul","div"])),parent,fc.random_tree(rng,features,2))
    else:
        child=(str(rng.choice(["add","sub","mul"])),parent,fc.random_tree(rng,features,2))
    return cap_tree(child,features,rng)


def evaluate_tree(tree, idx, data):
    st=ic_stats(tree,idx,data); st["tree"]=tree; st["formula"]=fc.expr(tree); st["fitness"]=fitness(st)
    return st


def tournament(records, rng, k=3):
    ids=rng.choice(len(records),size=min(k,len(records)),replace=False)
    return records[max(ids,key=lambda i:records[i]["fitness"])]


def run_evolution(data, seed=20261030, pop_size=40, gens=4):
    rng=np.random.default_rng(seed); feats=data["features"]; train_idx=fc.idx_for(data["panel_dates"],*fc.TRAIN)
    seeds=["ret5","ret20","max1d20","max5d20","udvol60","illiq20","amtrel20","pos20","vol20","madev60","ovn20","pvcorr20"]
    initial=[("feat",x) for x in seeds] + [("neg",("feat",x)) for x in seeds[:6]]
    pop=[]; seen=set()
    for t in initial:
        k=fc.tree_key(t)
        if k not in seen: pop.append(t); seen.add(k)
    while len(pop)<pop_size:
        t=cap_tree(fc.random_tree(rng,feats),feats,rng); k=fc.tree_key(t)
        if k not in seen: pop.append(t); seen.add(k)
    records={}; bykey={}
    for gen in range(gens):
        batch=[]
        if gen==0:
            batch=pop
        else:
            elites=sorted(records.values(),key=lambda x:x["fitness"],reverse=True)[:8]
            batch=[x["tree"] for x in elites]
            while len(batch)<pop_size:
                p1=tournament(list(records.values()),rng)["tree"]
                p2=tournament(list(records.values()),rng)["tree"]
                child=mutate(p1,rng,feats)
                if rng.random()<0.30: child=mutate(p2,rng,feats)
                child=cap_tree(child,feats,rng); k=fc.tree_key(child)
                if k not in bykey: batch.append(child)
        new=[]
        for t in batch:
            k=fc.tree_key(t)
            if k not in bykey:
                bykey[k]=evaluate_tree(t,train_idx,data)
            new.append(bykey[k])
        records={x["formula"]:x for x in bykey.values()}
        print(f"generation {gen+1}: best fitness={max(x['fitness'] for x in records.values()):.3f}, unique={len(records)}")
    return list(records.values()), bykey


def random_control(data, n, seed=20261031):
    rng=np.random.default_rng(seed); feats=data["features"]; idx=fc.idx_for(data["panel_dates"],*fc.TRAIN)
    out=[]; seen=set()
    while len(out)<n:
        t=cap_tree(fc.random_tree(rng,feats),feats,rng); k=fc.tree_key(t)
        if k in seen: continue
        seen.add(k); out.append(evaluate_tree(t,idx,data))
    return out


def validation_record(rec, data):
    tr=rec; valid_idx=fc.idx_for(data["panel_dates"],*fc.VALID)
    st=ic_stats(rec["tree"],valid_idx,data,force_direction=rec["direction"])
    retention=(st["icir_adj"]/tr["icir_adj"]) if tr["icir_adj"]>0 else -1.0
    ok=(abs(tr["ic"])>0.02 and (st["ic"]*tr["direction"])>0.015 and abs(tr["t_adj"])>2.0
        and st["t_adj"]>1.5 and retention>=0.45 and tr["nodes"]<=fc.MAX_NODES)
    combined=(0.25*tr["icir_adj"]+0.55*st["icir_adj"]+2.0*max(0,min(1,retention))
              -0.04*tr["nodes"]-0.3*(st["turnover_top1700"] if np.isfinite(st["turnover_top1700"]) else 0))
    return dict(tree=rec["tree"], formula=rec["formula"], train=tr, valid=st,
                retention=float(retention), combined=float(combined), gate_pass=bool(ok))


def sha256_file(p, limit=None):
    h=hashlib.sha256(); left=limit
    with open(p,"rb") as f:
        while True:
            n=1<<20 if left is None else min(1<<20,left)
            if n<=0: break
            b=f.read(n)
            if not b: break
            h.update(b)
            if left is not None: left-=len(b)
    return h.hexdigest()


def main():
    t0=time.time(); data=fc.build_base(); train_idx=fc.idx_for(data["panel_dates"],*fc.TRAIN)
    print(f"features={len(data['features'])}, train months={len(train_idx)}")
    evolved,bykey=run_evolution(data)
    random=fc.random_control(data,len(evolved)) if hasattr(fc,"random_control") else None
    # The random control is implemented here to keep the primitive module lean.
    if random is None:
        rng=np.random.default_rng(20261031); random=[]; seen=set()
        while len(random)<len(evolved):
            t=cap_tree(fc.random_tree(rng,data["features"]),data["features"],rng); k=fc.tree_key(t)
            if k in seen: continue
            seen.add(k); random.append(evaluate_tree(t,train_idx,data))
    ev_top=sorted(evolved,key=lambda x:x["fitness"],reverse=True)[:8]
    rn_top=sorted(random,key=lambda x:x["fitness"],reverse=True)[:8]
    vals=[]
    for r in ev_top:
        x=validation_record(r,data); x["role"]="evolved"; vals.append(x)
    for r in rn_top:
        x=validation_record(r,data); x["role"]="random"; vals.append(x)
    # simple baseline: negative 5-day reversal
    base=("neg",("feat","ret5")); bst=evaluate_tree(base,train_idx,data)
    bv=validation_record(bst,data); bv["role"]="rev5_baseline"; vals.append(bv)
    best_ev=max([x for x in vals if x["role"]=="evolved"],key=lambda x:x["combined"])
    # sign control and reproducibility
    vi=fc.idx_for(data["panel_dates"],*fc.VALID)
    ctrl=ic_stats(best_ev["tree"],vi,data,force_direction=-best_ev["train"]["direction"])
    rerun=ic_stats(best_ev["tree"],vi,data,force_direction=best_ev["train"]["direction"])
    passed=[x for x in vals if x["gate_pass"]]
    main_rec=max(passed,key=lambda x:x["combined"]) if passed else max(vals,key=lambda x:x["combined"])
    simple_pool=[x for x in passed if x["train"]["nodes"]<=5]
    simple=max(simple_pool,key=lambda x:x["combined"]) if simple_pool else None
    rows=[]
    for x in sorted(vals,key=lambda z:z["combined"],reverse=True):
        tr,va=x["train"],x["valid"]
        rows.append(dict(role=x["role"],formula=x["formula"],nodes=tr["nodes"],
            train_ic=tr["ic"],train_icir=tr["icir_adj"],train_t=tr["t_adj"],train_turn=tr["turnover_top1700"],
            valid_ic=va["ic"],valid_icir=va["icir_adj"],valid_t=va["t_adj"],valid_turn=va["turnover_top1700"],
            retention=x["retention"],combined=x["combined"],gate_pass=x["gate_pass"]))
    pd.DataFrame(rows).to_csv(fc.OUT/"formula_validation.csv",index=False,encoding="utf-8-sig")
    rb=max([x for x in vals if x["role"]=="random"],key=lambda x:x["combined"])
    freeze=dict(created_at=pd.Timestamp.now().isoformat(),seed=20261030,train=list(fc.TRAIN),validation=list(fc.VALID),oos_untouched=True,
        n_features=len(data["features"]),n_evolved=len(evolved),n_random=len(random),n_validation_looks=len(vals),
        selection=dict(main=dict(role=main_rec["role"],tree=main_rec["tree"],formula=main_rec["formula"],train=main_rec["train"],valid=main_rec["valid"],combined=main_rec["combined"]),
            simple=dict(role=simple["role"],tree=simple["tree"],formula=simple["formula"],train=simple["train"],valid=simple["valid"],combined=simple["combined"]) if simple else None,
            random_best=dict(role="random",tree=rb["tree"],formula=rb["formula"],train=rb["train"],valid=rb["valid"],combined=rb["combined"]),
            rev5_baseline=dict(tree=base,formula=fc.expr(base),train=bv["train"],valid=bv["valid"])),
        control=dict(signflip_valid_ic=ctrl["ic"],signflip_valid_icir=ctrl["icir_adj"],rerun_equal=bool(abs(rerun["icir_adj"]-best_ev["valid"]["icir_adj"])<1e-12)),
        data=dict(br_sha256=sha256_file(fc.CACHE/"br.npy",limit=8*1024*1024),br_bytes=(fc.CACHE/"br.npy").stat().st_size,
            y_sha256=sha256_file(fc.ROOT/"out/cache/evo/y20.npy"),
            universe_sha256=sha256_file(fc.ROOT/"out/cache/evo/universe.npy")),
        limits=["primitive bank is fixed; the GP evolves combinations, not new rolling operators",
                "feature/z-score preprocessing is causal but uses hand-designed windows",
                "validation shortlist is one look; GP itself used 40+ candidates per generation",
                "OOS 21 months cannot support precise annualized claims"])
    fc.OUT.joinpath("freeze.json").write_text(json.dumps(freeze,ensure_ascii=False,indent=2,default=float),encoding="utf-8")
    top=pd.DataFrame(rows).head(12)
    def md(df):
        return "\n".join(["| "+" | ".join(df.columns)+" |","| "+" | ".join(["---"]*len(df.columns))+" |"]+
                         ["| "+" | ".join(str(v) for v in row)+" |" for row in df.itertuples(index=False)])
    txt=f"""# 公式级自进化实验（Class4_回测）

- 训练：{fc.TRAIN[0]} ~ {fc.TRAIN[1]}（{len(train_idx)} 个月）
- 验证：{fc.VALID[0]} ~ {fc.VALID[1]}（单次短名单）
- 2025-2026：**freeze 后才可读取**
- 基元：{len(data['features'])} 个因果时序特征；公式树最大 {fc.MAX_NODES} 节点、深度 {fc.MAX_DEPTH}
- 进化候选：{len(evolved)}；匹配随机公式：{len(random)}
- 验证短名单：{len(vals)}；符号翻转控制：valid ICIR={ctrl['icir_adj']:.3f}

## 验证结果

{md(top)}

## 冻结选择

- 主公式：`{main_rec['formula']}`
- 主公式 valid ICIR={main_rec['valid']['icir_adj']:.3f}，retention={main_rec['retention']:.3f}
- 随机最佳 valid ICIR={rb['valid']['icir_adj']:.3f}
- 符号翻转控制：valid IC={ctrl['ic']:.4f}
- 复现：{rerun['icir_adj']:.6f} vs {best_ev['valid']['icir_adj']:.6f}

## 说明

公式层只允许对**预计算的因果基元**做横截面组合；因此这比“直接生成带未来函数的任意公式”安全，也不是完整的程序合成。看到 OOS 后不得改选。
"""
    fc.OUT.joinpath("train_formula_report.md").write_text(txt,encoding="utf-8")
    print(txt.encode("gbk",errors="replace").decode("gbk")); print("elapsed",time.time()-t0)

if __name__=="__main__": main()

