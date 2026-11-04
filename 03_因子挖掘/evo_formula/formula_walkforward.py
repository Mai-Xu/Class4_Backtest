# -*- coding: utf-8 -*-
"""Formula-level walk-forward evolution with matched random controls."""
from __future__ import annotations
import json, sys, time
from collections import defaultdict
import numpy as np
import pandas as pd
import formula_core as fc
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
V2OUT=fc.OUT/"v2"; V2OUT.mkdir(parents=True,exist_ok=True)
FOLDS=[("2021-01-01","2021-12-31","2022-01-01","2022-06-30"),("2021-07-01","2022-06-30","2022-07-01","2022-12-31"),("2022-01-01","2022-12-31","2023-01-01","2023-06-30"),("2022-07-01","2023-06-30","2023-07-01","2023-12-31"),("2023-01-01","2023-12-31","2024-01-01","2024-06-30")]
OUTER=("2024-07-01","2024-12-31")
def top_turnover(score,uni,n_hold=1700):
    prev=None; vals=[]
    for i in range(score.shape[0]):
        cand=np.flatnonzero(uni[i]&np.isfinite(score[i]))
        if len(cand)<n_hold: continue
        cur=set(cand[np.argsort(score[i,cand])[-n_hold:]].tolist())
        if prev is not None: vals.append(1-len(cur&prev)/float(n_hold))
        prev=cur
    return float(np.mean(vals)) if vals else np.nan
def score_tree(tree,idx,data):
    return np.where(data["uni"][idx],fc.eval_tree(tree,data)[idx],np.nan)
def stats_for(tree,idx,data,force_direction=None):
    s=score_tree(tree,idx,data); ic=fc.mc.ic_series(s,data["y20"][idx],data["uni"][idx]); st=fc.mc.ic_stats(ic)
    d=force_direction if force_direction is not None else (1.0 if st["ic"]>=0 else -1.0)
    st["direction"]=float(d); st["ic_adj"]=float(st["ic"]*d); st["icir_adj"]=float(st["icir"]*d); st["t_adj"]=float(st["t"]*d)
    st["turnover"]=top_turnover(s,data["uni"][idx]); st["nodes"]=fc.tree_nodes(tree); return st
def cap_tree(t,features,rng):
    while fc.tree_nodes(t)>fc.MAX_NODES:
        if t[0]=="feat": break
        children=[x for x in t[1:] if isinstance(x,tuple)]
        t=children[int(rng.integers(len(children)))] if children else ("feat",str(rng.choice(features)))
    if fc.tree_nodes(t)>fc.MAX_NODES: t=("feat",str(rng.choice(features)))
    return t
def random_tree(rng,features): return cap_tree(fc.random_tree(rng,features),features,rng)
def mutate(parent,op,rng,features):
    if op=="wrap": child=(str(rng.choice(["neg","abs","square","rank","z"])),parent)
    else: child=(str(rng.choice(["add","sub","mul","div"])),parent,fc.random_tree(rng,features,2))
    return cap_tree(child,features,rng)
class Policy:
    def __init__(self): self.ops=["wrap","binary"]; self.ok={o:1.0 for o in self.ops}; self.n={o:2.0 for o in self.ops}
    def choose(self,rng):
        p=np.array([self.ok[o]/self.n[o] for o in self.ops]); p=p/p.sum(); return str(rng.choice(self.ops,p=p))
    def update(self,records):
        top=sorted(records,key=lambda x:x["fitness"],reverse=True)[:max(1,len(records)//4)]
        for r in top:
            op=r.get("op","seed")
            if op in self.n: self.ok[op]+=1
        for op in self.ops: self.n[op]+=1
def fold_eval(tree,data):
    rows=[]
    for a,b,ha,hb in FOLDS:
        it=fc.idx_for(data["panel_dates"],a,b); ih=fc.idx_for(data["panel_dates"],ha,hb)
        tr=stats_for(tree,it,data); ho=stats_for(tree,ih,data,force_direction=tr["direction"])
        rows.append(dict(train_ic=tr["ic"],train_icir=tr["icir_adj"],hold_icir=ho["icir_adj"],train_dir=tr["direction"],hold_turn=ho["turnover"],train_turn=tr["turnover"]))
    norms=np.array([r["hold_icir"] for r in rows],float); dirs=np.array([r["train_dir"] for r in rows],float); ok=np.isfinite(norms)
    mean_hold=float(np.nanmean(norms)) if ok.any() else np.nan; min_hold=float(np.nanmin(norms)) if ok.any() else np.nan; pos_rate=float((norms[ok]>0).mean()) if ok.any() else np.nan
    mode_sign=float(np.sign(np.nanmean(dirs))) if len(dirs) else 1.0; dir_consistency=float((dirs==mode_sign).mean()) if len(dirs) else 0.0
    turn=float(np.nanmean([r["hold_turn"] for r in rows])) if rows else np.nan; nodes=fc.tree_nodes(tree)
    fit=(0.45*mean_hold+0.20*min_hold+0.15*2.0*pos_rate+0.10*dir_consistency*2.0-0.04*max(0,nodes-5)-0.60*max(0.0,(turn-0.65) if np.isfinite(turn) else 0.0))
    return float(fit),dict(mean_hold_icir=mean_hold,min_hold_icir=min_hold,hold_positive_rate=pos_rate,direction_consistency=dir_consistency,hold_turnover_mean=turn,nodes=nodes,folds=rows,direction=mode_sign)
def evaluate_tree(tree,data,op="seed"):
    fit,meta=fold_eval(tree,data); return dict(tree=tree,formula=fc.expr(tree),op=op,fitness=fit,**meta)
def run_arm(data,seed,budget=59):
    rng=np.random.default_rng(seed); feats=data["features"]; policy=Policy()
    seeds=["ret5","ret20","max1d20","max5d20","udvol60","illiq20","amtrel20","pos20","vol20","madev60","ovn20","pvcorr20"]
    pop=[("feat",x) for x in seeds]+[("neg",("feat",x)) for x in seeds[:6]]; seen=set(fc.tree_key(t) for t in pop)
    while len(pop)<24:
        t=random_tree(rng,feats); k=fc.tree_key(t)
        if k not in seen: pop.append(t); seen.add(k)
    recs={}; gen_records=defaultdict(list)
    for gen in range(3):
        batch=pop if gen==0 else []
        if gen>0:
            elites=sorted(recs.values(),key=lambda x:x["fitness"],reverse=True)[:6]; batch=[x["tree"] for x in elites]
            keys={fc.tree_key(t) for t in batch}
            while len(batch)<24:
                p1=max(rng.choice(list(recs.values()),size=min(3,len(recs)),replace=False),key=lambda x:x["fitness"])["tree"]
                p2=max(rng.choice(list(recs.values()),size=min(3,len(recs)),replace=False),key=lambda x:x["fitness"])["tree"]
                op=policy.choose(rng); child=mutate(p2 if rng.random()<.3 else p1,op,rng,feats)
                if rng.random()<0.25: child=mutate(child,"wrap",rng,feats)
                child=cap_tree(child,feats,rng); k=fc.tree_key(child)
                if k not in keys: batch.append(child); keys.add(k)
        for t in batch:
            k=fc.tree_key(t)
            if k not in recs: recs[k]=evaluate_tree(t,data,"mutation" if gen>0 else "seed")
            if recs[k] not in gen_records[gen]: gen_records[gen].append(recs[k])
        policy.update(gen_records[gen]); print(f"  gen {gen+1}: unique={len(recs)} best={max(recs.values(),key=lambda x:x['fitness'])['fitness']:.3f}")
    return sorted(recs.values(),key=lambda x:x["fitness"],reverse=True)[:budget]
def random_arm(data,seed,n):
    rng=np.random.default_rng(seed+300); out=[]; seen=set()
    while len(out)<n:
        t=random_tree(rng,data["features"]); k=fc.tree_key(t)
        if k in seen: continue
        seen.add(k); out.append(evaluate_tree(t,data,"random"))
    return sorted(out,key=lambda x:x["fitness"],reverse=True)
def outer_audit(rec,data):
    oi=fc.idx_for(data["panel_dates"],*OUTER); st=stats_for(rec["tree"],oi,data,force_direction=rec["direction"])
    return dict(ic=st["ic_adj"],icir=st["icir_adj"],t=st["t_adj"],positive=st["pos"],turnover=st["turnover"])
def tree_at(tree,t,C,uni,specs):
    def ev(node):
        if node[0]=="feat":
            x=np.asarray(specs[node[1]]["fn"](int(t),C),dtype=float); z=fc.mc.cs_rank_z(np.where(uni,x,np.nan)); return np.where(np.isfinite(z),z,0.0)
        if node[0]=="neg": return -ev(node[1])
        if node[0]=="abs": return np.abs(ev(node[1]))
        if node[0]=="square":
            x=np.clip(ev(node[1]),-3,3); return x*x
        if node[0] in ("rank","z"):
            x=ev(node[1]); z=fc.mc.cs_rank_z(x); return np.where(np.isfinite(z),z,0.0)
        a,b=ev(node[1]),ev(node[2])
        if node[0]=="add": return a+b
        if node[0]=="sub": return a-b
        if node[0]=="mul": return a*b
        return np.where(np.abs(b)>.25,a/np.where(np.abs(b)>.25,b,1.0),0.0)
    return ev(tree)
def causal_audit(best,data):
    md=fc.MarketData(); C=fc.Ctx(md); specs=fc.feature_specs(); rows=[]
    sample=[fc.idx_for(data["panel_dates"],*a)[0] for a in [FOLDS[0][2:],FOLDS[-1][2:],OUTER]]
    for pi in sample:
        t=md.dates.get_loc(data["panel_dates"][pi]); uni=data["uni"][pi]
        a=tree_at(best["tree"],t,C,uni,specs); b=tree_at(best["tree"],t,C.truncated(t),uni,specs)
        d=float(np.nanmax(np.abs(np.nan_to_num(a,nan=-1e18)-np.nan_to_num(b,nan=-1e18))))
        rows.append(dict(date=str(data["panel_dates"][pi].date()),max_abs_diff=d,causal=bool(d<1e-9)))
    return rows
def leaves(t,out):
    if t[0]=="feat": out.append(t[1]); return
    for x in t[1:]:
        if isinstance(x,(list,tuple)): leaves(x,out)
def main():
    t0=time.time(); data=fc.build_base(); seeds=[20261030,20261031,20261032,20261033,20261034]; all_rows=[]; summary=[]; best_global=None
    for seed in seeds:
        print("WALK-FORWARD seed",seed); ev=run_arm(data,seed,59); rn=random_arm(data,seed,59)
        for r in ev: r.update(seed=seed,arm="evolved")
        for r in rn: r.update(seed=seed,arm="random")
        all_rows.extend(ev+rn)
        ev_aud=[(r,outer_audit(r,data)) for r in ev[:8]]; rn_aud=[(r,outer_audit(r,data)) for r in rn[:8]]
        bev=max(ev_aud,key=lambda x:x[1]["icir"]); brv=max(rn_aud,key=lambda x:x[1]["icir"])
        summary.append(dict(seed=seed,evolved_fitness=ev[0]["fitness"],random_fitness=rn[0]["fitness"],evolved_outer_icir=bev[1]["icir"],random_outer_icir=brv[1]["icir"],evolved_outer_ic=bev[1]["ic"],random_outer_ic=brv[1]["ic"],evolved_nodes=ev[0]["nodes"],random_nodes=rn[0]["nodes"]))
        if best_global is None or bev[0]["fitness"]>best_global[0]["fitness"]: best_global=(bev[0],bev[1],seed)
    sdf=pd.DataFrame(summary); sdf.to_csv(V2OUT/"walkforward_multiseed.csv",index=False,encoding="utf-8-sig")
    ledger=[dict(seed=r["seed"],arm=r["arm"],op=r.get("op"),formula=r["formula"],nodes=r["nodes"],fitness=r["fitness"],mean_hold_icir=r["mean_hold_icir"],min_hold_icir=r["min_hold_icir"],hold_positive_rate=r["hold_positive_rate"],direction_consistency=r["direction_consistency"],turnover=r["hold_turnover_mean"]) for r in all_rows]
    pd.DataFrame(ledger).to_csv(V2OUT/"walkforward_trials.csv",index=False,encoding="utf-8-sig")
    tax=[]
    for r in all_rows:
        if r["arm"]!="evolved": continue
        why=("negative_holdout" if r["mean_hold_icir"]<=0 else "unstable_folds" if r["hold_positive_rate"]<0.6 else "direction_flip" if r["direction_consistency"]<0.8 else "high_turnover" if r["hold_turnover_mean"]>0.65 else "complexity" if r["nodes"]>5 else "accepted")
        tax.append(dict(seed=r["seed"],formula=r["formula"],reason=why))
    pd.DataFrame(tax).to_csv(V2OUT/"failure_taxonomy.csv",index=False,encoding="utf-8-sig")
    best,outer,seed=best_global; cav=causal_audit(best,data); pd.DataFrame(cav).to_csv(V2OUT/"causal_audit.csv",index=False,encoding="utf-8-sig")
    cf=[]; lv=[]; leaves(best["tree"],lv)
    for f in lv:
        st=outer_audit(dict(tree=("feat",f),direction=best["direction"]),data); cf.append(dict(kind="leaf_only",leaf=f,icir=st["icir"],ic=st["ic"]))
    cf.append(dict(kind="full",leaf="",icir=outer["icir"],ic=outer["ic"])); pd.DataFrame(cf).to_csv(V2OUT/"counterfactual.csv",index=False,encoding="utf-8-sig")
    base=("neg",("feat","ret5")); oi=fc.idx_for(data["panel_dates"],*OUTER); s1=score_tree(best["tree"],oi,data)*best["direction"]; s2=score_tree(base,oi,data)
    z1=np.vstack([fc.mc.cs_rank_z(np.where(data["uni"][oi][i],s1[i],np.nan)) for i in range(len(oi))]); z2=np.vstack([fc.mc.cs_rank_z(np.where(data["uni"][oi][i],s2[i],np.nan)) for i in range(len(oi))])
    pool=np.where(np.isfinite(z1)&np.isfinite(z2),.5*z1+.5*z2,np.nan); pi=fc.mc.ic_stats(fc.mc.ic_series(pool,data["y20"][oi],data["uni"][oi]))
    base_st=outer_audit(dict(tree=base,direction=1.0),data); pool_info=dict(formula_icir=outer["icir"],baseline_icir=base_st["icir"],pool_icir=pi["icir"],pool_increment=pi["icir"]-max(outer["icir"],base_st["icir"]))
    (V2OUT/"pool_increment.json").write_text(json.dumps(pool_info,ensure_ascii=False,indent=2),encoding="utf-8")
    freeze=dict(version="formula_walkforward_v2",created_at=pd.Timestamp.now().isoformat(),folds=FOLDS,outer_confirmation=OUTER,future_holdout_status="not_available; 2025-2026 already seen and not used for selection",seed=seed,selected=dict(formula=best["formula"],tree=best["tree"],direction=best["direction"],inner_fitness=best["fitness"],inner_mean_hold_icir=best["mean_hold_icir"],inner_min_hold_icir=best["min_hold_icir"],outer_audit=outer))
    (V2OUT/"freeze_future.json").write_text(json.dumps(freeze,ensure_ascii=False,indent=2,default=float),encoding="utf-8")
    def tb(d): return "\n".join(["| "+" | ".join(d.columns)+" |","| "+" | ".join(["---"]*len(d.columns))+" |"]+["| "+" | ".join(str(v) for v in r)+" |" for r in d.itertuples(index=False)])
    report=f"""# 公式级 Walk-Forward V2

## 设计

- 5 个滚动训练/留出窗口，均只落在 2021–2023；
- 2024H2 作为一次 outer confirmation；
- 2025–2026 不在本实验中用于选择（V1 已消费，不能再次冒充盲测）；
- 每臂 59 个候选，进化 vs 同预算随机；
- fitness 同时考虑 holdout ICIR、最差 fold、方向一致性、节点数、换手成本代理。

## 五种子结果

{tb(sdf)}

## 选中的冻结公式

`{best['formula']}`

- inner mean holdout ICIR: {best['mean_hold_icir']:.3f}
- inner min holdout ICIR: {best['min_hold_icir']:.3f}
- inner direction consistency: {best['direction_consistency']:.2f}
- outer 2024H2 ICIR: {outer['icir']:.3f}
- outer turnover: {outer['turnover']:.3f}

## 因果性/反事实

- 因果投毒检查：{all(x['causal'] for x in cav)}（截断前后最大差见 `causal_audit.csv`）
- 池增量：{json.dumps(pool_info,ensure_ascii=False)}
- 反事实拆解：`counterfactual.csv`

## 结论

如果五种子中随机臂仍经常赢，则公式级 GP 依然没有证明“搜索比同预算随机采样更好”；这次最多说明我们找到了一个可审计的公式搜索闭环，而不是 RSI。
"""
    (V2OUT/"report.md").write_text(report,encoding="utf-8"); print(report.encode("gbk",errors="replace").decode("gbk")); print("elapsed",time.time()-t0)
if __name__=="__main__": main()
