# -*- coding: utf-8 -*-
"""One-shot OOS test for the evolved formula + 04 fastsim backtest."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import formula_core as fc

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0,str(fc.ROOT/"04_策略选股器")); import fastsim


def score_formula(tree, idx, data, direction):
    s=fc.eval_tree(tree,data)[idx]*float(direction)
    return np.where(data["uni"][idx],s,np.nan)


def oos_stats(tree,direction,data):
    idx=fc.idx_for(data["panel_dates"],*fc.OOS)
    s=score_formula(tree,idx,data,direction)
    ic=fc.mc.ic_series(s,data["y20"][idx],data["uni"][idx]); st=fc.mc.ic_stats(ic)
    st["ic_adj"]=st["ic"]*direction; st["icir_adj"]=st["icir"]*direction; st["t_adj"]=st["t"]*direction
    st["turnover_top1700"]=__import__('train_formula',fromlist=['top_turnover']).top_turnover(s,data["uni"][idx]) if False else np.nan
    # Local turnover to avoid importing the training module's side effects.
    prev=None; vals=[]
    for i in range(s.shape[0]):
        cand=np.flatnonzero(data["uni"][idx][i]&np.isfinite(s[i]))
        if len(cand)<1700: continue
        cur=set(cand[np.argsort(s[i,cand])[-1700:]].tolist())
        if prev is not None: vals.append(1-len(cur&prev)/1700.0)
        prev=cur
    st["turnover_top1700"]=float(np.mean(vals)) if vals else np.nan
    return st,s,idx


def fastsim_run(tree,direction,data):
    meta=json.loads((fc.ROOT/"out/cache/selector/meta.json").read_text(encoding="utf-8"))
    sig=[pd.Timestamp(d) for d in meta["signal_dates"]]; si=np.array([data["panel_dates"].get_loc(d) for d in sig],dtype=int)
    scores=score_formula(tree,si,data,direction); uni=data["uni"][si]
    plan,_=fastsim.build_plan(scores,meta["signal_dates"],meta["trade_dates"],uni,n_hold=meta["is_n_hold"],buffer=1.0,liquidate_signal=meta["liq_signal"])
    px=np.load(fc.ROOT/"out/cache/selector/prices_oos.npz")
    sim=fastsim.simulate(plan,px["adj_open"],px["adj_close"],n_hold=meta["is_n_hold"])
    nav=fastsim.nav_ratio(sim["nav"]); st=fastsim.perf_stats(nav); st["年化换手"]=fastsim.turnover_ann(sim["turnover"])
    return st,nav


def leaves(t,out):
    if t[0]=="feat": out.add(t[1]); return
    for x in t[1:]:
        if isinstance(x,(list,tuple)): leaves(x,out)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--force",action="store_true"); a=ap.parse_args()
    fp=fc.OUT/"freeze.json"
    if not fp.exists(): raise SystemExit("freeze.json 不存在，禁止看 OOS")
    rp=fc.OUT/"oos_formula_report.md"
    if rp.exists() and not a.force: raise SystemExit("OOS 已消费；默认拒绝重复揭盲")
    t0=time.time(); fz=json.loads(fp.read_text(encoding="utf-8")); data=fc.build_base()
    roles={"evolved_formula":fz["selection"]["main"],"random_formula":fz["selection"]["random_best"],"rev5_baseline":fz["selection"]["rev5_baseline"]}
    rows=[]; navs={}
    for role,rec in roles.items():
        tree=rec["tree"]; direction=rec["train"]["direction"] if "direction" in rec["train"] else rec.get("direction",1.0)
        st,s,idx=oos_stats(tree,direction,data); fs,nav=fastsim_run(tree,direction,data); navs[role]=nav
        rows.append(dict(role=role,formula=rec["formula"],nodes=rec["train"]["nodes"],oos_ic_adj=st["ic_adj"],oos_icir_adj=st["icir_adj"],oos_t_adj=st["t_adj"],oos_pos=st["pos"],oos_turn=st["turnover_top1700"],cagr=fs["年化收益"],sharpe=fs["夏普"],mdd=fs["最大回撤"],ann_turnover=fs["年化换手"]))
    table=pd.DataFrame(rows); table.to_csv(fc.OUT/"oos_formula_metrics.csv",index=False,encoding="utf-8-sig")
    bench=json.loads((fc.ROOT/"out/cache/selector/benchmarks.json").read_text(encoding="utf-8"))
    main_nav=navs["evolved_formula"]; market=np.asarray(bench["market_ew"]["nav"]); mf=np.asarray(bench["mf1700"]["nav"]); ml=np.asarray(bench["ml1700"]["nav"])
    pd.DataFrame({"date":bench["dates"],"evolved_formula":main_nav[1:],"market_ew":market[1:],"mf1700":mf[1:],"ml1700":ml[1:]}).to_csv(fc.OUT/"oos_formula_nav.csv",index=False,encoding="utf-8-sig")
    ls=set(); leaves(fz["selection"]["main"]["tree"],ls)
    desc="\n".join(f"- `{x}` — {data['desc'].get(x,'')}" for x in sorted(ls))
    def tbl(df): return "\n".join(["| "+" | ".join(df.columns)+" |","| "+" | ".join(["---"]*len(df.columns))+" |"]+["| "+" | ".join(str(v) for v in row)+" |" for row in df.itertuples(index=False)])
    txt=f"""# 公式级进化：OOS 一次性检验

- 冻结时间：{fz['created_at']}
- OOS：{fc.OOS[0]} ~ {fc.OOS[1]}；只揭盲一次，不回改公式
- 主公式：`{fz['selection']['main']['formula']}`
- 方向：{fz['selection']['main']['train']['direction']:+.0f}（按训练段 IC 符号确定）
- 执行：`04_策略选股器/fastsim.py`，10 亿资金、N=1700、月频、万一佣金+万五印花税

## 结果

{tbl(table)}

## 主公式用到的基元

{desc}

## 基准

| 组合 | 年化 | 夏普 | 最大回撤 |
| --- | ---: | ---: | ---: |
| 进化公式 | {fastsim.perf_stats(main_nav)['年化收益']:.2%} | {fastsim.perf_stats(main_nav)['夏普']:.2f} | {fastsim.perf_stats(main_nav)['最大回撤']:.2%} |
| 全市场等权 | {fastsim.perf_stats(market)['年化收益']:.2%} | {fastsim.perf_stats(market)['夏普']:.2f} | {fastsim.perf_stats(market)['最大回撤']:.2%} |
| MF1700 | {fastsim.perf_stats(mf)['年化收益']:.2%} | {fastsim.perf_stats(mf)['夏普']:.2f} | {fastsim.perf_stats(mf)['最大回撤']:.2%} |
| ML1700 | {fastsim.perf_stats(ml)['年化收益']:.2%} | {fastsim.perf_stats(ml)['夏普']:.2f} | {fastsim.perf_stats(ml)['最大回撤']:.2%} |

## 诚实结论

- 这是**公式组合层**的一次进化，不是完整的程序合成；时序窗口和基元仍是人工给定。
- 同预算随机公式仍是必须看的对照；如果它接近或超过进化公式，就不能把提升归功于“进化”。
- OOS 仅 21 个月，年化标准误约 ±4pp；且 `fastsim` 不含涨跌停/停牌/参与率逐笔约束。
- 看到本报告后，不允许再在同一 OOS 上重选公式。
"""
    rp.write_text(txt,encoding="utf-8"); print(txt.encode("gbk",errors="replace").decode("gbk")); print("elapsed",time.time()-t0)

if __name__=="__main__": main()

