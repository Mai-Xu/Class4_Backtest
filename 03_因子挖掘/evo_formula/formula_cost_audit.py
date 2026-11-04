# -*- coding: utf-8 -*-
"""Cost sensitivity for the V2 formula. 2025-2026 is already seen."""
from __future__ import annotations
import json, sys
import numpy as np
import pandas as pd
import formula_core as fc
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0,str(fc.ROOT/"04_策略选股器")); import fastsim

def main():
    v2=fc.OUT/"v2"; fz=json.loads((v2/"freeze_future.json").read_text(encoding="utf-8")); rec=fz["selected"]
    data=fc.build_base(); meta=json.loads((fc.ROOT/"out/cache/selector/meta.json").read_text(encoding="utf-8"))
    sig=[pd.Timestamp(d) for d in meta["signal_dates"]]; si=np.array([data["panel_dates"].get_loc(d) for d in sig],int)
    score=fc.eval_tree(rec["tree"],data)[si]*float(rec["direction"]); uni=data["uni"][si]
    plan,_=fastsim.build_plan(score,meta["signal_dates"],meta["trade_dates"],uni,n_hold=meta["is_n_hold"],buffer=1.0,liquidate_signal=meta["liq_signal"])
    px=np.load(fc.ROOT/"out/cache/selector/prices_oos.npz"); rows=[]
    for slip in [0.0,0.001,0.002,0.003]:
        sim=fastsim.simulate(plan,px["adj_open"],px["adj_close"],n_hold=meta["is_n_hold"],fee_buy=fastsim.FEE_BUY+slip,fee_sell=fastsim.FEE_SELL+slip)
        nav=fastsim.nav_ratio(sim["nav"]); st=fastsim.perf_stats(nav)
        rows.append(dict(slippage=slip,cagr=st["年化收益"],sharpe=st["夏普"],mdd=st["最大回撤"],ann_turnover=fastsim.turnover_ann(sim["turnover"])))
    df=pd.DataFrame(rows); df.to_csv(v2/"cost_sensitivity.csv",index=False,encoding="utf-8-sig")
    table='\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']+['| '+' | '.join(str(v) for v in r)+' |' for r in df.itertuples(index=False)])
    txt=f"""# V2 公式成本敏感性（非盲测）

- 公式：`{rec['formula']}`，方向：{rec['direction']:+.0f}
- 样本：2025-01-02 ~ 2026-09-30，**V1 已经看过，不能当作新盲测**
- 口径：10 亿资金、N=1700、月频，`fastsim.py`，佣金+印花税，分别叠加 0/10/20/30bp 单边滑点
- 选择不看这张表；它只用于确认成本暴露。

## 结果

{table}

结论：如果高换手/滑点显著侵蚀收益，公式应在 fitness 中继续加重换手惩罚；但当前已消费的 OOS 不能再用于重新挑选。
"""
    (v2/"cost_audit.md").write_text(txt,encoding="utf-8"); print(txt.encode("gbk",errors="replace").decode("gbk"))
if __name__=="__main__": main()
