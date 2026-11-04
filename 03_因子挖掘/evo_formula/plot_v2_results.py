# -*- coding: utf-8 -*-
"""Charts for formula Walk-Forward V2."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]; V2=ROOT/"out/03_因子挖掘/evo_formula/v2"; FIG=V2/"figures"; FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei","DejaVu Sans"]; plt.rcParams["axes.unicode_minus"]=False
s=pd.read_csv(V2/"walkforward_multiseed.csv"); cf=pd.read_csv(V2/"counterfactual.csv"); cost=pd.read_csv(V2/"cost_sensitivity.csv"); fz=json.loads((V2/"freeze_future.json").read_text(encoding="utf-8"))
# 1 seed comparison
fig,ax=plt.subplots(figsize=(9,4.4)); x=np.arange(len(s)); w=.36
ax.bar(x-w/2,s.evolved_outer_icir,w,label="Evolved",color="#F4664A"); ax.bar(x+w/2,s.random_outer_icir,w,label="Random",color="#5B8FF9")
for i,(a,b) in enumerate(zip(s.evolved_outer_icir,s.random_outer_icir)):
    ax.text(i-w/2,a+.08,f"{a:.2f}",ha="center",fontsize=8); ax.text(i+w/2,b+.08,f"{b:.2f}",ha="center",fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([str(v) for v in s.seed],rotation=0); ax.set_ylabel("Outer 2024H2 ICIR"); ax.set_title("Walk-forward V2：随机对照 4 胜 1 平",fontsize=14,fontweight="bold"); ax.legend(frameon=False); ax.grid(axis="y",alpha=.2); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig(FIG/"v2_seed_compare.png",dpi=180); plt.close(fig)
# 2 counterfactual
fig,ax=plt.subplots(figsize=(8,4.2)); labels=[("Full" if k=="full" else str(v)) for k,v in zip(cf.kind,cf.leaf)]; vals=cf.icir; colors=["#F4664A" if k=="full" else "#5B8FF9" for k in cf.kind]
bars=ax.bar(labels,vals,color=colors)
for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+(.1 if v>=0 else -.1),f"{v:.2f}",ha="center",va="bottom" if v>=0 else "top",fontsize=9)
ax.axhline(0,color="#333",lw=.8); ax.set_ylabel("Outer 2024H2 ICIR"); ax.set_title("反事实：完整公式反而不如单个 intra20",fontsize=14,fontweight="bold"); ax.grid(axis="y",alpha=.2); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig(FIG/"v2_counterfactual.png",dpi=180); plt.close(fig)
# 3 cost
fig,ax=plt.subplots(figsize=(8,4.2)); ax.plot(cost.slippage*10000,cost.cagr*100,marker="o",label="CAGR",color="#F4664A"); ax2=ax.twinx(); ax2.plot(cost.slippage*10000,cost.sharpe,marker="s",label="Sharpe",color="#5B8FF9")
ax.set_xlabel("One-way slippage (bp)"); ax.set_ylabel("CAGR (%)"); ax2.set_ylabel("Sharpe"); ax.set_title("成本审计：高换手让收益快速缩水",fontsize=14,fontweight="bold"); ax.grid(alpha=.2); ax.spines[["top"]].set_visible(False); ax2.spines[["top"]].set_visible(False)
fig.tight_layout(); fig.savefig(FIG/"v2_cost.png",dpi=180); plt.close(fig)
# 4 2x2 summary
fig=plt.figure(figsize=(14,10)); gs=fig.add_gridspec(2,2,height_ratios=[1,1],width_ratios=[1,1.1]); ax1=fig.add_subplot(gs[0,0]); ax2=fig.add_subplot(gs[0,1]); ax3=fig.add_subplot(gs[1,0]); ax4=fig.add_subplot(gs[1,1])
ax1.bar(x-w/2,s.evolved_outer_icir,w,label="Evolved",color="#F4664A"); ax1.bar(x+w/2,s.random_outer_icir,w,label="Random",color="#5B8FF9"); ax1.set_xticks(x); ax1.set_xticklabels(["s1","s2","s3","s4","s5"]); ax1.set_title("Outer ICIR by seed",fontsize=12,fontweight="bold"); ax1.grid(axis="y",alpha=.2); ax1.legend(frameon=False,fontsize=8); ax1.spines[["top","right"]].set_visible(False)
ax2.bar(labels,vals,color=colors); ax2.axhline(0,color="#333",lw=.8); ax2.set_title("Counterfactual leaves",fontsize=12,fontweight="bold"); ax2.grid(axis="y",alpha=.2); ax2.spines[["top","right"]].set_visible(False)
ax3.plot(cost.slippage*10000,cost.cagr*100,marker="o",color="#F4664A"); ax3.set_xlabel("Slippage (bp)"); ax3.set_ylabel("CAGR (%)"); ax3.set_title("Cost sensitivity (seen OOS)",fontsize=12,fontweight="bold"); ax3.grid(alpha=.2); ax3.spines[["top","right"]].set_visible(False)
ax4.axis("off"); ax4.text(.02,.88,"V2 Walk-Forward 公式",fontsize=18,fontweight="bold",color="#1f2937"); ax4.text(.02,.70,fz["selected"]["formula"],fontsize=14,color="#F4664A"); ax4.text(.02,.52,"Inner mean holdout ICIR  4.14",fontsize=12,color="#334155"); ax4.text(.02,.40,"Outer 2024H2 ICIR  2.34",fontsize=12,color="#334155"); ax4.text(.02,.28,"随机对照 4 胜 1 平；intra20 单叶更强",fontsize=11,color="#5B8FF9"); ax4.text(.02,.10,"结论：公式闭环成立，但没有证明优于随机；成本与方向稳定性仍是硬约束。",fontsize=10.5,color="#475569")
fig.suptitle("Class4 公式级自进化 V2：从“能进化”到“知道哪里不行”",fontsize=17,fontweight="bold",y=.98); fig.tight_layout(rect=[0,0,1,.96]); fig.savefig(FIG/"v2_summary.png",dpi=180); plt.close(fig)
print("saved",FIG)

