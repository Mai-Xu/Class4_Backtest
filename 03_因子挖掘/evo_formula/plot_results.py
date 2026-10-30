# -*- coding: utf-8 -*-
"""Charts for the formula-evolution issue reply."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"out/03_因子挖掘/evo_formula"
FIG=OUT/"figures"; FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei","DejaVu Sans"]
plt.rcParams["axes.unicode_minus"]=False
plt.rcParams["figure.facecolor"]="white"
plt.rcParams["axes.facecolor"]="white"

fz=json.loads((OUT/"freeze.json").read_text(encoding="utf-8"))
vm=pd.read_csv(OUT/"oos_formula_metrics.csv")
val=pd.read_csv(OUT/"formula_validation.csv")
nav=pd.read_csv(OUT/"oos_formula_nav.csv",parse_dates=["date"])

# 1) ICIR path: train -> validation -> OOS
roles=[("进化公式","evolved_formula",fz["selection"]["main"]),
       ("随机公式","random_formula",fz["selection"]["random_best"]),
       ("neg(ret5)","rev5_baseline",fz["selection"]["rev5_baseline"])]
labels=[]; train=[]; valid=[]; oos=[]
for lab,key,rec in roles:
    labels.append(lab); train.append(rec["train"]["icir_adj"]); valid.append(rec["valid"]["icir_adj"])
    oos.append(float(vm.loc[vm.role==key,"oos_icir_adj"].iloc[0]))
x=np.arange(len(labels)); w=0.24
fig,ax=plt.subplots(figsize=(8.2,4.4))
b1=ax.bar(x-w,train,w,label="Train 2021–22",color="#5B8FF9")
b2=ax.bar(x,valid,w,label="Validation 2023–24",color="#61DDAA")
b3=ax.bar(x+w,oos,w,label="OOS 2025–26",color="#F4664A")
for bars in (b1,b2,b3):
    for b in bars:
        h=b.get_height(); ax.annotate(f"{h:.2f}",(b.get_x()+b.get_width()/2,h),ha="center",va="bottom" if h>=0 else "top",fontsize=9,xytext=(0,3 if h>=0 else -3),textcoords="offset points")
ax.axhline(0,color="#333",lw=.8)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("Direction-adjusted ICIR")
ax.set_title("公式级进化：验证段漂亮，OOS 方向翻转",fontsize=14,fontweight="bold")
ax.legend(frameon=False,ncol=3,loc="upper left"); ax.grid(axis="y",alpha=.2); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig(FIG/"formula_icir_path.png",dpi=180); plt.close(fig)

# 2) OOS NAV
fig,ax=plt.subplots(figsize=(9,4.6))
series=[("evolved_formula","进化公式","#F4664A",2.4),("market_ew","全市场等权","#8C8C8C",1.3),("mf1700","MF1700","#5B8FF9",1.3),("ml1700","ML1700","#61DDAA",1.3)]
for col,lab,color,lw in series:
    ax.plot(nav["date"],nav[col],label=lab,color=color,lw=lw)
    ax.annotate(f"{nav[col].iloc[-1]:.2f}",(nav["date"].iloc[-1],nav[col].iloc[-1]),xytext=(5,0),textcoords="offset points",va="center",fontsize=9,color=color)
ax.set_title("OOS 净值：公式跑得不错，但不要忽略 IC 方向已经反了",fontsize=14,fontweight="bold")
ax.set_ylabel("NAV"); ax.grid(axis="y",alpha=.2); ax.legend(frameon=False,ncol=4,loc="upper left"); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig(FIG/"formula_oos_nav.png",dpi=180); plt.close(fig)

# 3) train vs validation shortlist
fig,ax=plt.subplots(figsize=(7.2,5.0))
colors={"evolved":"#F4664A","random":"#5B8FF9","rev5_baseline":"#8C8C8C"}
for role,g in val.groupby("role"):
    ax.scatter(g["train_icir"],g["valid_icir"],s=35+g["nodes"]*8,alpha=.75,label={"evolved":"Evolved GP","random":"Random control","rev5_baseline":"neg(ret5)"}[role],color=colors.get(role,"#999"))
main=val.iloc[0]; ax.annotate("main: 3-node\n(amtlog60 + max5d20)",(main.train_icir,main.valid_icir),xytext=(12,8),textcoords="offset points",fontsize=8,arrowprops=dict(arrowstyle="->",color="#666",lw=.8))
ax.set_xlabel("Train ICIR (direction-adjusted)"); ax.set_ylabel("Validation ICIR (direction-adjusted)")
ax.set_title("验证短名单：随机公式也很强",fontsize=14,fontweight="bold")
ax.grid(alpha=.2); ax.legend(frameon=False); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig(FIG/"formula_validation_scatter.png",dpi=180); plt.close(fig)

# 4) balanced 2x2 issue summary
fig=plt.figure(figsize=(14,10))
gs=fig.add_gridspec(2,2,height_ratios=[1,1],width_ratios=[1,1.05])
ax1=fig.add_subplot(gs[0,0]); ax2=fig.add_subplot(gs[0,1]); ax3=fig.add_subplot(gs[1,0]); ax4=fig.add_subplot(gs[1,1])
ax1.bar(x-w,train,w,label="Train",color="#5B8FF9"); ax1.bar(x,valid,w,label="Valid",color="#61DDAA"); ax1.bar(x+w,oos,w,label="OOS",color="#F4664A")
ax1.axhline(0,color="#333",lw=.8); ax1.set_xticks(x); ax1.set_xticklabels(["Evolved","Random","neg(ret5)"],fontsize=9)
ax1.set_title("ICIR: validation vs OOS",fontsize=12,fontweight="bold"); ax1.grid(axis="y",alpha=.2); ax1.legend(frameon=False,fontsize=8); ax1.spines[["top","right"]].set_visible(False)
for col,lab,color,lw in series: ax2.plot(nav["date"],nav[col],label=lab,color=color,lw=lw)
ax2.set_title("OOS NAV",fontsize=12,fontweight="bold"); ax2.grid(axis="y",alpha=.2); ax2.legend(frameon=False,fontsize=8); ax2.spines[["top","right"]].set_visible(False)
for role,g in val.groupby("role"):
    ax3.scatter(g["train_icir"],g["valid_icir"],s=30+g["nodes"]*7,alpha=.75,label={"evolved":"Evolved GP","random":"Random control","rev5_baseline":"neg(ret5)"}[role],color=colors.get(role,"#999"))
ax3.set_title("Validation shortlist",fontsize=12,fontweight="bold"); ax3.set_xlabel("Train ICIR"); ax3.set_ylabel("Valid ICIR"); ax3.grid(alpha=.2); ax3.legend(frameon=False,fontsize=8); ax3.spines[["top","right"]].set_visible(False)
ax4.axis("off")
ax4.text(.02,.88,"主公式  -(amtlog60 + max5d20)",fontsize=17,fontweight="bold",color="#1f2937")
ax4.text(.02,.66,"Train ICIR  4.52",fontsize=13,color="#334155")
ax4.text(.02,.52,"Valid ICIR  2.60",fontsize=13,color="#334155")
ax4.text(.02,.38,"OOS ICIR  -2.11  ← 方向翻转",fontsize=13,color="#F4664A",fontweight="bold")
ax4.text(.02,.20,"同预算随机公式：Valid 2.44 / OOS -2.25",fontsize=11,color="#5B8FF9")
ax4.text(.02,.05,"结论：能挖到验证段信号，但没通过 OOS 方向稳定性；\n随机公式同样强，真正的进步来自 harness 和滚动验证。",fontsize=10.5,color="#475569")
fig.suptitle("Class4 公式级自进化：3 个节点，一个小惊喜，一个大跟头",fontsize=17,fontweight="bold",y=.98)
fig.tight_layout(rect=[0,0,1,.96]); fig.savefig(FIG/"formula_evolution_summary.png",dpi=180); plt.close(fig)
print("saved",FIG)
