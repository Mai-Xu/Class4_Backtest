# -*- coding: utf-8 -*-
"""v1 附加校验：参与率被精确触顶的笔数 + 被挡委托结构。"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))          # <项目根>/03_因子挖掘/复核
ROOT = os.path.dirname(os.path.dirname(HERE))            # 项目根（不写死本机路径）
sys.path.insert(0, ROOT)
from core.lib import MarketData
RUN = os.path.join(ROOT, "out", "03_因子挖掘", "runs", "23_raw", "MF1700_raw")
trades = pd.read_csv(os.path.join(RUN, "trades.csv"), parse_dates=["date"])
blocks = pd.read_csv(os.path.join(RUN, "blocks.csv"), parse_dates=["date"])
md = MarketData(verbose=False)
dpos = {d: i for i, d in enumerate(md.dates)}; spos = {s: i for i, s in enumerate(md.symbols)}
part = np.array([t.dv / md.amt.values[dpos[t.date], spos[t.symbol]] for t in trades.itertuples()])
print("成交笔数 %d；触及 10%% 上限（>0.099999）的笔数：%d" % (len(trades), int((part > 0.099999).sum())))
print("\n被挡委托结构（346 笔 / 合计 %.2f 万元）：" % (blocks.dv.sum() / 1e4))
g = blocks.groupby("reason")["dv"].agg(["size", "sum"])
g["sum_wan"] = g["sum"] / 1e4
print(g[["size", "sum_wan"]].sort_values("sum_wan", ascending=False).to_string())
print("\n参与率分位：P50=%.6f P90=%.6f P99=%.6f max=%.10f"
      % (np.quantile(part, 0.5), np.quantile(part, 0.9), np.quantile(part, 0.99), part.max()))
