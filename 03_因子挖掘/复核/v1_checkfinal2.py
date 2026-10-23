# -*- coding: utf-8 -*-
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))          # <项目根>/03_因子挖掘/复核
ROOT = os.path.dirname(os.path.dirname(HERE))            # 项目根（不写死本机路径）
sys.path.insert(0, ROOT)
from core.lib import MarketData
RUN = os.path.join(ROOT,'out','03_因子挖掘','runs','23_raw','MF1700_raw')
h = pd.read_csv(os.path.join(RUN,'holdings.csv'), parse_dates=['date'])
resid = h[h.date=='2026-09-30'].copy()
tr = pd.read_csv(os.path.join(RUN,'trades.csv'), parse_dates=['date'])
bl = pd.read_csv(os.path.join(RUN,'blocks.csv'), parse_dates=['date'])
last_tr = set(tr.loc[tr.date=='2026-09-30','symbol']); last_bl = set(bl.loc[bl.date=='2026-09-30','symbol'])
md = MarketData(verbose=False); i = md.dates.get_loc(pd.Timestamp('2026-09-30'))
print('%-11s %12s %12s %8s %8s' % ('symbol','value','adj_close','sold?','blocked?'))
for _, r in resid.iterrows():
    s = r['symbol']; ac = md.adj_close.values[i, md.symbols.get_loc(s)]
    print('%-11s %12s %12s %8s %8s' % (s, r['value'], ac, s in last_tr, s in last_bl))
print('\n末日 trades：%d 笔；blocks：%d 笔' % (len(last_tr), len(last_bl)))
print('9/29 持仓数=%d，末日卖出尝试=%d，剩余=%d' % (1711, len(last_tr)+len(last_bl), 1711-len(last_tr)-len(last_bl)))
