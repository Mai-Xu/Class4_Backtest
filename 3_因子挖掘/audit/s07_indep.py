# -*- coding: utf-8 -*-
"""s07_factor_lab.py 独立复核（复核项 2 / 4 / 5 / 6）。

运行（项目根目录）:
    $env:PYTHONIOENCODING="utf-8"; python 3_因子挖掘/audit/s07_indep.py

本脚本:
  * 不复用 s07 的任何中间变量，因子面板、IC、分层、方向全部从 feathers 重算；
  * 只用 core.lib / core.engine / core.factors / core.strategies 的权威定义；
  * 不修改 s07_factor_lab.py，只读它的源码用于静态定位。
产出:
    audit/s07_indep_out.json   全部关键数字
"""
from __future__ import annotations
import json
import os
import re
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from core.lib import MarketData, TAB, CACHE
from core.engine import Backtest
from core.strategies import PanelStrategy, build_factor_panels
from core.factors import FACTORS, Factors
from core import metrics

S07 = os.path.join(ROOT, "3_因子挖掘", "s07_factor_lab.py")
RUNS = os.path.join(ROOT, "out", "runs")
FUC = os.path.join(CACHE, "factors")
OUT = {}
T00 = time.time()


def sec(t):
    print("\n" + "=" * 92)
    print("## " + t)
    print("=" * 92)


def kv(k, v):
    OUT[k] = v
    print("   %-46s %s" % (k, v))


def fnum(x, nd=6):
    try:
        x = float(x)
    except Exception:
        return str(x)
    return "nan" if not np.isfinite(x) else ("%." + str(nd) + "g") % x


def maxabsdiff(a, b):
    """两个等形数组：有限-有限格的最大绝对差 + 单边有限格计数。"""
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    fa, fb = np.isfinite(a), np.isfinite(b)
    both = fa & fb
    d = np.abs(a[both] - b[both]).max() if both.any() else 0.0
    return float(d), int((fa & ~fb).sum()), int((fb & ~fa).sum()), int(both.sum())


def spearman(x, y, mask):
    """与 s04 / s07 完全相同的横截面 Spearman 定义（独立重写）。"""
    ok = mask & np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 50:
        return np.nan
    a = pd.Series(x[ok]).rank().values
    b = pd.Series(y[ok]).rank().values
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


# --------------------------------------------------------------------------- #
# 0. 载入数据 + s04/s05 产物 + 复现 s07 的 ts 网格
# --------------------------------------------------------------------------- #
sec("0. 载入（MarketData / vv_config.json / 调仓网格）")
md = MarketData(verbose=False)
cfg = json.load(open(os.path.join(RUNS, "vv_config.json"), encoding="utf-8"))
CFG_START, CFG_END, IS_END = cfg["start"], cfg["end"], cfg["is_end"]
FW = 21
ts = sorted(Backtest._rebalance_index(md.dates, cfg["rebal"]))
ts = [t for t in ts if md.dates[t] >= pd.Timestamp(CFG_START)]
dates = md.dates[list(ts)]
adj = md.adj_close                       # s07 用的前瞻收益基准
aopen = md.adj("open")                   # 引擎的实际成交价（复权开盘）
uni_all = md.universe()                  # s07: md.universe()
uni = uni_all.values[list(ts)]
print("   feathers: %d 交易日 x %d 证券, %s ~ %s"
      % (len(md.dates), len(md.symbols), md.dates[0].date(), md.dates[-1].date()))
kv("调仓网格 n_period", len(ts))
kv("调仓网格 首/末", "%s / %s" % (dates[0].date(), dates[-1].date()))
kv("is_end", IS_END)
kv("sel_factors(vv_config)", list(cfg["sel_factors"]))
kv("n_hold / rebal / cash", "%s / %s / %s" % (cfg["n_hold"], cfg["rebal"], cfg["cash"]))

# --------------------------------------------------------------------------- #
# 1. 前瞻收益对齐（s07 的 adj.shift(-h)/adj-1  vs  core.lib.ret_fwd）
# --------------------------------------------------------------------------- #
sec("1. 前瞻收益对齐（复核项 2）")
fwd_s07 = (adj.shift(-FW) / adj - 1.0)                      # s07 的写法（整块面板）
fwd_lib = md.ret_fwd(FW)                                    # core.lib 权威口径
d_full, only_s07, only_lib, n_both = maxabsdiff(fwd_s07.values, fwd_lib.values)
kv("全矩阵 max|s07_fwd - md.ret_fwd(21)|", fnum(d_full, 3))
kv("全矩阵 单边有限格 (s07独有 / lib独有)", "%d / %d（共有 %d 格）" % (only_s07, only_lib, n_both))
d_ts, o1, o2, n1 = maxabsdiff(fwd_s07.values[list(ts)], fwd_lib.values[list(ts)])
kv("调仓网格上 max|差分|", fnum(d_ts, 3))
kv("调仓网格上 单边有限格 (s07独有 / lib独有)", "%d / %d（共有 %d 格）" % (o1, o2, n1))
fwd = fwd_lib.values[list(ts)]                              # 复核统一用权威口径

# --- 独立复算：h 日复利 = prod_{j=1..h}(1 + 日收益_{t+j}) - 1 ---
rday = md.ret.values                                        # ret[t] = t-1 -> t
rng = np.random.default_rng(7)
n_pair, dmax, samples = 400, 0.0, []
_valid_i = [k for k in range(3, len(ts)) if ts[k] + FW < len(md.dates)]
for _ in range(n_pair):
    k = int(rng.integers(0, len(_valid_i)))
    i = _valid_i[k]
    j = int(rng.integers(0, len(md.symbols)))
    t0 = ts[i]
    if not (np.isfinite(adj.values[t0, j]) and np.isfinite(adj.values[t0 + FW, j])):
        continue
    win = rday[t0 + 1: t0 + FW + 1, j]
    if not np.all(np.isfinite(win)):
        continue
    alt = float(np.prod(1.0 + win) - 1.0)
    ref = float(adj.values[t0 + FW, j] / adj.values[t0, j] - 1.0)
    dmax = max(dmax, abs(alt - ref))
    samples.append((t0, j, ref))
kv("抽 %d 组 (股票,调仓日)：max|shift(-h) - 21日复利|" % len(samples), fnum(dmax, 3))

# --- 明确演示：不含 t 当天收益；且与「从 t-1 起算」不同 ---
t0, j, ref = samples[0]
sym = md.symbols[j]
with_prev = float(adj.values[t0 + FW, j] / adj.values[t0 - 1, j] - 1.0)
kv("样例 %s @ %s" % (sym, md.dates[t0].date()),
   "adj[t-1]=%.4f adj[t]=%.4f adj[t+21]=%.4f" % (adj.values[t0 - 1, j], adj.values[t0, j], adj.values[t0 + FW, j]))
kv("样例 fwd = adj[t+21]/adj[t]-1", fnum(ref, 8))
kv("样例 若含当天: adj[t+21]/adj[t-1]-1", fnum(with_prev, 8))
kv("样例 差值（当天收益的贡献）", fnum(with_prev - ref, 8))

# --- 引擎实际成交价口径：open[t+1] -> open[t+1+h] ---
fwd_open = np.full_like(fwd, np.nan)
ao = aopen.values
for i in range(len(ts)):
    t_in = ts[i] + 1
    t_out = t_in + FW
    if t_out < len(md.dates):
        with np.errstate(invalid="ignore", divide="ignore"):
            fwd_open[i] = ao[t_out] / ao[t_in] - 1.0
kv("close-close 与 open[t+1]->open[t+1+21] 均为有限的比例",
   "%.3f" % float((np.isfinite(fwd) & np.isfinite(fwd_open)).sum() / max(np.isfinite(fwd).sum(), 1)))
d_cc_oo, _, _, ncc = maxabsdiff(fwd, fwd_open)
kv("两者 max|差|（同格，仅说明口径不同）", fnum(d_cc_oo, 3))
OUT["_fwd_arrays"] = dict(fwd=fwd, fwd_open=fwd_open)

# --------------------------------------------------------------------------- #
# 2. 因子面板：缓存一致性 + 「删掉未来数据因子值不变」截断测试
# --------------------------------------------------------------------------- #
sec("2. 因子面板：缓存一致性 / 无未来函数截断测试（复核项 2·3）")
raw_c, zsc_c = build_factor_panels(md, FACTORS, ts, cache_dir=FUC)   # s07 实际吃的面板
F = Factors(md)
t_fac = time.time()
raw_f = {}
for nm in FACTORS:
    arr = np.vstack([np.asarray(F.get(nm, t), dtype="float64") for t in ts])
    raw_f[nm] = pd.DataFrame(arr, index=dates, columns=md.symbols)
kv("13 因子 x 70 调仓日 全量现算耗时(s)", "%.1f" % (time.time() - t_fac))
d_cache = {}
for nm in FACTORS:
    d, a1, a2, nb = maxabsdiff(raw_f[nm].values, raw_c[nm].values)
    d_cache[nm] = (d, a1, a2)
kv("缓存面板 vs 现算面板 最大 max|差分|", fnum(max(x[0] for x in d_cache.values()), 3))
kv("缓存面板 vs 现算面板 单边有限格合计", "%d" % sum(x[1] + x[2] for x in d_cache.values()))
kv("索引一致性 (缓存 index == md.dates[ts])",
   bool((raw_c[FACTORS[0]].index == dates).all()))


class _TruncMD(object):
    """把 t0 之后的数据全部抹掉，用来验证因子在 t0 上只用了 <= t0 的数据。"""

    def __init__(self, md_, t0_):
        idx, cols = md_.dates, md_.symbols

        def cut(df):
            v = df.values.astype("float64").copy()
            v[t0_ + 1:, :] = np.nan
            return pd.DataFrame(v, index=idx, columns=cols)

        self.ret = cut(md_.ret)
        self.adj_close = cut(md_.adj_close)
        self.amt = cut(md_.amt)
        self.num_trades = cut(md_.num_trades)
        l = md_.listed.values.copy()
        l[t0_ + 1:, :] = False
        self.listed = pd.DataFrame(l, index=idx, columns=cols)
        ro = md_.ret_ok.values.copy()
        ro[t0_ + 1:, :] = False
        self.ret_ok = pd.DataFrame(ro, index=idx, columns=cols)


def _fac_at(Fobj, t0):
    return {nm: np.asarray(Fobj.get(nm, t0), dtype="float64") for nm in FACTORS}


def _cmp(ta, tb):
    dab = rel = 0.0
    nbad = 0
    for nm in FACTORS:
        a, b = ta[nm], tb[nm]
        if (np.isfinite(a) != np.isfinite(b)).any():
            nbad += 1
            continue
        ok = np.isfinite(a)
        if ok.any():
            dab = max(dab, float(np.abs(a[ok] - b[ok]).max()))
            sc = float(np.abs(a[ok]).max())
            if sc > 0:
                rel = max(rel, float(np.abs(a[ok] - b[ok]).max()) / sc)
    return dab, rel, nbad


# 对照 A：不截断，仅把 DataFrame 换成同值新数组（检验浮点/内存布局噪声）
ctrl = _fac_at(Factors(_TruncMD(md, len(md.dates) - 1)), ts[5])
full = _fac_at(F, ts[5])
d_ctrl, r_ctrl, nb_ctrl = _cmp(full, ctrl)
kv("对照：不截断的同值拷贝 max|因子值差| / 最大相对差 / 掩码不一致因子数",
   "%s / %s / %d" % (fnum(d_ctrl, 3), fnum(r_ctrl, 3), nb_ctrl))
d_trunc, r_trunc, nb_trunc = 0.0, 0.0, 0
for t0 in (ts[5], ts[35], ts[-2]):
    d1, r1, nb1 = _cmp(_fac_at(F, t0), _fac_at(Factors(_TruncMD(md, t0)), t0))
    d_trunc = max(d_trunc, d1)
    r_trunc = max(r_trunc, r1)
    nb_trunc += nb1
kv("截断未来数据后 max|因子值差| (3 日期 x 13 因子)", fnum(d_trunc, 3))
kv("截断未来数据后 最大相对差 / 掩码不一致因子数", "%s / %d" % (fnum(r_trunc, 3), nb_trunc))
kv("结论", "因子值只依赖 <=t 的数据（差值 = 浮点噪声，相对 1e-16）"
   if (r_trunc <= 1e-12 and nb_trunc == 0) else "存在未来信息泄漏")
OUT["panel_cache_maxdiff"] = max(x[0] for x in d_cache.values())

# z-score 面板：逐日横截面标准化（无时间序列统计量）— 独立重写 cs_zscore
def cs_zscore_indep(row):
    v = np.asarray(row, dtype="float64")
    ok = np.isfinite(v)
    out = np.full_like(v, np.nan)
    if ok.sum() < 3:
        return out
    lo, hi = np.nanpercentile(v[ok], [2, 98])
    w = np.clip(v, lo, hi)
    m, s = np.nanmean(w[ok]), np.nanstd(w[ok])
    out[ok] = (w[ok] - m) / s if s > 0 else 0.0
    return out


zsc_f = {nm: pd.DataFrame(np.vstack([cs_zscore_indep(r) for r in raw_f[nm].values]),
                          index=dates, columns=md.symbols) for nm in FACTORS}
d_z = max(maxabsdiff(zsc_f[nm].values, zsc_c[nm].values)[0] for nm in FACTORS)
kv("我的 z-score vs s07 的 zsc 面板 max|差分|", fnum(d_z, 3))

# --------------------------------------------------------------------------- #
# 3. 股票池：动态 vs 固定 / 分层与毛收益同域 / 上市退市
# --------------------------------------------------------------------------- #
sec("3. 股票池（复核项 4）")
n_all_day = uni_all.sum(1)
kv("md.universe() 是逐日动态池", True)
kv("池规模 首/中/末", "%d / %d / %d" % (n_all_day.iloc[0], int(n_all_day.median()), n_all_day.iloc[-1]))
kv("池规模 min/max, 不同取值的个数", "%d / %d, %d 种" % (n_all_day.min(), n_all_day.max(), n_all_day.nunique()))
days = list(md.dates)
u0 = set(md.symbols[uni_all.values[0]])
u1 = set(md.symbols[uni_all.values[-1]])
kv("首日池 vs 末日池 交集/并集", "%d / %d" % (len(u0 & u1), len(u0 | u1)))

# 3.1 池的构成是否只用 <=t 数据：独立重算 t 日一行
t_probe = ts[-30]
sub = slice(0, t_probe + 1)
adv20 = md.amt.iloc[sub].rolling(20, min_periods=10).mean().iloc[-1]
adv20_ref = md.adv(20).iloc[t_probe]
d_adv, o1, o2, nb = maxabsdiff(adv20.values, adv20_ref.values)
kv("adv20 只用 <=t 数据重算 max|差分|", "%s (单边 %d/%d)" % (fnum(d_adv, 3), o1, o2))
listed_days_t = md.listed.iloc[sub].cumsum().iloc[-1] * md.listed.iloc[t_probe]
d_ld, o1, o2, nb = maxabsdiff(listed_days_t.values, md.listed_days.iloc[t_probe].values)
kv("listed_days 只用 <=t 数据重算 max|差分|", "%s (单边 %d/%d)" % (fnum(d_ld, 3), o1, o2))
own = (md.traded.iloc[t_probe].values
       & (listed_days_t.values >= 250)
       & (adv20.values >= 2e7)
       & (md.close.iloc[t_probe].values >= 2.0)
       & ~md.st_like.values
       & ~md.no_limit.iloc[t_probe].values.astype(bool)
       & ~md.one_line.iloc[t_probe].values.astype(bool))
kv("自建 t 日池 vs md.universe() 该行 不一致格数", int((own != uni_all.values[t_probe]).sum()))

# 3.2 唯一一个「全样本统计量」：st_like（逐股、用整段历史判定）
st = md.st_like
kv("st_like 命中证券数（全样本统计量）", "%d 只: %s" % (int(st.sum()), list(st.index[st])))
u_no_st = md.universe(exclude_st=False)
kv("关掉 exclude_st 后多出来的池格数", int((u_no_st.values & ~uni_all.values).sum()))
kv("占池格比例", "%.6f%%" % (100.0 * (u_no_st.values & ~uni_all.values).sum() / max(uni_all.values.sum(), 1)))

# 3.3 上市/退市
kv("池内 listed_days 最小值（应 >=250）", int(md.listed_days.values[uni_all.values].min()))
kv("池内未上市(NaN 报价)格数（应 =0）", int((uni_all.values & ~md.listed.values).sum()))
last_listed = np.where(md.listed.values.any(0),
                       md.listed.values.shape[0] - 1 - md.listed.values[::-1].argmax(0), -1)
delisted = np.flatnonzero(last_listed < md.listed.values.shape[0] - 1)
viol = 0
for j in delisted:
    jj = int(j)
    viol += int(uni_all.values[last_listed[jj] + 1:, jj].sum())
kv("样本内退市/停止报价证券数", int(len(delisted)))
kv("退市后仍出现在池里的格数（应 =0）", int(viol))

# 3.4 分层 (run_deciles) 与毛收益 (decile_gross) 是否同域
probe = PanelStrategy(pd.DataFrame(np.zeros((1, len(md.symbols))), index=[md.dates[ts[0]]],
                                   columns=md.symbols))
kv("PanelStrategy 默认池参数", str(probe._uni_kw))
kv("md.universe(**PanelStrategy._uni_kw) is md.universe()",
   bool(md.universe(min_listed_days=250, min_adv=2e7, price_min=2.0, boards=None,
                    exclude_st=True) is md.universe()))
# 毛收益口径用的域：uni(ts) & finite(fwd)
SIGNS = None

# --------------------------------------------------------------------------- #
# 4. IC / 方向：样本内定方向？（复核项 6 与 2）
# --------------------------------------------------------------------------- #
sec("4. IC 与方向一致性（复核项 2 / 6）")
investable = uni & np.isfinite(fwd)
kv("可投资域格数占池格比例", "%.4f" % (investable.sum() / max(uni.sum(), 1)))

ic_mine = {}
for nm in FACTORS:
    x = raw_f[nm].values
    ic_mine[nm] = np.array([spearman(x[i], fwd[i], investable[i]) for i in range(len(ts))])
ic_mine = pd.DataFrame(ic_mine, index=dates)
ic_pub = pd.read_csv(os.path.join(RUNS, "vv_ic.csv"), encoding="utf-8", index_col=0,
                     parse_dates=True)[list(FACTORS)]
d_ic, a1, a2, nb = maxabsdiff(ic_mine.values, ic_pub.values)
kv("我的 IC vs out/runs/vv_ic.csv max|差分|", "%s（单边 %d/%d，共有 %d 格）" % (fnum(d_ic, 3), a1, a2, nb))

istat = pd.read_csv(os.path.join(TAB, "04_factor_ic.csv"), encoding="utf-8", index_col=0)
rows = {}
for nm in FACTORS:
    s = pd.Series(ic_mine[nm]).dropna()
    si = s[s.index <= pd.Timestamp(IS_END)]
    so = s[s.index > pd.Timestamp(IS_END)]
    t_is = si.mean() / si.std() * np.sqrt(len(si))
    rows[nm] = dict(IC=float(s.mean()), 样本内IC=float(si.mean()), 样本内t=float(t_is),
                    样本外IC=float(so.mean()),
                    样本外t=float(so.mean() / so.std() * np.sqrt(len(so))) if len(so) > 1 else np.nan,
                    方向IS=float(np.sign(si.mean())),
                    方向OOS=float(np.sign(so.mean())),
                    方向CSV=float(istat.loc[nm, "方向"]))
MINE = pd.DataFrame(rows).T
kv("我的方向(样本内) vs 04_factor_ic.csv 方向 不一致数",
   int((MINE["方向IS"] != MINE["方向CSV"]).sum()))
kv("我的 IC 全样本 vs CSV 最大绝对差", fnum(float((MINE["IC"] - istat["IC全样本"].astype(float)).abs().max()), 3))
kv("我的 样本内t vs CSV 最大绝对差", fnum(float((MINE["样本内t"] - istat["样本内t"].astype(float)).abs().max()), 3))
sel_mine = [nm for nm in FACTORS if abs(float(MINE.loc[nm, "样本内t"])) >= 1.0]
kv("按「|样本内t|>=1」重算的 sel", sel_mine)
kv("与 vv_config.sel_factors 完全一致", bool(sel_mine == list(cfg["sel_factors"])))
flip = [nm for nm in cfg["sel_factors"] if MINE.loc[nm, "方向OOS"] != MINE.loc[nm, "方向IS"]]
kv("入闱因子中「样本外 IC 符号会翻」的个数", "%d %s" % (len(flip), flip))
sel_oos = [nm for nm in FACTORS if abs(float(MINE.loc[nm, "样本外t"])) >= 1.0]
kv("若改用样本外 |t|>=1 筛因子会得到的 sel（对照，s07 没用）", sel_oos)
kv("样本外筛法 vs s07 实际 sel 的对称差", sorted(set(sel_oos) ^ set(list(cfg["sel_factors"]))))

signs_is = {nm: (float(MINE.loc[nm, "方向IS"]) if np.isfinite(MINE.loc[nm, "方向IS"]) else 1.0)
            for nm in FACTORS}
signs_csv = {nm: float(istat.loc[nm, "方向"]) for nm in FACTORS}
sel = list(cfg["sel_factors"])
comp_is = sum(zsc_f[nm] * signs_is[nm] for nm in sel) / len(sel)
comp_csv = sum(zsc_f[nm] * signs_csv[nm] for nm in sel) / len(sel)
comp_s07 = sum(zsc_c[nm] * signs_csv[nm] for nm in sel) / len(sel)
d_cs, _, _, _ = maxabsdiff(comp_is.values, comp_s07.values)
kv("我独立重算的合成面板 vs s07 合成面板 max|差分|", fnum(d_cs, 3))
d_cs2, _, _, _ = maxabsdiff(comp_csv.values, comp_s07.values)
kv("（用 CSV 方向 + s07 zsc）复现合成面板 max|差分|", fnum(d_cs2, 3))
OUT["_comp"] = comp_s07

# 合成面板的有效覆盖：10 个因子全非空才算得出（NaN 传染口径）
cov_each = np.mean([np.isfinite(zsc_f[nm].values).sum(1) for nm in sel], axis=0)
cov_comp = np.isfinite(comp_s07.values).sum(1)
kv("单因子平均有效数 / 合成有效数（均值）", "%.0f / %.0f" % (cov_each.mean(), cov_comp.mean()))
kv("合成有效数占单因子平均的比例", "%.4f" % (cov_comp.mean() / max(cov_each.mean(), 1)))
kv("池内平均证券数 / 池内且有合成打分数（均值）",
   "%.0f / %.0f" % (uni.sum(1).mean(), (np.isfinite(comp_s07.values) & uni).sum(1).mean()))

# --------------------------------------------------------------------------- #
# 5. 十分层：整层等权？持仓数？独立开放端收益验证（复核项 5）
# --------------------------------------------------------------------------- #
sec("5. 十分层回测（复核项 5）")
comp = comp_s07
R = comp.rank(axis=1, pct=True)
N_DEC = 10
cnt = {}
for d in range(N_DEC):
    lo, hi = d / N_DEC, (d + 1) / N_DEC
    m = (R.values > lo) & (R.values <= hi) & uni & np.isfinite(fwd)
    cnt["D%d" % (d + 1)] = m.sum(1)
CNT = pd.DataFrame(cnt, index=dates)
tab07 = pd.read_csv(os.path.join(TAB, "07_decile.csv"), encoding="utf-8", index_col=0)
kv("各层(池内∩有收益)持仓数中位数 我算 vs 07_decile.csv", "")
cmp_cnt = pd.DataFrame({"我算": CNT.median().round(0), "s07引擎": tab07["持仓数"]})
print(cmp_cnt.to_string())
kv("D1/D10 持仓数差（引擎 - 我算）",
   "%d / %d" % (int(tab07.loc["D1", "持仓数"] - CNT["D1"].median()),
                int(tab07.loc["D10", "持仓数"] - CNT["D10"].median())))
kv("每层在池内的比例（D1..D10, 中位）",
   str((CNT.median() / uni.sum(1).mean()).round(3).to_dict()))

# 5.1 权重口径：PanelStrategy(n_hold=1e6, buffer_mult=0, max_weight=1) 是否整层等权
i_probe = ts[-13]
panel_d10 = comp.where((R > 0.9) & (R <= 1.0))
st_d10 = PanelStrategy(panel_d10, name="D10probe", n_hold=10 ** 6, buffer_mult=0.0, max_weight=1.0)
w = st_d10.weights(i_probe, md)
k = len(w)
kv("D10 一期目标权重 只数 / 该层池内可观测量",
   "%d / %d" % (k, int(((R.values[ts.index(i_probe)] > 0.9) & (R.values[ts.index(i_probe)] <= 1.0)
                        & investable[ts.index(i_probe)]).sum())))
kv("权重是否全部相等 1/k (max|w-1/k|)",
   fnum(float((w.values - 1.0 / k).__abs__().max()), 3))
kv("是否整层：目标数 == 层内可交易数", bool(k == int(
    ((R.values[ts.index(i_probe)] > 0.9) & (R.values[ts.index(i_probe)] <= 1.0)
     & investable[ts.index(i_probe)]).sum())))

# 5.2 真引擎复现 D1 / D10（与 07_decile.csv 比）
HG_DEC = dict(exec_price="open", max_participation=0.10, min_trade_notional=5e3,
              rebalance_band=1e-5, gross_limit=0.98, cash=1e8, exec_window=3)
eng = {}
for layer, (lo, hi) in (("D1", (0.0, 0.1)), ("D10", (0.9, 1.0))):
    panel = comp.where((R > lo) & (R <= hi))
    st = PanelStrategy(panel, name=layer, n_hold=10 ** 6, buffer_mult=0.0, max_weight=1.0)
    res = Backtest(md, **HG_DEC).run(st, start=CFG_START, end=CFG_END, rebalance=cfg["rebal"])
    st_ = metrics.perf_stats(res.nav)
    eng[layer] = dict(年化=st_["年化收益"], 夏普=st_["夏普"], 持仓=float(res.daily.n_holdings.median()),
                      换手=float(res.daily.turnover.mean() * 252), n_trade=len(res.trades),
                      fee=float(res.trades.fee.sum()) if len(res.trades) else 0.0)
    kv("引擎复现 %s 年化/夏普/持仓/换手" % layer,
       "%.4f / %.4f / %.0f / %.2f   (07表 %.4f / %.4f / %.0f / %.2f)"
       % (eng[layer]["年化"], eng[layer]["夏普"], eng[layer]["持仓"], eng[layer]["换手"],
          tab07.loc[layer, "年化"], tab07.loc[layer, "夏普"], tab07.loc[layer, "持仓数"],
          tab07.loc[layer, "年换手"]))
kv("引擎复现 D1/D10 年化差 |Δ|",
   fnum(abs(eng["D1"]["年化"] - tab07.loc["D1", "年化"]), 3) + " / "
   + fnum(abs(eng["D10"]["年化"] - tab07.loc["D10", "年化"]), 3))

# 5.2b 反证：调仓带 2e-3（基准）时持仓数会不会滚大（s07 注释的说法）
panel_d1 = comp.where((R > 0.0) & (R <= 0.1))
st_b = PanelStrategy(panel_d1, name="D1band", n_hold=10 ** 6, buffer_mult=0.0, max_weight=1.0)
res_b = Backtest(md, **dict(HG_DEC, rebalance_band=2e-3)).run(st_b, start=CFG_START,
                                                              end=CFG_END, rebalance=cfg["rebal"])
kv("对照：D1 用含基准调仓带 2e-3 的持仓数中位数/最大",
   "%.0f / %.0f" % (res_b.daily.n_holdings.median(), res_b.daily.n_holdings.max()))

# 5.3 独立实现：月末信号 -> 次日开盘等权买入 -> 下月次日开盘卖出（毛收益、无约束）
def decile_port(hold_open):
    """s07 口径(月度 21 日 c2c 均值年化) 与 实际链式(open2open) 并列。"""
    pick = {}
    for d in range(N_DEC):
        lo, hi = d / N_DEC, (d + 1) / N_DEC
        pick[d] = (R.values > lo) & (R.values <= hi) & investable
    rets = {d: [] for d in range(N_DEC)}
    rets_cc = {d: [] for d in range(N_DEC)}
    for i in range(len(ts) - 1):
        t_in, t_out = ts[i] + 1, ts[i + 1] + 1
        if t_out >= len(md.dates):
            break
        px_in, px_out = hold_open[t_in], hold_open[t_out]
        with np.errstate(invalid="ignore", divide="ignore"):
            rr = px_out / px_in - 1.0
        for d in range(N_DEC):
            m = pick[d][i] & np.isfinite(rr)
            rets[d].append(float(np.nanmean(rr[m])) if m.sum() > 10 else np.nan)
            m2 = pick[d][i] & np.isfinite(fwd[i])
            rets_cc[d].append(float(np.nanmean(fwd[i][m2])) if m2.sum() > 10 else np.nan)
    n_day = ts[-1] + 1 - (ts[0] + 1)
    yrs = n_day / 252.0
    out = {}
    for d in range(N_DEC):
        a = np.array(rets[d], dtype="float64")
        b = np.array(rets_cc[d], dtype="float64")
        nav = np.prod(1.0 + a[np.isfinite(a)])
        nav2 = np.prod(1.0 + b[np.isfinite(b)])
        aa = a[np.isfinite(a)]
        out["D%d" % (d + 1)] = dict(open2open_ann=float(nav ** (1 / yrs) - 1),
                                    cc21_ann=float(nav2 ** (1 / yrs) - 1),
                                    mean_ann=float((1.0 + aa.mean()) ** 12 - 1.0),
                                    mean_21d=float(aa.mean()), vol_m=float(aa.std(ddof=1)),
                                    n_period=int(np.isfinite(a).sum()))
    return out


mydec = decile_port(aopen.values)
gross_tab = pd.read_csv(os.path.join(TAB, "07_decile_gross.csv"), encoding="utf-8", index_col=0)["毛收益年化"]
# 独立复现 s07 的毛收益口径：(1 + 各期层内 21 日 c2c 平均收益)^12 - 1
mg = {}
for d in range(N_DEC):
    lo, hi = d / N_DEC, (d + 1) / N_DEC
    m = (R.values > lo) & (R.values <= hi) & investable & np.isfinite(fwd)
    mr = np.array([float(np.nanmean(fwd[i][m[i]])) if m[i].sum() > 10 else np.nan
                   for i in range(len(ts))])
    mg["D%d" % (d + 1)] = float((1.0 + np.nanmean(mr)) ** 12 - 1.0)
kv("独立复现 07_decile_gross.csv 的 (1+mean)^12-1 max|差|",
   fnum(max(abs(mg[L] - float(gross_tab[L])) for L in mg), 3))
kv("独立实现（open[t+1]->open[t+1+21]、等权、零成本）年化", "")
for L in ("D1", "D5", "D10"):
    kv("  %s" % L, "我算 %.4f / s07毛收益表(21d c2c) %.4f / s07引擎净 %.4f"
       % (mydec[L]["open2open_ann"], float(gross_tab[L]), tab07.loc[L, "年化"]))
kv("持仓数中位数 我算 vs 引擎(净)",
   str({L: (int(CNT[L].median()), int(tab07.loc[L, "持仓数"])) for L in ("D1", "D5", "D10")}))
kv("同口径分解 D1/D10：月度平均收益 -> 均值年化 vs 链式年化",
   "D1 月均 %+.4f%% -> 均值年化 %+.2f%% vs 链式 %+.2f%%（月波动 %.2f%%）; "
   "D10 月均 %+.4f%% -> %+.2f%% vs %+.2f%%（月波动 %.2f%%）"
   % (100 * mydec["D1"]["mean_21d"], 100 * mydec["D1"]["mean_ann"], 100 * mydec["D1"]["open2open_ann"],
      100 * mydec["D1"]["vol_m"], 100 * mydec["D10"]["mean_21d"], 100 * mydec["D10"]["mean_ann"],
      100 * mydec["D10"]["open2open_ann"], 100 * mydec["D10"]["vol_m"]))
kv("D10-D1（我算 open2open）", "%.4f" % (mydec["D10"]["open2open_ann"] - mydec["D1"]["open2open_ann"]))
kv("D10-D1（s07 毛收益表 21d close-close）", "%.4f" % (float(gross_tab["D10"]) - float(gross_tab["D1"])))
kv("D10-D1（s07 引擎净年化）", "%.4f" % (tab07.loc["D10", "年化"] - tab07.loc["D1", "年化"]))
OUT["_decile_open2open"] = mydec

# --------------------------------------------------------------------------- #
# 6. 留一法 / 随机对照 / IC 口径（复核项 2）
# --------------------------------------------------------------------------- #
sec("6. 合成 IC / 留一法 / 随机对照（复核项 2）")
ic_comp = np.array([spearman(comp.values[i], fwd[i], investable[i]) for i in range(len(ts))])
sc = pd.Series(ic_comp, index=dates).dropna()
si, so = sc[sc.index <= pd.Timestamp(IS_END)], sc[sc.index > pd.Timestamp(IS_END)]
kv("合成 IC 全样本 / 样本内 / 样本外", "%.4f / %.4f / %.4f" % (sc.mean(), si.mean(), so.mean()))
ic_comp_o = np.array([spearman(comp.values[i], fwd_open[i], investable[i]) for i in range(len(ts))])
so_ = pd.Series(ic_comp_o, index=dates).dropna()
kv("同一打分换 open[t+1] 口径 IC 全样本", "%.4f（close-close 为 %.4f）" % (so_.mean(), sc.mean()))

loo07 = pd.read_csv(os.path.join(TAB, "07_loo.csv"), encoding="utf-8", index_col=0)
base = sc.mean()
d_loo = {}
for nm in sel:
    rest = [f for f in sel if f != nm]
    pan = sum(zsc_f[f].values * signs_is[f] for f in rest) / len(rest)
    ics = np.array([spearman(pan[i], fwd[i], investable[i]) for i in range(len(ts))])
    d_loo[nm] = base - float(np.nanmean(ics))
kv("留一法 ΔIC 我的 vs 07_loo.csv max|差分|",
   fnum(max(abs(d_loo[nm] - float(loo07.loc[nm, "ΔIC"])) for nm in sel), 3))

# 随机对照：同 seed 复跑 s07 的口径
def random_control(ok, fwd_, n_draw=200, top_n=50, seed=20211016):
    rng = np.random.default_rng(seed)
    ann = []
    for _ in range(n_draw):
        tot = 1.0
        for i in range(len(ok)):
            idx = np.flatnonzero(ok[i])
            if len(idx) == 0:
                continue
            pick = rng.choice(idx, size=min(top_n, len(idx)), replace=False)
            r = float(np.nanmean(fwd_[i][pick]))
            if np.isfinite(r):
                tot *= (1.0 + r)
        ann.append(tot ** (12.0 / len(ok)) - 1.0)
    return np.array(ann)


rand = random_control(investable, fwd)
rc = pd.read_csv(os.path.join(TAB, "07_random_control.csv"), encoding="utf-8", index_col=0)["年化"]
kv("随机对照 中位数 我算 vs 表", "%.6f / %.6f" % (np.median(rand), float(rc.loc["中位数"])))
kv("随机对照 5% / 95% 分位 我算 vs 表",
   "%.4f/%.4f vs %.4f/%.4f" % (np.percentile(rand, 5), np.percentile(rand, 95),
                               float(rc.loc["5% 分位"]), float(rc.loc["95% 分位"])))
kv("随机对照 最好/最差 我算 vs 表",
   "%.4f/%.4f vs %.4f/%.4f" % (rand.max(), rand.min(), float(rc.loc["最好一组"]), float(rc.loc["最差一组"])))


def topn_gross(score, ok, fwd_, top_n=50):
    tot = 1.0
    for i in range(len(ok)):
        m = ok[i] & np.isfinite(score[i]) & np.isfinite(fwd_[i])
        idx = np.flatnonzero(m)
        if len(idx) == 0:
            continue
        k = min(top_n, len(idx))
        pick = idx[np.argsort(-score[i][idx], kind="stable")[:k]]
        tot *= (1.0 + float(np.nanmean(fwd_[i][pick])))
    return tot ** (12.0 / len(ok)) - 1.0


gc = topn_gross(comp.values, investable, fwd, top_n=int(cfg["n_hold"]))
kv("合成 top50 毛收益 我算 vs 表", "%.6f / %.6f" % (gc, float(rc.loc["合成 top50 毛收益"])))
kv("随机组里超过合成的比例 (1-该值 = 表里「高于 x% 的随机组」)",
   "%.4f → 高于 %.1f%%" % (float((rand < gc).mean()), 100 * float((rand < gc).mean())))

# 6.5 合成 top50：s07 的毛收益口径 vs 「次日开盘可执行」口径
pick_top = []
for i in range(len(ts)):
    m = investable[i] & np.isfinite(comp.values[i])
    idx = np.flatnonzero(m)
    k = min(int(cfg["n_hold"]), len(idx))
    pick_top.append(idx[np.argsort(-comp.values[i][idx], kind="stable")[:k]])
ch_c2c, ch_o2o = 1.0, 1.0
n_o2o = 0
for i in range(len(ts) - 1):
    if len(pick_top[i]) == 0:
        continue
    v = fwd[i][pick_top[i]]
    v = v[np.isfinite(v)]
    ch_c2c *= (1.0 + v.mean())
    t_in, t_out = ts[i] + 1, ts[i + 1] + 1
    if t_out < len(md.dates):
        rr = ao[t_out][pick_top[i]] / ao[t_in][pick_top[i]] - 1.0
        rr = rr[np.isfinite(rr)]
        if len(rr):
            ch_o2o *= (1.0 + rr.mean())
            n_o2o += 1
kv("合成 top50 链式毛收益(s07 21d c2c 口径) vs 表",
   "%.4f / %.4f" % (ch_c2c ** (12.0 / len(ts)) - 1.0, float(rc.loc["合成 top50 毛收益"])))
kv("合成 top50 链式毛收益(次日开盘->下月次日开盘, 零成本) vs 引擎净",
   "%.4f（%d 期）/ 引擎净 %.4f" % (ch_o2o ** (252.0 / (n_o2o * 21.0)) - 1.0, n_o2o,
                                  float(pd.read_csv(os.path.join(TAB, "04_vol_strategy_summary.csv"),
                                                    encoding="utf-8", index_col=0).loc["vv_comp", "年化收益"])))

# --------------------------------------------------------------------------- #
# 7. 静态扫描：全样本统计量的出现位置（复核项 3）
# --------------------------------------------------------------------------- #
sec("7. 静态扫描：s07 里全样本统计量的出现位置（复核项 3）")
src = open(S07, encoding="utf-8").read().split("\n")
func = None
pats = [("nanmean/np.mean", r"np\.nanmean|np\.mean|\.mean\(\)"),
        ("nanstd/std/quantile/percentile", r"nanstd|\.std\(|quantile|nanpercentile|np\.percentile"),
        ("corr/rank/cummax 跨期", r"\.corr\(|\.rank\(|autocorr|cummax|nanargmax|np\.median"),
        ("cumprod/compound", r"cumprod|np\.prod")]
hits = []
for i, l in enumerate(src, 1):
    m = re.match(r"^def\s+(\w+)", l)
    if m:
        func = m.group(1)
    for tag, p in pats:
        if re.search(p, l) and not l.strip().startswith("#"):
            hits.append((i, func, tag, l.strip()[:100]))
            break
for i, f, tag, l in hits:
    print("   L%-5d [%-22s] %s" % (i, f, l))
kv("含全样本统计量的函数（去重）", sorted(set(str(h[1]) for h in hits)))

# --------------------------------------------------------------------------- #
sec("8. 汇总")
el = time.time() - T00
kv("脚本总耗时(s)", "%.1f" % el)
OUT["_meta"] = dict(s07=str(S07), ts0=str(dates[0].date()), tsN=str(dates[-1].date()),
                    n_period=len(ts), is_end=IS_END, elapsed_sec=round(el, 1))
with open(os.path.join(_HERE, "s07_indep_out.json"), "w", encoding="utf-8") as fh:
    json.dump(OUT, fh, ensure_ascii=False, indent=1, default=str)
print("\n结果已写: %s" % os.path.join(_HERE, "s07_indep_out.json"))
