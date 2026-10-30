# -*- coding: utf-8 -*-
"""Formula-level factor evolution on Class4_Backtest data."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.lib import MarketData
from factor_zoo.ctx import Ctx
import importlib.util
_spec = importlib.util.spec_from_file_location("class4_mc", ROOT / "03_因子挖掘" / "mining_core.py")
mc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(mc)

CACHE = ROOT / "out/cache/evo_formula"; CACHE.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "out/03_因子挖掘/evo_formula"; OUT.mkdir(parents=True, exist_ok=True)
TRAIN = ("2021-01-01", "2022-12-31")
VALID = ("2023-01-01", "2024-12-31")
OOS = ("2025-01-01", "2026-09-30")
MAX_NODES = 9
MAX_DEPTH = 4


def idx_for(dates, lo, hi):
    return np.flatnonzero((dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi)))


def _mean(W):
    ok = np.isfinite(W); cnt = ok.sum(0)
    need = max(1, min(len(W), int(0.6 * len(W))))
    out = np.where(ok, W, 0.0).sum(0) / np.maximum(cnt, 1)
    return np.where(cnt >= need, out, np.nan)


def _std(W, min_frac=0.6):
    ok = np.isfinite(W); cnt = ok.sum(0)
    need = max(2, min(len(W), int(min_frac * len(W))))
    mu = np.where(ok, W, 0.0).sum(0) / np.maximum(cnt, 1)
    var = (np.where(ok, W - mu, 0.0) ** 2).sum(0) / np.maximum(cnt - 1, 1)
    return np.where(cnt >= need, np.sqrt(var), np.nan)


def _max(W):
    ok = np.isfinite(W); cnt = ok.sum(0); out = np.where(ok, W, -np.inf).max(0)
    return np.where(cnt >= max(1, min(len(W), int(0.6 * len(W)))), out, np.nan)


def _min(W):
    ok = np.isfinite(W); cnt = ok.sum(0); out = np.where(ok, W, np.inf).min(0)
    return np.where(cnt >= max(1, min(len(W), int(0.6 * len(W)))), out, np.nan)


def _corr(X, Y):
    ok = np.isfinite(X) & np.isfinite(Y)
    n = ok.sum(0)
    mx = np.where(ok, X, 0.0).sum(0) / np.maximum(n, 1)
    my = np.where(ok, Y, 0.0).sum(0) / np.maximum(n, 1)
    xc = np.where(ok, X - mx, 0.0); yc = np.where(ok, Y - my, 0.0)
    den = np.sqrt((xc * xc).sum(0) * (yc * yc).sum(0))
    out = (xc * yc).sum(0) / np.where(den > 0, den, np.nan)
    return np.where(n >= np.maximum(5, int(0.6 * len(X))), out, np.nan)


def _win(t, n, X):
    lo = max(int(t) - int(n) + 1, 0)
    return X[lo:int(t) + 1]


def _skew(W):
    ok = np.isfinite(W); n = ok.sum(0); mu = np.where(ok, W, 0).sum(0) / np.maximum(n, 1)
    x = np.where(ok, W - mu, 0.0); m2 = (x*x).sum(0) / np.maximum(n, 1); m3 = (x**3).sum(0) / np.maximum(n, 1)
    out = m3 / np.where(m2 > 0, m2 ** 1.5, np.nan)
    return np.where(n >= np.maximum(5, int(0.6 * len(W))), out, np.nan)


def feature_specs():
    specs = {}
    def add(name, desc, fn): specs[name] = dict(desc=desc, fn=fn)
    for n in [1,5,10,20,60,120,250]:
        add(f"ret{n}", f"过去{n}日复权收益", lambda t,C,n=n: np.where(np.isfinite(C.C[t-n]) & (C.C[t-n] > 0), C.C[t]/C.C[t-n]-1.0, np.nan) if t>=n else np.full(C.k,np.nan))
    for n in [5,20,60,120]:
        add(f"vol{n}", f"{n}日收益波动", lambda t,C,n=n: _std(_win(t,n,C.R)))
    for n in [20,60]:
        add(f"downvol{n}", f"{n}日下行波动", lambda t,C,n=n: _std(np.where(_win(t,n,C.R)<0,_win(t,n,C.R),np.nan),0.20))
        add(f"upvol{n}", f"{n}日上行波动", lambda t,C,n=n: _std(np.where(_win(t,n,C.R)>0,_win(t,n,C.R),np.nan),0.20))
    add("udvol60", "60日下行/上行波动比", lambda t,C: _std(np.where(_win(t,60,C.R)<0,_win(t,60,C.R),np.nan),0.20)/(_std(np.where(_win(t,60,C.R)>0,_win(t,60,C.R),np.nan),0.20)+1e-12))
    add("max1d20", "20日最大单日收益", lambda t,C: _max(_win(t,20,C.R)))
    add("max1d60", "60日最大单日收益", lambda t,C: _max(_win(t,60,C.R)))
    def max5d(t,C,n):
        W = C.C[max(0,t-n+1):t+1]; out=np.full(C.k,np.nan)
        if len(W)>=5:
            r5 = W[5:]/W[:-5]-1.0; out = np.nanmax(r5,axis=0)
        return out
    add("max5d20", "20日内最大5日收益", lambda t,C: max5d(t,C,20))
    add("max5d60", "60日内最大5日收益", lambda t,C: max5d(t,C,60))
    for n in [1,5,20]:
        add(f"range{n}", f"{n}日平均振幅", lambda t,C,n=n: _mean((_win(t,n,C.H)-_win(t,n,C.L))/(_win(t,n,C.C)+1e-12)))
    for n in [20,60]:
        add(f"amtlog{n}", f"log({n}日平均成交额)", lambda t,C,n=n: np.log1p(_mean(_win(t,n,C.A))))
        add(f"nlog{n}", f"log({n}日平均成交笔数)", lambda t,C,n=n: np.log1p(_mean(_win(t,n,C.NT))))
    for n in [5,20,60]:
        add(f"amtrel{n}", f"{n}日成交额/250日成交额", lambda t,C,n=n: _mean(_win(t,n,C.A))/(_mean(_win(t,250,C.A))+1e-12))
    add("tradesize20", "20日平均单笔成交额(log)", lambda t,C: np.log1p(_mean(_win(t,20,C.A)/(_win(t,20,C.NT)+1e-12))))
    for n in [5,20]:
        def ovn(t,C,n=n):
            i0=max(0,t-n); num=C.ADJ_O[i0+1:t+1]; den=C.C[i0:t]
            return _mean(num/den-1.0)
        def intra(t,C,n=n):
            return _mean(C.C[max(0,t-n+1):t+1]/C.ADJ_O[max(0,t-n+1):t+1]-1)
        add(f"ovn{n}", f"{n}日平均隔夜收益", ovn)
        add(f"intra{n}", f"{n}日平均日内收益", intra)
    for n in [20,60]:
        add(f"pvcorr{n}", f"{n}日收益-成交额变化相关", lambda t,C,n=n: _corr(C.R[max(1,t-n+1):t+1], np.diff(np.log1p(C.A[max(0,t-n):t+1]),axis=0)))
    for n in [20,60]:
        add(f"illiq{n}", f"{n}日非流动性", lambda t,C,n=n: np.log1p(_mean(np.abs(_win(t,n,C.R))/(_win(t,n,C.A)+1e-12))))
    for n in [20,60]:
        add(f"pos{n}", f"{n}日价格位置", lambda t,C,n=n: (C.C[t]-_min(_win(t,n,C.C)))/(_max(_win(t,n,C.C))-_min(_win(t,n,C.C))+1e-12))
        add(f"disthigh{n}", f"距{n}日高点的距离", lambda t,C,n=n: C.C[t]/(_max(_win(t,n,C.ADJ_H))+1e-12)-1)
        add(f"distlow{n}", f"距{n}日低点的距离", lambda t,C,n=n: C.C[t]/(_min(_win(t,n,C.ADJ_L))+1e-12)-1)
        add(f"madev{n}", f"收盘/MA{n}-1", lambda t,C,n=n: C.C[t]/(_mean(_win(t,n,C.C))+1e-12)-1)
    add("magap20_60", "MA20/MA60-1", lambda t,C: _mean(_win(t,20,C.C))/(_mean(_win(t,60,C.C))+1e-12)-1)
    for n in [20,60]:
        add(f"autocorr{n}", f"{n}日收益自相关", lambda t,C,n=n: _corr(C.R[max(1,t-n+1):t+1], C.R[max(0,t-n):t]))
        add(f"skew{n}", f"{n}日收益偏度", lambda t,C,n=n: _skew(_win(t,n,C.R)))
    add("closeloc20", "20日收盘位置均值", lambda t,C: _mean((_win(t,20,C.C)-_win(t,20,C.L))/(_win(t,20,C.H)-_win(t,20,C.L)+1e-12)))
    return specs


def build_base(force=False):
    meta_p = CACHE / "meta.json"; bz_p = CACHE / "br.npy"
    if not force and meta_p.exists() and bz_p.exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        return dict(panel_dates=pd.DatetimeIndex(meta["panel_dates"]), features=meta["features"],
                    desc=meta["desc"], BR=np.load(bz_p, mmap_mode="r"),
                    y20=np.load(ROOT/"out/cache/evo/y20.npy"),
                    uni=np.load(ROOT/"out/cache/evo/universe.npy"))
    panel_dates = pd.DatetimeIndex(json.loads((ROOT/"out/cache/evo/names.json").read_text(encoding="utf-8"))["panel_dates"])
    md = MarketData(); C = Ctx(md)
    ts = [md.dates.get_loc(d) for d in panel_dates]
    specs = feature_specs(); names = list(specs)
    print(f"building formula primitives: {len(names)} features x {len(ts)} months x {C.k} symbols")
    B = np.empty((len(ts), len(names), C.k), dtype=np.float32)
    for j,nm in enumerate(names):
        for i,t in enumerate(ts):
            try: B[i,j,:] = np.asarray(specs[nm]["fn"](int(t), C), dtype="float64")
            except Exception: B[i,j,:] = np.nan
        if j % 10 == 0 or j == len(names)-1: print(f"  {j+1}/{len(names)} {nm}")
    uni = np.load(ROOT/"out/cache/evo/universe.npy")
    BR = np.zeros_like(B)
    for i in range(len(ts)):
        for j in range(len(names)):
            z = mc.cs_rank_z(np.where(uni[i], B[i,j,:], np.nan))
            BR[i,j,:] = np.where(np.isfinite(z), z, 0.0)
    np.save(bz_p, BR)
    meta_p.write_text(json.dumps(dict(panel_dates=[str(x.date()) for x in panel_dates], features=names,
        desc={k: specs[k]["desc"] for k in names}), ensure_ascii=False, indent=2), encoding="utf-8")
    return dict(panel_dates=panel_dates, features=names, desc={k: specs[k]["desc"] for k in names}, BR=BR,
                y20=np.load(ROOT/"out/cache/evo/y20.npy"), uni=uni)


def tree_nodes(t): return 1 if t[0]=="feat" else 1+sum(tree_nodes(x) for x in t[1:] if isinstance(x,tuple))
def tree_depth(t): return 1 if t[0]=="feat" else 1+max(tree_depth(x) for x in t[1:] if isinstance(x,tuple))
def expr(t):
    if t[0]=="feat": return t[1]
    if len(t)==2: return f"{t[0]}({expr(t[1])})"
    return f"({expr(t[1])} {t[0]} {expr(t[2])})"


def random_tree(rng, features, depth=0):
    if depth>=MAX_DEPTH or rng.random()<0.30:
        return ("feat", str(rng.choice(features)))
    if rng.random()<0.30:
        op=str(rng.choice(["neg","abs","square","rank","z"]))
        return (op, random_tree(rng, features, depth+1))
    op=str(rng.choice(["add","sub","mul","div"]))
    return (op, random_tree(rng, features, depth+1), random_tree(rng, features, depth+1))


def eval_tree(t, data):
    if t[0]=="feat":
        j=data["features"].index(t[1]); return np.asarray(data["BR"][:,j,:],dtype="float64")
    if t[0]=="neg": return -eval_tree(t[1],data)
    if t[0]=="abs": return np.abs(eval_tree(t[1],data))
    if t[0]=="square":
        x=np.clip(eval_tree(t[1],data),-3,3); return x*x
    if t[0]=="rank":
        x=eval_tree(t[1],data); out=np.zeros_like(x)
        for i in range(x.shape[0]): out[i]=np.where(np.isfinite(x[i]),mc.cs_rank_z(x[i]),np.nan)
        return out
    if t[0]=="z":
        return eval_tree(("rank",t[1]),data)
    a=eval_tree(t[1],data); b=eval_tree(t[2],data)
    if t[0]=="add": return a+b
    if t[0]=="sub": return a-b
    if t[0]=="mul": return a*b
    return np.where(np.abs(b)>0.25, a/np.where(np.abs(b)>0.25,b,1.0), 0.0)


def tree_key(t): return expr(t)
