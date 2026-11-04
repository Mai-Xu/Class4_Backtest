# -*- coding: utf-8 -*-
"""Self-evolving factor-combination research on top of Class4_Backtest.

The factor *implementation* and forward-return convention are inherited from
Class4. This module evolves sparse factor-portfolio genomes (factor subset +
positive weights) using 2021-2022 for training, keeps 2023-2024 for a single
validation look, and never touches 2025-2026 until a freeze file is written.
"""
from __future__ import annotations
import hashlib, importlib.util, json, math, sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from factor_zoo import api  # noqa: E402

# Import the course mining core despite the numeric directory name.
_spec = importlib.util.spec_from_file_location("class4_mining_core", ROOT / "03_因子挖掘" / "mining_core.py")
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

CACHE = ROOT / "out" / "cache" / "evo"
CACHE.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "out" / "03_因子挖掘" / "evo"
OUT.mkdir(parents=True, exist_ok=True)
TRAIN = ("2021-01-01", "2022-12-31")
VALID = ("2023-01-01", "2024-12-31")
OOS = ("2025-01-01", "2026-09-30")
MAX_ACTIVE = 16
MAX_PER_FAMILY = 3
N_HOLD = 1700


def _rank_z_rows(arr: np.ndarray, uni: np.ndarray) -> np.ndarray:
    out = np.zeros(arr.shape, dtype=np.float32)
    for i in range(arr.shape[0]):
        x = np.asarray(arr[i], dtype="float64")
        ok = np.asarray(uni[i], bool) & np.isfinite(x)
        if ok.sum() < 50:
            continue
        # mc.cs_rank_z does not accept a mask; mask first, then restore.
        xm = np.where(ok, x, np.nan)
        z = mc.cs_rank_z(xm)
        out[i] = np.where(np.isfinite(z), z, 0.0).astype(np.float32)
    return out


def build_data(force=False):
    """Cache rank-z factor panels + Class4 forward returns + dynamic universe."""
    names_path = CACHE / "names.json"
    z_path = CACHE / "rankz.npy"
    y_path = CACHE / "y20.npy"
    u_path = CACHE / "universe.npy"
    if not force and names_path.exists() and z_path.exists() and y_path.exists() and u_path.exists():
        names = json.loads(names_path.read_text(encoding="utf-8"))
        return dict(names=names["names"], panel_dates=pd.DatetimeIndex(names["panel_dates"]),
                    Z=np.load(z_path, mmap_mode="r"), y20=np.load(y_path), uni=np.load(u_path),
                    families=names["families"], clusters=names["clusters"])
    fd = api.load_factor_dict()
    keeps = fd[(fd["coverage_is"] >= 0.5) & (fd["is_rep"] == True)].copy()
    names = list(keeps.index)
    panel_dates = api.panel_dates()
    meta = json.loads((ROOT / "out/cache/selector/meta.json").read_text(encoding="utf-8"))
    symbols = meta["symbols"]
    print(f"building evo cache: {len(names)} representatives x {len(panel_dates)} months x {len(symbols)} symbols")
    Z = np.zeros((len(panel_dates), len(names), len(symbols)), dtype=np.float32)
    for j, nm in enumerate(names):
        p = api.load_panel(nm).reindex(index=panel_dates, columns=symbols)
        Z[:, j, :] = _rank_z_rows(p.to_numpy(dtype="float64"), np.ones((len(panel_dates), len(symbols)), bool))
        if j % 40 == 0 or j == len(names) - 1:
            print(f"  {j+1}/{len(names)} {nm}")
    md = mc.MarketData()
    ts = [md.dates.get_loc(pd.Timestamp(d)) for d in panel_dates]
    uni = np.asarray(mc.universe_mask(md, ts), dtype=bool)
    # Recompute z within the dynamic universe (the first pass above used all symbols).
    for j in range(len(names)):
        Z[:, j, :] = _rank_z_rows(Z[:, j, :], uni)
    y20 = mc.fwd_returns(md, ts, horizons=(20,))[20].astype("float64")
    np.save(z_path, Z); np.save(y_path, y20); np.save(u_path, uni)
    names_path.write_text(json.dumps({
        "names": names,
        "panel_dates": [str(x.date()) for x in panel_dates],
        "families": keeps["family"].to_dict(),
        "clusters": {str(k): int(v) for k, v in keeps["cluster"].items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return dict(names=names, panel_dates=panel_dates, Z=Z, y20=y20, uni=uni,
                families=keeps["family"].to_dict(), clusters=keeps["cluster"].to_dict())


def idx_for(dates, lo, hi):
    return np.flatnonzero((dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi)))


def directions_on_train(data, train_idx):
    dz, y, u = data["Z"], data["y20"], data["uni"]
    dirs, stats = {}, {}
    for j, nm in enumerate(data["names"]):
        ic = mc.ic_series(dz[train_idx, j, :], y[train_idx], u[train_idx])
        st = mc.ic_stats(ic)
        if np.isfinite(st["ic"]) and st["n"] >= 18:
            dirs[nm] = 1.0 if st["ic"] >= 0 else -1.0
            stats[nm] = st
    pool = [nm for nm in data["names"] if nm in stats and abs(stats[nm]["ic"]) >= 0.01]
    return dirs, stats, pool


def normalized(genome, families, pool):
    g = {k: float(v) for k, v in genome.items() if k in pool and np.isfinite(v) and v > 0}
    if not g:
        return {}
    # Enforce family cap before the active-factor cap.
    while len(g) > MAX_ACTIVE:
        g.pop(min(g, key=lambda k: g[k]))
    for _ in range(8):
        cnt = Counter(families.get(k, "?") for k in g)
        over = [f for f, n in cnt.items() if n > MAX_PER_FAMILY]
        if not over:
            break
        fam = over[0]
        drop = min((k for k in g if families.get(k, "?") == fam), key=lambda k: g[k])
        g.pop(drop, None)
    return g


def random_genome(rng, pool, families, n=None):
    n = n or rng.integers(5, min(13, len(pool)) + 1)
    chosen, cnt = [], Counter()
    for nm in rng.permutation(pool):
        f = families.get(nm, "?")
        if cnt[f] >= 2:
            continue
        chosen.append(str(nm)); cnt[f] += 1
        if len(chosen) >= n:
            break
    weights = rng.choice([0.25, 0.5, 1.0, 1.5, 2.0, 3.0], size=len(chosen))
    return dict(zip(chosen, weights))


def top_turnover(score, uni, n_hold=N_HOLD):
    prev = None; vals = []
    for i in range(score.shape[0]):
        ok = uni[i] & np.isfinite(score[i])
        cand = np.flatnonzero(ok)
        if len(cand) < n_hold:
            continue
        order = cand[np.argsort(score[i, cand])[-n_hold:]]
        cur = set(order.tolist())
        if prev is not None:
            vals.append(1.0 - len(cur & prev) / float(n_hold))
        prev = cur
    return float(np.mean(vals)) if vals else np.nan


def score_genome(genome, idx, data, dirs):
    Z, u = data["Z"], data["uni"]
    score = np.zeros((len(idx), Z.shape[2]), dtype="float64")
    total = 0.0
    for nm, w in genome.items():
        j = data["names"].index(nm)
        score += float(w) * float(dirs[nm]) * np.asarray(Z[idx, j, :], dtype="float64")
        total += float(w)
    if total > 0:
        score /= total
    score = np.where(u[idx], score, np.nan)
    return score


def stats_for(idx, genome, data, dirs):
    s = score_genome(genome, idx, data, dirs)
    y, u = data["y20"], data["uni"]
    ic = mc.ic_series(s, y[idx], u[idx])
    st = mc.ic_stats(ic, periods_per_year=12.0)
    turn = top_turnover(s, u[idx])
    return st, turn, ic


def train_fitness(st, turn, n_active):
    fold = st.get("fold_min_icir", 0.0)
    fit = 0.85 * st["icir"] + 0.15 * min(0.0, fold)
    fit -= 0.04 * max(0, n_active - 8)
    fit -= 0.60 * max(0.0, (turn - 0.65) if np.isfinite(turn) else 0.0)
    return float(fit)


def fold_min_icir(idx, genome, data, dirs):
    # Four contiguous folds of the segment (24 months -> 6-month blocks).
    parts = np.array_split(np.arange(len(idx)), 4)
    vals = []
    for part in parts:
        sub = idx[part]
        if len(sub) < 3: continue
        s = score_genome(genome, sub, data, dirs)
        ic = mc.ic_series(s, data["y20"][sub], data["uni"][sub])
        vals.append(mc.ic_stats(ic)["icir"])
    return float(np.nanmin(vals)) if vals else np.nan

def evaluate(genome, idx, data, dirs, with_folds=True):
    st, turn, ic = stats_for(idx, genome, data, dirs)
    if with_folds:
        st["fold_min_icir"] = fold_min_icir(idx, genome, data, dirs)
    st["turnover_top1700"] = turn
    st["n_active"] = len(genome)
    return st


def mutate(parent, rng, pool, families):
    child = dict(parent)
    op = rng.choice(["add", "drop", "swap", "weight", "cross"])
    if op == "add" and len(child) < MAX_ACTIVE:
        cand = [x for x in pool if x not in child]
        rng.shuffle(cand)
        for nm in cand:
            if Counter(families.get(k, "?") for k in child).get(families.get(nm, "?"), 0) < MAX_PER_FAMILY:
                child[nm] = float(rng.choice([0.25, 0.5, 1.0, 1.5, 2.0, 3.0])); break
    elif op == "drop" and len(child) > 3:
        child.pop(rng.choice(list(child)))
    elif op == "swap" and child:
        old = rng.choice(list(child)); child.pop(old)
        cand = [x for x in pool if x not in child]
        rng.shuffle(cand)
        for nm in cand:
            if Counter(families.get(k, "?") for k in child).get(families.get(nm, "?"), 0) < MAX_PER_FAMILY:
                child[nm] = float(rng.choice([0.25, 0.5, 1.0, 1.5, 2.0, 3.0])); break
    elif op == "weight" and child:
        k = rng.choice(list(child)); child[k] = float(rng.choice([0.25, 0.5, 1.0, 1.5, 2.0, 3.0]))
    else:
        # A light crossover: swap a random subset with a second random parent.
        other = random_genome(rng, pool, families)
        keys = list(set(child) | set(other)); rng.shuffle(keys)
        for k in keys[: max(1, len(keys)//3)]:
            if k in other: child[k] = 0.5 * child.get(k, other[k]) + 0.5 * other[k]
            else: child.pop(k, None)
    return normalized(child, families, pool)


def tournament(pop, fitness, rng, k=3):
    ids = rng.choice(len(pop), size=min(k, len(pop)), replace=False)
    return pop[max(ids, key=lambda i: fitness[i])]


def run_evolution(data, dirs, stats, pool, seed=20261030, pop_size=24, gens=3):
    rng = np.random.default_rng(seed); families = data["families"]
    train_idx = idx_for(data["panel_dates"], *TRAIN)
    # Baseline: train-only ICIR top-8 with family cap 2.
    ranked = sorted(pool, key=lambda nm: abs(stats[nm]["icir"]), reverse=True)
    base, cnt = {}, Counter()
    for nm in ranked:
        f = families.get(nm, "?")
        if cnt[f] >= 2: continue
        base[nm] = abs(stats[nm]["icir"]); cnt[f] += 1
        if len(base) >= 8: break
    pop = [normalized(base, families, pool)]
    while len(pop) < pop_size:
        g = random_genome(rng, pool, families)
        if g not in pop: pop.append(g)
    all_records = {}; fitness = []
    for gen in range(gens):
        if gen == 0:
            batch = pop
        else:
            elites = [pop[i] for i in np.argsort(fitness)[-6:]]
            batch = list(elites)
            while len(batch) < pop_size:
                p1 = tournament(pop, fitness, rng)
                p2 = tournament(pop, fitness, rng)
                child = mutate(p1, rng, pool, families)
                if rng.random() < 0.25: child = mutate(p2, rng, pool, families)
                child = normalized(child, families, pool)
                if child not in batch: batch.append(child)
        new_fit = []
        for g in batch:
            key = tuple(sorted(g.items()))
            if key not in all_records:
                st = evaluate(g, train_idx, data, dirs, with_folds=True)
                st["generation"] = gen; st["genome"] = g; st["fitness"] = train_fitness(st, st["turnover_top1700"], len(g))
                all_records[key] = st
            new_fit.append(all_records[key]["fitness"])
        pop, fitness = batch, new_fit
        print(f"generation {gen+1}: best train fitness={max(fitness):.3f}, unique={len(all_records)}")
    return list(all_records.values()), base


def random_budget(pool, families, data, dirs, n, seed=20261031):
    rng = np.random.default_rng(seed); idx = idx_for(data["panel_dates"], *TRAIN)
    recs = []
    for i in range(n):
        g = random_genome(rng, pool, families)
        st = evaluate(g, idx, data, dirs, with_folds=True)
        st["generation"] = -1; st["genome"] = g; st["fitness"] = train_fitness(st, st["turnover_top1700"], len(g))
        recs.append(st)
    return recs


def validation_record(rec, data, dirs):
    vidx = idx_for(data["panel_dates"], *VALID)
    st = evaluate(rec["genome"], vidx, data, dirs, with_folds=True)
    tr = rec
    retention = (st["icir"] / tr["icir"]) if tr["icir"] > 0 else -1.0
    ok = (tr["ic"] > 0.02 and st["ic"] > 0.02 and tr["t"] > 2.0 and st["t"] > 1.5
          and retention >= 0.45 and rec["n_active"] <= MAX_ACTIVE)
    # v2 selection: validation dominance + penalise train->valid degradation.
    # This was changed BEFORE OOS after v1 ranked a 5-factor random fit first
    # despite its ICIR collapsing from 5.51 to 1.80.
    combined = (0.25 * tr["icir"] + 0.55 * st["icir"]
                + 2.0 * max(0.0, min(1.0, retention))
                - 0.03 * rec["n_active"]
                - 0.3 * (st["turnover_top1700"] if np.isfinite(st["turnover_top1700"]) else 0))
    return dict(genome=rec["genome"], generation=rec["generation"], train=tr, valid=st,
                retention=float(retention), combined=float(combined), gate_pass=bool(ok))


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
