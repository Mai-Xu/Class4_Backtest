# -*- coding: utf-8 -*-
"""把 factor_zoo 的因子铺成面板（与原 strategies.build_factor_panels 并存、互不影响）。

为什么单独写一份：`strategies.build_factor_panels` 只认 `Factors` 类里那 13 个内置因子
（签名 `f(t, **kw)`），而 zoo 因子的签名是 `f(t, md, C)`。为了不破坏正在被
`core/strategies.py` 使用的共享函数，这里单独实现。

缓存策略：文件名里带上因子所在模块的 mtime，模块一改缓存自动失效，
不会出现「改了因子但还在用旧面板」的隐蔽 bug。
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from .ctx import Ctx, cs_zscore


def _stamp(spec) -> int:
    """用因子所在模块文件的修改时间做缓存指纹。"""
    try:
        mod = __import__("factor_zoo." + spec["module"], fromlist=["x"])
        return int(os.path.getmtime(mod.__file__))
    except Exception:                                  # noqa: BLE001
        return 0


def build_zoo_panels(md, C: Ctx, specs: dict, ts, cache_dir=None, verbose=False):
    """在日期下标 ts 上把 specs 里的因子铺成 (raw, zsc) 两个 {因子名: DataFrame}。

    raw 是因子原值；zsc 是逐日横截面 z-score（2%/98% 去极值）。
    """
    raw, zsc = {}, {}
    todo = []
    for nm, spec in specs.items():
        p = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            p = os.path.join(cache_dir, "zoo_%s_%d.parquet" % (nm, _stamp(spec)))
        if p and os.path.exists(p):
            df = pd.read_parquet(p)
            raw[nm] = df
            zsc[nm] = _zpanel(df)
            if verbose:
                print("  命中缓存 %s" % nm)
        else:
            todo.append((nm, spec, p))
    if todo:
        dates = md.dates[list(ts)]
        for nm, spec, p in todo:
            t0 = time.time()
            f = spec["func"]
            arr = np.vstack([np.asarray(f(int(t), md, C), dtype="float64") for t in ts])
            arr[~np.isfinite(arr)] = np.nan
            df = pd.DataFrame(arr, index=dates, columns=md.symbols)
            raw[nm] = df
            zsc[nm] = _zpanel(df)
            if p:
                try:
                    df.to_parquet(p)
                except Exception:                      # noqa: BLE001
                    pass
            if verbose:
                print("  %-24s %5.1fs  覆盖 %.1f%%"
                      % (nm, time.time() - t0, 100 * np.isfinite(arr).mean()))
    return raw, zsc


def _zpanel(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.vstack([cs_zscore(r) for r in df.values]),
                        index=df.index, columns=df.columns)

# --------------------------------------------------------------------------- #
# 并行铺面板（300+ 因子时用；与上面的单进程版本并存，互不影响）
# --------------------------------------------------------------------------- #
_W = {}          # 子进程内的全局：MarketData / Ctx / registry


def _worker_init():
    """子进程初始化：只加载一次数据与上下文（约 1.8GB RSS/进程）。"""
    from core.lib import MarketData
    from . import registry
    _W["md"] = MarketData()
    _W["C"] = Ctx(_W["md"])
    _W["reg"] = registry.load_all(strict=False)


def _worker_one(args):
    """(因子名, ts) -> (因子名, float32 面板数组)。

    ts 随任务传递：spawn 出来的子进程无法继承父进程的全局变量。
    单个因子抛异常时返回全 NaN 面板，不炸整条流水线（异常会记录在报告里）。
    """
    name, ts = args
    spec = _W["reg"][name]
    f = spec["func"]
    md, C = _W["md"], _W["C"]
    arr = np.empty((len(ts), C.k), dtype="float32")
    for i, t in enumerate(ts):
        try:
            v = np.asarray(f(int(t), md, C), dtype="float64")
            if v.shape != (C.k,):
                v = np.full(C.k, np.nan)
        except Exception:                      # noqa: BLE001
            v = np.full(C.k, np.nan)
        arr[i] = v
    return name, arr


def build_panels_parallel(md, specs: dict, ts, cache_dir=None, workers: int = 6,
                          verbose: bool = True, force: bool = False):
    """并行把 specs 里的因子铺成 {因子名: DataFrame(index=日期, columns=证券, float32)}。

    缓存：`<cache_dir>/zoo2_<因子名>_<模块mtime>.parquet`；模块一改缓存自动失效。
    workers=1 时在本进程内直接算（调试用）。
    """
    import multiprocessing as mp

    ts = [int(t) for t in ts]
    dates = md.dates[ts]
    out, todo = {}, []
    for nm, spec in specs.items():
        p = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            p = os.path.join(cache_dir, "zoo2_%s_%d.parquet" % (nm, _stamp(spec)))
        if p and os.path.exists(p) and not force:
            try:
                out[nm] = pd.read_parquet(p)
                continue
            except Exception:                  # noqa: BLE001  坏缓存就重算
                pass
        todo.append((nm, p))
    if verbose:
        print("  面板：命中缓存 %d 个，待算 %d 个" % (len(out), len(todo)))
    if not todo:
        return out

    def _save(nm, arr, p):
        df = pd.DataFrame(np.asarray(arr, dtype="float32"), index=dates, columns=md.symbols)
        df.index.name = "date"
        if p:
            try:
                df.to_parquet(p)
            except Exception:                  # noqa: BLE001
                pass
        return df

    pmap = {nm: p for nm, p in todo}
    if workers and workers > 1 and len(todo) > 1:
        ctx = mp.get_context("spawn")
        jobs = [(nm, ts) for nm in pmap]
        with ctx.Pool(processes=int(workers), initializer=_worker_init) as pool:
            done = 0
            for nm, arr in pool.imap_unordered(_worker_one, jobs, chunksize=1):
                out[nm] = _save(nm, arr, pmap[nm])
                done += 1
                if verbose and (done % 20 == 0 or done == len(jobs)):
                    print("    已算 %d/%d" % (done, len(jobs)))
    else:
        _W["md"], _W["C"], _W["reg"] = md, Ctx(md), specs
        for nm, p in todo:
            _, arr = _worker_one((nm, ts))
            out[nm] = _save(nm, arr, p)
    return out
