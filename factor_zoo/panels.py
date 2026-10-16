# -*- coding: utf-8 -*-
"""把 factor_zoo 的因子铺成面板（与原 strategies.build_factor_panels 并存、互不影响）。

为什么单独写一份：`strategies.build_factor_panels` 只认 `Factors` 类里那 13 个内置因子
（签名 `f(t, **kw)`），而 zoo 因子的签名是 `f(t, md, C)`。为了不破坏正在被
`s07_factor_lab.py` 使用的共享函数，这里单独实现。

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
