# -*- coding: utf-8 -*-
"""v2 独立复核 · 被测对象执行器（唯一加载 factor_zoo 因子模块的文件）。

复核的参考实现在 v2_lib.py，完全不依赖被测模块；但验证被测模块本身
（投毒因果性、数值比对）必须执行被测代码，这里用 importlib 取各模块公开的
FACTORS 字典，只调用 spec["func"]（黑盒调用），不复用其内部函数，
也不复用 factor_zoo/registry.check_one。
"""
from __future__ import annotations

import copy
import importlib

import numpy as np
import pandas as pd

MODULES = ("reversal", "risk2", "liquidity2", "flow2", "chip", "trend2",
           "pattern", "calendar", "interaction", "marketrel")

FIELDS = ("open", "high", "low", "close", "amt", "num_trades", "cumadj")
#: Ctx.__init__ 实际读取的全部 md 属性（投毒时逐一同款处理）
CTX_KEYS = ("open", "high", "low", "close", "amt", "num_trades", "cumadj",
            "cumadj_clean", "adj_close", "ret", "listed", "ret_ok")


def load_modules(names=MODULES):
    """{模块名: FACTORS 字典}（只取公开接口）。"""
    out = {}
    for nm in names:
        mod = importlib.import_module("factor_zoo." + nm)
        out[nm] = getattr(mod, "FACTORS")
    return out


def poison_md(md, t, mode="nan", seed=20261022):
    """构造「t+1 行及以后全部改成 NaN / 随机值」的 md 副本。

    做法：把 Ctx 读取的全部面板（原始 7 字段 + cumadj_clean / adj_close /
    ret / listed / ret_ok）复制后覆盖 t+1: 行，再交给 Ctx 重新建上下文。
    这样 <= t 的所有元素与干净副本逐位相同，> t 的元素变成垃圾值；
    任何只读 <= t 的因子必须给出逐位相同的输出。
    """
    if mode not in ("copy", "nan", "rand"):
        raise ValueError(mode)
    mdp = copy.copy(md)
    cache = {}
    rng = np.random.default_rng(int(seed) + int(t))
    for k in CTX_KEYS:
        df = md.field(k) if k in FIELDS else getattr(md, k)
        arr = np.asarray(df.values)
        if mode == "copy":
            a = (arr.copy() if arr.dtype == bool
                 else np.asarray(arr, dtype="float64").copy())
        elif arr.dtype == bool:
            a = arr.copy()
            if mode == "nan":
                a[int(t) + 1:] = False
            else:
                a[int(t) + 1:] = rng.random(a[int(t) + 1:].shape) > 0.5
        else:
            a = np.asarray(arr, dtype="float64").copy()
            if mode == "nan":
                a[int(t) + 1:] = np.nan
            else:
                head = a[: int(t) + 1]
                mu = np.nanmean(head, axis=0)
                sd = np.nanstd(head, axis=0)
                shape = a[int(t) + 1:].shape
                r = rng.standard_normal(shape)
                scale = np.where(np.isfinite(sd), sd, 0.0)[None, :]
                vals = np.where(np.isfinite(mu)[None, :], mu[None, :] + scale * r, np.nan)
                if k in ("open", "high", "low", "close", "amt", "num_trades", "cumadj",
                         "cumadj_clean", "adj_close"):
                    vals = np.abs(vals)
                a[int(t) + 1:] = vals
        cache[k] = pd.DataFrame(a, index=df.index, columns=df.columns)
    mdp._cache = cache
    mdp._ratio_table = None
    mdp._st_like = None
    return mdp


def compare(a, b):
    """逐位比较两个因子输出。返回 dict。"""
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    rec = dict(shape_a=list(a.shape), shape_b=list(b.shape), shape_ok=bool(a.shape == b.shape))
    if not rec["shape_ok"]:
        rec.update(identical=False, n_mismatch=int(max(a.size, b.size)), max_abs=float("inf"),
                   max_rel=float("inf"), nan_a=int(np.isnan(a).sum()), nan_b=int(np.isnan(b).sum()))
        return rec
    fa = np.isfinite(a)
    fb = np.isfinite(b)
    mask_mismatch = int((fa != fb).sum())
    both = fa & fb
    if both.any():
        d = np.abs(a[both] - b[both])
        max_abs = float(d.max())
        rel = d / np.maximum(np.abs(b[both]), 1e-12)
        max_rel = float(rel.max())
        neq = int((a[both] != b[both]).sum())
    else:
        max_abs = 0.0
        max_rel = 0.0
        neq = 0
    rec.update(identical=bool(neq == 0 and mask_mismatch == 0),
               n_mismatch=int(neq + mask_mismatch),
               n_value_neq=neq, n_nan_mask_mismatch=mask_mismatch,
               max_abs=max_abs, max_rel=max_rel,
               n_valid=int(both.sum()), nan_a=int((~fa).sum()), nan_b=int((~fb).sum()))
    return rec