# -*- coding: utf-8 -*-
"""【模板】复制这个文件来写一个新分类。注意：本文件名以 _ 开头，不会被注册表加载。

规则（务必逐条遵守）
--------------------
1. 只允许使用下标 <= t 的数据。`Ctx.truncated(t)` 会做因果性检验。
2. 因子函数签名固定为 fn(t, md, C) -> np.ndarray(shape=(C.k,))，dtype=float64。
3. 数组里不要出现 inf；无法计算的位置返回 NaN。
4. 统计量用 C.mean / C.std（掩码版），窗口用 C.sl / C.stats。
5. 因子名加本分类前缀，避免与其他分类重名。
6. 别 import numpy/pandas 以外的东西（也不要 import lib / engine，太慢）。
"""
from __future__ import annotations

import numpy as np

from .ctx import Ctx, cs_zscore


# ------------------------------------------------------------------ #
def _ma_dev(t: int, C: Ctx, window: int = 20) -> np.ndarray:
    """示例：收盘价对 N 日均线的偏离（演示窗口/掩码写法）。"""
    lo, W = C.sl(t, window, C.C)                 # W = 窗口内的复权收盘价 (w, k)
    ok = np.isfinite(W)
    ma = C.mean(W, ok, min_cnt=max(3, window // 2))
    px = C.C[t]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ma > 0, px / ma - 1.0, np.nan)


# ------------------------------------------------------------------ #
FACTORS = {
    "xx_madev20": dict(func=lambda t, md, C: _ma_dev(t, C, 20),
                       desc="收盘价 / MA20 - 1", direction=-1),
}
