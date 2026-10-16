# -*- coding: utf-8 -*-
"""因子计算的共享上下文：一次把面板转成 numpy，因子只做切片运算。

**所有因子只能通过 Ctx 访问数据，且只允许使用下标 <= t 的行。**
`Ctx.truncated(t)` 会把所有数组截到 [:t+1]，注册表用它做因果性检验：
若 f(t, Ctx_full) != f(t, Ctx_truncated)，说明该因子偷看了未来数据。

约定
----
- 收益、复权价一律用复权口径（C / R / ADJ_*）
- 涨跌停、一字板判断才用未复权口径（RAW / O / H / L）
- 成交额单位是元；num_trades 单位是笔
- 所有统计量请用 Ctx.mean / Ctx.std 这两个掩码版本，
  不要用 np.nanmean（全 NaN 列会告警且慢）
"""
from __future__ import annotations

import numpy as np

#: 需要随 truncated() 一起截断的数组
_ARRAYS = ("R", "C", "RAW", "O", "H", "L", "A", "NT", "ADJ_O", "ADJ_H", "ADJ_L",
           "listed", "mkt", "mkt_fill", "VWAP_PROXY")


class Ctx:
    def __init__(self, md):
        self.md = md
        f = np.asarray(md.cumadj_clean.values, dtype="float64")
        self.R = np.asarray(md.ret.values, dtype="float64")        # 复权日收益
        self.C = np.asarray(md.adj_close.values, dtype="float64")   # 复权收盘
        self.RAW = np.asarray(md.close.values, dtype="float64")     # 未复权收盘
        self.O = np.asarray(md.open.values, dtype="float64")
        self.H = np.asarray(md.high.values, dtype="float64")
        self.L = np.asarray(md.low.values, dtype="float64")
        self.ADJ_O = self.O * f
        self.ADJ_H = self.H * f
        self.ADJ_L = self.L * f
        self.A = np.asarray(md.amt.values, dtype="float64")         # 成交额（元）
        self.NT = np.asarray(md.num_trades.values, dtype="float64")  # 成交笔数
        self.listed = np.asarray(md.listed.values, dtype=bool)
        self.mkt = np.asarray(md.ret_ok.mean(1).values, dtype="float64")   # 等权市场收益
        self.mkt_fill = np.nan_to_num(self.mkt)
        #: 代理 VWAP = 成交额 / 估算成交量；本数据集**没有成交量**，
        #: 这里用 (H+L+C)/3 作为当日均价代理，仅供形态类因子使用
        self.VWAP_PROXY = (self.H + self.L + self.RAW) / 3.0
        self.n, self.k = self.R.shape

    # ---------------------------------------------------------------- #
    def truncated(self, t: int) -> "Ctx":
        """返回只保留前 t+1 行的上下文（因果性检验用）。"""
        c = object.__new__(Ctx)
        c.md = self.md
        for a in _ARRAYS:
            setattr(c, a, getattr(self, a)[:int(t) + 1])
        c.n, c.k = int(t) + 1, self.k
        return c

    # ---------------------------------------------------------------- #
    @staticmethod
    def sl(t: int, w: int, X: np.ndarray):
        """取 [t-w+1, t] 窗口。返回 (起始行, 子矩阵)。"""
        lo = max(int(t) - int(w) + 1, 0)
        return lo, X[lo:int(t) + 1]

    @staticmethod
    def mean(X, ok, min_cnt):
        """掩码均值：ok 为 False 的位置不参与，有效数 < min_cnt 返回 NaN。"""
        cnt = ok.sum(0).astype("float64")
        mu = np.where(ok, X, 0.0).sum(0) / np.maximum(cnt, 1.0)
        return np.where(cnt >= min_cnt, mu, np.nan)

    @staticmethod
    def std(X, ok, min_cnt, ddof=1):
        cnt = ok.sum(0).astype("float64")
        mu = np.where(ok, X, 0.0).sum(0) / np.maximum(cnt, 1.0)
        c = np.where(ok, X - mu, 0.0)
        var = (c ** 2).sum(0) / np.maximum(cnt - ddof, 1.0)
        return np.where(cnt >= min_cnt, np.sqrt(var), np.nan)

    def stats(self, t, w, X=None, min_frac=0.6):
        """一次拿到窗口内的 (中心化矩阵, 有效计数, 均值, 方差, 坏列掩码)。"""
        X = self.R if X is None else X
        lo, W = self.sl(t, w, X)
        ok = np.isfinite(W)
        cnt = ok.sum(0).astype("float64")
        mu = np.where(ok, W, 0.0).sum(0) / np.maximum(cnt, 1.0)
        Wc = np.where(ok, W - mu, 0.0)
        var = (Wc ** 2).sum(0) / np.maximum(cnt - 1.0, 1.0)
        bad = cnt < max(3, int(min_frac * w))
        return dict(W=W, Wc=Wc, ok=ok, cnt=cnt, mu=mu, var=var, bad=bad, lo=lo)


def cs_zscore(x) -> np.ndarray:
    """横截面标准化（2%/98% 去极值 + z-score）；常数序列返回 0 而不是 NaN。"""
    v = np.asarray(x, dtype="float64")
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return np.full_like(v, np.nan)
    lo, hi = np.percentile(v[ok], [2, 98])
    v = np.clip(v, lo, hi)
    m = v[ok].mean()
    s = v[ok].std()
    out = np.full_like(v, np.nan)
    if s > 0:
        out[ok] = (v[ok] - m) / s
    else:
        out[ok] = 0.0
    return out
