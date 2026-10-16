# -*- coding: utf-8 -*-
"""示例策略：都只使用 t 日及之前的数据，等权持有 top-N。

    w = strategy.weights(t, md)      # 返回 t 日收盘后决定的目标权重（t+1 日开盘执行）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.lib import MarketData


class Strategy:
    """基类：截面打分 -> 选前 n -> 等权（或按分数加权）。"""

    name = "strategy"
    weight_by = "equal"          # equal | signal | inv_vol

    def __init__(self, n_hold: int = 50, max_weight: float = 0.05,
                 min_listed_days: int = 250, min_adv: float = 2e7,
                 price_min: float = 2.0, boards=None, exclude_st: bool = True,
                 lag: int = 0):
        self.n_hold = int(n_hold)
        self.max_weight = float(max_weight)
        self.lag = int(lag)      # 额外延迟，用于检验信号时效
        self.buffer_mult = 0.0   # 缓冲带倍数：0 = 无缓冲（每次都完全重排）
        self._prev = []          # 上一期持仓（用于缓冲带，只依赖历史，无未来信息）
        self._uni_kw = dict(min_listed_days=min_listed_days, min_adv=min_adv,
                            price_min=price_min, boards=boards, exclude_st=exclude_st)
        self._uni = None

    # ------------------------------------------------------------------ #
    def universe_row(self, md: MarketData, t: int) -> np.ndarray:
        if self._uni is None:
            self._uni = md.universe(**self._uni_kw).values
        return self._uni[t]

    def score(self, md: MarketData, t: int) -> np.ndarray:
        """截面打分（越大越买）。子类实现 signal() 或直接覆盖 score()。"""
        return np.asarray(self.signal(md, t), dtype="float64")

    def signal(self, md: MarketData, t: int) -> np.ndarray:
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    def _select(self, ok: np.ndarray, score: np.ndarray) -> np.ndarray:
        """按打分选股，支持**缓冲带**：已持有的标的要跌出 exit_rank 才卖。

        缓冲带能显著降低换手：只要老持仓还在前 n x buffer_mult 名内就继续留着，
        腾出来的名额才给新进入前 n 名的标的。buffer_mult=0 表示无缓冲（完全重排）。
        """
        cand = np.flatnonzero(ok)
        if len(cand) == 0:
            self._prev = []
            return cand
        order = cand[np.argsort(-score[cand], kind="stable")]
        rank = np.full(len(score), 10 ** 9, dtype="int64")
        rank[order] = np.arange(len(order))
        n = min(self.n_hold, len(order))
        prev = self._prev if (self.buffer_mult and self._prev) else None
        if prev is None:
            pick = order[:n]
        else:
            exit_rank = min(max(int(round(n * self.buffer_mult)), n + 1), len(order))
            keep = [j for j in prev if ok[j] and rank[j] < exit_rank]
            pick_set = set(keep)
            for j in order:
                if len(pick_set) >= n:
                    break
                pick_set.add(int(j))
            pick = np.array(sorted(pick_set), dtype="int64")
        self._prev = pick.tolist()
        return pick

    def weights(self, t: int, md: MarketData) -> pd.Series:
        t = int(t) - self.lag
        if t < 1:
            return pd.Series(dtype="float64")
        sig = np.asarray(self.score(md, t), dtype="float64")
        ok = self.universe_row(md, t) & np.isfinite(sig)
        pick = self._select(ok, np.where(np.isfinite(sig), sig, -np.inf))
        k = len(pick)
        if k == 0:
            return pd.Series(dtype="float64")
        w = np.zeros(len(md.symbols))
        if self.weight_by == "equal":
            w[pick] = 1.0 / k
        elif self.weight_by == "signal":
            s = sig[pick] - sig[pick].min() + 1e-9
            w[pick] = s / s.sum()
        elif self.weight_by == "inv_vol":
            vol = md.ret.values[max(t - 60, 0):t + 1][:, pick].std(0)
            iv = 1.0 / np.where(vol > 0, vol, np.nan)
            iv = np.nan_to_num(iv, nan=0.0)
            w[pick] = iv / iv.sum() if iv.sum() > 0 else 1.0 / k
        else:
            raise ValueError(self.weight_by)
        if self.max_weight > 0:
            w = np.minimum(w, self.max_weight)
        return pd.Series(w[w > 0], index=md.symbols[w > 0])

    def __repr__(self):
        return "%s(n_hold=%d)" % (type(self).__name__, self.n_hold)


def _cum_return(md: MarketData, t: int, lb: int, skip: int = 0) -> np.ndarray:
    """过去 lb 日累计复权收益（可选跳过最近 skip 日）。"""
    a = md.adj_close.values
    j = t - skip
    i = j - lb
    if i < 0:
        return np.full(a.shape[1], np.nan)
    return a[j] / a[i] - 1.0


class Momentum(Strategy):
    """截面动量：过去 lookback 日涨幅最高的 n 只（可跳过最近 skip 日）。"""

    name = "动量"

    def __init__(self, lookback: int = 20, skip: int = 0, **kw):
        super().__init__(**kw)
        self.lookback = int(lookback)
        self.skip = int(skip)
        self.name = "动量%d%s" % (lookback, "" if skip == 0 else "_跳%d" % skip)

    def signal(self, md, t):
        return _cum_return(md, t, self.lookback, self.skip)


class Reversal(Strategy):
    """短期反转：过去 lookback 日跌幅最大的 n 只。"""

    name = "反转"

    def __init__(self, lookback: int = 5, **kw):
        super().__init__(**kw)
        self.lookback = int(lookback)
        self.name = "反转%d" % lookback

    def signal(self, md, t):
        return -_cum_return(md, t, self.lookback)


class LowVol(Strategy):
    """低波动：过去 lookback 日日收益标准差最小的 n 只。"""

    name = "低波"

    def __init__(self, lookback: int = 60, **kw):
        super().__init__(**kw)
        self.lookback = int(lookback)
        self.name = "低波%d" % lookback

    def signal(self, md, t):
        r = md.ret.values[max(t - self.lookback + 1, 0):t + 1]
        if r.shape[0] < self.lookback:
            return np.full(md.ret.shape[1], np.nan)
        with np.errstate(invalid="ignore"):
            v = np.nanstd(r, axis=0, ddof=1)
        return -v


class Illiquidity(Strategy):
    """非流动性：20 日均额最小的 n 只（Amihud 类逻辑的规模代理）。"""

    name = "低流动性"

    def __init__(self, lookback: int = 20, **kw):
        super().__init__(**kw)
        self.lookback = int(lookback)
        self.name = "低流动性%d" % lookback

    def signal(self, md, t):
        a = md.amt.values[max(t - self.lookback + 1, 0):t + 1]
        if a.shape[0] < self.lookback:
            return np.full(md.amt.shape[1], np.nan)
        with np.errstate(invalid="ignore"):
            m = np.nanmean(a, axis=0)
        return -np.log(np.where(m > 0, m, np.nan))


class EqualWeightUniverse(Strategy):
    """基准：等权持有当期全部可交易股票池。"""

    name = "等权全市场"

    def __init__(self, cap: int = 10 ** 9, **kw):
        kw.setdefault("max_weight", 0.0)      # 不设单票上限
        kw.setdefault("min_adv", 2e7)
        super().__init__(n_hold=cap, **kw)    # cap 足够大 -> 选全部股票池成分

    def signal(self, md, t):
        return np.zeros(md.close.shape[1])


# --------------------------------------------------------------------------- #
# 波动率 + 成交量因子组合
# --------------------------------------------------------------------------- #
from core.factors import Factors, cs_zscore, DESC as FACTOR_DESC   # noqa: E402


class FactorPortfolio(Strategy):
    """单因子组合：sign=+1 买打分高的，sign=-1 买打分低的。"""

    def __init__(self, factor: str = "ivol", sign: int = -1, n_hold: int = 50,
                 buffer_mult: float = 2.0, **kw):
        super().__init__(n_hold=n_hold, **kw)
        self.factor = factor
        self.sign = int(sign)
        self.buffer_mult = float(buffer_mult)
        self._fac = None
        self.name = "%s%s" % (factor, "" if sign > 0 else "_neg")

    def score(self, md, t):
        if self._fac is None:
            self._fac = Factors(md)
        return self.sign * np.asarray(self._fac.get(self.factor, t), dtype="float64")


class CompositePortfolio(Strategy):
    """多因子等权 z-score 合成：score = sum_i sign_i * zscore(factor_i)。

    signs 未给出时从因子字典里取默认方向（见 VOLUME_DEFAULT_SIGNS）。
    """

    def __init__(self, factors: list, signs: dict | None = None, n_hold: int = 50,
                 buffer_mult: float = 2.0, winsor: bool = True, **kw):
        super().__init__(n_hold=n_hold, **kw)
        self.factors = list(factors)
        self.signs = dict(signs or {})
        self.buffer_mult = float(buffer_mult)
        self.winsor = winsor
        self._fac = None
        self.name = "composite%d" % len(self.factors)

    def score(self, md, t):
        if self._fac is None:
            self._fac = Factors(md)
        total = None
        for f in self.factors:
            x = np.asarray(self._fac.get(f, t), dtype="float64")
            z = cs_zscore(x) if self.winsor else x
            s = float(self.signs.get(f, -1.0))
            total = s * z if total is None else total + s * z
        return total if total is not None else np.full(len(md.symbols), np.nan)


#: 先验方向（-1 = 买低值）。最终方向由样本内 IC 决定，见 s04。
VOLUME_DEFAULT_SIGNS = {
    "ivol": -1, "beta": -1, "dvol": -1, "maxdd": -1, "skew": -1, "kurt": -1,
    "vol_trend": -1, "vol_of_vol": -1, "amt_trend": -1, "amt_cv": -1,
    "amihud": +1, "trade_size": +1, "pv_corr": -1,
}


class PanelStrategy(Strategy):
    """直接消费**预先算好的打分面板**（日期 x 证券），用于参数扫描。

    因子只在调仓日算一次（十几毫秒），之后所有配置共用同一块面板，
    这样跑几十组参数敏感性时不需要重算因子。
    """

    def __init__(self, panel: pd.DataFrame, name: str = "panel", n_hold: int = 50,
                 buffer_mult: float = 2.0, **kw):
        super().__init__(n_hold=n_hold, **kw)
        self._mat = np.asarray(panel.values, dtype="float64")
        self._pdates = np.asarray(panel.index.values, dtype="datetime64[ns]")
        self._rowmap = {d: i for i, d in enumerate(panel.index)}
        self.name = name
        self.buffer_mult = float(buffer_mult)

    def score(self, md, t):
        """取**最近一期不晚于 t** 的面板值。

        面板通常只铺在月频网格上；若引擎按周频/日频调仓，非网格日应沿用
        最近一期的信号（而不是变成空仓）。这样「提高调仓频率」比较的才是
        执行频率本身，而不是信号有没有取到。
        """
        d = np.datetime64(md.dates[t])
        i = int(np.searchsorted(self._pdates, d, side="right")) - 1
        if i < 0:
            return np.full(len(md.symbols), np.nan)
        return self._mat[i]

    def clone(self, **kw):
        p = dict(n_hold=self.n_hold, buffer_mult=self.buffer_mult, name=self.name)
        p.update(kw)
        return PanelStrategy(self._panel_ref, **p)


def build_factor_panels(md, names, ts, cache_dir=None, verbose=False):
    """在给定日期下标 ts 上把所有因子铺成面板，返回 (原始面板, z-score 面板)。

    结果缓存到 cache_dir（parquet），参数扫描时可反复读取。
    """
    import os
    import time
    raw, zsc = {}, {}
    todo = list(names)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        for nm in list(todo):
            p = os.path.join(cache_dir, "f_%s.parquet" % nm)
            if os.path.exists(p):
                df = pd.read_parquet(p)
                raw[nm] = df
                zsc[nm] = df.apply(cs_zscore, axis=1, result_type="broadcast") \
                    if False else pd.DataFrame(
                        np.vstack([cs_zscore(r) for r in df.values]),
                        index=df.index, columns=df.columns)
                todo.remove(nm)
                if verbose:
                    print("  cache hit %s" % nm)
    if todo:
        F = Factors(md)
        dates = md.dates[list(ts)]
        for nm in todo:
            t0 = time.time()
            arr = np.vstack([np.asarray(F.get(nm, t), dtype="float64") for t in ts])
            df = pd.DataFrame(arr, index=dates, columns=md.symbols)
            raw[nm] = df
            zsc[nm] = pd.DataFrame(np.vstack([cs_zscore(r) for r in arr]),
                                   index=dates, columns=md.symbols)
            if cache_dir:
                df.to_parquet(os.path.join(cache_dir, "f_%s.parquet" % nm))
            if verbose:
                print("  %-11s %5.1fs" % (nm, time.time() - t0))
    return raw, zsc
