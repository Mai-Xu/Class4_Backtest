# -*- coding: utf-8 -*-
"""因子库：波动率族 + 成交量族，全部**只用 t 日及之前**的数据。

设计
----
每个因子都是 `f(t) -> np.ndarray(len(symbols))`，逐日现算，不预计算整块面板：
调仓只有几十次，现算比铺满 1643 x 5451 快两个数量级，也天然避免未来函数。

缺失值处理不用 np.nanmean（对全 NaN 列会告警），而是先掩码再求和，
所有统计量都要求窗口内有效样本数 >= min_obs。

因子清单
--------
波动率族
    ivol        特质波动率：市场模型残差波动（对等权市场收益回归取残差）
    beta        市场模型 beta
    dvol        下行波动（半偏差，只统计负收益）
    maxdd       区间最大回撤 max(1 - close / cummax(close))
    skew        收益偏度
    kurt        收益峰度
    vol_trend   波动率变化：短窗波动 / 长窗波动 - 1
    vol_of_vol  波动率的不稳定性：滚动波动的标准差
成交量族
    amt_trend   量能放大：短窗均额 / 长窗均额 - 1
    amt_cv      成交量变异系数：std(amt) / mean(amt)
    amihud      Amihud 非流动性：mean(|r| / amt)
    trade_size  单笔均额（amt / num_trades），机构参与度代理
    pv_corr     量价相关：corr(收益, 成交量对数变化)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.lib import MarketData

FACTORS = ("ivol", "beta", "dvol", "maxdd", "skew", "kurt", "vol_trend", "vol_of_vol",
           "amt_trend", "amt_cv", "amihud", "trade_size", "pv_corr")

DESC = {
    "ivol": "特质波动率（市场模型残差）",
    "beta": "市场模型 beta",
    "dvol": "下行波动（半偏差）",
    "maxdd": "区间最大回撤",
    "skew": "收益偏度",
    "kurt": "收益峰度",
    "vol_trend": "波动率变化（短/长 - 1）",
    "vol_of_vol": "波动率的不稳定性",
    "amt_trend": "量能放大（短/长 - 1）",
    "amt_cv": "成交量变异系数",
    "amihud": "Amihud 非流动性",
    "trade_size": "单笔均额（机构参与度）",
    "pv_corr": "量价相关",
}


def cs_zscore(x: np.ndarray) -> np.ndarray:
    """横截面标准化（去极值 + z-score），返回同长度数组。"""
    v = np.asarray(x, dtype="float64")
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return np.full_like(v, np.nan)
    lo, hi = np.nanpercentile(v[ok], [2, 98])
    v = np.clip(v, lo, hi)
    m = np.nanmean(v[ok])
    s = np.nanstd(v[ok])
    out = np.full_like(v, np.nan)
    if s > 0:
        out[ok] = (v[ok] - m) / s
    else:
        out[ok] = 0.0        # 常数因子没有横截面信息：给中性值 0
    return out              # （返回 NaN 会顺着多因子求和传染，把组合打成空仓）


class Factors:
    def __init__(self, md: MarketData, mkt_window: int = 1):
        self.md = md
        self.R = np.asarray(md.ret.values, dtype="float64")        # 复权收益
        self.C = np.asarray(md.adj_close.values, dtype="float64")  # 复权价
        self.A = np.asarray(md.amt.values, dtype="float64")
        self.NT = np.asarray(md.num_trades.values, dtype="float64")
        self.listed = np.asarray(md.listed.values)
        self.n, self.k = self.R.shape
        self._mkt = None

    # ------------------------------------------------------------------ #
    @property
    def mkt(self) -> np.ndarray:
        """市场收益代理 = 全市场等权收益（ret_ok 的横截面均值）。"""
        if self._mkt is None:
            self._mkt = np.nan_to_num(np.asarray(self.md.ret_ok.mean(1).values,
                                                 dtype="float64"))
        return self._mkt

    def _slice(self, t: int, w: int, X: np.ndarray):
        lo = max(int(t) - int(w) + 1, 0)
        return lo, X[lo:int(t) + 1]

    @staticmethod
    def _mask_stats(X: np.ndarray):
        """返回 (掩码后中心化矩阵, 有效计数, 均值)。NaN 位置贡献 0。"""
        ok = np.isfinite(X)
        cnt = ok.sum(0).astype("float64")
        Xf = np.where(ok, X, 0.0)
        mu = Xf.sum(0) / np.maximum(cnt, 1.0)
        Xc = np.where(ok, X - mu, 0.0)
        return Xc, cnt, mu

    @staticmethod
    def _mean_ok(X, ok, min_cnt):
        """掩码均值：完全避开 np.nanmean 在全 NaN 列上的告警与开销。"""
        cnt = ok.sum(0).astype("float64")
        mu = np.where(ok, X, 0.0).sum(0) / np.maximum(cnt, 1.0)
        return np.where(cnt >= min_cnt, mu, np.nan)

    @staticmethod
    def _std_ok(X, ok, min_cnt):
        cnt = ok.sum(0).astype("float64")
        mu = np.where(ok, X, 0.0).sum(0) / np.maximum(cnt, 1.0)
        c = np.where(ok, X - mu, 0.0)
        var = (c ** 2).sum(0) / np.maximum(cnt - 1.0, 1.0)
        return np.where(cnt >= min_cnt, np.sqrt(var), np.nan)

    # ------------------------------------------------------------------ #
    # 波动率族
    # ------------------------------------------------------------------ #
    def _moments(self, t, w, min_obs=None, with_mkt=False):
        lo, X = self._slice(t, w, self.R)
        min_obs = min_obs or max(5, int(0.6 * w))
        Xc, cnt, mu = self._mask_stats(X)
        bad = cnt < min_obs
        n_eff = np.maximum(cnt - 1.0, 1.0)
        var = (Xc ** 2).sum(0) / n_eff
        out = dict(Xc=Xc, cnt=cnt, mu=mu, var=var, bad=bad, n_eff=n_eff, lo=lo, X=X)
        if with_mkt:
            m = self.mkt[lo:t + 1]
            mo = np.isfinite(m)
            mf = np.where(mo, m, 0.0)
            mbar = mf.sum() / max(mo.sum(), 1)
            mc = np.where(mo, m - mbar, 0.0)
            out.update(mc=mc, mvar=(mc ** 2).sum() / max(mo.sum() - 1, 1))
        return out

    def vol(self, t, window=60, min_obs=None) -> np.ndarray:
        """已实现波动（年化前，日频标准差）。"""
        s = self._moments(t, window, min_obs)
        v = np.sqrt(s["var"])
        return np.where(s["bad"], np.nan, v)

    def ivol(self, t, window=60, min_obs=None) -> np.ndarray:
        """特质波动率：市场模型残差的标准差。

        beta_i = cov(r_i, r_m) / var(r_m)，残差 = r_i - beta_i * r_m（含截距项，
        用去均值后的序列做，等价于 OLS）。
        """
        s = self._moments(t, window, min_obs, with_mkt=True)
        cov = (s["Xc"] * s["mc"][:, None]).sum(0) / s["n_eff"]
        beta = cov / max(s["mvar"], 1e-12)
        resid = s["Xc"] - beta[None, :] * s["mc"][:, None]
        iv = np.sqrt((resid ** 2).sum(0) / s["n_eff"])
        return np.where(s["bad"], np.nan, iv)

    def beta(self, t, window=60, min_obs=None) -> np.ndarray:
        s = self._moments(t, window, min_obs, with_mkt=True)
        cov = (s["Xc"] * s["mc"][:, None]).sum(0) / s["n_eff"]
        b = cov / max(s["mvar"], 1e-12)
        return np.where(s["bad"], np.nan, b)

    def dvol(self, t, window=60, min_obs=None) -> np.ndarray:
        """下行波动：只统计负收益的半偏差 sqrt(sum(min(r,0)^2)/n)。"""
        s = self._moments(t, window, min_obs)
        neg = np.minimum(s["Xc"], 0.0)
        d = np.sqrt((neg ** 2).sum(0) / s["n_eff"])
        return np.where(s["bad"], np.nan, d)

    def maxdd(self, t, window=60, **kw) -> np.ndarray:
        """区间最大回撤：max(1 - adj_close / cummax(adj_close))，取值 [0, 1)。"""
        lo, C = self._slice(t, window, self.C)
        ok = np.isfinite(C)
        cnt = ok.sum(0)
        C = np.where(ok, C, np.nan)
        run = np.fmax.accumulate(np.where(ok, C, -np.inf), axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            dd = 1.0 - C / run
        dd = np.where(np.isfinite(dd), dd, np.nan)
        m = np.nanmax(np.where(np.isfinite(dd), dd, -np.inf), axis=0)
        m = np.where(cnt >= max(5, int(0.6 * window)), m, np.nan)
        return np.where(np.isfinite(m), m, np.nan)

    def skew(self, t, window=60, min_obs=None) -> np.ndarray:
        s = self._moments(t, window, min_obs)
        sd = np.sqrt(s["var"])
        m3 = (s["Xc"] ** 3).sum(0) / s["n_eff"]
        with np.errstate(invalid="ignore", divide="ignore"):
            sk = np.where(sd > 0, m3 / sd ** 3, np.nan)
        return np.where(s["bad"], np.nan, sk)

    def kurt(self, t, window=60, min_obs=None) -> np.ndarray:
        s = self._moments(t, window, min_obs)
        v = s["var"]
        m4 = (s["Xc"] ** 4).sum(0) / s["n_eff"]
        with np.errstate(invalid="ignore", divide="ignore"):
            ku = np.where(v > 0, m4 / v ** 2 - 3.0, np.nan)   # 超额峰度
        return np.where(s["bad"], np.nan, ku)

    def vol_trend(self, t, short=20, long=60, **kw) -> np.ndarray:
        """波动率变化：短窗波动 / 长窗波动 - 1，>0 表示波动在上升。"""
        vs = self.vol(t, short)
        vl = self.vol(t, long)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(vl > 0, vs / vl - 1.0, np.nan)
        return r

    def vol_of_vol(self, t, window=120, sub=20, **kw) -> np.ndarray:
        """滚动波动的波动：把长窗切成若干短段，算各段波动的标准差。"""
        k = max(int(window) // int(sub), 2)
        segs = []
        for j in range(k):
            ti = int(t) - j * int(sub)
            if ti < sub:
                return np.full(self.k, np.nan)
            segs.append(self.vol(ti, sub, min_obs=max(5, int(0.6 * sub))))
        S = np.vstack(segs)
        ok = np.isfinite(S)
        cnt = ok.sum(0)
        mu = np.where(ok, S, 0.0).sum(0) / np.maximum(cnt, 1.0)
        v = np.where(ok, S - mu, 0.0)
        out = np.sqrt((v ** 2).sum(0) / np.maximum(cnt - 1, 1))
        return np.where(cnt >= max(2, int(0.6 * k)), out, np.nan)

    # ------------------------------------------------------------------ #
    # 成交量族
    # ------------------------------------------------------------------ #
    def amt_trend(self, t, short=5, long=60, **kw) -> np.ndarray:
        """量能放大：近 short 日均额 / 近 long 日均额 - 1。"""
        lo_s = max(int(t) - int(short) + 1, 0)
        lo_l = max(int(t) - int(long) + 1, 0)
        A = self.A
        Xs, Xl = A[lo_s:t + 1], A[lo_l:t + 1]
        a_s = self._mean_ok(Xs, np.isfinite(Xs) & (Xs > 0), max(2, short // 2))
        a_l = self._mean_ok(Xl, np.isfinite(Xl) & (Xl > 0), max(5, long // 3))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(a_l > 0, a_s / a_l - 1.0, np.nan)

    def amt_cv(self, t, window=60, **kw) -> np.ndarray:
        """成交量变异系数：std(amt)/mean(amt)，越大说明成交量越不稳定。"""
        lo, A = self._slice(t, window, self.A)
        ok = np.isfinite(A) & (A > 0)
        mn = max(5, int(0.6 * window))
        mu = self._mean_ok(A, ok, mn)
        sd = self._std_ok(A, ok, mn)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(mu > 0, sd / mu, np.nan)

    def amihud(self, t, window=20, **kw) -> np.ndarray:
        """Amihud 非流动性：mean(|r| / 成交额)，乘 1e8 便于量纲。"""
        lo, X = self._slice(t, window, self.R)
        A = self.A[lo:t + 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.abs(X) / np.where(A > 0, A, np.nan)
        return self._mean_ok(ratio, np.isfinite(ratio), max(5, int(0.6 * window))) * 1e8

    def trade_size(self, t, window=20, **kw) -> np.ndarray:
        """单笔均额 = mean(amt / num_trades)，机构参与度代理。"""
        lo = max(int(t) - int(window) + 1, 0)
        A, NT = self.A[lo:t + 1], self.NT[lo:t + 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            sz = A / np.where(NT > 0, NT, np.nan)
        lsz = np.log(np.where(sz > 0, sz, np.nan))
        return self._mean_ok(lsz, np.isfinite(lsz), max(5, int(0.6 * window)))

    def pv_corr(self, t, window=60, **kw) -> np.ndarray:
        """量价相关：corr(收益, 成交量对数变化)，>0 表示放量上涨。"""
        lo, X = self._slice(t, window, self.R)
        A = self.A[lo:t + 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            dA = np.diff(np.log(np.where(A > 0, A, np.nan)), axis=0)
        X = X[1:]
        ok = np.isfinite(X) & np.isfinite(dA)
        cnt = ok.sum(0)
        Xc, _, _ = self._mask_stats(np.where(ok, X, np.nan))
        Ac, _, _ = self._mask_stats(np.where(ok, dA, np.nan))
        num = (Xc * Ac).sum(0)
        den = np.sqrt((Xc ** 2).sum(0) * (Ac ** 2).sum(0))
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.where(den > 0, num / den, np.nan)
        return np.where(cnt >= max(10, int(0.6 * window)), c, np.nan)

    # ------------------------------------------------------------------ #
    def get(self, name: str, t: int, **kw) -> np.ndarray:
        if name not in FACTORS:
            raise KeyError("未知因子 %r，可选 %s" % (name, FACTORS))
        return getattr(self, name)(t, **kw)

    def panel(self, name: str, ts, **kw) -> pd.DataFrame:
        """在给定日期下标列表上把因子铺成面板（用于 IC 分析）。"""
        rows = np.vstack([self.get(name, t, **kw) for t in ts])
        return pd.DataFrame(rows, index=self.md.dates[list(ts)], columns=self.md.symbols)
