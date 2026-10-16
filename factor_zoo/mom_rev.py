# -*- coding: utf-8 -*-
"""动量与反转族因子（factor_zoo 分类：mom_rev）。

口径与数据说明
--------------
- 收益、复权价一律用**复权**口径：C.C（复权收盘）、C.ADJ_O/ADJ_H/ADJ_L、C.R（复权日收益）。
  未复权 OHLC 仅在识别涨跌停/一字板时才有用，本模块不需要。
- 本数据集**没有成交量**，只有成交额与成交笔数；本模块是纯价格族，不依赖
  C.A / C.NT，因此不存在「成交额代理换手率」这类口径问题（若日后加量类因子，
  必须注明 C.A 是**成交额代理**而非真实成交量/换手率）。
- 没有股本/市值/行业/指数成分/ST 标记，因子都是原始价格量，横截面中性化交给下游统一处理。

窗口与日历
----------
交易日以面板行数计：1 个月 = 21 日，3 个月 = 63 日，6 个月 = 126 日，12 个月 = 252 日，
3 年 = 756 日。面板共 1643 个交易日（2020-01-02 ~ 2026-10-16），因此：
- 3 年窗口只有样本后段能填满：窗口内有效交易日不足时退化为「可用历史 ≥ 252 个交易日」
  的累计收益，连 252 日都不足则返回 NaN（仍只使用 t 及之前的数据，不构成未来函数）。
- 新股上市前的价格为 NaN，价格比/复权收益的自然掩码即可，无需额外的 listed 判断。

因果性
------
所有函数只读取 [0, t] 行，窗口一律通过 C.sl / C.stats 取；统计量用 C.mean / C.std 的
掩码版本（不用 np.nanmean）；无法计算的位置返回 NaN，任何路径都不会产生 inf。

方向约定（与 registry 的「-1 = 买低值」一致）
-------------------------------------------
+1 表示因子值越大越看好，-1 表示值越小越看好。本模块的反转类因子按定义**已取负**
（值大 = 过去跌得多 = 反转逻辑看好），故 direction=+1；动量类因子值大 = 过去涨得多 =
动量逻辑看好，先验同样为 +1；只有「距 52 周最高价的距离」与「隔夜收益动量」按 A 股
经验取 -1（离高点越远/隔夜涨得越多，后续越弱）。最终方向仍由样本内 IC 决定，这里只给先验。

因子清单
--------
mr_rev5 / mr_rev20                 短期反转：过去 5 / 20 日复权收益取负
mr_mom3_1 / mr_mom6_1 / mr_mom12_1 中期动量：过去 3 / 6 / 12 个月复权收益，剔除最近 1 个月
mr_ltrev3y                         长期反转：过去约 3 年（756 交易日）复权收益取负
mr_nearhigh250                     52 周高点接近度：close / max(high, 250) - 1
mr_hh_dist250                      距 250 日最高价的距离：max(high, 250) / close - 1
mr_ovn_mom20                       隔夜动量：过去 20 日「今开/昨收 - 1」累乘
mr_intra_mom20                     日内动量：过去 20 日「今收/今开 - 1」累乘
mr_mom_acc                         动量加速度：近 1-4 个月动量 - 前 4-12 个月动量
mr_mom_ir12                        收益路径平滑度：12-1 区间日收益的信息比率（均值/波动 x sqrt(n)）
"""
from __future__ import annotations

import numpy as np

from .ctx import Ctx

#: 交易日近似：1 个月 / 3 个月 / 6 个月 / 12 个月 / 3 年
D1M, D3M, D6M, D1Y, D3Y = 21, 63, 126, 252, 756


# ------------------------------------------------------------------ #
# 基础工具
# ------------------------------------------------------------------ #
def _nan_vec(k: int) -> np.ndarray:
    return np.full(int(k), np.nan, dtype="float64")


def _px_ret(t: int, C: Ctx, window: int, skip: int = 0) -> np.ndarray:
    """复权价口径的区间收益 C[t-skip] / C[t-skip-window] - 1。

    用价格比而不是连乘日收益：停牌日价格前携 => 比值天然为 0 收益；
    上市前/退市后为 NaN => 比值自动传播为 NaN。左端点不足返回全 NaN。
    """
    i1 = int(t) - int(skip)               # 区间右端
    i0 = i1 - int(window)                 # 区间左端
    out = _nan_vec(C.k)
    if i0 < 0:
        return out
    num, den = C.C[i1], C.C[i0]
    ok = np.isfinite(num) & np.isfinite(den) & (den > 0.0)
    out[ok] = num[ok] / den[ok] - 1.0     # den>0 且有限 => 不会出现 inf
    return out


def _cum_ret(X: np.ndarray, min_cnt: int = 3, min_frac: float = 0.0) -> np.ndarray:
    """把日度收益矩阵 (w, k) 按列累乘成区间收益（log 空间求和，缺值跳过）。

    有效天数不足 max(min_cnt, min_frac*w) 的列返回 NaN。
    """
    w = X.shape[0]
    need = max(int(min_cnt), int(min_frac * w))
    out = _nan_vec(X.shape[1])
    if w == 0 or need <= 0:
        return out
    ok = np.isfinite(X)
    cnt = ok.sum(0).astype("float64")
    s = np.where(ok, np.log1p(np.where(ok, X, 0.0)), 0.0).sum(0)
    return np.where(cnt >= need, np.expm1(s), np.nan)


def _daily_part(t: int, C: Ctx, window: int, kind: str) -> np.ndarray:
    """窗口内每日的隔夜或日内收益矩阵 (w, k)，行 [-w+1, 0] 对应 t-w+1 .. t。

    kind="ovn"  : 隔夜 = 今开 / 昨收 - 1（复权口径 ADJ_O[t] / C[t-1]）
    kind="intra": 日内 = 今收 / 今开 - 1（复权口径 C[t] / ADJ_O[t]）
    """
    lo = max(int(t) - int(window) + 1, 0)
    i = np.arange(lo, int(t) + 1)
    if kind == "ovn":
        i = i[i >= 1]                     # 隔夜需要 t-1 行的昨收
        if i.size == 0:
            return np.zeros((0, C.k), dtype="float64")
        num, den = C.ADJ_O[i], C.C[i - 1]
    else:
        num, den = C.C[i], C.ADJ_O[i]
    ok = np.isfinite(num) & np.isfinite(den) & (den > 0.0)
    return np.where(ok, num / np.where(ok, den, 1.0) - 1.0, np.nan)


def _high_max(t: int, C: Ctx, window: int, min_frac: float = 0.6) -> np.ndarray:
    """窗口内复权最高价的最大值（掩码版 max），有效天数不足返回 NaN。"""
    lo, H = C.sl(t, window, C.ADJ_H)
    ok = np.isfinite(H)
    cnt = ok.sum(0).astype("float64")
    hmax = np.where(ok, H, -np.inf).max(0)
    good = (cnt >= max(3.0, min_frac * H.shape[0])) & (hmax > 0.0)
    return np.where(good, hmax, np.nan)


# ------------------------------------------------------------------ #
# 1) 短期反转：过去 5 / 20 日复权收益取负
# ------------------------------------------------------------------ #
def _rev(t: int, C: Ctx, window: int) -> np.ndarray:
    return -_px_ret(t, C, window)


# ------------------------------------------------------------------ #
# 2) 中期动量：过去 3 / 6 / 12 个月收益，剔除最近 1 个月
# ------------------------------------------------------------------ #
def _mom(t: int, C: Ctx, months: int, skip: int = D1M) -> np.ndarray:
    """经典「x-1」动量：C[t-21] / C[t-21-x*21] - 1（跳过最近 1 个月）。"""
    return _px_ret(t, C, months * D1M - skip, skip=skip)


# ------------------------------------------------------------------ #
# 3) 长期反转：过去约 3 年复权收益取负
# ------------------------------------------------------------------ #
def _lt_rev(t: int, C: Ctx, window: int = D3Y, min_cnt: int = 252) -> np.ndarray:
    """过去 window 个交易日复权累计收益取负。

    面板仅 1643 个交易日，早期日期填不满 3 年窗口：此时退化为「可用最长窗口」的
    累计收益（要求窗口内有效交易日 >= min_cnt），不足 min_cnt 返回 NaN。
    """
    lo, W = C.sl(t, window, C.R)
    if W.shape[0] < int(min_cnt):
        return _nan_vec(C.k)
    return -_cum_ret(W, min_cnt=int(min_cnt))


# ------------------------------------------------------------------ #
# 4) 52 周高点：接近度 + 距离
# ------------------------------------------------------------------ #
def _near_high(t: int, C: Ctx, window: int = 250) -> np.ndarray:
    """close / max(high, window) - 1（<= 0，越接近 0 离 52 周高点越近）。"""
    hmax = _high_max(t, C, window)
    px = C.C[t]
    out = _nan_vec(C.k)
    ok = np.isfinite(hmax) & np.isfinite(px) & (hmax > 0.0)
    out[ok] = px[ok] / hmax[ok] - 1.0
    return out


def _hh_dist(t: int, C: Ctx, window: int = 250) -> np.ndarray:
    """max(high, window) / close - 1（>= 0，越大离 52 周高点越远）。"""
    hmax = _high_max(t, C, window)
    px = C.C[t]
    out = _nan_vec(C.k)
    ok = np.isfinite(hmax) & np.isfinite(px) & (px > 0.0)
    out[ok] = hmax[ok] / px[ok] - 1.0
    return out


# ------------------------------------------------------------------ #
# 5) 隔夜 / 日内收益拆分
# ------------------------------------------------------------------ #
def _ovn_mom(t: int, C: Ctx, window: int = 20) -> np.ndarray:
    """过去 window 个交易日「隔夜收益（今开/昨收-1）」的累乘。"""
    return _cum_ret(_daily_part(t, C, window, "ovn"), min_cnt=3, min_frac=0.6)


def _intra_mom(t: int, C: Ctx, window: int = 20) -> np.ndarray:
    """过去 window 个交易日「日内收益（今收/今开-1）」的累乘。"""
    return _cum_ret(_daily_part(t, C, window, "intra"), min_cnt=3, min_frac=0.6)


# ------------------------------------------------------------------ #
# 6) 同族扩展
# ------------------------------------------------------------------ #
def _mom_acc(t: int, C: Ctx, fast: int = D3M, slow: int = D1Y - D1M - D3M) -> np.ndarray:
    """动量加速度：近 1-4 个月动量 - 前 4-12 个月动量（两段不重叠）。

    fast = C[t-21]/C[t-84] - 1，slow = C[t-84]/C[t-252] - 1；> 0 表示动量在加速。
    """
    a = _px_ret(t, C, fast, skip=D1M)
    b = _px_ret(t, C, slow, skip=D1M + fast)
    out = _nan_vec(C.k)
    ok = np.isfinite(a) & np.isfinite(b)
    out[ok] = a[ok] - b[ok]
    return out


def _mom_ir(t: int, C: Ctx, window: int = D1Y - D1M, skip: int = D1M,
            min_frac: float = 0.6) -> np.ndarray:
    """收益路径平滑度：12-1 区间日收益的信息比率 mean/std * sqrt(n)。

    同样幅度的区间收益，路径越平滑（日波动越小）说明动量越「干净」，>0 表示向上且稳。
    """
    end = int(t) - int(skip)
    out = _nan_vec(C.k)
    if end < 0:
        return out
    s = C.stats(end, window, C.R)
    mu, var, cnt = s["mu"], s["var"], s["cnt"]
    good = (~s["bad"]) & (var > 0.0) & np.isfinite(mu)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[good] = mu[good] / np.sqrt(var[good]) * np.sqrt(cnt[good])
    return out


# ------------------------------------------------------------------ #
def _spec(fn, desc: str, direction: int) -> dict:
    """把 (t, C) 形式的实现包装成注册表要求的 (t, md, C) 签名。"""
    return dict(func=lambda t, md, C, _fn=fn: _fn(t, C), desc=desc, direction=int(direction))


FACTORS = {
    # ---- 短期反转（取负：值大 = 过去跌得多）----
    "mr_rev5": _spec(lambda t, C: _rev(t, C, 5),
                     "过去 5 日复权收益取负（短期反转，值大 = 过去跌得多）", +1),
    "mr_rev20": _spec(lambda t, C: _rev(t, C, 20),
                      "过去 20 日复权收益取负（短期反转，值大 = 过去跌得多）", +1),
    # ---- 中期动量（剔除最近 1 个月）----
    "mr_mom3_1": _spec(lambda t, C: _mom(t, C, 3),
                       "过去 3 个月复权收益、剔除最近 1 个月（3-1 动量）", +1),
    "mr_mom6_1": _spec(lambda t, C: _mom(t, C, 6),
                       "过去 6 个月复权收益、剔除最近 1 个月（6-1 动量）", +1),
    "mr_mom12_1": _spec(lambda t, C: _mom(t, C, 12),
                        "过去 12 个月复权收益、剔除最近 1 个月（经典 12-1 动量）", +1),
    # ---- 长期反转（取负）----
    "mr_ltrev3y": _spec(lambda t, C: _lt_rev(t, C),
                        "过去约 3 年复权累计收益取负（窗口填不满时用可用历史，至少 252 日）", +1),
    # ---- 52 周高点 ----
    "mr_nearhigh250": _spec(lambda t, C: _near_high(t, C, 250),
                            "复权收盘 / 250 日复权最高价 - 1（52 周高点接近度，越接近 0 越强）", +1),
    "mr_hh_dist250": _spec(lambda t, C: _hh_dist(t, C, 250),
                           "250 日复权最高价 / 复权收盘 - 1（距 52 周最高价的距离，越大越弱）", -1),
    # ---- 隔夜 / 日内拆分 ----
    "mr_ovn_mom20": _spec(lambda t, C: _ovn_mom(t, C, 20),
                          "过去 20 日累计隔夜收益（今开/昨收-1 累乘；A 股隔夜收益常见反转）", -1),
    "mr_intra_mom20": _spec(lambda t, C: _intra_mom(t, C, 20),
                            "过去 20 日累计日内收益（今收/今开-1 累乘；日内收益常见动量）", +1),
    # ---- 同族扩展 ----
    "mr_mom_acc": _spec(lambda t, C: _mom_acc(t, C),
                        "动量加速度 = 近 1-4 月动量 - 前 4-12 月动量（>0 动量加速）", +1),
    "mr_mom_ir12": _spec(lambda t, C: _mom_ir(t, C),
                         "收益路径平滑度：12-1 区间日收益信息比率 mean/std*sqrt(n)", +1),
}

