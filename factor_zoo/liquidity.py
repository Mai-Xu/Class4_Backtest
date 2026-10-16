# -*- coding: utf-8 -*-
"""因子分类：流动性与交易活跃度（前缀 lq_）。

数据前提（重要，先看这里）
--------------------------
本数据集只有「未复权 OHLC + 成交额 amt（元）+ 成交笔数 num_trades + 累计复权因子」，
**没有成交量、股本、市值、行业**，因此：

1. 真实换手率无法构造（缺股本）。框架里「换手率代理 volume / mean(volume, N)」在本
   数据集只能用 **成交额相对自身历史的倍数** 来近似 —— 本模块里凡属此类，
   desc/注释都写明是「**成交额代理**」，请勿当成真实换手率使用。
   （amt 与 volume 只差一个当日价格因子，倍数关系定性一致。）
2. Amihud 非流动性 mean(|r| / amt) 不需要成交量：分母本来就是成交额（量 × 价），
   与「成交量口径」的非流动性等价，无需任何换算，只乘 1e8 调量纲。
3. amt <= 0（停牌日 amt == 0）与 NaN（未上市 / 已退市 / 数值缺口）一律视作缺失；
   num_trades 在 2026-10-16 整行缺失、另有 10 个交易日零星缺口，同样按缺失处理。

统计口径与因果性
----------------
* 窗口一律用 C.sl(t, w, X) 取 [t-w+1, t]，**只含第 t 行及之前**的数据；注册表会用
  Ctx.truncated(t) 复核因果性，因此这里不使用 md 上的任何全样本量。
* 均值 / 标准差 / 相关系数一律走掩码版本（C.mean / C.std 或本文件的掩码实现），
  绝不用 np.nanmean；窗口内有效样本数不足时返回 NaN。
* 所有因子返回 float64、shape=(C.k,)，无法计算处为 NaN，**绝不返回 inf**。

因子清单（14 个）
----------------
成交额水平/活跃度   lq_amt_log20（水平）、lq_amt_rel20_60（相对活跃度，成交额代理换手）
Amihud 非流动性     lq_amihud20、lq_amihud60
成交额趋势          lq_amt_trend20_120、lq_amt_slope60（对数线性斜率）
成交额波动/稳定性   lq_amt_cv60（变异系数）、lq_amt_stab60（log 额一阶自相关）
单笔均额（机构）     lq_trade_size20、lq_trade_size_chg、lq_trade_size_skew60
流动性冲击          lq_amt_impact60（-corr(Δlog 额, 收益)）
成交笔数            lq_nt_log20、lq_nt_trend20_120
"""
from __future__ import annotations

import numpy as np

from .ctx import Ctx


# ================================================================== #
# 工具：一律掩码统计，避开 np.nanmean 与 inf
# ================================================================== #
def _ok(x) -> np.ndarray:
    """有效观测掩码：有限且严格为正（amt <= 0 的停牌日、NaN 都算缺失）。"""
    return np.isfinite(x) & (x > 0.0)


def _pos_or_nan(x) -> np.ndarray:
    """只保留 >0 的有限值，其余置 NaN（给分子/分母做合法性预处理）。"""
    return np.where(_ok(x), x, np.nan)


def _min_cnt(w: int, frac: float = 0.5, floor: int = 4) -> int:
    """窗口最低有效样本数：max(下界, frac * 窗口)。"""
    return max(int(floor), int(round(float(frac) * int(w))))


def _log_ok(x, ok) -> np.ndarray:
    """log(x)，仅 ok 位置有意义；ok 之外填 1，避免 log(0/NaN) 产生 -inf。"""
    return np.log(np.where(ok, x, 1.0))


def _ratio(num, den) -> np.ndarray:
    """安全除法：分子或分母非有限、或分母 <= 0 时返回 NaN（永不产生 inf）。"""
    good = np.isfinite(num) & np.isfinite(den) & (den > 0.0)
    return np.where(good, num / np.where(good, den, 1.0), np.nan)


def _mean_ok(C: Ctx, t: int, w: int, X=None, frac: float = 0.5, floor: int = 4):
    """近 w 个交易日的掩码均值（只统计 >0 的有效值），有效数不足返回 NaN。"""
    X = C.A if X is None else X
    lo, W = C.sl(t, w, X)
    return C.mean(W, _ok(W), _min_cnt(w, frac, floor))


def _corr_ok(x, y, ok) -> np.ndarray:
    """掩码 Pearson 相关：只用 ok 位置（NaN/无效位置不参与），分母为 0 → NaN。"""
    cnt = ok.sum(0).astype("float64")
    xb = np.where(ok, x, 0.0).sum(0) / np.maximum(cnt, 1.0)
    yb = np.where(ok, y, 0.0).sum(0) / np.maximum(cnt, 1.0)
    xc = np.where(ok, x - xb, 0.0)
    yc = np.where(ok, y - yb, 0.0)
    return _ratio((xc * yc).sum(0), np.sqrt((xc ** 2).sum(0) * (yc ** 2).sum(0)))


def _log_trade_size(t: int, C: Ctx, w: int):
    """窗口内每日「单笔均额」的对数 log(amt / num_trades)，返回 (值矩阵, 有效掩码)。

    单笔均额 = 成交额 / 成交笔数，是机构参与度的代理：单笔均额越大，
    越可能是大资金 / 机构在交易。amt 与 num_trades 任一缺失即该日无效。
    """
    lo, A = C.sl(t, w, C.A)
    NT = C.NT[lo:int(t) + 1]
    ok = _ok(A) & _ok(NT)
    return _log_ok(A, ok) - _log_ok(NT, ok), ok


# ================================================================== #
# 1) 成交额水平与相对活跃度
# ================================================================== #
def _amt_log_level(t: int, C: Ctx, w: int = 20) -> np.ndarray:
    """成交额水平：log(近 w 日均成交额)。规模 / 流动性最基础的代理。"""
    mu = _mean_ok(C, t, w, frac=0.5)
    good = np.isfinite(mu) & (mu > 0.0)
    return np.where(good, np.log(np.where(good, mu, 1.0)), np.nan)


def _amt_relative(t: int, C: Ctx, w: int = 60) -> np.ndarray:
    """相对活跃度：当日成交额 / 近 w 日均额 - 1。

    **成交额代理的「换手率」**：本数据集没有股本，无法算真实换手率，
    这里退化成「今天的成交额相对自身近 w 日历史均值的倍数 - 1」，
    >0 表示相比过去放量。当日 amt 单日噪声较大，平滑口径见 lq_amt_trend20_120。
    """
    mu = _mean_ok(C, t, w, frac=0.5, floor=20)
    return _ratio(_pos_or_nan(C.A[int(t)]), mu) - 1.0


# ================================================================== #
# 2) Amihud 非流动性
# ================================================================== #
def _amihud(t: int, C: Ctx, w: int = 20) -> np.ndarray:
    """Amihud 非流动性：mean(|r| / amt) × 1e8，窗口 w 日。

    分母 amt 是成交额（元），故无需成交量 / 换手率即可直接计算；
    值越大 = 单位成交额推动的价格变动越大 = 越不流动。
    """
    lo, R = C.sl(t, w, C.R)
    A = C.A[lo:int(t) + 1]
    ok = np.isfinite(R) & _ok(A)
    ratio = np.abs(np.where(ok, R, 0.0)) / np.where(ok, A, 1.0)
    return C.mean(ratio, ok, _min_cnt(w, 0.6, 5)) * 1e8


# ================================================================== #
# 3) 成交额趋势
# ================================================================== #
def _amt_trend(t: int, C: Ctx, short: int = 20, long: int = 120) -> np.ndarray:
    """成交额趋势：近 short 日均额 / 近 long 日均额 - 1（>0 = 量能放大）。"""
    a_s = _mean_ok(C, t, short, frac=0.6, floor=5)
    a_l = _mean_ok(C, t, long, frac=0.5, floor=20)
    return _ratio(a_s, a_l) - 1.0


def _amt_slope(t: int, C: Ctx, w: int = 60) -> np.ndarray:
    """成交额对数线性回归斜率 ×100（%/ 日）：log(amt) ~ a + b·i，i = 0..w-1。

    b > 0 表示成交额在窗口内持续放大（趋势型放量），比两端均值比更平滑。
    """
    lo, A = C.sl(t, w, C.A)
    ok = _ok(A)
    i = np.arange(A.shape[0], dtype="float64")[:, None]
    cnt = ok.sum(0).astype("float64")
    ib = np.where(ok, i, 0.0).sum(0) / np.maximum(cnt, 1.0)
    L = _log_ok(A, ok)
    lb = np.where(ok, L, 0.0).sum(0) / np.maximum(cnt, 1.0)
    ic = np.where(ok, i - ib, 0.0)
    lc = np.where(ok, L - lb, 0.0)
    slope = _ratio((ic * lc).sum(0), (ic ** 2).sum(0))
    return np.where(cnt >= _min_cnt(w, 0.6, 12), slope * 100.0, np.nan)


# ================================================================== #
# 4) 成交额波动 / 稳定性
# ================================================================== #
def _amt_cv(t: int, C: Ctx, w: int = 60) -> np.ndarray:
    """成交额变异系数：std(amt) / mean(amt)，越大 = 成交额越不稳定（脉冲式放量）。"""
    lo, A = C.sl(t, w, C.A)
    ok = _ok(A)
    mc = _min_cnt(w, 0.6, 8)
    return _ratio(C.std(A, ok, mc), C.mean(A, ok, mc))


def _amt_stability(t: int, C: Ctx, w: int = 60) -> np.ndarray:
    """成交额稳定性：log(amt) 的一阶自相关（w 对样本）。

    越高 = 成交额序列越平滑 / 越有连续性（活跃度持续）；越低 = 成交额忽大忽小。
    """
    lo, A = C.sl(t, w + 1, C.A)
    ok = _ok(A)
    L = _log_ok(A, ok)
    okp = ok[1:] & ok[:-1]
    c = _corr_ok(L[1:], L[:-1], okp)
    return np.where(okp.sum(0) >= _min_cnt(w, 0.6, 10), c, np.nan)


# ================================================================== #
# 5) 单笔均额（机构参与度代理）
# ================================================================== #
def _trade_size(t: int, C: Ctx, w: int = 20) -> np.ndarray:
    """单笔均额水平：mean(log(amt / num_trades))（对数元），越大越是机构参与。"""
    sz, ok = _log_trade_size(t, C, w)
    return C.mean(sz, ok, _min_cnt(w, 0.6, 5))


def _trade_size_chg(t: int, C: Ctx, short: int = 20, long: int = 120) -> np.ndarray:
    """单笔均额变化：log 均额(近 short 日) - log 均额(近 long 日) ≈ 变化率。

    >0 表示单笔成交在变大（大资金参与度上升）；用对数差，天然是无量纲变化率。
    """
    s_s = _log_trade_size(t, C, short)
    s_l = _log_trade_size(t, C, long)
    m_s = C.mean(s_s[0], s_s[1], _min_cnt(short, 0.6, 5))
    m_l = C.mean(s_l[0], s_l[1], _min_cnt(long, 0.5, 20))
    return m_s - m_l


def _trade_size_skew(t: int, C: Ctx, w: int = 60) -> np.ndarray:
    """单笔均额的偏度：窗口内 log(amt/num_trades) 的三阶矩偏度。

    右偏（>0）= 平时多为小单、偶发少数巨额单 —— 机构 / 大资金进场的痕迹。
    """
    sz, ok = _log_trade_size(t, C, w)
    cnt = ok.sum(0).astype("float64")
    mu = np.where(ok, sz, 0.0).sum(0) / np.maximum(cnt, 1.0)
    c = np.where(ok, sz - mu, 0.0)
    n_eff = np.maximum(cnt - 1.0, 1.0)
    m2 = (c ** 2).sum(0) / n_eff
    m3 = (c ** 3).sum(0) / n_eff
    sk = _ratio(m3, np.sqrt(m2) ** 3)
    return np.where(cnt >= _min_cnt(w, 0.6, 20), sk, np.nan)


# ================================================================== #
# 6) 流动性冲击：大成交额伴随低涨幅
# ================================================================== #
def _amt_impact(t: int, C: Ctx, w: int = 60) -> np.ndarray:
    """流动性冲击：-corr(Δlog 成交额, 当日复权收益)，窗口 w 日。

    取负号后值越大，越符合「成交额放大 + 收益为负」的冲击形态（放量下跌），
    说明价格对资金的吸收越差、流动性冲击越强。逐日配对：r_i 与
    log(amt_i / amt_{i-1}) 对齐（同一天）。
    """
    lo, A = C.sl(t, w + 1, C.A)
    R = C.R[lo:int(t) + 1]
    ok = _ok(A)
    L = _log_ok(A, ok)
    d = L[1:] - L[:-1]                      # 当日 log 成交额变化
    r = R[1:]                               # 与 d 逐行对齐（同一天）
    okp = ok[1:] & ok[:-1] & np.isfinite(r)
    c = _corr_ok(d, r, okp)
    return np.where(okp.sum(0) >= _min_cnt(w, 0.6, 10), -c, np.nan)


# ================================================================== #
# 7) 成交笔数
# ================================================================== #
def _nt_log_level(t: int, C: Ctx, w: int = 20) -> np.ndarray:
    """成交笔数水平：log(近 w 日平均成交笔数)，交易参与广度（笔数 ≠ 金额）。"""
    mu = _mean_ok(C, t, w, X=C.NT, frac=0.5)
    good = np.isfinite(mu) & (mu > 0.0)
    return np.where(good, np.log(np.where(good, mu, 1.0)), np.nan)


def _nt_trend(t: int, C: Ctx, short: int = 20, long: int = 120) -> np.ndarray:
    """成交笔数趋势：近 short 日均笔数 / 近 long 日均笔数 - 1（>0 = 交易笔数变密）。"""
    n_s = _mean_ok(C, t, short, X=C.NT, frac=0.6, floor=5)
    n_l = _mean_ok(C, t, long, X=C.NT, frac=0.5, floor=20)
    return _ratio(n_s, n_l) - 1.0


# ================================================================== #
# 注册表（direction 只是先验，最终方向由样本内 IC 决定）
# ================================================================== #
FACTORS = {
    # --- 1) 成交额水平与相对活跃度 ---
    "lq_amt_log20": dict(
        func=lambda t, md, C: _amt_log_level(t, C, 20),
        desc="log(20日均成交额)：流动性水平/规模代理", direction=-1),
    "lq_amt_rel20_60": dict(
        func=lambda t, md, C: _amt_relative(t, C, 60),
        desc="当日额/60日均额-1：相对活跃度（成交额代理换手）", direction=-1),

    # --- 2) Amihud 非流动性 ---
    "lq_amihud20": dict(
        func=lambda t, md, C: _amihud(t, C, 20),
        desc="Amihud非流动性 mean(|r|/amt,20)×1e8", direction=1),
    "lq_amihud60": dict(
        func=lambda t, md, C: _amihud(t, C, 60),
        desc="Amihud非流动性 mean(|r|/amt,60)×1e8", direction=1),

    # --- 3) 成交额趋势 ---
    "lq_amt_trend20_120": dict(
        func=lambda t, md, C: _amt_trend(t, C, 20, 120),
        desc="均额20/120-1：中短期量能放大", direction=-1),
    "lq_amt_slope60": dict(
        func=lambda t, md, C: _amt_slope(t, C, 60),
        desc="log成交额60日回归斜率×100（%/日）", direction=-1),

    # --- 4) 成交额波动 / 稳定性 ---
    "lq_amt_cv60": dict(
        func=lambda t, md, C: _amt_cv(t, C, 60),
        desc="成交额变异系数 std/mean（60日）", direction=-1),
    "lq_amt_stab60": dict(
        func=lambda t, md, C: _amt_stability(t, C, 60),
        desc="log成交额一阶自相关（60对）：活跃度稳定性", direction=-1),

    # --- 5) 单笔均额（机构参与度代理） ---
    "lq_trade_size20": dict(
        func=lambda t, md, C: _trade_size(t, C, 20),
        desc="单笔均额 mean(log(amt/笔数),20)：机构参与度代理", direction=1),
    "lq_trade_size_chg": dict(
        func=lambda t, md, C: _trade_size_chg(t, C, 20, 120),
        desc="单笔均额变化 log均额20-log均额120", direction=1),
    "lq_trade_size_skew60": dict(
        func=lambda t, md, C: _trade_size_skew(t, C, 60),
        desc="单笔均额偏度（60日）：偶发大单（机构痕迹）", direction=1),

    # --- 6) 流动性冲击 ---
    "lq_amt_impact60": dict(
        func=lambda t, md, C: _amt_impact(t, C, 60),
        desc="-corr(Δlog成交额,收益)（60日）：放量下跌程度", direction=-1),

    # --- 7) 成交笔数 ---
    "lq_nt_log20": dict(
        func=lambda t, md, C: _nt_log_level(t, C, 20),
        desc="log(20日均成交笔数)：交易参与广度", direction=-1),
    "lq_nt_trend20_120": dict(
        func=lambda t, md, C: _nt_trend(t, C, 20, 120),
        desc="笔数20/120-1：成交笔数趋势（交易变密）", direction=-1),
}
