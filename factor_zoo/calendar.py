# -*- coding: utf-8 -*-
"""日历与季节性因子（factor_zoo 分类：calendar，前缀 cal_，共 20 个）。

本模块实现开发规范 §5.8 的全部 cal_* 因子。

设计要点
--------
1. **因果性**：日历状态只由 ``md.dates[t]``（当前交易日）与固定规则（月/日/星期、
   长假固定锚点）推算，绝不引用 ``md.dates[t+1]`` 或任何 t 之后的数据；个股状态
   只切 ``C`` 中下标 <= t 的窗口。``Ctx.truncated(t)`` 下结果逐位一致。
2. **横截面非退化**：注册表要求每天横截面标准差 > 0，纯日历哑变量（如「是否 1 月」
   全市场同值）会被判失败。因此所有因子都写成「日历状态 × 个股自身状态」；日历状态
   在月频调仓日（月末）上必须非零，否则该因子在月频选股里没有信息。
   - 指示型状态（月末月初、周一/周五、月份、长假前）在窗口外为 0；
   - 季末 / 年末 / 长假后三类事件窗口很窄，若用 0/1 指示，绝大多数月末调仓日状态为 0，
     因子退化成常数，所以改用**以事件为峰值的衰减权重**（三角 / 双曲），
     既保留事件附近的强权重，又保证月末调仓日有非零状态（口径妥协见下）。
3. 状态为 0 的位置返回 0.0 而不是 NaN，保证调仓日之外的日期仍有覆盖率；个股未上市 /
   数据不足的位置返回 NaN，出口统一收敛 ±inf。
4. 长假固定规则（节假日安排提前公布，属于静态规则，不是未来信息）：
   元旦 1/1~1/3、春节 1/25~2/20、清明 4/4、劳动 5/1~5/6、端午 6 月、国庆 9/25~10/8。
   本模块用「假期首日 / 假期末日」锚点 + 自然日距离近似「节前 / 节后 / 距下一个长假」。
5. 性能：窗口一律整块 numpy 掩码运算，无逐股 Python 循环；当日基础量按 (key, t)
   缓存在 Ctx 实例上（只保留当前 t，内存有界，且 truncated 是全新对象，不影响因果性检验）。

口径妥协 / 局限
---------------
- 农历假期（春节 / 端午 / 中秋）无法用固定公历规则精确复原：本模块用固定锚点
  （春节 1/25~2/20、端午 6 月、国庆 9/25~10/8 等）近似，与当年真实假期可能差数日。
- 「长假前 3 个交易日」用「假期首日前 4 个自然日内的工作日」近似（不读交易日历）。
- 季末 / 年末改用三角衰减权重（|d| <= 7 取 1、|d| >= 45 取 0），长假后改用双曲衰减
  权重 1/(1+d/5) 代替 0/1 指示：这三类事件窗口很窄，若用 0/1 指示，绝大多数月末
  调仓日状态恒为 0，因子在月频选股里退化成常数（注册表也会判「所有日期都是常数」）。
- 状态为 0 的日期返回 0.0（而非 NaN），个股数据不足 / 未上市返回 NaN；不做任何横截面
  标准化、不取负之外的额外变换，下游自行去极值 + z-score。
- cal_seasonal_mom / cal_seasonal_mom_max 分别需要 273 / 819 个交易日历史，数据从
  2020-01-02 开始，所以样本前段（2021-2022）整行 NaN、覆盖率偏低；2025 年起的调仓日
  不受影响（3 年历史已齐备）。
- cal_seasonal_mom_max 的「过去可用年份」按 252 交易日/年回溯，最多 5 年；不足 3 个
  可用年度（含个股自身缺失）返回 NaN。
"""
from __future__ import annotations

import numpy as np

from .ctx import Ctx

# ------------------------------------------------------------------ #
# 固定日历规则
# ------------------------------------------------------------------ #
#: 非闰年 / 闰年「各月之前」的累计天数：把 (月, 日) 转成 day-of-year
_CUM = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)
_CUM_L = (0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335)

#: 长假「假期首日」固定锚点：(月, 日)
_ONSETS = ((1, 1), (1, 25), (4, 4), (5, 1), (6, 1), (10, 1))
#: 长假「假期末日」固定锚点：(月, 日)
_ENDS = ((1, 3), (2, 20), (5, 6), (6, 25), (10, 8))
#: 季末锚点：(月, 日)
_QENDS = ((3, 31), (6, 30), (9, 30), (12, 31))

_PRE_DAYS = 4                 # 长假前自然日窗口（含周末，约 2~3 个交易日）
_POST_HL = 5.0                # 长假后衰减权重 1/(1+d/_POST_HL)，d 为距假期末日自然日数
_Q_INNER, _Q_OUTER = 7, 45    # 季末 / 年末三角权重：|d| <= 7 取 1，|d| >= 45 取 0


# ------------------------------------------------------------------ #
# 通用工具
# ------------------------------------------------------------------ #
def _fin(x) -> np.ndarray:
    """出口统一把 ±inf / NaN 收敛成 NaN。"""
    x = np.asarray(x, dtype="float64")
    return np.where(np.isfinite(x), x, np.nan)


def _nan(C: Ctx) -> np.ndarray:
    return np.full(int(C.k), np.nan, dtype="float64")


def _need(w: int) -> int:
    """窗口最低有效样本数：短窗 max(3, ceil(0.6w))，长窗 max(20, ceil(0.6w))。"""
    w = int(w)
    if w <= 0:
        return 1
    floor = 20 if w >= 120 else 3
    return max(1, min(w, max(floor, int(np.ceil(0.6 * w)))))


def _cached(C: Ctx, key: str, t: int, fn):
    """按 (key, t) 在 Ctx 实例上缓存当日基础量；只保留当前 t，内存有界。"""
    store = C.__dict__.setdefault("_cal_cache", {})
    if store.get("_t") != int(t):
        store.clear()
        store["_t"] = int(t)
    v = store.get(key)
    if v is None:
        v = fn()
        store[key] = v
    return v


# ------------------------------------------------------------------ #
# 日期工具（只吃 md.dates[t] 这一个时点）
# ------------------------------------------------------------------ #
def _ts(md, t: int):
    """当前交易日的 pandas.Timestamp（只取 md.dates[t]）。"""
    return md.dates[int(t)]


def _leap(y: int) -> bool:
    return (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)


def _ylen(y: int) -> int:
    return 366 if _leap(y) else 365


def _doy(y: int, m: int, d: int) -> int:
    """(年, 月, 日) -> day-of-year（1 起）。"""
    cum = _CUM_L if _leap(int(y)) else _CUM
    return cum[int(m) - 1] + int(d)


def _signed(ts, m: int, d: int) -> int:
    """ts 到「本年度 (m,d)」的带符号自然日差（负 = 该日在 ts 之前）。"""
    return _doy(int(ts.year), m, d) - int(ts.dayofyear)


def _to_next(ts, m: int, d: int) -> int:
    """ts 到「下一个 (m,d)」的自然日数（>= 0）。"""
    v = _signed(ts, m, d)
    return v if v >= 0 else v + _ylen(int(ts.year))


def _since_last(ts, m: int, d: int) -> int:
    """「上一个 (m,d)」到 ts 的自然日数（>= 0）。"""
    v = -_signed(ts, m, d)
    return v if v >= 0 else v + _ylen(int(ts.year) - 1)


def _kernel(d, inner: int, outer: int) -> float:
    """三角权重：|d| <= inner 取 1，|d| >= outer 取 0，中间线性衰减。"""
    a = abs(int(d))
    if a <= inner:
        return 1.0
    if a >= outer:
        return 0.0
    return 1.0 - (a - inner) / float(outer - inner)


# ------------------------------------------------------------------ #
# 日历状态（纯函数：只依赖当前日期 + 固定规则）
# ------------------------------------------------------------------ #
def _s_tom(ts) -> float:
    """月末 / 月初状态：日号 <= 5 或 >= 25 记 1，其余 0。"""
    d = int(ts.day)
    return 1.0 if (d <= 5 or d >= 25) else 0.0


def _s_dom(ts) -> float:
    """月内进度 (day-1)/20（不截断，月内单调递增）。"""
    return (int(ts.day) - 1) / 20.0


def _s_quarter(ts) -> float:
    """季末状态：到最近季末的三角权重（|d| <= 7 为 1，|d| >= 45 为 0）。"""
    y = int(ts.year)
    best = None
    for m, d in _QENDS:
        v = _signed(ts, m, d)
        for cand in (v, v - _ylen(y), v + _ylen(y)):
            if best is None or abs(cand) < abs(best):
                best = cand
    return _kernel(best, _Q_INNER, _Q_OUTER)


def _s_yearend(ts) -> float:
    """年末状态：到最近 12/31 的三角权重（|d| <= 7 为 1，|d| >= 45 为 0）。"""
    y = int(ts.year)
    v = _signed(ts, 12, 31)
    best = min((v, v - _ylen(y), v + _ylen(y)), key=abs)
    return _kernel(best, _Q_INNER, _Q_OUTER)


def _s_pre_holiday(ts) -> float:
    """长假前状态：假期首日前 _PRE_DAYS 个自然日内的工作日记 1（周末不可交易）。"""
    if int(ts.weekday()) >= 5:
        return 0.0
    for m, d in _ONSETS:
        k = _to_next(ts, m, d)
        if 1 <= k <= _PRE_DAYS:
            return 1.0
    return 0.0


def _s_post_holiday(ts) -> float:
    """长假后状态：1/(1+d/5)，d = 距上一个长假末日的自然日数（恒 > 0，越近越大）。"""
    d = min(_since_last(ts, m, dd) for m, dd in _ENDS)
    return 1.0 / (1.0 + d / _POST_HL)


def _s_weekday(ts) -> float:
    """周一 / 周五状态：周五 +1、周一 -1、其余 0。"""
    wd = int(ts.weekday())
    if wd == 4:
        return 1.0
    if wd == 0:
        return -1.0
    return 0.0


def _s_month(ts, months) -> float:
    """月份状态：落在给定月份集合内记 1，其余 0。"""
    return 1.0 if int(ts.month) in months else 0.0


def _s_days_to_holiday(ts) -> float:
    """距下一个长假交易日数：0~20 截断后取负（越接近假期越接近 0，远离为 -20）。"""
    k = min(_to_next(ts, m, d) for m, d in _ONSETS)
    return -float(min(int(k), 20))


def _s_year_progress(ts) -> float:
    """年内进度 (dayofyear-1)/当年天数，取值 [0, 1)。"""
    y = int(ts.year)
    return (int(ts.dayofyear) - 1) / float(_ylen(y))


# ------------------------------------------------------------------ #
# 个股基础状态（窗口 <= t）
# ------------------------------------------------------------------ #
def _ret(t: int, C: Ctx, w: int) -> np.ndarray:
    """复权价区间收益 C[t]/C[t-w]-1（端点缺失返回 NaN）。"""
    out = _nan(C)
    i1, i0 = int(t), int(t) - int(w)
    if i0 < 0:
        return out
    a, b = C.C[i1], C.C[i0]
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0.0) & (b > 0.0)
    out[ok] = a[ok] / b[ok] - 1.0
    return _fin(out)


def _vol(t: int, C: Ctx, w: int) -> np.ndarray:
    """窗口内复权日收益样本标准差（停牌日 R=0 计入，未上市 NaN 跳过）。"""
    _, W = C.sl(t, w, C.R)
    ok = np.isfinite(W)
    return _fin(C.std(W, ok, _need(w), ddof=1))


def _amihud(t: int, C: Ctx, w: int) -> np.ndarray:
    """Amihud 非流动性 mean(|R|/A)*1e8（只用 A>0 的正常成交日）。"""
    lo, A = C.sl(t, w, C.A)
    R = C.R[lo:int(t) + 1]
    ok = np.isfinite(A) & (A > 0.0) & np.isfinite(R)
    val = np.abs(R) / np.where(ok, A, 1.0)
    return _fin(C.mean(val, ok, _need(w)) * 1e8)


def _amt_ratio(t: int, C: Ctx, w: int) -> np.ndarray:
    """当日成交额 / 窗口平均成交额（均值只在 A>0 的正常成交日上取）。"""
    lo, A = C.sl(t, w, C.A)
    ok = np.isfinite(A) & (A > 0.0)
    mu = C.mean(A, ok, _need(w))
    cur = C.A[int(t)]
    good = np.isfinite(cur) & (cur > 0.0) & np.isfinite(mu) & (mu > 0.0)
    return _fin(np.where(good, cur / np.where(good, mu, 1.0), np.nan))


# ---- 带缓存的常用基础量 ------------------------------------------- #
def _ret5(t, C):
    return _cached(C, "ret5", t, lambda: _ret(t, C, 5))


def _ret20(t, C):
    return _cached(C, "ret20", t, lambda: _ret(t, C, 20))


def _ret60(t, C):
    return _cached(C, "ret60", t, lambda: _ret(t, C, 60))


def _vol20(t, C):
    return _cached(C, "vol20", t, lambda: _vol(t, C, 20))


def _vol60(t, C):
    return _cached(C, "vol60", t, lambda: _vol(t, C, 60))


def _illiq60(t, C):
    return _cached(C, "illiq60", t, lambda: _amihud(t, C, 60))


def _ratio60(t, C):
    return _cached(C, "ratio60", t, lambda: _amt_ratio(t, C, 60))


def _rel_amt60(t, C):
    """A[t]/mean(A,60) - 1。"""
    return _fin(_ratio60(t, C) - 1.0)


def _log_rel_amt60(t, C):
    """log(A[t]/mean(A,60))。"""
    r = _ratio60(t, C)
    good = np.isfinite(r) & (r > 0.0)
    return _fin(np.where(good, np.log(np.where(good, r, 1.0)), np.nan))


# ------------------------------------------------------------------ #
# 交互出口
# ------------------------------------------------------------------ #
def _gate(state, base, listed) -> np.ndarray:
    """日历状态 × 个股状态：state==0 处返回 0.0，未上市位置返回 NaN。"""
    s = np.asarray(state, dtype="float64")
    b = np.asarray(base, dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        v = np.where(s != 0.0, s * b, 0.0)
    return _fin(np.where(np.asarray(listed, dtype=bool), v, np.nan))


# ------------------------------------------------------------------ #
# 因子实现
# ------------------------------------------------------------------ #
def _f_tom_rev5(t, md, C):
    return _gate(_s_tom(_ts(md, t)), -_ret5(t, C), C.listed[int(t)])


def _f_tom_vol20(t, md, C):
    return _gate(_s_tom(_ts(md, t)), -_vol20(t, C), C.listed[int(t)])


def _f_tom_illiq(t, md, C):
    return _gate(_s_tom(_ts(md, t)), _illiq60(t, C), C.listed[int(t)])


def _f_tom_amt(t, md, C):
    return _gate(_s_tom(_ts(md, t)), _rel_amt60(t, C), C.listed[int(t)])


def _f_dom_rev5(t, md, C):
    return _gate(_s_dom(_ts(md, t)), -_ret5(t, C), C.listed[int(t)])


def _f_quarter_rev20(t, md, C):
    return _gate(_s_quarter(_ts(md, t)), -_ret20(t, C), C.listed[int(t)])


def _f_quarter_lowvol60(t, md, C):
    return _gate(_s_quarter(_ts(md, t)), -_vol60(t, C), C.listed[int(t)])


def _f_yearend_rev20(t, md, C):
    return _gate(_s_yearend(_ts(md, t)), -_ret20(t, C), C.listed[int(t)])


def _f_pre_holiday_ret5(t, md, C):
    return _gate(_s_pre_holiday(_ts(md, t)), -_ret5(t, C), C.listed[int(t)])


def _f_pre_holiday_amt(t, md, C):
    return _gate(_s_pre_holiday(_ts(md, t)), _log_rel_amt60(t, C), C.listed[int(t)])


def _f_post_holiday_mom20(t, md, C):
    return _gate(_s_post_holiday(_ts(md, t)), _ret20(t, C), C.listed[int(t)])


def _f_post_holiday_lowvol20(t, md, C):
    return _gate(_s_post_holiday(_ts(md, t)), -_vol20(t, C), C.listed[int(t)])


def _f_weekday_rev5(t, md, C):
    return _gate(_s_weekday(_ts(md, t)), -_ret5(t, C), C.listed[int(t)])


def _f_weekday_illiq(t, md, C):
    return _gate(_s_weekday(_ts(md, t)), _illiq60(t, C), C.listed[int(t)])


def _f_month_mom60(t, md, C):
    return _gate(_s_month(_ts(md, t), (1, 3, 10)), _ret60(t, C), C.listed[int(t)])


def _f_month_rev60(t, md, C):
    return _gate(_s_month(_ts(md, t), (4, 8)), -_ret60(t, C), C.listed[int(t)])


def _f_days_to_holiday_rev(t, md, C):
    return _gate(_s_days_to_holiday(_ts(md, t)), -_ret5(t, C), C.listed[int(t)])


def _f_year_progress_mom60(t, md, C):
    return _gate(_s_year_progress(_ts(md, t)), _ret60(t, C), C.listed[int(t)])


def _f_seasonal_mom(t, md, C):
    """去年同月 21 交易日收益 C[t-252]/C[t-273]-1。"""
    i = int(t)
    i1, i0 = i - 252, i - 273
    if i0 < 0:
        return _nan(C)
    a, b = C.C[i1], C.C[i0]
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0.0) & (b > 0.0)
    v = np.where(ok, a / np.where(ok, b, 1.0) - 1.0, np.nan)
    return _fin(np.where(C.listed[i], v, np.nan))


def _f_seasonal_mom_max(t, md, C):
    """过去可用年份同月 21 交易日收益的最大值（不足 3 年返回 NaN，最多回溯 5 年）。"""
    i = int(t)
    vals = []
    for j in range(1, 6):
        i1, i0 = i - 252 * j, i - 273 * j
        if i0 < 0:
            break
        a, b = C.C[i1], C.C[i0]
        ok = np.isfinite(a) & np.isfinite(b) & (a > 0.0) & (b > 0.0)
        vals.append(np.where(ok, a / np.where(ok, b, 1.0) - 1.0, np.nan))
    out = _nan(C)
    if len(vals) < 3:
        return out
    M = np.vstack(vals)
    cnt = np.isfinite(M).sum(0)
    good = cnt >= 3
    vmax = np.where(np.isfinite(M), M, -np.inf).max(0)
    out[good] = vmax[good]
    return _fin(np.where(C.listed[i], out, np.nan))


# ------------------------------------------------------------------ #
FACTORS = {
    # ---- 月末 / 月初（turn of month）-------------------------------- #
    "cal_tom_x_rev5": dict(
        func=_f_tom_rev5,
        desc="月末月初(日号<=5或>=25) × 近5日反转(-ret5)",
        direction=+1),
    "cal_tom_x_vol20": dict(
        func=_f_tom_vol20,
        desc="月末月初 × (-vol20)，20日复权日收益标准差取负",
        direction=+1),
    "cal_tom_x_illiq": dict(
        func=_f_tom_illiq,
        desc="月末月初 × Amihud非流动性 mean(|R|/A,60)*1e8",
        direction=+1),
    "cal_tom_x_amt": dict(
        func=_f_tom_amt,
        desc="月末月初 × (A[t]/mean(A,60)-1) 成交额相对度",
        direction=+1),
    "cal_dom_x_rev5": dict(
        func=_f_dom_rev5,
        desc="月内进度(day-1)/20 × 近5日反转(-ret5)",
        direction=+1),
    # ---- 季末 / 年末 ------------------------------------------------- #
    "cal_quarter_x_rev20": dict(
        func=_f_quarter_rev20,
        desc="季末三角权重(|d|<=7取1,|d|>=45取0) × 近20日反转(-ret20)",
        direction=+1),
    "cal_quarter_x_lowvol60": dict(
        func=_f_quarter_lowvol60,
        desc="季末三角权重 × (-vol60)，60日复权日收益标准差取负",
        direction=+1),
    "cal_yearend_x_rev20": dict(
        func=_f_yearend_rev20,
        desc="年末(12/31)三角权重 × 近20日反转(-ret20)",
        direction=+1),
    # ---- 长假前 / 长假后 --------------------------------------------- #
    "cal_pre_holiday_x_ret5": dict(
        func=_f_pre_holiday_ret5,
        desc="长假前<=4自然日的工作日 × 近5日反转(-ret5)",
        direction=+1),
    "cal_pre_holiday_x_amt": dict(
        func=_f_pre_holiday_amt,
        desc="长假前<=4自然日的工作日 × log(A[t]/mean(A,60))",
        direction=-1),
    "cal_post_holiday_x_mom20": dict(
        func=_f_post_holiday_mom20,
        desc="长假后衰减权重1/(1+d/5) × 近20日动量(ret20)",
        direction=+1),
    "cal_post_holiday_x_lowvol20": dict(
        func=_f_post_holiday_lowvol20,
        desc="长假后衰减权重1/(1+d/5) × (-vol20)",
        direction=+1),
    # ---- 周内 / 月份效应 --------------------------------------------- #
    "cal_weekday_x_rev5": dict(
        func=_f_weekday_rev5,
        desc="周一(-1)/周五(+1)状态 × 近5日反转(-ret5)",
        direction=+1),
    "cal_weekday_x_illiq": dict(
        func=_f_weekday_illiq,
        desc="周一(-1)/周五(+1)状态 × Amihud非流动性(60日)",
        direction=+1),
    "cal_month_x_mom60": dict(
        func=_f_month_mom60,
        desc="1/3/10月状态 × 近60日动量(ret60)",
        direction=+1),
    "cal_month_x_rev60": dict(
        func=_f_month_rev60,
        desc="4/8月财报季状态 × 近60日反转(-ret60)",
        direction=+1),
    # ---- 季节性 / 距假期 --------------------------------------------- #
    "cal_seasonal_mom": dict(
        func=_f_seasonal_mom,
        desc="去年同月21交易日收益 C[t-252]/C[t-273]-1",
        direction=+1),
    "cal_days_to_holiday_x_rev": dict(
        func=_f_days_to_holiday_rev,
        desc="距下一个长假交易日数(0~20截断取负) × 近5日反转(-ret5)",
        direction=+1),
    "cal_year_progress_x_mom60": dict(
        func=_f_year_progress_mom60,
        desc="年内进度(dayofyear-1)/当年天数 × 近60日动量(ret60)",
        direction=+1),
    "cal_seasonal_mom_max": dict(
        func=_f_seasonal_mom_max,
        desc="过去可用年份(<=5)同月21交易日收益最大值，不足3年返回NaN",
        direction=+1),
}
