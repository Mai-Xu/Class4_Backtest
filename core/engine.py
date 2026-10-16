# -*- coding: utf-8 -*-
"""回测引擎：目标权重驱动 + 逐日撮合，尽量贴近 A 股实盘约束。

============================ 记账口径（复权） ============================
**不能用「原始股数 x 复权价」记账**：`cumadj` 是累乘因子（000001.SZ 达 98 倍），
原始股数乘复权价会把持仓价值凭空虚增 98 倍。所以本引擎用**复权份额**记账：

    复权份额 units = 原始股数 / cumadj_t
    持仓市值 value_t = units x adj_close_t = 原始股数 x 原始价_t

买入 dv 元：units += dv / (adj_exec x (1+滑点))，现金 -= dv + 费用
估值：NAV = 现金 + units @ adj_close_t

这样 NAV 天然是**全收益**口径：除权日原始价下跌、复权价连续，分红自动留在净值里，
不需要单独记分红流水，也不会在除权日产生假亏损。

============================ 撮合约束（实盘） ============================
执行价默认取**次日开盘价**（信号在 t 日收盘后产生，t+1 日开盘成交），逐笔检查：

1. **停牌**（amt==0）：买入、卖出全部拒绝，挂单次日重试。
2. **涨停**：执行价 >= 涨停价 - 半跳 -> 买入拒绝（无卖盘）。
      - 一字涨停自然包含在内（开盘即涨停）。
      - 但**一字跌停时买入是允许的**（卖盘堆积，买方反而容易成交）。
3. **跌停**：执行价 <= 跌停价 + 半跳 -> 卖出拒绝（无买盘）。
      - **一字涨停时卖出是允许的**（买盘堆积）。
4. **T+1**：当日买入的份额计入冻结，当日不可卖出。
5. **参与率上限**：单日单票成交额不超过该股当日成交额的 max_participation
      （默认 10%），超出部分本次不成交，次日按新目标重试。
6. **现金约束**：先卖后买，买入总额受可用现金约束，超出现金时按比例缩减。
7. **成本**：佣金（万 2.5，最低 5 元）+ 过户费（万 0.1，双向）
      + 印花税（卖出，2023-08-28 前千 1、之后万 5）+ 滑点（默认 5bp）
      + 可选平方根冲击成本 impact_coef x sqrt(成交额/当日成交额)。

============================ 其他约定 ============================
- **只做多、不加杠杆**：负权重被截断为 0，权重和超过 1 时按比例缩放。
- 信号在 t 日收盘产生，**最早 t+1 日开盘成交**，结构上不存在未来函数。
- 目标权重会被「持续跟踪」：某日因涨跌停/停牌没买上的部分，次日在未更新目标前
  继续尝试，直到新信号覆盖。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.lib import MarketData, HALF_TICK

EPS = 1e-12


# --------------------------------------------------------------------------- #
# 成本模型
# --------------------------------------------------------------------------- #
@dataclass
class CostModel:
    """A 股交易成本。默认值取当前主流券商与税率。"""

    commission_rate: float = 2.5e-4        # 佣金 万 2.5
    commission_min: float = 5.0            # 单笔最低 5 元
    transfer_fee_rate: float = 1e-5        # 过户费 万 0.1（双向）
    stamp_duty_rate: float = 5e-4          # 印花税 万 5（2023-08-28 起，仅卖出）
    stamp_duty_rate_old: float = 1e-3      # 印花税 千 1（2023-08-28 前）
    stamp_duty_cut: str = "2023-08-28"
    slippage_bps: float = 5.0              # 固定滑点（单边，基点）
    impact_coef: float = 0.01              # 冲击成本 = coef x sqrt(成交额/当日成交额)

    def stamp_rate(self, date) -> float:
        return (self.stamp_duty_rate if pd.Timestamp(date) >= pd.Timestamp(self.stamp_duty_cut)
                else self.stamp_duty_rate_old)

    def fee(self, date, amount: float, side: str) -> float:
        """amount 为实际成交金额（正数）。"""
        if amount <= 0:
            return 0.0
        comm = max(amount * self.commission_rate, self.commission_min)
        fee = comm + amount * self.transfer_fee_rate
        if side == "sell":
            fee += amount * self.stamp_rate(date)
        return fee

    def slip(self, amount: float, adv_amount: float) -> float:
        """往返单边的滑点（比例）。adv_amount 为当日该股成交额。"""
        s = self.slippage_bps / 1e4
        if self.impact_coef > 0 and adv_amount > 0:
            s += self.impact_coef * np.sqrt(max(amount, 0.0) / adv_amount)
        return s


# --------------------------------------------------------------------------- #
# 账户
# --------------------------------------------------------------------------- #
class Portfolio:
    """现金 + 复权份额。units 与 frozen 均为长度 n_symbols 的数组。"""

    def __init__(self, n_symbols: int, cash: float):
        self.cash = float(cash)
        self.units = np.zeros(n_symbols)
        self.frozen = np.zeros(n_symbols)      # 当日买入、T+1 不可卖出的份额
        self.initial_cash = float(cash)

    def unlock(self):
        """开盘前解锁：昨日买入的份额今日起可卖。"""
        self.frozen[:] = 0.0

    def market_value(self, adj_px: np.ndarray) -> float:
        return float(np.nansum(self.units * adj_px))

    def nav(self, adj_px: np.ndarray) -> float:
        return self.cash + self.market_value(adj_px)

    def buy(self, j: int, dv: float, adj_exec: float, slip: float):
        """买入 mv 为 dv（元）的市值，返回（成交金额, 获得份额）。"""
        units_gain = dv / (adj_exec * (1.0 + slip))
        self.units[j] += units_gain
        self.frozen[j] += units_gain
        return dv, units_gain

    def sell(self, j: int, dv: float, adj_exec: float, slip: float):
        """卖出市值 dv（元）的持仓，返回（成交金额, 减少份额）。"""
        units_loss = min(dv / adj_exec, max(self.units[j] - self.frozen[j], 0.0))
        self.units[j] -= units_loss
        return units_loss * adj_exec * (1.0 - slip), units_loss


# --------------------------------------------------------------------------- #
# 结果
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    name: str
    nav: pd.Series
    daily: pd.DataFrame
    trades: pd.DataFrame
    blocks: pd.DataFrame
    holdings: pd.DataFrame
    target_log: pd.DataFrame
    params: dict = field(default_factory=dict)

    @property
    def ret(self) -> pd.Series:
        return self.nav.pct_change().fillna(0.0)


# --------------------------------------------------------------------------- #
# 引擎
# --------------------------------------------------------------------------- #
class Backtest:
    def __init__(self, md: MarketData, cost: CostModel | None = None,
                 exec_price: str = "open", max_participation: float = 0.10,
                 min_trade_notional: float = 5e3, rebalance_band: float = 2e-3,
                 gross_limit: float = 0.98, cash: float = 1e8,
                 use_adjusted: bool = True, enforce_limits: bool = True,
                 exec_window: int | None = 3):
        """use_adjusted / enforce_limits 只用于做「反事实对照实验」：
        分别关掉复权记账、关掉涨跌停与停牌约束，看结果被高估多少。生产回测必须都为 True。"""
        assert exec_price in ("open", "close", "(o+c)/2")
        self.md = md
        self.cost = cost or CostModel()
        self.exec_price = exec_price
        self.max_participation = float(max_participation)
        self.min_trade_notional = float(min_trade_notional)
        self.rebalance_band = float(rebalance_band)
        self.gross_limit = float(gross_limit)
        self.cash0 = float(cash)
        self.use_adjusted = bool(use_adjusted)
        self.enforce_limits = bool(enforce_limits)
        # 调仓信号的执行窗口（交易日）。信号在 t 日收盘产生，在 t+1..t+exec_window
        # 内按目标权重下单，被涨跌停/停牌挡下的部分在窗口内逐日重试；
        # 窗口结束后不再主动交易，持仓自然漂移直到下一次信号。
        # None = 无限制持续跟踪（会造成不真实的日频补单，仅用于对照）。
        self.exec_window = exec_window

        # 预取成 numpy，避免逐日从 DataFrame 取值
        self.dates = md.dates
        self.symbols = md.symbols
        self._adj_close = md.adj_close.values.astype("float64")
        if not self.use_adjusted:
            self._adj_close = md.close.values.astype("float64")
        if exec_price == "open":
            self._px_raw = md.open.values.astype("float64")
        elif exec_price == "close":
            self._px_raw = md.close.values.astype("float64")
        else:
            self._px_raw = ((md.open.values + md.close.values) / 2.0).astype("float64")
        factor = md.cumadj_clean.values.astype("float64")
        if not self.use_adjusted:
            factor = np.ones_like(factor)          # 反事实：不做复权记账
        self._adj_exec = self._px_raw * factor

        self._suspended = md.suspended.values
        self._one_line_up = md.one_line_up.values
        self._one_line_down = md.one_line_down.values
        self._up = md.limit_up_price.values
        self._dn = md.limit_down_price.values
        self._amt = np.nan_to_num(md.amt.values.astype("float64"))
        self._listed = md.listed.values

    # ---------------------------------------------------------------- #
    def _normalize(self, w) -> pd.Series:
        """权重清洗：只做多、无杠杆。"""
        if w is None:
            return pd.Series(dtype="float64")
        s = pd.Series(w).astype("float64")
        s = s.reindex(self.symbols) if not s.index.equals(self.symbols) else s
        s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
        tot = float(s.sum())
        if tot > self.gross_limit:
            s = s * (self.gross_limit / tot)
        return s[s > 0]

    @staticmethod
    def _rebalance_index(dates: pd.DatetimeIndex, rule) -> set:
        if rule is None:
            return set()
        if not isinstance(rule, str):
            want = pd.DatetimeIndex(pd.to_datetime(list(rule)))
            return set(np.flatnonzero(dates.isin(want)).tolist())
        if rule.upper() == "D":
            return set(range(len(dates)))
        if rule.upper() == "W":
            key = dates.to_period("W")
        elif rule.upper() == "M":
            key = dates.to_period("M")
        else:
            raise ValueError("rebalance 只支持 'D'/'W'/'M' 或日期序列")
        s = pd.Series(np.arange(len(dates)), index=dates)
        return set(s.groupby(key).max().tolist())

    # ---------------------------------------------------------------- #
    def run(self, strategy, start=None, end=None, rebalance: str = "M",
            verbose: bool = False) -> Result:
        n_day, n_sym = self._adj_close.shape
        i0 = 0 if start is None else int(self.dates.searchsorted(pd.Timestamp(start)))
        i1 = n_day - 1 if end is None else int(self.dates.searchsorted(pd.Timestamp(end), "right") - 1)
        i0 = max(i0, 0)          # 允许从样本首日开始：首日收盘产生信号，次日开盘成交

        rebal = self._rebalance_index(self.dates, rebalance)
        pf = Portfolio(n_sym, self.cash0)

        nav_hist, daily_rows = [], []
        trades, blocks, hold_rows, tgt_rows = [], [], [], []
        target = None            # None = 尚未收到任何信号；空 Series = 收到「持有现金」信号
        exec_until = -1          # 当前目标的有效执行窗口末日（位置下标）

        for i in range(i0, i1 + 1):
            date = self.dates[i]
            pf.unlock()                                   # 1) T+1 解锁（可卖）
            adj_px = self._adj_close[i]
            adj_exec = self._adj_exec[i]
            px_raw = self._px_raw[i]

            # 2) 开盘撮合：向目标权重靠拢
            traded = 0.0
            if target is not None and i <= exec_until:
                traded = self._trade_to_target(i, date, pf, target, adj_px, adj_exec,
                                               px_raw, trades, blocks)
            # 3) 收盘估值
            nav = pf.nav(adj_px)
            nav_hist.append((date, nav))

            # 4) 收盘后生成新目标（只能用 <= t 的数据）
            if i in rebal and i + 1 <= i1:
                w = strategy.weights(i, self.md) if hasattr(strategy, "weights") else strategy(i, self.md)
                target = self._normalize(w)
                exec_until = i + self.exec_window if self.exec_window is not None else n_day
                tgt_rows.append((date, len(target), float(target.sum())))

            # 持仓明细（仅记录非零，控制体积）
            held = np.flatnonzero(pf.units > 1e-9)
            if len(held) and nav > 0:
                val = pf.units[held] * adj_px[held]
                hold_rows.append(pd.DataFrame({
                    "date": date, "symbol": self.symbols[held],
                    "value": val, "weight": val / nav}))

            daily_rows.append(dict(date=date, nav=nav, cash=pf.cash, traded=traded,
                                   turnover=traded / nav if nav > 0 else 0.0,
                                   n_holdings=len(held)))
            if verbose and (i - i0) % 250 == 0:
                print("  %s  NAV=%.4g  持仓=%d" % (date.date(), nav, len(held)))

        nav = pd.Series({d: v for d, v in nav_hist}).sort_index()
        daily = pd.DataFrame(daily_rows).set_index("date")
        trade_df = pd.DataFrame(trades, columns=["date", "symbol", "side", "dv",
                                                 "price", "units", "fee", "slip_bps"])
        block_df = pd.DataFrame(blocks, columns=["date", "symbol", "side", "dv", "reason"])
        hold_df = pd.concat(hold_rows, ignore_index=True) if hold_rows else pd.DataFrame(
            columns=["date", "symbol", "value", "weight"])
        tgt_df = pd.DataFrame(tgt_rows, columns=["date", "n_target", "gross"]).set_index("date")
        return Result(name=getattr(strategy, "name", strategy.__class__.__name__),
                      nav=nav, daily=daily, trades=trade_df, blocks=block_df,
                      holdings=hold_df, target_log=tgt_df,
                      params=dict(exec_price=self.exec_price, rebalance=rebalance,
                                  max_participation=self.max_participation,
                                  cash=self.cash0,
                                  cost=self.cost.__dict__.copy()))

    # ---------------------------------------------------------------- #
    def _buy_block(self, i, j, px_raw):
        """买入的硬约束：返回被挡原因，None 表示可买。"""
        if self._suspended[i, j]:
            return "suspended"                       # 停牌，无法成交
        if not self.enforce_limits:                  # 反事实：假定涨跌停也能成交
            return None
        if self._one_line_up[i, j]:
            return "one_line_up"                     # 一字涨停，买盘排不上
        if np.isfinite(self._up[i, j]) and px_raw[j] >= self._up[i, j] - HALF_TICK:
            return "limit_up"                        # 执行价已在涨停价，无卖盘
        return None

    def _sell_block(self, i, j, px_raw):
        """卖出的硬约束：返回被挡原因，None 表示可卖。

        注意方向性：一字**跌停**时卖出被挡（无买盘），但买入反而容易成交；
        一字**涨停**时卖出容易成交，买入被挡。所以这里只检查跌停方向。
        """
        if self._suspended[i, j]:
            return "suspended"
        if not self.enforce_limits:                  # 反事实：假定涨跌停也能成交
            return None
        if self._one_line_down[i, j]:
            return "one_line_down"
        if np.isfinite(self._dn[i, j]) and px_raw[j] <= self._dn[i, j] + HALF_TICK:
            return "limit_down"
        return None

    # ---------------------------------------------------------------- #
    def _trade_to_target(self, i, date, pf, target, adj_px, adj_exec, px_raw,
                         trades, blocks) -> float:
        """把组合拉向目标权重，返回当日成交金额合计。"""
        n_sym = len(self.symbols)
        w = np.zeros(n_sym)
        idx = self.symbols.get_indexer(target.index)
        keep = idx >= 0
        w[idx[keep]] = target.values[keep]

        valid = np.isfinite(adj_exec) & (adj_exec > 0)
        cur_val = np.where(valid, pf.units * adj_exec, 0.0)
        # 目标权重以「执行时刻（开盘）」的市值为基数：用昨收会在跳空开盘时
        # 系统性高估可投金额，产生大量假的现金不足。
        nav_pre = pf.cash + float(np.nansum(cur_val))
        if nav_pre <= 0:
            return 0.0
        delta = w * nav_pre - cur_val
        band = max(self.min_trade_notional, self.rebalance_band * nav_pre)
        delta = np.where(valid & (np.abs(delta) > band), delta, 0.0)

        # ---- 1) 先卖：回款当日即可用于买入（A 股资金 T+0） ----
        traded = 0.0
        for j in np.flatnonzero(delta < 0):
            traded += self._do_sell(i, date, pf, j, -float(delta[j]),
                                    adj_exec, px_raw, trades, blocks)

        # ---- 2) 再买：先剔除硬约束，再按参与率、现金统一缩放 ----
        cand = []
        for j in np.flatnonzero(delta > 0):
            reason = self._buy_block(i, j, px_raw)
            if reason:
                blocks.append(dict(date=date, symbol=self.symbols[j], side="buy",
                                   dv=float(delta[j]), reason=reason))
                continue
            limit = self.max_participation * self._amt[i, j]
            cand.append((j, float(delta[j]), limit))
        if cand:
            want = np.array([min(d, l) for _, d, l in cand])
            for (j, d, l), wv in zip(cand, want):
                if d - wv > 1.0:
                    blocks.append(dict(date=date, symbol=self.symbols[j], side="buy",
                                       dv=d - wv, reason="participation"))
            room = max(pf.cash / (1.0 + self.cost.commission_rate
                                   + self.cost.transfer_fee_rate) - len(cand) * self.cost.commission_min,
                       0.0)
            tot = float(want.sum())
            if tot > room and tot > 0:
                scale = room / tot
                for (j, d, l), wv in zip(cand, want):
                    if wv * (1.0 - scale) > 1.0:
                        blocks.append(dict(date=date, symbol=self.symbols[j], side="buy",
                                           dv=wv * (1.0 - scale), reason="cash"))
                want = want * scale
            for (j, d, l), wv in zip(cand, want):
                traded += self._do_buy(i, date, pf, j, float(wv),
                                       adj_exec, px_raw, trades, blocks)
        return traded

    # ---------------------------------------------------------------- #
    def _affordable(self, cash: float, dv: float) -> float:
        """在现金 cash 下最多能买入的市值：预留手续费（含最低佣金）。"""
        if cash <= 0:
            return 0.0
        rate = self.cost.commission_rate + self.cost.transfer_fee_rate
        dv = min(dv, cash - self.cost.commission_min)      # 预留最低佣金
        dv = min(dv, cash / (1.0 + rate))
        return max(dv, 0.0)

    def _do_buy(self, i, date, pf, j, dv, adj_exec, px_raw, trades, blocks) -> float:
        """买入市值 dv 元。调用前必须已通过 _buy_block 与缩放。"""
        dv = self._affordable(pf.cash, float(dv))
        if dv <= 1.0:
            return 0.0
        amt = self._amt[i, j]
        slip = self.cost.slip(dv, amt)
        exec_amount, units_gain = pf.buy(j, dv, adj_exec[j], slip)
        fee = self.cost.fee(date, exec_amount, "buy")
        pf.cash -= (exec_amount + fee)
        trades.append(dict(date=date, symbol=self.symbols[j], side="buy",
                           dv=exec_amount, price=px_raw[j], units=units_gain,
                           fee=fee, slip_bps=slip * 1e4))
        return exec_amount

    # ---------------------------------------------------------------- #
    def _do_sell(self, i, date, pf, j, dv, adj_exec, px_raw, trades, blocks) -> float:
        """卖出市值 dv 元。"""
        if dv <= 1.0:
            return 0.0
        amt = self._amt[i, j]
        reason = self._sell_block(i, j, px_raw)
        if reason:
            blocks.append(dict(date=date, symbol=self.symbols[j], side="sell",
                               dv=float(dv), reason=reason))
            return 0.0
        limit = self.max_participation * amt
        sellable = max(pf.units[j] - pf.frozen[j], 0.0) * adj_exec[j]
        if sellable <= 0:
            blocks.append(dict(date=date, symbol=self.symbols[j], side="sell",
                               dv=float(dv), reason="t1_frozen"))
            return 0.0
        cap = min(dv, limit, sellable)
        if dv - cap > 1.0:
            blocks.append(dict(date=date, symbol=self.symbols[j], side="sell",
                               dv=float(dv - cap),
                               reason="t1_frozen" if sellable <= limit else "participation"))
        if cap <= 1.0:
            return 0.0
        slip = self.cost.slip(cap, amt)
        exec_amount, units_loss = pf.sell(j, cap, adj_exec[j], slip)
        fee = self.cost.fee(date, exec_amount, "sell")
        pf.cash += (exec_amount - fee)
        trades.append(dict(date=date, symbol=self.symbols[j], side="sell",
                           dv=exec_amount, price=px_raw[j], units=units_loss,
                           fee=fee, slip_bps=slip * 1e4))
        return exec_amount
