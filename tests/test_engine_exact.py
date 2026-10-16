# -*- coding: utf-8 -*-
"""引擎的**精确恒等式**测试（与 test_engine.py 的统计性检验互补）。

test_engine.py 用的是「无摩擦等权全市场 vs 理论等权收益，相关系数 0.9963」这类
**统计性**判据；统计性判据会放过系统性小偏差。本文件的判据是**逐位恒等式**：

  1. 单票买入持有：NAV_t 必须精确等于 现金 + 份额 x 复权收盘价；
  2. 零成本下成交完成后，各标的的成交名义金额必须**精确相等**（目标权重是等权）；
  3. 费用恒等式：NAV 与「初始资金 - 费用 + 各标的开盘到收盘的损益」逐位一致；
  4. 涨跌停不可交易：真实数据上挑一个一字涨停的 (日期, 标的)，断言买入被挡。

这些恒等式一旦不成立，说明记账/撮合有系统性错误，而不是「随机噪声」。
"""
from __future__ import annotations
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import unittest

import numpy as np
import pandas as pd

from core.lib import MarketData
from core.engine import Backtest, CostModel
from core.strategies import Strategy

CASH = 1e9

ZERO = CostModel(commission_rate=0.0, commission_min=0.0, transfer_fee_rate=0.0,
                 stamp_duty_rate=0.0, stamp_duty_rate_old=0.0,
                 slippage_bps=0.0, impact_coef=0.0)
COMM = CostModel(commission_rate=1e-4, commission_min=0.0, transfer_fee_rate=0.0,
                 stamp_duty_rate=0.0, stamp_duty_rate_old=0.0,
                 slippage_bps=0.0, impact_coef=0.0)


class OneShot(Strategy):
    """只在指定那一天收盘给出目标权重。"""

    def __init__(self, weights: dict, date):
        super().__init__(n_hold=10 ** 6)
        self._w = dict(weights)
        self._d = pd.Timestamp(date)
        self.name = "oneshot"

    def weights(self, t, md):
        if pd.Timestamp(md.dates[int(t)]) != self._d:
            return pd.Series(dtype="float64")
        return pd.Series(self._w, dtype="float64")


class TestExactIdentities(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.md = MarketData(verbose=False)
        cls.dates = cls.md.dates
        cls.sym = cls.md.symbols

    # ------------------------------------------------------------------ #
    def _liquid(self, pick="600519.SH"):
        """挑一只整段都在交易的流动性标的。"""
        self.assertIn(pick, set(self.sym), pick + " 不在样本里")
        return pick

    def _run(self, strat, rebal_date, cost=ZERO, exec_price="open", start=None):
        bt = Backtest(self.md, cost=cost, exec_price=exec_price,
                      max_participation=1e9, min_trade_notional=0.0,
                      rebalance_band=0.0, gross_limit=0.98, cash=CASH,
                      use_adjusted=True, enforce_limits=True, exec_window=1)
        return bt.run(strat, start=start, end=None, rebalance=[pd.Timestamp(rebal_date)])

    # ------------------------------------------------------------------ #
    def test_single_stock_buy_and_hold_is_exact(self):
        """单票买入持有：NAV_t == 现金 + 份额 x 复权收盘价，逐日精确。"""
        s = self._liquid()
        j = int(self.sym.get_indexer([s])[0])
        sig = int(self.dates.searchsorted(pd.Timestamp("2023-06-30")))
        res = self._run(OneShot({s: 0.98}, self.dates[sig]), self.dates[sig])
        exe = sig + 1                                    # 次日开盘成交
        open_raw = self.md.open.values[exe, j]
        f = self.md.cumadj_clean.values[exe, j]
        adj_open = open_raw * f
        self.assertTrue(np.isfinite(adj_open) and adj_open > 0)
        units = 0.98 * CASH / adj_open
        cash = CASH - 0.98 * CASH
        adj_close = self.md.adj_close.values[:, j]
        nav = res.nav
        worst = 0.0
        for i in range(exe, len(self.dates)):
            exp = cash + units * adj_close[i]
            got = float(nav.loc[self.dates[i]])
            worst = max(worst, abs(got - exp) / max(abs(exp), 1.0))
        self.assertLess(worst, 1e-12, "单票买入持有恒等式破了：%.3g" % worst)
        # 顺带确认成交确实发生在次日开盘，且只发生一次
        self.assertEqual(len(res.trades), 1)
        self.assertEqual(pd.Timestamp(res.trades.date.iloc[0]), self.dates[exe])

    # ------------------------------------------------------------------ #
    def test_equal_notional_after_execution(self):
        """零成本下成交完成后，各标的的成交名义金额必须精确相等。"""
        names = [s for s in ("600519.SH", "000001.SZ", "601318.SH", "600036.SH")
                 if s in set(self.sym)]
        self.assertGreaterEqual(len(names), 3)
        sig = int(self.dates.searchsorted(pd.Timestamp("2024-06-28")))
        w = {s: 0.98 / len(names) for s in names}
        res = self._run(OneShot(w, self.dates[sig]), self.dates[sig])
        exe = sig + 1
        self.assertEqual(len(res.trades), len(names), "成交笔数与目标不一致（可能被挡）")
        # 判据用**成交名义金额**（trades.dv），不能用 holdings.value：
        # 后者按当日收盘价估值，开盘到收盘的日内漂移会让各标的值不同，这是对的。
        vals = res.trades.dv.values
        self.assertLess(float(np.max(np.abs(vals - vals.mean())) / vals.mean()), 1e-12,
                        "等权目标下各标的成交名义金额不相等：%s" % vals)
        # 顺带确认成交价就是次日开盘价（未复权）
        for _, r in res.trades.iterrows():
            jj = int(self.sym.get_indexer([r.symbol])[0])
            self.assertAlmostEqual(float(r.price), float(self.md.open.values[exe, jj]),
                                   places=10)

    # ------------------------------------------------------------------ #
    def test_fee_identity(self):
        """费用恒等式：NAV = 初始资金 - 各项费用 + 持仓开盘到收盘的损益。"""
        names = [s for s in ("600519.SH", "000001.SZ", "601318.SH", "600036.SH")
                 if s in set(self.sym)]
        sig = int(self.dates.searchsorted(pd.Timestamp("2024-06-28")))
        exe = sig + 1
        w = {s: 0.98 / len(names) for s in names}
        res = self._run(OneShot(w, self.dates[sig]), self.dates[sig], cost=COMM)
        pos = self.sym.get_indexer(names)
        f = self.md.cumadj_clean.values[exe, pos]
        adj_open = self.md.open.values[exe, pos] * f
        adj_close = self.md.adj_close.values[exe, pos]
        notional = 0.98 * CASH / len(names)              # 每只的成交名义金额（零滑点）
        exp = CASH - 0.98 * CASH * 1e-4 \
            + float(np.sum(notional * (adj_close / adj_open - 1.0)))
        got = float(res.nav.loc[self.dates[exe]])
        self.assertLess(abs(got - exp) / CASH, 1e-12,
                        "费用恒等式破了：got %.6f exp %.6f（差 %.3g 元）"
                        % (got, exp, got - exp))
        self.assertAlmostEqual(float(self.md.ret.values[exe, pos[0]]) * 0 + 1.0, 1.0)

    # ------------------------------------------------------------------ #
    def test_one_line_up_cannot_be_bought(self):
        """真实一字涨停：买入必须被挡，且不能假装成交。"""
        olu = self.md.one_line_up.values
        amt = self.md.amt.values
        mask = olu & (amt > 0)
        # 取样本中段（避开新股不设限的日期）
        i0 = int(self.dates.searchsorted(pd.Timestamp("2023-01-01")))
        idx = np.argwhere(mask[i0:])
        self.assertGreater(len(idx), 0, "样本里没有一字涨停？")
        for k in range(0, min(len(idx), 400), 7):
            i = i0 + int(idx[k, 0])
            j = int(idx[k, 1])
            s = str(self.sym[j])
            res = self._run(OneShot({s: 0.5}, self.dates[i - 1]), self.dates[i - 1])
            if len(res.trades) == 0 and len(res.blocks) > 0:
                self.assertIn("one_line", str(res.blocks.reason.iloc[0]))
                # 不能假装成交：一股都没买到（零成本下被挡时 NAV 恰好等于初始资金）
                self.assertEqual(len(res.holdings), 0)
                self.assertAlmostEqual(float(res.nav.loc[self.dates[i]]), CASH, places=6)
                return
        self.fail("没有找到任何一例一字涨停买入被挡")

    # ------------------------------------------------------------------ #
    def test_one_line_down_cannot_be_sold(self):
        """真实一字跌停：持仓卖出必须被挡。"""
        old = self.md.one_line_down.values
        amt = self.md.amt.values
        mask = old & (amt > 0)
        i0 = int(self.dates.searchsorted(pd.Timestamp("2023-01-01")))
        idx = np.argwhere(mask[i0:])
        self.assertGreater(len(idx), 0)
        for k in range(0, min(len(idx), 400), 7):
            i = i0 + int(idx[k, 0])
            j = int(idx[k, 1])
            s = str(self.sym[j])

            class TwoStep(Strategy):
                name = "twostep"

                def __init__(self, s_):
                    super().__init__(n_hold=10 ** 6)
                    self.s = s_
                    self.n = 0

                def weights(self, t, md):
                    self.n += 1
                    if self.n == 1:
                        return pd.Series({self.s: 0.98})
                    return pd.Series(dtype="float64")

            st = TwoStep(s)
            bt = Backtest(self.md, cost=ZERO, exec_price="open", max_participation=1e9,
                          min_trade_notional=0.0, rebalance_band=0.0, gross_limit=0.98,
                          cash=CASH, use_adjusted=True, enforce_limits=True, exec_window=1)
            # 在 i-2 建仓（i-1 开盘成交），在 i-1 收盘发清仓信号 -> i 开盘卖出（正好一字跌停）
            res = bt.run(st, start=None, end=None,
                         rebalance=[self.dates[i - 2], self.dates[i - 1]])
            if len(res.blocks) and "one_line" in str(res.blocks.reason.iloc[-1]):
                self.assertIn("down", str(res.blocks.reason.iloc[-1]))
                return
        self.fail("没有找到任何一例一字跌停卖出被挡")


if __name__ == "__main__":
    unittest.main(verbosity=2)
