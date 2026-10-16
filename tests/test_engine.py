# -*- coding: utf-8 -*-
"""回测引擎口径测试：复权记账、涨跌停不可交易、T+1、成本与容量约束。

    cd Class4_回测
    python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.lib import MarketData, round_tick
from core.engine import Backtest, CostModel
from core.strategies import Reversal
from core import metrics

MD = None
FREE = CostModel(commission_rate=0.0, commission_min=0.0, transfer_fee_rate=0.0,
                 stamp_duty_rate=0.0, stamp_duty_rate_old=0.0,
                 slippage_bps=0.0, impact_coef=0.0)


def setUpModule():
    global MD
    MD = MarketData()


# --------------------------------------------------------------------------- #
# 小面板：用来精确控制涨跌停 / 停牌 / 除权等边界情形
# --------------------------------------------------------------------------- #
class FakeMD:
    def __init__(self, close, open_=None, high=None, low=None, cumadj=None, amt=None,
                 ratio=0.10, dates=None, symbols=None):
        close = np.atleast_2d(np.asarray(close, dtype="float64"))
        n, k = close.shape
        self.dates = pd.DatetimeIndex(dates if dates is not None else pd.bdate_range("2021-01-04", periods=n))
        self.symbols = pd.Index(symbols if symbols is not None else ["S%d" % i for i in range(k)])
        self.close = pd.DataFrame(close, index=self.dates, columns=self.symbols)
        self.open = self.close if open_ is None else pd.DataFrame(
            np.asarray(open_, dtype="float64"), index=self.dates, columns=self.symbols)
        # 默认给一个普通交易日（而非 high==low 的一字板）
        hi = close * 1.01 if high is None else np.asarray(high, "float64")
        lo = close * 0.99 if low is None else np.asarray(low, "float64")
        self.high = pd.DataFrame(hi, index=self.dates, columns=self.symbols)
        self.low = pd.DataFrame(lo, index=self.dates, columns=self.symbols)
        self.cumadj = pd.DataFrame(1.0 if cumadj is None else np.asarray(cumadj, "float64"),
                                   index=self.dates, columns=self.symbols)
        self.cumadj_clean = self.cumadj
        self.amt = pd.DataFrame(1e9 if amt is None else np.asarray(amt, "float64"),
                                index=self.dates, columns=self.symbols)
        self.listed = self.close.notna()
        self.traded = self.listed & (self.amt > 0)
        self.suspended = self.listed & ~self.traded
        self.one_line = self.traded & (self.high == self.low)
        self.ret_raw = (self.close / self.close.shift(1) - 1.0).where(
            self.listed & self.listed.shift(1, fill_value=False))
        self.one_line_up = self.one_line & (self.ret_raw > 0)
        self.one_line_down = self.one_line & (self.ret_raw < 0)
        prev_ref = (self.close * self.cumadj).shift(1) / self.cumadj
        self.limit_up_price = prev_ref.mul(1 + ratio).map(round_tick)
        self.limit_down_price = prev_ref.mul(1 - ratio).map(round_tick)
        self.adj_close = self.close * self.cumadj
        self.no_limit = pd.DataFrame(False, index=self.dates, columns=self.symbols)


def normal(p):
    """普通交易日：(开, 高, 低, 收)。"""
    return (p, p * 1.01, p * 0.99, p)


def one_line(p):
    """一字板：全天单一价格。"""
    return (p, p, p, p)


def panel(day_list, ratio=0.10, amt=None):
    """day_list: [(o,h,l,c), ...] 单只股票的日序列 -> FakeMD。"""
    return FakeMD([[c] for _, _, _, c in day_list],
                  open_=[[o] for o, _, _, _ in day_list],
                  high=[[h] for _, h, _, _ in day_list],
                  low=[[l] for _, _, l, _ in day_list],
                  ratio=ratio, amt=amt)


class Fixed:
    """按日期表设定目标权重；未列出的日期沿用最近一次设定（便于表达「继续持有」）。"""

    def __init__(self, table, name="fixed"):
        self.keys = sorted(pd.Timestamp(k) for k in table)
        self.vals = [pd.Series(table[k]) for k in
                     sorted(table, key=lambda x: pd.Timestamp(x))]
        self.name = name

    def weights(self, t, md):
        d = md.dates[t]
        pos = np.searchsorted(np.array(self.keys, dtype="datetime64[ns]"),
                              np.datetime64(d), side="right") - 1
        return self.vals[pos] if pos >= 0 else pd.Series(dtype="float64")


# --------------------------------------------------------------------------- #
class TestPriceLimitRules(unittest.TestCase):
    """涨跌停不可交易 —— 用户明确要求的硬约束。"""

    def _run(self, md, table, **kw):
        bt = Backtest(md, cost=FREE, max_participation=1.0, min_trade_notional=0,
                      rebalance_band=0.0, gross_limit=1.0, cash=1e6, **kw)
        return bt.run(Fixed(table), rebalance="D")

    def test_one_line_limit_up_blocks_buy(self):
        """一字涨停：全天单一价格，买不进；打开涨停后次日补单。"""
        md = panel([normal(10), one_line(11), normal(11.5)])
        res = self._run(md, {md.dates[0]: {"S0": 1.0}})
        d1, d2 = md.dates[1], md.dates[2]
        self.assertFalse(((res.trades.date == d1) & (res.trades.side == "buy")).any(),
                         "一字涨停日竟然成交了买入")
        self.assertIn("one_line_up", set(res.blocks.reason))
        self.assertTrue(((res.trades.date == d2) & (res.trades.side == "buy")).any(),
                        "涨停打开后未补单")

    def test_open_at_limit_up_blocks_buy(self):
        """开盘即涨停（盘中打开）也买不到 —— 开盘价已是涨停价，无卖盘。"""
        md = panel([normal(10), (11.0, 11.5, 10.8, 11.2), normal(11.5)])
        res = self._run(md, {md.dates[0]: {"S0": 1.0}})
        d1 = md.dates[1]
        blk = res.blocks[res.blocks.date == d1]
        self.assertIn("limit_up", set(blk.reason))
        self.assertFalse(((res.trades.date == d1) & (res.trades.side == "buy")).any())

    def test_one_line_limit_down_blocks_sell(self):
        """一字跌停：卖不出（无买盘）。"""
        md = panel([normal(10), normal(10), one_line(9)])
        res = self._run(md, {md.dates[0]: {"S0": 1.0}, md.dates[1]: {}})
        d1, d2 = md.dates[1], md.dates[2]
        self.assertTrue(((res.trades.date == d1) & (res.trades.side == "buy")).any())
        self.assertIn("one_line_down", set(res.blocks[res.blocks.date == d2].reason))
        self.assertFalse(((res.trades.date == d2) & (res.trades.side == "sell")).any())

    def test_limit_down_allows_buy(self):
        """一字跌停时**买入**允许：卖盘堆积，买方反而容易成交。"""
        md = panel([normal(10), one_line(9)])
        res = self._run(md, {md.dates[0]: {"S0": 1.0}})
        d1 = md.dates[1]
        self.assertTrue(((res.trades.date == d1) & (res.trades.side == "buy")).any(),
                        "一字跌停日买入被错误拒绝")

    def test_limit_up_allows_sell(self):
        """一字涨停时**卖出**允许：买盘堆积，卖方容易成交。"""
        md = panel([normal(10), normal(10), one_line(11)])
        res = self._run(md, {md.dates[0]: {"S0": 1.0}, md.dates[1]: {}})
        d2 = md.dates[2]
        self.assertTrue(((res.trades.date == d2) & (res.trades.side == "sell")).any(),
                        "一字涨停日卖出被错误拒绝")

    def test_limit_up_price_is_grid_aligned(self):
        """涨停价必须落在 0.01 元网格上，且用四舍五入（非银行家舍入）。"""
        md = panel([normal(5.75), normal(5.75)])
        self.assertAlmostEqual(float(md.limit_up_price.iloc[1, 0]), 6.33, places=6)

    def test_suspension_blocks_all(self):
        """停牌：买卖全部拒绝。"""
        close = [[10.0, 10.0], [10.0, 10.0], [10.0, 10.0]]
        md = FakeMD(close, open_=close, amt=[[1e9, 1e9], [0.0, 1e9], [1e9, 1e9]])
        res = self._run(md, {md.dates[0]: {"S0": 0.4, "S1": 0.4}})
        d1 = md.dates[1]
        self.assertIn("suspended", set(res.blocks.reason))
        self.assertFalse(((res.trades.date == d1) & (res.trades.symbol == "S0")).any())
        self.assertTrue(((res.trades.date == d1) & (res.trades.symbol == "S1")).any())


class TestTradeMechanics(unittest.TestCase):
    def _run(self, md, table, **kw):
        kw.setdefault("max_participation", 1.0)
        kw.setdefault("min_trade_notional", 0)
        kw.setdefault("rebalance_band", 0.0)
        kw.setdefault("gross_limit", 1.0)
        kw.setdefault("cash", 1e6)
        bt = Backtest(md, cost=FREE, **kw)
        return bt.run(Fixed(table), rebalance="D")

    def test_signal_then_next_open(self):
        """信号在 t 日收盘产生，最早 t+1 日开盘成交，同日绝不成交。"""
        md = panel([normal(10.0 + 0.02 * i) for i in range(8)])
        res = self._run(md, {md.dates[0]: {"S0": 1.0}})
        self.assertFalse((res.trades.date == md.dates[0]).any())
        self.assertTrue((res.trades.date == md.dates[1]).any())

    def test_participation_cap(self):
        """参与率上限：单日成交额不超过该股当日成交额的规定比例。"""
        md = panel([normal(10)] * 6, amt=[[1e6]] * 6)
        res = self._run(md, {md.dates[0]: {"S0": 1.0}}, max_participation=0.10)
        self.assertLessEqual(res.trades.dv.max(), 1e5 + 1)
        self.assertIn("participation", set(res.blocks.reason))

    def test_cash_scaling_no_overdraft(self):
        """目标权重之和超过 1 时按比例缩放，现金绝不为负。"""
        md = FakeMD(np.ones((8, 3)) * 10.0)
        res = self._run(md, {md.dates[0]: {"S0": 1.0, "S1": 1.0, "S2": 1.0}})
        self.assertGreaterEqual(res.daily.cash.min(), 0.0)
        w = res.holdings[res.holdings.date == md.dates[3]].set_index("symbol")["weight"]
        self.assertAlmostEqual(float(w.sum()), 1.0, places=2)
        self.assertAlmostEqual(float(w["S0"]), 1 / 3, places=2)

    def test_no_short_and_no_leverage(self):
        """负权重被截断、总权重被限制在 gross_limit 内。"""
        md = FakeMD(np.ones((6, 3)) * 10.0)
        bt = Backtest(md, cost=FREE, max_participation=1.0, min_trade_notional=0,
                      rebalance_band=0.0, gross_limit=0.5, cash=1e6)
        res = bt.run(Fixed({md.dates[0]: {"S0": -1.0, "S1": 1.0, "S2": 1.0}}), rebalance="D")
        self.assertNotIn("S0", set(res.holdings.symbol))
        w = res.holdings[res.holdings.date == md.dates[4]].set_index("symbol")["weight"]
        self.assertLessEqual(float(w.sum()), 0.5 + 1e-6)

    def test_t1_frozen_never_triggers(self):
        """T+1：先卖后买 + 每日单次下单，结构上不可能卖出当日买入。"""
        bt = Backtest(MD, cash=1e7)
        r = bt.run(Reversal(lookback=5, n_hold=30, max_weight=0.05),
                   start="2022-01-04", end="2023-12-29", rebalance="M")
        self.assertNotIn("t1_frozen", set(r.blocks.reason))


class TestAdjustment(unittest.TestCase):
    """复权：除权日不产生假亏损，净值必须是全收益口径。"""

    def test_exdiv_is_return_neutral(self):
        f = 10.0 / 9.5                       # day1 除权：10 -> 9.5，因子同步跳升
        md = FakeMD([[10.0], [9.5], [9.8]], open_=[[10.0], [9.5], [9.8]],
                    high=[[10.0], [9.5], [9.8]], low=[[10.0], [9.5], [9.8]],
                    cumadj=[[1.0], [f], [f]])
        bt = Backtest(md, cost=FREE, max_participation=1.0, min_trade_notional=0,
                      rebalance_band=0.0, gross_limit=1.0, cash=1e6)
        res = bt.run(Fixed({md.dates[0]: {"S0": 1.0}}), rebalance="D")
        nav = res.nav
        self.assertAlmostEqual(nav.iloc[1] / nav.iloc[0], 1.0, places=6)      # 除权日不亏
        self.assertAlmostEqual(nav.iloc[2] / nav.iloc[1], 9.8 / 9.5, places=6)  # 之后按真实涨跌

    def test_raw_price_accounting_would_be_wrong(self):
        """反例：若用原始价记账，除权日会凭空亏 5%。"""
        f = 10.0 / 9.5
        md = FakeMD([[10.0], [9.5], [9.5]], open_=[[10.0], [9.5], [9.5]],
                    high=[[10.0], [9.5], [9.5]], low=[[10.0], [9.5], [9.5]],
                    cumadj=[[1.0], [f], [f]])
        units = 1e6 / 10.0                                  # 用复权口径买入的份额
        naive = units * 9.5                                 # 用原始价估值
        self.assertAlmostEqual(naive / 1e6 - 1.0, -0.05, places=6)
        self.assertAlmostEqual(units * 9.5 * f / 1e6 - 1.0, 0.0, places=6)

    def test_nav_identity_on_real_data(self):
        bt = Backtest(MD, cash=1e7, cost=CostModel(slippage_bps=3))
        res = bt.run(Reversal(lookback=5, n_hold=30, max_weight=0.05),
                     start="2022-01-04", end="2023-12-29", rebalance="M")
        mv = res.holdings.groupby("date")["value"].sum().reindex(res.daily.index).fillna(0.0)
        self.assertTrue(np.allclose(res.daily.nav, res.daily.cash + mv, rtol=1e-9))
        self.assertGreaterEqual(res.daily.cash.min(), 0.0)

    def test_costs_reduce_nav(self):
        st = Reversal(lookback=5, n_hold=30, max_weight=0.05)
        cheap = Backtest(MD, cost=FREE, cash=1e7).run(
            st, start="2022-01-04", end="2023-12-29", rebalance="M")
        dear = Backtest(MD, cost=CostModel(slippage_bps=20, impact_coef=0.05),
                        cash=1e7).run(st, start="2022-01-04", end="2023-12-29", rebalance="M")
        self.assertLess(dear.nav.iloc[-1], cheap.nav.iloc[-1])
        self.assertGreater(dear.trades.fee.sum(), cheap.trades.fee.sum())

    def test_fee_schedule(self):
        c = CostModel()
        self.assertAlmostEqual(c.stamp_rate("2023-08-25"), 1e-3)     # 印花税减半前
        self.assertAlmostEqual(c.stamp_rate("2023-09-01"), 5e-4)
        self.assertAlmostEqual(c.fee("2024-01-02", 1000, "buy"), 5.0 + 1000 * 1e-5)
        self.assertAlmostEqual(c.fee("2024-01-02", 1e6, "sell"), 250.0 + 10.0 + 500.0)


class TestBenchmark(unittest.TestCase):
    """整体正确性：无摩擦下等权全市场应逼近理论等权收益。"""

    def test_equal_weight_tracks_market(self):
        from core.strategies import EqualWeightUniverse
        bt = Backtest(MD, cost=FREE, max_participation=1.0, min_trade_notional=0,
                      rebalance_band=0.0, gross_limit=1.0, cash=1e7)
        res = bt.run(EqualWeightUniverse(), start="2022-01-04", end="2023-12-29",
                     rebalance="M")
        th = MD.ret_ok.loc[res.nav.index].mean(1)
        self.assertGreater(res.ret.corr(th), 0.95)
        a = float(res.nav.iloc[-1] / res.nav.iloc[0])
        b = float((1 + th).prod())
        self.assertLess(abs(a / b - 1.0), 0.15, "与理论等权收益偏离过大：%.4f vs %.4f" % (a, b))


class TestMetrics(unittest.TestCase):
    def test_drawdown_and_stats(self):
        nav = pd.Series([1.0, 1.2, 0.9, 1.0, 1.5],
                        index=pd.bdate_range("2023-01-02", periods=5))
        d = metrics.max_drawdown_detail(nav)
        self.assertAlmostEqual(d["mdd"], 0.9 / 1.2 - 1.0, places=10)
        s = metrics.perf_stats(nav)
        self.assertAlmostEqual(s["累计收益"], 0.5, places=10)

    def test_execution_report_shape(self):
        bt = Backtest(MD, cash=1e7)
        res = bt.run(Reversal(lookback=5, n_hold=20, max_weight=0.05),
                     start="2022-01-04", end="2023-06-30", rebalance="M")
        rep = metrics.execution_report(res)
        self.assertIn("成交笔数", rep.index)
        known = {"suspended", "limit_up", "limit_down", "one_line_up", "one_line_down",
                 "t1_frozen", "participation", "cash"}
        for k in rep.index:
            if k.startswith("被挡："):
                self.assertIn(k.split("：")[1].split("（")[0], known)


if __name__ == "__main__":
    unittest.main(verbosity=2)
