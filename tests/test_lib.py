# -*- coding: utf-8 -*-
"""数据层单元测试：把「坑」固化成断言，防止后续改动把口径改坏。

    cd Class4_回测/1_数据真实性验证
    python -m unittest discover -s tests -p "test_lib.py" -v
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

# 数据层 lib.py 在项目根目录（回测系统也依赖它）；向上找到含 lib.py 的目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.lib import (MarketData, FIELDS, board_new_days, round_tick,
                 CUMADJ_EPS, RATIO_MAIN, RATIO_GEM)

MD = None


def setUpModule():
    global MD
    MD = MarketData()


class TestStructure(unittest.TestCase):
    def test_shape_and_alignment(self):
        self.assertEqual(len(MD.dates), 1643)
        self.assertEqual(len(MD.symbols), 5451)
        self.assertEqual(MD.dates[0], pd.Timestamp("2020-01-02"))
        self.assertEqual(MD.dates[-1], pd.Timestamp("2026-10-16"))
        self.assertTrue(MD.dates.is_monotonic_increasing)
        self.assertTrue(MD.dates.is_unique)
        self.assertTrue(MD.symbols.is_unique)

    def test_all_fields_share_columns(self):
        cols = list(MD.close.columns)
        for f in FIELDS:
            self.assertEqual(list(MD.field(f).columns), cols, f)
            self.assertEqual(list(MD.field(f).index), list(MD.dates), f)

    def test_mask_alignment(self):
        ref = MD.close.isna().values
        for f in ("open", "high", "low", "amt"):
            self.assertEqual(int((MD.field(f).isna().values != ref).sum()), 0, f)
        self.assertLessEqual(int((MD.cumadj.isna().values != ref).sum()), 2)

    def test_ohlc_sanity(self):
        bad = ((MD.high < MD.low) | (MD.open > MD.high) | (MD.open < MD.low)
               | (MD.close > MD.high) | (MD.close < MD.low))
        self.assertEqual(int(np.nansum(bad.values)), 0)


class TestMasks(unittest.TestCase):
    def test_listed_partition(self):
        n = MD.traded.values.sum() + MD.suspended.values.sum()
        self.assertEqual(int(n), int(MD.listed.values.sum()))

    def test_suspension_convention(self):
        """停牌 = amt==0 且 OHLC 四价全等（前收盘携程）。"""
        m = MD.suspended
        self.assertTrue(bool(m.values.any()))
        cell = m.stack()
        cell = cell[cell]
        self.assertGreater(len(cell), 20000)
        t, s = cell.index[0]
        px = set(MD.field("open").at[t, s] for _ in (0,)) | {
            MD.close.at[t, s], MD.high.at[t, s], MD.low.at[t, s]}
        self.assertEqual(len(px), 1)                      # 四价全等
        self.assertEqual(float(MD.amt.at[t, s]), 0.0)     # 无成交额
        self.assertFalse(bool(MD.traded.at[t, s]))

    def test_one_line_is_traded_but_flat(self):
        m = MD.one_line
        self.assertTrue(bool(m.values.any()))
        self.assertFalse(bool((m & MD.suspended).values.any()))
        self.assertFalse(bool((m & ~(MD.high == MD.low)).values.any()))
        # 一字板全集 = 停牌 + 一字板
        self.assertEqual(int(MD.all_flat.values.sum()),
                         int(MD.suspended.values.sum()) + int(MD.one_line.values.sum()))

    def test_one_line_blocks_trade_direction(self):
        self.assertFalse(bool((MD.can_buy & MD.one_line_up).values.any()))
        self.assertFalse(bool((MD.can_sell & MD.one_line_down).values.any()))
        self.assertFalse(bool((MD.one_line_up & (MD.ret_raw <= 0)).values.any()))
        self.assertFalse(bool((MD.one_line_down & (MD.ret_raw >= 0)).values.any()))

    def test_covid_reopening_day(self):
        """2020-02-03（疫情后开市首日）应出现大规模一字跌停。"""
        t = pd.Timestamp("2020-02-03")
        self.assertGreater(int(MD.one_line_down.loc[t].sum()), 1500)
        self.assertGreater(int(MD.at_limit_down.loc[t].sum()), 2500)


class TestCumadj(unittest.TestCase):
    def test_denoise_and_preserve(self):
        raw = MD.cumadj.pct_change(fill_method=None).abs().values
        cln = MD.cumadj_clean.pct_change(fill_method=None).abs().values
        # 噪声带被清空
        self.assertEqual(int(((cln >= 1e-6) & (cln < CUMADJ_EPS)).sum()), 0)
        # 真实事件数量一格不少
        self.assertEqual(int((raw >= CUMADJ_EPS).sum()), int((cln >= CUMADJ_EPS).sum()))
        self.assertGreater(int((cln >= 1e-3).sum()), 20000)

    def test_clean_is_constant_within_non_event_days(self):
        t = "000001.SZ"
        a = MD.cumadj[t]
        b = MD.cumadj_clean[t]
        keep = a.notna()
        self.assertTrue(np.allclose(a[keep].values, b[keep].values, rtol=1e-3))

    def test_adjusted_price_continuity(self):
        """除权日的复权收益应当连续，不应出现原始价的跳空。"""
        t = "000001.SZ"
        ev = MD.cumadj_clean[t].pct_change(fill_method=None).abs()
        days = ev.index[ev > 1e-3]
        self.assertGreater(len(days), 5)
        for d in days:
            i = MD.dates.get_loc(d)
            raw_jump = abs(MD.ret_raw[t].iloc[i])
            adj_jump = abs(MD.ret[t].iloc[i])
            self.assertLess(adj_jump, max(raw_jump, 1e-3) + 0.05)


class TestRounding(unittest.TestCase):
    def test_half_up_not_bankers(self):
        self.assertAlmostEqual(round_tick(6.325), 6.33, places=6)   # 银行家舍入会得 6.32
        self.assertAlmostEqual(round_tick(5.005), 5.01, places=6)
        self.assertAlmostEqual(round_tick(4.550 * 1.1), 5.01, places=6)
        self.assertAlmostEqual(round_tick(6.324), 6.32, places=6)
        self.assertAlmostEqual(round_tick(6.326), 6.33, places=6)
        self.assertAlmostEqual(round_tick(np.array([1.005, 1.015])).tolist()[0], 1.01, places=6)


class TestLimits(unittest.TestCase):
    def test_prior_by_board(self):
        r = MD.limit_ratio
        t_late = MD.dates[-1]
        t_early = pd.Timestamp("2020-01-10")
        self.assertAlmostEqual(float(r.loc[t_late, "600519.SH"]), RATIO_MAIN)
        self.assertAlmostEqual(float(r.loc[t_late, "688981.SH"]), RATIO_GEM)
        self.assertAlmostEqual(float(r.loc[t_late, "300750.SZ"]), RATIO_GEM)
        self.assertAlmostEqual(float(r.loc[t_early, "300750.SZ"]), RATIO_MAIN)   # 创业板注册制前

    def test_cap_never_violated_on_normal_days(self):
        viol = ((MD.close > MD.limit_up_price + 0.005) & MD.listed & ~MD.no_limit)
        self.assertLess(int(viol.values.sum()), 60)       # 只剩极个别复牌/异常日

    def test_limit_price_grid(self):
        """涨停价必须落在 0.01 元的整数价格上。"""
        cap = MD.limit_up_price
        v = cap.values
        v = v[np.isfinite(v)]
        self.assertTrue(np.allclose(v * 100, np.round(v * 100), atol=1e-6))

    def test_extreme_days_match_real_events(self):
        dn = MD.at_limit_down.sum(1).nlargest(3)
        self.assertIn(pd.Timestamp("2020-02-03"), dn.index)
        self.assertIn(pd.Timestamp("2025-04-07"), dn.index)
        up = MD.at_limit_up.sum(1).nlargest(3)
        self.assertIn(pd.Timestamp("2024-10-08"), up.index)

    def test_st_detection_is_conservative(self):
        st = MD.st_like
        self.assertLessEqual(int(st.sum()), 6)             # 样本基本不含 ST
        for sym in ("601398.SH", "601939.SH", "601288.SH", "600900.SH", "601988.SH"):
            self.assertNotIn(sym, set(st.index[st]), "大盘银行股被误判为 5% 标的")

    def test_st_rule_wall(self):
        T = MD.ratio_table()
        st = MD.st_like
        if st.any():
            sub = T.loc[st[st].index]
            self.assertTrue(bool((sub["v05"] == 0).all()))
            self.assertTrue(bool((sub["p99"] >= 0.045).all()))
            self.assertTrue(bool((sub["days"] >= 100).all()))

    def test_new_stock_window_respects_regime(self):
        d = board_new_days(MD.board, MD.dates)
        self.assertEqual(float(d.loc[MD.dates[-1], "688981.SH"]), 5.0)
        self.assertEqual(float(d.loc[MD.dates[-1], "600519.SH"]), 5.0)
        self.assertEqual(float(d.loc[pd.Timestamp("2023-01-03"), "600519.SH"]), 1.0)
        self.assertEqual(float(d.loc[pd.Timestamp("2020-01-10"), "300750.SZ"]), 1.0)
        self.assertEqual(float(d.loc[MD.dates[-1], "300750.SZ"]), 5.0)
        # 样本首日的存量证券不得被当成次新股
        self.assertEqual(int(MD._new_stock.iloc[0].sum()), 0)


class TestReturns(unittest.TestCase):
    def test_ret_fwd_alignment(self):
        r1 = MD.ret_fwd(1)
        r = MD.ret
        t = MD.dates[100]
        s = "000001.SZ"
        self.assertAlmostEqual(float(r1.loc[t, s]), float(r.loc[MD.dates[101], s]), places=10)

    def test_suspension_return_is_zero(self):
        """停牌且当日无除权时，复权收益应为 0（头寸价值不变）。"""
        m = MD.suspended & (MD.cumadj_clean.pct_change(fill_method=None).abs() < CUMADJ_EPS)
        r = MD.ret
        v = r.values[m.values]
        self.assertGreater(len(v), 10000)
        self.assertTrue(np.allclose(v, 0.0, atol=1e-12))

    def test_ret_ok_is_subset_of_ret(self):
        a = MD.ret_ok.notna().values
        b = MD.ret.notna().values
        self.assertFalse(bool((a & ~b).any()))

    def test_market_vol_reasonable(self):
        v = MD.equal_weight_ret().std() * np.sqrt(252)
        self.assertTrue(0.15 < v < 0.40, v)


class TestUniverse(unittest.TestCase):
    def test_subset_and_no_untradable(self):
        u = MD.universe()
        self.assertFalse(bool((u & ~MD.traded).values.any()))
        self.assertFalse(bool((u & MD.one_line).values.any()))
        self.assertFalse(bool((u & MD.no_limit).values.any()))
        self.assertFalse(bool((u & MD.suspended).values.any()))

    def test_min_listed_days_respected(self):
        u = MD.universe()
        self.assertFalse(bool((u & (MD.listed_days < 250)).values.any()))

    def test_size_evolves(self):
        n = MD.n_eligible()
        self.assertEqual(int(n.iloc[0]), 0)                 # 需要 250 日历史
        self.assertGreater(int(n.iloc[-1]), 4000)
        self.assertLess(int(n.iloc[-1]), int(MD.traded.iloc[-1].sum()))

    def test_no_lookahead_in_adv(self):
        """滚动均值只能用当日及此前数据。"""
        w = 20
        a = MD.adv(w)
        t = 800
        s = "000001.SZ"
        manual = MD.amt[s].iloc[t - w + 1:t + 1].mean()
        self.assertAlmostEqual(float(a[s].iloc[t]), float(manual), places=6)

    def test_listed_days_is_cumulative_count(self):
        s = "600519.SH"
        ld = MD.listed_days[s]
        base = MD.listed[s].astype(int)
        self.assertEqual(int(ld.iloc[500]), int(base.iloc[:501].sum()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
