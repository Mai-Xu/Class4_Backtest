# -*- coding: utf-8 -*-
"""因子与策略的单元测试：重点是无未来函数、横截面标准化、缓冲带与组合构造。

    cd Class4_回测
    python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.lib import MarketData
from core.factors import Factors, FACTORS, cs_zscore
from core.strategies import PanelStrategy, Strategy, FactorPortfolio, CompositePortfolio
from core.engine import Backtest

MD = None
F = None


def setUpModule():
    global MD, F
    MD = MarketData()
    F = Factors(MD)


class TestNoLookahead(unittest.TestCase):
    """因子只能用 t 日及之前的数据 —— 这是最容易出错、也最致命的一条。"""

    def test_factor_is_causal(self):
        """把 t 之后的数据整段改成 NaN，t 日的因子值必须一字不变。"""
        t = MD.dates.get_loc("2024-06-28")
        for name in ("ivol", "dvol", "maxdd", "skew", "amt_trend", "amihud",
                     "trade_size", "pv_corr", "vol_trend"):
            a = F.get(name, t)
            R, C, A, NT = F.R.copy(), F.C.copy(), F.A.copy(), F.NT.copy()
            try:
                F.R[t + 1:] = np.nan
                F.C[t + 1:] = np.nan
                F.A[t + 1:] = np.nan
                F.NT[t + 1:] = np.nan
                b = F.get(name, t)
            finally:
                F.R, F.C, F.A, F.NT = R, C, A, NT
            np.testing.assert_allclose(np.nan_to_num(a, nan=-9.9),
                                       np.nan_to_num(b, nan=-9.9), rtol=0, atol=0,
                                       err_msg="%s 用到了未来数据" % name)

    def test_market_proxy_is_causal(self):
        """市场收益代理 ret_ok 的横截面均值本身是逐日独立的。"""
        m = MD.ret_ok.mean(1)
        t = MD.dates.get_loc("2024-06-28")
        manual = np.nanmean(MD.ret_ok.iloc[t].values)
        self.assertAlmostEqual(float(m.iloc[t]), float(manual), places=12)

    def test_amihud_uses_only_past_window(self):
        t = MD.dates.get_loc("2024-06-28")
        f = F.amihud(t, window=20)
        lo = t - 19
        A = MD.amt.values[lo:t + 1]
        R = MD.ret.values[lo:t + 1]
        j = MD.symbols.get_loc("600519.SH")
        manual = np.nanmean(np.abs(R[:, j]) / A[:, j]) * 1e8
        self.assertAlmostEqual(float(f[j]), float(manual), places=6)


class TestFactorValues(unittest.TestCase):
    def test_all_factors_finite_on_normal_day(self):
        t = MD.dates.get_loc("2024-06-28")
        u = MD.universe().values[t]
        for name in FACTORS:
            x = F.get(name, t)
            ok = np.isfinite(x) & u
            self.assertGreater(ok.sum(), 3000, "%s 有效值太少" % name)

    def test_ivol_le_total_vol(self):
        """特质波动率 <= 总波动（残差不可能比原序列更分散太多）。"""
        t = MD.dates.get_loc("2024-06-28")
        iv = F.ivol(t, 60)
        tv = F.vol(t, 60)
        ok = np.isfinite(iv) & np.isfinite(tv) & (tv > 0)
        self.assertTrue((iv[ok] <= tv[ok] * 1.15).mean() > 0.98)

    def test_dvol_le_vol(self):
        t = MD.dates.get_loc("2024-06-28")
        d, v = F.dvol(t, 60), F.vol(t, 60)
        ok = np.isfinite(d) & np.isfinite(v)
        self.assertTrue((d[ok] <= v[ok] * 1.001).all())

    def test_maxdd_range(self):
        t = MD.dates.get_loc("2024-06-28")
        m = F.maxdd(t, 60)
        ok = np.isfinite(m)
        self.assertTrue((m[ok] >= 0).all() and (m[ok] < 1).all())
        # 与直接定义一致
        j = MD.symbols.get_loc("600519.SH")
        c = MD.adj_close["600519.SH"].iloc[t - 59:t + 1].values
        self.assertAlmostEqual(float(m[j]), float(np.max(1 - c / np.maximum.accumulate(c))),
                               places=10)

    def test_beta_of_market_is_one(self):
        """等权市场组合对自己的 beta 应该接近 1。"""
        t = MD.dates.get_loc("2024-06-28")
        b = F.beta(t, 120)
        self.assertAlmostEqual(float(np.nanmedian(b)), 1.0, delta=0.35)

    def test_skew_kurt_signs(self):
        t = MD.dates.get_loc("2024-06-28")
        sk, ku = F.skew(t, 250), F.kurt(t, 250)
        self.assertLess(abs(float(np.nanmedian(sk))), 1.5)
        self.assertLess(abs(float(np.nanmedian(ku))), 12.0)

    def test_amt_trend_matches_manual(self):
        t = MD.dates.get_loc("2024-06-28")
        f = F.amt_trend(t, short=5, long=60)
        j = MD.symbols.get_loc("000001.SZ")
        A = MD.amt["000001.SZ"].values
        s = np.nanmean(np.where(A[t - 4:t + 1] > 0, A[t - 4:t + 1], np.nan))
        l = np.nanmean(np.where(A[t - 59:t + 1] > 0, A[t - 59:t + 1], np.nan))
        self.assertAlmostEqual(float(f[j]), float(s / l - 1), places=8)


class TestCrossSection(unittest.TestCase):
    def test_zscore_is_standardised(self):
        rng = np.random.default_rng(0)
        x = rng.lognormal(0, 1, 5000)
        z = cs_zscore(x)
        ok = np.isfinite(z)
        self.assertAlmostEqual(float(np.nanmean(z)), 0.0, places=6)
        self.assertAlmostEqual(float(np.nanstd(z)), 1.0, places=6)
        # 去极值：标准化后尾部应明显收窄（对比不 winsorize 的版本）
        raw_z = (x - np.nanmean(x)) / np.nanstd(x)
        self.assertLess(float(np.nanmax(np.abs(z))), float(np.nanmax(np.abs(raw_z))))
        self.assertLess(float(np.nanmax(np.abs(z))), 6.0)

    def test_zscore_handles_nan_and_degenerate(self):
        x = np.array([np.nan, np.nan, np.nan])
        self.assertTrue(np.all(np.isnan(cs_zscore(x))))
        x = np.array([1.0, 1.0, 1.0, 1.0, np.nan])
        z = cs_zscore(x)
        self.assertTrue(np.all(np.isfinite(z[:4])))     # 零方差不应变 NaN
        x = np.array([1.0, 2.0, np.nan])
        self.assertFalse(np.isfinite(cs_zscore(x)[2]))


class TestBuffer(unittest.TestCase):
    """缓冲带：老持仓要跌出 exit_rank 才卖，用来压低换手。"""

    def _mk(self, n_hold=3, buffer_mult=2.0):
        st = Strategy.__new__(Strategy)
        Strategy.__init__(st, n_hold=n_hold)
        st.buffer_mult = buffer_mult
        return st

    def test_no_buffer_is_pure_topn(self):
        st = self._mk(buffer_mult=0.0)
        score = np.array([5.0, 4, 3, 2, 1])
        pick = st._select(np.ones(5, bool), score)
        np.testing.assert_array_equal(pick, np.array([0, 1, 2]))

    def test_buffer_keeps_incumbent(self):
        st = self._mk(n_hold=3, buffer_mult=2.0)
        st._select(np.ones(5, bool), np.array([5.0, 4, 3, 2, 1]))
        # 老持仓 [0,1,2]；新排序让 3、4 冲到前面
        pick = st._select(np.ones(5, bool), np.array([1.0, 2, 3, 9, 8]))
        # 0/1/2 的排名分别是 4/3/2，exit_rank = 6 -> 全部保留
        np.testing.assert_array_equal(pick, np.array([0, 1, 2]))

    def test_buffer_eventually_exits(self):
        st = self._mk(n_hold=2, buffer_mult=1.5)
        st._select(np.ones(5, bool), np.array([5.0, 4, 3, 2, 1]))      # 持有 [0,1]
        pick = st._select(np.ones(5, bool), np.array([0.0, 0, 9, 8, 7]))
        # order = [2,3,4,0,1]，rank[0]=3、rank[1]=4，均 >= exit_rank(3) -> 老持仓全部出局
        np.testing.assert_array_equal(pick, np.array([2, 3]))

    def test_buffer_respects_universe(self):
        st = self._mk(n_hold=2, buffer_mult=2.0)
        st._select(np.ones(4, bool), np.array([4.0, 3, 2, 1]))
        ok = np.array([False, True, True, True])
        pick = st._select(ok, np.array([0.0, 1, 2, 3]))
        self.assertNotIn(0, pick)                        # 掉出股票池必须卖掉


class TestWeightConstruction(unittest.TestCase):
    def _panel(self):
        rng = np.random.default_rng(7)
        dates = MD.dates[sorted(Backtest._rebalance_index(MD.dates, "M"))]
        return pd.DataFrame(rng.standard_normal((len(dates), len(MD.symbols))),
                            index=dates, columns=MD.symbols)

    def test_weights_respect_universe_and_cap(self):
        st = PanelStrategy(self._panel(), n_hold=40, buffer_mult=2.0, max_weight=0.05)
        t = MD.dates.get_loc("2024-06-28")
        w = st.weights(t, MD)
        self.assertGreater(len(w), 0)
        self.assertLessEqual(len(w), 40)
        self.assertLessEqual(float(w.sum()), 1.0 + 1e-12)
        self.assertLessEqual(float(w.max()), 0.05 + 1e-12)
        ok = MD.universe().iloc[t]
        self.assertTrue(bool(ok.reindex(w.index).all()))

    def test_panel_strategy_uses_latest_signal(self):
        """周频调仓时，非面板日应沿用最近一期信号，而不是变成空仓。"""
        p = self._panel()
        st = PanelStrategy(p, n_hold=20, buffer_mult=2.0, max_weight=0.1)
        # 必须取 2021 年以后：默认股票池要求上市满 250 日，2020 年成分是空的
        d0 = p.index[p.index >= "2022-01-01"][2]
        t_mid = int(MD.dates.searchsorted(d0)) + 3      # d0 之后的非面板日
        w_mid = st.weights(t_mid, MD)
        self.assertGreater(len(w_mid), 0)
        np.testing.assert_array_equal(st.score(MD, t_mid), st.score(MD, int(MD.dates.searchsorted(d0))))

    def test_weights_are_causal(self):
        """信号面板只影响 t 及之后，不会倒回去污染已发生的净值。"""
        p = self._panel()
        bt = Backtest(MD, cash=1e7)
        a = bt.run(PanelStrategy(p, n_hold=20, buffer_mult=2.0, max_weight=0.1),
                   start="2023-01-04", end="2023-12-29", rebalance="M")
        p2 = p.copy()
        p2.iloc[len(p2) // 2:] = 0.0                    # 只改后半段信号
        b = bt.run(PanelStrategy(p2, n_hold=20, buffer_mult=2.0, max_weight=0.1),
                   start="2023-01-04", end="2023-12-29", rebalance="M")
        cut = p.index[len(p) // 2]
        self.assertTrue(np.allclose(a.nav[a.nav.index < cut], b.nav[b.nav.index < cut]))


class TestSystemInvariants(unittest.TestCase):
    def test_column_permutation_invariance(self):
        """把证券顺序整体打乱，净值应当不变（浮点求和顺序误差除外）。"""
        rng = np.random.default_rng(1)
        dates = MD.dates[sorted(Backtest._rebalance_index(MD.dates, "M"))]
        panel = pd.DataFrame(rng.standard_normal((len(dates), len(MD.symbols))),
                             index=dates, columns=MD.symbols)
        bt = Backtest(MD, cash=1e7)
        a = bt.run(PanelStrategy(panel, n_hold=20, buffer_mult=2.0, max_weight=0.1),
                   start="2023-01-04", end="2023-12-29", rebalance="M")
        perm = rng.permutation(len(MD.symbols))
        pmd = _PermutedMD(MD, perm)
        b = Backtest(pmd, cash=1e7).run(
            PanelStrategy(panel.iloc[:, perm], n_hold=20, buffer_mult=2.0, max_weight=0.1),
            start="2023-01-04", end="2023-12-29", rebalance="M")
        self.assertLess(abs(a.nav.iloc[-1] - b.nav.iloc[-1]) / a.nav.iloc[-1], 1e-9)

    def test_deterministic(self):
        rng = np.random.default_rng(2)
        dates = MD.dates[sorted(Backtest._rebalance_index(MD.dates, "M"))]
        panel = pd.DataFrame(rng.standard_normal((len(dates), len(MD.symbols))),
                             index=dates, columns=MD.symbols)
        bt = Backtest(MD, cash=1e7)
        a = bt.run(PanelStrategy(panel, n_hold=20, buffer_mult=2.0, max_weight=0.1),
                   start="2023-01-04", end="2023-12-29", rebalance="M")
        b = bt.run(PanelStrategy(panel, n_hold=20, buffer_mult=2.0, max_weight=0.1),
                   start="2023-01-04", end="2023-12-29", rebalance="M")
        self.assertEqual(float(a.nav.iloc[-1]), float(b.nav.iloc[-1]))


class _PermutedMD:
    def __init__(self, md, perm):
        self.symbols = md.symbols[perm]
        self.dates = md.dates
        self._md = md
        self._perm = perm
        for a in ("open", "close", "high", "low", "amt", "num_trades", "listed",
                  "suspended", "one_line_up", "one_line_down", "limit_up_price",
                  "limit_down_price", "cumadj_clean", "adj_close"):
            setattr(self, a, getattr(md, a).iloc[:, perm])

    def universe(self, **kw):
        return self._md.universe(**kw).iloc[:, self._perm]


if __name__ == "__main__":
    unittest.main(verbosity=2)
