# -*- coding: utf-8 -*-
"""04 策略选股器：快速口径 + WebUI 服务端的单元测试。

    python -m unittest tests.test_selector -v
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "03_因子挖掘"), os.path.join(ROOT, "04_策略选股器")):
    if p not in sys.path:
        sys.path.insert(0, p)

import fastsim as fs                                     # noqa: E402
import mining_core as mc                                 # noqa: E402
from core import metrics                                 # noqa: E402

SEL = os.path.join(ROOT, "out", "cache", "selector")
HAS_ARTIFACTS = os.path.exists(os.path.join(SEL, "meta.json"))


class TestRankZ(unittest.TestCase):
    def test_matches_mining_core(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=500)
        x[rng.random(500) < 0.1] = np.nan
        ok = rng.random(500) > 0.2
        a = fs.rank_z(x, ok)                       # 在 ok 掩码内排序
        b = mc.cs_rank_z(np.where(ok, x, np.nan))  # 同一批样本
        m = np.isfinite(a)
        np.testing.assert_allclose(a[m], b[m], atol=1e-12)
        self.assertEqual(int(np.isfinite(a).sum()), int((ok & np.isfinite(x)).sum()))
        # 掩码会改变分母（只在池内排序），与全样本版不同
        self.assertFalse(np.allclose(fs.rank_z(x, ok)[ok], mc.cs_rank_z(x)[ok]))

    def test_range_and_center(self):
        z = fs.rank_z(np.arange(101, dtype="float64"))
        self.assertAlmostEqual(float(z[0]), -1.7320508075688772, places=10)
        self.assertAlmostEqual(float(z[-1]), 1.7320508075688772, places=10)
        self.assertLess(abs(float(np.nanmean(z))), 1e-12)

    def test_ties_get_average_rank(self):
        z = fs.rank_z(np.array([1.0, 2.0, 2.0, 2.0, 5.0]))
        np.testing.assert_allclose(z[1:4], z[1], atol=1e-12)


class TestCompositeScore(unittest.TestCase):
    def test_weight_and_direction(self):
        z = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
        s = fs.composite_score(z, [1.0, 1.0], [1.0, 1.0])
        np.testing.assert_allclose(s, [0.0, 1.0, 2.0])
        s2 = fs.composite_score(z, [1.0, 1.0], [1.0, -1.0])
        np.testing.assert_allclose(s2, [1.0, 1.0, 1.0])

    def test_weight_scale_invariance(self):
        z = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
        a = fs.composite_score(z, [1.0, 3.0], [1.0, 1.0])
        b = fs.composite_score(z, [10.0, 30.0], [1.0, 1.0])
        np.testing.assert_allclose(a, b)


class TestSelectWeights(unittest.TestCase):
    def test_top_n_equal_weight(self):
        score = np.arange(100, dtype="float64")
        ok = np.ones(100, dtype=bool)
        w, pick = fs.select_weights(score, ok, 10, max_weight=0.02)
        self.assertEqual(len(pick), 10)
        self.assertEqual(sorted(pick.tolist()), list(range(90, 100)))
        self.assertAlmostEqual(float(w.sum()), fs.GROSS_LIMIT, places=12)
        np.testing.assert_allclose(w[pick], fs.GROSS_LIMIT / 10, atol=1e-12)

    def test_respects_universe_mask(self):
        score = np.arange(20, dtype="float64")
        ok = np.ones(20, dtype=bool)
        ok[19] = False
        w, pick = fs.select_weights(score, ok, 3)
        self.assertNotIn(19, pick.tolist())

    def test_buffer_keeps_old_holdings(self):
        score = np.arange(100, dtype="float64")
        ok = np.ones(100, dtype=bool)
        prev = np.array([80, 81, 82], dtype="int64")          # 分数中等的老持仓
        w, pick = fs.select_weights(score, ok, 10, prev=prev, buffer=2.0)
        self.assertEqual(len(pick), 10)
        self.assertTrue(set(prev.tolist()).issubset(set(pick.tolist())))
        w2, pick2 = fs.select_weights(score, ok, 10, prev=prev, buffer=1.0)
        self.assertFalse(set(prev.tolist()).issubset(set(pick2.tolist())))

    def test_blacklist_nan_score(self):
        score = np.array([1.0, np.nan, 3.0, 4.0])
        w, pick = fs.select_weights(score, np.ones(4, bool), 3)
        self.assertNotIn(1, pick.tolist())


class TestSimulate(unittest.TestCase):
    def _prices(self, path_open, path_close):
        return np.asarray(path_open, dtype="float64"), np.asarray(path_close, dtype="float64")

    def test_single_symbol_buy_hold(self):
        # 第 0 天开盘价 10 → 收盘 11；第 1 天收盘 12
        ao, ac = self._prices([[10.0], [11.0]], [[11.0], [12.0]])
        plan = {0: np.array([0.98])}
        res = fs.simulate(plan, ao, ac, cash0=1e9, n_hold=1)
        # 第一天买 0.98×1e9 元 @10 元（先按佣金缩放）
        spend = 0.98e9
        units = spend / 10.0
        cash = 1e9 - spend * (1 + fs.FEE_BUY)
        nav0 = cash + units * 11.0
        self.assertAlmostEqual(res["nav"][0], nav0, places=4)
        # 第二天只受价格影响
        self.assertAlmostEqual(res["nav"][1], cash + units * 12.0, places=4)

    def test_liquidation_clears_positions(self):
        ao, ac = self._prices([[10.0], [10.0]], [[10.0], [10.0]])
        plan = {0: np.array([0.98]), 1: np.zeros(1)}
        res = fs.simulate(plan, ao, ac, cash0=1e9, n_hold=1)
        self.assertEqual(int(res["n_holdings"][1]), 0)
        # 只投入 98% 目标仓位：买入付万一、卖出付万一+万五，其余现金不变
        expect = 1e9 - 1e9 * fs.GROSS_LIMIT * (fs.FEE_BUY + fs.FEE_SELL)
        self.assertAlmostEqual(res["nav"][1], expect, places=2)

    def test_band_blocks_tiny_trades(self):
        ao, ac = self._prices([[10.0], [10.0]], [[10.0], [10.0]])
        # 目标权重与已有仓位只差 1 分钱 → 死区挡住，不成交
        plan = {0: np.array([0.0])}
        res = fs.simulate(plan, ao, ac, cash0=1e9, n_hold=1)
        self.assertAlmostEqual(res["nav"][0], 1e9, places=6)

    def test_zero_cash_portfolio_no_negative_nav(self):
        ao, ac = self._prices([[10.0, 20.0], [3.0, 20.0]], [[10.0, 20.0], [3.0, 20.0]])
        plan = {0: np.array([0.49, 0.49])}
        res = fs.simulate(plan, ao, ac, n_hold=2)
        self.assertTrue(np.all(res["nav"] > 0))
        self.assertLess(res["nav"][1], res["nav"][0])          # 一只腰斩，净值下跌


class TestPerfStats(unittest.TestCase):
    def test_matches_core_metrics(self):
        rng = np.random.default_rng(0)
        r = rng.normal(0.0006, 0.012, 500)
        nav = np.concatenate([[1.0], np.cumprod(1 + r)])
        a = fs.perf_stats(nav)
        b = metrics.perf_stats(pd.Series(nav))
        for k in ("累计收益", "年化收益", "年化波动", "夏普", "最大回撤", "卡玛", "日胜率"):
            self.assertAlmostEqual(a[k], b[k], places=9, msg=k)

    def test_nav_ratio_prepends_one(self):
        nav = fs.nav_ratio(np.array([1.1e9, 1.2e9]))
        np.testing.assert_allclose(nav, [1.0, 1.1, 1.2])

    def test_tracking_stats_identical_series(self):
        r = np.array([0.01, -0.005, 0.002] * 100)
        ex = fs.tracking_stats(r, r)
        self.assertAlmostEqual(ex["超额年化"], 0.0, places=12)
        self.assertAlmostEqual(ex["跟踪误差"], 0.0, places=12)


class TestBuildPlan(unittest.TestCase):
    def test_executes_next_trading_day(self):
        sig = ["2025-03-31"]
        trade = ["2025-03-31", "2025-04-01", "2025-04-02"]
        score = np.arange(10, dtype="float64")[None, :]
        uni = np.ones((1, 10), dtype=bool)
        plan, log = fs.build_plan(score, sig, trade, uni, 3, liquidate=False)
        self.assertEqual(sorted(plan), [1])                    # 3-31 收盘信号 → 4-01 开盘执行
        self.assertEqual(log[0]["exec"], "2025-04-01")

    def test_flat_cross_section_skips_rebalance(self):
        """因子当期在横截面上无差异（例如日历因子非事件月恒为 0）→ 不调仓、不按代码顺序乱选。"""
        sig = ["2025-03-31", "2025-04-30"]
        trade = ["2025-03-31", "2025-04-01", "2025-04-30", "2025-05-06"]
        uni = np.ones((2, 10), dtype=bool)
        s1 = np.arange(10, dtype="float64")[None, :]          # 有效信号
        s2 = np.zeros((1, 10))                                # 横截面无差异
        score = np.vstack([s1, s2])
        plan, log = fs.build_plan(score, sig, trade, uni, 3, liquidate=False)
        self.assertEqual(sorted(plan), [1])                   # 只执行第一期
        self.assertIsNone(log[1]["n_hold"])
        self.assertIn("无差异", log[1]["note"])

    def test_flat_first_signal_holds_cash(self):
        score = np.zeros((1, 10))
        plan, log = fs.build_plan(score, ["2025-03-31"],
                                  ["2025-03-31", "2025-04-01"], np.ones((1, 10), bool), 3,
                                  liquidate=False)
        self.assertEqual(plan, {})                            # 没有目标 → 一直空仓，且不会强制清仓
        self.assertIn("空仓", log[0]["note"])

    def test_select_weights_flat_keeps_previous(self):
        score = np.zeros(20)
        prev = np.array([1, 5, 9])
        w, pick = fs.select_weights(score, np.ones(20, bool), 5, prev=prev)
        self.assertEqual(sorted(pick.tolist()), [1, 5, 9])
        self.assertAlmostEqual(float(w.sum()), fs.GROSS_LIMIT, places=12)
        w2, pick2 = fs.select_weights(score, np.ones(20, bool), 5)
        self.assertEqual(len(pick2), 0)
        self.assertEqual(float(w2.sum()), 0.0)

    def test_liquidation_uses_given_signal(self):
        sig = ["2025-03-31"]
        trade = ["2025-03-31", "2025-04-01", "2025-04-02"]
        score = np.arange(10, dtype="float64")[None, :]
        uni = np.ones((1, 10), dtype=bool)
        plan, log = fs.build_plan(score, sig, trade, uni, 3,
                                  liquidate_signal="2025-04-01")
        self.assertIn(2, plan)
        self.assertEqual(float(np.abs(plan[2]).sum()), 0.0)


@unittest.skipUnless(HAS_ARTIFACTS, "缺少 out/cache/selector 产物，先跑 build_ui_data.py")
class TestArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json
        with open(os.path.join(SEL, "meta.json"), encoding="utf-8") as f:
            cls.meta = json.load(f)
        with open(os.path.join(SEL, "benchmarks.json"), encoding="utf-8") as f:
            cls.bench = json.load(f)
        cls.z = np.load(os.path.join(SEL, "z", cls.meta["factors"][0] + ".npy"))
        cls.uni = np.load(os.path.join(SEL, "universe.npy"))

    def test_shapes(self):
        n_sig, n_sym = self.uni.shape
        self.assertEqual(self.z.shape, (n_sig, n_sym))
        self.assertEqual(len(self.meta["trade_dates"]), 424)
        self.assertEqual(len(self.meta["signal_dates"]), n_sig)
        self.assertEqual(len(self.meta["symbols"]), n_sym)

    def test_z_inside_universe(self):
        vals = np.abs(self.z[self.uni])
        self.assertGreater(vals.size, 0)
        self.assertLessEqual(float(vals.max()), 1.7320509)     # ±√3
        self.assertEqual(float(np.abs(self.z[~self.uni]).max()), 0.0)

    def test_benchmarks_aligned(self):
        for k in ("market_ew", "mf1700", "ml1700"):
            self.assertIn(k, self.bench)
            self.assertEqual(len(self.bench[k]["nav"]), len(self.meta["trade_dates"]) + 1)
            self.assertAlmostEqual(self.bench[k]["nav"][0], 1.0, places=9)

    def test_published_benchmark_levels(self):
        # 与 03_因子挖掘 报告一致：池内等权 +36.54%、MF1700_raw +40.83%、ML1700 +54.40%
        self.assertAlmostEqual(self.bench["market_ew"]["nav"][-1], 1.3654, places=3)
        self.assertAlmostEqual(self.bench["mf1700"]["nav"][-1], 1.4083, places=3)
        self.assertAlmostEqual(self.bench["ml1700"]["nav"][-1], 1.5440, places=3)


@unittest.skipUnless(HAS_ARTIFACTS, "缺少 out/cache/selector 产物，先跑 build_ui_data.py")
class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app as webapp
        webapp.ST = webapp.Store()
        webapp.APP.config.update(TESTING=True)
        cls.client = webapp.APP.test_client()

    def test_bootstrap(self):
        r = self.client.get("/api/bootstrap")
        self.assertEqual(r.status_code, 200)
        d = r.get_json()
        self.assertTrue(d["ok"])
        self.assertEqual(len(d["factors"]), 390)
        self.assertIn("families", d["meta"])
        for f in d["factors"]:
            self.assertIsNotNone(f["spark"])
        self.assertNotIn("Infinity", r.get_data(as_text=True))

    def test_run_score_mode(self):
        payload = dict(mode="score", n_hold=200, buffer=1.0,
                       factors=[dict(name="mr_rev20", weight=1.0),
                                dict(name="lq_amihud20", weight=3.0)])
        r = self.client.post("/api/run", json=payload)
        self.assertEqual(r.status_code, 200)
        d = r.get_json()
        self.assertTrue(d["ok"])
        self.assertEqual([c["key"] for c in d["curves"]][:2], ["mine", "factor_avg"])
        self.assertEqual(len(d["curves"][0]["nav"]), len(d["dates"]))
        mine = [s for s in d["stats"] if s["key"] == "mine"][0]
        self.assertTrue(np.isfinite(mine["年化收益"]))
        self.assertEqual(d["n_hold"], 200)
        self.assertEqual(len(d["factors"]), 2)
        self.assertLessEqual(len(d["holdings"]), 20)

    def test_run_allocate_mode_weights_sum_to_one(self):
        payload = dict(mode="allocate", n_hold=200,
                       factors=[dict(name="mr_rev5", weight=9.0), dict(name="mr_rev20", weight=1.0)])
        d = self.client.post("/api/run", json=payload).get_json()
        self.assertTrue(d["ok"])
        self.assertAlmostEqual(sum(d["weights"]), 1.0, places=12)
        self.assertTrue(all(abs(f["weight"]) <= 1 for f in d["factors"]))

    def test_bad_inputs(self):
        for payload, code in [
            (dict(mode="nope", factors=[dict(name="mr_rev5")]), 400),
            (dict(mode="score", factors=[]), 400),
            (dict(mode="score", factors=[dict(name="不存在")]), 400),
            (dict(mode="score", n_hold=5, factors=[dict(name="mr_rev5")]), 400),
        ]:
            r = self.client.post("/api/run", json=payload)
            self.assertEqual(r.status_code, code, payload)
            self.assertFalse(r.get_json()["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)