# -*- coding: utf-8 -*-
"""组合策略选股器 · WebUI 服务端（Flask，纯本地）。

    python 04_策略选股器/app.py                 # 默认 http://127.0.0.1:8770
    python 04_策略选股器/app.py --port 9000 --no-browser

接口
----
- `GET  /`               单页 WebUI
- `GET  /api/bootstrap`  口径信息 + 390 个因子字典（含样本内历史表现）+ 3 条基准曲线
- `POST /api/run`        {mode, factors:[{name,weight}], n_hold, buffer} -> 净值曲线 + 绩效 + 持仓

两种模式
--------
- `score`（打分合成）：Σ wᵢ·方向ᵢ·zᵢ → 取分数前 N 只等权 —— 一个组合、一份资金。
- `allocate`（资金分配）：每个因子各自跑一个「前 N 只等权」策略，资金按权重分给各策略，
  组合净值 = Σ wᵢ × 净值ᵢ（等价于把资金分账户执行，权重随净值自然漂移）。

服务端只读 `out/cache/selector/`（由 `build_ui_data.py` 生成），不依赖 core.lib / 原始行情。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import webbrowser

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import fastsim as fs                                     # noqa: E402

SEL = os.path.join(ROOT, "out", "cache", "selector")
WEB = os.path.join(HERE, "web")
MAX_FACTORS = 20
Z_CACHE_LIMIT = 160

try:
    from flask import Flask, jsonify, request, send_from_directory
except ImportError:                                     # pragma: no cover
    print("需要 Flask：pip install flask", file=sys.stderr)
    raise


class Store:
    """紧凑产物 + 惰性加载的因子 z 面板。"""

    def __init__(self, sel=SEL):
        self.dir = sel
        need = ["meta.json", "factor_meta.json", "benchmarks.json",
                "universe.npy", "prices_oos.npz", os.path.join("z")]
        miss = [p for p in need if not os.path.exists(os.path.join(sel, p))]
        if miss:
            raise FileNotFoundError(
                "缺少预计算产物 %s（%s）。请先运行：python 04_策略选股器/build_ui_data.py"
                % (miss, sel))
        self.meta = self._json("meta.json")
        self.fm = self._json("factor_meta.json")
        self.bench = self._json("benchmarks.json")
        self.uni = np.load(os.path.join(sel, "universe.npy"))
        z = np.load(os.path.join(sel, "prices_oos.npz"))
        self.adj_open = z["adj_open"].astype("float64")
        self.adj_close = z["adj_close"].astype("float64")
        self.symbols = np.asarray(self.meta["symbols"])
        self.signal_dates = self.meta["signal_dates"]
        self.trade_dates = self.meta["trade_dates"]
        self._zc: dict = {}
        self._zq: list = []

    def _json(self, name):
        with open(os.path.join(self.dir, name), encoding="utf-8") as f:
            return json.load(f)

    def z(self, name):
        if name not in self._zc:
            a = np.load(os.path.join(self.dir, "z", name + ".npy")).astype("float64")
            self._zc[name] = a
            self._zq.append(name)
            if len(self._zq) > Z_CACHE_LIMIT:
                self._zc.pop(self._zq.pop(0), None)
        return self._zc[name]


ST = None
APP = Flask(__name__, static_folder=None)
APP.json.ensure_ascii = False          # 中文直出，别转义成 \uXXXX


def json_safe(o):
    """把 NaN / ±Infinity / numpy 标量转成 JSON 合法值（前端 JSON.parse 不接受 Infinity）。"""
    if isinstance(o, dict):
        return {k: json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if np.isfinite(f) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return o


def J(payload, status=200):
    resp = jsonify(json_safe(payload))
    return (resp, status) if status != 200 else resp


# ------------------------------------------------------------------ 计算
def _plan_and_sim(score, n_hold, buffer):
    plan, log = fs.build_plan(score, ST.signal_dates, ST.trade_dates, ST.uni, n_hold,
                              buffer=buffer, liquidate_signal=ST.meta["liq_signal"])
    res = fs.simulate(plan, ST.adj_open, ST.adj_close, n_hold=n_hold)
    return plan, log, res


def _last_informative(score, uni, note_flat=False):
    """最近一个横截面有差异的信号日（因子当期为常数时不能用来选股/展示）。

    返回 (信号日下标, 候选下标, 是否为退化的最近一期)。
    """
    for i in range(len(uni) - 1, -1, -1):
        cand = np.flatnonzero(uni[i] & np.isfinite(score[i]))
        if cand.size and float(np.ptp(np.asarray(score[i])[cand])) > 0:
            return i, cand, (i < len(uni) - 1)
    return None, None, True


def _top_holdings(score, uni, n_hold, k=20, as_of=None):
    """按最近有效信号日取打分前 k 名（附：该信号日是否不是最后一期）。"""
    i, cand, stale = _last_informative(score, uni)
    if i is None:
        return [], None
    order = cand[np.argsort(-np.asarray(score[i])[cand], kind="stable")][:k]
    n_pick = max(min(int(n_hold), int(cand.size)), 1)
    wt = fs.GROSS_LIMIT / n_pick
    out = [dict(symbol=str(ST.symbols[j]), score=float(score[i][j]), weight=float(wt))
           for j in order]
    date = ST.signal_dates[i]
    note = "最近一期信号 %s" % date
    if stale:
        note += "（该期横截面无差异，组合保持上一期持仓）"
    if as_of:
        note = as_of % note
    return out, note


def _curve_stats(nav, ret, bench_ret, turnover=None):
    st = fs.perf_stats(nav, ret)
    ex = fs.tracking_stats(ret, bench_ret)
    st.update(dict(超额年化=ex["超额年化"], 跟踪误差=ex["跟踪误差"], 信息比率=ex["信息比率"]))
    st["年换手"] = float(np.sum(turnover) / (len(ret) / fs.T252)) if turnover is not None else None
    return st


def run_request(payload):
    t0 = time.time()
    mode = str(payload.get("mode", "score"))
    if mode not in ("score", "allocate"):
        raise ValueError("mode 只能是 score 或 allocate")
    items = payload.get("factors") or []
    if not items:
        raise ValueError("请至少选择一个因子")
    if len(items) > MAX_FACTORS:
        raise ValueError("一次最多选 %d 个因子" % MAX_FACTORS)
    n_hold = int(payload.get("n_hold", 1700))
    if not (30 <= n_hold <= 3000):
        raise ValueError("持仓只数需要在 30 ~ 3000 之间")
    buffer = float(payload.get("buffer", 1.0) or 1.0)
    buffer = min(max(buffer, 1.0), 2.0)

    names, weights, signs = [], [], []
    for it in items:
        nm = str(it.get("name"))
        if nm not in ST.fm:
            raise ValueError("未知因子：%s" % nm)
        w = float(it.get("weight", 1.0))
        if not np.isfinite(w) or w <= 0:
            raise ValueError("因子 %s 的权重必须为正数" % nm)
        names.append(nm)
        weights.append(w)
        signs.append(float(ST.fm[nm]["direction"]))
    w = np.asarray(weights, dtype="float64")
    w = w / w.sum()
    signs = np.asarray(signs, dtype="float64")

    # ---------- 每个因子各跑一条 sleeve（对照组：所选因子等权；也是模式 B 的构件） ----------
    sleeves = []
    for nm, wi, si in zip(names, w, signs):
        Zi = ST.z(nm)
        _, log_i, res_i = _plan_and_sim(si * Zi, n_hold, buffer)
        nav_i = fs.nav_ratio(res_i["nav"])
        sleeves.append(dict(name=nm, weight=float(wi), nav=nav_i,
                            ret=nav_i[1:] / nav_i[:-1] - 1.0,
                            turnover=res_i["turnover"], log=log_i))
    idx = ST.meta["symbols"]  # noqa: F841  (占位，便于阅读)

    # ---------- 我的策略 ----------
    holdings, plan_log, mine_hold_src, plan_src = [], [], None, "打分合成（每月一次月频调仓）"
    if mode == "score":
        Z = np.stack([ST.z(nm) for nm in names])                 # (k, 21, n_sym)
        score = fs.composite_score(Z, w, signs)
        plan, log, res = _plan_and_sim(score, n_hold, buffer)
        nav_mine = fs.nav_ratio(res["nav"])
        ret_mine = nav_mine[1:] / nav_mine[:-1] - 1.0
        turn_mine = res["turnover"]
        plan_log = log
        holdings, mine_hold_src = _top_holdings(
            score, ST.uni, n_hold, as_of="合成分数前 20 名 · %s")
    else:
        navs = np.stack([s["nav"] for s in sleeves])
        nav_mine = (navs * w[:, None]).sum(axis=0)               # 资金分账户：组合净值 = Σ wᵢ × 净值ᵢ
        ret_mine = nav_mine[1:] / nav_mine[:-1] - 1.0
        turn_mine = np.sum([s["turnover"] * wi for s, wi in zip(sleeves, w)], axis=0)
        big = int(np.argmax(w))
        nm = sleeves[big]["name"]
        plan_log = sleeves[big]["log"]
        plan_src = "最大仓位子策略「%s」（其余 %d 个子策略的调仓节奏相同）" % (nm, len(sleeves) - 1)
        s_big = signs[big] * ST.z(nm)
        holdings, mine_hold_src = _top_holdings(
            s_big, ST.uni, n_hold,
            as_of="最大仓位子策略「" + nm + "」的打分前 20 名 · %s")

    # ---------- 对照曲线 ----------
    nav_avg = np.stack([s["nav"] for s in sleeves]).mean(axis=0)  # 所选因子等权
    ret_avg = nav_avg[1:] / nav_avg[:-1] - 1.0
    bench_dates = ST.bench["dates"]
    bench_curves = {k: np.asarray(v["nav"], dtype="float64")
                    for k, v in ST.bench.items() if isinstance(v, dict)}
    ew = bench_curves["market_ew"]
    ew_ret = ew[1:] / ew[:-1] - 1.0

    series = [("mine", "我的策略", nav_mine, ret_mine, turn_mine, 3.0),
              ("factor_avg", "所选因子等权", nav_avg, ret_avg, None, 1.8),
              ("market_ew", ST.bench["market_ew"]["label"], ew, ew_ret, None, 2.0),
              ("mf1700", ST.bench["mf1700"]["label"] if "mf1700" in ST.bench else "最佳线性打分",
               bench_curves.get("mf1700"), None, None, 1.8),
              ("ml1700", ST.bench["ml1700"]["label"] if "ml1700" in ST.bench else "最佳打分模型",
               bench_curves.get("ml1700"), None, None, 2.0)]

    curves, stats = [], []
    for key, label, nav, ret, turn, lw in series:
        if nav is None:
            continue
        if ret is None:
            ret = nav[1:] / nav[:-1] - 1.0
        st = _curve_stats(nav, ret, ew_ret, turn)
        curves.append(dict(key=key, label=label, nav=[round(float(x), 6) for x in nav]))
        stats.append(dict(key=key, label=label, **{k: (None if v is None or not np.isfinite(v) else float(v))
                                                   for k, v in st.items()}))

    # ---------- 分年度 / 月度 / 回撤 ----------
    import pandas as pd
    dates = pd.DatetimeIndex([ST.meta["base_date"]] + ST.trade_dates)
    annual = {}
    for key, label, nav, *_ in series:
        if nav is None:
            continue
        s = pd.Series(nav, index=dates)
        annual[key] = {str(y): float((1 + v.pct_change().dropna()).prod() - 1)
                       for y, v in s.groupby(s.index.year)}
    monthly = {}
    for key, label, nav, *_ in series:
        if nav is None or key not in ("mine", "market_ew", "ml1700", "factor_avg"):
            continue
        s = pd.Series(nav, index=dates)
        r = s.pct_change().dropna()
        monthly[key] = {str(p): float((1 + v).prod() - 1) for p, v in r.groupby(r.index.to_period("M"))}
    dd = {}
    for key, label, nav, *_ in series:
        if nav is None or key not in ("mine", "factor_avg", "market_ew", "ml1700"):
            continue
        dd[key] = [round(float(x), 5) for x in fs.drawdown(nav)]

    # ---------- 因子级明细（sleeve） ----------
    fac_rows = []
    for s, wi in zip(sleeves, w):
        st = fs.perf_stats(s["nav"], s["ret"])
        ex = fs.tracking_stats(s["ret"], ew_ret)
        fm = ST.fm[s["name"]]
        fac_rows.append(dict(name=s["name"], family=fm["family"], weight=float(wi),
                             年化=float(st["年化收益"]), 夏普=float(st["夏普"]),
                             最大回撤=float(st["最大回撤"]), 超额年化=float(ex["超额年化"]),
                             年换手=float(np.sum(s["turnover"]) / (len(s["ret"]) / fs.T252)),
                             ic20=fm.get("ic20"), icir=fm.get("icir")))

    return dict(
        ok=True, mode=mode, n_hold=n_hold, buffer=buffer,
        elapsed_ms=int((time.time() - t0) * 1000),
        dates=[ST.meta["base_date"]] + ST.trade_dates,
        curves=curves, stats=stats, annual=annual, monthly=monthly, drawdown=dd,
        factors=fac_rows, holdings=holdings, holdings_note=mine_hold_src,
        plan=plan_log, plan_src=plan_src, weights=[float(x) for x in w], names=names,
        gross_limit=fs.GROSS_LIMIT,
        final_nav=float(nav_mine[-1]), final_asset=float(nav_mine[-1] * fs.CASH0),
    )


# ------------------------------------------------------------------ 缓存
class LRU:
    def __init__(self, cap=32):
        self.cap, self.d, self.q = cap, {}, []

    def get(self, key):
        return self.d.get(key)

    def put(self, key, val):
        if key not in self.d:
            self.q.append(key)
        self.d[key] = val
        while len(self.q) > self.cap:
            self.d.pop(self.q.pop(0), None)


RUN_CACHE = LRU(48)


# ------------------------------------------------------------------ 路由
@APP.route("/")
def index():
    return send_from_directory(WEB, "index.html")


@APP.route("/<path:fn>")
def static_files(fn):
    if fn.startswith("api/"):
        return J(dict(ok=False, error="not found"), 404)
    return send_from_directory(WEB, fn)


@APP.route("/api/bootstrap")
def api_bootstrap():
    factors = []
    for nm in ST.meta["factors"]:
        m = ST.fm[nm]
        spark = [1.0]
        for _, r in m.get("is_monthly") or []:
            spark.append(spark[-1] * (1.0 + (r or 0.0)))
        factors.append(dict(
            name=nm, family=m["family"], module=m["module"], desc=m["desc"],
            direction=m["direction"], ic20=m["ic20"], icir=m["icir"],
            ls_net_ann=m["ls_net_ann"], mono=m["mono"], coverage=m["coverage"],
            oos_ic20=m["oos_ic20"], cluster=m["cluster"], is_rep=m["is_rep"],
            is_stats=m["is_stats"], spark=[round(float(x), 4) for x in spark]))
    return J(dict(
        ok=True,
        meta=dict(trade_dates=ST.trade_dates, base_date=ST.meta["base_date"],
                  signal_dates=ST.meta["signal_dates"], liq_signal=ST.meta["liq_signal"],
                  is_n_hold=ST.meta["is_n_hold"], cash=ST.meta["cash"],
                  commission=ST.meta["commission"], stamp=ST.meta["stamp"],
                  gross_limit=ST.meta["gross_limit"], universe_size=ST.meta["universe_size"],
                  families=ST.meta["families"], built_at=ST.meta["built_at"],
                  n_symbols=len(ST.meta["symbols"]), max_factors=MAX_FACTORS),
        factors=factors,
        benchmarks=[dict(key=k, label=v["label"], note=v["note"], nav=v["nav"])
                    for k, v in ST.bench.items() if isinstance(v, dict)],
        default=dict(mode="score", n_hold=1700, buffer=1.0),
    ))


@APP.route("/api/run", methods=["POST"])
def api_run():
    payload = request.get_json(force=True, silent=True) or {}
    try:
        key = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    except TypeError:
        key = None
    hit = RUN_CACHE.get(key) if key else None
    if hit is not None:
        out = dict(hit)
        out["cached"] = True
        return J(out)
    try:
        out = run_request(payload)
    except ValueError as e:
        return J(dict(ok=False, error=str(e)), 400)
    except Exception as e:                                # noqa: BLE001
        import traceback
        traceback.print_exc()
        return J(dict(ok=False, error="计算失败：%s" % e), 500)
    out["cached"] = False
    if key:
        RUN_CACHE.put(key, out)
    return J(out)


def main(argv=None):
    global ST
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args(argv)
    ST = Store()
    print("已加载因子 %d 个 / 信号 %d 个 / 交易日 %d 个（产物 %s）"
          % (len(ST.meta["factors"]), len(ST.signal_dates), len(ST.trade_dates), SEL))
    url = "http://%s:%d/" % (a.host, a.port)
    print("WebUI：%s" % url)
    if not a.no_browser and not a.debug:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    APP.run(host=a.host, port=a.port, debug=a.debug, threaded=True)


if __name__ == "__main__":
    main()
