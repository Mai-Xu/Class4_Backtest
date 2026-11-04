# -*- coding: utf-8 -*-
"""ONE-SHOT OOS check + 04_策略选股器 fastsim backtest.

Refuses to run unless freeze.json exists. Refuses to overwrite an existing OOS
report unless --force is explicit; by default the sample is consumed once.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import pandas as pd
import evo_core as e

ROOT = e.ROOT
sys.path.insert(0, str(ROOT / "04_策略选股器"))
import fastsim


def oos_ic(genome, data, dirs):
    idx = e.idx_for(data["panel_dates"], *e.OOS)
    st, turn, ic = e.stats_for(idx, genome, data, dirs)
    return st, turn, ic, idx


def fastsim_run(genome, data, dirs):
    meta = json.loads((ROOT / "out/cache/selector/meta.json").read_text(encoding="utf-8"))
    sig = [pd.Timestamp(d) for d in meta["signal_dates"]]
    sig_idx = np.array([data["panel_dates"].get_loc(d) for d in sig], dtype=int)
    scores = e.score_genome(genome, sig_idx, data, dirs)
    uni = data["uni"][sig_idx]
    plan, _ = fastsim.build_plan(scores, meta["signal_dates"], meta["trade_dates"], uni,
                                 n_hold=meta["is_n_hold"], buffer=1.0,
                                 liquidate_signal=meta["liq_signal"])
    px = np.load(ROOT / "out/cache/selector/prices_oos.npz")
    sim = fastsim.simulate(plan, px["adj_open"], px["adj_close"], n_hold=meta["is_n_hold"])
    nav = fastsim.nav_ratio(sim["nav"])
    stats = fastsim.perf_stats(nav)
    stats["年化换手"] = fastsim.turnover_ann(sim["turnover"])
    return stats, nav, sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    freeze_path = e.OUT / "freeze.json"
    if not freeze_path.exists():
        raise SystemExit("freeze.json 不存在：先完成训练/验证并冻结，禁止直接看 OOS")
    report_path = e.OUT / "oos_report.md"
    if report_path.exists() and not args.force:
        raise SystemExit("OOS 报告已存在；默认不重复消费样本。确需重跑请显式 --force。")
    t0 = time.time(); freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    data = e.build_data(); train_idx = e.idx_for(data["panel_dates"], *e.TRAIN)
    dirs, _, pool = e.directions_on_train(data, train_idx)
    roles = {"evolved_main": freeze["selection"]["main"]["genome"],
             "train_icir8_baseline": freeze["selection"]["train_icir8_baseline"]["genome"],
             "random_best": freeze["selection"]["random_best"]["genome"]}
    rows, navs = [], {}
    for role, g in roles.items():
        st, turn, ic, idx = oos_ic(g, data, dirs)
        fs, nav, sim = fastsim_run(g, data, dirs)
        navs[role] = nav
        rows.append(dict(role=role, n_active=len(g), oos_ic=st["ic"], oos_icir=st["icir"], oos_t=st["t"],
                         oos_pos=st["pos"], oos_turn=turn, cagr=fs["年化收益"], sharpe=fs["夏普"],
                         mdd=fs["最大回撤"], ann_turnover=fs["年化换手"]))
    bench = json.loads((ROOT / "out/cache/selector/benchmarks.json").read_text(encoding="utf-8"))
    main_nav = navs["evolved_main"]
    market = np.asarray(bench["market_ew"]["nav"], dtype=float)
    mf = np.asarray(bench["mf1700"]["nav"], dtype=float)
    ml = np.asarray(bench["ml1700"]["nav"], dtype=float)
    main_stats = fastsim.perf_stats(main_nav)
    super_market = fastsim.tracking_stats(main_nav[1:] / main_nav[:-1] - 1,
                                          market[1:] / market[:-1] - 1)
    main_stats["超额年化_vs_market"] = super_market["超额年化"]
    main_stats["IR_vs_market"] = super_market["信息比率"]
    table = pd.DataFrame(rows)
    table.to_csv(e.OUT / "oos_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"date": bench["dates"], "evolved_main": main_nav[1:],
                  "market_ew": market[1:], "mf1700": mf[1:], "ml1700": ml[1:]}).to_csv(
        e.OUT / "oos_nav.csv", index=False, encoding="utf-8-sig")
    fd = e.api.load_factor_dict()
    fac_lines = []
    for k, w in sorted(freeze["selection"]["main"]["genome"].items(), key=lambda kv: -kv[1]):
        desc = str(fd.loc[k, "desc"]) if k in fd.index else ""
        fac_lines.append(f"- `{k}`  w={w:.2f} — {desc}")
    def tbl(df):
        return "\n".join(["| "+" | ".join(df.columns)+" |", "| "+" | ".join(["---"]*len(df.columns))+" |"] +
                         ["| "+" | ".join(str(v) for v in row)+" |" for row in df.itertuples(index=False)])
    txt = f"""# Class4 自进化因子组合：OOS 一次性检验

- 冻结时间：{freeze['created_at']}
- OOS：{e.OOS[0]} ~ {e.OOS[1]}（只评估一次；结果不回改选择）
- 主候选：{len(roles['evolved_main'])} 因子；验证 ICIR={freeze['selection']['main']['valid']['icir']:.3f}
- 匹配随机候选：{len(roles['random_best'])} 因子
- 执行口径：`04_策略选股器/fastsim.py`，10 亿资金，N=1700，月频，万一佣金+万五印花税，未含涨跌停/参与率逐笔约束

## OOS 因子统计与快速回测

{tbl(table)}

## 主候选因子

{chr(10).join(fac_lines)}

## 基准对照

| 曲线 | 年化 | 夏普 | 最大回撤 |
| --- | --- | --- | --- |
| 主候选（evolved_main） | {main_stats['年化收益']:.2%} | {main_stats['夏普']:.2f} | {main_stats['最大回撤']:.2%} |
| 全市场等权 | {fastsim.perf_stats(market)['年化收益']:.2%} | {fastsim.perf_stats(market)['夏普']:.2f} | {fastsim.perf_stats(market)['最大回撤']:.2%} |
| MF1700 线性基准 | {fastsim.perf_stats(mf)['年化收益']:.2%} | {fastsim.perf_stats(mf)['夏普']:.2f} | {fastsim.perf_stats(mf)['最大回撤']:.2%} |
| ML1700 树模型基准 | {fastsim.perf_stats(ml)['年化收益']:.2%} | {fastsim.perf_stats(ml)['夏普']:.2f} | {fastsim.perf_stats(ml)['最大回撤']:.2%} |

## 读法（不做事后改选）

1. OOS 是**一次性揭盲**。若结果好，只说明这条路径值得进入逐笔引擎；若变差，也不能回头在同一 OOS 上重选。
2. 21 个月的样本外年化标准误约 ±4pp；不要用 1–2pp 差异做强结论。
3. `fastsim` 不含涨跌停/停牌/参与率约束，年化通常比逐笔引擎高约 0.1–0.3pp；正式验收应跑 `core.engine`。
4. 进化 vs 匹配随机若只差很小，则真正的改进来自“预算约束下的稳健筛选”，不是神秘的自我进化。
"""
    report_path.write_text(txt, encoding="utf-8")
    print(txt.encode("gbk", errors="replace").decode("gbk")); print("elapsed", time.time()-t0)

if __name__ == "__main__": main()
