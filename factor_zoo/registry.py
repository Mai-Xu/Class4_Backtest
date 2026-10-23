# -*- coding: utf-8 -*-
"""因子注册表：收集各分类模块的 FACTORS，并做统一校验。

每个分类模块必须暴露一个字典：

    FACTORS = {
        "mr_ret20": dict(func=fn, desc="过去 20 日复权收益", direction=-1),
        ...
    }

其中 fn(t, md, C) -> np.ndarray(len(C.k))，t 为调仓日在 md.dates 中的下标。
`direction` 只是先验方向（-1 = 买低值），最终方向由样本内 IC 决定。
"""
from __future__ import annotations

import importlib

import numpy as np

#: 因子模块加载顺序。新增模块只要加到这里即可被主线自动纳入（缺失的模块自动跳过）。
CATEGORIES = ("mom_rev", "reversal", "marketrel",
              "risk", "risk2",
              "liquidity", "liquidity2",
              "flow", "flow2", "chip",
              "trend", "trend2", "pattern",
              "calendar", "interaction")

#: 模块 -> 家族（下游合成按家族聚合；同一家族的多个模块会被合并成一族）
FAMILY = {
    "mom_rev": "动量与反转", "reversal": "动量与反转", "marketrel": "动量与反转",
    "risk": "波动率与风险", "risk2": "波动率与风险",
    "liquidity": "流动性与交易活跃度", "liquidity2": "流动性与交易活跃度",
    "flow": "量价关系与资金流", "flow2": "量价关系与资金流", "chip": "量价关系与资金流",
    "trend": "趋势与形态", "trend2": "趋势与形态", "pattern": "趋势与形态",
    "calendar": "日历与季节性",
    "interaction": "交互与合成",
}


def load_all(strict: bool = False) -> dict:
    """导入全部分类模块，返回 {因子名: spec}。缺模块自动跳过（可增量开发）。"""
    out = {}
    for mod_name in CATEGORIES:
        try:
            mod = importlib.import_module("factor_zoo." + mod_name)
        except ModuleNotFoundError as e:
            if strict:
                raise
            print("  [registry] 跳过 %s（%s）" % (mod_name, e))
            continue
        got = getattr(mod, "FACTORS", None)
        if not isinstance(got, dict) or not got:
            print("  [registry] %s 未定义非空 FACTORS" % mod_name)
            continue
        for name, spec in got.items():
            if name in out:
                raise ValueError("因子重名：%s（%s 与 %s）" % (name, out[name]["module"], mod_name))
            if not callable(spec.get("func")):
                raise ValueError("因子 %s 缺少可调用的 func" % name)
            s = dict(spec)
            s.setdefault("desc", "")
            s.setdefault("direction", -1)
            s["module"] = mod_name
            s["family"] = FAMILY.get(mod_name, mod_name)
            out[name] = s
    return out


def check_one(name: str, spec: dict, C, ts, tol=0.0) -> dict:
    """单个因子的形状 / 覆盖率 / 因果性检查。"""
    f = spec["func"]
    k = C.k
    rep = dict(factor=name, ok=False, reason="", coverage=0.0, n_valid_days=0,
               causal=True, worst_diff=0.0)
    vals = None
    for t in ts:
        try:
            v = np.asarray(f(int(t), C.md, C), dtype="float64")
        except Exception as e:                       # noqa: BLE001
            rep["reason"] = "%s: %s" % (type(e).__name__, e)
            return rep
        if v.shape != (k,):
            rep["reason"] = "返回形状 %s，应为 (%d,)" % (v.shape, k)
            return rep
        if vals is None:
            vals = []
        vals.append(v)
    V = np.vstack(vals)
    cov = np.isfinite(V).mean()
    rep["coverage"] = float(cov)
    rep["n_valid_days"] = int((np.isfinite(V).sum(1) > 100).sum())
    if rep["n_valid_days"] < max(3, len(ts) // 3):
        rep["reason"] = "有效天数太少（%d/%d）" % (rep["n_valid_days"], len(ts))
        return rep
    # 常量或近常量因子没有横截面信息
    sd = np.nanstd(V, axis=1)
    if np.nanmax(sd) <= 0:
        rep["reason"] = "所有日期都是常数"
        return rep
    # 因果性：把所有数组截到 t 之后必须得到同样的值
    worst = 0.0
    for t in ts[:4]:
        try:
            a = np.asarray(f(int(t), C.md, C.truncated(int(t))), dtype="float64")
        except Exception as e:                       # noqa: BLE001
            rep["reason"] = "truncated 调用失败: %s" % e
            return rep
        b = np.asarray(f(int(t), C.md, C), dtype="float64")
        if a.shape != b.shape:
            rep["causal"] = False
            worst = np.inf
            break
        d = np.nanmax(np.abs(np.nan_to_num(a, nan=-1e18) - np.nan_to_num(b, nan=-1e18))) \
            if len(a) else 0.0
        worst = max(worst, float(d))
    rep["worst_diff"] = worst
    if worst > 1e-9:
        rep["causal"] = False
        rep["reason"] = "疑似使用未来数据（截断后最大差 %.3g）" % worst
        return rep
    rep["ok"] = True
    return rep


def check_all(factors: dict, C, ts) -> "list[dict]":
    return [check_one(n, s, C, ts) for n, s in factors.items()]
