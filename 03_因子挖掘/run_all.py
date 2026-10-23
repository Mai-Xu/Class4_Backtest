# -*- coding: utf-8 -*-
"""一键跑完「03_因子挖掘」全流程（按顺序调用各步脚本）。

    python 03_因子挖掘/run_all.py [--workers 6] [--random 20] [--skip 20,25]

步骤
----
s20 因子库自检（形状/覆盖/因果性）
s21 因子评价（IC/ICIR/衰减/分层/相关性/冗余簇）+ 面板缓存
s22 导出因子字典 + 下游读取 API 说明
s23 多因子打分选股 + 指定口径回测（raw 与规模中性化两套）
s25 稳健性（参数/成本/家族留一/随机对照）
s24 报告 + 看板
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import subprocess
import sys
import time

HERE = _os.path.dirname(_os.path.abspath(__file__))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--random", type=int, default=20)
    ap.add_argument("--skip", default="", help="逗号分隔要跳过的步骤号，如 25")
    a = ap.parse_args(argv)
    skip = {s.strip() for s in a.skip.split(",") if s.strip()}

    W = str(a.workers)
    steps = [
        ("20", "s20_zoo_check.py", ["--dates", "12"]),
        ("21", "s21_factor_eval.py", ["--workers", W]),
        ("22", "s22_export_panels.py", ["--workers", W]),
        ("23", "s23_multifactor_bt.py", ["--workers", W, "--random-controls", str(a.random)]),
        ("23ns", "s23_multifactor_bt.py", ["--workers", W, "--random-controls", str(a.random),
                                           "--neutralize-adv"]),
        ("25", "s25_robustness.py", ["--workers", W, "--random-controls", str(a.random)]),
        ("24", "s24_report.py", []),
    ]
    t0 = time.time()
    for no, script, args in steps:
        if no.rstrip("ns") in skip:
            print("== 跳过 s%s ==" % no)
            continue
        cmd = [sys.executable, _os.path.join(HERE, script)] + args
        print("\n========== s%s: %s ==========" % (no, " ".join(args)))
        t1 = time.time()
        r = subprocess.run(cmd, cwd=_os.path.dirname(HERE))
        print("---- s%s 结束，返回码 %d，用时 %.1fs ----" % (no, r.returncode, time.time() - t1))
        if r.returncode != 0:
            raise SystemExit("步骤 s%s 失败" % no)
    print("\n全流程完成，总用时 %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
