# -*- coding: utf-8 -*-
"""下游「多因子打分选股程序」读取接口（稳定 API，不依赖挖掘脚本）。

    from factor_zoo.api import load_factor_dict, load_panel_matrix, universe_mask
    fd    = load_factor_dict()                    # 因子字典（含样本内 IC/ICIR/方向/簇）
    names = fd.index[fd["is_rep"]].tolist()       # 非冗余代表因子
    X     = load_panel_matrix(names)              # {因子名: DataFrame(信号日 x 证券)}
    uni   = universe_mask()                       # DataFrame(bool)，同日期索引

面板来源：`out/cache/zoo2/zoo2_<因子名>_<mtime>.parquet`（由 `03_因子挖掘/s22_export_panels.py` 生成）。
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL_DIR = os.path.join(ROOT, "out", "cache", "zoo2")
FACTOR_DICT_JSON = os.path.join(PANEL_DIR, "factor_dict.json")
FACTOR_DICT_CSV = os.path.join(ROOT, "out", "03_因子挖掘", "tables", "22_factor_dict.csv")


def load_factor_dict(path: str | None = None) -> pd.DataFrame:
    """因子字典：index=因子名，列含 family/desc/direction_prior/direction_is/ic20/icir/cluster/is_rep。"""
    p = path or (FACTOR_DICT_JSON if os.path.exists(FACTOR_DICT_JSON) else FACTOR_DICT_CSV)
    if not os.path.exists(p):
        raise FileNotFoundError("找不到因子字典，请先跑 03_因子挖掘/s22_export_panels.py：%s" % p)
    if p.endswith(".json"):
        with open(p, encoding="utf-8") as f:
            df = pd.DataFrame(json.load(f))
    else:
        df = pd.read_csv(p)
    return df.set_index("factor")


def panel_path(name: str, cache_dir: str | None = None) -> str:
    d = cache_dir or PANEL_DIR
    hits = sorted(glob.glob(os.path.join(d, "zoo2_%s_*.parquet" % name)))
    if not hits:
        raise FileNotFoundError("没有 %s 的面板缓存，请先跑 s22_export_panels.py" % name)
    return hits[-1]


def load_panel(name: str, cache_dir: str | None = None) -> pd.DataFrame:
    """单个因子面板：index=信号日，columns=证券代码。"""
    return pd.read_parquet(panel_path(name, cache_dir))


def load_panel_matrix(names=None, cache_dir: str | None = None, dtype="float64") -> dict:
    """多个因子面板 -> {因子名: DataFrame}；names=None 时取字典里全部因子。"""
    if names is None:
        names = list(load_factor_dict().index)
    return {nm: load_panel(nm, cache_dir).astype(dtype) for nm in names}


def panel_dates(cache_dir: str | None = None) -> pd.DatetimeIndex:
    """面板的信号日索引（任取一个面板）。"""
    nm = sorted(os.path.basename(p) for p in glob.glob(os.path.join(cache_dir or PANEL_DIR, "zoo2_*.parquet")))[0]
    f = nm[len("zoo2_"):].rsplit("_", 1)[0]
    return load_panel(f, cache_dir).index


def universe_mask(**kw) -> pd.DataFrame:
    """非 ST 全市场动态股票池（与回测同一口径）。

    默认：上市满 250 交易日、20 日均成交额 >= 2000 万、股价 >= 2 元、
    剔除 5% 涨跌幅标的（ST 近似）、无涨跌幅限制日、一字板与停牌。
    """
    from core.lib import MarketData
    md = MarketData()
    return md.universe(**kw)


def cs_rank_z(x) -> np.ndarray:
    """横截面秩 z（0~1 分位线性映射，抗极值）；NaN 保留。"""
    x = np.asarray(x, dtype="float64")
    ok = np.isfinite(x)
    out = np.full(x.shape, np.nan)
    if ok.sum() < 3:
        return out
    v = x[ok]
    order = np.argsort(v, kind="stable")
    r = np.empty(len(v), dtype="float64")
    r[order] = np.arange(len(v), dtype="float64")
    out[ok] = (r / max(len(r) - 1.0, 1.0) - 0.5) * 3.4641016151377544
    return out
