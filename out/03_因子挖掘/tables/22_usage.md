# 下游「多因子打分选股程序」用法

因子面板与字典都在 `out/cache/zoo2/`：

- `zoo2_<因子名>_<模块mtime>.parquet`：`index=调仓信号日(月末最后一个交易日)`、
  `columns=证券代码`、`values=因子原值(float32)`。
- `factor_dict.json` / `../03_因子挖掘/tables/22_factor_dict.csv`：每个因子的家族、口径、
  样本内 IC/ICIR、样本内方向 `direction_is`、冗余簇 `cluster`、是否簇代表 `is_rep`。
- `panel_manifest.csv`：面板文件清单 + sha1。

推荐用法（`factor_zoo/api.py` 已封装）：

```python
from factor_zoo.api import load_factor_dict, load_panel_matrix, universe_mask

fd = load_factor_dict()                     # DataFrame，index=因子名
names = fd[(fd.is_rep) & (fd.ic20.abs() >= 0.02)].index.tolist()
X = load_panel_matrix(names)                # {因子名: DataFrame(日期 x 证券)}
uni = universe_mask()                       # DataFrame(bool)，同一日期索引
```

打分要点：

1. **只用样本内（2021-01 ~ 2024-12）的 IC/ICIR 定方向与权重**，`direction_is` 已经给出符号；
2. 横截面先做秩 z（`mining_core.cs_rank_z`），再按家族合成；
3. 冗余簇（`cluster`）只取代表（`is_rep=True`），避免同一 bet 重复下注；
4. 需要规模中性化时对 `log(ADV20)` 做横截面回归取残差（见 `s23_multifactor_bt.py`）。
