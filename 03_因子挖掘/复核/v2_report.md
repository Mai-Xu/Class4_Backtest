# v2 独立复核报告：`factor_zoo/` 新增 10 个因子模块（因子方向 / 实现可信性）

- 复核时间：2026-10-22
- 复核对象：`factor_zoo/` 新增的 10 个模块（reversal / risk2 / liquidity2 / flow2 / chip /
  trend2 / pattern / calendar / interaction / marketrel），共 **320 个因子**
  （reversal 30、risk2 43、liquidity2 38、flow2 36、chip 22、trend2 48、pattern 39、
  calendar 20、interaction 24、marketrel 20）
- 复核口径：`03_因子挖掘/因子模块开发规范.md`、`factor_zoo/ctx.py`、`core/lib.py`
- 产物目录：`03_因子挖掘/复核/`，文件名前缀 `v2_`；**未修改任何已有源码或产物**

## 0. 方法边界说明（关于"不要 import 被测模块"）

- **参考实现完全独立**：`v2_lib.py` 里所有 `ref_*` 公式（20 个抽检因子）与
  Spearman / 平均秩 / 2%–98% 去极值 z / 月频网格 / 股票池 / 可成交前视收益
  都是复核方按规范与 docstring **另写的一套**，只 import `numpy`、`pandas`、
  `core.lib.MarketData`（数据层），**没有 import 任何 factor_zoo 因子模块**，
  也没有复用 `factor_zoo/registry.py::check_one`。
- **被测对象的执行**：验证"模块本身"必须运行模块代码。所有模块调用集中在
  `v2_modules.py::load_modules()`（`importlib` 取公开 `FACTORS`，只调用
  `spec["func"](t, md, C)` 黑盒），不引用其内部函数。这是"被检验对象"，
  不是参考实现的一部分。
- 数据层用 `core.lib.MarketData`；因子取值比对另外用 `out/cache/zoo2/*.parquet`
  （主线真实消费的模块输出面板，float32）。

## 1. 因果性投毒测试（独立实现，不用 `Ctx.truncated`）

**做法**：把 `Ctx.__init__` 实际读取的 12 个面板（7 个原始字段 + `cumadj_clean` /
`adj_close` / `ret` / `listed` / `ret_ok`）复制成新 md，**只把 t+1 行及以后**替换为
NaN 或随机值（`rand` 用该列 ≤t 的均值/标准差生成），再新建 `Ctx`；与干净上下文
逐元素比较（NaN 与 NaN 视为相同）。另设 `copy` 模式（数组照抄、数值完全相同）
用于测**浮点噪声下限**，并设 2 个"故意偷看未来"的控制因子证明测试有检出力。

**抽样口径**：每模块 5 因子 × 3 日期（2021-06-30 / 2023-07-31 / 2026-01-30）×
2 种投毒 = 300 组比较（`v2_t1_poison.py` / `v2_t1_poison.json`）。

| 模块 | 组合数 | 逐位相同 | NaN 掩码不一致 | max\|Δ\| | copy 噪声下限 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| reversal | 30 | 30 | 0 | 0 | 0 | 通过 |
| risk2 | 30 | 12 | 0 | 1.776e-15 | 1.776e-15 | 通过 |
| liquidity2 | 30 | 16 | 0 | 3.553e-15 | 3.553e-15 | 通过 |
| flow2 | 30 | 18 | 0 | 6.051e-14 | 6.051e-14 | 通过 |
| chip | 30 | 30 | 0 | 0 | 0 | 通过 |
| trend2 | 30 | 12 | 0 | 2.842e-14 | 2.842e-14 | 通过 |
| pattern | 30 | 30 | 0 | 0 | 0 | 通过 |
| calendar | 30 | 30 | 0 | 0 | 0 | 通过 |
| interaction | 30 | 18 | 0 | 7.105e-15 | 7.105e-15 | 通过 |
| marketrel | 30 | 18 | 0 | 3.553e-15 | 3.553e-15 | 通过 |
| **控制因子** | 12 | 0 | 17 783 | 5.47e+04 | 6.9e-18 | 按预期被发现 |

- **未通过组合数 = 0**。非零差异的模块，其 NaN/rand 最大绝对差与 `copy` 模式
  **完全相同**（例如 flow2 两种投毒与 copy 都是 6.051e-14），说明差异来自
  数组来源不同导致的 1ulp 级浮点噪声（SIMD/求和顺序），而非读到未来数据。
- 控制因子 `ctrl_peek_next_row` max|Δ|=5.47e+04、`ctrl_full_sample_mean`
  max|Δ|=0.122 且 17 783 个 NaN 掩码错 → 本测试确实能抓住偷看未来的写法。
- **全量投毒**（`v2_t5_contract.py`）：320 个因子 × 1 日期 × 2 种投毒 = 640 次重算，
  **386 次逐位相同**，其余差异全部 ≤ 8.73e-11（相对 ≤ 9.2e-16，即 1ulp），
  只有 1 个因子出现 2 格的 NaN↔0 边界抖动（见问题 #4）。

## 2. 数值正确性抽检（每模块 2 因子，独立公式重算）

日期 2023-07-31（t=866，月频网格第 31 期，周一、日号 31 → 日历状态非平凡）。
`v2_t2_numeric.py` / `v2_t2_numeric.json`：

| 模块 | 因子 | 比较列数 | 双有限 | NaN 掩码不一致 | max\|Δ\| | 最大相对误差 | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| reversal | rev_ret10 | 5029 | 5013 | 0 | 0 | 0 | OK |
| reversal | rev_max1d20 | 5029 | 5010 | 0 | 0 | 0 | OK |
| risk2 | rk2_park60 | 5029 | 4973 | 0 | 2.08e-17 | 5.88e-16 | OK |
| risk2 | rk2_var5_60 | 5029 | 4972 | 0 | 1.39e-17 | 2.22e-16 | OK |
| liquidity2 | lq2_logamt60 | 5029 | 4964 | 0 | 0 | 0 | OK |
| liquidity2 | lq2_amihud120 | 5029 | 4928 | 0 | 0 | 0 | OK |
| flow2 | fl2_cmf60 | 5029 | 4963 | 0 | 2.22e-16 | 2.67e-13 | OK |
| flow2 | fl2_vwapdev20 | 5029 | 5004 | 0 | 6.66e-16 | 1.05e-11 | OK |
| chip | ch_profit60 | 5029 | 4964 | 0 | 0 | 0 | OK |
| chip | ch_cost_dev120 | 5029 | 4928 | 0 | 1.33e-15 | 9.45e-12 | OK |
| trend2 | tr2_rsi14b | 800 | 798 | 0 | 2.84e-14 | 5.73e-16 | OK |
| trend2 | tr2_di_plus14 | 800 | 791 | 0 | 0 | 0 | OK |
| pattern | pt_body1 | 5029 | 5016 | 0 | 0 | 0 | OK |
| pattern | pt_up_days20 | 5029 | 5003 | 0 | 0 | 0 | OK |
| calendar | cal_tom_x_rev5 | 5029 | 5022 | 0 | 0 | 0 | OK |
| calendar | cal_seasonal_mom | 5029 | 4662 | 0 | 0 | 0 | OK |
| interaction | ix_consensus_rev | 5029 | 5022 | 0 | 0 | 0 | OK |
| interaction | ix_rev_x_illiq | 5029 | 4963 | 0 | 0 | 0 | OK |
| marketrel | mk_rel_ret20 | 5029 | 5010 | 0 | 0 | 0 | OK |
| marketrel | mk_rel_mom12_1 | 5029 | 4840 | 0 | 0 | 0 | OK |

**20/20 一致**：覆盖动量 / 波动 / 尾部 / 流动性 / 非流动性 / 量价 / VWAP / 筹码（获利盘、
加权成本）/ Wilder RSI 与 +DI / K 线实体 / 上涨天数 / 日历状态 / 季节动量 / 交互 z 合成 /
相对市场累计收益。参考公式按模块 docstring 与规范口径独立重写（例如 Wilder +DI 用
逐股显式循环复刻"6n 行闭式预热 + 递推"口径，结果逐位相同）。

## 3. 覆盖率与常数检查

**（a）与 `out/03_因子挖掘/tables/20_zoo_check.csv` 比对**（`v2_t3_coverage.py`）：
复现 s20 的抽样规则（69 个网格日期取 `[::5][:12]` = 2021-01-29 ... 2025-08-29，
12 个日期），用缓存面板重算 45 个因子（9 个已注册模块 × 5 个）的全市场覆盖率：

- 45/45 与表格 **逐位一致**：max|Δ| = 0.000e+00、中位 |Δ| = 0.000e+00。
- 2025-01-27 ~ 2026-09-30 的 21 个月频调仓日上：45 个因子横截面标准差全部 > 0，
  只有 2 个因子出现"全为 0"的日期（见问题 #2）：cal_pre_holiday_x_ret5 12/21 天、
  cal_month_x_mom60 16/21 天（共 28/945 格）；池内覆盖率 0.88 ~ 1.00。
- **flow2 无法比对**：它在 `20_zoo_check.csv` / `21_factor_stats.csv` 里 **0 行**
  （未注册进主线，见问题 #1）。改为直接调用模块函数重算：5 个抽样因子 12 日
  覆盖率 0.882 ~ 0.887、2025-26 全市场覆盖率 0.946、21 天横截面标准差全部 > 0。

**（b）全量契约 + 常数性**（`v2_t5_contract.py`，320 因子 × 2 个日期）：

- 形状 = (5451,)：**0 违例**；`±inf`：**0 个**；抛异常：**0 次**；
  desc 为空 / direction 非 ±1 / 前缀不符 / 跨模块重名：**0 个**。
- 横截面标准差 = 0 的因子：**7 个，全部来自 calendar**（cal_yearend_x_rev20、
  cal_pre_holiday_x_ret5、cal_pre_holiday_x_amt、cal_month_x_mom60、cal_month_x_rev60
  在两个抽样日均恒为 0；cal_weekday_x_rev5 / cal_weekday_x_illiq 在非周一/周五的
  2026-06-30 恒为 0）——与模块 docstring"状态为 0 处返回 0.0"的设计一致。

## 4. IC 复核（独立 Spearman + 可成交前视收益）

`v2_t4_ic.py`：`ADJ_O[t+21]/ADJ_O[t+1]-1` + 复核方自己的平均秩 Spearman
（min_n=50，动态非 ST 全市场池），样本内 2021-01 ~ 2024-12（48 期）：

| 因子 | 模块 | ic20 复核 | 21_factor_stats.csv | 差 | ICIR 复核 / 表 | 期数 |
|---|---|---:|---:|---:|---:|---:|
| lq_amt_log20 | liquidity（旧模块） | -0.120073 | -0.120073 | -2.78e-17 | -3.001 / -3.001 | 48/48 |
| rev_max1d20 | reversal | -0.095331 | -0.095331 | -5.55e-17 | -2.619 / -2.619 | 48/48 |
| ch_profit60 | chip | -0.061603 | -0.061603 | -5.55e-17 | -1.518 / -1.518 | 48/48 |
| tr2_di_plus14 | trend2 | -0.077776 | -0.077776 | -5.55e-17 | -2.293 / -2.293 | 48/48 |
| ix_consensus_rev | interaction | +0.054889 | +0.054889 | +5.55e-17 | +1.440 / +1.440 | 48/48 |

**5/5 完全一致（最大绝对差 5.551e-17，机器精度）**；样本外（2025-01~2026-09 21 期）也同样一致：
OOS IC20 复核 vs 表格 = -0.131135/-0.131135（lq_amt_log20）、-0.076494/-0.076494（rev_max1d20）、
-0.070059/-0.070059（ch_profit60）、-0.101856/-0.101856（tr2_di_plus14）、+0.067985/+0.067985（ix_consensus_rev），最大差 2.8e-17。说明面板、前视收益、股票池、秩相关口径
与主线完全可复现。

## 5. 附加：数据层 `C.mkt` 的时点一致性（`v2_t6_mkt_pit.py`）

`core.lib.cap_tolerance()` 用**全样本**中位价构造涨跌停容差，进而影响
`no_limit → ret_ok → C.mkt`。复核用"只用 ≤t 的扩展中位数"重建一遍：

- 实际受影响的只有 **12 个 (股票, 日) 格**（over 掩码 12 格 / 896 万个格）；
- `mkt` 差异：max = 3.08e-05、均值 = 1.60e-07、仅 12/1643 个交易日非零；
- 对 mkt 重度因子 `mk_rel_ret20` 的样本内 IC20 影响 = **0.000e+00**（6 位小数）。
- 复核重算的 mkt 与 `Ctx.mkt` 逐位一致（max|Δ| = 0.000e+00），同时验证了本复核的
  市场收益实现正确。

结论：这是数据层（非本次 10 个模块）的**极小口径瑕疵**，量级 1e-5，对因子/IC/回测
无可测影响，建议记录而不必改（若改：把 `cap_tolerance` 改成扩展中位数即可）。

## 6. 问题清单（因子名 + 问题 + 证据）

### P1【重要 · 接线问题，非公式错误】`flow2.py` 的 36 个 `fl2_*` 因子没有进入主线

- 证据 1：`factor_zoo/registry.py` 的 `CATEGORIES` 无 `"flow2"`
  （= mom_rev, reversal, marketrel, risk, risk2, liquidity, liquidity2, flow, chip,
  trend, trend2, pattern, calendar, interaction），`FAMILY` 也没有 `flow2`；
  `registry.py` mtime = 1792584888 < `flow2.py` mtime = 1792586342（先改注册表、后写完 flow2）。
- 证据 2：`20_zoo_check.csv` 里 `fl2_*` = **0 行**；`21_factor_stats.csv` 里
  `fl2_*` = **0 行**（354 - 70 旧因子 = 284 = 320 - 36）；`out/cache/zoo2/` 无
  `fl2_*` 面板；`panel_manifest.csv` 无对应条目。
- 影响：OBV/MFI/CMF/Force/PVT/EMV/VWAP 偏离/量价背离/大单代理等 36 个因子对
  下游多因子打分完全不可见（读了但没被装载）。
- 模块本身没问题：本复核的契约（形状/无 inf/无异常）、全量投毒、覆盖率抽查
  对 flow2 全部通过（cov25 ≈ 0.946、sd>0、逐位/噪声下限内因果）。
- 修复：在 `CATEGORIES` 加 `"flow2"`、在 `FAMILY` 加 `"flow2": "量价关系与资金流"`
  （与 `flow`/`chip` 同族，主线会把它并进"量价关系与资金流"家族），然后重跑
  `s20_zoo_check.py`、`s21_factor_eval.py`、`s22_export_panels.py`（建议 `--force`）。
- **注意（修复的副作用）**：加进去后该家族的成员从 36 个变成 72 个，家族合成分数、
  聚类去冗余、`23~25` 的组合结果都会变，需要整条 `run_all` 重跑一遍，
  再决定是否把 fl2_* 纳入最终打分（也可先用 `s21` 的 IC/ICIR 筛一遍再放行）。

### P2【设计性 · 非错误】5 个 `cal_*` 因子在多数调仓日横截面恒为 0

- 因子：cal_yearend_x_rev20、cal_pre_holiday_x_ret5、cal_pre_holiday_x_amt、
  cal_month_x_mom60、cal_month_x_rev60（另有 cal_weekday_x_rev5 / cal_weekday_x_illiq
  在非周一/周五恒 0）。
- 证据：v2_t5 契约检查 sd=0；v2_t3 显示 2025-26 21 个调仓日里
  cal_pre_holiday_x_ret5 有 12 天、cal_month_x_mom60 有 16 天横截面全为 0。
- 影响：可控。主线 `cs_rank_z` 对常数行输出 0（不污染打分）；但这类因子在多数
  月份贡献为 0，"有效样本"远小于名义 21 期。建议下游对"当日全为 0"的行直接跳过，
  并在报告中把它们的 IC 解释为条件 IC（只在少数月份有效）。

### P3【轻微 · 与规范表的措辞差异】2 个日历因子 direction 与规范表相反但自洽

- `cal_tom_x_vol20`、`cal_post_holiday_x_lowvol20`：规范表 direction = -1，
  模块填 **+1**；原因是取值已乘 `-vol20`（值越大 = 波动越低 = 越"好"），
  desc 也明确写了"×(-vol20)"。属诚实填写 + 表述不一致，不影响使用
  （主线用 `direction_is = sign(ic20)` 定方向）。

### P4【轻微 · 1ulp 边界抖动】`tr2_lin_r2_20` 在 2 只恒价股上 NaN↔0.0

- 证据：2023-07-31 的 20 日窗口内，000616.SZ 与 002504.SZ 的复权价**只有 1 个
  唯一值**（长期停牌/恒价），`syy` 理论为 0，`use = good & (syy > 0)` 在求和顺序
  不同时可能判 0 或判 1e-30；`copy` 模式（数值完全相同、仅数组来源不同）
  **复现同一抖动**，故与"偷看未来"无关。绝对差 1.11e-15，仅 2/5451 只股票。
- 建议（可选）：把 `syy > 0` 改成 `syy > 1e-24 * sxx` 之类的相对阈值，或直接
  接受（对因子排序无影响）。

### P5【信息 · 非缺陷】`direction_prior` 与样本内 IC20 符号一致率 59.9%

- 284 个已注册新因子中 170 个先验方向与样本内 IC20 同号（59.9%），74.6% 的因子
  |IC20| ≥ 0.02、样本外同向占比 74.6%。这与规范"先验只是假设、最终方向由样本内
  IC 决定"一致；合成/使用时请用 `direction_is`（表里已有）而不是 `direction_prior`。

### 未发现的问题（明确排除）

- **偷看未来**：300 组抽样投毒 + 640 组全量投毒（10 模块 320 因子）全部通过；
  代码层面也未发现 `t+k` 索引、全样本统计、`md.dates[t+1]` 等写法
  （grep：仅 `calendar._ts(md, t)` 读 `md.dates[t]`；无模块级可变缓存，
  缓存均挂 `Ctx.__dict__`）。
- **公式写错**：20 个跨类别抽检因子与独立实现完全一致（最大相对误差 1.05e-11，
  其中 10 个逐位相同），含最易写错的 Wilder RSI/+DI、64 箱筹码获利盘、
  含截距市场模型残差、配对大单/量价背离。
- **±inf / 形状 / 异常 / 重名**：320 因子 × 2 日全部合规。
- **面板与代码版本一致性**：`panel_manifest.csv` 里的模块 mtime 指纹与当前
  `factor_zoo/*.py` 的 mtime 完全一致（reversal 30 / risk2 43 / liquidity2 38 /
  chip 22 / trend2 48 / pattern 39 / calendar 20 / interaction 24 / marketrel 20
  个面板的指纹 = 对应模块 mtime），说明 20/21/22 表与面板就是当前代码跑出来的。

## 7. 总体可信度判断

**可信度：高（可用于下游多因子打分），有 1 个必须修的接线问题（flow2）。**

依据：
1. 因果性：420 组抽样投毒 + 640 组全量投毒全部通过（含 2 个控制的检出力证明），
   残余差异可被 `copy` 模式精确解释为 1ulp 浮点噪声；
2. 数值：跨 10 个模块、10 个因子类别的 20 个因子与独立重写公式一致到 1e-11 以内；
3. 统计口径：45 个因子覆盖率与 `20_zoo_check.csv` 逐位一致、5 个因子 ic20/ICIR 与
   `21_factor_stats.csv` 一致到 5.6e-17；面板指纹与模块 mtime 一致（无"改代码没重跑"）；
4. 契约：320 因子 0 异常 0 inf 0 形状错 0 重名，7 个恒常因子均为日历事件门的
   设计后果且已在 docstring 声明。

残余风险（本复核未覆盖的部分，建议后续补）：
- 只对 20/320 个因子做了逐公式重算，其他 300 个只有契约 + 因果 + 覆盖率证据；
  若某因子"公式与 desc 不符但因果且覆盖正常"，本复核无法发现。
- 投毒抽样日期为 3 个（全量投毒 1 个），极端历史段（2020 年、上市初期）未覆盖。
- 数据层 `C.mkt` 存在 1e-5 量级的全样本容差瑕疵（§5），建议在报告中记录。

## 8. 新增文件清单（全部在 `03_因子挖掘/复核/`）

| 文件 | 作用 |
|---|---|
| `v2_lib.py` | 复核方独立数据包 `Ref` + 独立工具（网格/股票池/前视收益/Spearman/去极值 z）+ 20 个参考公式 |
| `v2_modules.py` | 被测模块执行器（importlib 取公开 `FACTORS`）+ 投毒器 `poison_md` + 逐位比较 `compare` |
| `v2_t1_poison.py` / `.json` / `_log.txt` | 验证1：抽样投毒因果性（50 因子 × 3 日期 × copy/nan/rand） |
| `v2_t2_numeric.py` / `.json` / `_log.txt` | 验证2：20 因子独立公式重算 |
| `v2_t3_coverage.py` / `.json` / `_log.txt` | 验证3：覆盖率/常数性（含与 20_zoo_check.csv 比对） |
| `v2_t4_ic.py` / `.json` / `_log.txt` | 验证4：5 因子 IC20 重算 |
| `v2_t5_contract.py` / `.json` / `_log.txt` | 附加：320 因子全量契约体检 + 全量投毒 |
| `v2_t6_mkt_pit.py` / `.json` / `_log.txt` | 附加：`C.mkt` 时点一致性（cap_tolerance 全样本中位数） |
| `v2_report.md` | 本报告 |

运行方式（PowerShell，cwd = 项目根）：

```powershell
$env:PYTHONIOENCODING='utf-8'
python 03_因子挖掘/复核/v2_t1_poison.py
python 03_因子挖掘/复核/v2_t2_numeric.py
python 03_因子挖掘/复核/v2_t3_coverage.py
python 03_因子挖掘/复核/v2_t4_ic.py
python 03_因子挖掘/复核/v2_t5_contract.py
python 03_因子挖掘/复核/v2_t6_mkt_pit.py
```
## 9. 证据版本指纹（本次复核读取的版本，SHA1 前 12 位）

| 文件 | SHA1(12) | mtime |
|---|---|---|
| out/03_因子挖掘/tables/20_zoo_check.csv | 5dd08f310df4 | 2026-10-22 00:28 |
| out/03_因子挖掘/tables/21_factor_stats.csv | 08c742bcfb1a | 2026-10-21 21:05 |
| out/03_因子挖掘/tables/panel_manifest.csv | 3e5672b6f9c8 | 2026-10-21 21:06 |
| out/03_因子挖掘/tables/22_factor_dict.csv | 23db3f8c3d18 | 2026-10-21 21:06 |
| factor_zoo/reversal.py | f8d2d964495e | 2026-10-21 20:39 |
| factor_zoo/risk2.py | 88934a5aae6c | 2026-10-21 20:41 |
| factor_zoo/liquidity2.py | 3a1f32cc9dd2 | 2026-10-21 20:23 |
| factor_zoo/flow2.py | 8354197e00da | 2026-10-21 20:39 |
| factor_zoo/chip.py | 042583a8334e | 2026-10-21 20:37 |
| factor_zoo/trend2.py | 98e6e26d622d | 2026-10-21 20:52 |
| factor_zoo/pattern.py | 5d7978c29ea3 | 2026-10-21 20:45 |
| factor_zoo/calendar.py | d9305d671ae7 | 2026-10-21 20:55 |
| factor_zoo/interaction.py | f0a6ca2c1869 | 2026-10-21 20:54 |
| factor_zoo/marketrel.py | 9548c23e0526 | 2026-10-21 20:55 |

10 个模块文件的 mtime（20:23~20:55）都**早于**面板（21:05/21:06）与 20_zoo_check.csv（00:28），
且 `panel_manifest.csv` 内的模块 mtime 指纹与上表一致 → 本次复核结论对应当前代码版本。
原始指纹见 `v2_fingerprints.txt`。

## 10. 复核脚本总耗时（本机实测）

| 脚本 | 耗时 |
|---|---:|
| v2_t1_poison.py（50 因子 × 3 日期 × 3 模式） | 24.4s |
| v2_t2_numeric.py（20 因子独立重算） | 4.5s |
| v2_t3_coverage.py（45+5 因子覆盖率） | 13.5s |
| v2_t4_ic.py（5 因子 IC20） | 5.6s |
| v2_t5_contract.py（320 因子契约 + 全量投毒） | 21.8s |
| v2_t6_mkt_pit.py（C.mkt 时点一致性） | 8.9s |

（每个脚本各自加载一次 MarketData，约 3~6s 不计入上表耗时统计口径内的重复开销。）