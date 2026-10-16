# factor_zoo/flow.py 审计报告（独立重算）

- 审计对象：`factor_zoo/flow.py`（349 行，sha256 `f12e00f47bf82a3db696e7feb622adc0a64596efa386a86e865d89d609385557`）。
  审计期间**未修改**该文件（文件 LastWriteTime 2026-10-16 21:46:07，早于本次审计开始时间）。
- 审计脚本：`3_因子挖掘/audit/naive_flow.py`（全部对照实现为「逐股票 Python 循环 + 朴素 numpy」，
  不调用 `factor_zoo.flow` 的任何内部函数，只把 `registry.load_all()` 里的因子函数当被比较对象）。
- 完整输出：`3_因子挖掘/audit/naive_flow_output.txt`；项目自检输出：`3_因子挖掘/audit/selfcheck_flow_output.txt`。
- 数据：1643 交易日 × 5451 证券（2020-01-02 ~ 2026-10-16），价格未复权、复权价 = 原始价 × cumadj_clean；
  成交额 `amt`（元）、成交笔数 `num_trades`（flow.py 全文只在 docstring 里出现一次 `num_trades`，**14 个因子没有一个用到笔数**）。
- 对照日期（题目要求 6 个 + 1 个补测，均为交易日）：2020-03-31(t=57)、2020-08-31(t=160)、2021-06-30(t=360)、
  2022-12-30(t=727)、2023-07-31(t=866)、2024-06-28(t=1086)、2025-06-30(t=1328)。
  判定标准：全横截面逐格 `max|diff| ≤ 1e-8` 且 **NaN 位置完全一致**（掩码不一致格数 = 0）。

复跑命令（项目根目录）：

```powershell
$env:PYTHONIOENCODING="utf-8"; python 3_因子挖掘/audit/naive_flow.py
$env:PYTHONIOENCODING="utf-8"; python -m factor_zoo.selfcheck flow
```

## 0. 结论摘要

- **14/14 因子通过独立重算**：7 个日期 × 全横截面（约 3.7k~5.1k 只有效股票/日）逐格比较，
  `max|diff| ≤ 2.84e-14`（纯浮点求和顺序差异），NaN 掩码不一致 **0 格**，输出中 **0 个 inf**。
- 项目自带因果性检验 `python -m factor_zoo.selfcheck flow`：14/14 OK、截断差全部 `0.00e+00`。
  代码走查同样确认所有窗口都用 `C.sl(t, w, ·)` 截到 `<= t`；`_pvdiv_z` 的截面 z-score 只同一天截面；`_chip` 缓存键含 `C.n`，截断上下文不会命中全量上下文。
- **没有发现「公式抄错 / 偷看未来 / 数值算错」级别的问题**。有 5 个需要标注的争议点（见 §4）：
  1) MFI/OBV 对「精确并列日」的处理是硬开关，数值上脆弱（MFI 1e-13 扰动可跳 15~22 分）；
  2) CMF 把一字板日从分子分母同时剔除，金额被整日丢掉（口径选择，未写进对外 desc）；
  3) `_obv_slope` 注释「起点不参与斜率」与实现不符（首点确实参与回归）；
  4) `_pvcorr` 零方差返回 NaN 而不是 0（与 `ctx.cs_zscore` 的常数→0 策略不一致，但实测 0 例）；
  5) `direction` 先验：`fl_amtmom20=-1` 与 `fl_amtmom60=+1` 对同一公式给了相反先验；14 个里 7 个与实测 IC 符号相反（registry 明确说方向只是先验、最终由样本内 IC 决定，故不算代码错）。

## 1. 逐因子结果表

- `max diff` = 7 个对照日期的全横截面最大绝对差（每个日期都单独比过，表里取最大）；
- `NaN 掩码` = 「模块 NaN 而重算非 NaN」或反之的格数，全部为 0；
- `inf` = 模块输出里 ±inf 的格数，全部为 0。

| 因子 | 声称定义（源码 docstring / registry desc） | 重算 max diff | NaN 掩码 | 判定 |
| --- | --- | ---: | :---: | --- |
| fl_pvcorr20 | 20 日 `corr(复权收益 r, Δlog 成交额)`，>0=放量上涨 | 2.22e-16 | 一致 | 通过 |
| fl_pvcorr60 | 60 日同上 | 3.33e-16 | 一致 | 通过 |
| fl_amtmom20 | 20 日成交额加权收益 `Σ(amt·r)/Σamt` | 5.55e-17 | 一致 | 通过 |
| fl_amtmom60 | 60 日同上 | 1.39e-17 | 一致 | 通过 |
| fl_pvdiv60 | 量价背离：60 日价格分位 − 成交额分位（价高量缩） | 0.00e+00 | 一致 | 通过 |
| fl_pvdiv_z20 | 量价背离(连续)：`z(ret20d) − z(dlog成交额20d)` | 8.88e-16 | 一致 | 通过 |
| fl_obvlvl20 | 20 日归一化 OBV `Σsign(r)·amt/Σamt`，成交额替代成交量 | 1.11e-16 | 一致 | 有争议¹ |
| fl_obvslp20 | OBV 20 日斜率 / 日均成交额，成交额替代成交量 | 3.33e-16 | 一致 | 有争议¹ |
| fl_mfi14 | 14 日 `MFI=100−100/(1+正/负资金流)`，资金流=典型价×成交额 | 2.84e-14 | 一致 | 有争议² |
| fl_mfi28 | 28 日 MFI，成交额替代成交量 | 2.84e-14 | 一致 | 有争议² |
| fl_cmf20 | 20 日 CMF `Σ(CLV·amt)/Σamt`，`CLV=((C−L)−(H−C))/(H−L)` | 2.78e-16 | 一致 | 有争议³ |
| fl_chip_prof250 | 获利盘比例：250 日成本低于现价的成交额占比 | 1.09e-14 | 一致 | 通过⁴ |
| fl_chip_dev250 | 现价 / 250 日加权平均成本 − 1 | 1.13e-14 | 一致 | 通过⁴ |
| fl_chip_conc250 | 250 日成本集中度（归一化 Herfindahl） | 1.14e-14 | 一致 | 通过⁴ |

¹ 对「当日收益恰为 0」是硬开关（实盘 20 日窗口内 R==0 的格数占有效日 2.0%~3.4%，每日期 1502~3525 格），
  见 §4-P1。代数公式本身与 desc 完全一致。
² 对「典型价恰不变」是硬开关，实盘每 14 日窗口 290~738 个精确并列、273~617 只股票，见 §4-P1。
³ 一字板（H==L 且有成交）整日从分子**和分母**剔除；该选择写在 `_cmf` 的行内注释里，
  但没写进对外 desc，见 §4-P5。
⁴ 复权口径正确、分桶实现与声明一致；网格截断与 5% 桶宽是已声明的近似，实测影响见 §3-4。

备注：`fl_chip_prof250` 的 `maxrel≈5.4e-5` 出现在 `prof≈1e-11` 的极小值格上（绝对差 1.1e-14），不是相对意义上的偏差；
其余所有因子的 `maxrel ≤ 1e-11`。

## 2. 逐因子：声称定义 ↔ 实现细节

约定：`C.sl(t, w, X)` 的窗口是 **`[t-w+1, t]`（含当日）**，`lo = max(t-w+1, 0)`（历史不足时窗口自动变短）；
`_ok_window = listed & (amt > 0)`（停牌 amt=0 与未上市都在窗口里被剔除）。

1. **fl_pvcorr20/60**（`_pvcorr(t,C,w)`）
   - 声称：过去 w 日 `corr(r, Δlog amt)`；w=20/60。
   - 实现：窗口内**相邻两日对**（共 w−1 对，首日收益 `R[lo]` 不参与），要求两日都 listed 且 amt>0、r 与 dlog 有限；
     样本数门槛 `n >= max(10, 0.6w)`（20 日→12 对，60 日→36 对）；相关系数经 `_safe_div` 后 clip 到 [−1,1]。
   - 口径：成交额（非笔数）；复权收益 `R`；零方差时 `den=0 → NaN`（不是 0，见 §4-P4）。
2. **fl_amtmom20/60**（`_amtmom`）
   - 声称：`Σ(amt·r)/Σamt`。
   - 实现：窗口内 listed 且 amt>0 且 r 有限的日；门槛 `n >= max(5, 0.6w)`（12/36 天）；分母是该子集的实际成交额之和。
3. **fl_pvdiv60**（`_pvdiv_rank`，w=60）
   - 声称：60 日价格分位 − 成交额分位（价高量缩）。
   - 实现：在窗口内 `listed & amt>0 & C 有限` 的交易日上算经验分位，**含当日自身**，排名数用 `≤` 计数，
     归一化 `(count−1)/(cnt−1)`；门槛 `cnt >= max(20, 0.6·60)=36`；要求当日复权价有限且 >0、当日 amt 有限。
     停牌日（当日 amt=0）不剔除 t 本身，只剔除窗口内其它停牌日。
4. **fl_pvdiv_z20**（`_pvdiv_z`，w=20）
   - 声称：`z(ret20d) − z(dlog成交额20d)`（截面 z）。
   - 实现：`t < 40` 直接全 NaN；近 20 日平均成交额 vs 再前 20 日平均成交额（各自只对 amt>0 的日求均值），
     `dlog = log(m_now/m_pv)`；`ret = C[t]/C[t−20] − 1`；然后 `cs_zscore(ret) − cs_zscore(dlog)`
     （截面 2%/98% 去极值 + 总体标准差 z；该 z 是**同日横截面**，不引入未来）。
5. **fl_obvlvl20**（`_obv_lvl`）
   - 声称：`Σ sign(r)·amt / Σamt`（成交额替代成交量）。
   - 实现：窗口内 listed & amt>0 & r 有限的日；`sign(0)=0`（持平日分子记 0、分母仍计入 amt）；
     门槛 `n >= max(5, 12)=12`。
6. **fl_obvslp20**（`_obv_slope`）
   - 声称：OBV 20 日斜率 / 日均成交额。
   - 实现：窗口内 `flow = sign(r)·amt`（非有效日记 0），`Y = cumsum(flow)`；
     对日序 x（中心化）做掩码最小二乘，`slope = Sxy/Sxx`，再除以有效日均成交额；门槛 `n >= max(8,12)=12`。
     窗口**首个有效点确实参与回归**（注释「起点不参与斜率」不准确，见 §4-P3）。
7. **fl_mfi14/28**（`_mfi(t,C,n)`）
   - 声称：`100 − 100/(1 + 正/负资金流)`，资金流 = 典型价 × 成交额。
   - 实现：取 `n+1` 行窗口（即 n 个典型价差分）；典型价 `tp=(ADJ_H+ADJ_L+C)/3`（**复权**）；
     有效对要求两日都 listed & amt>0 & tp 有限且 >0；`mf = tp·amt`；`Δtp>0` 进正流、`Δtp<0` 进负流、
     `Δtp==0` **两边都不计**；门槛 `有效对数 >= 0.6n`（14→9 对，28→17 对）；
     只有正流时=100，两流全 0 时=NaN，最后 clip 到 [0,100]。
8. **fl_cmf20**（`_cmf`）
   - 声称：`CMF = Σ(MLF)/Σamt`，`MLF = ((C−L)−(H−C))/(H−L) × amt`。
   - 实现：窗口内 listed & amt>0 & `H>L` & C 有限；乘数 clip 到 [−1,1]；`H==L` 的日子**分子分母同时剔除**；
     门槛 `n >= max(8,12)=12`。
9. **fl_chip_prof250 / dev250 / conc250**（`_chip_calc`，一次分桶三个输出）
   - 声称：250 日成交额按**复权** [low, high] 对数价格区间均匀铺到价格桶，派生获利盘/成本偏离/集中度。
   - 实现：窗口 `[t−249, t]`（含当日）；每日 `u=log(price/当日复权收盘)`，把 `[log(L/px), log(H/px)]` 截断到 ±1.5
     （价格比 0.223~4.482），该日全部成交额按区间长度均匀落在 60 个桶（桶宽 0.05 对数 ≈ 5% 相对价距）；
     一字板/退化日整日进单桶；要求窗口内有效日 `>= max(60, 125)=125` 且总权重 >0。
     - `prof`：u<0（成本低于现价）的成交额占比，clip [0,1]；
     - `dev = exp(−Σ(hist·桶中心)/Σhist) − 1 = 现价/几何平均成本 − 1`，clip [−0.95,100]；
     - `conc = (HHI − 1/60)/(1 − 1/60)`，clip [0,1]。
   - 复权/未复权：**用复权价**（`ADJ_H/ADJ_L` 与复权收盘 `C`），量纲一致；§3-4 有反事实量化。

## 3. 逐条易错点检查（是/否 + 证据）

### 3-1 OBV / MFI / CMF 的成交量替代：是否标注？替代后是否自洽？
**标注：是。** `flow.py` 顶部 docstring（第 6~10 行）明确写「本数据集没有成交量……OBV / MFI / CMF / VWAP 等经典
『量价』指标一律用成交额 amt 替代 volume」，并且 `fl_obvlvl20 / fl_obvslp20 / fl_mfi14 / fl_mfi28 / fl_cmf20`
的 desc 都带「成交额替代成交量」。

**自洽性：基本自洽，但 MFI 的资金流口径与「按金额」口径有可测量的差别。**
- OBV 水平/斜率、CMF 都是「分子分母同量纲相除」：OBV 的 amt 在 `Σsign·amt / Σamt` 里约掉，
  CMF 的 amt 在 `ΣCLV·amt/Σamt` 里约掉；替代只改变**各日权重按价格水平倾斜**的程度。
- MFI 不是比值形式：`mf = tp·amt`，而经典式是 `tp·volume`，用 `amt ≈ volume·price` 代入会变成 `tp²·volume`，
  即**价格水平被平方加权**。实测（探针 [H]，7 日期）：
  - 用「纯金额」口径 `mf=amt` 对照：MFI 中位差 0.34~0.43、p95 1.2~1.7、最大 4.1~7.5（满分 100），秩相关 0.999；
  - 用成交量粗代理 `V=amt/未复权收盘` 对照：MFI 中位差 0.52、最大 9.1，秩相关 ≥0.9989；
    OBV 两因子中位差 0.010、最大 0.21；CMF 中位差 0.006、最大 0.13。
  → 结论：替代已在 docstring 标注、公式内部同口径（典型价与成交额同一天、同为复权口径），
  但「tp×amt」相对经典定义多乘了一次价格，窗口内价格波动越大偏差越大；现有数据上对横截面排序影响很小（秩相关 ≥0.997）。
  **修正建议**：若要与经典 MFI 完全同口径，用 `mf = amt`（金额流）或 `mf = tp·amt/VWAP`（成交量流）并在 desc 注明。
  审计判定：**有争议（但已标注、已量化，非实现错误）**。

### 3-2 CMF 的 `((C−L)−(H−C))/(H−L)` 在 H==L 时会不会除零出 inf？
**否，不会产生 inf。** 证据：
- 全样本一字板（`H==L` 且 `amt>0`）共 **21869 格**；`_cmf` 里 `ok` 要求 `rng>0`，乘数用 `_safe_div`（分母 >0 才算），
  不满足的位置先得 NaN，再被 `ml = np.where(ok, ml, 0.0)` 置 0，且该日金额也被排除出 `Σamt`。
- 合成探针 [A]：构造 1 个 `H==L` 且 `C` 越界的脏数据格（裸除法必然出 inf）后，
  裸乘数 `((C−L)−(H−C))/(H−L)` 的统计是 **inf=1、NaN=6**；而 `fl_cmf20` 的输出 **inf=0、NaN=0**，与朴素重算逐格相等（max|diff|=0）。
- 7 个对照日期全横截面：`fl_cmf20` 输出 `inf=0`，`max|value| ≤ 1`（clip 生效）。
审计判定：**是（要求满足）**；但「整日剔除（分子分母都不计）」是口径选择，见 §4-P5。

### 3-3 MFI 累计区间在零涨跌日（Δtp==0）怎么处理？
**两边都不计入（正/负资金流都不加该日金额），与常见 MFI 惯例一致；但对「精确并列」的判定是硬开关。** 证据：
- 合成探针 [B]：15 日窗口内 d>0 6 天、d<0 4 天、d==0 4 天；模块=`60.315893`，朴素（零涨跌两边都不计）完全相同；
  若改成「零涨跌算作流入」= `71.630205` → 该约定影响很大（相差 11.3 分）。
- 实盘 14 日窗口内**精确** Δtp==0 的对数：每日期 290/391/610/504/738/639/645 个，
  涉及股票 273~617 只，并列日金额最大可占窗口总资金流的 **19.7%~36.8%**。
- 刀锋效应：把 `ADJ_H` 乘 `(1+1e-13)`（就足以打破精确并列，但远小于任何真实数据精度）后，
  `fl_mfi14` 最大变化 **21.94 分**、2635 格变化 >1；`fl_mfi28` 最大变化 **19.51 分**、3373 格变化 >1。
- 最小复现（实测输出 `max|d|=15.18, |d|>1 的股票数=433 / 5131`，`002562.SZ` 从 68.55 变 61.28）：
  ```powershell
  $env:PYTHONIOENCODING="utf-8"; python -c "import sys; sys.path.insert(0,'3_因子挖掘/audit'); import numpy as np, pandas as pd, naive_flow as nf; from core.lib import MarketData; from factor_zoo.ctx import Ctx; from factor_zoo.registry import load_all; md=MarketData(); C=Ctx(md); P=nf.build_panels(md); f=load_all(); t=int(md.dates.searchsorted(pd.Timestamp('2025-06-30'))); P2=dict(P); P2['ADJ_H']=P['ADJ_H']*(1+1e-13); s2=nf.make_stub(P2,*P['A'].shape); a=np.asarray(f['fl_mfi14']['func'](t,md,C)); b=np.asarray(f['fl_mfi14']['func'](t,None,s2)); d=np.abs(a-b); print('max|d|=%.2f, |d|>1 的股票数=%d/%d'%(np.nanmax(d), int((d>1).sum()), int(np.isfinite(d).sum())))"
  ```
  审计判定：**有争议（口径与主流惯例一致，但精确并列是 knife-edge，建议加容差或写进 docstring）**。

### 3-4 筹码分布代理：复权还是未复权？长期分箱是否可比？
**用复权价（正确）。** `_chip_calc` 读 `C.ADJ_H / C.ADJ_L`，现价用复权收盘 `C.C[t]`，都是 `原始价 × cumadj_clean`，
同一股票跨年可直接比较；`cumadj` 实测全样本最大 **10765.4 倍**（不是 98 倍量级——比题目给的还要大），
若用未复权价做 250 日分箱会系统性错位。反事实量化（把同一套分桶换成未复权 [low, high]、现价用未复权收盘）：
- `dev`：中位差 0.002~0.004，但 p95 0.076~0.227，**5%~10% 的股票偏差 >0.10**，dev 符号翻转 2.3%~6.8%；
- `prof`：中位差 ≈0.0002~0.0047，5.2%~11.7% 的股票偏差 >0.10。
另有两点已声明的近似需要知道：250 日窗口内价格比超出 `exp(±1.5)=[0.223, 4.482]` 的日子会被截断到网格边缘，
每个对照日期有 **0.2%~1.24% 的合格股票**至少有一天被截断；桶宽 5%（对数价）导致单一成本簇跨桶时 `conc` 只能到 ≈0.5
（合成探针实测 0.54），`dev` 的几何均价与 desc 字面的「加权平均成本（算术）」也有少量差异
（中位 0.006~0.010，p95 ≤0.05，最大 0.35，秩相关 ≥0.996）。审计判定：**是（复权口径正确）**，上述均为已声明/可量化近似。

### 3-5 量价相关：窗口端点、零方差返回 NaN 还是 0？
- 窗口端点：`[t−w+1, t]` 含当日；协方差用窗口内**相邻对**，共 w−1 对，首日收益 `R[lo]` 不参与；
  有效对还要求两个相邻日都 `listed & amt>0`；门槛 12 对（20 日）/36 对（60 日）。
- 零方差：`den = sqrt(Sxx·Syy)`，用 `_safe_div(num, den)`，`den==0` 时返回 **NaN 而不是 0**（也绝不返回 inf）。
  合成探针（R 恒为 0.01、A 恒定）实测 `fl_pvcorr20 = nan`；但 7 个对照日期 × 5451 只 × 两个窗口，
  **0 例**「样本数达标却零方差」。这与 `ctx.cs_zscore` 对常数序列返回 0 的策略不一致，但在现有数据上没有实际影响。
  审计判定：**存在性=是（NaN，不是 0），实际影响=0 例**。

### 3-6 停牌日（amt==0）是否混进均值/相关/累计？
**不进任何窗口统计。** 扰动检验（探针 [E]）：把 23118 个**非估值日**的停牌格的 H×127、L×0.37
（`amt`、收益、收盘价全部不动；7 个对照日期当天的 111 个停牌格不扰动，其余 23118 格全部扰动），重算 14 个因子在 7 个对照日期的输出，
与未扰动面板逐格比较：`max|diff| ≤ 1.14e-14`（纯浮点噪声）、NaN 掩码不一致 0 格、inf 0 个；
被检验到的含停牌日窗口共 3970 个（20/60/250 日三种窗口）。
对照检验：把 `2025-06-27` 当天已成交行的 H/L 改成 2x/0.5x，`fl_cmf20/fl_mfi14/fl_chip_dev250` 分别有
5443/5447/5437 只股票发生变化 → 检验本身有灵敏度。

附带说明（不是错误，但要知道）：**估值日当天停牌的股票，因子仍会给值**（用截到 t 的陈旧窗口），
例如 2022-12-30 当日 7 只停牌股里 `fl_pvdiv60` 7 只全非 NaN（因为 t 的 amt=0 被当成「量分位最低」）；
`fl_chip_*` 在停牌日几乎总是给值（网格中心用携程收盘价）。回测的 `md.universe()` 默认剔除停牌/一字板，
所以不影响实际选股；但若有人直接拿因子面板做 `rank`，这些值需要自己剔除。

### 3-7 docstring / direction 与代码实际方向
- **代码代数符号与 desc 语义一致**（合成探针 [F]，全部为预期符号）：
  r 全为 +1% 时 `amtmom=+0.010`、`obvlvl=+1.0`、`obvslp=+1.0`；典型价单调上行时 `mfi=100`；
  收=最高时 `cmf=+1.0`；R 与 ΔlogA 同相位时 `pvcorr=+0.875`（同号）；价格递增+成交额递减时 `pvdiv60=+1.0`；
  成本 5 元单桶、现价 10 元时 `prof=0.996`、`dev=0.959`、`conc=0.992`。
- **声明的 direction（registry 说只是先验）**：14 个里 7 个与本次自算的 2021-01~2024-12 月度 IC 符号一致，
  7 个相反（`fl_amtmom60 / fl_pvdiv60 / fl_obvlvl20 / fl_obvslp20 / fl_cmf20 / fl_chip_prof250 / fl_chip_dev250`）；
  其中 `fl_amtmom20(direction=-1)` 与 `fl_amtmom60(direction=+1)` 是**同一个公式的 20/60 日版本，却给了相反先验**。
  由于 `s08_factor_mining.py` 实际按「样本内 IC 符号」定向并只把 direction 当先验，这不是计算错误，
  但作为文档/先验质量是**有争议**的（见 §4-P2）。

## 4. 发现的真实问题（按严重度排序；全部给了最小复现命令）

> 先说明：**没有**发现公式抄错、未来函数、形状/复权错误这类「有错」级问题。
> 下面 5 条是本次审计真正落到数字上的问题，性质分别是：数值脆弱 2 条、文档不一致 1 条、
> 口径/策略不一致 2 条。全部可复跑。

### P1（有争议·数值脆弱）MFI 零涨跌日 / OBV 零收益日是「精确相等」硬开关
- 位置：`factor_zoo/flow.py` `_mfi`（`d == 0` 走不进 pos/neg）与 `_obvlvl/_obvslp`（`np.sign(0)=0`，分子记 0、分母照算金额）。
- 事实：实盘 20 日窗口内 `R==0`（当日收盘与昨收**逐位相等**且有成交）的格数占有效日 2.0%~3.4%
  （每日期 1502~3525 格、1142~2071 只股票）；14 日 MFI 窗口内 `Δtp==0` 的对数 290~738 个/日期、
  273~617 只股票/日期，并列日金额最大占窗口总资金流的 19.7%~36.8%。
  只要给价格一个极小的相对扰动（`ADJ_H × (1+1e-13)`，比真实行情精度小 6~8 个数量级）就会让这些「并列日」整日金额重新计入正流或负流：
  `fl_mfi14` 最大跳 **21.94 分**（2635 格 >1）、`fl_mfi28` 最大跳 **19.51 分**（3373 格 >1）；
  把 `R==0` 改成 `+1e-13`（等价于把持平日改判为上涨日）后，7 个日期上 OBV 两因子合计最大跳 **0.946**（>1e-3 的格数 23179）；单看 2025-06-30：`fl_obvlvl20` 最大 0.5217（2017 只）、`fl_obvslp20` 最大 0.5667（1934 只）。
- 最小复现（MFI，实测量级 15.18、433/5131 只）：
  ```powershell
  $env:PYTHONIOENCODING="utf-8"; python -c "import sys; sys.path.insert(0,'3_因子挖掘/audit'); import numpy as np, pandas as pd, naive_flow as nf; from core.lib import MarketData; from factor_zoo.ctx import Ctx; from factor_zoo.registry import load_all; md=MarketData(); C=Ctx(md); P=nf.build_panels(md); f=load_all(); t=int(md.dates.searchsorted(pd.Timestamp('2025-06-30'))); P2=dict(P); P2['ADJ_H']=P['ADJ_H']*(1+1e-13); s2=nf.make_stub(P2,*P['A'].shape); a=np.asarray(f['fl_mfi14']['func'](t,md,C)); b=np.asarray(f['fl_mfi14']['func'](t,None,s2)); d=np.abs(a-b); print('max|d|=%.2f, |d|>1 的股票数=%d/%d'%(np.nanmax(d), int((d>1).sum()), int(np.isfinite(d).sum())))"
  ```
- 最小复现（OBV，实测 2025-06-30：`fl_obvlvl20` max|d|=0.5217、2017 只；`fl_obvslp20` 0.5667、1934 只）：
  ```powershell
  $env:PYTHONIOENCODING="utf-8"; python -c "import sys; sys.path.insert(0,'3_因子挖掘/audit'); import numpy as np, pandas as pd, naive_flow as nf; from core.lib import MarketData; from factor_zoo.ctx import Ctx; from factor_zoo.registry import load_all; md=MarketData(); C=Ctx(md); P=nf.build_panels(md); f=load_all(); t=int(md.dates.searchsorted(pd.Timestamp('2025-06-30'))); P3=dict(P); P3['R']=np.where(P['R']==0.0,1e-13,P['R']); s=nf.make_stub(P3,*P['A'].shape); a=np.asarray(f['fl_obvlvl20']['func'](t,md,C)); b=np.asarray(f['fl_obvlvl20']['func'](t,None,s)); d=np.abs(a-b); print('max|d|=%.4f, 变化>1e-3 的股票数=%d'%(np.nanmax(d), int((d>1e-3).sum())))"
  ```
- 评价与修正建议：模块的口径与主流惯例一致（「收盘不变 OBV 不变」「典型价不变不计入正负流」），
  在**同一份数据、同一条代码路径**下结果完全确定，所以不是 bug；
  但它是没有容差的开关，任何数据源的 ulp 差异都会造成 O(1) 级跳变。
  建议在 docstring 里写明该约定，或把判定改成容差式（如 `|Δtp| <= 1e-12·tp` 记为不变、`|r| <= 1e-12` 记为持平日），并回归自检。
  影响：`fl_mfi14/fl_mfi28`（有争议）；`fl_obvlvl20/fl_obvslp20`（同类，程度稍轻，判定为有争议）。

### P2（有争议·文档/先验）`direction` 先验与实际不符、且同一公式两个窗口先验相反
- 位置：`flow.py` 的 `FACTORS` 字典（`fl_amtmom20 direction=-1`、`fl_amtmom60 direction=+1`；OBV/CMF/两个 chip 因子为 +1）。
- 事实：`registry.py` 明确写「direction 只是先验（-1=买低值），最终方向由样本内 IC 决定」；
  本次自算 2021-01~2024-12 月度 IC（48 期，`md.universe()` 默认池、20 日前向收益）显示
  **7/14 的声明方向与实测 IC 符号相反**：`fl_amtmom60(+1 vs IC −0.0894)`、`fl_pvdiv60(−1 vs +0.0023)`、
  `fl_obvlvl20(+1 vs −0.0353)`、`fl_obvslp20(+1 vs −0.0313)`、`fl_cmf20(+1 vs −0.0341)`、
  `fl_chip_prof250(+1 vs −0.0732)`、`fl_chip_dev250(+1 vs −0.0834)`；其中 `fl_amtmom20/60` 是同一公式的不同窗口却给了相反先验。
- 最小复现：
  ```powershell
  $env:PYTHONIOENCODING="utf-8"; python 3_因子挖掘/audit/naive_flow.py     # 看 [G] 段
  $env:PYTHONIOENCODING="utf-8"; python -c "import sys; sys.path.insert(0,'.'); from factor_zoo.registry import load_all; f=load_all(); [print('%-18s direction=%+d'%(k,v['direction'])) for k,v in f.items() if k.startswith('fl_')]"
  ```
- 修正建议：direction 只是先验、主线用的是样本内 IC，故不影响回测结果；但建议把这两个 amtmom 的先验统一
  （按 2021-2024 实测，两个都应为 -1），并让 desc 与先验的表述自洽，避免下游读者按 desc 的「量价配合」直觉用错方向。

### P3（文档不一致）`_obv_slope` 注释「起点不参与斜率」与实现不符
- 位置：`flow.py` `_obv_slope`：`Y = np.cumsum(flow, axis=0)` 后 `ok`、`x`、掩码均值都包含窗口第一个有效点，
  首点**确实参与** OLS；注释写的「起点不参与斜率」不成立（把累加起点平移一个常数确实不影响斜率，
  但首个观测点被计入了回归）。
- 实测（2024-06-28）：含首点 vs 去掉首个有效点，中位 |Δ|=0.0256、最大 |Δ|=0.3095、横截面秩相关 0.99094（n=5097）；
  7 个日期都算过（脚本 `naive_flow.py` 的 [OBV斜率] 段逐日打印），秩相关 0.989~0.995、中位|Δ| 0.016~0.026、最大|Δ| 0.12~0.31。
- 最小复现（实测输出 `含首点 vs 去掉首点: 中位|d|=0.0256 最大|d|=0.3095 秩相关=0.99094 (n=5097)`）：
  ```powershell
  $env:PYTHONIOENCODING="utf-8"; python -c "
  import sys; sys.path.insert(0,'3_因子挖掘/audit')
  import numpy as np, pandas as pd, naive_flow as nf
  from core.lib import MarketData
  from factor_zoo.ctx import Ctx
  from factor_zoo.registry import load_all
  md=MarketData(); C=Ctx(md); P=nf.build_panels(md); f=load_all()
  t=int(md.dates.searchsorted(pd.Timestamp('2024-06-28'))); lo=t-19
  R,A,LI=P['R'][lo:t+1],P['A'][lo:t+1],P['listed'][lo:t+1]
  nr=R.shape[0]; x=np.arange(nr,dtype=float)-0.5*(nr-1)
  a=np.asarray(f['fl_obvslp20']['func'](t,md,C)); b=np.full(A.shape[1],np.nan)
  for j in range(A.shape[1]):
      g=LI[:,j]&(A[:,j]>0)&np.isfinite(R[:,j])
      if g.sum()<12: continue
      Y=np.cumsum(np.where(g,np.sign(np.where(g,R[:,j],0.0))*np.where(g,A[:,j],0.0),0.0))
      g2=g.copy(); g2[np.argmax(g)]=False
      if g2.sum()<11: continue
      dx,dy=x[g2]-x[g2].mean(),Y[g2]-Y[g2].mean()
      b[j]=(dx*dy).sum()/((dx*dx).sum()*A[g2,j].mean())
  m=np.isfinite(a)&np.isfinite(b)
  print('含首点 vs 去掉首点: 中位|d|=%.4f 最大|d|=%.4f 秩相关=%.5f (n=%d)'%(np.median(np.abs(a[m]-b[m])),np.abs(a[m]-b[m]).max(),pd.Series(a[m]).rank().corr(pd.Series(b[m]).rank()),m.sum()))
  "
  ```
- 修正建议：把注释改成「累计 OBV 的起算基准对斜率无影响，但窗口首个有效点计入回归」，或直接把首点排除并同步注释。

### P4（口径/策略不一致，实测 0 影响）`fl_pvcorr` 零方差返回 NaN 而非 0
- 位置：`_pvcorr` 末段 `_safe_div(num, den)`：`den==0`（某只股票窗口内收益或 Δlog 成交额恒定）→ NaN。
- 实测：7 个对照日期 × 5451 只 × (20,60) 两个窗口，**0 例**「样本数达标但零方差」；
  合成常数序列（R 恒 0.01、A 恒定）实测返回 `nan`。
- 与 `ctx.cs_zscore`（常数序列返回 0，有 docstring 说明）策略不一致；由于 0 例，对现有数据无影响。
- 最小复现：
  ```powershell
  $env:PYTHONIOENCODING="utf-8"; python 3_因子挖掘/audit/naive_flow.py     # 看 [D] 段
  ```
- 修正建议：如果下游会做「NaN 丢弃后排名」，NaN 与 0 的区别只影响零方差股票；建议在 `_pvcorr` 里注明策略，或统一为 0。

### P5（口径选择，已部分声明）CMF 一字板整日剔除；筹码分箱的三项近似；停牌日仍给陈旧值
- CMF：`H==L`（有成交的一字板）的日子分子分母同时剔除。实盘 20 日窗口内这种日子每天 50~133 只股票涉及、
  每日期 81~410 格；个别股票的**整窗金额都来自一字板**（一字板金额占窗口金额比可达 100%），此时 CMF 由剩余日子决定甚至 NaN。
  该选择写在 `_cmf` 行内注释（「分子分母同取有效日，口径一致」），但没写进对外 desc；
  与「把 CLV 记 0 但金额留在分母」的替代口径在 1e-13 扰动下最大差 0.185。
- 筹码：复权口径正确（见 §3-4）；网格 ±1.5 对数截断影响每日期 0.2%~1.24% 的合格股票；
  5% 桶宽使 `conc` 的分辨率受限（单桶成本跨桶时只能到 ≈0.5，实测截面 max 0.50~1.00、中位 0.10）；
  `dev` 实际是「现价/几何平均成本−1」，与 desc 字面的「加权平均成本」相差中位 0.6~1.0 个百分点、最大 0.35。
- 停牌日估值：见 §3-6 表，因素在停牌日会给陈旧窗口值（0%~100% 非 NaN，因因子/日期而异）；
  `md.universe()` 默认剔除停牌，回测不受影响。
- 最小复现：
  ```powershell
  $env:PYTHONIOENCODING="utf-8"; python 3_因子挖掘/audit/naive_flow.py     # [A]/[C]/[D]/[E]/[OBV斜率] 段
  ```
- 修正建议：把 `_cmf` 的剔除口径与筹码网格/桶宽写进模块顶部 docstring；`fl_chip_dev250` 的 desc 改为「几何平均成本」；
  若下游可能直接对因子面板排名，建议对停牌/一字板格子在因子层也返回 NaN（或明确由消费方过滤）。

## 5. 未能验证 / 限制

1. **真实成交量口径无法验证**：数据只有成交额与笔数，没有 volume/股本；
   §3-1 的对照用的是 `V=amt/未复权收盘` 的粗代理（RAW 不是当日 VWAP），只能衡量「替代口径的相对影响」，
   不能断言真实 volume 口径下的因子值或 IC。
2. **筹码分布没有外部真值**：无龙虎榜/成本分布/筹码数据可对账，本报告只能证明「实现 == docstring 声明」，
   「把成交额按当日 [L,H] 对数均匀铺开」这一经济建模假设本身未被验证（这是该因子族的定义假设）。
3. **抽查范围**：独立重算只做了 7 个日期（题目要求 ≥6，已满足）；其余日期依赖项目自带 truncated 检验（24 个日期，14/14 OK）
   与代码走查；未做全样本 1643 日的逐格重算（成本考虑），因此极端日期（如 2020 年 2 月疫情期、2026 年新上市股）可能有未覆盖到的边界。
4. **只审 flow 模块**：`mom_rev / risk / liquidity / trend` 不在本次范围（本目录下另有其它审计产物）。
5. 审计脚本运行耗时约 114 s（朴素循环 + 分桶），输出保存在 `naive_flow_output.txt`；
   自检输出保存在 `selfcheck_flow_output.txt`。
