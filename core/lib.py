# -*- coding: utf-8 -*-
"""Class4 回测 · 数据层：把 feathers/ 的 7 个日频矩阵变成可直接回测的面板。

    from core.lib import MarketData
    md  = MarketData()
    px  = md.adj_close                  # 复权价
    uni = md.universe()                 # 动态股票池（逐日布尔矩阵）
    r   = md.ret.where(uni)             # 可用的日收益

数据事实（经全样本核验，数字见 1_数据真实性验证/out/data_audit.md）
------------------------------------------------------------------
1. 7 个矩阵结构完全一致：1643 交易日 x 5451 证券，date + float64，列名与顺序全同。
2. 价格是**未复权**成交价；复权价 = 原始价 x cumadj（后复权口径）。
3. open/high/low/close/amt/cumadj 的 NaN 掩码完全一致（仅 2 格不同）；
   num_trades 另有 2026-10-16 一整行及 10 个交易日的零星缺口，共 5363 格。
4. 停牌：amt==0 且 OHLC 四价全等（用前收盘价携程），23229 格，平均每天约 14 只，
   1638/1643 个交易日都存在。做收益或因子时必须剔除。
5. 一字板：high==low 且 amt>0，共 21869 格，全天只有单一价格、**无法按市价成交**，
   回测须按方向屏蔽买入（一字涨停）与卖出（一字跌停）。
6. cumadj 在非调整日的相对抖动只有 1e-6~1e-5 量级（浮点噪声），真实分红送转
   事件才 >1e-3。直接 pct_change 会凭空造出上万条假收益，必须先做去噪。
7. 涨跌幅限制用「板块先验 + 时代规则 + 逐股违规法反推」；全样本只有 4 只 5% 特殊
   标的（000752.SZ / 002872.SZ / 600182.SH / 600677.SH），说明样本基本不含 ST。
8. 新股上市后前 N 个交易日不设涨跌幅（主板 2023-02-17 起 N=5、此前 N=1；
   创业板 2020-08-24 起 N=5、此前 N=1；科创板 N=5），复牌/退市整理日亦无常规限制。
   这些日子单日涨跌幅最高 +488%，已全部落在 no_limit 中。
9. 样本只有价 + 成交额 + 成交笔数 + 复权因子：**没有成交量、流通股本、市值、行业、
   指数成分、ST 标记、停牌公告**。换手率类因子、行业中性化、基准指数需外部数据；
   VWAP 无法还原，成交价只能用 open/close 近似。
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"

# 项目根目录 = 本文件所在目录（不写死本机路径）
# 项目根 = core/ 的上一级（core/ 只放库代码，产物统一落在项目根的 out/ 下）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: 产物按「工作」分目录。脚本在 import core.lib **之前**设置
#:     os.environ.setdefault("CLASS4_TRACK", "02_回测系统")
#: 之后 FIG/TAB/RUNS 会自动落到 out/02_回测系统/{figures,tables,runs}。
#: 不设置时退回老的 out/{figures,tables,runs}（兼容单独 import core.lib 的场景）。
#: 因子面板缓存 **不随工作切换**：out/cache/ 由 s04/s05/s07/s08/s10 共享，避免重复计算。
TRACK = os.environ.get("CLASS4_TRACK", "").strip()
OUT = os.path.join(ROOT, "out", TRACK) if TRACK else os.path.join(ROOT, "out")
FIG = os.path.join(OUT, "figures")
TAB = os.path.join(OUT, "tables")
RUNS = os.path.join(OUT, "runs")
CACHE = os.path.join(ROOT, "out", "cache")
#: 只有设了 CLASS4_TRACK 才创建产物目录；单独 import core.lib（测试、体检）不该建空目录。
#: 缓存目录始终存在（它是共享的、可再生的）。
if TRACK:
    for _d in (FIG, TAB, RUNS):
        os.makedirs(_d, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

FIELDS = ("open", "high", "low", "close", "amt", "num_trades", "cumadj")
ENV_VARS = ("CLASS4_DATA_DIR", "FEATHERS_DIR")

TICK = 0.01                 # A 股最小变动价位（元）
HALF_TICK = TICK / 2.0      # 价格比较容差
CUMADJ_EPS = 1e-4           # cumadj 相对变化的去噪阈值（噪声 <1e-5，真实事件 >1e-3）

GEM_REG = pd.Timestamp("2020-08-24")     # 创业板注册制：20% 涨跌幅，新股前 5 日不设限
MAIN_REG = pd.Timestamp("2023-02-17")    # 主板全面注册制：新股前 5 日不设限
NEW_DAYS = 5                             # 注册制下新股不设涨跌幅的交易日数

RATIO_MAIN, RATIO_GEM, RATIO_ST = 0.10, 0.20, 0.05


# --------------------------------------------------------------------------- #
# 数据目录定位（与 Class3 同一套约定，不写死个人路径）
# --------------------------------------------------------------------------- #
def resolve_dir(argv=None, start=None) -> str:
    """优先级：命令行 --data-dir > 环境变量 CLASS4_DATA_DIR/FEATHERS_DIR
    > 本项目下 feathers/ > 当前工作目录 > 逐级向上搜索 close.feather。"""
    argv = list(argv if argv is not None else [])
    cands = []
    if "--data-dir" in argv:
        i = argv.index("--data-dir")
        if i + 1 < len(argv):
            cands.append(argv[i + 1])
    for var in ENV_VARS:
        p = os.environ.get(var)
        if p:
            cands.append(p)
    roots = [start or ROOT, ROOT, os.getcwd()]
    for base in roots:
        cands.append(os.path.join(base, "feathers"))
        cands.append(base)
    for c in cands:
        if not c:
            continue
        p = os.path.abspath(os.path.expanduser(c))
        if os.path.isfile(os.path.join(p, "close.feather")):
            return p
    for base in roots:
        for hit in sorted(glob.glob(os.path.join(base, "**", "close.feather"), recursive=True)):
            return os.path.dirname(os.path.abspath(hit))
    raise SystemExit(
        "找不到日频数据目录（需含 close.feather 等 7 个文件）。\n"
        "请用 --data-dir 指定，或设置环境变量 CLASS4_DATA_DIR，\n"
        "或把 feathers/ 放在本项目目录下。")


def portable_path(path) -> str:
    """把目录/文件路径转成可移植写法，写报告/日志时用。

    - 在项目根内 → 相对路径（如 ``feathers``）
    - 在项目根外 → ``<外部数据目录>/<目录名>``

    这样交付物里不会出现本机盘符绝对路径。
    """
    try:
        p = os.path.abspath(os.path.expanduser(str(path)))
    except Exception:                                     # noqa: BLE001
        return str(path)
    try:
        rel = os.path.relpath(p, ROOT)
    except ValueError:                                    # Windows 跨盘符
        return "<外部数据目录>/" + os.path.basename(p)
    if rel == ".":
        return "."
    if rel.startswith(".."):
        return "<外部数据目录>/" + os.path.basename(p)
    return rel.replace("\\", "/")



def _read_feather(path: str) -> pd.DataFrame:
    """Feather V2 即 Arrow IPC 文件格式；pyarrow>=24 已弃用 feather.read_table。"""
    try:
        import pyarrow.ipc as ipc
        tbl = ipc.open_file(path).read_all()
    except AttributeError:                                  # 老版本 pyarrow 兜底
        import pyarrow.feather as feather
        tbl = feather.read_table(path)
    df = tbl.to_pandas().set_index("date")
    df.index.name = "date"
    df.index = pd.DatetimeIndex(df.index).normalize()
    return df


def round_tick(x):
    """交易所口径：四舍五入到 0.01 元。

    numpy/pandas 的 round 是「四舍六入五取偶」（banker's rounding），
    昨收 5.75 的涨停价交易所算 6.33、pandas 算 6.32，恰好差一跳；
    全样本因此会产生约 2700 格假的「突破涨停」。
    """
    a = np.asarray(x, dtype="float64")
    return np.floor(a * 100.0 + 0.5 + 1e-7) / 100.0


def board_of(symbols) -> pd.Series:
    """按代码前缀判断上市板块：MAIN 主板 / GEM 创业板 / STAR 科创板。"""
    s = pd.Index(symbols).astype(str)
    code = s.str[:3]
    out = np.where(s.str.startswith(("688", "689")), "STAR",
          np.where(code.isin(["300", "301", "302"]), "GEM", "MAIN"))
    return pd.Series(out, index=s, name="board")


def board_new_days(board: pd.Series, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """新股上市后不设涨跌幅的交易日数矩阵（逐日、逐股）。

    科创板自 2019-07 起 5 日；创业板 2020-08-24 注册制起 5 日、此前 1 日；
    主板 2023-02-17 全面注册制起 5 日、此前 1 日。
    """
    b = np.asarray(board.values)
    is_main = b == "MAIN"
    main_n = np.where(dates < MAIN_REG, 1.0, float(NEW_DAYS))[:, None]
    gem_n = np.where(dates < GEM_REG, 1.0, float(NEW_DAYS))[:, None]
    star_n = np.full((len(dates), 1), float(NEW_DAYS))
    n = np.where(is_main[None, :], main_n, np.where((b == "GEM")[None, :], gem_n, star_n))
    return pd.DataFrame(n, index=dates, columns=board.index)


def _limit_ratio_prior(board: pd.Series, dates: pd.DatetimeIndex, st: pd.Series) -> pd.DataFrame:
    """板块先验 + 时代规则 + 5% 特殊标的，返回 (日期 x 证券) 的涨跌幅上限矩阵。"""
    base = pd.Series(np.where(np.asarray(board.values) == "MAIN", RATIO_MAIN, RATIO_GEM),
                     index=board.index, dtype="float64")
    base[st.reindex(base.index).fillna(False).astype(bool)] = RATIO_ST
    ratio = pd.DataFrame(np.tile(base.values, (len(dates), 1)),
                         index=dates, columns=base.index)
    early = np.asarray(dates < GEM_REG)
    if early.any():
        gem_early = (np.asarray(board.values) == "GEM") & (base.values != RATIO_ST)
        if gem_early.any():
            sub = ratio.values.copy()
            sub[np.ix_(early, gem_early)] = RATIO_MAIN
            ratio = pd.DataFrame(sub, index=dates, columns=base.index)
    return ratio


# --------------------------------------------------------------------------- #
# 主类
# --------------------------------------------------------------------------- #
class MarketData:
    """日频面板：载入 / 掩码 / 复权 / 涨跌停 / 收益 / 股票池。

    参数
    ----
    data_dir : str | None   数据目录，缺省自动定位
    float32  : bool         读入时转 float32，常驻内存减半（约 502MB -> 251MB）
    verbose  : bool         打印载入进度
    """

    def __init__(self, data_dir=None, float32=False, verbose=False):
        self.dir = resolve_dir(None if data_dir is None else ["--data-dir", str(data_dir)])
        self.float32 = bool(float32)
        self.verbose = bool(verbose)
        self._cache = {}
        template = self._load("close")
        self.dates = template.index
        self.symbols = template.columns
        self.board = board_of(self.symbols)
        self._ratio_table = None
        self._st_like = None

    # ---------------- 载入 ----------------
    def _load(self, field: str) -> pd.DataFrame:
        if field in self._cache:
            return self._cache[field]
        path = os.path.join(self.dir, field + ".feather")
        if not os.path.isfile(path):
            raise FileNotFoundError("缺少数据文件: " + path)
        df = _read_feather(path)
        if self.float32:
            df = df.astype("float32")
        if self.verbose:
            print("loaded %-11s %s" % (field, df.shape))
        self._cache[field] = df
        return df

    def field(self, name: str) -> pd.DataFrame:
        """按名取字段：open/high/low/close/amt/num_trades/cumadj。"""
        if name not in FIELDS:
            raise KeyError("未知字段 %r，可选 %s" % (name, FIELDS))
        return self._load(name)

    @property
    def open(self):
        return self._load("open")

    @property
    def high(self):
        return self._load("high")

    @property
    def low(self):
        return self._load("low")

    @property
    def close(self):
        return self._load("close")

    @property
    def amt(self):
        return self._load("amt")

    @property
    def num_trades(self):
        return self._load("num_trades")

    @property
    def cumadj(self):
        return self._load("cumadj")

    def _memo(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def _row(self, key, values) -> pd.DataFrame:
        """把一维序列按列广播成 (日期 x 证券) 矩阵，带缓存。"""
        if key not in self._cache:
            arr = np.tile(np.asarray(values), (len(self.dates), 1))
            self._cache[key] = pd.DataFrame(arr, index=self.dates, columns=self.symbols)
        return self._cache[key]

    # ---------------- 基础掩码 ----------------
    @property
    def listed(self) -> pd.DataFrame:
        """有报价（含停牌日的携程价）。"""
        return self._memo("listed", lambda: self.close.notna())

    @property
    def traded(self) -> pd.DataFrame:
        """当日真实成交过（成交额 > 0）。"""
        return self._memo("traded", lambda: self.listed & (self.amt > 0))

    @property
    def suspended(self) -> pd.DataFrame:
        """停牌：有价无成交，OHLC 全部等于前收盘价。"""
        return self._memo("suspended", lambda: self.listed & ~(self.amt > 0))

    @property
    def all_flat(self) -> pd.DataFrame:
        """全天只有一个价格（停牌或一字板）。"""
        return self._memo("all_flat", lambda: self.listed & (self.high == self.low))

    @property
    def one_line(self) -> pd.DataFrame:
        """一字板：有成交但全天同一价格，无法按市价成交。"""
        return self._memo("one_line", lambda: self.traded & (self.high == self.low))

    @property
    def listed_days(self) -> pd.DataFrame:
        """截至当日的累计上市交易日数（含当日），用于剔除次新股。"""
        def build():
            m = self.listed.values
            n = np.cumsum(m, axis=0) * m          # 停牌/退市区间保持原值
            return pd.DataFrame(n.astype("float64"), index=self.dates, columns=self.symbols)
        return self._memo("listed_days", build)

    @property
    def first_pos(self) -> pd.Series:
        """每只证券首个有报价日在 self.dates 中的位置（从未出现为 -1）。"""
        def build():
            m = self.listed.values
            any_ = m.any(0)
            pos = np.where(any_, m.argmax(0), -1).astype("int64")
            return pd.Series(pos, index=self.symbols, name="first_pos")
        return self._memo("first_pos", build)

    @property
    def _new_stock(self) -> pd.DataFrame:
        """样本内新上市、且仍在「上市后前 N 个交易日」窗口内的格子。

        注意：样本首日就已存在的证券无法判断其上市日期，一律**不**视为次新股
        （否则会把 2020-01-02 的全部 3760 只标的误标为无涨跌幅限制）。
        """
        def build():
            days = board_new_days(self.board, self.dates)
            fresh = self._row("fresh", (self.first_pos > 0).values).astype("bool")
            return (self.listed & (self.listed_days <= days) & fresh).fillna(False)
        return self._memo("_new_stock", build)

    # ---------------- 复权 ----------------
    @property
    def cumadj_clean(self) -> pd.DataFrame:
        """去噪后的累计复权因子。

        非调整日的相对抖动（<1e-4）视作浮点噪声置为 1，逐日连乘重建因子；
        真实分红送转事件原样保留。未上市区间仍为 NaN。
        """
        def build():
            f = self.cumadj
            rel = f / f.shift(1)
            step = rel.where((rel - 1.0).abs() >= CUMADJ_EPS, 1.0)
            base = f.bfill().iloc[0]                     # 每列首个有效因子
            clean = step.fillna(1.0).cumprod().mul(base, axis=1)
            arr = np.where(f.notna().values, clean.values, np.nan)
            return pd.DataFrame(arr, index=f.index, columns=f.columns)
        return self._memo("cumadj_clean", build)

    def adj(self, field: str = "close") -> pd.DataFrame:
        """复权价 = 原始价 x 去噪复权因子。"""
        return self.field(field) * self.cumadj_clean

    @property
    def adj_close(self):
        return self._memo("adj_close", lambda: self.adj("close"))

    # ---------------- 收益 ----------------
    @property
    def ret(self) -> pd.DataFrame:
        """复权日收益（持仓口径）。

        - 停牌日价格为前收盘携程 -> 收益自然为 0，头寸价值不变（可继续持有）；
          若停牌期间除权，当日收益体现为复权因子跳升，属经济上应计的收益。
        - 未上市 / 退市后为 NaN；新股上市首日跳幅未剔除（见 no_limit）。
        """
        def build():
            a = self.adj_close
            r = a / a.shift(1) - 1.0
            return r.where(self.listed & self.listed.shift(1, fill_value=False))
        return self._memo("ret", build)

    @property
    def ret_raw(self) -> pd.DataFrame:
        """未复权日收益，仅用于识别涨跌停与一字板方向。"""
        def build():
            c = self.close
            return (c / c.shift(1) - 1.0).where(self.listed & self.listed.shift(1, fill_value=False))
        return self._memo("ret_raw", build)

    @property
    def prev_ref(self) -> pd.DataFrame:
        """涨跌停的基准价 = 昨收还原到今日复权基准。

        除权除息日的涨跌幅以除权参考价为基准，而非原始昨收：
            ref = 昨收 x f_{t-1} / f_t = adj_close.shift(1) / f_t
        非除权日该值恰等于原始昨收。
        """
        def build():
            return self.adj_close.shift(1) / self.cumadj_clean
        return self._memo("prev_ref", build)

    def ret_fwd(self, k: int = 1) -> pd.DataFrame:
        """未来 k 日复权收益（t 收盘建仓、t+k 收盘平仓）。"""
        a = self.adj_close
        return (a.shift(-k) / a - 1.0).where(self.listed & self.listed.shift(-k, fill_value=False))

    @property
    def ret_ok(self) -> pd.DataFrame:
        """可直接用于因子评估/回测的收益：两端都在正常交易状态。"""
        def build():
            good = self.traded & self.traded.shift(1, fill_value=False) & ~self.no_limit
            return self.ret.where(good)
        return self._memo("ret_ok", build)

    # ---------------- 涨跌停 ----------------
    def cap_tolerance(self) -> pd.Series:
        """逐股的上限判定容差 = 0.2 个百分点 + 四舍五入余量 0.005/价格。

        涨跌停价是按 0.01 元四舍五入后的整数分价格，因此低价股的「上限收益」
        天然会比标称幅度高出最多 0.005/价格（1.5 元股约 0.33pp），
        用固定容差会把低价股全部误判为违规。
        """
        return self._memo("cap_tol", lambda: 0.002 + HALF_TICK / self.close.where(self.listed).median(0))

    def ratio_table(self, cand=(RATIO_ST, RATIO_MAIN, RATIO_GEM)) -> pd.DataFrame:
        """逐股的涨跌停上限诊断表（违规法）。

        - `v05/v10/v20`：复权收益突破各候选上限的天数（容差见 cap_tolerance）。
        - `p99/p999/max_abs`：收益分布的极端分位，用来判断「上限是否真的顶住」。
        - `wall_<r>`：该候选上限是否被数据顶住（p99 已逼近 + 无一格突破）。
        - `st_strict`：严格的 5% 判定（见 st_like）。
        """
        if self._ratio_table is not None:
            return self._ratio_table
        r = self.ret                                     # 复权收益：除权跳空不算违规
        ok = self.listed & self.listed.shift(1, fill_value=False) & ~self._new_stock
        absr = r.abs().where(ok)
        tol = self.cap_tolerance()
        T = pd.DataFrame({("v%02d" % (x * 100)): ((absr > x + tol) & ok).sum(0) for x in cand})
        T["days"] = ok.sum(0)
        T["median_px"] = self.close.where(ok).median(0)
        T["p99"] = absr.quantile(0.99)
        T["p999"] = absr.quantile(0.999)
        T["max_abs"] = absr.max(0)
        for x in cand:
            T["wall_%02d" % (x * 100)] = (T["v%02d" % (x * 100)] == 0) & (T["p99"] >= x - 0.005)
        T["st_strict"] = T["wall_05"] & (T["days"] >= 100)
        # 按各自的板块先验逐格比对：正常标的应为 0 格违规
        prior = _limit_ratio_prior(self.board, self.dates,
                                   pd.Series(False, index=self.symbols))
        exceed = (absr.values > prior.values + tol.values[None, :]) & ok.values
        T["v_prior"] = exceed.sum(0)
        T["v_prior_pct"] = 100.0 * T["v_prior"] / T["days"].replace(0.0, np.nan)
        self._ratio_table = T
        return T

    @property
    def st_like(self) -> pd.Series:
        """5% 涨跌幅标的布尔序列（S 股 / 风险警示）。

        只做**保守**判定：要求复权收益在整段历史上从未突破 5%+容差（一格都不许），
        且收益的 99 分位已逼近 5%（说明 5% 这道墙确实被顶住），且历史 >=100 个交易日。

        为什么必须这么严：从价格数据无法区分「5% 上限但从未涨过 5%」和
        「10% 上限但从未涨过 5%」。大盘银行股（工行/建行/农行）六年里单日
        最大涨幅只有 7%~9%，放宽容差就会把它们误判成 5%。本样本最终只识别出
        2 只（600182.SH / 600485.SH），说明数据源基本已剔除 ST 名单。
        """
        if self._st_like is None:
            T = self.ratio_table()
            self._st_like = T["st_strict"].fillna(False).rename("st_like")
        return self._st_like

    @property
    def limit_ratio(self) -> pd.DataFrame:
        """(日期 x 证券) 涨跌幅上限矩阵。

        以**监管先验**为准（主板 10%、科创板 20%、创业板 2020-08-24 起 20% 此前 10%），
        只对通过严格检验的 5% 标的做例外处理。先验的可靠性由 ratio_table 的
        违规列直接验证：按先验判定，全样本没有一格收盘价突破上限。
        """
        return self._memo("limit_ratio",
                          lambda: _limit_ratio_prior(self.board, self.dates, self.st_like))

    def _limit_price(self, sign: float) -> pd.DataFrame:
        ref = self.prev_ref.values
        arr = round_tick(ref * (1.0 + sign * self.limit_ratio.values))
        return pd.DataFrame(arr, index=self.dates, columns=self.symbols)

    @property
    def limit_up_price(self) -> pd.DataFrame:
        """当日涨停价（交易所口径四舍五入），除权日以除权参考价为基准。"""
        return self._memo("limit_up_price", lambda: self._limit_price(1.0))

    @property
    def limit_down_price(self) -> pd.DataFrame:
        """当日跌停价（交易所口径四舍五入），除权日以除权参考价为基准。"""
        return self._memo("limit_down_price", lambda: self._limit_price(-1.0))

    @property
    def no_limit(self) -> pd.DataFrame:
        """不适用常规涨跌幅的日子：新股前 N 日、复牌/退市整理、数据异常。

        判定：仍在上市前 N 个交易日内（按板块与时代规则），
        或单日涨跌幅超出上限 0.3 个百分点以上。
        """
        def build():
            cap = self.limit_ratio.values + self.cap_tolerance().values[None, :]
            over = self.ret.abs().values > cap
            out = self.listed.values & (self._new_stock.values | over)
            return pd.DataFrame(out, index=self.dates, columns=self.symbols)
        return self._memo("no_limit", build)

    @property
    def at_limit_up(self) -> pd.DataFrame:
        """最高价触及涨停（含一字板）。"""
        def build():
            touch = (self.high - self.limit_up_price).abs() < HALF_TICK
            return (touch & self.listed & ~self.no_limit).fillna(False)
        return self._memo("at_limit_up", build)

    @property
    def at_limit_down(self) -> pd.DataFrame:
        def build():
            touch = (self.low - self.limit_down_price).abs() < HALF_TICK
            return (touch & self.listed & ~self.no_limit).fillna(False)
        return self._memo("at_limit_down", build)

    @property
    def one_line_up(self) -> pd.DataFrame:
        """一字涨停：全天单一价格且上涨，当日买不进。"""
        return self._memo("one_line_up", lambda: self.one_line & (self.ret_raw > 0))

    @property
    def one_line_down(self) -> pd.DataFrame:
        """一字跌停：全天单一价格且下跌，当日卖不出。"""
        return self._memo("one_line_down", lambda: self.one_line & (self.ret_raw < 0))

    @property
    def can_buy(self) -> pd.DataFrame:
        """当日可买入（有成交且非一字涨停）。"""
        return self._memo("can_buy", lambda: self.traded & ~self.one_line_up)

    @property
    def can_sell(self) -> pd.DataFrame:
        """当日可卖出（有成交且非一字跌停）。"""
        return self._memo("can_sell", lambda: self.traded & ~self.one_line_down)

    # ---------------- 流动性 / 微观 ----------------
    def adv(self, window: int = 20) -> pd.DataFrame:
        """window 日平均成交额（元），最少需 window/2 个观测。"""
        key = "adv%d" % window
        return self._memo(key, lambda: self.amt.rolling(window, min_periods=max(2, window // 2)).mean())

    @property
    def avg_trade_size(self) -> pd.DataFrame:
        """单笔平均成交金额（元）= amt / num_trades；2026-10-16 无笔数数据。"""
        def build():
            nt = self.num_trades
            return (self.amt / nt).where((nt > 0) & self.amt.notna())
        return self._memo("avg_trade_size", build)

    @property
    def amount_yi(self) -> pd.DataFrame:
        """成交额（亿元），仅用于展示。"""
        return self.amt / 1e8

    # ---------------- 动态股票池 ----------------
    def universe(self,
                 min_listed_days: int = 250,
                 adv_window: int = 20,
                 min_adv: float | None = 2e7,
                 price_min: float | None = 2.0,
                 boards=None,
                 exclude_st: bool = True,
                 exclude_no_limit: bool = True,
                 exclude_one_line: bool = True) -> pd.DataFrame:
        """逐日动态可交易股票池。

        默认：上市满 250 个交易日、20 日均成交额 >= 2000 万元、股价 >= 2 元、
        剔除 5% 涨跌幅标的、剔除新股/复牌等无限制日、剔除一字板与停牌。
        条件只用当日及此前数据，**不含未来信息**。
        """
        key = ("universe", min_listed_days, adv_window, min_adv, price_min,
               None if boards is None else tuple(sorted(boards)),
               exclude_st, exclude_no_limit, exclude_one_line)
        if key in self._cache:
            return self._cache[key]                    # 只读，勿就地修改
        u = self.traded.copy()
        u &= self.listed_days >= min_listed_days
        if min_adv is not None:
            u &= self.adv(adv_window) >= min_adv
        if price_min is not None:
            u &= self.close >= price_min
        if boards is not None:
            u &= self._row("board:" + ",".join(sorted(boards)),
                           self.board.isin(list(boards)).values).astype("bool")
        if exclude_st:
            u &= ~self._row("st_like", self.st_like.values).astype("bool")
        if exclude_no_limit:
            u &= ~self.no_limit
        if exclude_one_line:
            u &= ~self.one_line
        out = u.astype("bool")
        self._cache[key] = out
        return out

    def n_eligible(self, **kw) -> pd.Series:
        """每日股票池规模。"""
        s = self.universe(**kw).sum(1)
        s.name = "n"
        return s

    # ---------------- 便捷工具 ----------------
    def equal_weight_ret(self) -> pd.Series:
        """等权多头组合的日收益（简单平均，仅用于数据体检）。"""
        s = self.ret_ok.mean(1)
        s.name = "ew_ret"
        return s

    def mask_report(self) -> pd.DataFrame:
        """各类掩码的规模统计。"""
        n_listed = max(int(self.listed.values.sum()), 1)
        rows = []
        for nm in ("listed", "traded", "suspended", "all_flat", "one_line",
                   "one_line_up", "one_line_down", "no_limit",
                   "at_limit_up", "at_limit_down"):
            a = np.asarray(getattr(self, nm).values, dtype=bool)
            rows.append(dict(mask=nm, cells=int(a.sum()),
                             pct_of_listed=100.0 * a.sum() / n_listed,
                             days_with=int(a.any(1).sum()),
                             syms_with=int(a.any(0).sum()),
                             max_per_day=int(a.sum(1).max())))
        return pd.DataFrame(rows).set_index("mask")

    def describe(self, field: str = "close") -> pd.Series:
        """字段的横截面描述统计（只统计有值格）。"""
        v = np.asarray(self.field(field).values, dtype="float64")
        return pd.Series(v[np.isfinite(v)]).describe(percentiles=[.01, .25, .5, .75, .99])

    def __repr__(self):
        return ("MarketData(%s | %d 交易日 x %d 证券 | %s ~ %s | 已载入 %s)"
                % (self.dir, len(self.dates), len(self.symbols),
                   self.dates[0].date(), self.dates[-1].date(),
                   ",".join(sorted(k for k in self._cache if k in FIELDS))))


# --------------------------------------------------------------------------- #
# 出图 / 出表（沿用课程约定）
# --------------------------------------------------------------------------- #
def savefig(fig, fname):
    p = os.path.join(FIG, fname)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fig.savefig(p)
    plt.close(fig)
    return p


def savetab(df, fname, index=True):
    p = os.path.join(TAB, fname)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    df.to_csv(p, index=index, encoding="utf-8-sig")
    return p


def main(argv=None):
    """命令行体检：python lib.py"""
    md = MarketData(verbose=True)
    print(md)
    print("\n各类掩码统计:")
    print(md.mask_report().to_string())

    T = md.ratio_table()
    print("\n涨跌幅上限诊断（逐股，按板块先验 10%% / 20%% 判定）:")
    bad = T["v_prior"] > 0
    print("  按自身板块先验判定，违规格数为 0 的证券: %d / %d"
          % (int((~bad).sum()), len(T)))
    print("  有违规的 %d 只，违规格合计 %d（多为复牌/退市整理等异常日）"
          % (int(bad.sum()), int(T["v_prior"].sum())))
    print("  5%% 严格判定标的: %s" % list(md.st_like[md.st_like].index))
    print("  上限被数据顶住的证券数: 10%% -> %d, 20%% -> %d"
          % (int(T["wall_10"].fillna(False).sum()), int(T["wall_20"].fillna(False).sum())))

    a = md.cumadj.pct_change(fill_method=None).abs()
    b = md.cumadj_clean.pct_change(fill_method=None).abs()
    print("\n复权因子去噪: |日变化| 落在 [1e-6,1e-4) 的格子 由 %d 降为 %d（真实事件 %d 格全部保留）"
          % (int(((a >= 1e-6) & (a < 1e-4)).to_numpy().sum()),
             int(((b >= 1e-6) & (b < 1e-4)).to_numpy().sum()),
             int((b >= 1e-4).to_numpy().sum())))

    print("\n涨跌停/一字板交叉验证:")
    print(md.at_limit_up.sum(1).nlargest(3).rename("涨停最多").to_string())
    print(md.at_limit_down.sum(1).nlargest(3).rename("跌停最多").to_string())

    n = md.n_eligible()
    print("\n股票池规模: 起 %d / 中 %d / 末 %d"
          % tuple(n.iloc[[0, len(n) // 2, -1]].tolist()))
    print("等权日收益年化波动: %.1f%%" % (md.equal_weight_ret().std() * np.sqrt(252) * 100))
    return md


if __name__ == "__main__":
    main()
