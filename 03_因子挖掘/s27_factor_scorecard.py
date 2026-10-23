# -*- coding: utf-8 -*-
"""s27: 390 因子独立评价 / 评分卡。

输入：
  tables/20_zoo_check.csv                 因果性、覆盖率
  tables/21_factor_stats.csv              IC / ICIR / 分层 / 稳定性 / 聚类
  tables/21_decile.csv                    十层收益
  tables/21_exposure.csv                  规模、波动、20日动量暴露
  tables/27_neutralized_ic_clean.csv      干净样本切分 + 中性化 IC
输出：
  tables/27_factor_scorecard.csv
  tables/27_family_scorecard.csv
  tables/27_module_scorecard.csv
  figures/27_*.png
  factor_evaluation_report.md
  factor_evaluation_dashboard.html

评价口径：
- 样本内信号日截至 2024-11-30，避免 20 日标签跨入 2025 年；
- 样本外信号日为 2024-12 至 2026-08，有效 OOS 21 期；
- IC 使用 Spearman RankIC；t 值用 Newey-West(lag=3)；
- 独立增量 IC 对 log(ADV20)、60日波动、20日收益三个横截面暴露做秩中性化；
- 多重检验用 Benjamini-Hochberg FDR 校正。
"""
from __future__ import annotations

import base64
import io
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "03_因子挖掘"
TAB = OUT / "tables"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

GRADE_ORDER = {"S": 0, "A": 1, "B": 2, "C": 3, "D": 4}
GRADE_LABEL = {
    "S": "S 核心增量",
    "A": "A 优先备选",
    "B": "B 有效但偏 beta/冗余",
    "C": "C 观察/仅组合输入",
    "D": "D 低优先级",
}


def read_table(name: str) -> pd.DataFrame:
    return pd.read_csv(TAB / name)


def pct_rank(s: pd.Series, ascending: bool = True) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.rank(pct=True, ascending=ascending).fillna(0.5).clip(0, 1)


def orient_sign(x: pd.Series) -> pd.Series:
    s = np.sign(pd.to_numeric(x, errors="coerce"))
    return s.replace(0, np.nan).fillna(1.0)


def q_values(t: pd.Series, direction: pd.Series, alternative: str = "greater") -> pd.Series:
    t = pd.to_numeric(t, errors="coerce")
    direction = pd.to_numeric(direction, errors="coerce")
    if alternative == "greater":
        p = np.where(direction > 0, norm.sf(t), norm.sf(-t))
    else:
        p = 2.0 * norm.sf(np.abs(t))
    p = np.asarray(p, dtype=float)
    p[~np.isfinite(p)] = 1.0
    return pd.Series(multipletests(p, method="fdr_bh")[1], index=t.index)


def build_scorecard() -> pd.DataFrame:
    stats = read_table("21_factor_stats.csv")
    neut = read_table("27_neutralized_ic_clean.csv")
    dec = read_table("21_decile.csv")
    expo = read_table("21_exposure.csv")
    qual = read_table("20_zoo_check.csv")

    x = (
        stats.merge(neut, on="factor", how="left", suffixes=("_old", "_clean"))
        .merge(dec[["factor", "spread", "mono"]], on="factor", how="left", suffixes=("", "_decile"))
        .merge(expo, on="factor", how="left")
        .merge(qual[["factor", "ok", "causal", "coverage"]], on="factor", how="left")
    )

    direction = orient_sign(x["raw_ic_is"])
    x["direction_audit"] = direction
    for c in [
        "raw_ic_is", "raw_ic_oos", "raw_icir_is", "raw_icir_oos",
        "res_ic_is", "res_ic_oos", "res_icir_is", "res_icir_oos",
    ]:
        x[c + "_adj"] = x[c] * direction
    x["mono_adj"] = x["mono"] * direction
    x["spread_adj"] = x["spread"] * direction
    x["net_adj"] = x["ls_net_ann"] * direction

    x["same_raw"] = np.sign(x["raw_ic_oos"]) == direction
    x["same_res"] = np.sign(x["res_ic_oos"]) == direction
    x["abs_raw_ic_is"] = x["raw_ic_is"].abs()
    x["abs_raw_ic_oos"] = x["raw_ic_oos"].abs()
    x["abs_res_ic_is"] = x["res_ic_is"].abs()
    x["abs_res_ic_oos"] = x["res_ic_oos"].abs()
    x["abs_raw_ir_is"] = x["raw_icir_is"].abs()
    x["abs_raw_ir_oos"] = x["raw_icir_oos"].abs()
    x["abs_res_ir_is"] = x["res_icir_is"].abs()
    x["abs_res_ir_oos"] = x["res_icir_oos"].abs()
    x["abs_mono"] = x["mono"].abs()

    x["raw_q_hac"] = q_values(x["raw_t_hac_oos"], direction)
    x["res_q_hac"] = q_values(x["res_t_hac_oos"], direction)
    x["raw_fdr05"] = x["raw_q_hac"] < 0.05
    x["res_fdr05"] = x["res_q_hac"] < 0.05

    expo_cols = ["corr_log_adv", "corr_vol60", "corr_ret20"]
    x["max_abs_expo"] = x[expo_cols].abs().max(axis=1)
    x["size_beta"] = x["corr_log_adv"].abs() >= 0.80
    x["vol_beta"] = x["corr_vol60"].abs() >= 0.80
    x["momentum_beta"] = x["corr_ret20"].abs() >= 0.80
    x["high_turnover"] = x["ls_turnover_ann"] >= 10.0
    x["low_persistence"] = x["stability"] < 0.20
    x["is_duplicate"] = ~x["is_rep"].fillna(False).astype(bool)
    x["raw_oos_retention"] = np.where(
        x["abs_raw_ic_is"] > 1e-9,
        x["raw_ic_oos_adj"] / x["abs_raw_ic_is"],
        np.nan,
    )
    x["res_oos_retention"] = np.where(
        x["abs_raw_ic_oos"] > 1e-9,
        x["res_ic_oos_adj"] / x["abs_raw_ic_oos"],
        np.nan,
    )

    eligible = (
        x["ok"].fillna(False).astype(bool)
        & x["causal"].fillna(False).astype(bool)
        & (x["coverage"] >= 0.60)
    )
    s_rule = (
        eligible
        & x["raw_fdr05"] & x["res_fdr05"]
        & x["same_raw"] & x["same_res"]
        & (x["abs_res_ic_oos"] >= 0.015)
        & (x["abs_mono"] >= 0.50)
        & (x["net_adj"] > 0)
    )
    a_rule = (
        eligible & ~s_rule
        & x["raw_fdr05"] & x["same_raw"] & x["same_res"]
        & (x["abs_res_ic_oos"] >= 0.015)
        & (x["abs_mono"] >= 0.50)
        & (x["net_adj"] > 0)
    )
    b_rule = (
        eligible & ~s_rule & ~a_rule
        & x["raw_fdr05"] & x["same_raw"]
        & (x["abs_raw_ic_oos"] >= 0.02)
        & (x["abs_mono"] >= 0.50)
        & (x["net_adj"] > 0)
    )
    c_rule = (
        eligible & ~s_rule & ~a_rule & ~b_rule
        & (
            (x["same_raw"] & (x["abs_raw_ic_oos"] >= 0.02))
            | (x["same_res"] & (x["abs_res_ic_oos"] >= 0.01))
        )
    )
    x["grade"] = np.select(
        [s_rule, a_rule, b_rule, c_rule],
        ["S", "A", "B", "C"],
        default="D",
    )
    x["grade_order"] = x["grade"].map(GRADE_ORDER)

    # 相对评分：预测力、独立增量、形状、交易性、稳健/分散度。
    raw_power = (
        0.50 * pct_rank(x["abs_raw_ic_oos"])
        + 0.30 * pct_rank(x["abs_raw_ir_oos"])
        + 0.20 * pct_rank(-np.log10(x["raw_q_hac"].clip(lower=1e-12)))
    )
    incr_power = (
        0.40 * pct_rank(x["abs_res_ic_oos"])
        + 0.30 * pct_rank(x["abs_res_ir_oos"])
        + 0.30 * pct_rank(-np.log10(x["res_q_hac"].clip(lower=1e-12)))
    )
    shape = 0.60 * pct_rank(x["abs_mono"]) + 0.40 * pct_rank(x["spread_adj"])
    trade = (
        0.45 * pct_rank(x["net_adj"])
        + 0.35 * pct_rank(x["ls_turnover_ann"], ascending=False)
        + 0.20 * pct_rank(x["stability"])
    )
    robustness = (
        0.40 * pct_rank(x["coverage"])
        + 0.40 * pct_rank(x["max_abs_expo"], ascending=False)
        + 0.20 * x["is_rep"].fillna(False).astype(float)
    )
    x["score"] = 100.0 * (
        0.25 * raw_power
        + 0.35 * incr_power
        + 0.15 * shape
        + 0.15 * trade
        + 0.10 * robustness
    )

    # 每个相关簇选一个审计代表：优先等级，其次综合分。
    x = x.sort_values(["cluster", "grade_order", "score"], ascending=[True, True, False]).copy()
    x["audit_rep"] = ~x.duplicated("cluster", keep="first")
    x["research_pool"] = x["audit_rep"] & x["grade"].isin(["S", "A"])
    x = x.sort_values(["grade_order", "score", "abs_res_ic_oos"], ascending=[True, False, False])
    return x.reset_index(drop=True)


def summarize(x: pd.DataFrame, key: str) -> pd.DataFrame:
    g = x.groupby(key, dropna=False)
    out = pd.DataFrame({
        "n": g.size(),
        "S": g["grade"].apply(lambda s: int((s == "S").sum())),
        "A": g["grade"].apply(lambda s: int((s == "A").sum())),
        "B": g["grade"].apply(lambda s: int((s == "B").sum())),
        "C": g["grade"].apply(lambda s: int((s == "C").sum())),
        "D": g["grade"].apply(lambda s: int((s == "D").sum())),
        "audit_rep": g["audit_rep"].sum().astype(int),
        "research_pool_clusters": g["research_pool"].sum().astype(int),
        "raw_fdr_rate": g["raw_fdr05"].mean(),
        "res_fdr_rate": g["res_fdr05"].mean(),
        "median_abs_raw_ic_oos": g["abs_raw_ic_oos"].median(),
        "median_abs_res_ic_oos": g["abs_res_ic_oos"].median(),
        "median_res_retention": g["res_oos_retention"].median(),
        "median_net_adj": g["net_adj"].median(),
        "median_turnover": g["ls_turnover_ann"].median(),
        "median_max_expo": g["max_abs_expo"].median(),
    })
    out["S_A"] = out["S"] + out["A"]
    out["S_A_share"] = out["S_A"] / out["n"]
    out = out.sort_values(["S_A_share", "median_abs_res_ic_oos"], ascending=False)
    return out.reset_index()


def fmt_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "factor", "family", "module", "grade", "score",
        "raw_ic_oos_adj", "raw_q_hac", "res_ic_oos_adj", "res_q_hac",
        "mono_adj", "net_adj", "ls_turnover_ann", "max_abs_expo",
        "cluster", "audit_rep",
    ]
    z = df[cols].copy()
    for c in ["score", "raw_ic_oos_adj", "res_ic_oos_adj", "mono_adj", "net_adj", "ls_turnover_ann", "max_abs_expo"]:
        z[c] = pd.to_numeric(z[c], errors="coerce").round(4)
    for c in ["raw_q_hac", "res_q_hac"]:
        z[c] = pd.to_numeric(z[c], errors="coerce").map(lambda v: "" if pd.isna(v) else f"{v:.2e}")
    return z


def make_figures(x: pd.DataFrame, family: pd.DataFrame) -> list[Path]:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    paths = []
    colors = {"S": "#c0392b", "A": "#e67e22", "B": "#2980b9", "C": "#7f8c8d", "D": "#bdc3c7"}

    fig, ax = plt.subplots(figsize=(9, 6))
    for grade in ["D", "C", "B", "A", "S"]:
        z = x[x["grade"] == grade]
        ax.scatter(z["raw_ic_oos_adj"], z["res_ic_oos_adj"], s=18, alpha=.72,
                   c=colors[grade], label=GRADE_LABEL[grade], edgecolors="none")
    ax.axhline(0, color="black", lw=.8)
    ax.axvline(0, color="black", lw=.8)
    ax.axhline(0.015, color="gray", lw=.8, ls="--")
    ax.axhline(-0.015, color="gray", lw=.8, ls="--")
    ax.axvline(0.02, color="gray", lw=.8, ls="--")
    ax.axvline(-0.02, color="gray", lw=.8, ls="--")
    ax.set_xlabel("OOS Raw RankIC20 (方向统一)")
    ax.set_ylabel("OOS 中性化后 RankIC20")
    ax.set_title("原始 OOS IC vs 剔除规模/波动/动量后的独立 IC")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    p = FIG / "27_raw_vs_residual_ic.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    paths.append(p)

    counts = x.groupby("grade").size().reindex(["S", "A", "B", "C", "D"]).fillna(0)
    reps = x[x["audit_rep"]].groupby("grade").size().reindex(counts.index).fillna(0)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    y = np.arange(len(counts))
    ax.barh(y - .18, counts.values, height=.34, label="因子数", color="#4c78a8")
    ax.barh(y + .18, reps.values, height=.34, label="独立簇代表数", color="#f58518")
    ax.set_yticks(y, [GRADE_LABEL[g] for g in counts.index])
    ax.invert_yaxis()
    ax.set_xlabel("数量")
    ax.set_title("390 因子评价分级")
    ax.legend()
    fig.tight_layout()
    p = FIG / "27_grade_counts.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    paths.append(p)

    fam = family.copy()
    fam = fam.sort_values("S_A_share", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(fam["family"], fam["S_A_share"] * 100, color="#2a9d8f")
    ax.set_xlabel("S/A 因子占比 (%)")
    ax.set_title("按因子家族的独立增量证据覆盖")
    fig.tight_layout()
    p = FIG / "27_family_sa_share.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    paths.append(p)

    top = x[x["research_pool"]].sort_values(["grade_order", "score"], ascending=[True, False]).head(25).copy()
    top = top.sort_values("score")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["factor"], top["score"], color=[colors[g] for g in top["grade"]])
    ax.set_xlabel("综合评分")
    ax.set_title("独立簇代表因子综合评分前 25")
    fig.tight_layout()
    p = FIG / "27_top_factor_scores.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    paths.append(p)
    return paths


def img_data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def build_report(x: pd.DataFrame, family: pd.DataFrame, module: pd.DataFrame, fig_paths: list[Path]) -> str:
    n = len(x)
    s, a, b, c, d = [int((x["grade"] == g).sum()) for g in ["S", "A", "B", "C", "D"]]
    sa = s + a
    raw_fdr = int(x["raw_fdr05"].sum())
    res_fdr = int(x["res_fdr05"].sum())
    raw_rep = int((x["raw_fdr05"] & x["audit_rep"]).sum())
    res_rep = int((x["res_fdr05"] & x["audit_rep"]).sum())
    pool_clusters = int(x["research_pool"].sum())
    top = x[x["research_pool"]].sort_values(["grade_order", "score"], ascending=[True, False]).head(30)
    top_tab = fmt_table(top)

    def md_table(df: pd.DataFrame) -> str:
        return df.to_markdown(index=False)

    report = []
    report.append("# 390 因子评价报告\n")
    report.append("> 口径：非 ST 全市场动态池；RankIC20；样本内信号截至 **2024-11-30**；样本外为 2024-12 至 2026-08，有效 21 期；OOS t 使用 Newey-West(lag=3)；独立增量 IC 对 log(ADV20)、60 日波动、20 日收益做横截面秩中性化；显著性使用 BH-FDR。\n")
    report.append("## 1. 结论\n")
    report.append(f"- **因子工程完成度高，但不等于 390 个因子都可用。** 390/390 通过因果性/形状校验，388 个覆盖率 ≥60%；但只有 {raw_rep} 个独立簇代表通过原始 OOS FDR，剔除规模/波动/动量后进一步降到 {res_rep} 个。\n")
    report.append(f"- **可进入研究池的高度候选为 {sa} 个因子，来自 {pool_clusters} 个独立相关簇**；其中 S 级 {s} 个，A 级 {a} 个。\n")
    report.append(f"- **分级结果：S={s}, A={a}, B={b}, C={c}, D={d}。** S/A 有独立增量证据；B 主要是原始有效但 beta 暴露或冗余较高；C 适合作为非线性模型/组合输入，不建议单因子实盘；D 为低优先级。\n")
    report.append(f"- **多重检验后：** 原始 OOS FDR<0.05 共 {raw_fdr} 个；中性化后仍有独立 OOS FDR<0.05 的仅 {res_fdr} 个。这个收缩幅度说明大量原始 IC 来自规模、波动或反转暴露。\n")
    report.append("- **最值得保留的方向：** 隔夜/日内反转与量价背离、流动性和成交结构、波动率偏斜及趋势强度。它们的共同点是 OOS 和中性化后仍保留较大 IC；但换手普遍偏高，必须用 10bp 以上滑点复核。\n")
    report.append("- **不建议直接做的事：** 把 390 个因子全部等权喂给线性模型；把同一相关簇的多个近似因子当作独立证据；或在没有滑点、没有规模中性化的口径下直接宣称稳定 alpha。\n")

    report.append("## 2. 评价方法\n")
    report.append("评分由五部分构成：原始预测力 25%、独立增量 35%、分层形状 15%、交易可行性 15%、稳健/分散度 10%。评级不是只看 IC，而是同时要求样本外同向、FDR、单调性、扣费后收益、覆盖率和风险暴露。\n")
    report.append("关键评级规则：\n")
    report.append("| 等级 | 规则摘要 | 含义 |\n|---|---|---|\n| S | raw/res 双 FDR；OOS 同向；|ResIC20|≥0.015；|单调性|≥0.5；扣费后收益>0 | 核心独立增量候选 |\n| A | raw FDR；OOS 同向；|ResIC20|≥0.015；单调性/扣费收益合格，但 Res FDR 未过 | 优先备选 |\n| B | raw FDR、OOS 同向、单调性合格，但独立增量或冗余条件不足 | 有效但偏 beta/冗余 |\n| C | OOS 同向且达到中等 IC，或仅有中性化后增量 | 观察/组合输入 |\n| D | 覆盖率不足、样本外反向、IC 过弱或成本后不合格 | 低优先级 |\n")

    report.append("## 3. 家族评价\n")
    fam_show = family.copy()
    for c0 in ["raw_fdr_rate", "res_fdr_rate", "S_A_share"]:
        fam_show[c0] = (fam_show[c0] * 100).round(1).astype(str) + "%"
    for c0 in ["median_abs_raw_ic_oos", "median_abs_res_ic_oos", "median_net_adj", "median_turnover"]:
        fam_show[c0] = fam_show[c0].round(3)
    report.append(md_table(fam_show[[
        "family", "n", "S", "A", "B", "C", "D", "S_A_share",
        "median_abs_raw_ic_oos", "median_abs_res_ic_oos",
        "raw_fdr_rate", "res_fdr_rate", "median_net_adj", "median_turnover",
    ]]))
    report.append("\n解读：\n")
    report.append(f"- **流动性与交易活跃度**的原始命中率最高，但许多因子在剔除 log(ADV) 后 IC 大幅衰减，属于“有效但混有规模/流动性 beta”。\n")
    report.append(f"- **量价关系与资金流、趋势与形态、波动率与风险**提供了较多中性化后仍存在增量 IC 的候选；其中波动率家族整体换手较低，交易可行性相对更好。\n")
    report.append(f"- **交互与合成、日历与季节性**的原始 IC 看似较强，但独立簇覆盖和多层单调性较弱，更适合作为非线性模型的交互输入，而不是单独实盘因子。\n")

    report.append("## 4. 独立簇代表：综合评分前 30\n")
    report.append(md_table(top_tab))
    report.append("\n说明：`raw_ic_oos_adj` 和 `res_ic_oos_adj` 已按样本内方向统一符号，正值表示方向正确；`audit_rep=True` 表示该因子是当前相关簇下按“等级优先、评分次之”重新选出的代表。\n")

    report.append("## 5. 主要风险与建议\n")
    report.append("1. **多重检验和因子相关性**：390 个因子的检验并不独立。报告中的 FDR 是对 390 个假设做校正，但仍不能完全消除共同因子、共同样本和反复试验带来的选择偏差。\n")
    report.append("2. **样本外只有 21 个月**：月度 IC 的有效样本很短，且 20 日收益重叠。S/A 结论应看作“进入下一轮验证”，不是“已验证可实盘”。\n")
    report.append("3. **规模/流动性 beta**：大量强 IC 来自 log(ADV) 或流动性结构；中性化后消失的因子不能作为独立 alpha。若要保留小市值 beta，应与 alpha 分开管理。\n")
    report.append("4. **换手与滑点**：多数候选年化换手 8~11 倍。现有线性组合在加 10bp 滑点后超额几乎归零，因此评分高的因子也必须通过成本压力测试。\n")
    report.append("5. **组合建议**：先取 S/A 的独立簇代表（建议 51 个以内），再做相关性和正交化。B/C 因子只进入树模型或条件交互池；D 因子暂不作为生产线输入。\n")
    report.append("6. **需要补齐的数据**：真实流通市值、行业、ST 标识、成交量/换手率、指数成分和逐笔或分钟数据，用于完成真正的行业/市值中性化、拥挤度和容量评估。\n")

    report.append("## 6. 产物\n")
    report.append("- `tables/27_factor_scorecard.csv`：390 个因子全量评分卡。\n")
    report.append("- `tables/27_neutralized_ic_clean.csv`：干净样本切分下的原始/中性化 IC。\n")
    report.append("- `tables/27_ic_series.parquet`：390×2 条月度 IC 序列。\n")
    report.append("- `tables/27_family_scorecard.csv` / `tables/27_module_scorecard.csv`：家族、模块汇总。\n")
    report.append("- `factor_evaluation_dashboard.html`：可搜索的全量评分看板。\n")
    return "\n".join(report)


def build_html(x: pd.DataFrame, fig_paths: list[Path], report_md: str) -> str:
    top = x[x["research_pool"]].sort_values(["grade_order", "score"], ascending=[True, False])
    show_cols = [
        "factor", "family", "grade", "score", "raw_ic_oos_adj", "raw_q_hac",
        "res_ic_oos_adj", "res_q_hac", "abs_mono", "net_adj",
        "ls_turnover_ann", "max_abs_expo", "cluster", "audit_rep",
        "size_beta", "vol_beta", "momentum_beta", "high_turnover",
    ]
    table = x[show_cols].copy()
    table["grade"] = table["grade"].map(GRADE_LABEL)
    for c in ["score", "raw_ic_oos_adj", "res_ic_oos_adj", "abs_mono", "net_adj", "ls_turnover_ann", "max_abs_expo"]:
        table[c] = table[c].round(4)
    for c in ["raw_q_hac", "res_q_hac"]:
        table[c] = table[c].map(lambda v: "" if pd.isna(v) else f"{v:.2e}")
    top_html = top[show_cols].to_html(index=False, classes="score-table", border=0, escape=False)
    all_html = table.to_html(index=True, classes="score-table", border=0, escape=False)
    figs = "".join(f'<figure><img src="{img_data_uri(p)}" alt="{p.name}"></figure>' for p in fig_paths)
    md_body = ""
    # 使用简单转义后的 pre 保留 Markdown 结论，HTML 主体以图表和表格为主。
    import html as _html
    md_body = "<pre class='report'>" + _html.escape(report_md) + "</pre>"
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>390 因子独立评价</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Microsoft YaHei',sans-serif;margin:0;background:#f5f6f8;color:#17212b}}
header{{background:#13293d;color:#fff;padding:28px 36px}}
header h1{{margin:0;font-size:26px}} header p{{margin:8px 0 0;color:#c7d3df}}
main{{padding:24px 30px 50px;max-width:1500px;margin:auto}}
section{{background:#fff;border-radius:12px;box-shadow:0 2px 12px #0000000d;padding:20px 24px;margin-bottom:18px}}
h2{{border-left:5px solid #2a9d8f;padding-left:10px}}
figure{{margin:14px 0;text-align:center}} img{{max-width:100%;border-radius:8px}}
table{{border-collapse:collapse;font-size:12px;width:100%;white-space:nowrap}}
th{{position:sticky;top:0;background:#eaf0f6;text-align:left;z-index:1}}
th,td{{border-bottom:1px solid #e5e9ef;padding:5px 7px}}
tr:hover{{background:#f7fbfa}} .table-wrap{{overflow:auto;max-height:70vh}}
pre.report{{white-space:pre-wrap;font-family:inherit;line-height:1.55;font-size:13px}}
input{{padding:9px 12px;border:1px solid #b8c2cc;border-radius:7px;width:320px;margin:4px 0 14px}}
</style></head><body>
<header><h1>390 因子独立评价</h1><p>RankIC20 · 严格样本内截止 · Newey-West HAC · BH-FDR · 规模/波动/动量中性化 · 相关簇去重</p></header>
<main>
<section><h2>结论摘要</h2>{md_body}</section>
<section><h2>图表</h2>{figs}</section>
<section><h2>研究池：S/A 独立簇代表</h2><div class="table-wrap">{top_html}</div></section>
<section><h2>全量 390 因子评分卡</h2><input id="filter" placeholder="搜索因子名、家族或等级..."><div class="table-wrap">{all_html}</div></section>
</main>
<script>
const inp=document.getElementById('filter');
inp.addEventListener('input',()=>{{const q=inp.value.toLowerCase();document.querySelectorAll('.score-table tbody tr').forEach(tr=>{{tr.style.display=tr.innerText.toLowerCase().includes(q)?'':'none';}});}});
</script></body></html>"""


def main() -> None:
    x = build_scorecard()
    fam = summarize(x, "family")
    mod = summarize(x, "module")
    x.to_csv(TAB / "27_factor_scorecard.csv", index=False, encoding="utf-8-sig")
    fam.to_csv(TAB / "27_family_scorecard.csv", index=False, encoding="utf-8-sig")
    mod.to_csv(TAB / "27_module_scorecard.csv", index=False, encoding="utf-8-sig")
    figs = make_figures(x, fam)
    report = build_report(x, fam, mod, figs)
    (OUT / "factor_evaluation_report.md").write_text(report, encoding="utf-8")
    (OUT / "factor_evaluation_dashboard.html").write_text(build_html(x, figs, report), encoding="utf-8")
    print("grade counts:", x["grade"].value_counts().reindex(["S", "A", "B", "C", "D"]).to_dict())
    print("audit reps:", x.groupby("grade")["audit_rep"].sum().reindex(["S", "A", "B", "C", "D"]).to_dict())
    print("research pool clusters:", int(x["research_pool"].sum()))
    print("saved:", TAB / "27_factor_scorecard.csv")
    print("saved:", OUT / "factor_evaluation_report.md")
    print("saved:", OUT / "factor_evaluation_dashboard.html")


if __name__ == "__main__":
    main()
