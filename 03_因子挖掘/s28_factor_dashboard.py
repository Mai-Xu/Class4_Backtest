# -*- coding: utf-8 -*-
"""s28: build one self-contained interactive HTML factor evaluation dashboard."""
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "03_因子挖掘"
TAB = OUT / "tables"
TARGET = OUT / "factor_evaluation_dashboard.html"
GRADES = ["S", "A", "B", "C", "D"]
LABEL = {"S": "S 核心增量", "A": "A 优先备选", "B": "B 有效偏 beta/冗余", "C": "C 观察/组合输入", "D": "D 低优先级"}
COLOR = {"S": "#c0392b", "A": "#e67e22", "B": "#2471a3", "C": "#7f8c8d", "D": "#bdc3c7"}

x = pd.read_csv(TAB / "27_factor_scorecard.csv")
fam = pd.read_csv(TAB / "27_family_scorecard.csv")
x["grade"] = pd.Categorical(x["grade"], categories=GRADES, ordered=True)
n = len(x); cov = int((x.coverage >= .6).sum()); rawq = int(x.raw_fdr05.sum()); resq = int(x.res_fdr05.sum())
sa = int(x.grade.isin(["S", "A"]).sum()); pool_n = int(x.research_pool.sum())
cnt = {g: int((x.grade == g).sum()) for g in GRADES}
rep = {g: int(((x.grade == g) & x.audit_rep).sum()) for g in GRADES}

fig1 = px.scatter(x, x="raw_ic_oos_adj", y="res_ic_oos_adj", color="grade", color_discrete_map=COLOR,
                  category_orders={"grade": GRADES}, hover_data=["factor", "family", "module", "score", "raw_q_hac", "res_q_hac", "net_adj", "ls_turnover_ann"],
                  labels={"raw_ic_oos_adj": "OOS 原始 RankIC20", "res_ic_oos_adj": "OOS 中性化后的独立 RankIC20", "grade": "等级"},
                  title="原始 IC 与独立增量的关系")
fig1.update_traces(marker=dict(size=7, opacity=.78, line=dict(width=0)))
fig1.add_hline(y=0, line_color="#333"); fig1.add_vline(x=0, line_color="#333")
for z in [.015, -.015]: fig1.add_hline(y=z, line_color="#999", line_dash="dot")
for z in [.02, -.02]: fig1.add_vline(x=z, line_color="#999", line_dash="dot")
fig1.update_layout(height=560, margin=dict(l=50, r=20, t=65, b=50))

fig2 = go.Figure()
fig2.add_bar(x=[LABEL[g] for g in GRADES], y=[cnt[g] for g in GRADES], name="因子数", marker_color=[COLOR[g] for g in GRADES], text=[cnt[g] for g in GRADES], textposition="outside")
fig2.add_bar(x=[LABEL[g] for g in GRADES], y=[rep[g] for g in GRADES], name="独立簇代表", marker_color="#17212b", opacity=.68, text=[rep[g] for g in GRADES], textposition="outside")
fig2.update_layout(barmode="group", height=390, title="评级分布与去重后的独立簇", margin=dict(l=45, r=20, t=65, b=45), legend=dict(orientation="h", y=1.12))

f = fam.sort_values("S_A_share").copy()
fig3 = go.Figure()
fig3.add_bar(y=f.family, x=f.S_A_share * 100, name="S/A 因子占比", marker_color="#2a9d8f", text=(f.S_A_share * 100).round(1).astype(str) + "%", textposition="outside")
fig3.add_bar(y=f.family, x=f.res_fdr_rate * 100, name="中性化后 FDR<5% 占比", marker_color="#e9c46a", text=(f.res_fdr_rate * 100).round(1).astype(str) + "%", textposition="outside")
fig3.update_layout(barmode="group", height=450, title="家族层面的独立增量证据", xaxis_title="占比（%）", margin=dict(l=145, r=45, t=65, b=45), legend=dict(orientation="h", y=1.12))

fig4 = px.box(x, x="grade", y="res_ic_oos_adj", color="grade", color_discrete_map=COLOR, category_orders={"grade": GRADES}, points="outliers",
              labels={"grade": "等级", "res_ic_oos_adj": "中性化 OOS RankIC20"}, title="不同评级的独立 IC 分布")
fig4.update_layout(height=420, showlegend=False, margin=dict(l=55, r=20, t=65, b=45)); fig4.add_hline(y=0, line_color="#333")
for z in [.015, -.015]: fig4.add_hline(y=z, line_color="#999", line_dash="dot")

top = x[x.research_pool].sort_values(["grade", "score"], ascending=[True, False]).head(25).sort_values("score")
fig5 = px.bar(top, x="score", y="factor", orientation="h", color="grade", color_discrete_map=COLOR, category_orders={"grade": GRADES},
              hover_data=["family", "module", "raw_ic_oos_adj", "res_ic_oos_adj", "raw_q_hac", "res_q_hac", "net_adj", "ls_turnover_ann"],
              labels={"score": "综合评分", "factor": "因子", "grade": "等级"}, title="S/A 独立簇代表：综合评分前 25")
fig5.update_layout(height=700, margin=dict(l=145, r=30, t=65, b=45), legend_title_text="等级")

def ph(fig, first=False):
    return pio.to_html(fig, include_plotlyjs="inline" if first else False, full_html=False,
                       config={"displaylogo": False, "responsive": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
plots = "".join([f'<article class="chart wide">{ph(fig1, True)}</article>', f'<article class="chart">{ph(fig2)}</article>', f'<article class="chart wide">{ph(fig3)}</article>', f'<article class="chart">{ph(fig4)}</article>', f'<article class="chart wide">{ph(fig5)}</article>'])

cols = ["grade","score","factor","family","module","desc","raw_ic_oos_adj","raw_q_hac","res_ic_oos_adj","res_q_hac","abs_mono","net_adj","ls_turnover_ann","max_abs_expo","corr_log_adv","corr_vol60","corr_ret20","cluster","audit_rep","research_pool","size_beta","vol_beta","momentum_beta","high_turnover"]
t = x[cols].copy()
for c in ["score","raw_ic_oos_adj","res_ic_oos_adj","abs_mono","net_adj","ls_turnover_ann","max_abs_expo","corr_log_adv","corr_vol60","corr_ret20"]:
    t[c] = pd.to_numeric(t[c], errors="coerce").round(5)
data_json = t.to_json(orient="records", force_ascii=False)

css = """
<style>
:root{--navy:#102a43;--ink:#17212b;--muted:#66788a;--line:#e4e9ef;--bg:#f4f7fa;--teal:#2a9d8f;--red:#c0392b;--orange:#e67e22}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif}
.hero{background:linear-gradient(135deg,#0d2235,#163b5c 65%,#1f5b72);color:#fff;padding:52px 36px 45px}.hero-inner{max-width:1480px;margin:auto}.hero h1{font-size:38px;margin:0 0 12px}.hero p{max-width:980px;color:#c8d8e6;line-height:1.65;margin:0}
.kpis{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;max-width:1480px;margin:-24px auto 0;padding:0 26px;position:relative;z-index:2}.kpi{background:#fff;border:1px solid var(--line);border-radius:14px;padding:15px 16px;box-shadow:0 7px 22px #0e2a4012}.kpi b{font-size:26px;display:block}.kpi span{font-size:12px;color:var(--muted)}
nav{position:sticky;top:0;z-index:10;background:#fffffff2;backdrop-filter:blur(12px);border-bottom:1px solid var(--line)}nav .inner{max-width:1480px;margin:auto;padding:11px 26px;display:flex;gap:8px;overflow:auto}nav a{color:#324b61;text-decoration:none;padding:7px 11px;border-radius:8px;white-space:nowrap;font-size:13px}nav a:hover{background:#eaf2f7}
main{max-width:1480px;margin:auto;padding:22px 26px 70px}.section{background:#fff;border:1px solid var(--line);border-radius:16px;padding:24px;margin-bottom:18px;box-shadow:0 5px 18px #0e2a4009}.section-head{display:flex;align-items:baseline;gap:14px;margin-bottom:16px}.eyebrow{font-size:11px;text-transform:uppercase;letter-spacing:.13em;color:var(--teal);font-weight:800}h2{font-size:23px;margin:0}
.callout-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.callout{border-radius:12px;padding:17px;border-left:5px solid;background:#f9fbfc}.callout p{line-height:1.62;color:#415466;margin:8px 0 0}.danger{border-color:var(--red)}.success{border-color:var(--teal)}.warning{border-color:var(--orange)}
.method-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:15px}.method-grid div{background:#f7fafc;border:1px solid var(--line);border-radius:11px;padding:14px}.method-grid span{font-size:11px;color:var(--teal);font-weight:900}.method-grid b{display:block;margin:4px 0 5px}.method-grid p{margin:0;color:var(--muted);font-size:12px;line-height:1.55}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:16px}.chart{background:#fff;border:1px solid var(--line);border-radius:14px;padding:8px}.chart.wide{grid-column:1/-1}
.grade-row{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:15px}.grade-card{background:#f8fafc;border:1px solid var(--line);border-top:4px solid var(--grade);border-radius:11px;padding:12px}.grade-card p{margin:0;color:var(--muted);font-size:12px}.grade-chip{display:inline-flex;align-items:center;justify-content:center;background:var(--grade);color:#fff;width:24px;height:24px;border-radius:7px;font-weight:800}
.toolbar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:12px}input,select,button{font:inherit;border:1px solid #c5d0da;border-radius:8px;padding:9px 11px;background:#fff}input{min-width:320px}button{cursor:pointer}button:hover{background:#f1f7fa}.toolbar .active{background:#173f5e;color:#fff;border-color:#173f5e}.status{margin-left:auto;color:var(--muted);font-size:13px}
.table-wrap{max-height:72vh;overflow:auto;border:1px solid var(--line);border-radius:11px}table{border-collapse:separate;border-spacing:0;width:100%;font-size:12px;white-space:nowrap}th,td{padding:7px 9px;border-bottom:1px solid #edf1f4;text-align:left}th{position:sticky;top:0;background:#eff4f8;z-index:2;cursor:pointer}tr:hover td{background:#f7fbfc}.grade-badge{font-weight:900;padding:2px 7px;border-radius:99px;color:#fff}.grade-S{background:#c0392b}.grade-A{background:#e67e22}.grade-B{background:#2471a3}.grade-C{background:#7f8c8d}.grade-D{background:#bdc3c7}.flag{display:inline-block;background:#eef3f6;color:#51697d;padding:1px 5px;border-radius:5px;margin-right:3px;font-size:10px}.pager{display:flex;gap:8px;align-items:center;justify-content:flex-end;margin-top:10px}.small{font-size:12px;color:var(--muted);line-height:1.65}.foot{color:#6b7c8d;font-size:12px;text-align:center;padding:15px}
@media(max-width:1050px){.kpis{grid-template-columns:repeat(3,1fr)}.callout-grid,.method-grid{grid-template-columns:1fr 1fr}.charts{grid-template-columns:1fr}.chart.wide{grid-column:auto}.grade-row{grid-template-columns:1fr 1fr}}
@media(max-width:650px){.hero{padding:35px 20px 40px}.hero h1{font-size:30px}.kpis{grid-template-columns:1fr 1fr;padding:0 15px}main{padding:16px 12px 50px}.section{padding:17px}.callout-grid,.method-grid{grid-template-columns:1fr}.grade-row{grid-template-columns:1fr}input{min-width:0;width:100%}.status{margin-left:0;width:100%}}
</style>
"""

body = f"""
<header class="hero"><div class="hero-inner"><h1>390 因子独立评价</h1><p>从样本外预测力、独立增量、相关簇冗余、交易成本和风险暴露五个角度，而不是只看 IC 排名。样本内截至 2024-11，样本外有效 21 期；OOS t 使用 Newey-West HAC，显著性使用 BH-FDR。</p></div></header>
<div class="kpis"><div class="kpi"><b>{n}</b><span>因子总数</span></div><div class="kpi"><b>{cov}</b><span>覆盖率≥60%</span></div><div class="kpi"><b>{rawq}</b><span>原始 OOS FDR&lt;5%</span></div><div class="kpi"><b>{resq}</b><span>中性化后 FDR&lt;5%</span></div><div class="kpi"><b>{sa}</b><span>S/A 因子</span></div><div class="kpi"><b>{pool_n}</b><span>研究池独立簇</span></div></div>
<nav><div class="inner"><a href="#summary">结论</a><a href="#charts">可视化</a><a href="#grades">评级</a><a href="#table">390 评分卡</a><a href="#method">方法边界</a><a href="#recommend">建议</a></div></nav>
<main>
<section class="section" id="summary"><div class="section-head"><span class="eyebrow">Executive Summary</span><h2>结论先说</h2></div><div class="callout-grid">
<article class="callout danger"><strong>390 ≠ 390 个可用因子</strong><p>{cov}/390 覆盖率合格，{rawq} 个通过原始 OOS FDR；剔除规模、波动和 20 日动量后只剩 <b>{resq}</b> 个仍在 FDR 5% 下显著。</p></article>
<article class="callout success"><strong>研究池：{sa} 个因子 / {pool_n} 个独立簇</strong><p>S 级 {cnt['S']} 个、A 级 {cnt['A']} 个。建议先保留每个相关簇的代表，再进入正交化、成本压力和组合回测。</p></article>
<article class="callout warning"><strong>最强信号普遍高换手</strong><p>候选因子年换手多在 8~11 倍。已有线性组合在 10bp 滑点下超额几乎归零，IC 好不能替代“扣摩擦后仍好”。</p></article></div>
<div class="method-grid"><div><span>01</span><b>干净切分</b><p>样本内仅用 2021-01 至 2024-11 的信号，20 日标签不跨入 2025。</p></div><div><span>02</span><b>相关簇去重</b><p>240 个相关簇，每个簇只保留一个审计代表，避免重复公式伪装成独立证据。</p></div><div><span>03</span><b>独立增量</b><p>对 log(ADV20)、60 日波动、20 日收益中性化，检查剩余 RankIC。</p></div><div><span>04</span><b>交易与风险</b><p>同时看 FDR、单调性、扣费多空收益、换手和风险暴露。</p></div></div></section>
<section class="section" id="charts"><div class="section-head"><span class="eyebrow">Visual Diagnostics</span><h2>可视化诊断</h2></div><div class="charts">{plots}</div></section>
<section class="section" id="grades"><div class="section-head"><span class="eyebrow">Rating</span><h2>评级图例</h2></div><div class="grade-row">{''.join(f'<div class="grade-card" style="--grade:{COLOR[g]}"><span class="grade-chip">{g}</span><b>{LABEL[g]}</b><p>{cnt[g]} 个因子 · {rep[g]} 个独立簇代表</p></div>' for g in GRADES)}</div><div class="small"><b>S：</b>原始与中性化后均通过 FDR，OOS 同向，独立 IC≥0.015、单调性≥0.5 且扣费收益为正。<b>A：</b>原始 FDR 通过、独立增量≥0.015，但严格 Res FDR 未过。<b>B：</b>原始显著但独立增量或去重条件不足。<b>C：</b>观察或仅适合组合输入。<b>D：</b>低优先级。</div></section>
<section class="section" id="table"><div class="section-head"><span class="eyebrow">Interactive Scorecard</span><h2>390 因子评分卡</h2></div><div class="toolbar"><input id="q" placeholder="搜索因子 / 家族 / 模块"><select id="grade"><option value="">全部等级</option><option>S</option><option>A</option><option>B</option><option>C</option><option>D</option></select><button id="poolBtn">只看 S/A 研究池</button><button id="repBtn">只看独立簇代表</button><button id="resetBtn">重置</button><button id="csvBtn">导出当前筛选 CSV</button><span class="status" id="status"></span></div><div class="table-wrap"><table id="tbl"><thead><tr><th data-k="grade">等级</th><th data-k="score">评分</th><th data-k="factor">因子</th><th data-k="family">家族</th><th data-k="module">模块</th><th data-k="raw_ic_oos_adj">OOS原始IC</th><th data-k="raw_q_hac">原始q</th><th data-k="res_ic_oos_adj">中性化IC</th><th data-k="res_q_hac">中性化q</th><th data-k="abs_mono">|单调性|</th><th data-k="net_adj">扣费多空年化</th><th data-k="ls_turnover_ann">年换手</th><th data-k="max_abs_expo">最大暴露</th><th data-k="cluster">簇</th><th>标记</th></tr></thead><tbody></tbody></table></div><div class="pager"><button id="prev">上一页</button><span id="pageInfo"></span><button id="next">下一页</button></div></section>
<section class="section" id="method"><div class="section-head"><span class="eyebrow">Limits</span><h2>方法和边界</h2></div><div class="callout-grid"><article class="callout success"><strong>本次额外做了什么</strong><p>严格样本切分、HAC t、FDR 多重检验、风格中性化、相关簇去重，并把成本和单调性纳入评级。</p></article><article class="callout warning"><strong>仍不能过度解读</strong><p>OOS 只有 21 个月，供研究的因子很多。FDR 不能完全消除共同样本、共同因子和研究者自由度。</p></article><article class="callout danger"><strong>数据缺口</strong><p>缺少真实市值、行业、ST、成交量/换手率、指数成分和逐笔/分钟数据，无法完成真正行业中性化、拥挤度和容量评估。</p></article></div></section>
<section class="section" id="recommend"><div class="section-head"><span class="eyebrow">Next Step</span><h2>建议</h2></div><ol class="small"><li><b>先做簇内去重：</b>从 {pool_n} 个 S/A 独立簇代表开始，不把同一相关簇的多个公式同时视为独立证据。</li><li><b>再做正交化和组合：</b>比较线性 IC_IR、正交化和树模型合成的增量 IC 与净超额。</li><li><b>最后做成本压力：</b>加 10bp、20bp 滑点及参与率约束，高换手但不抗摩擦的因子降级为条件信号。</li><li><b>补数据后重评：</b>拿到真实市值、行业、成交量后重新中性化和评估拥挤度。</li></ol></section><div class="foot">单文件 HTML · 图表和数据已内嵌 · 可离线打开</div></main>
"""

script = r"""
<script>
const ALL=__DATA__;let view=[],page=1,size=40,sk='score',sa=false,pool=false,rep=false;const $=s=>document.querySelector(s);const v=x=>(x===null||x===undefined||x==='')?'—':x;const num=x=>(x===null||x===undefined||Number.isNaN(Number(x)))?null:Number(x);
function ok(r){const q=$('#q').value.trim().toLowerCase(),g=$('#grade').value;if(q&&!(`${r.factor} ${r.family} ${r.module}`.toLowerCase().includes(q)))return false;if(g&&r.grade!==g)return false;if(pool&&!r.research_pool)return false;if(rep&&!r.audit_rep)return false;return true}
function row(r){const tags=[];if(r.audit_rep)tags.push('代表');if(r.research_pool)tags.push('研究池');if(r.size_beta)tags.push('规模beta');if(r.vol_beta)tags.push('波动beta');if(r.momentum_beta)tags.push('动量beta');if(r.high_turnover)tags.push('高换手');const qv=x=>x==null?'—':Number(x).toExponential(2);return `<tr title="${r.desc||''}"><td><span class="grade-badge grade-${r.grade}">${r.grade}</span></td><td>${num(r.score)?.toFixed(2)??'—'}</td><td><b>${r.factor}</b></td><td>${r.family}</td><td>${r.module}</td><td>${num(r.raw_ic_oos_adj)?.toFixed(4)??'—'}</td><td>${qv(r.raw_q_hac)}</td><td>${num(r.res_ic_oos_adj)?.toFixed(4)??'—'}</td><td>${qv(r.res_q_hac)}</td><td>${num(r.abs_mono)?.toFixed(3)??'—'}</td><td>${num(r.net_adj)?.toFixed(3)??'—'}</td><td>${num(r.ls_turnover_ann)?.toFixed(2)??'—'}</td><td>${num(r.max_abs_expo)?.toFixed(3)??'—'}</td><td>${r.cluster}</td><td>${tags.map(t=>`<span class="flag">${t}</span>`).join('')}</td></tr>`}
function render(){view=ALL.filter(ok);const pages=Math.max(1,Math.ceil(view.length/size));page=Math.min(page,pages);view.sort((a,b)=>{let x=a[sk],y=b[sk];if(typeof x==='string'||typeof y==='string'){x=String(x||'');y=String(y||'');return sa?x.localeCompare(y,'zh-CN'):y.localeCompare(x,'zh-CN')}x=num(x);y=num(y);if(x==null&&y==null)return 0;if(x==null)return 1;if(y==null)return -1;return sa?x-y:y-x});$('#tbl tbody').innerHTML=view.slice((page-1)*size,page*size).map(row).join('');$('#pageInfo').textContent=`第 ${page} / ${pages} 页，共 ${view.length} 条`;$('#status').textContent=`显示 ${view.length} / ${ALL.length} 个因子`}
document.querySelectorAll('th[data-k]').forEach(th=>th.onclick=()=>{const k=th.dataset.k;sa=sk===k?!sa:false;sk=k;render()});$('#q').oninput=()=>{page=1;render()};$('#grade').onchange=()=>{page=1;render()};$('#poolBtn').onclick=()=>{pool=!pool;$('#poolBtn').classList.toggle('active',pool);page=1;render()};$('#repBtn').onclick=()=>{rep=!rep;$('#repBtn').classList.toggle('active',rep);page=1;render()};$('#resetBtn').onclick=()=>{$('#q').value='';$('#grade').value='';pool=rep=false;$('#poolBtn').classList.remove('active');$('#repBtn').classList.remove('active');page=1;sk='score';sa=false;render()};$('#prev').onclick=()=>{if(page>1){page--;render()}};$('#next').onclick=()=>{if(page*size<view.length){page++;render()}};$('#csvBtn').onclick=()=>{const k=['grade','score','factor','family','module','raw_ic_oos_adj','raw_q_hac','res_ic_oos_adj','res_q_hac','abs_mono','net_adj','ls_turnover_ann','max_abs_expo','cluster','audit_rep','research_pool'];const lines=[k.join(',')].concat(view.map(r=>k.map(x=>`"${String(r[x]??'').replaceAll('"','""')}"`).join(',')));const b=new Blob(['\ufeff'+lines.join('\n')],{type:'text/csv;charset=utf-8'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='factor_scorecard_filtered.csv';a.click();URL.revokeObjectURL(a.href)};render();
</script>
""".replace("__DATA__", data_json)

html = "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>390 因子独立评价</title>" + css + "</head><body>" + body + script + "</body></html>"
TARGET.write_text(html, encoding="utf-8")
print("saved", TARGET, "bytes", TARGET.stat().st_size)
