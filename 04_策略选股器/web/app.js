/* ============================================================
   因子组合策略选股器 · 前端
   ============================================================ */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));

const COLOR = { mine:'#5b5bd6', factor_avg:'#0ea5a5', market_ew:'#94a3b8', mf1700:'#f0a020', ml1700:'#e11d48' };
const DASH  = { market_ew:'4 3', mf1700:'5 3', ml1700:'2 2' };
const FAMS  = ['动量与反转','波动率与风险','流动性与交易活跃度','量价关系与资金流','趋势与形态','日历与季节性','交互与合成'];

const pct = (v, d = 2) => (v === null || v === undefined || !isFinite(v)) ? '—' : (v * 100).toFixed(d) + '%';
const sgn = (v, d = 2) => (v === null || v === undefined || !isFinite(v)) ? '—' : (v >= 0 ? '+' : '') + (v * 100).toFixed(d) + '%';
const num = (v, d = 2) => (v === null || v === undefined || !isFinite(v)) ? '—' : Number(v).toFixed(d);
const yi  = v => (v / 1e8).toFixed(2) + ' 亿元';
const clsP = v => (v >= 0 ? 'pos' : 'neg');

const S = {
  boot: null, mode: 'score', sel: new Map(), nHold: 1700, buffer: 1.0,
  q: '', fam: 'all', expanded: new Set(), off: new Set(), log: false,
  result: null, running: false, tab: 'hold',
};

/* ---------------------------------------------------------- 启动 */
fetch('/api/bootstrap').then(r => r.json()).then(d => {
  if (!d.ok) throw new Error('bootstrap failed');
  S.boot = d;
  S.nHold = d.default.n_hold;
  renderChips();
  renderFamilies();
  renderFactorList();
  applyPresetIcir(true);
  bindTop();
  $('#placeholder').classList.remove('hidden');
}).catch(e => {
  document.body.insertAdjacentHTML('afterbegin',
    '<div style="margin:14px 22px;padding:12px 16px;border-radius:12px;background:#fdeef1;color:#a8324c;font-size:13px">' +
    '初始化失败：' + e.message + '（请确认已运行 python 04_策略选股器/build_ui_data.py）</div>');
  console.error(e);
});

function renderChips() {
  const m = S.boot.meta;
  const u = m.universe_size;
  const avg = Math.round(u.reduce((a, b) => a + b, 0) / u.length);
  $('#topChips').innerHTML = [
    `<div class="chip">初始资金 <b>10 亿元</b></div>`,
    `<div class="chip">佣金 <b>万一</b>（单边）</div>`,
    `<div class="chip">印花税 <b>万五</b>（卖出）</div>`,
    `<div class="chip">区间 <b>2025-01-02 → 2026-09-30</b></div>`,
    `<div class="chip">股票池 <b>非 ST 全市场 · 约 ${avg} 只</b></div>`,
    `<div class="chip">调仓 <b>月度</b>（月末信号 · 次日开盘成交）</div>`,
  ].join('');
}

function renderFamilies() {
  const fs = S.boot.meta.families;
  const counts = {};
  S.boot.factors.forEach(f => counts[f.family] = (counts[f.family] || 0) + 1);
  $('#familyFilter').innerHTML =
    `<button class="fam on" data-fam="all">全部 ${S.boot.factors.length}</button>` +
    FAMS.filter(f => counts[f]).map(f =>
      `<button class="fam" data-fam="${f}">${f} ${counts[f]}</button>`).join('');
  $$('#familyFilter .fam').forEach(b => b.onclick = () => {
    S.fam = b.dataset.fam;
    $$('#familyFilter .fam').forEach(x => x.classList.toggle('on', x === b));
    renderFactorList();
  });
}

/* ---------------------------------------------------------- 因子列表 */
function factorByName(n) { return S.boot.factors.find(f => f.name === n); }

function visibleFactors() {
  const q = S.q.trim().toLowerCase();
  return S.boot.factors.filter(f => {
    if (S.fam !== 'all' && f.family !== S.fam) return false;
    if (!q) return true;
    return (f.name + ' ' + f.module + ' ' + f.desc + ' ' + f.family).toLowerCase().includes(q);
  });
}

function renderFactorList() {
  const list = visibleFactors();
  const host = $('#factorList');
  $('#factorCount').textContent = `（共 ${S.boot.factors.length} 个，当前显示 ${list.length} 个）`;
  host.innerHTML = list.map(f => factorRow(f)).join('') || `<div class="empty-sel" style="border:0">没有匹配的因子</div>`;
  $$('.fitem', host).forEach(el => {
    const nm = el.dataset.f;
    $('[data-role=toggle]', el).onclick = () => toggleFactor(nm);
    $('[data-role=more]', el).onclick = (e) => {
      e.stopPropagation();
      S.expanded.has(nm) ? S.expanded.delete(nm) : S.expanded.add(nm);
      renderFactorList();
    };
  });
  list.forEach(f => {
    const el = $(`.fitem[data-f="${cssEsc(f.name)}"]`, host);
    if (el && S.expanded.has(f.name)) drawSpark($('.spark', el), f.spark);
  });
}
function cssEsc(s) { return s.replace(/["\\]/g, '\\$&'); }

function factorRow(f) {
  const on = S.sel.has(f.name);
  const e = S.expanded.has(f.name);
  const st = f.is_stats || {};
  return `<div class="fitem ${on ? 'on' : ''}" data-f="${f.name}">
    <div class="frow" data-role="toggle">
      <div class="tick">✓</div>
      <div class="fmeta">
        <div class="fline1">
          <span class="fname">${f.name}</span>
          <span class="fam-tag">${f.family}</span>
          ${f.is_rep ? '<span class="fam-tag" title="冗余簇代表因子">代表</span>' : ''}
        </div>
        <div class="fdesc" title="${esc(f.desc)}">${esc(f.desc)}</div>
        <div class="fst">IC <b>${num(f.ic20, 3)}</b> · ICIR <b>${num(f.icir, 2)}</b>
          · 样本内年化 <b>${pct(st.年化收益, 1)}</b> · 夏普 ${num(st.夏普, 2)}</div>
        ${e ? factorDetail(f) : ''}
      </div>
      <div class="fmore" data-role="more">${e ? '收起' : '详情'}</div>
    </div>
  </div>`;
}

function factorDetail(f) {
  const st = f.is_stats || {};
  const dup = [...S.sel.keys()].filter(n => n !== f.name && factorByName(n) && factorByName(n).cluster === f.cluster);
  return `<div class="fdetail" data-role="detail">
    <div class="dgrid">
      <div class="dcell" title="样本内 20 日前视收益的秩 IC 均值"><span>IC20</span><b>${num(f.ic20, 3)}</b></div>
      <div class="dcell" title="样本内 IC 的信息比率（= IC 均值 / IC 标准差）"><span>ICIR</span><b>${num(f.icir, 2)}</b></div>
      <div class="dcell" title="样本内十分层多空组合的净年化（含成本，因子原值口径、与 IC 同号）"><span>多空年化</span><b>${pct(f.ls_net_ann, 1)}</b></div>
      <div class="dcell" title="样本内按该因子打分前 1700 只等权的年化收益"><span>策略年化</span><b>${pct(st.年化收益, 1)}</b></div>
      <div class="dcell" title="样本内策略夏普（年化收益 / 年化波动）"><span>夏普</span><b>${num(st.夏普, 2)}</b></div>
      <div class="dcell" title="样本内策略最大回撤"><span>最大回撤</span><b>${pct(st.最大回撤, 1)}</b></div>
      <div class="dcell" title="股票池内的有效取值覆盖率"><span>覆盖率</span><b>${pct(f.coverage, 0)}</b></div>
      <div class="dcell" title="冗余簇编号：同一簇的因子信息高度重叠"><span>冗余簇</span><b>#${f.cluster}</b></div>
    </div>
    <svg class="spark"></svg>
    <div class="note">上文与曲线：<b>样本内 2021-01 ~ 2024-12</b>，按<b>打分前 1700 只等权</b>月频调仓测算（方向按样本内 IC 符号，同 WebUI 口径）。
      样本外 IC（2025-2026，事后参考、不参与选因子）：<span class="grey">${num(f.oos_ic20, 3)}</span></div>
    ${dup.length ? `<div class="warn">⚠ 与已选因子 <b>${dup.join('、')}</b> 属于同一个冗余簇（#${f.cluster}），信息高度重叠，一起用等于加倍下注同一个 bet。</div>` : ''}
  </div>`;
}

function drawSpark(svg, vals) {
  if (!svg || !vals || vals.length < 2) return;
  const W = svg.clientWidth || 300, H = 52, pad = 4;
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const X = i => pad + i / (vals.length - 1) * (W - 2 * pad);
  const Y = v => H - pad - (v - lo) / (hi - lo || 1) * (H - 2 * pad);
  const line = vals.map((v, i) => `${i ? 'L' : 'M'}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join('');
  const up = vals[vals.length - 1] >= vals[0];
  const c = up ? '#d3405c' : '#1f9d63';
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('height', H);
  svg.innerHTML = `<path d="${line}L${X(vals.length - 1)},${H}L${X(0)},${H}Z" fill="${c}" opacity=".08"></path>
    <path d="${line}" fill="none" stroke="${c}" stroke-width="1.5"></path>`;
}

/* ---------------------------------------------------------- 选择与权重 */
function toggleFactor(nm) {
  S.sel.has(nm) ? S.sel.delete(nm) : S.sel.set(nm, 1);
  renderFactorList();
  renderSelected();
}

function renderSelected() {
  const host = $('#selectedList');
  const names = [...S.sel.keys()];
  $('#weightTitle').textContent = S.mode === 'score' ? '调权重（合成分数）' : '分配资金（仓位）';
  if (!names.length) {
    host.innerHTML = `<div class="empty-sel">还没选因子 —— 上方列表点一下即选中，或用「ICIR 精选」</div>`;
    $('#weightSum').textContent = '';
    return;
  }
  const tot = names.reduce((a, n) => a + S.sel.get(n), 0) || 1;
  host.innerHTML = names.map(n => {
    const f = factorByName(n), w = S.sel.get(n);
    const st = f.is_stats || {};
    return `<div class="selrow" data-n="${n}"
        title="${esc(f.desc)}｜样本内 ICIR ${num(f.icir, 2)} · 年化 ${pct(st.年化收益, 1)} · 夏普 ${num(st.夏普, 2)}">
      <span class="selname">${n}</span>
      <span class="dir ${f.direction < 0 ? 'neg' : ''}">${f.direction < 0 ? '反' : '正'}</span>
      <input type="range" min="1" max="100" step="1" value="${Math.round(w)}" data-r>
      <input type="number" min="0.01" step="0.1" value="${(+w).toFixed(2)}" data-n-input>
      <span class="pct" data-pct>${pct(w / tot, 1)}</span>
      <button class="del" data-del title="移除">×</button>
    </div>`;
  }).join('');
  $$('.selrow', host).forEach(el => {
    const n = el.dataset.n;
    const r = $('[data-r]', el), box = $('[data-n-input]', el);
    const upd = v => {
      S.sel.set(n, Math.max(0.01, v));
      const t = [...S.sel.values()].reduce((a, b) => a + b, 0) || 1;
      $$('.selrow', host).forEach(x => { $('[data-pct]', x).textContent = pct(S.sel.get(x.dataset.n) / t, 1); });
      $('#weightSum').textContent = `权重已自动归一（合计 ${t.toFixed(1)}）`;
    };
    r.oninput = () => { box.value = (+r.value).toFixed(2); upd(+r.value); };
    box.onchange = () => { r.value = Math.max(1, Math.min(100, +box.value)); upd(+box.value); };
    $('[data-del]', el).onclick = () => toggleFactor(n);
  });
  $('#weightSum').textContent = `权重已自动归一（合计 ${tot.toFixed(1)}）`;
}

function applyPresetIcir(silent) {
  const pool = S.boot.factors
    .filter(f => f.is_rep && Math.abs(f.ic20 || 0) >= 0.02 && Math.abs(f.icir || 0) >= 0.5)
    .sort((a, b) => Math.abs(b.icir) - Math.abs(a.icir));
  const famCnt = {}, pick = [];
  for (const f of pool) {
    if ((famCnt[f.family] || 0) >= 2) continue;
    famCnt[f.family] = (famCnt[f.family] || 0) + 1;
    pick.push(f);
    if (pick.length >= 8) break;
  }
  S.sel.clear();
  pick.forEach(f => S.sel.set(f.name, Math.max(0.05, Math.abs(f.icir))));
  renderFactorList(); renderSelected();
}

/* ---------------------------------------------------------- 运行 */
function bindTop() {
  $$('#modePicker .mode-card').forEach(b => b.onclick = () => setMode(b.dataset.mode));
  setMode(S.mode);

  $('#search').oninput = e => { S.q = e.target.value; renderFactorList(); };
  $('#nHold').oninput = e => {
    S.nHold = +e.target.value;
    $('#nHoldLabel').textContent = S.nHold;
    updNHoldHint();
    $$('#nPresets button').forEach(x => x.classList.toggle('on', +x.dataset.n === S.nHold));
  };
  $$('#nPresets button').forEach(b => b.onclick = () => {
    S.nHold = +b.dataset.n; $('#nHold').value = S.nHold; $('#nHoldLabel').textContent = S.nHold;
    $$('#nPresets button').forEach(x => x.classList.toggle('on', x === b));
    updNHoldHint();
  });
  $('#buffer').oninput = e => { S.buffer = +e.target.value; $('#bufferLabel').textContent = S.buffer.toFixed(1) + '×'; };
  $('#run').onclick = run;
  $('#tryDefault').onclick = run;
  $('#presetIcir').onclick = () => applyPresetIcir();
  $('#presetReps').onclick = () => {
    S.sel.clear();
    S.boot.factors.filter(f => f.is_rep && Math.abs(f.ic20 || 0) >= 0.02 && Math.abs(f.icir || 0) >= 0.5)
      .slice(0, 20).forEach(f => S.sel.set(f.name, 1));
    renderFactorList(); renderSelected();
  };
  $('#clearSel').onclick = () => { S.sel.clear(); renderFactorList(); renderSelected(); };
  $('#wEqual').onclick = () => { [...S.sel.keys()].forEach(n => S.sel.set(n, 1)); renderSelected(); };
  $('#wIcir').onclick = () => { [...S.sel.keys()].forEach(n => S.sel.set(n, Math.max(.05, Math.abs(factorByName(n).icir || 1)))); renderSelected(); };
  $('#wInvIcir').onclick = () => { [...S.sel.keys()].forEach(n => S.sel.set(n, 1 / Math.max(.05, Math.abs(factorByName(n).icir || 1)))); renderSelected(); };
  updNHoldHint();
  window.addEventListener('resize', () => { if (S.result) renderResults(false); });
}

function updNHoldHint() {
  const u = S.boot.meta.universe_size;
  const avg = u.reduce((a, b) => a + b, 0) / u.length;
  $('#nHoldHint').textContent = `≈ 全市场（约 ${Math.round(avg)} 只）里分数最高的 ${(100 * S.nHold / avg).toFixed(0)}%，等权持有`;
}

function setMode(m) {
  S.mode = m;
  $$('#modePicker .mode-card').forEach(b => b.classList.toggle('on', b.dataset.mode === m));
  const t = $('#weightTitle');
  if (t) renderSelected();
}

async function run() {
  if (S.running) return;
  if (!S.sel.size) { $('#runErr').textContent = '请至少选择一个因子'; return; }
  if (S.sel.size > S.boot.meta.max_factors) { $('#runErr').textContent = `一次最多选 ${S.boot.meta.max_factors} 个因子`; return; }
  $('#runErr').textContent = '';
  S.running = true;
  const btn = $('#run');
  btn.disabled = true;
  $('.run-label', btn).textContent = '回测中…';
  $('.run-time', btn).innerHTML = '<span class="spinner"></span>';
  const t0 = performance.now();
  try {
    const res = await fetch('/api/run', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        mode: S.mode, n_hold: S.nHold, buffer: S.buffer,
        factors: [...S.sel.entries()].map(([name, weight]) => ({ name, weight })),
      }),
    });
    const d = await res.json();
    if (!d.ok) throw new Error(d.error || '计算失败');
    S.result = d;
    S.off.clear();
    renderResults(true);
    const ms = Math.round(performance.now() - t0);
    $('.run-time', btn).textContent = `${ms} ms`;
    $('#results').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (e) {
    $('#runErr').textContent = e.message;
    $('.run-time', btn).textContent = '';
  } finally {
    S.running = false; btn.disabled = false;
    $('.run-label', btn).textContent = '开始回测';
  }
}

/* ---------------------------------------------------------- 结果渲染 */
function statOf(key) { return (S.result.stats || []).find(s => s.key === key) || {}; }
function curveOf(key) { return (S.result.curves || []).find(c => c.key === key) || {}; }

function renderResults(rebuild) {
  const d = S.result;
  $('#placeholder').classList.add('hidden');
  const body = $('#resultBody');
  body.classList.remove('hidden');
  const mine = statOf('mine'), ew = statOf('market_ew');
  const ms = mine.年化 > 0 ? 'success' : '';

  const kpis = [
    ['期末资产', yi(d.final_asset), `净值 ${num(d.final_nav, 4)}（起点 10 亿）`, 'hi'],
    ['累计收益', sgn(mine.累计收益), `全市场等权 ${sgn(ew.累计收益)}`, ''],
    ['年化收益', pct(mine.年化收益), `超额 ${sgn(mine.超额年化)} / 年`, 'hi'],
    ['夏普', num(mine.夏普, 2), `全市场等权 ${num(ew.夏普, 2)}`, ''],
    ['最大回撤', pct(mine.最大回撤), `全市场等权 ${pct(ew.最大回撤)}`, ''],
    ['年化换手', mine.年换手 ? num(mine.年换手, 1) + '×' : '—', '单边成交额 / 净值', ''],
  ];

  body.innerHTML = `
    <div class="kpis">${kpis.map(k => `<div class="kpi ${k[3]}"><span>${k[0]}</span><b>${k[1]}</b><small>${k[2]}</small></div>`).join('')}</div>

    <div class="card">
      <div class="card-head">
        <div>
          <h3>净值对比</h3>
          <p>${S.mode === 'score' ? '打分合成：一批股票、一份资金（分数前 N 等权）' : '资金分配：各因子独立建仓，资金按权重分账户'} ·
             起点 2024-12-31 = 1.0</p>
        </div>
        <div class="toggles">
          <button class="mini" id="logBtn">${S.log ? '线性坐标' : '对数坐标'}</button>
        </div>
      </div>
      <div class="legend" id="legend"></div>
      <div class="chartwrap"><svg class="chart" id="mainChart"></svg></div>
    </div>

    <div class="grid3">
      <div class="card">
        <div class="card-head"><div><h3>分年度收益</h3><p>2025 / 2026（截至 9-30）</p></div></div>
        <svg class="subchart" id="annualChart"></svg>
      </div>
      <div class="card">
        <div class="card-head"><div><h3>月度收益</h3><p>我的策略 · 红涨绿跌</p></div></div>
        <div id="heat"></div>
      </div>
      <div class="card">
        <div class="card-head"><div><h3>回撤</h3><p>我的策略 vs 全市场等权</p></div></div>
        <svg class="subchart" id="ddChart"></svg>
      </div>
    </div>

    <div class="card">
      <div class="card-head"><div><h3>横向对比</h3><p>同一区间、同一资金、同一费用口径</p></div></div>
      <div class="scrollx">${compareTable()}</div>
    </div>

    <div class="card">
      <div class="tabs" id="tabs">
        <button class="tab on" data-t="hold">组合持仓</button>
        <button class="tab" data-t="fac">因子贡献</button>
        <button class="tab" data-t="plan">调仓记录</button>
        <button class="tab" data-t="doc">口径与说明</button>
      </div>
      <div class="tabpane on" id="pane-hold"></div>
      <div class="tabpane" id="pane-fac"></div>
      <div class="tabpane" id="pane-plan"></div>
      <div class="tabpane doc" id="pane-doc"></div>
    </div>`;

  renderLegend();
  drawMainChart();
  drawAnnualChart();
  drawHeat();
  drawDdChart();
  renderPanes();
  $('#logBtn').onclick = () => { S.log = !S.log; renderResults(false); };
  $$('#tabs .tab').forEach(b => b.onclick = () => {
    $$('#tabs .tab').forEach(x => x.classList.toggle('on', x === b));
    $$('.tabpane', body).forEach(p => p.classList.toggle('on', p.id === 'pane-' + b.dataset.t));
  });
}

function renderLegend() {
  $('#legend').innerHTML = S.result.curves.map(c => {
    const off = S.off.has(c.key) ? 'off' : '';
    const dash = DASH[c.key] ? 'dash' : '';
    return `<div class="lg ${off}" data-k="${c.key}">
      <i class="${dash}" style="background:${COLOR[c.key] || '#888'};color:${COLOR[c.key] || '#888'}"></i>${c.label}</div>`;
  }).join('');
  $$('#legend .lg').forEach(el => el.onclick = () => {
    const k = el.dataset.k;
    S.off.has(k) ? S.off.delete(k) : S.off.add(k);
    renderLegend(); drawMainChart();
  });
}

function compareTable() {
  const rows = S.result.stats;
  const head = ['组合', '累计收益', '年化', '波动', '夏普', '最大回撤', '卡玛', '超额年化', '信息比率', '年换手'];
  const body = rows.map(s => `<tr class="${s.key === 'mine' ? 'mine' : ''}">
      <td>${s.label}${s.key === 'mine' ? '<span class="pill">我的策略</span>' : ''}</td>
      <td class="${clsP(s.累计收益)}">${sgn(s.累计收益)}</td>
      <td class="${clsP(s.年化收益)}">${pct(s.年化收益)}</td>
      <td>${pct(s.年化波动)}</td>
      <td>${num(s.夏普)}</td>
      <td class="neg">${pct(s.最大回撤)}</td>
      <td>${num(s.卡玛)}</td>
      <td class="${clsP(s.超额年化)}">${s.key === 'market_ew' ? '—' : sgn(s.超额年化)}</td>
      <td>${s.key === 'market_ew' ? '—' : num(s.信息比率)}</td>
      <td>${s.年换手 ? num(s.年换手, 1) + '×' : '—'}</td>
    </tr>`).join('');
  return `<table><thead><tr>${head.map(h => `<th>${h}</th>`).join('')}</tr></thead><tbody>${body}</tbody></table>`;
}

/* ---------------------------------------------------------- 图表 */
function niceTicks(lo, hi, k = 5) {
  const span = hi - lo || 1;
  const raw = span / (k - 1);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw) || mag * 10;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

function drawMainChart() {
  const host = $('#mainChart');
  const W = host.clientWidth || 900, H = 360;
  const pad = { l: 56, r: 16, t: 12, b: 26 };
  const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;
  const dates = S.result.dates;
  const n = dates.length;
  const vis = S.result.curves.filter(c => !S.off.has(c.key));
  const trans = v => S.log ? Math.log(Math.max(v, 1e-6)) : v;
  let lo = Infinity, hi = -Infinity;
  vis.forEach(c => c.nav.forEach(v => { const t = trans(v); if (t < lo) lo = t; if (t > hi) hi = t; }));
  if (!isFinite(lo)) { lo = 0.9; hi = 1.1; }
  const padv = (hi - lo) * 0.07 || 0.02;
  lo -= padv; hi += padv;
  const X = i => pad.l + (n <= 1 ? 0 : i / (n - 1) * iw);
  const Y = v => pad.t + ih - (trans(v) - lo) / (hi - lo) * ih;

  let g = '';
  niceTicks(lo, hi, 5).forEach(v => {
    const y = Y(S.log ? Math.exp(v) : v);
    if (y < pad.t - 1 || y > pad.t + ih + 1) return;
    const label = ((S.log ? Math.exp(v) : v) - 1) * 100;
    g += `<line class="gridline" x1="${pad.l}" x2="${W - pad.r}" y1="${y.toFixed(1)}" y2="${y.toFixed(1)}"></line>
          <text x="${pad.l - 8}" y="${(y + 3.5).toFixed(1)}" text-anchor="end">${label.toFixed(label > 99 ? 0 : 1)}%</text>`;
  });
  const tickIdx = [];
  dates.forEach((d, i) => {
    const mm = d.slice(5, 7);
    const firstOfMonth = i === 0 || dates[i - 1].slice(0, 7) !== d.slice(0, 7);
    if (firstOfMonth && (i === 0 || mm === '01' || mm === '04' || mm === '07' || mm === '10')) tickIdx.push(i);
  });
  tickIdx.forEach(i => {
    const x = X(i);
    g += `<line class="axisline" x1="${x.toFixed(1)}" x2="${x.toFixed(1)}" y1="${pad.t + ih}" y2="${pad.t + ih + 4}"></line>
          <text x="${x.toFixed(1)}" y="${H - 6}" text-anchor="middle">${dates[i].slice(0, 7)}</text>`;
  });
  g += `<line class="axisline" x1="${pad.l}" x2="${W - pad.r}" y1="${pad.t + ih}" y2="${pad.t + ih}"></line>`;

  vis.forEach(c => {
    const col = COLOR[c.key] || '#888';
    const dash = DASH[c.key];
    const d = c.nav.map((v, i) => `${i ? 'L' : 'M'}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join('');
    g += `<path d="${d}" fill="none" stroke="${col}" stroke-width="${c.key === 'mine' ? 2.6 : 1.7}"
            ${dash ? `stroke-dasharray="${dash}"` : ''} stroke-linejoin="round"></path>`;
  });
  g += `<line class="hoverline" id="hline" x1="0" x2="0" y1="${pad.t}" y2="${pad.t + ih}" style="display:none"></line>
        <g id="hdots"></g>
        <rect id="hit" x="${pad.l}" y="${pad.t}" width="${iw}" height="${ih}" fill="transparent"></rect>`;
  host.setAttribute('viewBox', `0 0 ${W} ${H}`);
  host.setAttribute('height', H);
  host.innerHTML = g;

  const hit = $('#hit', host), hline = $('#hline', host), dots = $('#hdots', host);
  hit.onmousemove = ev => {
    const rect = host.getBoundingClientRect();
    const px = (ev.clientX - rect.left) / rect.width * W;
    let i = Math.round((px - pad.l) / (iw || 1) * (n - 1));
    i = Math.max(0, Math.min(n - 1, i));
    hline.style.display = '';
    hline.setAttribute('x1', X(i)); hline.setAttribute('x2', X(i));
    dots.innerHTML = vis.map(c => `<circle cx="${X(i).toFixed(1)}" cy="${Y(c.nav[i]).toFixed(1)}" r="3.2"
        fill="#fff" stroke="${COLOR[c.key] || '#888'}" stroke-width="2"></circle>`).join('');
    showTip(ev, `<div class="tt-date">${dates[i]}</div>` + vis.slice().sort((a, b) => b.nav[i] - a.nav[i])
      .map(c => `<div class="tt-row"><span><i style="background:${COLOR[c.key] || '#888'}"></i>${shortLabel(c)}</span>
        <b>${num(c.nav[i], 4)}<span style="color:#9aa0b5"> ${sgn(c.nav[i] - 1, 1)}</span></b></div>`).join(''));
  };
  hit.onmouseleave = () => { hline.style.display = 'none'; dots.innerHTML = ''; hideTip(); };
}

function shortLabel(c) {
  return { mine: '我的策略', factor_avg: '所选因子等权', market_ew: '全市场等权', mf1700: '最佳线性打分', ml1700: '最佳树模型' }[c.key] || c.label;
}

function showTip(ev, html) {
  const t = $('#tooltip');
  t.innerHTML = html;
  t.style.opacity = 1;
  const x = Math.min(ev.clientX + 14, window.innerWidth - t.offsetWidth - 12);
  const y = Math.min(ev.clientY + 14, window.innerHeight - t.offsetHeight - 12);
  t.style.left = x + 'px'; t.style.top = y + 'px';
}
function hideTip() { $('#tooltip').style.opacity = 0; }

function drawAnnualChart() {
  const host = $('#annualChart');
  const W = host.clientWidth || 320, H = 132;
  const years = Object.keys(S.result.annual.mine || {}).filter(y => (+y) >= 2025);
  const keys = ['mine', 'market_ew'];
  const pad = { l: 34, r: 8, t: 8, b: 18 };
  const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;
  let mx = 0.02;
  years.forEach(y => keys.forEach(k => { const v = (S.result.annual[k] || {})[y]; if (v != null) mx = Math.max(mx, Math.abs(v)); }));
  const Y = v => pad.t + ih / 2 - v / mx * (ih / 2);
  let g = `<line class="axisline" x1="${pad.l}" x2="${W - pad.r}" y1="${Y(0)}" y2="${Y(0)}"></line>`;
  years.forEach((y, i) => {
    const bw = iw / years.length;
    keys.forEach((k, j) => {
      const v = (S.result.annual[k] || {})[y]; if (v == null) return;
      const x = pad.l + i * bw + bw * (0.18 + j * 0.34), w = bw * 0.3;
      const y0 = Y(0), y1 = Y(v);
      g += `<rect x="${x.toFixed(1)}" y="${Math.min(y0, y1).toFixed(1)}" width="${w.toFixed(1)}"
             height="${Math.abs(y1 - y0).toFixed(1)}" rx="3" fill="${COLOR[k]}" opacity="${k === 'mine' ? .9 : .45}"></rect>
            <text x="${(x + w / 2).toFixed(1)}" y="${(v >= 0 ? y1 - 3 : y1 + 11).toFixed(1)}" text-anchor="middle"
             style="font-size:10px;fill:${k === 'mine' ? '#171a26' : '#8b93a7'}">${(v * 100).toFixed(1)}%</text>`;
    });
    g += `<text x="${(pad.l + i * bw + bw / 2).toFixed(1)}" y="${H - 5}" text-anchor="middle">${y}</text>`;
  });
  g += `<text x="${pad.l - 6}" y="${Y(mx) + 9}" text-anchor="end" style="font-size:10px">${(mx * 100).toFixed(0)}%</text>
        <text x="${pad.l - 6}" y="${Y(-mx)}" text-anchor="end" style="font-size:10px">-${(mx * 100).toFixed(0)}%</text>`;
  host.setAttribute('viewBox', `0 0 ${W} ${H}`); host.setAttribute('height', H); host.innerHTML = g;
}

function drawHeat() {
  const m = S.result.monthly.mine || {};
  const months = Object.keys(m).sort();
  const years = [...new Set(months.map(x => x.slice(0, 4)))];
  const mx = Math.max(0.02, ...Object.values(m).map(v => Math.abs(v)));
  const cell = v => {
    if (v == null) return `<div class="hcell" style="background:#f7f8fc"></div>`;
    const a = Math.min(1, Math.abs(v) / mx);
    const c = v >= 0 ? `rgba(211,64,92,${0.10 + a * 0.75})` : `rgba(31,157,99,${0.10 + a * 0.75})`;
    const fg = a > 0.45 ? '#fff' : '#5c6478';
    return `<div class="hcell" style="background:${c};color:${fg}" title="${(v * 100).toFixed(2)}%">${(v * 100).toFixed(1)}</div>`;
  };
  let html = `<div class="heat" style="grid-template-columns:38px repeat(12,1fr)">
    <div class="hrow"></div>${Array.from({ length: 12 }, (_, i) => `<div class="hrow" style="text-align:center;white-space:nowrap">${i + 1}</div>`).join('')}`;
  years.forEach(y => {
    html += `<div class="hrow">${y}</div>`;
    for (let mo = 1; mo <= 12; mo++) html += cell(m[`${y}-${String(mo).padStart(2, '0')}`]);
  });
  html += `</div><div class="cols-legend"><i style="background:rgba(31,157,99,.75)"></i>跌
    <i style="background:#f7f8fc;border:1px solid #e7e9f2"></i>无交易
    <i style="background:rgba(211,64,92,.75)"></i>涨 &nbsp; 深浅 = 幅度（最大 ${(mx * 100).toFixed(1)}%）</div>`;
  $('#heat').innerHTML = html;
}

function drawDdChart() {
  const host = $('#ddChart');
  const W = host.clientWidth || 320, H = 132, pad = { l: 36, r: 8, t: 8, b: 18 };
  const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;
  const dd = S.result.drawdown;
  const n = (dd.mine || []).length;
  let mdd = -0.02;
  ['mine', 'market_ew'].forEach(k => (dd[k] || []).forEach(v => { mdd = Math.min(mdd, v); }));
  const X = i => pad.l + i / (n - 1) * iw, Y = v => pad.t + (-v / -mdd) * (ih - 6);
  let g = `<line class="gridline" x1="${pad.l}" x2="${W - pad.r}" y1="${pad.t}" y2="${pad.t}"></line>`;
  ['market_ew', 'mine'].forEach(k => {
    if (!dd[k]) return;
    const d = dd[k].map((v, i) => `${i ? 'L' : 'M'}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join('');
    g += `<path d="${d}L${X(n - 1)},${pad.t}L${pad.l},${pad.t}Z" fill="${COLOR[k]}" opacity=".08"></path>
          <path d="${d}" fill="none" stroke="${COLOR[k]}" stroke-width="${k === 'mine' ? 1.9 : 1.3}"
            ${k === 'market_ew' ? 'stroke-dasharray="4 3"' : ''}></path>`;
  });
  g += `<text x="${pad.l - 6}" y="${pad.t + 9}" text-anchor="end" style="font-size:10px">0%</text>
        <text x="${pad.l - 6}" y="${Y(mdd)}" text-anchor="end" style="font-size:10px">${(mdd * 100).toFixed(0)}%</text>
        <text x="${(pad.l + iw / 2).toFixed(0)}" y="${H - 4}" text-anchor="middle" style="font-size:10px">
        我的最大回撤 ${(Math.min(...(dd.mine || [0])) * 100).toFixed(1)}% · 等权 ${(Math.min(...(dd.market_ew || [0])) * 100).toFixed(1)}%</text>`;
  host.setAttribute('viewBox', `0 0 ${W} ${H}`); host.setAttribute('height', H); host.innerHTML = g;
}

/* ---------------------------------------------------------- 标签页 */
function renderPanes() {
  const d = S.result;
  $('#pane-hold').innerHTML = `
    <p class="legend-note">${esc(d.holdings_note || '')} · 全部持仓等权，
      单票权重 ≈ ${(d.gross_limit / d.n_hold * 100).toFixed(3)}%（总仓位 ${(d.gross_limit * 100).toFixed(0)}%，留 2% 现金缓冲）</p>
    <div class="scrollx"><table><thead><tr><th>证券代码</th><th>合成分数</th><th>目标权重</th></tr></thead>
    <tbody>${(d.holdings || []).map(h => `<tr><td>${h.symbol}</td><td>${num(h.score, 3)}</td>
      <td>${(h.weight * 100).toFixed(3)}%</td></tr>`).join('')}</tbody></table></div>`;
  $('#pane-fac').innerHTML = `<p class="legend-note">每个因子<b>单独</b>按同样口径跑一遍（前 ${d.n_hold} 只等权），
      ${d.mode === 'allocate' ? '它们按你给的权重合成就是「我的策略」' : '把它们等权组合就是图中的「所选因子等权」'}。</p>
    <div class="scrollx"><table><thead><tr><th>因子</th><th>家族</th><th>${d.mode === 'allocate' ? '资金权重' : '打分权重'}</th>
      <th>年化</th><th>夏普</th><th>最大回撤</th><th>超额年化</th><th>年换手</th></tr></thead>
    <tbody>${(d.factors || []).map(f => `<tr><td>${f.name}</td><td>${f.family}</td><td>${(f.weight * 100).toFixed(1)}%</td>
      <td class="${clsP(f.年化)}">${pct(f.年化)}</td><td>${num(f.夏普)}</td><td class="neg">${pct(f.最大回撤)}</td>
      <td class="${clsP(f.超额年化)}">${sgn(f.超额年化)}</td><td>${num(f.年换手, 1)}×</td></tr>`).join('')}</tbody></table></div>`;
  $('#pane-plan').innerHTML = `<p class="legend-note">${esc(d.plan_src || '')} · 调仓 22 次：21 次月度打分 + 1 次末日清仓。
      因子在某一期横截面上完全没有差异时（例如日历因子在非事件月恒为常数），该期<b>不调仓</b>，避免按代码顺序乱选。</p>
    <div class="scrollx"><table><thead><tr><th>信号日（收盘）</th><th>执行日（开盘）</th><th>动作</th></tr></thead>
    <tbody>${(d.plan || []).map(p => `<tr><td>${p.signal}</td><td>${p.exec}</td><td>
      ${p.note ? p.note : (p.n_hold ? '调仓至 ' + p.n_hold + ' 只' : '清仓')}</td></tr>`).join('')}</tbody></table></div>`;
  $('#pane-doc').innerHTML = docHtml();
}

function docHtml() {
  const m = S.boot.meta;
  return `
  <h4>一、用户指定口径</h4>
  <ul>
    <li>初始资金 <code>10 亿元</code>；佣金 <code>万一</code>（单边，已含过户费的简化处理）；印花税 <code>万五</code>（仅卖出）。</li>
    <li>回测区间：<code>2025-01-02</code>（当日开盘建仓）→ <code>2026-09-30</code>（当日开盘清仓），共 ${m.trade_dates.length} 个交易日、21 次月度调仓。</li>
    <li>股票池：非 ST 全市场动态池（上市满 250 交易日、20 日均成交额 ≥ 2000 万、股价 ≥ 2 元、剔除一字板 / 停牌 / 无涨跌幅限制标的），
        每期约 ${Math.min(...m.universe_size)} ~ ${Math.max(...m.universe_size)} 只。</li>
    <li>信号在<b>每月最后一个交易日收盘</b>产生，<b>次日开盘</b>按复权价成交（复权份额记账，分红自动留在净值里）。</li>
  </ul>
  <h4>二、两种玩法怎么算</h4>
  <ul>
    <li><b>打分合成</b>：每个因子先做当日横截面<b>秩 z</b>（抗极值），方向取样本内 IC 符号，按你的权重合成总分 → 买分数前 N 只等权（目标总仓位 98%，留 2% 现金缓冲）。</li>
    <li><b>资金分配</b>：每个因子各自选前 N 只等权做成一个子策略，资金按权重分给各子策略，组合净值 = Σ wᵢ × 净值ᵢ（分账户执行，权重随净值自然漂移，不再平衡）。</li>
  </ul>
  <h4>三、图上四条对照</h4>
  <ul>
    <li><b>全市场等权</b>：池内所有股票日频等权，项目 02/03 报告的同一基准。</li>
    <li><b>所选因子等权</b>：把你选的每个因子各跑一个子策略、资金等分——用来回答「加权比等权强吗」。</li>
    <li><b>最佳线性打分</b>：项目 s23 的 165 因子 ICIR 两级加权模型（引擎口径 22.51%/年、夏普 1.21）。</li>
    <li><b>最佳打分模型</b>：项目 s26 的树模型打分（样本内 2024-11 前训练，引擎口径 29.38%/年、夏普 1.34）。</li>
  </ul>
  <h4>四、快速口径 vs 逐笔撮合引擎</h4>
  <p>WebUI 用向量化口径复刻引擎的记账规则（复权份额、次日开盘成交、98% 目标仓位、调仓死区、先卖后买、指定费用），
     但<b>不模拟</b> T+1、涨跌停/停牌挡单、参与率上限与执行窗口内逐日补单——实测对拍（<code>verify_vs_engine.py</code>）
     日收益相关 &gt; 0.999、年化偏高 0.13 ~ 0.31pp（挡单摩擦的成本）。要逐笔落库请直接跑 <code>03_因子挖掘/s23_multifactor_bt.py</code>。</p>
  <h4>五、必须知道的坑</h4>
  <ul>
    <li>因子的方向与权重<b>只能用样本内（2021-01 ~ 2024-12）</b>的信息，样本外（2025 起）不参与任何选择；因子卡片里的「样本内表现」同样存在样本内选择偏差。</li>
    <li>样本外只有 21 个月，年化差异的标准误约 ±4pp，随机对照（同池同 N 选 10 组）的年化区间就有 [19.5%, 22.7%]，别把 1~2pp 的差异当规律。</li>
    <li>宽持仓（N≈1700）是项目结论：多头端 α 主要来自「避开最差的一档」，集中 top-100 在 10 亿资金上会被参与率上限吃掉大半。</li>
    <li>本页是课程作业的教学回测，不构成任何投资建议。</li>
  </ul>`;
}

function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}