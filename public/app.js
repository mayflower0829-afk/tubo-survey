/* 乡橙调查 · TuboSurvey — 市场调查数据交互分析
   统计函数（卡方/Spearman/Cronbach's α）为前端手写实现，固定种子数据可复现。 */

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => (
  { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
));

const VARS = {
  age: { label: '年龄组', vals: ['18-24岁', '25-34岁', '35-44岁', '45岁及以上'] },
  channel: { label: '招募渠道', vals: ['线上', '线下'] },
  use: { label: '购买用途', vals: ['日常自用', '送礼馈赠', '家庭聚会', '其他'] },
  attr: { label: '首要关注点', vals: ['口感风味', '新鲜度', '价格', '品牌与产地', '包装设计'] },
  pack: { label: '包装偏好', vals: ['简易防护包装', '礼盒装', '高端定制装'] },
  willing: { label: '99元方案购买意愿', vals: ['不愿意', '愿意'] },
};
const D = window.SURVEY.records;

function priceBand(p) {
  if (p < 80) return '<80元';
  if (p < 100) return '80–99元';
  if (p < 130) return '100–129元';
  return '≥130元';
}
const PRICE_BANDS = ['<80元', '80–99元', '100–129元', '≥130元'];
const VARS_ALL = Object.assign({}, VARS, {
  price_band: { label: '最高接受总价', vals: PRICE_BANDS },
});

// ================= 手写统计函数 =================
function gammaln(x) {  // Lanczos
  const g = [676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012,
    9.9843695780195716e-6, 1.5056327351493116e-7];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - gammaln(1 - x);
  x -= 1;
  let a = 0.99999999999980993;
  const t = x + 7.5;
  for (let i = 0; i < 8; i++) a += g[i] / (x + i + 1);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}
function gammaincSeries(s, x) {  // P(s,x)，x < s+1
  let sum = 1 / s, term = sum;
  for (let n = 1; n < 500; n++) {
    term *= x / (s + n);
    sum += term;
    if (term < sum * 1e-12) break;
  }
  return sum * Math.exp(-x + s * Math.log(x) - gammaln(s));
}
function gammaincCF(s, x) {  // Q(s,x)，x >= s+1（连分式）
  const FPMIN = 1e-300;
  let b = x + 1 - s, c = 1 / FPMIN, d = 1 / b, h = d;
  for (let i = 1; i < 500; i++) {
    const an = -i * (i - s);
    b += 2;
    d = an * d + b; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = b + an / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-12) break;
  }
  return Math.exp(-x + s * Math.log(x) - gammaln(s)) * h;
}
function chi2Sf(x, k) {  // 卡方上侧概率 P(X > x)
  if (x <= 0) return 1;
  const s = k / 2, xx = x / 2;
  return xx < s + 1 ? 1 - gammaincSeries(s, xx) : gammaincCF(s, xx);
}
function chi2Test(table) {  // r×c 列联表卡方独立性检验
  const r = table.length, c = table[0].length;
  const rowSum = table.map((row) => row.reduce((a, b) => a + b, 0));
  const colSum = table[0].map((_, j) => table.reduce((a, row) => a + row[j], 0));
  const n = rowSum.reduce((a, b) => a + b, 0);
  let chi2 = 0, lowExp = 0, cells = 0;
  for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) {
    const e = rowSum[i] * colSum[j] / n;
    if (e > 0) chi2 += (table[i][j] - e) ** 2 / e;
    if (e < 5) lowExp += 1;
    cells += 1;
  }
  const df = (r - 1) * (c - 1);
  const v = df > 0 && n > 0 ? Math.sqrt(chi2 / (n * Math.min(r - 1, c - 1))) : 0;
  return { chi2, df, p: chi2Sf(chi2, df), v, n, lowExpPct: cells ? lowExp / cells * 100 : 0 };
}
function rankAvg(arr) {  // 平均秩（并列取平均）
  const idx = arr.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
  const ranks = new Array(arr.length);
  let i = 0;
  while (i < idx.length) {
    let j = i;
    while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j += 1;
    const r = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) ranks[idx[k][1]] = r;
    i = j + 1;
  }
  return ranks;
}
function pearson(xs, ys) {
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b, 0) / n, my = ys.reduce((a, b) => a + b, 0) / n;
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    sxy += (xs[i] - mx) * (ys[i] - my);
    sxx += (xs[i] - mx) ** 2;
    syy += (ys[i] - my) ** 2;
  }
  return sxx && syy ? sxy / Math.sqrt(sxx * syy) : 0;
}
function spearman(xs, ys) {  // t 近似 p 值（n=400 下与精确值差异可忽略）
  const rx = rankAvg(xs), ry = rankAvg(ys);
  const rho = pearson(rx, ry);
  const n = xs.length;
  const t = rho * Math.sqrt((n - 2) / Math.max(1 - rho * rho, 1e-12));
  const x = n - 2 + 2, a = (n - 2) / 2, b = 0.5;
  const pTwo = t == 0 ? 1 : betacfIbeta(a, b, n / (n + t * t));  // 双侧
  return { rho, p: Math.min(1, Math.max(pTwo, 0)) };
}
function betacf(a, b, x) {  // 正则化不完全贝塔的连分式
  const FPMIN = 1e-300;
  const qab = a + b, qap = a + 1, qam = a - 1;
  let c = 1, d = 1 - qab * x / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d; let h = d;
  for (let m = 1; m <= 300; m++) {
    const m2 = 2 * m;
    let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d; h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-12) break;
  }
  return h;
}
function betacfIbeta(a, b, x) {  // 正则化 I_x(a,b)
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const lbeta = gammaln(a + b) - gammaln(a) - gammaln(b);
  const bt = Math.exp(lbeta + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
}
function cronbach(cols) {  // cols: 数组的数组（每人每题得分）
  const k = cols.length, n = cols[0].length;
  const itemVar = cols.map((c) => {
    const m = c.reduce((a, b) => a + b, 0) / n;
    return c.reduce((s, x) => s + (x - m) ** 2, 0) / (n - 1);
  });
  const tot = [];
  for (let i = 0; i < n; i++) tot.push(cols.reduce((s, c) => s + c[i], 0));
  const tm = tot.reduce((a, b) => a + b, 0) / n;
  const totVar = tot.reduce((s, x) => s + (x - tm) ** 2, 0) / (n - 1);
  return k / (k - 1) * (1 - itemVar.reduce((a, b) => a + b, 0) / totVar);
}

// ================= 视图 =================
let charts = {};
function ensureChart(key, el) {
  if (!charts[key]) charts[key] = echarts.init(el);
  return charts[key];
}
function mean(arr) { return arr.reduce((a, b) => a + b, 0) / arr.length; }
function col(field) { return D.map((r) => r[field]); }

function renderOverview() {
  $('kpiRow').innerHTML = `
    <div class="kpi"><b>${D.length}</b><span>模拟问卷样本</span><em>4 组 × 2 渠道 × 50</em></div>
    <div class="kpi"><b>${SURVEY.willing_rate}%</b><span>99 元方案购买意愿</span><em>报告校准口径</em></div>
    <div class="kpi"><b>¥${SURVEY.price_median}</b><span>5 斤最高接受总价中位数</span><em>报告口径 95 元</em></div>
    <div class="kpi"><b>${(Object.values(SURVEY.alpha).reduce((a, b) => a + b, 0) / 4).toFixed(2)}</b><span>量表 α 均值</span><em>均 &gt; 0.7 可接受</em></div>`;
  // 配额结构
  const quota = VARS.age.vals.map((a) => ({
    name: a,
    线上: D.filter((r) => r.age === a && r.channel === '线上').length,
    线下: D.filter((r) => r.age === a && r.channel === '线下').length,
  }));
  ensureChart('quota', $('chartQuota')).setOption({
    tooltip: { trigger: 'axis' },
    legend: { top: 0, textStyle: { fontSize: 11 } },
    grid: { left: 40, right: 12, top: 30, bottom: 26 },
    xAxis: { type: 'category', data: quota.map((q) => q.name), axisLabel: { fontSize: 11 } },
    yAxis: { type: 'value', axisLabel: { fontSize: 10 } },
    series: [
      { name: '线上', type: 'bar', data: quota.map((q) => q.线上), itemStyle: { color: '#D97706' }, barMaxWidth: 30 },
      { name: '线下', type: 'bar', data: quota.map((q) => q.线下), itemStyle: { color: '#9AA5B1' }, barMaxWidth: 30 },
    ],
  });
  // 用途
  const useCnt = VARS.use.vals.map((v) => ({ name: v, value: D.filter((r) => r.use === v).length }));
  ensureChart('use', $('chartUse')).setOption({
    tooltip: { trigger: 'item', formatter: '{b}: {c} 人（{d}%）' },
    legend: { type: 'scroll', top: 0, textStyle: { fontSize: 11 } },
    series: [{ type: 'pie', radius: ['38%', '66%'], data: useCnt, label: { fontSize: 11, formatter: '{b} {d}%' } }],
  });
}

// ---- 频数分析 ----
function freqData(field) {
  const def = VARS_ALL[field];
  const get = (r) => field === 'price_band' ? priceBand(r.price) : String(r[field]);
  const counts = def.vals.map((v) => D.filter((r) => get(r) === v).length);
  return { def, counts, get };
}
function renderFreq() {
  const field = $('freqVar').value;
  const { def, counts } = freqData(field);
  const total = D.length;
  ensureChart('freq', $('chartFreq')).setOption({
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    grid: { left: 44, right: 16, top: 16, bottom: field === 'age' ? 26 : 56 },
    xAxis: { type: 'category', data: def.vals, axisLabel: { fontSize: 10, interval: 0, rotate: def.vals.some((v) => v.length > 5) ? 20 : 0 } },
    yAxis: { type: 'value', axisLabel: { fontSize: 10 } },
    series: [{ type: 'bar', data: counts, itemStyle: { color: '#D97706' }, barMaxWidth: 40, label: { show: true, position: 'top', fontSize: 11 } }],
  }, { notMerge: true });
  $('freqTable').innerHTML = `<table class="tbl">
    <thead><tr><th>取值</th><th class="num">频数</th><th class="num">占比</th></tr></thead>
    <tbody>${def.vals.map((v, i) => `<tr><td>${esc(v)}</td><td class="num">${counts[i]}</td><td class="num">${(counts[i] / total * 100).toFixed(2)}%</td></tr>`).join('')}
    <tr class="hl"><td>合计</td><td class="num">${total}</td><td class="num">100%</td></tr></tbody></table>`;
}

// ---- 交叉分析 ----
function crossTable(rowF, colF) {
  const rd = VARS_ALL[rowF], cd = VARS_ALL[colF];
  const getRow = (r) => rowF === 'price_band' ? priceBand(r.price) : String(r[rowF]);
  const getCol = (r) => colF === 'price_band' ? priceBand(r.price) : String(r[colF]);
  const table = rd.vals.map(() => cd.vals.map(() => 0));
  D.forEach((r) => {
    const i = rd.vals.indexOf(getRow(r)), j = cd.vals.indexOf(getCol(r));
    if (i >= 0 && j >= 0) table[i][j] += 1;
  });
  return { rd, cd, table };
}
function renderCross() {
  const rowF = $('crossRow').value, colF = $('crossCol').value;
  const { rd, cd, table } = crossTable(rowF, colF);
  const res = chi2Test(table);
  // 100% 堆叠条形
  const series = cd.vals.map((v, j) => ({
    name: v, type: 'bar', stack: 's', barMaxWidth: 30,
    data: table.map((row) => {
      const tot = row.reduce((a, b) => a + b, 0);
      return tot ? +(row[j] / tot * 100).toFixed(1) : 0;
    }),
  }));
  ensureChart('cross', $('chartCross')).setOption({
    tooltip: { trigger: 'axis', valueFormatter: (v) => v + '%' },
    legend: { top: 0, textStyle: { fontSize: 11 } },
    grid: { left: 44, right: 12, top: 30, bottom: 46 },
    xAxis: { type: 'category', data: rd.vals, axisLabel: { fontSize: 10, interval: 0, rotate: rd.vals.some((v) => v.length > 5) ? 15 : 0 } },
    yAxis: { type: 'value', axisLabel: { formatter: '{value}%', fontSize: 10 } },
    series,
  }, { notMerge: true });

  const sig = res.p < 0.05;
  $('chiResult').innerHTML = `
    <div class="chi-box">
      <div class="chi-item"><b>${res.chi2.toFixed(2)}</b><span>χ² 统计量</span></div>
      <div class="chi-item"><b>${res.df}</b><span>自由度</span></div>
      <div class="chi-item"><b>${res.p < 1e-4 ? res.p.toExponential(1) : res.p.toFixed(4)}</b><span>p 值</span></div>
      <div class="chi-item"><b>${res.v.toFixed(3)}</b><span>Cramér's V</span></div>
    </div>
    <p class="hint" style="margin-top:10px">
      <span class="sig" style="background:${sig ? 'var(--danger)' : 'var(--muted)'}">${sig ? '显著' : '不显著'}</span>
      ${sig ? `在 α=0.05 下拒绝独立性假设：${rd.label} 与 ${cd.label} 存在统计关联（关联强度 Cramér's V = ${res.v.toFixed(3)}，${res.v < 0.1 ? '极弱' : res.v < 0.3 ? '较弱' : res.v < 0.5 ? '中等' : '较强'}）。` : `在 α=0.05 下不能拒绝独立性假设：现有数据不足以支持 ${rd.label} 与 ${cd.label} 存在关联。`}
      ${res.lowExpPct > 20 ? `⚠️ ${res.lowExpPct.toFixed(0)}% 的单元格期望频数 &lt; 5，卡方近似需谨慎解读。` : '期望频数条件满足（&lt;5 的单元格不超过 20%）。'}
    </p>`;
  // 列联表
  $('crossTable').innerHTML = `<table class="tbl">
    <thead><tr><th>${esc(rd.label)} \\ ${esc(cd.label)}</th>${cd.vals.map((v) => `<th class="num">${esc(v)}</th>`).join('')}<th class="num">合计</th></tr></thead>
    <tbody>${rd.vals.map((rv, i) => `<tr><td>${esc(rv)}</td>${table[i].map((c) => `<td class="num">${c}</td>`).join('')}<td class="num">${table[i].reduce((a, b) => a + b, 0)}</td></tr>`).join('')}
    <tr class="hl"><td>合计</td>${cd.vals.map((_, j) => `<td class="num">${table.reduce((s, row) => s + row[j], 0)}</td>`).join('')}<td class="num">${D.length}</td></tr></tbody></table>`;
}

// ---- 量表与相关 ----
function renderScale() {
  const SCALES = {
    trust: { label: '信息信任', cols: ['T1', 'T2', 'T3'] },
    value: { label: '感知价格价值', cols: ['V1', 'V2', 'V3'] },
    risk: { label: '物流风险', cols: ['R1', 'R2', 'R3'] },
    willing: { label: '购买意愿', cols: ['W1', 'W2', 'W3'] },
  };
  const alphas = {};
  for (const [key, s] of Object.entries(SCALES)) alphas[key] = cronbach(s.cols.map((c) => col(c)));
  $('alphaTable').innerHTML = `<table class="tbl">
    <thead><tr><th>量表</th><th>题项</th><th class="num">Cronbach's α</th><th>判定</th></tr></thead>
    <tbody>${Object.entries(SCALES).map(([k, s]) => {
      const a = alphas[k];
      const verdict = a >= 0.8 ? '<span class="sig" style="background:var(--good)">良好</span>' : a >= 0.7 ? '<span class="sig" style="background:var(--accent)">可接受</span>' : '<span class="sig" style="background:var(--danger)">偏低</span>';
      return `<tr><td>${esc(s.label)}</td><td>${s.cols.join('、')}</td><td class="num">${a.toFixed(3)}</td><td>${verdict}</td></tr>`;
    }).join('')}</tbody></table>`;

  const keys = Object.keys(SCALES);
  const sums = keys.map((k) => col(SCALES[k].cols[0]).map((_, i) => SCALES[k].cols.reduce((s, c) => s + col(c)[i], 0) / SCALES[k].cols.length));
  const labels = keys.map((k) => SCALES[k].label);
  const mat = [];
  for (let i = 0; i < keys.length; i++) {
    mat.push([]);
    for (let j = 0; j < keys.length; j++) {
      mat[i][j] = i === j ? 1 : +spearman(sums[i], sums[j]).rho.toFixed(3);
    }
  }
  ensureChart('spear', $('chartSpear')).setOption({
    tooltip: { position: 'top', formatter: (p) => `${labels[p.value[1]]} × ${labels[p.value[0]]}<br>ρ = ${p.value[2]}` },
    grid: { left: 110, right: 20, top: 10, bottom: 90 },
    xAxis: { type: 'category', data: labels, axisLabel: { fontSize: 10, rotate: 20 } },
    yAxis: { type: 'category', data: labels, axisLabel: { fontSize: 10 } },
    visualMap: { min: -0.5, max: 0.5, calculable: true, orient: 'horizontal', left: 'center', bottom: 0,
      inRange: { color: ['#2A6F97', '#FFFFFF', '#D97706'] }, text: ['负相关', '正相关'], textStyle: { fontSize: 10 } },
    series: [{ type: 'heatmap', data: (() => { const c = []; labels.forEach((_, i) => labels.forEach((_, j) => c.push([j, i, mat[i][j]]))); return c; })(),
      label: { show: true, fontSize: 12, formatter: (p) => p.value[2] } }],
  }, { notMerge: true });

  const sp = SURVEY.spearman;
  $('spearConclusion').innerHTML = `
    <p>· <b>感知价格价值 ↔ 购买意愿</b>：ρ = ${sp.value_willing}（正向）——认为"99 元方案划算"的消费者更愿意购买，指向价格构成透明化的运营动作。</p>
    <p>· <b>信息信任 ↔ 购买意愿</b>：ρ = ${sp.trust_willing}（正向）——产地信息、品控与履约信任是转化的前提，对应项目的产品分级与溯源环节。</p>
    <p>· <b>物流风险 ↔ 购买意愿</b>：ρ = ${sp.risk_willing}（负向）——坏果与配送顾虑抑制下单，对应简易防护包装、坏果包赔等设计。</p>
    <p>· 三个方向与《市场调查报告》结论一致；Spearman 选择的原因是 Likert 题为有序变量且分布偏态，秩相关不要求正态假设。</p>`;
}

// ---------- 路由与初始化 ----------
function switchTab(name) {
  document.querySelectorAll('#tabnav button').forEach((b) => b.classList.toggle('active', b.dataset.tab === name));
  document.querySelectorAll('.tab').forEach((s) => s.classList.toggle('active', s.id === 'tab-' + name));
  requestAnimationFrame(() => {
    if (name === 'overview') renderOverview();
    if (name === 'freq') renderFreq();
    if (name === 'cross') renderCross();
    if (name === 'scale') renderScale();
  });
}

function init() {
  if (!window.SURVEY || !SURVEY.records) {
    $('main').innerHTML = '<div class="card"><div class="card-title">数据未加载</div><p class="hint">data.js 缺失或为空。</p></div>';
    return;
  }
  document.querySelectorAll('#tabnav button').forEach((b) =>
    b.addEventListener('click', () => switchTab(b.dataset.tab)));
  // 频数变量
  const freqOpts = Object.entries(VARS_ALL).map(([k, v]) => `<option value="${k}">${v.label}</option>`).join('');
  $('freqVar').innerHTML = freqOpts;
  $('freqVar').addEventListener('change', renderFreq);
  // 交叉变量（行/列，排除自身组合）
  const crossOpts = Object.entries(VARS_ALL).map(([k, v]) => `<option value="${k}">${v.label}</option>`).join('');
  $('crossRow').innerHTML = crossOpts;
  $('crossCol').innerHTML = crossOpts;
  $('crossRow').value = 'age';
  $('crossCol').value = 'willing';
  $('crossRow').addEventListener('change', renderCross);
  $('crossCol').addEventListener('change', renderCross);
  renderOverview();
  window.addEventListener('resize', () => Object.values(charts).forEach((c) => c.resize()));
}

init();
