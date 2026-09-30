import { aggregatePack, categorySeriesRows, monthlySeries, selectedAgencyTotals, selectedSeriesStats, sumMaps, sumNestedMaps } from './data.js';

const APP_BUILD_ID = '__BUILD_ID__';
const $ = id => document.getElementById(id);
const number = new Intl.NumberFormat('ru-RU');
const percent = new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 1 });
const format = value => value === null || value === undefined ? '—' : number.format(value);
const formatPercent = value => value === null ? '—' : `${percent.format(value)}%`;
const fullDate = value => new Date(`${value}T12:00:00Z`).toLocaleDateString('ru-RU', { day: 'numeric', month: 'long', year: 'numeric', timeZone: 'UTC' });
const shortDate = value => new Date(`${value}T12:00:00Z`).toLocaleDateString('ru-RU', { day: 'numeric', month: 'short', timeZone: 'UTC' });
const monthName = value => new Date(`${value.slice(0, 7)}-01T12:00:00Z`).toLocaleDateString('ru-RU', { month: 'long', timeZone: 'UTC' });
const SVG = 'http://www.w3.org/2000/svg';
let manifest;
let renderVersion = 0;
let currentRows = [];
let currentSeries = [];
let currentSummary;
let currentSelected = [];
const packs = new Map();
let requestSequence = 0;
let lastBuildCheck = 0;
const fresh = () => `${Date.now()}-${++requestSequence}`;
async function fetchManifest() {
  let error;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const response = await fetch(`./data/manifest.json?fresh=${fresh()}`, { cache: 'no-store' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      if (!data.buildId || !data.years) throw new Error('Неполный манифест');
      return data;
    } catch (cause) { error = cause; }
  }
  throw new Error(`manifest.json недоступен: ${error.message}. Откройте сайт через локальный сервер.`);
}
function restartForBuild(buildId) {
  const previous = JSON.parse(sessionStorage.getItem('buildReload') || '{}');
  if (previous.id === buildId && Date.now() - previous.at < 30_000) {
    throw new Error('Новая версия сайта ещё публикуется. Повторите загрузку через минуту.');
  }
  sessionStorage.setItem('buildReload', JSON.stringify({ id: buildId, at: Date.now() }));
  const url = new URL(location.href);
  url.searchParams.set('__build', buildId);
  location.replace(url);
}
async function checkLatestBuild() {
  if (Date.now() - lastBuildCheck < 60_000) return;
  lastBuildCheck = Date.now();
  const latest = await fetchManifest();
  if (latest.buildId !== manifest.buildId) restartForBuild(latest.buildId);
}
function unfilteredState() {
  return { metric: 'decisions', year: String(manifest?.latestYear ?? 2026), country: 'all', group: 'all',
    institution: 'all', caseType: 'all', marker: 'all', series: { decisions: null, applications: null },
    mode: 'cumulative', range: 'all', zoom: null, zoomBaseRange: null, showAll: false };
}
function defaultState() {
  return { ...unfilteredState(), year: manifest?.years?.['2026'] ? '2026' : String(manifest?.latestYear ?? 2026),
    country: '241', institution: '810', caseType: '4',
    series: { decisions: ['4', '11', '6'], applications: null }, mode: 'snapshot' };
}
const state = defaultState();
const seriesColors = { 4: '#168a72', 11: '#566ac3', 6: '#cc0000', 8: '#b88331', 1: '#879596',
  3: '#9c6eb3', 5: '#3b96a6', 9: '#b86699', 12: '#887e6d', 21: '#43a58e', 22: '#b4a05b' };
const caseTypeColors = { 1: '#168a72', 2: '#566ac3', 3: '#b88331', 4: '#9c6eb3' };
const labels = { decisions: ['Решения', 'РЕШЕНИЙ В СРЕЗЕ', 'Как менялось число решений', 'По результатам'],
  applications: ['Заявления', 'ЗАЯВЛЕНИЙ В СРЕЗЕ', 'Как менялось число заявлений', 'По типам дел'],
  statuses: ['Статусы', 'СТАТУСОВ В СРЕЗЕ', 'Как менялось число статусов', 'По статусам'] };
const groupLabels = { PSG: 'Placówki Straży Granicznej', OSG: 'Oddziały Straży Granicznej',
  WOJ: 'Wojewodowie', MIN: 'Ministerstwo' };
const groupOrder = ['PSG', 'OSG', 'WOJ', 'MIN'];

function create(tag, className, content) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (content !== undefined) el.textContent = content;
  return el;
}
function option(value, name) { const el = document.createElement('option'); el.value = String(value); el.textContent = name; return el; }
function fillSelect(select, items, selected, allLabel) {
  select.replaceChildren();
  if (allLabel) select.append(option('all', allLabel));
  for (const item of items) select.append(option(item.id, item.name));
  select.value = selected;
  if (select.selectedIndex < 0) select.value = 'all';
  return select.value;
}
function dictionary(name, id) { return manifest.dictionaries[name].find(item => item.id === id)?.name ?? `ID ${id}`; }
function groups() { return new Map(manifest.dictionaries.institutions.map(item => [item.id, item.authorityCode])); }
function groupLabel(code) {
  const name = groupLabels[code] ?? manifest.dictionaries.institutions.find(item => item.authorityCode === code)?.authorityName ?? code;
  return `${code} · ${name}`;
}
function syncControls() {
  const years = Object.keys(manifest.years).sort((a, b) => Number(b) - Number(a));
  fillSelect($('yearFilter'), years.map(year => ({ id: year, name: `${year} год` })), state.year, 'Все годы');
  const countries = [...manifest.dictionaries.countries].sort((a,b) => a.name.localeCompare(b.name, 'pl'));
  const popular = ['BY', 'UA', 'RU'];
  countries.sort((a,b) => {
    const left = popular.indexOf(a.code), right = popular.indexOf(b.code);
    if (left !== -1 || right !== -1) return (left === -1 ? 99 : left) - (right === -1 ? 99 : right);
    return a.name.localeCompare(b.name, 'pl');
  });
  state.country = fillSelect($('countryFilter'), countries.map(item => ({ id: item.id, name: `${item.name}${item.code && item.code !== 'undefined' ? ` · ${item.code}` : ''}` })), state.country, 'Все гражданства');
  const metricInstitutions = new Set(manifest.institutionsByMetric[state.metric]);
  const availableInstitutions = manifest.dictionaries.institutions.filter(item => metricInstitutions.has(item.id));
  const availableGroups = new Set(availableInstitutions.map(item => item.authorityCode));
  const orderedGroups = [...availableGroups].sort((a, b) => {
    const left = groupOrder.indexOf(a), right = groupOrder.indexOf(b);
    if (left !== -1 || right !== -1) return (left === -1 ? 99 : left) - (right === -1 ? 99 : right);
    return a.localeCompare(b, 'pl');
  });
  $('groupFilter').replaceChildren(option('all', 'Все органы'),
    ...orderedGroups.map(group => option(group, groupLabel(group))));
  if (!availableGroups.has(state.group)) state.group = 'all';
  $('groupFilter').value = state.group;
  const institutions = availableInstitutions.filter(item => state.group === 'all' || item.authorityCode === state.group)
    .sort((a,b) => a.name.localeCompare(b.name, 'pl'));
  state.institution = fillSelect($('institutionFilter'), institutions, state.institution, 'Все органы группы');
  state.caseType = fillSelect($('caseFilter'), manifest.dictionaries.caseTypes, state.caseType, 'Все типы дел');
  const markerList = state.metric === 'statuses' ? manifest.dictionaries.statuses : manifest.dictionaries.decisionMarkers;
  state.marker = fillSelect($('markerFilter'), markerList, state.marker, state.metric === 'statuses' ? 'Все статусы' : 'Все результаты');
  $('caseBlock').hidden = state.metric === 'statuses';
  $('markerBlock').hidden = state.metric === 'applications';
  $('markerLabel').textContent = state.metric === 'statuses' ? 'Статус' : 'Результат решения';
  document.querySelectorAll('[data-metric]').forEach(tab => {
    const active = tab.dataset.metric === state.metric;
    tab.classList.toggle('active', active);
    tab.setAttribute('aria-selected', String(active));
  });
  document.querySelectorAll('[data-mode]').forEach(button => {
    button.classList.toggle('selected', state.year !== 'all' && button.dataset.mode === state.mode);
    button.disabled = state.year === 'all';
  });
  document.querySelectorAll('[data-range]').forEach(button => button.classList.toggle('selected', button.dataset.range === state.range));
  $('resetZoom').hidden = state.zoom === null;
}
function storeUrl() {
  const params = new URLSearchParams();
  for (const key of ['metric','year','country','group','institution','caseType','marker','mode']) {
    if (key === 'year' && state.year === 'all') { params.set('year', 'all'); continue; }
    if (state[key] !== 'all' && state[key] !== 'decisions' && state[key] !== 'cumulative') params.set(key, state[key]);
  }
  if ((state.metric === 'decisions' && state.marker === 'all' || state.metric === 'applications' && state.caseType === 'all')
      && state.series[state.metric]?.length) params.set('series', state.series[state.metric].join(','));
  history.replaceState(null, '', `${location.pathname}${params.size ? `?${params}` : ''}`);
}
function restoreUrl() {
  const params = new URLSearchParams(location.search);
  for (const key of ['metric','year','country','group','institution','caseType','marker','mode']) {
    if (params.has(key)) state[key] = params.get(key);
  }
  if (!['decisions', 'applications'].includes(state.metric)) state.metric = 'decisions';
  if (!manifest.years[state.year] && state.year !== 'all') state.year = String(manifest.latestYear);
  state.group = { voivodeship: 'WOJ', chief: 'MIN', other: 'MIN', psg: 'PSG', sg_branch: 'OSG' }[state.group] ?? state.group;
  if (!['cumulative','snapshot','month'].includes(state.mode)) state.mode = 'snapshot';
  if (params.has('series')) state.series[state.metric] = params.get('series').split(',').filter(id => /^\d+$/.test(id));
}
async function getPack(year, country) {
  if (country !== 'all' && !manifest.years[year].countries.includes(Number(country))) return null;
  const key = `${year}/${country}`;
  if (!packs.has(key)) packs.set(key, (async () => {
    const expected = manifest.years[year].packHashes[country];
    if (!expected) throw new Error(`Нет контрольной суммы для ${key}`);
    for (let attempt = 0; attempt < 3; attempt++) {
      const suffix = attempt ? `&fresh=${fresh()}` : '';
      const response = await fetch(`./data/${key}.json?build=${manifest.buildId}${suffix}`,
        { cache: attempt ? 'no-store' : 'default' });
      if (!response.ok) {
        if (attempt === 2) throw new Error(`Не удалось открыть данные ${key}`);
        continue;
      }
      const bytes = await response.arrayBuffer();
      const digest = await crypto.subtle.digest('SHA-256', bytes);
      const actual = [...new Uint8Array(digest)].map(byte => byte.toString(16).padStart(2, '0')).join('');
      if (actual === expected) return JSON.parse(new TextDecoder().decode(bytes));
      const latest = await fetchManifest();
      if (latest.buildId !== manifest.buildId) restartForBuild(latest.buildId);
    }
    throw new Error('Данные разных версий не совпали. Повторите загрузку через минуту.');
  })());
  try { return await packs.get(key); }
  catch (error) { packs.delete(key); throw error; }
}
function filters() { return { group: state.group, institution: state.institution, caseType: state.caseType, marker: state.marker }; }
async function yearResult(year) {
  const points = manifest.years[year].snapshots;
  const pack = await getPack(year, state.country);
  return { year, points, pack, ...aggregatePack(pack, state.metric, points, filters(), groups()) };
}
function chartRows(result) {
  if (state.year === 'all') return result.annual.map(item => ({ date: item.date, label: item.year, value: item.total,
    coverage: item.date.endsWith('12-31') ? 'Срез на конец года' : `Последний срез: ${fullDate(item.date)}` }));
  const { points, cumulative, changes } = result;
  if (state.mode === 'month') return monthlySeries(points, cumulative);
  return points.map((point, i) => ({ date: point.date, label: point.date,
    value: state.mode === 'snapshot' && i === 0 ? null : state.mode === 'snapshot' ? changes[i] : cumulative[i],
    coverage: state.mode === 'snapshot' ? i === 0 ? 'Первый снимок года — базовый итог' : `С предыдущего снимка ${points[i-1].date}` : 'Накопительный итог' }));
}
function availableCategories(summary) {
  return [...summary.categories].filter(([, value]) => value > 0)
    .sort((a, b) => b[1] - a[1]).map(([id]) => String(id));
}
function selectedCategories(summary) {
  const available = availableCategories(summary);
  const selectedFilter = state.metric === 'decisions' ? state.marker : state.caseType;
  if (selectedFilter !== 'all') return [selectedFilter];
  if (!available.length) return [];
  if (state.series[state.metric] === null) {
    const preferred = state.metric === 'decisions' ? ['4', '11'].filter(id => available.includes(id)) : available;
    state.series[state.metric] = preferred.length ? preferred : available.slice(0, 2);
  }
  state.series[state.metric] = state.series[state.metric].filter(id => available.includes(id));
  if (!state.series[state.metric].length) state.series[state.metric] = state.metric === 'applications' ? available : available.slice(0, 2);
  return state.series[state.metric];
}
function renderSeriesPicker(summary, selected) {
  const picker = $('seriesPicker');
  picker.hidden = !(state.metric === 'decisions' && state.marker === 'all' || state.metric === 'applications' && state.caseType === 'all');
  if (picker.hidden) return;
  const applications = state.metric === 'applications';
  $('seriesPickerLabel').textContent = applications ? 'СРАВНИТЬ ТИПЫ ДЕЛ' : 'СРАВНИТЬ РЕЗУЛЬТАТЫ';
  $('seriesButtons').setAttribute('aria-label', applications ? 'Типы дел на графике' : 'Результаты на графике');
  $('seriesButtons').replaceChildren(...availableCategories(summary).map(id => {
    const button = create('button', 'series-button', dictionary(applications ? 'caseTypes' : 'decisionMarkers', Number(id)));
    const active = selected.includes(id);
    button.type = 'button'; button.dataset.series = id; button.setAttribute('aria-pressed', String(active));
    button.style.setProperty('--series-color', (applications ? caseTypeColors : seriesColors)[id] ?? '#526b9d');
    button.prepend(create('i'));
    button.addEventListener('click', () => {
      if (active && state.series[state.metric].length === 1) return;
      state.series[state.metric] = active ? state.series[state.metric].filter(item => item !== id) : [...state.series[state.metric], id];
      state.zoom = null; state.zoomBaseRange = null; update();
    });
    return button;
  }));
}
function chartData(result, selected) {
  if (!selected.length) {
    return { rows: chartRows(result).map(row => ({ ...row, values: { total: row.value } })),
      series: [{ key: 'total', label: labels[state.metric][0], color: '#168a72' }] };
  }
  const applications = state.metric === 'applications';
  const series = selected.map(id => ({ key: id, label: dictionary(applications ? 'caseTypes' : 'decisionMarkers', Number(id)),
    color: (applications ? caseTypeColors : seriesColors)[id] ?? '#526b9d' }));
  if (state.year === 'all') {
    const rows = result.perYear.map(item => ({ date: item.points.at(-1).date, label: item.year,
      values: Object.fromEntries(selected.map(id => [id, aggregatePack(item.pack, state.metric, item.points,
        { ...filters(), [applications ? 'caseType' : 'marker']: id }, groups()).total])),
      coverage: item.points.at(-1).date.endsWith('12-31') ? 'Срез на конец года' : `Последний срез: ${fullDate(item.points.at(-1).date)}` }));
    return { rows, series };
  }
  return { rows: categorySeriesRows(result.pack, state.metric, result.points, filters(), groups(), selected, state.mode), series };
}
function scopedRows(rows) {
  if (state.year === 'all' || state.mode === 'month') return rows;
  let output = rows;
  if (state.range !== 'all') {
    const cutoff = Date.parse(`${rows.at(-1)?.date}T00:00:00Z`) - Number(state.range) * 86400000;
    output = rows.filter(row => Date.parse(`${row.date}T00:00:00Z`) >= cutoff);
  }
  if (state.zoom) output = output.slice(state.zoom[0], state.zoom[1] + 1);
  return output;
}
function s(tag, attrs = {}) {
  const element = document.createElementNS(SVG, tag);
  for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
  return element;
}
function nice(value) {
  if (value === 0) return 0;
  const exponent = 10 ** Math.floor(Math.log10(Math.abs(value)));
  const factor = Math.abs(value) / exponent;
  return Math.ceil(factor / (factor <= 2 ? .5 : factor <= 5 ? 1 : 2)) * (factor <= 2 ? .5 : factor <= 5 ? 1 : 2) * exponent;
}
function showTooltip(event, row, series) {
  const tip = $('chartTooltip');
  const stats = selectedSeriesStats(row.values, series.map(item => item.key));
  tip.replaceChildren(create('span', 'tooltip-date', state.year === 'all' ? `${row.label} год` :
    state.mode === 'month' ? `${monthName(row.date)} ${row.date.slice(0, 4)} г.` : fullDate(row.date)));
  tip.append(create('span', 'tooltip-scope', $('chartScope').textContent));
  for (const item of series) {
    const line = create('div', 'tooltip-series');
    const label = create('span');
    const dot = create('i'); dot.style.background = item.color;
    label.append(dot, document.createTextNode(item.label));
    const amount = create('div', 'tooltip-amount');
    amount.append(create('strong', '', format(row.values[item.key])));
    amount.append(create('span', 'tooltip-percent', formatPercent(stats.percentages[item.key])));
    line.append(label, amount);
    tip.append(line);
  }
  const total = create('div', 'tooltip-total');
  total.append(create('span', '', 'Итого выбранных'), create('strong', '', format(stats.total)));
  tip.append(total);
  tip.append(create('small', '', row.coverage ?? ''));
  tip.hidden = false;
  const bounds = tip.getBoundingClientRect();
  tip.style.left = `${Math.max(8, Math.min(window.innerWidth - bounds.width - 8, event.clientX + 16))}px`;
  tip.style.top = `${Math.max(8, Math.min(window.innerHeight - bounds.height - 8, event.clientY - bounds.height - 12))}px`;
}
function renderLegend(rows, series) {
  const latest = rows.at(-1);
  $('chartLegend').replaceChildren(...series.map(item => {
    const label = create('span', 'legend-key');
    const dot = create('i'); dot.style.background = item.color;
    label.append(dot, document.createTextNode(item.label));
    if (latest && latest.values[item.key] !== null) label.append(create('strong', '', format(latest.values[item.key])));
    return label;
  }));
}
function drawChart(rows, series = currentSeries) {
  currentRows = rows; currentSeries = series;
  const area = $('chartArea');
  const visible = scopedRows(rows);
  area.replaceChildren();
  renderLegend(visible, series);
  if (!visible.length || visible.every(row => series.every(item => row.values[item.key] === null))) {
    area.append(create('div', 'empty', 'Для этих фильтров нет сопоставимых снимков.'));
    $('chartTooltip').hidden = true;
    return;
  }
  const w = 960, h = 300, left = 65, right = 20, top = 16, bottom = 41;
  const width = w - left - right, height = h - top - bottom;
  const svg = s('svg', { viewBox: `0 0 ${w} ${h}`, role: 'img', 'aria-label': `График: ${visible.length} точек` });
  const values = visible.flatMap(row => series.map(item => row.values[item.key])).filter(value => value !== null && value !== undefined);
  const low = Math.min(0, ...values), high = Math.max(0, ...values);
  const yMin = low < 0 ? -nice(-low) : 0, yMax = nice(high || 1);
  const y = value => top + height - (value - yMin) / (yMax - yMin) * height;
  const bars = state.year === 'all' || state.mode === 'month';
  const dates = visible.map(row => Date.parse(`${row.date}T00:00:00Z`));
  const first = dates[0], span = Math.max(1, dates.at(-1) - first);
  const x = i => bars ? left + (i + .5) * width / visible.length : left + (dates[i] - first) / span * width;
  for (let i = 0; i <= 4; i++) {
    const tick = yMin + (yMax - yMin) * (4 - i) / 4;
    const yy = top + i * height / 4;
    svg.append(s('line', { x1: left, x2: w - right, y1: yy, y2: yy, class: 'grid' }));
    const text = s('text', { x: left - 11, y: yy + 4, 'text-anchor': 'end', class: 'axis-label' });
    text.textContent = number.format(Math.round(tick)); svg.append(text);
  }
  const tickCount = Math.min(5, visible.length);
  for (let i = 0; i < tickCount; i++) {
    const index = Math.round(i * (visible.length - 1) / Math.max(1, tickCount - 1));
    const text = s('text', { x: x(index), y: h - 12, 'text-anchor': i === 0 ? 'start' : i === tickCount - 1 ? 'end' : 'middle', class: 'axis-label' });
    text.textContent = state.year === 'all' ? visible[index].label : state.mode === 'month' ? monthName(visible[index].date).slice(0, 3) : shortDate(visible[index].date);
    svg.append(text);
  }
  if (bars) {
    const groupWidth = Math.min(70, width / visible.length * .74);
    const barWidth = groupWidth / series.length;
    visible.forEach((row, i) => {
      if (state.mode === 'month' && series.every(item => row.values[item.key] === null)) {
        const missing = s('text', { x: x(i), y: y(0) - 10, 'text-anchor': 'middle', class: 'missing-month-label' });
        missing.textContent = 'нет данных'; svg.append(missing);
      }
      series.forEach((item, j) => {
        const value = row.values[item.key];
        if (value === null || value === undefined || value === 0) return;
        const baseline = y(0), valueY = y(value);
        const barHeight = Math.max(4, Math.abs(baseline - valueY));
        svg.append(s('rect', { x: x(i) - groupWidth / 2 + j * barWidth,
          y: value > 0 ? baseline - barHeight : baseline,
          width: Math.max(1, barWidth - 2), height: barHeight,
          rx: 2, class: 'bar', style: `fill:${item.color}` }));
      });
    });
  } else {
    for (const item of series) {
      let started = false;
      const line = visible.map((row, i) => {
        const value = row.values[item.key];
        if (value === null || value === undefined) { started = false; return ''; }
        const command = started ? 'L' : 'M'; started = true;
        return `${command}${x(i).toFixed(2)},${y(value).toFixed(2)}`;
      }).filter(Boolean).join(' ');
      svg.append(s('path', { d: line, class: 'line', style: `stroke:${item.color}` }));
      if (visible.length <= 2) visible.forEach((row, i) => {
        const value = row.values[item.key];
        if (value !== null && value !== undefined) svg.append(s('circle', { cx: x(i), cy: y(value), r: 3, fill: item.color }));
      });
    }
  }
  const focusLine = s('line', { x1: 0, x2: 0, y1: top, y2: top + height, class: 'focus-line', visibility: 'hidden' });
  const focusDots = series.map(item => s('circle', { cx: 0, cy: 0, r: 5, class: 'focus-dot', style: `stroke:${item.color}`, visibility: 'hidden' }));
  const selection = s('rect', { x: 0, y: top, width: 0, height, class: 'selection', visibility: 'hidden' });
  svg.append(focusLine, ...focusDots, selection);
  let drag = null;
  const coordinate = event => {
    const rect = svg.getBoundingClientRect();
    return Math.min(w - right, Math.max(left, (event.clientX - rect.left) / rect.width * w));
  };
  const closest = pos => {
    let best = 0;
    for (let i = 1; i < visible.length; i++) if (Math.abs(x(i) - pos) < Math.abs(x(best) - pos)) best = i;
    return best;
  };
  svg.addEventListener('pointermove', event => {
    const pos = coordinate(event), i = closest(pos);
    focusLine.setAttribute('x1', x(i)); focusLine.setAttribute('x2', x(i)); focusLine.setAttribute('visibility', 'visible');
    focusDots.forEach((dot, j) => {
      const value = visible[i].values[series[j].key];
      dot.setAttribute('cx', x(i)); dot.setAttribute('cy', y(value ?? 0));
      dot.setAttribute('visibility', value === null || value === undefined ? 'hidden' : 'visible');
    });
    showTooltip(event, visible[i], series);
    if (drag !== null) { selection.setAttribute('x', Math.min(drag, pos)); selection.setAttribute('width', Math.abs(drag - pos)); selection.setAttribute('visibility', 'visible'); }
  });
  svg.addEventListener('pointerleave', () => { if (drag === null) $('chartTooltip').hidden = true; focusLine.setAttribute('visibility', 'hidden'); focusDots.forEach(dot => dot.setAttribute('visibility', 'hidden')); });
  svg.addEventListener('pointerdown', event => { if (state.year === 'all' || state.mode === 'month') return; drag = coordinate(event); svg.setPointerCapture(event.pointerId); });
  svg.addEventListener('pointerup', event => {
    if (drag === null) return;
    const end = coordinate(event);
    if (Math.abs(end - drag) > 20) {
      const lowIndex = closest(Math.min(drag, end)), highIndex = closest(Math.max(drag, end));
      const initial = scopedRows(rows);
      const firstIndex = rows.indexOf(initial[0]);
      if (state.zoom === null) state.zoomBaseRange = state.range;
      state.range = 'all'; state.zoom = [firstIndex + lowIndex, firstIndex + highIndex];
      $('chartTooltip').hidden = true;
      syncControls(); drawChart(rows, series);
    }
    drag = null; selection.setAttribute('visibility', 'hidden');
  });
  area.append(svg);
}

function renderBreakdown(summary, date) {
  $('breakdownHeading').textContent = labels[state.metric][3];
  $('breakdownDate').textContent = date ? shortDate(date) : '';
  const items = [...summary.categories].filter(([,value]) => value > 0).sort((a,b) => b[1] - a[1]);
  const key = state.metric === 'decisions' ? 'decisionMarkers' : state.metric === 'applications' ? 'caseTypes' : 'statuses';
  const max = items[0]?.[1] || 1;
  $('breakdownRows').replaceChildren(...(items.length ? items.slice(0, 8).map(([id,value]) => {
    const row = create('div', 'breakdown-item');
    const head = create('div', 'breakdown-item-top'); head.append(create('span', '', dictionary(key,id)), create('strong', '', format(value)));
    const bar = create('div', 'track'); const fill = create('i');
    fill.style.width = `${value / max * 100}%`;
    fill.style.backgroundColor = (state.metric === 'applications' ? caseTypeColors : seriesColors)[id] ?? '#879596';
    bar.append(fill); row.append(head,bar);
    return row;
  }) : [create('div','empty','Нет данных для выбранных фильтров.') ]));
}
function renderAgencies(summary, selected) {
  const agencyTotals = selected.length ? selectedAgencyTotals(summary.agencyCategories, selected) : summary.agencies;
  const items = [...agencyTotals].filter(([,value]) => value > 0).sort((a,b) => b[1] - a[1]);
  $('agenciesScope').textContent = selected.length === 1
    ? dictionary(state.metric === 'applications' ? 'caseTypes' : 'decisionMarkers', Number(selected[0]))
    : selected.length ? `Сумма выбранных ${state.metric === 'applications' ? 'типов дел' : 'результатов'} (${selected.length})`
      : `Все ${state.metric === 'applications' ? 'типы дел' : 'результаты'}`;
  const search = $('agencySearch').value.trim().toLocaleLowerCase('pl');
  const filtered = search ? items.filter(([id]) => dictionary('institutions',id).toLocaleLowerCase('pl').includes(search)) : items;
  $('agenciesCount').textContent = `${items.length} органов`;
  const shown = state.showAll || search ? filtered : filtered.slice(0, 8);
  $('agencyRows').replaceChildren(...(shown.length ? shown.map(([id,value]) => {
    const row = create('button', 'agency-row'); row.type = 'button'; row.title = 'Показать динамику этого органа';
    row.append(create('span','agency-rank',String(items.findIndex(item => item[0] === id) + 1).padStart(2,'0')),
      create('span','agency-name',dictionary('institutions',id)),create('span','agency-val',format(value)));
    row.addEventListener('click', () => { state.institution = String(id); state.group = 'all'; state.zoom = null; state.zoomBaseRange = null; syncControls(); update(); });
    return row;
  }) : [create('div','empty','Органы не найдены.') ]));
  $('showAllAgencies').hidden = Boolean(search) || filtered.length <= 8;
  $('showAllAgencies').textContent = state.showAll ? 'Свернуть список ↑' : `Показать все ${filtered.length} органов ↓`;
}
function renderCards(summary, date, monthly) {
  $('totalLabel').textContent = state.year === 'all' ? `${labels[state.metric][0].toUpperCase()} ЗА ВСЕ ГОДЫ` : labels[state.metric][1];
  $('totalValue').textContent = format(summary.total);
  $('totalFoot').textContent = date ? `По данным на ${fullDate(date)}` : 'Нет данных';
  const current = monthly.at(-1);
  $('monthLabel').textContent = state.year === 'all' ? 'ПОСЛЕДНИЙ ГОД' : `ИЗМЕНЕНИЕ ЗА ${current ? monthName(current.date).toUpperCase() : 'МЕСЯЦ'}`;
  $('monthValue').textContent = state.year === 'all' ? format(summary.lastYearTotal) : format(current?.value);
  $('monthFoot').textContent = state.year === 'all' ? `На ${fullDate(date)}` : current?.coverage ?? 'Нет снимков';
  $('agencyValue').textContent = format([...summary.agencies.values()].filter(value => value > 0).length);
}
function downloadCsv() {
  const data = [['date', ...currentSeries.map(item => item.label)],
    ...currentRows.map(row => [row.date, ...currentSeries.map(item => row.values[item.key] ?? '')])];
  const csv = '\ufeff' + data.map(row => row.map(value => `"${String(value).replaceAll('"','""')}"`).join(',')).join('\r\n');
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
  const link = create('a'); link.href = url; link.download = `pl-decisions-${state.metric}-${state.year}.csv`; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function renderText(result) {
  $('updatedDate').textContent = fullDate(manifest.years[String(manifest.latestYear)].lastUpdated);
  const country = state.country === 'all' ? 'Все гражданства' : dictionary('countries', Number(state.country));
  const agency = state.institution !== 'all' ? dictionary('institutions',Number(state.institution)) :
    state.group === 'all' ? 'все органы' : groupLabel(state.group);
  $('scopeLabel').textContent = `${country} · ${agency}`;
  const caseType = state.metric === 'statuses' ? '' : state.caseType === 'all' ? 'все типы дел' :
    dictionary('caseTypes', Number(state.caseType));
  $('chartScope').textContent = [country, agency, caseType].filter(Boolean).join(' · ');
  $('chartHeading').textContent = state.year === 'all' ? `${labels[state.metric][0]} по годам` : labels[state.metric][2];
  $('chartSubtitle').textContent = state.year === 'all' ? 'Последний доступный срез каждого года' :
    state.mode === 'snapshot' ? `Изменение между снимками за ${state.year} год` :
    state.mode === 'month' ? `Изменение по месяцам за ${state.year} год · оценка по снимкам` : `Накопительные данные за ${state.year} год`;
  $('chartHint').textContent = state.year === 'all' || state.mode === 'month' ? 'Наведите для точной даты и значения' : 'Наведите для деталей · потяните для приближения';
  $('coverageNotice').textContent = state.year === 'all' ? 'Неполные годы показаны по дате последнего доступного среза.' :
    state.mode === 'month' ? 'Каждый столбец — разница между первым и последним снимками этого же месяца. Подсказка показывает даты сравнения; один снимок даёт 0, отсутствие снимков — неизвестное значение. Разницу с предыдущим срезом смотрите в режиме «По снимкам».' :
    state.mode === 'snapshot' ? 'Первый снимок года — базовый итог; отрицательные значения отражают исправления.' : '';
}
async function update() {
  const version = ++renderVersion;
  syncControls(); storeUrl();
  $('error').hidden = true;
  try {
    let result, summary, date, monthly = [];
    if (state.year === 'all') {
      const years = Object.keys(manifest.years).sort();
      const results = await Promise.all(years.map(yearResult));
      if (version !== renderVersion) return;
      const annual = results.map(item => ({ year: item.year, date: item.points.at(-1).date, total: item.total }));
      summary = { total: annual.reduce((sum,row) => sum + row.total,0), agencies: new Map(), categories: new Map(), agencyCategories: new Map(), lastYearTotal: annual.at(-1).total };
      for (const item of results) { sumMaps(summary.agencies,item.agencies); sumMaps(summary.categories,item.categories); sumNestedMaps(summary.agencyCategories,item.agencyCategories); }
      date = annual.at(-1).date; result = { annual, perYear: results };
    } else {
      result = await yearResult(state.year);
      if (version !== renderVersion) return;
      summary = result; date = result.points.at(-1)?.date;
      monthly = monthlySeries(result.points,result.cumulative);
    }
    const selected = selectedCategories(summary);
    const { rows, series } = chartData(result, selected);
    renderText(result); renderCards(summary,date,monthly); renderBreakdown(summary,date); renderAgencies(summary, selected);
    renderSeriesPicker(summary, selected); drawChart(rows, series); storeUrl();
    currentSummary = summary; currentSelected = selected;
    document.querySelector('.range-buttons').hidden = state.year === 'all' || state.mode === 'month';
  } catch (error) {
    if (version !== renderVersion) return;
    $('error').textContent = `Не удалось загрузить данные: ${error.message}`;
    $('error').hidden = false;
    console.error(error);
  }
}
function bind() {
  const filters = { yearFilter: 'year', countryFilter: 'country', groupFilter: 'group', institutionFilter: 'institution', caseFilter: 'caseType', markerFilter: 'marker' };
  for (const [id,key] of Object.entries(filters)) $(id).addEventListener('change', event => {
    state[key] = event.target.value;
    if (key === 'group') state.institution = 'all';
    if (key === 'marker' || key === 'caseType') state.series[state.metric] = null;
    state.zoom = null; state.zoomBaseRange = null; state.range = 'all'; update();
  });
  document.querySelectorAll('[data-metric]').forEach(button => button.addEventListener('click', () => {
    state.metric = button.dataset.metric; state.marker = 'all'; state.caseType = 'all'; state.zoom = null; state.zoomBaseRange = null; update();
  }));
  document.querySelectorAll('[data-mode]').forEach(button => button.addEventListener('click', () => {
    if (state.year === 'all') return; state.mode = button.dataset.mode; state.zoom = null; state.zoomBaseRange = null; state.range = 'all'; update();
  }));
  document.querySelectorAll('[data-range]').forEach(button => button.addEventListener('click', () => {
    state.range = button.dataset.range; state.zoom = null; state.zoomBaseRange = null; syncControls(); drawChart(currentRows);
  }));
  $('resetZoom').addEventListener('click', () => {
    if (state.zoom === null) return;
    state.zoom = null; state.range = state.zoomBaseRange ?? 'all'; state.zoomBaseRange = null;
    syncControls(); drawChart(currentRows);
  });
  $('resetFilters').addEventListener('click', () => {
    Object.assign(state, unfilteredState());
    $('agencySearch').value = ''; update();
  });
  $('agencySearch').addEventListener('input', () => { if (currentSummary) renderAgencies(currentSummary, currentSelected); });
  $('showAllAgencies').addEventListener('click', () => { state.showAll = !state.showAll; if (currentSummary) renderAgencies(currentSummary, currentSelected); });
  $('downloadCsv').addEventListener('click', downloadCsv);
}
async function init() {
  try {
    manifest = await fetchManifest();
    if (manifest.buildId !== APP_BUILD_ID) { restartForBuild(manifest.buildId); return; }
    const params = new URLSearchParams(location.search);
    params.delete('__build');
    Object.assign(state, params.size ? unfilteredState() : defaultState());
    restoreUrl(); bind(); update();
    document.addEventListener('visibilitychange', () => {
      if (!document.hidden) checkLatestBuild().catch(console.error);
    });
    window.addEventListener('focus', () => checkLatestBuild().catch(console.error));
    setInterval(() => {
      if (!document.hidden) checkLatestBuild().catch(console.error);
    }, 5 * 60_000);
  } catch (error) { $('error').textContent = error.message; $('error').hidden = false; $('updatedDate').textContent = 'Данные недоступны'; }
}
init();
