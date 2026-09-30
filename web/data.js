// Pure data transformations shared by the dashboard and regression checks.
export function aggregatePack(pack, metric, snapshots, filters, institutionGroups, cutoff = snapshots.length - 1) {
  const changes = Array(snapshots.length).fill(0);
  const agencies = new Map();
  const categories = new Map();
  const agencyCategories = new Map();
  for (const row of pack?.[metric] ?? []) {
    const [index, institution, dimension, fourth, fifth] = row;
    const value = metric === 'decisions' ? fifth : fourth;
    const category = metric === 'decisions' ? fourth : dimension;
    if (index > cutoff) continue;
    if (filters.group !== 'all' && institutionGroups.get(institution) !== filters.group) continue;
    if (filters.institution !== 'all' && institution !== Number(filters.institution)) continue;
    if (metric !== 'statuses' && filters.caseType !== 'all' && dimension !== Number(filters.caseType)) continue;
    if (metric !== 'applications' && filters.marker !== 'all' && category !== Number(filters.marker)) continue;
    changes[index] += value;
    agencies.set(institution, (agencies.get(institution) ?? 0) + value);
    categories.set(category, (categories.get(category) ?? 0) + value);
    if (!agencyCategories.has(institution)) agencyCategories.set(institution, new Map());
    const agency = agencyCategories.get(institution);
    agency.set(category, (agency.get(category) ?? 0) + value);
  }
  let total = 0;
  const cumulative = changes.map(value => (total += value));
  return { changes, cumulative, agencies, categories, agencyCategories, total };
}

export function monthlySeries(snapshots, cumulative, cutoff = snapshots.length - 1) {
  const first = new Map();
  const latest = new Map();
  for (let i = 0; i <= cutoff; i++) {
    const month = snapshots[i].date.slice(0, 7);
    if (!first.has(month)) first.set(month, i);
    latest.set(month, i);
  }
  if (!latest.size) return [];
  const year = snapshots[0].date.slice(0, 4);
  const lastMonth = Number(snapshots[cutoff].date.slice(5, 7));
  const rows = [];
  for (let month = 1; month <= lastMonth; month++) {
    const key = `${year}-${String(month).padStart(2, '0')}`;
    const start = first.get(key);
    const end = latest.get(key);
    const date = end === undefined ? `${key}-01` : snapshots[end].date;
    // The last snapshot before the month is the baseline. Otherwise a month
    // with just one snapshot would incorrectly show zero despite a change.
    const baseline = start === undefined ? null : Math.max(0, start - 1);
    const startDate = baseline === null ? null : snapshots[baseline].date;
    const value = end === undefined ? null : cumulative[end] - cumulative[baseline];
    const coverage = end === undefined ? 'Нет снимков за месяц' : baseline === end ?
      `Первый снимок года (${date}) — базовый итог; изменение 0` :
      `Разница снимков ${startDate} — ${date}`;
    rows.push({ date, label: key, value, coverage, startDate });
  }
  return rows;
}

export function categorySeriesRows(pack, metric, snapshots, filters, institutionGroups, categoryIds, mode) {
  const dimension = metric === 'applications' ? 'caseType' : 'marker';
  const series = categoryIds.map(id => {
    const result = aggregatePack(pack, metric, snapshots, { ...filters, [dimension]: String(id) }, institutionGroups);
    return { id: String(id), result, monthly: mode === 'month' ? monthlySeries(snapshots, result.cumulative) : null };
  });
  const points = mode === 'month' ? series[0]?.monthly ?? [] : snapshots;
  return points.map((point, index) => ({
    date: point.date,
    label: point.label ?? point.date,
    startDate: mode === 'month' ? point.startDate : null,
    values: Object.fromEntries(series.map(({ id, result, monthly }) => [id,
      mode === 'month' ? monthly[index].value : mode === 'snapshot' ? index === 0 ? null : result.changes[index] : result.cumulative[index]])),
    coverage: mode === 'month' ? point.coverage : mode === 'snapshot'
      ? index === 0 ? 'Первый снимок года — базовый итог' : `С предыдущего снимка ${snapshots[index - 1].date}`
      : 'Накопительный итог',
  }));
}

export function decisionSeriesRows(pack, snapshots, filters, institutionGroups, markerIds, mode) {
  return categorySeriesRows(pack, 'decisions', snapshots, filters, institutionGroups, markerIds, mode);
}

export function selectedSeriesStats(values, keys) {
  const selected = keys.map(key => values[key]);
  if (!selected.length || selected.some(value => !Number.isFinite(value))) {
    return { total: null, percentages: Object.fromEntries(keys.map(key => [key, null])) };
  }
  const total = selected.reduce((sum, value) => sum + value, 0);
  const allZero = selected.every(value => value === 0);
  return { total, percentages: Object.fromEntries(keys.map((key, index) => [key,
    total === 0 ? allZero ? 0 : null : selected[index] / total * 100])) };
}

export function visibleSeriesTotals(rows, keys, cumulative, baseline = null) {
  return Object.fromEntries(keys.map(key => {
    const values = rows.map(row => row.values[key]).filter(Number.isFinite);
    if (!values.length) return [key, null];
    const baselineValue = baseline?.values[key];
    return [key, cumulative ? values.at(-1) - (Number.isFinite(baselineValue) ? baselineValue : 0) :
      values.reduce((sum, value) => sum + value, 0)];
  }));
}

export function visibleChartRows(rows, range = 'all', zoom = null) {
  if (!rows.length) return [];
  let visible = rows;
  if (range !== 'all') {
    const cutoff = Date.parse(`${rows.at(-1).date}T00:00:00Z`) - Number(range) * 86400000;
    visible = rows.filter(row => Date.parse(`${row.date}T00:00:00Z`) >= cutoff);
  }
  return zoom ? visible.slice(zoom[0], zoom[1] + 1) : visible;
}

export function chartCsv(rows, series, period = 'date') {
  const monthly = period === 'month';
  const data = [[period, ...series.map(item => item.label)],
    ...rows.map(row => [monthly ? row.label ?? row.date.slice(0, 7) : row.date,
      ...series.map(item => row.values[item.key] ?? '')])];
  return '\ufeff' + data.map(row => row.map(value => `"${String(value).replaceAll('"', '""')}"`).join(',')).join('\r\n');
}

export function sumMaps(target, source) {
  for (const [key, value] of source) target.set(key, (target.get(key) ?? 0) + value);
  return target;
}

export function sumNestedMaps(target, source) {
  for (const [key, values] of source) {
    if (!target.has(key)) target.set(key, new Map());
    sumMaps(target.get(key), values);
  }
  return target;
}

export function selectedAgencyTotals(agencyCategories, categoryIds) {
  const selected = categoryIds.map(Number);
  return new Map([...agencyCategories].map(([institution, categories]) => [institution,
    selected.reduce((total, id) => total + (categories.get(id) ?? 0), 0)]));
}
