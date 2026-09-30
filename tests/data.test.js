import test from 'node:test';
import assert from 'node:assert/strict';
import { aggregatePack, categorySeriesRows, chartCsv, decisionSeriesRows, monthlySeries, selectedAgencyTotals, selectedSeriesStats, sumNestedMaps, visibleChartRows, visibleSeriesTotals } from '../web/data.js';

test('filters citizenship shard, institutions, case and result before calculating series', () => {
  const snapshots = [{date:'2026-01-31'},{date:'2026-02-28'}];
  const pack = { decisions: [
    [0, 873, 4, 4, 10], [0, 810, 4, 6, 3],
    [1, 873, 4, 4, 5], [1, 810, 4, 6, -1], [1, 873, 1, 4, 2],
  ] };
  const groups = new Map([[873,'WOJ'],[810,'MIN']]);
  const selection = aggregatePack(pack,'decisions',snapshots,
    {group:'WOJ',institution:'all',caseType:'4',marker:'4'},groups);
  assert.deepEqual(selection.cumulative,[10,15]);
  assert.equal(selection.agencies.get(873),15);
  assert.equal(selection.categories.get(4),15);
});

test('monthly values compare with the last snapshot before the month', () => {
  const points = ['2023-02-10','2023-02-28','2023-03-31','2023-06-10','2023-06-29']
    .map(date => ({date}));
  const months = monthlySeries(points,[10,20,30,40,70]);
  assert.equal(months[1].value,10);
  assert.equal(months[2].value,10);
  assert.deepEqual(months.slice(3,5).map(row => row.value),[null,null]);
  assert.equal(months[5].value,40);
  assert.equal(months[5].coverage,'Разница снимков 2023-03-31 — 2023-06-29');
});

test('negative corrections remain visible in monthly changes', () => {
  const points = [{date:'2026-01-05'},{date:'2026-01-31'}];
  assert.equal(monthlySeries(points,[100,96])[0].value,-4);
});

test('early month snapshots still produce bars with their actual comparison dates', () => {
  const points = ['2026-04-02','2026-04-30','2026-05-01','2026-05-06','2026-06-07','2026-06-10',
    '2026-07-16','2026-07-31','2026-08-01','2026-08-31','2026-09-08','2026-09-30']
    .map(date => ({date}));
  const months = monthlySeries(points,[90,100,105,110,112,115,1715,1815,1820,1915,1920,2015]);
  assert.deepEqual(months.slice(3).map(row => row.value), [10,10,5,1700,100,100]);
  assert.equal(months[5].coverage, 'Разница снимков 2026-05-06 — 2026-06-10');
  assert.equal(months[6].coverage, 'Разница снимков 2026-06-10 — 2026-07-31');
});

test('a single September snapshot includes its change since August', () => {
  const points = ['2025-08-27', '2025-09-11'].map(date => ({date}));
  const rows = monthlySeries(points, [86, 90]);
  assert.equal(rows[8].value, 4);
  assert.equal(rows[8].startDate, '2025-08-27');
  assert.equal(rows[8].coverage, 'Разница снимков 2025-08-27 — 2025-09-11');
});

test('two decision results retain independent values at the same snapshots', () => {
  const points = [{date:'2026-01-01'},{date:'2026-01-03'},{date:'2026-01-06'}];
  const pack = { decisions: [
    [0, 810, 4, 4, 10], [0, 810, 4, 11, 3],
    [1, 810, 4, 4, 2], [1, 810, 4, 11, 5],
    [2, 810, 4, 4, -1], [2, 810, 4, 11, 4],
    [2, 873, 4, 4, 100],
  ] };
  const filters = {group:'all', institution:'810', caseType:'4', marker:'all'};
  const groups = new Map([[810,'MIN'],[873,'WOJ']]);
  const snapshot = decisionSeriesRows(pack, points, filters, groups, ['4','11'], 'snapshot');
  assert.deepEqual(snapshot.map(row => row.values), [
    {'4':null,'11':null}, {'4':2,'11':5}, {'4':-1,'11':4},
  ]);
  assert.equal(snapshot[2].coverage, 'С предыдущего снимка 2026-01-03');
  const cumulative = decisionSeriesRows(pack, points, filters, groups, ['4','11'], 'cumulative');
  assert.deepEqual(cumulative[2].values, {'4':11,'11':12});
  const monthly = decisionSeriesRows(pack, points, filters, groups, ['4','11'], 'month');
  assert.deepEqual(monthly[0].values, {'4':1,'11':9});
});

test('application case types stay separate across snapshots and months', () => {
  const points = ['2026-01-01','2026-01-10','2026-02-01','2026-02-28'].map(date => ({date}));
  const pack = { applications: [
    [0, 873, 1, 10], [0, 873, 2, 4], [0, 3201, 1, 100],
    [1, 873, 1, 3], [1, 873, 2, -1],
    [2, 873, 1, 20], [2, 873, 2, 2],
    [3, 873, 1, -2], [3, 873, 2, 6],
  ] };
  const filters = {group:'WOJ', institution:'all', caseType:'all', marker:'all'};
  const groups = new Map([[873,'WOJ'],[3201,'PSG']]);
  const rows = mode => categorySeriesRows(pack,'applications',points,filters,groups,['1','2'],mode);
  assert.deepEqual(rows('cumulative')[3].values, {'1':31,'2':11});
  assert.deepEqual(rows('snapshot').map(row => row.values), [
    {'1':null,'2':null}, {'1':3,'2':-1}, {'1':20,'2':2}, {'1':-2,'2':6},
  ]);
  assert.deepEqual(rows('month').map(row => row.values), [
    {'1':3,'2':-1}, {'1':18,'2':8},
  ]);
});

test('agency comparison uses the sum of selected categories for decisions and applications', () => {
  const points = [{date:'2026-01-31'}];
  const groups = new Map([[810,'MIN'],[873,'WOJ']]);
  const filters = {group:'all', institution:'all', caseType:'all', marker:'all'};
  const decisions = aggregatePack({decisions:[
    [0,810,4,4,10], [0,810,4,6,2], [0,873,4,4,3], [0,873,4,6,20],
  ]},'decisions',points,filters,groups);
  assert.deepEqual([...selectedAgencyTotals(decisions.agencyCategories,['4'])], [[810,10],[873,3]]);
  assert.deepEqual([...selectedAgencyTotals(decisions.agencyCategories,['4','6'])], [[810,12],[873,23]]);

  const applications = aggregatePack({applications:[
    [0,810,1,7], [0,810,2,1], [0,873,1,5], [0,873,2,4],
  ]},'applications',points,filters,groups);
  const merged = sumNestedMaps(new Map(), applications.agencyCategories);
  sumNestedMaps(merged, applications.agencyCategories);
  assert.deepEqual([...selectedAgencyTotals(merged,['1','2'])], [[810,16],[873,18]]);
});

test('percentages use only selected results and handle zero or unavailable totals', () => {
  const selected = selectedSeriesStats({'4':66,'11':95,'8':168,'6':1156,'1':177}, ['4','11','8','6']);
  assert.equal(selected.total,1485);
  assert.equal(Math.round(selected.percentages['6'] * 10) / 10,77.8);
  assert.equal(Object.hasOwn(selected.percentages,'1'),false);
  assert.deepEqual(selectedSeriesStats({'4':0,'6':0},['4','6']),
    {total:0,percentages:{'4':0,'6':0}});
  assert.deepEqual(selectedSeriesStats({'4':null,'6':5},['4','6']),
    {total:null,percentages:{'4':null,'6':null}});
});

test('legend totals sum visible changes but use the final cumulative value', () => {
  const changes = [
    {values:{positive:null,negative:null}},
    {values:{positive:3,negative:5}},
    {values:{positive:-1,negative:0}},
  ];
  assert.deepEqual(visibleSeriesTotals(changes,['positive','negative'],false), {positive:2,negative:5});
  assert.deepEqual(visibleSeriesTotals(changes.slice(1,2),['positive','negative'],false), {positive:3,negative:5});
  assert.deepEqual(visibleSeriesTotals(changes.slice(0,1),['positive','negative'],false), {positive:null,negative:null});
  const cumulative = [{values:{positive:10}}, {values:{positive:13}}, {values:{positive:12}}];
  assert.deepEqual(visibleSeriesTotals(cumulative,['positive'],true), {positive:12});
  assert.deepEqual(visibleSeriesTotals(cumulative.slice(1),['positive'],true,cumulative[0]), {positive:2});
  assert.deepEqual(visibleSeriesTotals(cumulative.slice(1,2),['positive'],true,cumulative[1]), {positive:0});
});

test('CSV contains only selected series and dates inside the visible chart range', () => {
  const rows = [
    {date:'2026-05-31',values:{positive:1,negative:2}},
    {date:'2026-07-10',values:{positive:2,negative:3}},
    {date:'2026-08-31',values:{positive:3,negative:4}},
    {date:'2026-09-10',values:{positive:5,negative:6}},
    {date:'2026-09-30',values:{positive:7,negative:8}},
  ];
  assert.deepEqual(visibleChartRows(rows, '30').map(row => row.date), ['2026-08-31','2026-09-10','2026-09-30']);
  assert.deepEqual(visibleChartRows(rows, '90').map(row => row.date), ['2026-07-10','2026-08-31','2026-09-10','2026-09-30']);
  assert.deepEqual(visibleChartRows(rows, 'all', [2,3]).map(row => row.date), ['2026-08-31','2026-09-10']);
  const csv = chartCsv(visibleChartRows(rows, 'all', [2,3]), [{key:'positive',label:'Pozytywna'}]);
  assert.equal(csv, '\ufeff"date","Pozytywna"\r\n"2026-08-31","3"\r\n"2026-09-10","5"');
});

test('monthly CSV uses calendar months rather than last snapshot dates', () => {
  const rows = [
    {date:'2026-01-31',label:'2026-01',values:{positive:5}},
    {date:'2026-02-27',label:'2026-02',values:{positive:9}},
    {date:'2026-03-01',label:'2026-03',values:{positive:null}},
  ];
  assert.equal(chartCsv(rows, [{key:'positive',label:'Pozytywna'}], 'month'),
    '\ufeff"month","Pozytywna"\r\n"2026-01","5"\r\n"2026-02","9"\r\n"2026-03",""');
});
