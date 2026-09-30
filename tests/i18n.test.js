import test from 'node:test';
import assert from 'node:assert/strict';
import { languages, localeTags, messages, translate } from '../web/i18n.js';

test('interface translations cover the same keys in every supported language', () => {
  assert.deepEqual(languages, ['by', 'pl', 'ukr', 'en', 'ru']);
  const keys = Object.keys(messages.ru).sort();
  for (const language of languages) {
    assert.ok(localeTags[language]);
    assert.deepEqual(Object.keys(messages[language]).sort(), keys);
    for (const key of keys) {
      assert.ok(messages[language][key].trim(), `${language}.${key}`);
      const placeholders = value => [...value.matchAll(/\{(\w+)\}/g)].map(match => match[1]).sort();
      assert.deepEqual(placeholders(messages[language][key]), placeholders(messages.ru[key]), `${language}.${key} placeholders`);
    }
  }
});

test('translated descriptions interpolate dates without changing source values', () => {
  assert.equal(translate('en', 'monthDifference', { start: '2026-09-08', end: '2026-09-30' }),
    'Difference between snapshots 2026-09-08 — 2026-09-30');
  assert.equal(translate('pl', 'yearOption', { year: 2026 }), '2026 r.');
  assert.equal(translate('by', 'monthDifference', { start: '2025-08-27', end: '2025-09-11' }),
    'Розніца паміж зрэзамі 2025-08-27 — 2025-09-11');
  assert.equal(translate('ukr', 'monthDifference', { start: '2025-08-27', end: '2025-09-11' }),
    'Різниця між зрізами 2025-08-27 — 2025-09-11');
  assert.equal(translate('unknown', 'loading'), messages.by.loading);
});
