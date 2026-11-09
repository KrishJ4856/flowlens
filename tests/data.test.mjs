import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
const meta = JSON.parse(readFileSync('public/data/datasets.json'));
for (const partition of ['train', 'test'])
  test(`Complete ${partition} data and filters retain all records and correct labels`, () => {
    const m = meta[partition],
      label = m.columns.indexOf('label'),
      category = m.columns.indexOf('attack_cat');
    assert.equal(m.columns.length, 45);
    for (const [filter, info] of Object.entries(m.filters)) {
      let count = 0;
      for (let chunk = 0; chunk < info.chunks; chunk++) {
        const rows = JSON.parse(
          gunzipSync(readFileSync(`public/data/${partition}/${filter}/${chunk}.json.gz`)),
        );
        for (const row of rows) {
          assert.equal(row.length, 45);
          if (filter === 'all') assert.equal(row[0], count + 1);
          else if (filter === 'Attack') assert.equal(row[label], 1);
          else assert.equal(row[category], filter);
          count++;
        }
      }
      assert.equal(count, info.count);
    }
    assert.equal(m.filters.all.count, partition === 'train' ? 175341 : 82332);
  });
