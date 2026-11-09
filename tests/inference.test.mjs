import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { createHash } from 'node:crypto';
import { prepare,predictEncoded } from '../public/inference.mjs';
const manifest=JSON.parse(readFileSync('public/models/manifest.json'));
const forests=Object.fromEntries(['xgboost','randomforest'].map(name=>{const data=gunzipSync(readFileSync(`public/models/${name}.bin.gz`));return [name,new DataView(data.buffer,data.byteOffset,data.byteLength)];}));
const fixtures=JSON.parse(readFileSync('tests/fixtures/python-predictions.json'));
for(const example of fixtures)test(`Python parity: ${example.name}`,()=>{
 const result=predictEncoded(prepare(example.values,manifest),manifest,forests);
 for(const name of ['xgboost','randomforest']){
  assert.equal(result[name].prediction,example.expected[name].prediction);
  assert.ok(Math.abs(result[name].score-example.expected[name].score)<(name==='xgboost'?1e-6:1e-12));
 }
});
test('Fitted binary exports match their recorded checksums',()=>{
 for(const name of ['xgboost','randomforest']){
  const data=gunzipSync(readFileSync(`public/models/${name}.bin.gz`));
  assert.equal(createHash('sha256').update(data).digest('hex'),manifest.models[name].binarySha256);
 }
});
test('Answers, ID and input order cannot change the prepared features',()=>{
 const original=fixtures[0].values;
 const reordered=Object.fromEntries(Object.entries({...original,id:-1,label:1,attack_cat:'Attack'}).reverse());
 assert.deepEqual(prepare(original,manifest),prepare(reordered,manifest));
});
test('A required column cannot be silently omitted',()=>{const input={...fixtures[0].values};delete input.dur;assert.throws(()=>prepare(input,manifest),/Missing measurement: dur/);});
test('Bad numeric inputs fail clearly',()=>{
 for(const value of ['abc',-1,Infinity])assert.throws(()=>prepare({...fixtures[0].values,dur:value},manifest),/dur needs/);
});
test('Full-test parity report contains zero decision mismatches',()=>{
 const r=JSON.parse(readFileSync('public/data/model-parity.json'));assert.equal(r.records,82332);assert.equal(r.predictionMismatches.xgboost,0);assert.equal(r.predictionMismatches.randomforest,0);
});
