import { readFileSync, writeFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { predictEncoded, prepare } from '../public/inference.mjs';
const manifest = JSON.parse(readFileSync('public/models/manifest.json'));
const forests = Object.fromEntries(['xgboost','randomforest'].map(name => {
 const b=gunzipSync(readFileSync(`public/models/${name}.bin.gz`));
 return [name,new DataView(b.buffer,b.byteOffset,b.byteLength)];
}));
const directory = process.argv[2] || '.cache/model-verification';
const input = readFileSync(`${directory}/test-encoded.bin`);
const matrix = new Float32Array(input.buffer,input.byteOffset,input.byteLength/4);
const expected = JSON.parse(readFileSync(`${directory}/expected.json`));
const errors={xgboost:0,randomforest:0}, mismatches={xgboost:0,randomforest:0};
for(let row=0;row<expected.xgboost.length;row++){
 const result=predictEncoded(matrix.subarray(row*194,(row+1)*194),manifest,forests);
 for(const name of Object.keys(errors)){
  errors[name]=Math.max(errors[name],Math.abs(result[name].score-expected[name][row]));
  mismatches[name]+=Number(result[name].prediction !== Number(expected[name][row]>=manifest.models[name].threshold));
 }
}
const report={records:expected.xgboost.length,maxAbsoluteScoreError:errors,predictionMismatches:mismatches,tolerance:{xgboost:1e-6,randomforest:1e-12},scope:'All frozen final test records. Browser inference uses exported fitted trees; no model was retrained.'};
console.log(JSON.stringify(report,null,2));
writeFileSync('public/data/model-parity.json',JSON.stringify(report,null,2));
if(mismatches.xgboost||mismatches.randomforest||errors.xgboost>1e-6||errors.randomforest>1e-12)process.exit(1);
