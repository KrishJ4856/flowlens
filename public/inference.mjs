// Inference only. These functions traverse the original fitted trees; no training occurs.
export function prepare(values, manifest) {
  const result = new Float32Array(194);
  for (let i = 0; i < manifest.numeric.length; i++) {
    const name = manifest.numeric[i];
    const value = values[name];
    if (value === undefined) throw new Error(`Missing measurement: ${name}`);
    const number = value === null || value === '' ? manifest.medians[i] : Number(value);
    if (!Number.isFinite(number) || number < 0) throw new Error(`${name} needs a finite, non-negative number, or a blank value.`);
    result[i] = number;
  }
  let offset = manifest.numeric.length;
  for (let i = 0; i < manifest.categorical.length; i++) {
    const name = manifest.categorical[i];
    if (values[name] === undefined) throw new Error(`Missing measurement: ${name}`);
    const raw = values[name];
    const value = raw === null || raw === '' ? manifest.categoryModes[i] : String(raw).trim();
    const position = manifest.categories[i].indexOf(value);
    if (position >= 0) result[offset + position] = 1;
    offset += manifest.categories[i].length;
  }
  return result;
}
export function predictEncoded(vector, manifest, forests) {
  const results = {};
  for (const name of ['xgboost', 'randomforest']) {
    const config = manifest.models[name], nodes = forests[name];
    let sum = config.baseMargin;
    for (const root of config.roots) {
      let node = root;
      while (true) {
        const at = node * 20, feature = nodes.getInt32(at + 8, true);
        if (feature < 0) {
          const leaf = nodes.getFloat64(at + 12, true);
          sum = name === 'xgboost' ? Math.fround(sum + leaf) : sum + leaf;
          break;
        }
        const threshold = nodes.getFloat64(at + 12, true);
        const left = name === 'xgboost' ? vector[feature] < threshold : vector[feature] <= threshold;
        node = nodes.getInt32(at + (left ? 0 : 4), true);
      }
    }
    const score = name === 'xgboost' ? Math.fround(1 / (1 + Math.exp(-sum))) : sum / config.roots.length;
    results[name] = { score, prediction: Number(score >= config.threshold), threshold: config.threshold, trees: config.treeCount };
  }
  return results;
}
