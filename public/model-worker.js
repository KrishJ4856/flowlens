import { prepare, predictEncoded } from './inference.mjs';
let loaded;
async function gunzip(url) {
  const response = await fetch(url);
  if (!response.ok)
    throw new Error(
      `Could not load model (${response.status}). Check your connection and try again.`,
    );
  if (!response.body || typeof DecompressionStream === 'undefined')
    throw new Error(
      'This browser does not support model decompression. Please use a recent Chrome, Edge, Firefox or Safari.',
    );
  return new Response(response.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
}
async function load() {
  const response = await fetch('/models/manifest.json');
  if (!response.ok) throw new Error('Could not load the model manifest.');
  const manifest = await response.json();
  const names = ['xgboost', 'randomforest'];
  const buffers = await Promise.all(
    names.map(async (name) => {
      const buffer = await gunzip(`/models/${name}.bin.gz`);
      const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', buffer)))
        .map((b) => b.toString(16).padStart(2, '0'))
        .join('');
      if (digest !== manifest.models[name].binarySha256)
        throw new Error(`The ${name} model file failed its integrity check. Reload the page.`);
      return new DataView(buffer);
    }),
  );
  return { manifest, forests: Object.fromEntries(names.map((name, i) => [name, buffers[i]])) };
}
self.onmessage = async (event) => {
  const { id, values } = event.data;
  try {
    loaded ??= load().catch((error) => {
      loaded = undefined;
      throw error;
    });
    const { manifest, forests } = await loaded;
    if (!values) {
      self.postMessage({ id, ready: true });
      return;
    }
    const start = performance.now();
    const results = predictEncoded(prepare(values, manifest), manifest, forests);
    self.postMessage({ id, results, milliseconds: performance.now() - start });
  } catch (error) {
    self.postMessage({ id, error: error.message || 'Prediction failed. Please try again.' });
  }
};
