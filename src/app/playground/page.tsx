'use client';
import { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import Papa from 'papaparse';
import {
  ArrowRight,
  Download,
  FlaskConical,
  LoaderCircle,
  Play,
  RotateCcw,
  Upload,
  CheckCircle2,
  AlertTriangle,
  Eye,
  Cpu,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Explain, Note, PageHeading } from '@/components/common';
import { groups, details, type Values, type Sample } from '@/lib/features';
type ModelResult = { score: number; prediction: number; threshold: number; trees: number };
type Results = Record<'xgboost' | 'randomforest', ModelResult>;
const featureNames = groups.flatMap((g) => g.fields);
function download(name: string, text: string, type: string) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}
export default function Playground() {
  const [samples, setSamples] = useState<Sample[]>([]),
    [sample, setSample] = useState<Sample | null>(null),
    [values, setValues] = useState<Values>({});
  const [group, setGroup] = useState('connection'),
    [results, setResults] = useState<Results | null>(null),
    [milliseconds, setMilliseconds] = useState(0),
    [ready, setReady] = useState(false),
    [pending, setPending] = useState(false),
    [error, setError] = useState(''),
    [revealed, setRevealed] = useState(false),
    [importNote, setImportNote] = useState('');
  const [categories, setCategories] = useState<Record<string, string[]>>({});
  const worker = useRef<Worker | null>(null),
    file = useRef<HTMLInputElement>(null),
    requestId = useRef(0);
  const edited = sample ? JSON.stringify(values) !== JSON.stringify(sample.values) : false;
  const invalid = featureNames.filter(
    (name) =>
      !['proto', 'service', 'state'].includes(name) &&
      values[name] !== '' &&
      values[name] !== null &&
      values[name] !== undefined &&
      (!Number.isFinite(Number(values[name])) || Number(values[name]) < 0),
  );
  function choose(record: Sample) {
    setSample(record);
    setValues({ ...record.values });
    setResults(null);
    setRevealed(false);
    setError('');
    setImportNote('');
  }
  useEffect(() => {
    let active = true;
    Promise.all([
      fetch('/data/samples.json').then((r) => {
        if (!r.ok) throw Error('Sample records could not be loaded.');
        return r.json();
      }),
      fetch('/models/manifest.json').then((r) => {
        if (!r.ok) throw Error('Model details could not be loaded.');
        return r.json();
      }),
    ])
      .then(([records, manifest]) => {
        if (!active) return;
        setSamples(records);
        setCategories(
          Object.fromEntries(
            manifest.categorical.map((name: string, i: number) => [name, manifest.categories[i]]),
          ),
        );
        let selected = records[0];
        try {
          const saved = sessionStorage.getItem('flowlens-record');
          if (saved) {
            selected = JSON.parse(saved);
            sessionStorage.removeItem('flowlens-record');
          }
        } catch {}
        choose(selected);
      })
      .catch((e) => active && setError(e.message));
    const w = new Worker('/model-worker.js', { type: 'module' });
    worker.current = w;
    w.onmessage = ({ data }) => {
      if (!active) return;
      if (data.id === 0) {
        if (data.error) setError(data.error);
        else setReady(true);
        return;
      }
      if (data.id !== requestId.current) return;
      setPending(false);
      if (data.error) {
        setError(data.error);
        return;
      }
      setReady(true);
      setResults(data.results);
      setMilliseconds(data.milliseconds);
    };
    w.onerror = () => {
      if (active) {
        setError('The model worker could not start. Reload the page to try again.');
        setPending(false);
      }
    };
    w.postMessage({ id: 0 });
    return () => {
      active = false;
      w.terminate();
    };
  }, []);
  function predict() {
    if (!sample || invalid.length) return;
    setPending(true);
    setError('');
    setRevealed(false);
    const id = ++requestId.current;
    worker.current?.postMessage({ id, values });
  }
  function edit(name: string, value: string) {
    setValues((v) => ({
      ...v,
      [name]:
        value === '' ? '' : ['proto', 'service', 'state'].includes(name) ? value : Number(value),
    }));
    setResults(null);
    setRevealed(false);
  }
  function importCsv(event: React.ChangeEvent<HTMLInputElement>) {
    const f = event.target.files?.[0];
    if (!f) return;
    if (f.size > 2 * 1024 * 1024) {
      setError('For a single-record import, please choose a CSV smaller than 2 MB.');
      event.target.value = '';
      return;
    }
    Papa.parse<Record<string, string>>(f, {
      header: true,
      skipEmptyLines: true,
      complete(parsed) {
        const row = parsed.data[0];
        const missing = featureNames.filter((name) => !row || !(name in row));
        if (parsed.errors.length || missing.length) {
          setError(
            missing.length
              ? `Missing input columns: ${missing.join(', ')}`
              : 'The CSV could not be read. Check its header and values.',
          );
          return;
        }
        const inputs: Values = {};
        for (const name of featureNames) {
          const raw = row[name].trim();
          if (
            !['proto', 'service', 'state'].includes(name) &&
            raw !== '' &&
            (!Number.isFinite(Number(raw)) || Number(raw) < 0)
          ) {
            setError(`Invalid number in ${name}.`);
            return;
          }
          inputs[name] = ['proto', 'service', 'state'].includes(name)
            ? raw
            : raw === ''
              ? ''
              : Number(raw);
        }
        choose({
          id: 'custom',
          title: 'Your imported record',
          source: 'Imported CSV',
          values: inputs,
          truth: null,
          category: 'Unknown',
          recordId: Number(row.id) || 0,
        });
        setImportNote(
          `Loaded the first record from ${f.name}${parsed.data.length > 1 ? ` (${parsed.data.length} rows in file)` : ''}. Answer columns are excluded; no verified answer is assumed.`,
        );
      },
      error(e) {
        setError(e.message);
      },
    });
    event.target.value = '';
  }
  const current = groups.find((g) => g.id === group)!;
  return (
    <div className="container page-content">
      <PageHeading
        eyebrow="EXPERIMENT WITH A CONNECTION"
        title="The playground"
        description="Start with a complete record, change a measurement, and ask both models what they see. Every prediction uses the actual trained trees."
      >
        <span className="pill">
          <Cpu size={13} /> Runs in your browser
        </span>
      </PageHeading>
      <div className="playground-layout">
        <section className="panel input-panel" aria-labelledby="input-title">
          <div className="panel-header">
            <div>
              <h2 id="input-title">Your input measurements</h2>
              <p>One record. The same input for both models.</p>
            </div>
            <span className={`pill ${edited ? 'amber' : ''}`}>
              {edited ? 'Edited record' : '42 features'}
            </span>
          </div>
          <div className="sample-bar">
            <div className="sample-select">
              <label htmlFor="sample">Start with a recorded example</label>
              <select
                id="sample"
                className="select-input"
                value={sample?.id || ''}
                disabled={pending || !samples.length}
                onChange={(e) => {
                  const next = samples.find((s) => s.id === e.target.value);
                  if (next) choose(next);
                }}
              >
                {sample && !samples.some((s) => s.id === sample.id) && (
                  <option value={sample.id}>{sample.title}</option>
                )}
                {['Starter', 'Error case', 'Explore'].map((source) => (
                  <optgroup
                    label={
                      source === 'Error case' ? 'Deliberately selected XGBoost mistakes' : source
                    }
                    key={source}
                  >
                    {samples
                      .filter((s) => s.source === source)
                      .map((s) => (
                        <option value={s.id} key={s.id}>
                          {s.title}
                        </option>
                      ))}
                  </optgroup>
                ))}
              </select>
            </div>
            <Button
              variant="outline"
              size="sm"
              disabled={!sample || pending}
              onClick={() => sample && choose(sample)}
            >
              <RotateCcw size={13} /> Reset
            </Button>
          </div>
          <div className="flex items-center justify-between gap-3">
            <button className="drop-file" disabled={pending} onClick={() => file.current?.click()}>
              <Upload size={12} /> Import a record CSV
            </button>
            <button
              className="drop-file"
              disabled={!sample || pending}
              onClick={() => download('flowlens-input.csv', Papa.unparse([values]), 'text/csv')}
            >
              <Download size={12} /> Export input
            </button>
            <input
              type="file"
              accept=".csv,text/csv"
              hidden
              ref={file}
              onChange={importCsv}
              aria-label="Import a CSV record"
            />
          </div>
          {importNote && <p className="result-detail">{importNote}</p>}
          {sample?.source === 'Training data' && (
            <div className="mt-4">
              <Note>
                This is a training record the models learned from. Its result is not independent
                test evidence.
              </Note>
            </div>
          )}
          <div
            className="group-tabs"
            role="tablist"
            aria-label="Measurement groups"
            onKeyDown={(event) => {
              if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
              event.preventDefault();
              const index = groups.findIndex((g) => g.id === group);
              const next =
                event.key === 'Home'
                  ? 0
                  : event.key === 'End'
                    ? groups.length - 1
                    : (index + (event.key === 'ArrowRight' ? 1 : -1) + groups.length) %
                      groups.length;
              setGroup(groups[next].id);
              event.currentTarget.querySelectorAll<HTMLButtonElement>('button')[next]?.focus();
            }}
          >
            {groups.map((g) => (
              <button
                role="tab"
                tabIndex={group === g.id ? 0 : -1}
                aria-selected={group === g.id}
                aria-controls="measurement-panel"
                id={`tab-${g.id}`}
                key={g.id}
                className={group === g.id ? 'selected' : ''}
                onClick={() => setGroup(g.id)}
              >
                {g.title}
              </button>
            ))}
          </div>
          <div role="tabpanel" id="measurement-panel" aria-labelledby={`tab-${group}`}>
            <div className="group-intro">
              <h3>{current.title}</h3>
              <p>{current.description}</p>
            </div>
            {!sample ? (
              <div className="loading">
                <LoaderCircle className="spin" size={20} /> Loading sample measurements…
              </div>
            ) : (
              <fieldset
                disabled={pending}
                className="fields-grid"
                style={{ border: 0, padding: 0, margin: 0 }}
              >
                {current.fields.map((name) => {
                  const changed = String(values[name] ?? '') !== String(sample.values[name] ?? '');
                  return (
                    <div key={name} className={`measurement-field ${changed ? 'changed' : ''}`}>
                      <label htmlFor={`field-${name}`}>
                        <span>{details[name]?.[0] || name}</span>
                        <Explain>{details[name]?.[1]}</Explain>
                      </label>
                      <Input
                        id={`field-${name}`}
                        name={name}
                        list={categories[name] ? `options-${name}` : undefined}
                        type={categories[name] ? 'text' : 'number'}
                        min={categories[name] ? undefined : 0}
                        step="any"
                        value={values[name] ?? ''}
                        placeholder="Use training default"
                        onChange={(e) => edit(name, e.target.value)}
                        aria-invalid={invalid.includes(name)}
                        aria-describedby={`description-${name}`}
                      />
                      {categories[name] && (
                        <datalist id={`options-${name}`}>
                          {categories[name].map((value) => (
                            <option key={value} value={value} />
                          ))}
                        </datalist>
                      )}
                      <span className="field-code" id={`description-${name}`}>
                        {name}
                        {changed && (
                          <>
                            <span className="changed-dot" /> edited
                          </>
                        )}
                      </span>
                      {invalid.includes(name) && (
                        <div className="field-error">Enter a finite number, zero or greater.</div>
                      )}
                    </div>
                  );
                })}
              </fieldset>
            )}
          </div>
          <div className="input-footer">
            <span>
              {edited
                ? `${featureNames.filter((n) => String(values[n] ?? '') !== String(sample?.values[n] ?? '')).length} measurement(s) changed`
                : 'Original recorded measurements'}
            </span>
            <span>
              ID and answers never enter the model{' '}
              <Explain>
                The record ID is only a reference. label and attack_cat would give away the answer,
                so both are excluded from the inputs.
              </Explain>
            </span>
          </div>
          <details className="explainer">
            <summary>What am I changing here?</summary>
            <p>
              A flow is a summary of a network communication. Source is the sending side;
              destination is the receiving side. Each field is one measurement. Blank numeric values
              use the saved training median; blank categories use the training mode. Unknown
              categories become all-zero indicators.
            </p>
            <p>
              Changing one value is a what-if experiment. Related measurements are not recalculated,
              so arbitrary changes can create an unrealistic communication. An unchanged prediction
              is also a valid result.
            </p>
          </details>
        </section>
        <aside className="result-column">
          <section className="panel" aria-labelledby="result-title">
            <div className="panel-header">
              <div>
                <h2 id="result-title">Two models. One input.</h2>
                <p>Compare their independent decisions.</p>
              </div>
            </div>
            <div className="ready-status">
              {ready ? (
                <>
                  <span className="status-dot" /> Both saved models loaded
                </>
              ) : (
                <>
                  <LoaderCircle size={12} className="spin" /> Loading fitted trees · about 6.8 MB
                  once
                </>
              )}
            </div>
            <div aria-live="polite">
              {results ? (
                <div className="result-model-wrap">
                  {(['xgboost', 'randomforest'] as const).map((name, i) => {
                    const result = results[name];
                    return (
                      <div className={`result-model ${i ? 'rf' : ''}`} key={name}>
                        <div className="result-model-header">
                          <div className="model-name">
                            <span className={`model-orb ${i ? 'teal' : ''}`}>{i ? 'RF' : 'X'}</span>
                            {i ? 'Random Forest' : 'XGBoost'}
                          </div>
                          <span className="pill">{result.trees} trees</span>
                        </div>
                        <div
                          className={`result-decision ${result.prediction ? 'attack' : 'normal'}`}
                        >
                          {result.prediction ? 'Attack' : 'Normal'}
                        </div>
                        <div className="result-sub">
                          {result.prediction
                            ? 'Flagged as potentially suspicious'
                            : 'Classified as ordinary traffic'}
                        </div>
                        <div className="score-row">
                          <span>
                            Attack score{' '}
                            <Explain>
                              A model score, not a calibrated real-world probability. It is compared
                              against this model’s saved cutoff.
                            </Explain>
                          </span>
                          <b>{result.score.toFixed(6)}</b>
                        </div>
                        <div
                          className="score-track"
                          aria-label={`Attack score ${result.score.toFixed(6)}; cutoff ${result.threshold}`}
                        >
                          <div className="score-fill" style={{ width: `${result.score * 100}%` }} />
                          <span
                            className="cutoff-marker"
                            style={{ left: `${result.threshold * 100}%` }}
                          />
                        </div>
                        <div className="score-caption">
                          <span>0 · Normal</span>
                          <span>Cutoff {result.threshold.toFixed(2)}</span>
                          <span>Attack · 1</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="empty-results">
                  <FlaskConical size={30} />
                  <h3>A connection to investigate</h3>
                  <p>
                    Keep the sample as it is, or change a measurement. Then run both models to
                    compare their answers.
                  </p>
                </div>
              )}
            </div>
            {error && (
              <div className="error-message" role="alert">
                {error}
              </div>
            )}
            <Button
              className="run-button"
              size="lg"
              disabled={pending || !sample || !!invalid.length}
              onClick={predict}
            >
              {pending ? (
                <>
                  <LoaderCircle className="spin" size={16} /> Running both models…
                </>
              ) : (
                <>
                  <Play size={15} fill="currentColor" /> {results ? 'Run again' : 'Run both models'}
                </>
              )}
            </Button>
            {results && (
              <div className="flex justify-between items-center mt-3">
                <span className="result-detail" style={{ margin: 0 }}>
                  {milliseconds.toFixed(1)} ms · after loading
                </span>
                <button
                  className="drop-file"
                  onClick={() =>
                    download(
                      'flowlens-prediction.json',
                      JSON.stringify(
                        {
                          inputs: values,
                          results,
                          edited,
                          caveat: 'Uncalibrated benchmark model scores.',
                        },
                        null,
                        2,
                      ),
                      'application/json',
                    )
                  }
                >
                  <Download size={11} /> Save result
                </button>
              </div>
            )}
            <div className="result-caveat">
              A score at or above the model’s cutoff becomes <b>Attack</b>. Scores are not verified
              risk percentages, and predictions can be wrong.
            </div>
          </section>
          <section className="panel truth-panel">
            <h3>How do I check the answer?</h3>
            {edited ? (
              <p>
                Your input has changed. Its actual answer is unknown—the original label cannot
                verify a hypothetical communication.
              </p>
            ) : sample?.truth === null ? (
              <p>
                This imported record has no independently verified answer attached. A model’s own
                prediction cannot check itself.
              </p>
            ) : (
              <>
                <p>
                  The original recorded example has a known answer. Make the predictions first, then
                  reveal it.
                </p>
                <Button
                  size="sm"
                  variant="outline"
                  className="mt-3"
                  disabled={!results}
                  onClick={() => setRevealed(!revealed)}
                >
                  <Eye size={13} /> {revealed ? 'Hide known answer' : 'Reveal known answer'}
                </Button>
                {revealed && sample && (
                  <div className="truth-outcome">
                    <strong>Actual: {sample.truth ? 'Attack' : 'Normal'}</strong>
                    <div className="mt-1 text-xs">Category: {sample.category}</div>
                    {results &&
                      (['xgboost', 'randomforest'] as const).map((name) => (
                        <div key={name} className="mt-2 flex gap-1.5 items-center text-xs">
                          {results[name].prediction === sample.truth ? (
                            <CheckCircle2 size={13} className="text-[var(--teal)]" />
                          ) : (
                            <AlertTriangle size={13} className="text-[var(--amber)]" />
                          )}
                          {name === 'xgboost' ? 'XGBoost' : 'Random Forest'}:{' '}
                          {results[name].prediction === sample.truth ? 'correct' : 'incorrect'}
                        </div>
                      ))}
                  </div>
                )}
              </>
            )}
            {sample?.source === 'Error case' && (
              <p className="mt-3">
                This example was deliberately selected because XGBoost got it wrong. It is not an
                accuracy sample.
              </p>
            )}
            <Link href="/analysis/" className="subtle-link mt-3">
              See the full test results <ArrowRight size={12} />
            </Link>
          </section>
          <Note>
            No inputs leave this browser. The site uses recorded measurements and does not inspect
            your network.
          </Note>
        </aside>
      </div>
    </div>
  );
}
