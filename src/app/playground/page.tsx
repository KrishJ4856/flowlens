'use client';
import { useState, useEffect, useRef } from 'react';
import Papa from 'papaparse';
import {
  Download,
  LoaderCircle,
  Play,
  RotateCcw,
  Upload,
  CheckCircle2,
  AlertTriangle,
  Eye,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Explain, PageHeading } from '@/components/common';
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
    [ready, setReady] = useState(false),
    [pending, setPending] = useState(false),
    [error, setError] = useState(''),
    [revealed, setRevealed] = useState(false),
    [importNote, setImportNote] = useState('');
  const [categories, setCategories] = useState<Record<string, string[]>>({});
  const worker = useRef<Worker | null>(null),
    file = useRef<HTMLInputElement>(null),
    requestId = useRef(0);
  const resultsRef = useRef<HTMLElement>(null);
  const edited = sample ? JSON.stringify(values) !== JSON.stringify(sample.values) : false;
  const invalid = featureNames.filter(
    (name) =>
      !['proto', 'service', 'state'].includes(name) &&
      values[name] !== '' &&
      values[name] !== null &&
      values[name] !== undefined &&
      (!Number.isFinite(Math.fround(Number(values[name]))) || Number(values[name]) < 0),
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
    if (window.matchMedia('(max-width: 640px)').matches) {
      resultsRef.current?.scrollIntoView({
        behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth',
        block: 'start',
      });
    }
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
            (!Number.isFinite(Math.fround(Number(raw))) || Number(raw) < 0)
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
    <div className="container page-content playground-page">
      <PageHeading
        title="Playground"
        description="Choose a record. Change a value. Compare the models."
      />
      <div className="mobile-run">
        <Button size="lg" disabled={pending || !sample || !!invalid.length} onClick={predict}>
          {pending ? (
            <>
              <LoaderCircle className="spin" size={16} /> Running…
            </>
          ) : (
            <>
              <Play size={15} fill="currentColor" />
              {results ? 'Run again' : 'Run both models'}
            </>
          )}
        </Button>
      </div>
      <div className="playground-layout">
        <section className="panel input-panel" aria-label="Input measurements">
          <div className="sample-bar">
            <div className="sample-select">
              <label htmlFor="sample">Example record</label>
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
                    label={source === 'Error case' ? 'Selected model mistakes' : source}
                    key={source}
                  >
                    {samples
                      .filter((s) => s.source === source)
                      .map((s) => (
                        <option key={s.id} value={s.id}>
                          {s.title}
                        </option>
                      ))}
                  </optgroup>
                ))}
              </select>
            </div>
            <Button
              variant="outline"
              size="icon"
              aria-label="Reset measurements"
              disabled={!sample || pending}
              onClick={() => sample && choose(sample)}
            >
              <RotateCcw size={16} />
            </Button>
          </div>
          <div className="input-status">
            <span>{edited ? 'Edited record' : 'Original record'}</span>
            <span>
              42 measurements{' '}
              <Explain>
                ID, label and attack_cat are excluded. Only the 42 network measurements enter the
                models.
              </Explain>
            </span>
          </div>
          {importNote && <p className="inline-note">{importNote}</p>}
          {sample?.source === 'Training data' && (
            <p className="inline-note">Training example: this is not an independent test.</p>
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
            {groups.map((g, i) => (
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
                {['Connection', 'Packets & data', 'Timing', 'Activity'][i]}
              </button>
            ))}
          </div>
          <div role="tabpanel" id="measurement-panel" aria-labelledby={`tab-${group}`}>
            <p className="group-description">{current.description}</p>
            {!sample ? (
              <div className="loading">
                <LoaderCircle className="spin" size={18} /> Loading record…
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
                      <div className="measurement-label">
                        <label htmlFor={`field-${name}`}>
                          {details[name]?.[0] || name}
                          {changed && <span className="changed-dot" aria-label="edited" />}
                        </label>
                        <Explain>
                          <b>{name}</b>
                          <br />
                          {details[name]?.[1]}
                        </Explain>
                      </div>
                      <Input
                        id={`field-${name}`}
                        name={name}
                        list={categories[name] ? `options-${name}` : undefined}
                        type={categories[name] ? 'text' : 'number'}
                        min={categories[name] ? undefined : 0}
                        step="any"
                        value={values[name] ?? ''}
                        placeholder="Training default"
                        onChange={(e) => edit(name, e.target.value)}
                        aria-invalid={invalid.includes(name)}
                      />
                      {categories[name] && (
                        <datalist id={`options-${name}`}>
                          {categories[name].map((value) => (
                            <option key={value} value={value} />
                          ))}
                        </datalist>
                      )}
                      {invalid.includes(name) && (
                        <div className="field-error">
                          Use a non-negative number in the supported range.
                        </div>
                      )}
                    </div>
                  );
                })}
              </fieldset>
            )}
          </div>
          <div className="input-footer">
            <button className="text-link" disabled={pending} onClick={() => file.current?.click()}>
              <Upload size={14} /> Import CSV
            </button>
            <button
              className="text-link"
              disabled={!sample || pending}
              onClick={() => download('flowlens-input.csv', Papa.unparse([values]), 'text/csv')}
            >
              <Download size={14} /> Export CSV
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
          <details className="explainer">
            <summary>How to use these measurements</summary>
            <p>
              Source is the sending side; destination is the receiving side. Hover or tap an info
              icon for a field’s name and meaning. Blank values use the saved training defaults.
            </p>
            <p>
              Edits create a what-if record. Related fields are not recalculated, and its true
              answer is unknown. Importing a CSV loads its first row and requires all 42 measurement
              columns.
            </p>
          </details>
        </section>
        <aside className="result-column" ref={resultsRef}>
          <section className="prediction-panel" aria-labelledby="result-title">
            <div className="prediction-heading">
              <h2 id="result-title">Predictions</h2>
              <span className="ready-status">
                {ready ? (
                  <>
                    <span className="status-dot" /> Ready
                  </>
                ) : (
                  <>
                    <LoaderCircle className="spin" size={12} /> Loading models
                  </>
                )}
              </span>
            </div>
            <Button
              className="run-button"
              size="lg"
              disabled={pending || !sample || !!invalid.length}
              onClick={predict}
            >
              {pending ? (
                <>
                  <LoaderCircle className="spin" size={16} /> Running…
                </>
              ) : (
                <>
                  <Play size={15} fill="currentColor" />
                  {results ? 'Run again' : 'Run both models'}
                </>
              )}
            </Button>
            {error && (
              <div className="error-message" role="alert">
                {error}
              </div>
            )}
            <div className="result-model-wrap" aria-live="polite">
              {(['xgboost', 'randomforest'] as const).map((name, i) => {
                const result = results?.[name];
                return (
                  <div className={`result-model ${i ? 'rf' : ''}`} key={name}>
                    <div className="result-model-header">
                      <span className="model-name">
                        <span className={`model-marker ${i ? 'rf' : ''}`} />
                        {i ? 'Random Forest' : 'XGBoost'}
                      </span>
                      <Explain>
                        {i
                          ? '200 randomized decision trees, averaged together.'
                          : '700 trees built progressively.'}{' '}
                        Score at or above {i ? '0.55' : '0.50'} becomes Attack.
                      </Explain>
                    </div>
                    <div
                      className={`result-decision ${result ? (result.prediction ? 'attack' : 'normal') : 'waiting'}`}
                    >
                      {result ? (result.prediction ? 'Attack' : 'Normal') : '—'}
                    </div>
                    <div className="score-row">
                      <span>
                        Attack score{' '}
                        <Explain>
                          A model output, not a calibrated risk percentage. The marker shows the
                          cutoff for an Attack decision.
                        </Explain>
                      </span>
                      <b>{result ? result.score.toFixed(6) : '—'}</b>
                    </div>
                    <div
                      className="score-track"
                      aria-label={
                        result
                          ? `Attack score ${result.score.toFixed(6)}; cutoff ${result.threshold}`
                          : 'Waiting for a prediction'
                      }
                    >
                      <div
                        className="score-fill"
                        style={{ width: `${(result?.score || 0) * 100}%` }}
                      />
                      <span
                        className="cutoff-marker"
                        style={{ left: `${(i ? 0.55 : 0.5) * 100}%` }}
                      />
                    </div>
                    <div className="score-caption">
                      <span>Normal</span>
                      <span>Cutoff {i ? '0.55' : '0.50'}</span>
                      <span>Attack</span>
                    </div>
                  </div>
                );
              })}
            </div>
            <div className="answer-check">
              {edited ? (
                <span className="inline-note">Edited input · actual answer unknown</span>
              ) : sample?.truth === null ? (
                <span className="inline-note">Imported input · no answer key</span>
              ) : (
                <>
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={!results}
                    onClick={() => setRevealed(!revealed)}
                  >
                    <Eye size={14} />
                    {revealed ? 'Hide answer' : 'Check known answer'}
                  </Button>
                  {revealed && sample && (
                    <div className="truth-outcome">
                      <div>
                        <span>Actual</span>
                        <b>{sample.truth ? 'Attack' : 'Normal'}</b>
                        <small>{sample.category}</small>
                      </div>
                      {results &&
                        (['xgboost', 'randomforest'] as const).map((name) => (
                          <span className="answer-verdict" key={name}>
                            {results[name].prediction === sample.truth ? (
                              <CheckCircle2 size={14} />
                            ) : (
                              <AlertTriangle size={14} />
                            )}{' '}
                            {name === 'xgboost' ? 'XGBoost' : 'Random Forest'} ·{' '}
                            {results[name].prediction === sample.truth ? 'correct' : 'incorrect'}
                          </span>
                        ))}
                    </div>
                  )}
                </>
              )}
              {sample?.source === 'Error case' && (
                <span className="inline-note">Deliberately selected XGBoost mistake.</span>
              )}
            </div>
            <div className="prediction-foot">
              <span>Model scores, not risk percentages.</span>
              <Explain>
                Predictions can be wrong. The models classify recorded UNSW-NB15 measurements;
                live-network performance has not been established. Your inputs stay in this browser.
              </Explain>
            </div>
            {results && (
              <button
                className="text-link save-result"
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
                <Download size={13} /> Save result
              </button>
            )}
          </section>
        </aside>
      </div>
    </div>
  );
}
