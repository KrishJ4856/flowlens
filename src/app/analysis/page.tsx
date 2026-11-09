'use client';
import { useEffect, useState } from 'react';
import Link from 'next/link';
import { ArrowRight, CheckCircle2, LoaderCircle, Download } from 'lucide-react';
import {
  BarChart,
  Bar,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
  Cell,
} from 'recharts';
import { PageHeading, Explain, Note } from '@/components/common';
import { Button } from '@/components/ui/button';
import { formatNumber, repository } from '@/lib/features';
type Metric = {
  model: string;
  subset: string;
  accuracy: number;
  attack_recall: number;
  attack_precision: number;
  false_positive_rate: number;
  tn: number;
  fp: number;
  fn: number;
  tp: number;
  macro_f1: number;
  records: number;
};
type Analysis = {
  metrics: Metric[];
  importance: Record<string, { feature: string; importance: number }[]>;
  trainingRows: number;
  testingRows: number;
  exactOverlap: number;
  fitRows: number;
  validationRows: number;
};
const tooltipStyle = {
  background: 'var(--popover)',
  border: '1px solid var(--border)',
  borderRadius: 9,
  fontSize: 11,
  color: 'var(--foreground)',
};
export default function Analysis() {
  const [data, setData] = useState<Analysis | null>(null),
    [model, setModel] = useState('xgb_depth7'),
    [view, setView] = useState('official_test'),
    [error, setError] = useState('');
  useEffect(() => {
    fetch('/data/analysis.json')
      .then((r) => {
        if (!r.ok) throw Error('The saved analysis could not be loaded.');
        return r.json();
      })
      .then(setData)
      .catch((e) => setError(e.message));
  }, []);
  const chosen = data?.metrics.find((m) => m.model === model && m.subset === view),
    label = model === 'xgb_depth7' ? 'XGBoost' : 'Random Forest';
  const comparison = ['xgb_depth7', 'rf_depth24'].map((name) => {
    const m = data?.metrics.find((m) => m.model === name && m.subset === view);
    return {
      name: name === 'xgb_depth7' ? 'XGBoost' : 'Random Forest',
      Accuracy: (m?.accuracy || 0) * 100,
      'Attack recall': (m?.attack_recall || 0) * 100,
      'False positive rate': (m?.false_positive_rate || 0) * 100,
    };
  });
  return (
    <div className="container page-content">
      <PageHeading
        eyebrow="MEASURED RESULTS, EXPLAINED"
        title="What did the models learn?"
        description="A useful detector should catch attacks and avoid false alarms. These charts show both, using predictions from the frozen experiment."
      >
        <Button asChild variant="outline" size="sm">
          <a href={`${repository}/blob/main/docs/REPORT.md`} target="_blank" rel="noreferrer">
            <Download size={14} /> Full experiment report
          </a>
        </Button>
      </PageHeading>
      {error && (
        <div className="error-message" role="alert">
          {error}
        </div>
      )}
      {!data ? (
        <div className="loading">
          <LoaderCircle className="spin" size={20} /> Loading measured results…
        </div>
      ) : (
        <>
          <div className="flex flex-wrap items-center justify-between gap-3 mb-5">
            <div className="analysis-selector" aria-label="Model for detailed results">
              {[
                ['xgb_depth7', 'XGBoost'],
                ['rf_depth24', 'Random Forest'],
              ].map(([id, title]) => (
                <button
                  key={id}
                  className={model === id ? 'selected' : ''}
                  onClick={() => setModel(id)}
                  aria-pressed={model === id}
                >
                  {title}
                </button>
              ))}
            </div>
            <div className="flex items-center gap-2">
              <label htmlFor="test-view" className="text-[10px] text-muted-foreground">
                Test view
              </label>
              <select
                id="test-view"
                className="select-input"
                value={view}
                onChange={(e) => setView(e.target.value)}
              >
                <option value="official_test">Complete test · 82,332 records</option>
                <option value="unseen_features">No exact training match · 73,791</option>
                <option value="unique_unseen_feature_target">
                  Unique unseen feature + answer · 52,648
                </option>
              </select>
            </div>
          </div>
          {chosen && (
            <div className="metric-grid">
              <div className="metric-box">
                <span className="metric-label">
                  Overall accuracy{' '}
                  <Explain>
                    All correct Normal/Attack decisions divided by all test records.
                  </Explain>
                </span>
                <strong>{(chosen.accuracy * 100).toFixed(2)}%</strong>
                <small>
                  {label} · {formatNumber(chosen.records)} records
                </small>
              </div>
              <div className="metric-box">
                <span className="metric-label">
                  Attacks caught{' '}
                  <Explain>
                    Recall: the proportion of actual attacks that the detector correctly flagged.
                  </Explain>
                </span>
                <strong>{(chosen.attack_recall * 100).toFixed(2)}%</strong>
                <small>Higher is better</small>
              </div>
              <div className="metric-box">
                <span className="metric-label">
                  False alarms{' '}
                  <Explain>
                    Normal records incorrectly flagged as Attack, divided by all actual normal
                    records. This is the false positive rate.
                  </Explain>
                </span>
                <strong style={{ color: 'var(--amber)' }}>
                  {(chosen.false_positive_rate * 100).toFixed(2)}%
                </strong>
                <small>Lower is better</small>
              </div>
              <div className="metric-box">
                <span className="metric-label">
                  Flags that were attacks{' '}
                  <Explain>
                    Precision: among all records flagged as Attack, how many really had an attack
                    label?
                  </Explain>
                </span>
                <strong>{(chosen.attack_precision * 100).toFixed(2)}%</strong>
                <small>Attack precision</small>
              </div>
            </div>
          )}
          <Note>
            <b>The main trade-off:</b> both models catch roughly 98% of attacks on the complete
            test, but wrongly flag about one in four normal records. These are useful benchmark
            results with a substantial false-alarm limitation.
          </Note>
          <div className="chart-grid">
            <section className="panel">
              <h2>Two algorithms, side by side</h2>
              <p className="chart-description">The same test records. Three different questions.</p>
              <div
                className="chart-frame"
                role="img"
                aria-label={`Comparison: ${comparison.map((m) => `${m.name}: accuracy ${m.Accuracy.toFixed(2)}%, recall ${m['Attack recall'].toFixed(2)}%, false-positive rate ${m['False positive rate'].toFixed(2)}%`).join('; ')}`}
              >
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={comparison} margin={{ top: 10, right: 0, left: -25, bottom: 0 }}>
                    <CartesianGrid vertical={false} stroke="var(--border)" strokeDasharray="3 3" />
                    <XAxis
                      dataKey="name"
                      tick={{ fontSize: 11 }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <YAxis
                      domain={[0, 100]}
                      tickFormatter={(v) => `${v}%`}
                      tick={{ fontSize: 10 }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <Tooltip
                      contentStyle={tooltipStyle}
                      formatter={(value) => `${Number(value).toFixed(2)}%`}
                    />
                    <Legend wrapperStyle={{ fontSize: 10, paddingTop: 15 }} />
                    <Bar dataKey="Accuracy" fill="#5b85ec" radius={[4, 4, 0, 0]} maxBarSize={32} />
                    <Bar
                      dataKey="Attack recall"
                      fill="#32a898"
                      radius={[4, 4, 0, 0]}
                      maxBarSize={32}
                    />
                    <Bar
                      dataKey="False positive rate"
                      fill="#d8a15b"
                      radius={[4, 4, 0, 0]}
                      maxBarSize={32}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <p className="chart-description">
                Taller is better for accuracy and recall. A <b>shorter</b> false-positive bar is
                better. Random Forest performed slightly better on the final test; XGBoost had been
                selected earlier using validation.
              </p>
            </section>
            <section className="panel">
              <h2>What a mistake looks like</h2>
              <p className="chart-description">
                {label} · four possible outcomes on the selected test view.
              </p>
              {chosen && (
                <div className="confusion-grid">
                  <div className="confusion-cell">
                    <strong>{formatNumber(chosen.tn)}</strong>
                    <span>Normal correctly accepted</span>
                    <small>Actual Normal → predicted Normal</small>
                  </div>
                  <div className="confusion-cell mistake">
                    <strong>{formatNumber(chosen.fp)}</strong>
                    <span>Normal wrongly flagged</span>
                    <small>False positive · a false alarm</small>
                  </div>
                  <div className="confusion-cell mistake">
                    <strong>{formatNumber(chosen.fn)}</strong>
                    <span>Attacks missed</span>
                    <small>False negative · an attack let through</small>
                  </div>
                  <div className="confusion-cell">
                    <strong>{formatNumber(chosen.tp)}</strong>
                    <span>Attacks correctly caught</span>
                    <small>Actual Attack → predicted Attack</small>
                  </div>
                </div>
              )}
              <p className="chart-description">
                These counts reconcile to all {formatNumber(chosen?.records || 0)} records in this
                view. Accuracy alone hides the difference between a missed attack and a false alarm.
              </p>
            </section>
            <section className="panel">
              <h2>Why check repeated patterns?</h2>
              <p className="chart-description">
                Some test inputs exactly match measurements seen in training.
              </p>
              <div
                className="chart-frame"
                role="img"
                aria-label="XGBoost accuracy: 87.12% on complete test; 86.35% without exact training matches; 82.59% on unique unseen feature-plus-answer records"
              >
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={[
                      {
                        name: 'Complete test',
                        accuracy:
                          (data.metrics.find(
                            (m) => m.model === 'xgb_depth7' && m.subset === 'official_test',
                          )?.accuracy || 0) * 100,
                      },
                      {
                        name: 'No exact match',
                        accuracy:
                          (data.metrics.find(
                            (m) => m.model === 'xgb_depth7' && m.subset === 'unseen_features',
                          )?.accuracy || 0) * 100,
                      },
                      {
                        name: 'Unique unseen',
                        accuracy:
                          (data.metrics.find(
                            (m) =>
                              m.model === 'xgb_depth7' &&
                              m.subset === 'unique_unseen_feature_target',
                          )?.accuracy || 0) * 100,
                      },
                    ]}
                    margin={{ top: 10, right: 0, left: -25, bottom: 0 }}
                  >
                    <CartesianGrid vertical={false} stroke="var(--border)" strokeDasharray="3 3" />
                    <XAxis
                      dataKey="name"
                      axisLine={false}
                      tickLine={false}
                      tick={{ fontSize: 10 }}
                    />
                    <YAxis
                      domain={[0, 100]}
                      tickFormatter={(v) => `${v}%`}
                      tick={{ fontSize: 10 }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <Tooltip
                      contentStyle={tooltipStyle}
                      formatter={(value) => `${Number(value).toFixed(2)}%`}
                    />
                    <Bar
                      dataKey="accuracy"
                      name="XGBoost accuracy"
                      fill="#5b85ec"
                      radius={[5, 5, 0, 0]}
                      maxBarSize={48}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <p className="chart-description">
                <b>8,541</b> test records have an exact training-feature match. Removing them leaves{' '}
                <b>73,791</b> records. The unique unseen view also counts each feature-plus-answer
                pair once, changing the weighting. These views reveal dataset limitations; they do
                not prove live-network performance.
              </p>
            </section>
            <section className="panel">
              <h2>Which measurements mattered?</h2>
              <p className="chart-description">
                Top training-derived feature importances for {label}.
              </p>
              <div className="chart-frame" role="img" aria-label={`Top features for ${label}`}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    layout="vertical"
                    data={data.importance[label]?.slice(0, 7) || []}
                    margin={{ top: 0, right: 15, left: 0, bottom: 0 }}
                  >
                    <CartesianGrid
                      horizontal={false}
                      stroke="var(--border)"
                      strokeDasharray="3 3"
                    />
                    <XAxis type="number" tick={{ fontSize: 9 }} axisLine={false} tickLine={false} />
                    <YAxis
                      type="category"
                      dataKey="feature"
                      width={90}
                      tick={{ fontSize: 10 }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <Tooltip
                      contentStyle={tooltipStyle}
                      formatter={(value) => Number(value).toFixed(4)}
                    />
                    <Bar
                      dataKey="importance"
                      name="Training importance"
                      fill="#32a898"
                      radius={[0, 4, 4, 0]}
                      barSize={15}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <p className="chart-description">
                Importance summarizes how the fitted model used features. It is <b>not a cause</b>,
                and it does not explain one individual prediction. Importances from different
                algorithms are not directly interchangeable.
              </p>
            </section>
          </div>
          <section className="panel">
            <div className="panel-header">
              <div>
                <h2>How we kept the experiment honest</h2>
                <p>Choices made before opening the final test answers.</p>
              </div>
              <CheckCircle2 size={22} style={{ color: 'var(--teal)' }} />
            </div>
            <div className="glossary">
              <div>
                <b>Keep the answers out of the inputs</b>
                <p>
                  ID, label and attack_cat never enter a detector. Both models receive the same 42
                  measurements.
                </p>
              </div>
              <div>
                <b>Use validation for choosing settings</b>
                <p>
                  140,273 fitting rows and 35,068 validation rows. Identical feature groups stay
                  together. Preparation is learned inside the fitting split.
                </p>
              </div>
              <div>
                <b>Freeze, refit, then test</b>
                <p>
                  Freeze model settings and cutoffs using validation, refit on all training rows,
                  and evaluate the separate test partition.
                </p>
              </div>
              <div>
                <b>Check the website against Python</b>
                <p>
                  The browser inference produced zero decision mismatches against both original
                  models over all 82,332 test records.{' '}
                  <a
                    href="/data/model-parity.json"
                    target="_blank"
                    className="underline text-primary"
                  >
                    Read the parity check
                  </a>
                  .
                </p>
              </div>
            </div>
            <Link href="/project/" className="subtle-link mt-5">
              Follow the complete workflow <ArrowRight size={12} />
            </Link>
          </section>
        </>
      )}
    </div>
  );
}
