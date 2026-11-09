'use client';
import { useEffect, useState } from 'react';
import { ArrowUpRight, LoaderCircle } from 'lucide-react';
import {
  BarChart,
  Bar,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { PageHeading, Explain } from '@/components/common';
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
  records: number;
};
type Analysis = {
  metrics: Metric[];
  importance: Record<string, { feature: string; importance: number }[]>;
};
const tooltipStyle = {
  background: 'var(--popover)',
  border: '1px solid var(--border)',
  borderRadius: 9,
  fontSize: 13,
  color: 'var(--foreground)',
  fontFamily: 'inherit',
};
export default function Analysis() {
  const [data, setData] = useState<Analysis | null>(null),
    [model, setModel] = useState('xgb_depth7'),
    [view, setView] = useState('official_test'),
    [error, setError] = useState('');
  useEffect(() => {
    let active = true;
    fetch('/data/analysis.json')
      .then((r) => {
        if (!r.ok) throw Error('Results could not be loaded.');
        return r.json();
      })
      .then((d) => active && setData(d))
      .catch((e) => active && setError(e.message));
    return () => {
      active = false;
    };
  }, []);
  const chosen = data?.metrics.find((m) => m.model === model && m.subset === view),
    label = model === 'xgb_depth7' ? 'XGBoost' : 'Random Forest';
  const comparison = ['xgb_depth7', 'rf_depth24'].map((name) => {
    const m = data?.metrics.find((m) => m.model === name && m.subset === view);
    return {
      name: name === 'xgb_depth7' ? 'XGBoost' : 'Random Forest',
      Accuracy: (m?.accuracy || 0) * 100,
      'Attacks caught': (m?.attack_recall || 0) * 100,
      'False alarms': (m?.false_positive_rate || 0) * 100,
    };
  });
  return (
    <div className="container page-content">
      <PageHeading
        title="The results."
        description="Catch attacks. Avoid false alarms. Both matter."
      />
      {error && (
        <div className="error-message" role="alert">
          {error}
        </div>
      )}
      {!data ? (
        <div className="loading">
          <LoaderCircle className="spin" size={18} /> Loading results…
        </div>
      ) : (
        <>
          <div className="analysis-toolbar">
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
            <select
              id="test-view"
              aria-label="Test view"
              className="select-input"
              value={view}
              onChange={(e) => setView(e.target.value)}
            >
              <option value="official_test">Complete test · 82,332</option>
              <option value="unseen_features">No exact training match · 73,791</option>
              <option value="unique_unseen_feature_target">Unique unseen records · 52,648</option>
            </select>
          </div>
          {chosen && (
            <div className="metric-grid">
              <div className="metric-box">
                <span>
                  Overall accuracy{' '}
                  <Explain>All correct decisions divided by all records in this test view.</Explain>
                </span>
                <strong>
                  {(chosen.accuracy * 100).toFixed(2)}
                  <small>%</small>
                </strong>
                <p>Correct decisions</p>
              </div>
              <div className="metric-box">
                <span>
                  Attacks caught{' '}
                  <Explain>Recall: the proportion of actual attacks correctly flagged.</Explain>
                </span>
                <strong>
                  {(chosen.attack_recall * 100).toFixed(2)}
                  <small>%</small>
                </strong>
                <p>Higher is better</p>
              </div>
              <div className="metric-box">
                <span>
                  False alarms{' '}
                  <Explain>
                    Normal records wrongly flagged, divided by all actual normal records. This is
                    the false positive rate.
                  </Explain>
                </span>
                <strong className="amber-text">
                  {(chosen.false_positive_rate * 100).toFixed(2)}
                  <small>%</small>
                </strong>
                <p>Lower is better</p>
              </div>
            </div>
          )}
          <div className="analysis-note">
            <span className="warning-dot" />
            <b>The trade-off:</b> about 1 in 4 normal flows is wrongly flagged on the complete test.
          </div>
          <div className="chart-grid">
            <section className="panel">
              <h2>Compare the models</h2>
              <div
                className="chart-frame"
                role="img"
                aria-label={`Test comparison: ${comparison.map((m) => `${m.name}, accuracy ${m.Accuracy.toFixed(2)}%, attacks caught ${m['Attacks caught'].toFixed(2)}%, false alarms ${m['False alarms'].toFixed(2)}%`).join('; ')}`}
              >
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={comparison} margin={{ top: 15, right: 5, left: -20, bottom: 0 }}>
                    <CartesianGrid vertical={false} stroke="var(--border)" strokeDasharray="3 4" />
                    <XAxis
                      dataKey="name"
                      tick={{ fontSize: 13 }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <YAxis
                      domain={[0, 100]}
                      tickFormatter={(v) => `${v}%`}
                      tick={{ fontSize: 11 }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <Tooltip
                      contentStyle={tooltipStyle}
                      formatter={(v) => `${Number(v).toFixed(2)}%`}
                    />
                    <Legend wrapperStyle={{ fontSize: 12, paddingTop: 12 }} />
                    <Bar dataKey="Accuracy" fill="#2d7360" radius={[4, 4, 0, 0]} maxBarSize={32} />
                    <Bar
                      dataKey="Attacks caught"
                      fill="#8ab6a7"
                      radius={[4, 4, 0, 0]}
                      maxBarSize={32}
                    />
                    <Bar
                      dataKey="False alarms"
                      fill="#d3a16a"
                      radius={[4, 4, 0, 0]}
                      maxBarSize={32}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <p className="chart-caption">Higher accuracy and recall. Lower false alarms.</p>
            </section>
            <section className="panel">
              <div className="section-title">
                <h2>Every outcome</h2>
                <span className="muted-label">{label}</span>
              </div>
              {chosen && (
                <div className="confusion-grid">
                  {[
                    [chosen.tn, 'Normal accepted', false],
                    [chosen.fp, 'False alarms', true],
                    [chosen.fn, 'Attacks missed', true],
                    [chosen.tp, 'Attacks caught', false],
                  ].map(([n, text, mistake]) => (
                    <div
                      className={`confusion-cell ${mistake ? 'mistake' : ''}`}
                      key={String(text)}
                    >
                      <strong>{formatNumber(Number(n))}</strong>
                      <span>{text}</span>
                    </div>
                  ))}
                </div>
              )}
              <p className="chart-caption">
                {formatNumber(chosen?.records || 0)} records in this view.
              </p>
            </section>
          </div>
          <details className="advanced-analysis">
            <summary>
              Go deeper <span>Feature importance, repeated patterns & method</span>
            </summary>
            <div className="advanced-content">
              <div className="chart-grid">
                <section className="panel">
                  <h2>Which measurements mattered?</h2>
                  <div
                    className="chart-frame"
                    role="img"
                    aria-label={`Top training features for ${label}`}
                  >
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        layout="vertical"
                        data={data.importance[label]?.slice(0, 7) || []}
                        margin={{ top: 0, right: 15, left: 0, bottom: 0 }}
                      >
                        <CartesianGrid
                          horizontal={false}
                          stroke="var(--border)"
                          strokeDasharray="3 4"
                        />
                        <XAxis
                          type="number"
                          tick={{ fontSize: 11 }}
                          axisLine={false}
                          tickLine={false}
                        />
                        <YAxis
                          type="category"
                          dataKey="feature"
                          width={105}
                          tick={{ fontSize: 11 }}
                          axisLine={false}
                          tickLine={false}
                        />
                        <Tooltip
                          contentStyle={tooltipStyle}
                          formatter={(v) => Number(v).toFixed(4)}
                        />
                        <Bar
                          dataKey="importance"
                          name="Training importance"
                          fill="#2d7360"
                          radius={[0, 4, 4, 0]}
                          barSize={15}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <p className="chart-caption">
                    Training-derived importance. Not a cause or an explanation of one prediction.
                  </p>
                </section>
                <section className="method-notes">
                  <h3>Repeated patterns</h3>
                  <p>
                    8,541 test records exactly match training measurements. The no-exact-match view
                    removes them. The unique unseen view also counts each feature-plus-answer pair
                    once.
                  </p>
                  <h3>How settings were chosen</h3>
                  <p>
                    Grouped validation, then frozen settings and score cutoffs. XGBoost was chosen
                    on validation; Random Forest did slightly better on the final test.
                  </p>
                  <h3>Browser verification</h3>
                  <p>
                    Zero decision mismatches against Python across all 82,332 test records. This
                    does not establish live-network protection.
                  </p>
                  <a href="/data/model-parity.json" target="_blank" className="text-link">
                    Verification record <ArrowUpRight size={13} />
                  </a>
                </section>
              </div>
            </div>
          </details>
          <div className="analysis-bottom">
            <span>Recorded UNSW-NB15 flows. Live-network performance is untested.</span>
            <a
              href={`${repository}/blob/main/docs/REPORT.md`}
              target="_blank"
              rel="noreferrer"
              className="text-link"
            >
              Full report <ArrowUpRight size={13} />
            </a>
          </div>
        </>
      )}
    </div>
  );
}
