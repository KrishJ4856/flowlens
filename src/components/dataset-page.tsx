'use client';
import { useEffect, useState, useRef } from 'react';
import { useRouter } from 'next/navigation';
import {
  ArrowRight,
  ChevronLeft,
  ChevronRight,
  Download,
  Eye,
  LoaderCircle,
  Search,
  TableProperties,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Dialog, DialogContent, DialogTitle, DialogDescription } from '@/components/ui/dialog';
import { Explain, PageHeading } from '@/components/common';
import { details, formatNumber, readGzip, repository, type Values } from '@/lib/features';
type RecordRow = Record<string, string | number>;
type DatasetMeta = {
  rows: number;
  columns: string[];
  categories: Record<string, number>;
  filters: Record<string, { count: number; chunks: number }>;
};
export function DatasetPage({ partition }: { partition: 'train' | 'test' }) {
  const train = partition === 'train',
    router = useRouter();
  const [meta, setMeta] = useState<DatasetMeta | null>(null),
    [filter, setFilter] = useState('all'),
    [page, setPage] = useState(0),
    [chunk, setChunk] = useState<(string | number)[][]>([]),
    [loading, setLoading] = useState(true),
    [error, setError] = useState(''),
    [selected, setSelected] = useState<RecordRow | null>(null),
    [full, setFull] = useState(false),
    [search, setSearch] = useState(''),
    [searching, setSearching] = useState(false),
    [attempt, setAttempt] = useState(0);
  const cache = useRef(new Map<string, (string | number)[][]>());
  useEffect(() => {
    let alive = true;
    fetch('/data/datasets.json')
      .then((r) => {
        if (!r.ok) throw Error('Dataset details could not be loaded.');
        return r.json();
      })
      .then((data) => alive && setMeta(data[partition]))
      .catch((e) => alive && setError(e.message));
    return () => {
      alive = false;
    };
  }, [partition, attempt]);
  const chunkIndex = Math.floor((page * 25) / 500);
  useEffect(() => {
    let alive = true;
    setLoading(true);
    setError('');
    const key = `${partition}/${filter}/${chunkIndex}`;
    const cached = cache.current.get(key);
    if (cached) {
      setChunk(cached);
      setLoading(false);
      return;
    }
    readGzip(`/data/${partition}/${encodeURIComponent(filter)}/${chunkIndex}.json.gz`)
      .then((data) => {
        if (!alive) return;
        cache.current.set(key, data);
        setChunk(data);
        setLoading(false);
      })
      .catch((e) => {
        if (alive) {
          setError(e.message);
          setLoading(false);
        }
      });
    return () => {
      alive = false;
    };
  }, [partition, filter, chunkIndex, attempt]);
  const columns = full
    ? meta?.columns || []
    : [
        'id',
        'proto',
        'service',
        'state',
        'dur',
        'spkts',
        'dpkts',
        'sbytes',
        'dbytes',
        'label',
        'attack_cat',
      ];
  const count = meta?.filters[filter]?.count || 0,
    pages = Math.ceil(count / 25),
    offset = (page * 25) % 500;
  const rows = chunk
    .slice(offset, offset + 25)
    .map(
      (row) =>
        Object.fromEntries((meta?.columns || []).map((col, i) => [col, row[i]])) as RecordRow,
    );
  const normal = meta?.categories.Normal || 0,
    attack = (meta?.rows || 0) - normal;
  async function findRecord(event: React.FormEvent) {
    event.preventDefault();
    if (!meta) return;
    const id = Number(search);
    if (!Number.isInteger(id) || id < 1 || id > meta.rows) {
      setError(`Enter a record ID from 1 to ${formatNumber(meta.rows)}.`);
      return;
    }
    setSearching(true);
    setError('');
    try {
      const index = Math.floor((id - 1) / 500),
        key = `${partition}/all/${index}`;
      const data =
        cache.current.get(key) || (await readGzip(`/data/${partition}/all/${index}.json.gz`));
      cache.current.set(key, data);
      const row = data[(id - 1) % 500];
      setSelected(Object.fromEntries(meta.columns.map((col, i) => [col, row[i]])));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSearching(false);
    }
  }
  function useRecord(record: RecordRow) {
    const values = Object.fromEntries(
      Object.entries(record).filter(([name]) => !['id', 'label', 'attack_cat'].includes(name)),
    ) as Values;
    sessionStorage.setItem(
      'flowlens-record',
      JSON.stringify({
        id: `${partition}-${record.id}`,
        title: `${train ? 'Training' : 'Test'} · record ${record.id}`,
        source: train ? 'Training data' : 'Testing data',
        values,
        truth: Number(record.label),
        category: record.attack_cat,
        recordId: Number(record.id),
      }),
    );
    router.push('/playground/');
  }
  return (
    <div className="container page-content">
      <PageHeading
        title={train ? 'Training data' : 'Testing data'}
        description={
          train
            ? 'The examples used to teach the models.'
            : 'The separate records used to check them.'
        }
      >
        <Button asChild variant="outline" size="sm">
          <a
            href={`https://raw.githubusercontent.com/KrishJ4856/flowlens/main/ml/data/raw/${partition}.csv`}
          >
            <Download size={15} /> Download CSV
          </a>
        </Button>
      </PageHeading>
      <div className="dataset-summary">
        <div>
          <strong>{meta ? formatNumber(meta.rows) : '—'}</strong>
          <span>records</span>
        </div>
        <div>
          <i className="inline-dot" />
          <strong>{meta ? formatNumber(normal) : '—'}</strong>
          <span>Normal</span>
        </div>
        <div>
          <i className="inline-dot attack" />
          <strong>{meta ? formatNumber(attack) : '—'}</strong>
          <span>Attack</span>
        </div>
        <span className="dataset-features">
          42 measurements{' '}
          <Explain>
            Each row has 42 inputs, an ID and two answer columns. The ID, label and attack_cat are
            excluded from prediction inputs.
          </Explain>
        </span>
      </div>
      <section className="panel data-panel">
        <div className="data-toolbar">
          <div className="data-filters">
            <label htmlFor="dataset-filter" className="sr-only">
              Filter the full dataset
            </label>
            <select
              id="dataset-filter"
              className="select-input"
              value={filter}
              onChange={(e) => {
                setFilter(e.target.value);
                setPage(0);
              }}
            >
              <option value="all">All records</option>
              <option value="Normal">Normal only</option>
              <option value="Attack">All attacks</option>
              {Object.keys(meta?.categories || {})
                .filter((c) => c !== 'Normal')
                .sort()
                .map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
            </select>
            <Button
              variant={full ? 'secondary' : 'outline'}
              size="sm"
              onClick={() => setFull(!full)}
            >
              <TableProperties size={14} />
              {full ? 'Key columns' : 'All columns'}
            </Button>
          </div>
          <form className="data-search" onSubmit={findRecord}>
            <Input
              type="number"
              min="1"
              max={meta?.rows}
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Record ID"
              aria-label="Find a record by ID"
            />
            <Button
              type="submit"
              variant="outline"
              size="sm"
              disabled={searching || !search}
              aria-label="Find record"
            >
              {searching ? <LoaderCircle className="spin" size={15} /> : <Search size={15} />}
            </Button>
          </form>
        </div>
        {error && (
          <div className="error-message" role="alert">
            {error}{' '}
            <button className="underline" onClick={() => setAttempt((a) => a + 1)}>
              Retry
            </button>
          </div>
        )}
        <div
          className="table-scroll"
          tabIndex={0}
          aria-label="Network flow records; scroll horizontally for more columns"
        >
          {loading || !meta ? (
            <div className="loading">
              <LoaderCircle className="spin" size={18} /> Loading records…
            </div>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Record</th>
                  {columns.map((col) => (
                    <th key={col}>
                      {col === 'label' ? 'Answer' : col === 'attack_cat' ? 'Category' : col}
                      {details[col] && <Explain>{details[col][1]}</Explain>}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((record) => (
                  <tr key={String(record.id)}>
                    <td>
                      <Button
                        variant="ghost"
                        size="xs"
                        onClick={() => setSelected(record)}
                        aria-label={`Inspect record ${record.id}`}
                      >
                        <Eye size={13} /> View
                      </Button>
                    </td>
                    {columns.map((col) => (
                      <td key={col}>
                        {col === 'label' ? (
                          <span className={`pill ${record[col] === 0 ? 'normal' : 'attack'}`}>
                            {record[col] === 0 ? 'Normal' : 'Attack'}
                          </span>
                        ) : (
                          String(record[col])
                        )}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
        <div className="table-footer">
          <span>
            {loading
              ? 'Loading…'
              : `${formatNumber(page * 25 + 1)}–${formatNumber(Math.min((page + 1) * 25, count))} of ${formatNumber(count)}`}
          </span>
          <div className="pagination">
            <Button
              size="sm"
              variant="outline"
              disabled={page === 0 || loading}
              onClick={() => setPage((p) => p - 1)}
              aria-label="Previous page"
            >
              <ChevronLeft size={15} />
            </Button>
            <span>
              {page + 1} / {formatNumber(pages || 1)}
            </span>
            <Button
              size="sm"
              variant="outline"
              disabled={page + 1 >= pages || loading}
              onClick={() => setPage((p) => p + 1)}
              aria-label="Next page"
            >
              <ChevronRight size={15} />
            </Button>
          </div>
        </div>
      </section>
      <div className="data-bottom">
        <span>Open a record to inspect all measurements or try it in the playground.</span>
        <details className="explainer">
          <summary>About this dataset</summary>
          <p>
            UNSW-NB15 records describe network communications, not message contents. Normal is label
            0; Attack is label 1.
          </p>
          <p>
            {train
              ? 'Training was split into 140,273 fitting and 35,068 validation rows. After choosing settings, both models were refitted on all training rows.'
              : 'Test answers were used only after model choices were frozen. Some measurements repeat training patterns; the Analysis page reports a separate no-exact-match view.'}
          </p>
          <a
            href={`${repository}/blob/main/ml/data/source_manifest.json`}
            target="_blank"
            rel="noreferrer"
            className="text-link"
          >
            View data sources <ArrowRight size={13} />
          </a>
        </details>
      </div>
      <Dialog open={!!selected} onOpenChange={(open) => !open && setSelected(null)}>
        <DialogContent className="max-w-2xl max-h-[85vh] overflow-y-auto">
          <DialogTitle>Record {selected?.id}</DialogTitle>
          <DialogDescription>
            {train ? 'Training' : 'Testing'} data · 42 measurements
          </DialogDescription>
          {selected && (
            <>
              <div className="record-dialog-actions">
                <span className={`pill ${Number(selected.label) === 0 ? 'normal' : 'attack'}`}>
                  Actual: {Number(selected.label) === 0 ? 'Normal' : 'Attack'}
                </span>
                <span className="record-category">{selected.attack_cat}</span>
                <Button size="sm" onClick={() => useRecord(selected)}>
                  Try in playground <ArrowRight size={14} />
                </Button>
              </div>
              {train && <p className="inline-note">Training example—not an independent test.</p>}
              <div className="record-fields">
                {Object.entries(selected)
                  .filter(([name]) => !['id', 'label', 'attack_cat'].includes(name))
                  .map(([name, value]) => (
                    <div key={name}>
                      <span>
                        {details[name]?.[0] || name}
                        <Explain>
                          <b>{name}</b>
                          <br />
                          {details[name]?.[1]}
                        </Explain>
                      </span>
                      <b>{value}</b>
                    </div>
                  ))}
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
