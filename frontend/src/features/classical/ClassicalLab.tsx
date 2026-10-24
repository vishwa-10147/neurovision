/**
 * Classical ML Lab: fit a scikit-learn algorithm on a small dataset and watch
 * how it decides, step by step. Every region, tree and centroid comes from
 * the fitted model on the server.
 */
import { Pause, Play } from 'lucide-react'
import { lazy, Suspense, useEffect, useState } from 'react'
import { ALGORITHMS, CLASSIC_DATASETS, fitClassical, type ClassicRequest, type ClassicResult } from '../../api/classical'
import { ErrorBoundary } from '../../components/states/ErrorBoundary'
import { EmptyState, ErrorState, LoadingState } from '../../components/states/StateViews'
import { TreeDiagram } from '../../components/tree/TreeDiagram'
import { classColor } from '../../lib/classColors'
import type { RunPoint } from '../../types/runs'
import { ALGORITHM_INFO, DEFAULT_REQUEST, PARAM_INFO } from './classical'
import { RegionPlot } from './RegionPlot'

const RunMetricCharts = lazy(() => import('../training-lab/MetricCharts').then((m) => ({ default: m.RunMetricCharts })))

const DATASET_LABELS: Record<string, string> = {
  moons: 'Two moons (2D)',
  circles: 'Circles (2D)',
  spiral: 'Spiral, 3 classes (2D)',
  blobs: 'Blobs, 4 clusters (2D)',
  digits: 'Handwritten digits (64 pixels, 10 classes)',
}

type Status = { state: 'idle' } | { state: 'fitting' } | { state: 'error'; error: unknown }

export function ClassicalLab() {
  const [form, setForm] = useState<ClassicRequest>(DEFAULT_REQUEST)
  const [result, setResult] = useState<ClassicResult | null>(null)
  const [status, setStatus] = useState<Status>({ state: 'idle' })
  const [frameIndex, setFrameIndex] = useState(0)
  const [playing, setPlaying] = useState(false)
  const info = ALGORITHM_INFO[form.algorithm]
  const frames = result?.frames ?? []
  const frame = frames[Math.min(frameIndex, frames.length - 1)] ?? null

  useEffect(() => {
    if (!playing) return
    const timer = window.setInterval(() => {
      setFrameIndex((i) => {
        if (i >= frames.length - 1) {
          setPlaying(false)
          return i
        }
        return i + 1
      })
    }, 350)
    return () => window.clearInterval(timer)
  }, [playing, frames.length])

  const update = <K extends keyof ClassicRequest>(key: K, value: ClassicRequest[K]) =>
    setForm((f) => ({ ...f, [key]: value }))

  const chooseAlgorithm = (algorithm: ClassicRequest['algorithm']) => {
    const datasets = ALGORITHM_INFO[algorithm].datasets
    setForm((f) => ({ ...f, algorithm, dataset: datasets.includes(f.dataset) ? f.dataset : datasets[0] }))
  }

  const run = async () => {
    setStatus({ state: 'fitting' })
    setPlaying(false)
    try {
      const fitted = await fitClassical(form)
      setResult(fitted)
      setFrameIndex(fitted.frames.length > 1 ? 0 : fitted.frames.length - 1)
      setPlaying(fitted.frames.length > 1)
      setStatus({ state: 'idle' })
    } catch (error) {
      setStatus({ state: 'error', error })
    }
  }

  const curve: RunPoint[] = frames
    .filter((f) => f.train_accuracy !== null)
    .map((f, i) => ({
      epoch: 0,
      step: i + 1,
      metrics: { accuracy: f.train_accuracy as number, test_accuracy: f.test_accuracy as number },
    }))

  return (
    <section aria-labelledby="classical-title" className="training">
      <header className="studio-header">
        <h1 id="classical-title">Classical ML</h1>
        <p className="muted">Fit scikit-learn algorithms and watch how they decide, step by step.</p>
      </header>
      <div className="training-grid">
        <form
          className="card training-form"
          noValidate
          onSubmit={(e) => {
            e.preventDefault()
            void run()
          }}
        >
          <label className="field">
            <span>Algorithm</span>
            <select value={form.algorithm} onChange={(e) => chooseAlgorithm(e.target.value as ClassicRequest['algorithm'])}>
              {ALGORITHMS.map((a) => (
                <option key={a} value={a}>
                  {ALGORITHM_INFO[a].family}: {ALGORITHM_INFO[a].label}
                </option>
              ))}
            </select>
          </label>
          <p className="muted algorithm-about">{info.about}</p>
          <label className="field">
            <span>Dataset</span>
            <select value={form.dataset} onChange={(e) => update('dataset', e.target.value as ClassicRequest['dataset'])}>
              {CLASSIC_DATASETS.filter((d) => info.datasets.includes(d)).map((d) => (
                <option key={d} value={d}>
                  {DATASET_LABELS[d]}
                </option>
              ))}
            </select>
          </label>
          <NumberField label="Samples" value={form.samples} min={40} max={2000} step={10} onChange={(v) => update('samples', v)} />
          {form.dataset !== 'digits' && (
            <NumberField label="Noise" value={form.noise} min={0} max={0.5} step={0.01} onChange={(v) => update('noise', v)} />
          )}
          {info.params.map((p) => (
            <NumberField
              key={p}
              label={PARAM_INFO[p].label}
              value={form[p]}
              min={PARAM_INFO[p].min}
              max={PARAM_INFO[p].max}
              step={PARAM_INFO[p].step}
              onChange={(v) => update(p, v)}
            />
          ))}
          <NumberField label="Seed" value={form.seed} min={0} max={99999} step={1} onChange={(v) => update('seed', v)} />
          <div className="training-actions">
            <button type="submit" className="button button-primary" disabled={status.state === 'fitting'}>
              {status.state === 'fitting' ? 'Fitting…' : 'Fit'}
            </button>
          </div>
        </form>

        <div className="training-main">
          {status.state === 'error' && <ErrorState error={status.error} onRetry={() => void run()} />}
          {status.state === 'fitting' && <LoadingState label="Fitting in scikit-learn…" />}
          {!result && status.state === 'idle' && (
            <div className="card">
              <EmptyState title="Choose an algorithm and press Fit">
                Nine algorithms, from logistic regression to gradient boosting, k-means and PCA, on moons, circles,
                spiral, blobs or real handwritten digits.
              </EmptyState>
            </div>
          )}
          {result && (
            <>
              <div className="card classical-result">
                <h2 className="run-title">
                  {ALGORITHM_INFO[result.algorithm].label} on {DATASET_LABELS[result.dataset]}
                </h2>
                <div className="classical-body">
                  {result.bounds || result.pca ? (
                    <RegionPlot result={result} frame={frame} />
                  ) : (
                    <p className="muted">
                      This dataset has {result.features} features, so there are no 2D regions to draw; the scores
                      below are measured on it.
                    </p>
                  )}
                  <div className="classical-side">
                    {frames.length > 1 && (
                      <div className="frame-controls" role="group" aria-label="Animation">
                        <button type="button" className="icon-button" onClick={() => setPlaying((p) => !p)} aria-label={playing ? 'Pause' : 'Play'}>
                          {playing ? <Pause size={16} /> : <Play size={16} />}
                        </button>
                        <input
                          type="range"
                          min={0}
                          max={frames.length - 1}
                          value={Math.min(frameIndex, frames.length - 1)}
                          aria-label={`Step through ${ALGORITHM_INFO[result.algorithm].animates ?? 'steps'}`}
                          onChange={(e) => {
                            setPlaying(false)
                            setFrameIndex(Number(e.target.value))
                          }}
                        />
                        <span className="frame-label">{frame?.label}</span>
                      </div>
                    )}
                    <dl className="metric-tiles" aria-label="Scores">
                      {frame?.train_accuracy != null && (
                        <div>
                          <dt>Train accuracy</dt>
                          <dd>{(frame.train_accuracy * 100).toFixed(1)}%</dd>
                        </div>
                      )}
                      {frame?.test_accuracy != null && (
                        <div>
                          <dt>Test accuracy</dt>
                          <dd>{(frame.test_accuracy * 100).toFixed(1)}%</dd>
                        </div>
                      )}
                      {frame?.inertia != null && (
                        <div>
                          <dt>Inertia (within-cluster)</dt>
                          <dd>{frame.inertia.toFixed(1)}</dd>
                        </div>
                      )}
                      <div>
                        <dt>Fit time</dt>
                        <dd>{result.fit_ms.toFixed(0)} ms</dd>
                      </div>
                    </dl>
                    <Legend result={result} />
                    {result.notes.map((n) => (
                      <p key={n} className="muted">
                        {n}
                      </p>
                    ))}
                  </div>
                </div>
              </div>
              {curve.length > 1 && (
                <ErrorBoundary fallback={(_, reset) => <ErrorState error={new Error('The chart could not be shown.')} onRetry={reset} />}>
                  <Suspense fallback={<LoadingState label="Loading chart…" />}>
                    <RunMetricCharts points={curve} />
                  </Suspense>
                </ErrorBoundary>
              )}
              {result.tree && (
                <TreeDiagram
                  nodes={result.tree}
                  featureNames={['x', 'y']}
                  description={`${result.tree.length} nodes. Each split sends points left when the condition holds. Color: majority class, stronger when the node is purer.${result.tree_note ? ` ${result.tree_note}` : ''}`}
                />
              )}
              {result.feature_importance && <Importance values={result.feature_importance} />}
              {result.pca && <Variance ratios={result.pca.explained_variance_ratio} />}
            </>
          )}
        </div>
      </div>
    </section>
  )
}

function NumberField({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string
  value: number
  min: number
  max: number
  step: number
  onChange: (value: number) => void
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => {
          const v = Number(e.target.value)
          if (Number.isFinite(v)) onChange(Math.min(max, Math.max(min, v)))
        }}
      />
    </label>
  )
}

function Legend({ result }: { result: ClassicResult }) {
  const classes = result.algorithm === 'kmeans' ? 0 : Math.min(result.classes, 10)
  return (
    <ul className="plain-list classical-legend" aria-label="Legend">
      {Array.from({ length: classes }, (_, c) => (
        <li key={c}>
          <span className="swatch-dot" style={{ background: classColor(c) }} aria-hidden="true" /> class {c}
        </li>
      ))}
      {result.test.x.length > 0 && <li>● training point · ○ test point</li>}
      {result.support_vectors && <li>ringed: support vector</li>}
      {result.algorithm === 'kmeans' && <li>✕ centroid; colors are clusters found, not labels</li>}
      {result.pca?.axes && <li>red / blue line: first / second principal component (±2 std)</li>}
      {result.algorithm === 'kmeans' ? (
        <li>shading: the nearest centroid</li>
      ) : (
        result.frames.some((f) => f.grid_class.length > 0) && <li>shading: predicted class, darker = more confident</li>
      )}
    </ul>
  )
}

function Importance({ values }: { values: number[] }) {
  const named = values.map((v, i) => ({ name: values.length === 2 ? ['x', 'y'][i] : `pixel ${i}`, v }))
  const top = [...named].sort((a, b) => b.v - a.v).slice(0, 10)
  const max = Math.max(...top.map((t) => t.v), 1e-9)
  return (
    <section className="card" aria-label="Feature importance">
      <h3>Feature importance</h3>
      <p className="muted">Share of the impurity reduction each feature provides{values.length > 10 ? ' (top 10)' : ''}.</p>
      <ul className="plain-list bars">
        {top.map(({ name, v }) => (
          <li key={name}>
            <span className="bar-label">{name}</span>
            <span className="bar" style={{ width: `${(v / max) * 100}%` }} />
            <span className="bar-value">{(v * 100).toFixed(1)}%</span>
          </li>
        ))}
      </ul>
    </section>
  )
}

function Variance({ ratios }: { ratios: number[] }) {
  const cumulative = ratios.map((_, i) => ratios.slice(0, i + 1).reduce((a, b) => a + b, 0))
  return (
    <section className="card" aria-label="Explained variance">
      <h3>Explained variance</h3>
      <p className="muted">Share of the data’s variance along each principal component.</p>
      <ul className="plain-list bars">
        {ratios.map((r, i) => {
          return (
            <li key={i}>
              <span className="bar-label">PC{i + 1}</span>
              <span className="bar" style={{ width: `${(r / ratios[0]) * 100}%` }} />
              <span className="bar-value">
                {(r * 100).toFixed(1)}% (total {(cumulative[i] * 100).toFixed(0)}%)
              </span>
            </li>
          )
        })}
      </ul>
    </section>
  )
}
