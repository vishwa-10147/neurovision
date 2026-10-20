import { lazy, Suspense, useCallback, useEffect, useState } from 'react'
import { API_BASE_URL } from '../../api/client'
import { listRuns } from '../../api/runs'
import { ErrorBoundary } from '../../components/states/ErrorBoundary'
import { EmptyState, ErrorState, LoadingState } from '../../components/states/StateViews'
import { formatValue } from '../../lib/values'
import { useLiveRunStore, type WatchedRun } from '../../stores/liveRunStore'
import type { RunInfo, RunPoint } from '../../types/runs'
import { ActivationStrips } from '../training-lab/TrainingResults'
import { useEventStream } from '../training-lab/useTrainingStream'

const RunMetricCharts = lazy(() => import('../training-lab/MetricCharts').then((m) => ({ default: m.RunMetricCharts })))

/** How often the run list is refreshed while this view is open. */
export const POLL_MS = 3000

const STATUS_LABELS = { running: 'Running', completed: 'Completed', failed: 'Failed' } as const

export function LiveRuns() {
  const { runs, error, refresh } = useRunList()
  const run = useLiveRunStore((s) => s.run)
  const points = useLiveRunStore((s) => s.points)
  const activations = useLiveRunStore((s) => s.activations)
  const stream = useLiveRunStore((s) => s.stream)
  const watch = useLiveRunStore((s) => s.watch)
  const applyEvents = useLiveRunStore((s) => s.applyEvents)
  const setStream = useLiveRunStore((s) => s.setStream)
  const { reconnect } = useEventStream(run?.runId ?? null, run !== null, applyEvents, setStream)

  return (
    <section aria-labelledby="runs-title" className="training">
      <header className="studio-header">
        <h1 id="runs-title">Live runs</h1>
        <p className="muted">Watch your own PyTorch training scripts stream metrics and activations.</p>
      </header>
      <div className="training-grid">
        <div className="card run-list">
          <h2 className="run-list-title">Runs</h2>
          {error !== null && <ErrorState error={error} onRetry={refresh} />}
          {runs === null && error === null && <LoadingState label="Loading runs…" />}
          {runs !== null && runs.length === 0 && (
            <EmptyState title="No runs yet">Start one from a training script; it appears here within seconds.</EmptyState>
          )}
          {runs !== null && runs.length > 0 && (
            <ul aria-label="External runs">
              {runs.map((info) => (
                <li key={info.run_id}>
                  <button
                    type="button"
                    className="run-item"
                    aria-pressed={run?.runId === info.run_id}
                    onClick={() => watch(info)}
                  >
                    <span className="run-name">{info.name}</span>
                    <span className={`status-badge status-${info.status}`}>{STATUS_LABELS[info.status]}</span>
                    <span className="muted run-meta">
                      Epoch {info.epoch} of {info.total_epochs} · {info.steps_logged.toLocaleString()} steps
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
        <div className="training-main">
          {run ? (
            <>
              <div className="card">
                <RunProgress run={run} points={points} />
                {run.failure && <ErrorState error={new Error(run.failure)} />}
                {stream === 'error' && (
                  <ErrorState error={new Error('Lost the live connection to this run.')} onRetry={reconnect} />
                )}
              </div>
              {points.length > 0 && (
                <ErrorBoundary
                  fallback={(_, reset) => <ErrorState error={new Error('The charts could not be shown.')} onRetry={reset} />}
                >
                  <Suspense fallback={<LoadingState label="Loading charts…" />}>
                    <RunMetricCharts points={points} />
                  </Suspense>
                </ErrorBoundary>
              )}
              <ActivationStrips
                activations={activations}
                layerNames={{}}
                description="Sent by the training script with its latest step that included activations."
              />
            </>
          ) : (
            <div className="card">
              <GettingStarted hasRuns={(runs?.length ?? 0) > 0} />
            </div>
          )}
        </div>
      </div>
    </section>
  )
}

function useRunList() {
  const [runs, setRuns] = useState<RunInfo[] | null>(null)
  const [error, setError] = useState<unknown>(null)
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout> | null = null
    const load = async () => {
      try {
        setRuns(await listRuns(controller.signal))
        setError(null)
        timer = setTimeout(() => void load(), POLL_MS)
      } catch (caught) {
        // Stop polling on errors; "Try again" restarts it.
        if (!controller.signal.aborted) setError(caught)
      }
    }
    void load()
    return () => {
      controller.abort()
      if (timer) clearTimeout(timer)
    }
  }, [attempt])

  const refresh = useCallback(() => setAttempt((n) => n + 1), [])
  return { runs, error, refresh }
}

function RunProgress({ run, points }: { run: WatchedRun; points: RunPoint[] }) {
  const latest = points[points.length - 1]
  const epoch = latest?.epoch ?? 0
  const percent = Math.min(100, Math.round((epoch / run.totalEpochs) * 100))
  return (
    <div className="progress">
      <h2 className="run-title">{run.name}</h2>
      <span className={`status-badge status-${run.status}`} role="status">
        {STATUS_LABELS[run.status]}
      </span>
      <div
        className="progress-bar"
        role="progressbar"
        aria-label="Run progress"
        aria-valuemin={0}
        aria-valuemax={run.totalEpochs}
        aria-valuenow={Math.min(epoch, run.totalEpochs)}
      >
        <span style={{ width: `${percent}%` }} />
      </div>
      <p className="muted">
        Epoch {epoch} of {run.totalEpochs}
        {latest && ` · step ${latest.step.toLocaleString()}`}
      </p>
      {latest && Object.keys(latest.metrics).length > 0 && (
        <dl className="metric-tiles" aria-label="Latest metrics">
          {Object.entries(latest.metrics).map(([name, value]) => (
            <div key={name}>
              <dt>{name}</dt>
              <dd>{formatValue(value, 4)}</dd>
            </div>
          ))}
        </dl>
      )}
      {!latest && <p className="muted">Waiting for the first logged step…</p>}
    </div>
  )
}

// Scripts should use 127.0.0.1: on Windows, "localhost" tries IPv6 first and
// costs about 2 seconds per request.
const SCRIPT_URL = API_BASE_URL.replace('//localhost', '//127.0.0.1')

function GettingStarted({ hasRuns }: { hasRuns: boolean }) {
  return (
    <div className="getting-started">
      <h2>{hasRuns ? 'Choose a run to watch' : 'Stream a training script'}</h2>
      <p className="muted">
        Copy <code>sdk/neurovision_client.py</code> next to your script (it needs only the Python standard
        library), then log from your training loop:
      </p>
      <pre className="code-block">
        <code>{`import neurovision_client as nv

run = nv.start_run("my experiment", total_epochs=EPOCHS,
                   url="${SCRIPT_URL}")
for epoch in range(1, EPOCHS + 1):
    ...  # train one epoch
    run.log(epoch=epoch, metrics={"loss": loss, "accuracy": acc})
run.finish()`}</code>
      </pre>
      <p className="muted">
        Or run the example: <code>python sdk/examples/train_spiral.py --url {SCRIPT_URL}</code>. See{' '}
        <code>docs/live-runs.md</code> for activations and options.
      </p>
    </div>
  )
}
