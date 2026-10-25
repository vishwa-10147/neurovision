import { lazy, Suspense, useCallback, useState } from 'react'
import { runMetricsCsvUrl } from '../../api/runs'
import { ErrorBoundary } from '../../components/states/ErrorBoundary'
import { ErrorState, LoadingState } from '../../components/states/StateViews'
import type { StreamState } from '../../stores/trainingStore'
import type { TrainingEvent } from '../../types/events'
import type { RunInfo, RunPoint } from '../../types/runs'
import { useEventStream } from '../training-lab/useTrainingStream'
import { COMPARE_COLORS, type ComparedRun } from './series'

const RunMetricCharts = lazy(() => import('../training-lab/MetricCharts').then((m) => ({ default: m.RunMetricCharts })))

/**
 * Overlays the metric curves of the selected runs by step. Each run's steps
 * come from its own event stream, which replays finished runs in full.
 */
export function RunComparison({ runs }: { runs: RunInfo[] }) {
  const [points, setPoints] = useState<Record<string, RunPoint[]>>({})
  const [streams, setStreams] = useState<Record<string, StreamState>>({})

  const onEvents = useCallback((runId: string, events: TrainingEvent[]) => {
    setPoints((previous) => {
      const known = previous[runId] ?? []
      let last = known[known.length - 1]?.step ?? -1
      const added: RunPoint[] = []
      for (const event of events) {
        // A reconnect replays steps already received.
        if (event.event_type !== 'training.step' || event.step <= last) continue
        added.push({ epoch: event.epoch, step: event.step, metrics: event.metrics })
        last = event.step
      }
      return added.length === 0 ? previous : { ...previous, [runId]: [...known, ...added] }
    })
  }, [])
  const onStream = useCallback((runId: string, stream: StreamState) => {
    setStreams((previous) => ({ ...previous, [runId]: stream }))
  }, [])

  const names = runs.map((info) => info.name)
  const compared: ComparedRun[] = runs.map((info) => ({
    runId: info.run_id,
    // Repeated runs of one script share a name; tell them apart in legends.
    name: names.filter((n) => n === info.name).length > 1 ? `${info.name} (${info.run_id})` : info.name,
    points: points[info.run_id] ?? [],
  }))
  const lost = compared.filter((run) => streams[run.runId] === 'error')

  return (
    <section className="card compare" aria-labelledby="compare-title">
      {runs.map((info) => (
        <RunFeed key={info.run_id} runId={info.run_id} onEvents={onEvents} onStream={onStream} />
      ))}
      <h2 id="compare-title" className="run-title">
        Comparing {runs.length} runs
      </h2>
      <p className="muted">Metric curves by step, one color per run; validation and other variants are dashed.</p>
      <ul className="compare-key" aria-label="Compared runs">
        {compared.map((run, i) => (
          <li key={run.runId}>
            <span
              className="compare-swatch"
              aria-hidden="true"
              style={{ background: COMPARE_COLORS[i % COMPARE_COLORS.length] }}
            />
            <span className="run-name">{run.name}</span>
            <span className="muted">
              {run.points.length.toLocaleString()} steps ·{' '}
              <a href={runMetricsCsvUrl(run.runId)} download aria-label={`Download metrics of ${run.name} (CSV)`}>
                CSV
              </a>
            </span>
          </li>
        ))}
      </ul>
      {lost.length > 0 && (
        <ErrorState error={new Error(`Lost the connection to ${lost.map((run) => run.name).join(', ')}.`)} />
      )}
      {compared.some((run) => run.points.length > 0) ? (
        <ErrorBoundary
          fallback={(_, reset) => <ErrorState error={new Error('The charts could not be shown.')} onRetry={reset} />}
        >
          <Suspense fallback={<LoadingState label="Loading charts…" />}>
            <RunMetricCharts runs={compared} />
          </Suspense>
        </ErrorBoundary>
      ) : (
        <p className="muted">Waiting for logged steps…</p>
      )}
    </section>
  )
}

function RunFeed({
  runId,
  onEvents,
  onStream,
}: {
  runId: string
  onEvents: (runId: string, events: TrainingEvent[]) => void
  onStream: (runId: string, stream: StreamState) => void
}) {
  const applyEvents = useCallback((events: TrainingEvent[]) => onEvents(runId, events), [runId, onEvents])
  const setStream = useCallback((stream: StreamState) => onStream(runId, stream), [runId, onStream])
  useEventStream(runId, true, applyEvents, setStream)
  return null
}
