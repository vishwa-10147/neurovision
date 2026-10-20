import { activationFill, formatValue } from '../../lib/values'
import type { TrainingJob } from '../../stores/trainingStore'
import type { EpochMetrics } from '../../types/training'

const STATUS_LABELS = {
  running: 'Training',
  completed: 'Completed',
  cancelled: 'Cancelled',
  failed: 'Failed',
} as const

export function TrainingProgress({ job, history }: { job: TrainingJob; history: EpochMetrics[] }) {
  const epoch = history[history.length - 1]?.epoch ?? 0
  const percent = Math.round((epoch / job.totalEpochs) * 100)
  const latest = history[history.length - 1]
  return (
    <div className="progress">
      <span className={`status-badge status-${job.status}`} role="status">
        {STATUS_LABELS[job.status]}
      </span>
      <div
        className="progress-bar"
        role="progressbar"
        aria-label="Training progress"
        aria-valuemin={0}
        aria-valuemax={job.totalEpochs}
        aria-valuenow={epoch}
      >
        <span style={{ width: `${percent}%` }} />
      </div>
      <p className="muted">
        Epoch {epoch} of {job.totalEpochs}
      </p>
      {latest && (
        <dl className="metric-tiles" aria-label="Latest metrics">
          <div>
            <dt>Training loss</dt>
            <dd>{formatValue(latest.loss, 4)}</dd>
          </div>
          <div>
            <dt>Validation loss</dt>
            <dd>{formatValue(latest.val_loss, 4)}</dd>
          </div>
          <div>
            <dt>Training accuracy</dt>
            <dd>{(latest.accuracy * 100).toFixed(1)}%</dd>
          </div>
          <div>
            <dt>Validation accuracy</dt>
            <dd>{(latest.val_accuracy * 100).toFixed(1)}%</dd>
          </div>
        </dl>
      )}
    </div>
  )
}

const PROBE_DESCRIPTION = 'Measured on the first validation example after each epoch.'

export function ActivationStrips({
  activations,
  layerNames,
  description = PROBE_DESCRIPTION,
}: {
  activations: Record<string, number[]>
  layerNames: Record<string, string>
  /** Where the values come from. */
  description?: string
}) {
  const entries = Object.entries(activations)
  if (entries.length === 0) return null
  return (
    <section className="activation-strips card" aria-label="Streamed activations">
      <h3>Activations during training</h3>
      <p className="muted">
        {description} Colors are scaled within each layer: orange negative, blue positive. Hover a
        cell for its value.
      </p>
      {entries.map(([nodeId, values]) => {
        const maxAbs = Math.max(0, ...values.map(Math.abs))
        return (
          <div key={nodeId} className="strip">
            <span>{layerNames[nodeId] ?? nodeId}</span>
            <div className="strip-cells" role="list" aria-label={`${layerNames[nodeId] ?? nodeId} activations`}>
              {values.map((v, i) => (
                <span
                  key={i}
                  role="listitem"
                  title={`Unit ${i + 1}: ${formatValue(v)}`}
                  aria-label={`Unit ${i + 1}: ${formatValue(v)}`}
                  style={{ background: activationFill(v, maxAbs) }}
                />
              ))}
            </div>
          </div>
        )
      })}
    </section>
  )
}
