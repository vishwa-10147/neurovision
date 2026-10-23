import { Play } from 'lucide-react'
import { useId, useState } from 'react'
import { runImported } from '../../api/imports'
import { ErrorState, LoadingState } from '../../components/states/StateViews'
import { formatValue } from '../../lib/values'
import { useExplorerStore } from '../../stores/explorerStore'
import { FILLS, RUN_LIMITS, type ImportReport, type RunInput, type TensorValues } from '../../types/imports'
import { PhotoPicker } from './PhotoPicker'
import { initialDraft, photoSize, toRunInput, topValues, type Draft, type Mode } from './runInputs'

const MODE_LABELS: Record<Mode, string> = {
  random: 'Random (seeded)',
  zeros: 'All zeros',
  ones: 'All ones',
  range: 'Evenly spaced −1 → 1',
  custom: 'Type values',
  image: 'Photo',
}
export function RunPanel({ modelId, report }: { modelId: string; report: ImportReport }) {
  const run = useExplorerStore((s) => s.run)
  const setRun = useExplorerStore((s) => s.setRun)
  const capture = useExplorerStore((s) => s.capture)
  const [drafts, setDrafts] = useState<Record<string, Draft>>(() =>
    Object.fromEntries(report.inputs.map((spec) => [spec.name, initialDraft(spec)])),
  )
  const [status, setStatus] = useState<{ state: 'idle' | 'running' } | { state: 'error'; error: unknown }>({ state: 'idle' })
  const id = useId()

  if (report.inference_blockers.length > 0) {
    return (
      <section className="editor run-panel" aria-labelledby={`${id}-title`}>
        <h2 id={`${id}-title`}>Run the model</h2>
        <p className="muted">This model cannot be run here:</p>
        <ul className="warnings">
          {report.inference_blockers.map((b) => (
            <li key={b}>{b}</li>
          ))}
        </ul>
        <p className="muted">Its graph and weights can still be inspected.</p>
      </section>
    )
  }

  const checked = report.inputs.map((spec) => ({ spec, result: toRunInput(spec, drafts[spec.name]) }))
  const problem = checked.find((c) => 'error' in c.result)
  const update = (name: string, changes: Partial<Draft>) =>
    setDrafts((d) => ({ ...d, [name]: { ...d[name], ...changes } }))

  const execute = async () => {
    if (problem) return
    const inputs = Object.fromEntries(checked.map(({ spec, result }) => [spec.name, (result as { input: RunInput }).input]))
    setStatus({ state: 'running' })
    try {
      setRun(await runImported(modelId, inputs, undefined, capture), inputs)
      setStatus({ state: 'idle' })
    } catch (error) {
      setStatus({ state: 'error', error })
    }
  }

  return (
    <section className="editor run-panel" aria-labelledby={`${id}-title`}>
      <h2 id={`${id}-title`}>Run the model</h2>
      <p className="muted">
        Runs in ONNX Runtime in a separate, time-limited process on the server, and shows the values
        each operator really produces.
      </p>
      <form
        className="training-form"
        noValidate
        onSubmit={(e) => {
          e.preventDefault()
          void execute()
        }}
      >
        {report.inputs.map((spec) => {
          const draft = drafts[spec.name]
          const count = draft.dims.reduce((a, d) => a * (Number(d) || 0), 1)
          return (
            <fieldset key={spec.name} className="run-input">
              <legend>
                <span className="mono">{spec.name}</span> <span className="muted">{spec.dtype}</span>
              </legend>
              <div className="dims" role="group" aria-label={`Shape of ${spec.name}`}>
                {draft.dims.map((dim, axis) => {
                  const declared = spec.shape?.[axis]
                  const fixed = typeof declared === 'number'
                  return (
                    <label key={axis} className="dim">
                      <span className="muted">{fixed ? `dim ${axis}` : String(declared ?? `dim ${axis}`)}</span>
                      <input
                        type="number"
                        min={1}
                        value={dim}
                        disabled={fixed}
                        aria-label={`${spec.name} dimension ${axis}${fixed ? ' (fixed)' : ''}`}
                        onChange={(e) => {
                          const dims = [...draft.dims]
                          dims[axis] = e.target.value
                          update(spec.name, { dims })
                        }}
                      />
                    </label>
                  )
                })}
              </div>
              <label className="field">
                <span>Values</span>
                <select
                  value={draft.mode}
                  aria-label={`Values for ${spec.name}`}
                  onChange={(e) => update(spec.name, { mode: e.target.value as Mode })}
                >
                  {[...FILLS, 'custom' as const, ...(photoSize(draft.dims) ? ['image' as const] : [])].map((m) => (
                    <option key={m} value={m} disabled={m === 'custom' && count > RUN_LIMITS.maxTypedValues}>
                      {MODE_LABELS[m]}
                      {m === 'custom' && count > RUN_LIMITS.maxTypedValues ? ' (too many values)' : ''}
                    </option>
                  ))}
                </select>
              </label>
              {draft.mode === 'random' && (
                <label className="field">
                  <span>Seed</span>
                  <input
                    type="number"
                    min={0}
                    value={draft.seed}
                    aria-label={`Seed for ${spec.name}`}
                    onChange={(e) => update(spec.name, { seed: e.target.value })}
                  />
                </label>
              )}
              {draft.mode === 'image' && (
                <PhotoPicker
                  inputName={spec.name}
                  size={photoSize(draft.dims)}
                  draft={draft}
                  onChange={(changes) => update(spec.name, changes)}
                />
              )}
              {draft.mode === 'custom' && (
                <label className="field">
                  <span>{count.toLocaleString()} values, row-major, separated by commas or spaces</span>
                  <textarea
                    rows={3}
                    value={draft.custom}
                    aria-label={`Typed values for ${spec.name}`}
                    onChange={(e) => update(spec.name, { custom: e.target.value })}
                  />
                </label>
              )}
            </fieldset>
          )
        })}
        {problem && 'error' in problem.result && <p className="field-error">{problem.result.error}</p>}
        <div className="training-actions">
          <button type="submit" className="button button-primary" disabled={Boolean(problem) || status.state === 'running'}>
            <Play size={16} aria-hidden="true" />
            {status.state === 'running' ? 'Running…' : 'Run'}
          </button>
        </div>
      </form>

      {status.state === 'running' && <LoadingState label="Running in ONNX Runtime…" />}
      {status.state === 'error' && <ErrorState error={status.error} onRetry={() => void execute()} />}
      {run && run.model_id === modelId && status.state !== 'running' && <RunResults />}
    </section>
  )
}

function RunResults() {
  const run = useExplorerStore((s) => s.run)
  if (!run) return null
  return (
    <div className="run-results" role="status" aria-live="polite" aria-label="Run results">
      <p>
        <strong>{run.elapsed_ms.toFixed(run.elapsed_ms < 10 ? 2 : 1)} ms</strong> in {run.runtime}
        {run.intermediates.length > 0 && ` · ${run.intermediates.length} intermediate tensors captured`}
      </p>
      {run.notes.map((note) => (
        <p key={note} className="muted">
          {note}
        </p>
      ))}
      {run.outputs.map((tensor) => (
        <OutputView key={tensor.name} tensor={tensor} />
      ))}
      <p className="muted">Select an operator in the graph to see its output from this run.</p>
    </div>
  )
}

function OutputView({ tensor }: { tensor: TensorValues }) {
  const top = topValues(tensor)
  return (
    <div className="initializer">
      <p>
        Output <span className="mono">{tensor.name}</span>{' '}
        <span className="muted">
          [{tensor.shape.join(' × ')}] {tensor.dtype}
        </span>
      </p>
      {top ? (
        <p>
          Top {top.length} of the first row (index: value):{' '}
          <span className="mono">{top.map(([i, v]) => `${i}: ${formatValue(v, 4)}`).join(', ')}</span>
        </p>
      ) : (
        <p className="mono">
          {tensor.values
            .slice(0, 8)
            .map((v) => (v === null ? 'NaN/∞' : formatValue(v, 4)))
            .join(', ')}
          {tensor.count > 8 ? ', …' : ''}
        </p>
      )}
    </div>
  )
}
