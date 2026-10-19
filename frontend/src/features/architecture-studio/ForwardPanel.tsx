import { Dices, Play } from 'lucide-react'
import { useId, useState } from 'react'
import { runForward } from '../../api/training'
import { ErrorState } from '../../components/states/StateViews'
import { unitCount } from '../../lib/graph'
import { formatValue } from '../../lib/values'
import { currentForward, useModelStore } from '../../stores/modelStore'
import { TRAINING_LIMITS } from '../../types/training'

type RunState = { state: 'idle' } | { state: 'running' } | { state: 'error'; error: unknown }

/**
 * Runs one example through the real model. The 2D view and the inspector then
 * show the values the network actually computed for it.
 */
export function ForwardPanel() {
  const graph = useModelStore((s) => s.graph)
  const modelId = useModelStore((s) => s.modelId)
  const weightsVersion = useModelStore((s) => s.weightsVersion)
  const forward = useModelStore(currentForward)
  const stale = useModelStore((s) => s.forward !== null && currentForward(s) === null)
  const setForward = useModelStore((s) => s.setForward)
  const inputNode = graph?.nodes.find((n) => n.op_type === 'Input')
  const inputUnits = (inputNode && unitCount(inputNode)) ?? 0

  const [draft, setDraft] = useState<string[]>([])
  const [run, setRun] = useState<RunState>({ state: 'idle' })
  const id = useId()
  // Keep one field per input unit as the architecture changes.
  const values = Array.from({ length: inputUnits }, (_, i) => draft[i] ?? '0')
  const parsed = values.map((v) => (v.trim() === '' ? Number.NaN : Number(v)))
  const invalid = parsed.map((v) => !Number.isFinite(v) || Math.abs(v) > TRAINING_LIMITS.inputMagnitude)

  if (!graph || !modelId || inputUnits === 0) return null

  const execute = async () => {
    if (invalid.some(Boolean)) return
    setRun({ state: 'running' })
    try {
      const response = await runForward(modelId, [parsed])
      setForward({ modelId, inputs: parsed, response, weightsVersion })
      setRun({ state: 'idle' })
    } catch (error) {
      setRun({ state: 'error', error })
    }
  }

  const randomize = () =>
    setDraft(Array.from({ length: inputUnits }, () => (Math.random() * 2 - 1).toFixed(2)))

  const prediction = forward?.response.predictions[0]
  const outputs = forward?.response.outputs[0]

  return (
    <section className="forward-panel" aria-labelledby={`${id}-title`}>
      <h2 id={`${id}-title`}>Forward pass</h2>
      <p className="muted">Run one example through the model to see the values it computes.</p>
      <form
        className="forward-inputs"
        onSubmit={(e) => {
          e.preventDefault()
          void execute()
        }}
        noValidate
      >
        {values.map((value, i) => (
          <label key={i} className="forward-input">
            <span>x{i + 1}</span>
            <input
              type="number"
              step="any"
              value={value}
              aria-label={`Input ${i + 1}`}
              aria-invalid={invalid[i]}
              onChange={(e) => {
                const next = [...values]
                next[i] = e.target.value
                setDraft(next)
              }}
            />
          </label>
        ))}
        {invalid.some(Boolean) && (
          <p className="field-error">Each input must be a number between −1,000,000 and 1,000,000.</p>
        )}
        <div className="forward-actions">
          <button type="button" className="button" onClick={randomize}>
            <Dices size={16} aria-hidden="true" />
            Random input
          </button>
          <button type="submit" className="button button-primary" disabled={run.state === 'running' || invalid.some(Boolean)}>
            <Play size={16} aria-hidden="true" />
            {run.state === 'running' ? 'Running…' : 'Run'}
          </button>
        </div>
      </form>

      {run.state === 'error' && <ErrorState error={run.error} onRetry={() => void execute()} />}
      {stale && (
        <p className="muted" role="status">
          The weights changed since the last run, so its values are hidden. Run again to update them.
        </p>
      )}
      {forward && outputs && (
        <div className="forward-result" role="status" aria-live="polite">
          <p>
            Predicted class <strong>{prediction}</strong> for ({forward.inputs.map((v) => formatValue(v)).join(', ')})
          </p>
          <p className="muted">Outputs: {outputs.map((v) => formatValue(v)).join(', ')}</p>
        </div>
      )}
    </section>
  )
}
