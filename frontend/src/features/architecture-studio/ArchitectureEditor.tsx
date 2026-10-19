import { Plus, RotateCcw, Trash2 } from 'lucide-react'
import { useId, type ReactNode } from 'react'
import { useModelStore } from '../../stores/modelStore'
import {
  ACTIVATION_LABELS,
  ACTIVATIONS,
  LIMITS,
  type Activation,
  type ConfigErrors,
  type DenseNetworkConfig,
} from '../../types/architecture'

const DEFAULT_NEW_LAYER = { units: 4, activation: 'relu' as const }

const parseCount = (raw: string) => (raw.trim() === '' ? Number.NaN : Number(raw))
const showCount = (value: number) => (Number.isNaN(value) ? '' : String(value))

export function ArchitectureEditor({ errors }: { errors: ConfigErrors }) {
  const config = useModelStore((s) => s.config)
  const updateConfig = useModelStore((s) => s.updateConfig)
  const resetToExample = useModelStore((s) => s.resetToExample)

  const patch = (changes: Partial<DenseNetworkConfig>) => updateConfig((c) => ({ ...c, ...changes }))
  const patchLayer = (index: number, changes: Partial<DenseNetworkConfig['hidden_layers'][number]>) =>
    updateConfig((c) => ({
      ...c,
      hidden_layers: c.hidden_layers.map((layer, i) => (i === index ? { ...layer, ...changes } : layer)),
    }))
  const atMaxLayers = config.hidden_layers.length >= LIMITS.hiddenLayers

  return (
    <form className="editor" aria-label="Network architecture" onSubmit={(e) => e.preventDefault()} noValidate>
      <div className="editor-header">
        <h2>Architecture</h2>
        <button type="button" className="button" onClick={resetToExample}>
          <RotateCcw size={16} aria-hidden="true" />
          Reset to example
        </button>
      </div>

      <Field label="Name" error={errors.name}>
        {(props) => (
          <input {...props} type="text" value={config.name} maxLength={80} onChange={(e) => patch({ name: e.target.value })} />
        )}
      </Field>

      <Field label="Input units" hint={`1–${LIMITS.inputUnits}`} error={errors.input_units}>
        {(props) => (
          <input
            {...props}
            type="number"
            min={1}
            max={LIMITS.inputUnits}
            value={showCount(config.input_units)}
            onChange={(e) => patch({ input_units: parseCount(e.target.value) })}
          />
        )}
      </Field>

      <fieldset className="editor-layers">
        <legend>
          Hidden layers <span className="muted">(up to {LIMITS.hiddenLayers}, 1–{LIMITS.hiddenUnits} units each)</span>
        </legend>
        {config.hidden_layers.length === 0 && (
          <p className="muted">No hidden layers: inputs connect straight to the output layer.</p>
        )}
        {config.hidden_layers.map((layer, index) => (
          <div className="layer-row" key={index}>
            <span className="layer-row-name">Hidden {index + 1}</span>
            <Field label={`Hidden layer ${index + 1} units`} visuallyHiddenLabel error={errors[`hidden_layers.${index}.units`]}>
              {(props) => (
                <input
                  {...props}
                  type="number"
                  min={1}
                  max={LIMITS.hiddenUnits}
                  value={showCount(layer.units)}
                  onChange={(e) => patchLayer(index, { units: parseCount(e.target.value) })}
                />
              )}
            </Field>
            <Field label={`Hidden layer ${index + 1} activation`} visuallyHiddenLabel error={errors[`hidden_layers.${index}.activation`]}>
              {(props) => (
                <select
                  {...props}
                  value={layer.activation}
                  onChange={(e) => patchLayer(index, { activation: e.target.value as Activation })}
                >
                  {ACTIVATIONS.map((a) => (
                    <option key={a} value={a}>
                      {ACTIVATION_LABELS[a]}
                    </option>
                  ))}
                </select>
              )}
            </Field>
            <button
              type="button"
              className="icon-button"
              aria-label={`Remove hidden layer ${index + 1}`}
              title="Remove layer"
              onClick={() => updateConfig((c) => ({ ...c, hidden_layers: c.hidden_layers.filter((_, i) => i !== index) }))}
            >
              <Trash2 size={16} aria-hidden="true" />
            </button>
          </div>
        ))}
        {errors.hidden_layers && <p className="field-error" role="alert">{errors.hidden_layers}</p>}
        <button
          type="button"
          className="button"
          disabled={atMaxLayers}
          onClick={() => updateConfig((c) => ({ ...c, hidden_layers: [...c.hidden_layers, DEFAULT_NEW_LAYER] }))}
        >
          <Plus size={16} aria-hidden="true" />
          Add hidden layer
        </button>
        {atMaxLayers && <p className="muted">Maximum of {LIMITS.hiddenLayers} hidden layers reached.</p>}
      </fieldset>

      <Field label="Output units" hint={`1–${LIMITS.outputUnits}`} error={errors.output_units}>
        {(props) => (
          <input
            {...props}
            type="number"
            min={1}
            max={LIMITS.outputUnits}
            value={showCount(config.output_units)}
            onChange={(e) => patch({ output_units: parseCount(e.target.value) })}
          />
        )}
      </Field>

      <Field label="Output activation" error={errors.output_activation}>
        {(props) => (
          <select
            {...props}
            value={config.output_activation ?? ''}
            onChange={(e) => patch({ output_activation: (e.target.value || null) as Activation | null })}
          >
            <option value="">None (raw outputs)</option>
            {ACTIVATIONS.map((a) => (
              <option key={a} value={a}>
                {ACTIVATION_LABELS[a]}
              </option>
            ))}
          </select>
        )}
      </Field>
    </form>
  )
}

interface ControlProps {
  id: string
  'aria-invalid': boolean
  'aria-describedby'?: string
}

function Field({
  label,
  hint,
  error,
  visuallyHiddenLabel = false,
  children,
}: {
  label: string
  hint?: string
  error?: string
  visuallyHiddenLabel?: boolean
  children: (props: ControlProps) => ReactNode
}) {
  const id = useId()
  const errorId = `${id}-error`
  return (
    <div className={`field${error ? ' has-error' : ''}`}>
      <label htmlFor={id} className={visuallyHiddenLabel ? 'visually-hidden' : undefined}>
        {label}
        {hint && <span className="muted"> ({hint})</span>}
      </label>
      {children({ id, 'aria-invalid': Boolean(error), 'aria-describedby': error ? errorId : undefined })}
      {error && (
        <p id={errorId} className="field-error">
          {error}
        </p>
      )}
    </div>
  )
}
