import { Play, Square } from 'lucide-react'
import { useId, type ReactNode } from 'react'
import { useTrainingStore } from '../../stores/trainingStore'
import { DATASETS, FIXED_CLASSES, type DatasetName, type TrainingConfig } from '../../types/training'

export type TrainingErrors = Record<string, string>

const DATASET_LABELS: Record<DatasetName, string> = {
  moons: 'Two moons (2 classes)',
  circles: 'Circles (2 classes)',
  spiral: 'Spiral (one arm per output)',
}

const SPEEDS = [
  { label: 'Full speed', value: 0 },
  { label: 'Watchable (20 ms per epoch)', value: 20 },
  { label: 'Slow (100 ms per epoch)', value: 100 },
]

const num = (raw: string) => (raw.trim() === '' ? Number.NaN : Number(raw))
const show = (v: number) => (Number.isNaN(v) ? '' : String(v))

interface Props {
  errors: TrainingErrors
  layers: { id: string; name: string }[]
  outputUnits: number
  running: boolean
  starting: boolean
  onStart: () => void
  onCancel: () => void
}

export function TrainingForm({ errors, layers, outputUnits, running, starting, onStart, onCancel }: Props) {
  const config = useTrainingStore((s) => s.config)
  const update = useTrainingStore((s) => s.updateConfig)
  const captureLayers = useTrainingStore((s) => s.captureLayers)
  const setCaptureLayers = useTrainingStore((s) => s.setCaptureLayers)
  const captured = new Set(captureLayers ?? layers.map((l) => l.id))
  const invalid = Object.keys(errors).length > 0
  const numberField = (key: keyof TrainingConfig, label: string, step: string) => (
    <Field label={label} error={errors[key]}>
      {(props) => (
        <input
          {...props}
          type="number"
          step={step}
          value={show(config[key] as number)}
          disabled={running}
          onChange={(e) => update({ [key]: num(e.target.value) })}
        />
      )}
    </Field>
  )

  return (
    <form
      className="editor training-form"
      aria-label="Training settings"
      noValidate
      onSubmit={(e) => {
        e.preventDefault()
        if (!running) onStart()
      }}
    >
      <h2>Training settings</h2>
      <Field label="Dataset" error={errors.dataset}>
        {(props) => (
          <select
            {...props}
            value={config.dataset}
            disabled={running}
            onChange={(e) => update({ dataset: e.target.value as DatasetName })}
          >
            {DATASETS.map((d) => {
              const fixed = FIXED_CLASSES[d]
              const fits = fixed === undefined || outputUnits <= fixed
              return (
                <option key={d} value={d} disabled={!fits}>
                  {DATASET_LABELS[d]}
                  {fits ? '' : ' — needs 1 or 2 outputs'}
                </option>
              )
            })}
          </select>
        )}
      </Field>
      <div className="field-row">
        {numberField('samples', 'Samples', '50')}
        {numberField('noise', 'Noise', '0.05')}
      </div>
      <div className="field-row">
        {numberField('epochs', 'Epochs', '10')}
        {numberField('batch_size', 'Batch size', '1')}
      </div>
      <div className="field-row">
        {numberField('learning_rate', 'Learning rate', '0.01')}
        <Field label="Optimizer" error={errors.optimizer}>
          {(props) => (
            <select
              {...props}
              value={config.optimizer}
              disabled={running}
              onChange={(e) => update({ optimizer: e.target.value as TrainingConfig['optimizer'] })}
            >
              <option value="adam">Adam</option>
              <option value="sgd">SGD with momentum</option>
            </select>
          )}
        </Field>
      </div>
      <div className="field-row">
        {numberField('validation_split', 'Validation share', '0.05')}
        {numberField('seed', 'Seed', '1')}
      </div>
      <Field label="Speed" error={errors.epoch_delay_ms}>
        {(props) => (
          <select
            {...props}
            value={config.epoch_delay_ms}
            disabled={running}
            onChange={(e) => update({ epoch_delay_ms: Number(e.target.value) })}
          >
            {SPEEDS.map((s) => (
              <option key={s.value} value={s.value}>
                {s.label}
              </option>
            ))}
          </select>
        )}
      </Field>
      <fieldset disabled={running}>
        <legend>Stream activations for</legend>
        {layers.map((layer) => (
          <label key={layer.id} className="checkbox">
            <input
              type="checkbox"
              checked={captured.has(layer.id)}
              onChange={(e) => {
                const next = new Set(captured)
                if (e.target.checked) next.add(layer.id)
                else next.delete(layer.id)
                setCaptureLayers(layers.map((l) => l.id).filter((id) => next.has(id)))
              }}
            />
            {layer.name}
          </label>
        ))}
        {errors.capture_layers && <p className="field-error">{errors.capture_layers}</p>}
      </fieldset>

      <div className="training-actions">
        {running ? (
          <button type="button" className="button button-danger" onClick={onCancel}>
            <Square size={16} aria-hidden="true" />
            Cancel training
          </button>
        ) : (
          <button type="submit" className="button button-primary" disabled={invalid || starting}>
            <Play size={16} aria-hidden="true" />
            {starting ? 'Starting…' : 'Start training'}
          </button>
        )}
      </div>
    </form>
  )
}

interface ControlProps {
  id: string
  'aria-invalid': boolean
  'aria-describedby'?: string
}

function Field({ label, error, children }: { label: string; error?: string; children: (p: ControlProps) => ReactNode }) {
  const id = useId()
  return (
    <div className={`field${error ? ' has-error' : ''}`}>
      <label htmlFor={id}>{label}</label>
      {children({ id, 'aria-invalid': Boolean(error), 'aria-describedby': error ? `${id}-error` : undefined })}
      {error && (
        <p id={`${id}-error`} className="field-error">
          {error}
        </p>
      )}
    </div>
  )
}
