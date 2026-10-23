/**
 * Training-health views for an external run (ADR 0008): alerts raised by the
 * server's rules, and the latest per-layer statistics the script measured.
 */
import { AlertOctagon, AlertTriangle, Info } from 'lucide-react'
import { formatValue } from '../../lib/values'
import type { RunAlert, StepHealth } from '../../types/events'

/** A commonly cited healthy range for the update ratio ‖Δw‖/‖w‖. */
const HEALTHY_UPDATE_RATIO: [number, number] = [1e-4, 1e-2]

const SEVERITY_ICONS = { critical: AlertOctagon, warning: AlertTriangle, info: Info } as const
const SEVERITY_LABELS = { critical: 'Critical', warning: 'Warning', info: 'Info' } as const

export function RunAlerts({ alerts }: { alerts: RunAlert[] }) {
  return (
    <section className="card run-alerts" aria-label="Health alerts">
      <h3>Health alerts</h3>
      {alerts.length === 0 ? (
        <p className="muted">
          No problems detected so far. Alerts appear here when gradients explode or vanish, units die or
          saturate, updates are far too large or small, or the loss turns NaN, spikes, plateaus or starts to
          overfit.
        </p>
      ) : (
        <ul>
          {alerts.map((alert) => {
            const Icon = SEVERITY_ICONS[alert.severity]
            return (
              <li key={alert.alert_id} className={`alert alert-${alert.severity}`}>
                <Icon size={18} aria-hidden="true" />
                <div>
                  <p className="alert-message">
                    <span className="visually-hidden">{SEVERITY_LABELS[alert.severity]}: </span>
                    {alert.message}
                  </p>
                  <p className="muted alert-advice">{alert.advice}</p>
                  <p className="muted alert-meta">
                    Step {alert.step.toLocaleString()}
                    {alert.layer && ` · ${alert.layer}`}
                  </p>
                </div>
              </li>
            )
          })}
        </ul>
      )}
    </section>
  )
}

function fmt(value: number | null | undefined, digits = 3): string {
  return value === null || value === undefined ? '–' : formatValue(value, digits)
}

function percent(value: number | null | undefined): string {
  return value === null || value === undefined ? '–' : `${Math.round(value * 100)}%`
}

function ratioClass(ratio: number | null): string {
  if (ratio === null) return ''
  const [low, high] = HEALTHY_UPDATE_RATIO
  return ratio > 0.1 || ratio < 1e-6 ? 'value-bad' : ratio < low || ratio > high ? 'value-warn' : 'value-ok'
}

export function LayerHealth({ health }: { health: { step: number; stats: StepHealth } | null }) {
  if (!health) return null
  const { stats, step } = health
  const parameters = Object.entries(stats.parameters)
  const activations = Object.entries(stats.activations)
  return (
    <section className="card layer-health" aria-label="Layer health">
      <h3>Layer health</h3>
      <p className="muted">
        Measured by the training script at step {step.toLocaleString()}
        {stats.grad_norm !== null && ` · total gradient norm ${fmt(stats.grad_norm)}`}
        {stats.nonfinite_grads > 0 && ` · ${stats.nonfinite_grads.toLocaleString()} NaN/∞ gradient values`}.
      </p>
      {parameters.length > 0 && (
        <div className="table-scroll">
          <table className="health-table">
            <caption className="visually-hidden">Parameter groups</caption>
            <thead>
              <tr>
                <th scope="col">Parameters</th>
                <th scope="col">Count</th>
                <th scope="col">Gradient norm</th>
                <th scope="col">Weight norm</th>
                <th scope="col">
                  <abbr title="‖Δw‖ / ‖w‖ since the previous measurement; around 1e-3 is a common target">
                    Update ratio
                  </abbr>
                </th>
              </tr>
            </thead>
            <tbody>
              {parameters.map(([name, group]) => (
                <tr key={name}>
                  <th scope="row">{name}</th>
                  <td>{group.parameters.toLocaleString()}</td>
                  <td>{fmt(group.grad_norm)}</td>
                  <td>{fmt(group.weight_norm)}</td>
                  <td className={ratioClass(group.update_ratio)}>
                    {group.update_ratio === null ? '–' : group.update_ratio.toExponential(1)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {activations.length > 0 && (
        <div className="table-scroll">
          <table className="health-table">
            <caption className="visually-hidden">Activations on the probe batch</caption>
            <thead>
              <tr>
                <th scope="col">Activation</th>
                <th scope="col">Type</th>
                <th scope="col">Mean</th>
                <th scope="col">Std</th>
                <th scope="col">Dead units</th>
                <th scope="col">Saturated</th>
              </tr>
            </thead>
            <tbody>
              {activations.map(([name, layer]) => (
                <tr key={name}>
                  <th scope="row">{name}</th>
                  <td>{layer.kind}</td>
                  <td>{fmt(layer.mean)}</td>
                  <td>{fmt(layer.std)}</td>
                  <td className={(layer.dead_fraction ?? 0) >= 0.5 ? 'value-bad' : ''}>{percent(layer.dead_fraction)}</td>
                  <td className={(layer.saturated_fraction ?? 0) >= 0.5 ? 'value-bad' : ''}>
                    {percent(layer.saturated_fraction)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}
