import { AlertTriangle, Check, X } from 'lucide-react'
import { useState } from 'react'
import type { Capabilities, ModelGraph } from '../../types/graph'
import type { ImportReport, RunResponse } from '../../types/imports'
import type { ViewerTarget } from '../../types/maps'
import { tensorTarget } from './viewerTargets'

const CAPABILITIES: { key: keyof Capabilities; label: string; unavailable: string }[] = [
  { key: 'graph', label: 'Graph inspection', unavailable: '' },
  { key: 'inference', label: 'Running the model', unavailable: 'It cannot run here; see "Run the model" for why.' },
  { key: 'activations', label: 'Activation values', unavailable: 'Needs the model to run.' },
  { key: 'training', label: 'Training', unavailable: 'Imported models cannot be trained here.' },
  { key: 'gradients', label: 'Gradients', unavailable: 'Only available for models trained here.' },
]

const WARNINGS_SHOWN = 5

export function ImportReportPanel({
  graph,
  report,
  run = null,
  onView,
}: {
  graph: ModelGraph
  report: ImportReport
  run?: RunResponse | null
  onView?: (target: ViewerTarget) => void
}) {
  const [allWarnings, setAllWarnings] = useState(false)
  const onnx = graph.model.metadata.onnx as Record<string, unknown> | undefined
  const producer = [onnx?.producer_name, onnx?.producer_version].filter(Boolean).join(' ')
  const warnings = allWarnings ? report.warnings : report.warnings.slice(0, WARNINGS_SHOWN)

  return (
    <section className="editor import-report" aria-label="Import report">
      <h2>{graph.model.name}</h2>
      <dl className="facts">
        <dt>File</dt>
        <dd>
          {report.file_name} <span className="muted">({formatBytes(report.file_bytes)})</span>
        </dd>
        {producer && (
          <>
            <dt>Produced by</dt>
            <dd>{producer}</dd>
          </>
        )}
        <dt>Opsets</dt>
        <dd>{report.opsets.map((o) => `${o.domain} ${o.version}`).join(', ') || 'none declared'}</dd>
        <dt>Operators</dt>
        <dd>{report.operator_count.toLocaleString()}</dd>
        <dt>Tensors (edges)</dt>
        <dd>{report.edge_count.toLocaleString()}</dd>
        <dt>Initializers</dt>
        <dd>
          {report.initializer_count.toLocaleString()}{' '}
          <span className="muted">({report.parameter_count.toLocaleString()} values)</span>
        </dd>
      </dl>

      <div>
        <h3>What you can do with this model</h3>
        <ul className="capabilities">
          {CAPABILITIES.map(({ key, label, unavailable }) => {
            const ok = graph.model.capabilities[key]
            return (
              <li key={key} className={ok ? 'is-available' : 'is-unavailable'}>
                {ok ? <Check size={14} aria-hidden="true" /> : <X size={14} aria-hidden="true" />}
                <span>
                  {label}
                  <span className="visually-hidden">{ok ? ': available' : ': not available'}</span>
                  {!ok && <span className="muted"> — {unavailable}</span>}
                </span>
              </li>
            )
          })}
        </ul>
      </div>

      {report.attention.length > 0 && (
        <div role="region" aria-label="Attention">
          <h3>Attention ({report.attention.length})</h3>
          <p className="muted">
            Found by pattern: a softmax over the last axis of a square [..., S, S] tensor. Fused attention
            operators, and layers whose sequence sizes are unknown, are not detected.
          </p>
          <ul className="plain-list">
            {report.attention.map((a) => {
              const output = graph.nodes.find((n) => n.id === a.node_id)?.output_tensors.indexOf(a.tensor) ?? 0
              return (
                <li key={a.node_id}>
                  <span className="mono">{a.name}</span>{' '}
                  {a.tensor !== a.name && <span className="muted mono">→ {a.tensor} </span>}
                  <span className="muted">[{a.layout.join(', ')}]</span>{' '}
                  {onView && (
                    <button
                      type="button"
                      className="link-button"
                      disabled={!run?.retained}
                      onClick={() => {
                        // Use the run's real shape: symbolic sizes are filled in by then.
                        const measured = [...(run?.intermediates ?? []), ...(run?.outputs ?? [])].find(
                          (t) => t.node_id === a.node_id && t.name === a.tensor,
                        )
                        if (measured) onView(tensorTarget(a.node_id, Math.max(0, output), measured.shape, a.tensor, a))
                      }}
                    >
                      View attention
                    </button>
                  )}
                </li>
              )
            })}
          </ul>
          {onView && !run?.retained && <p className="muted">Run the model first to see the weights.</p>}
        </div>
      )}

      {report.unsupported_operators.length > 0 && (
        <div>
          <h3>Unsupported operators</h3>
          <ul className="plain-list">
            {report.unsupported_operators.map((op) => (
              <li key={`${op.domain}/${op.op_type}`}>
                <code>{op.op_type}</code> <span className="muted">({op.domain}) × {op.count}</span>
              </li>
            ))}
          </ul>
          <p className="muted">They stay in the graph with dashed outlines so nothing is hidden.</p>
        </div>
      )}

      {report.warnings.length > 0 && (
        <div role="region" aria-label="Import warnings">
          <h3>
            <AlertTriangle size={14} aria-hidden="true" /> Warnings ({report.warnings.length})
          </h3>
          <ul className="warnings">
            {warnings.map((warning, i) => (
              <li key={i}>{warning}</li>
            ))}
          </ul>
          {report.warnings.length > WARNINGS_SHOWN && (
            <button type="button" className="link-button" onClick={() => setAllWarnings((v) => !v)}>
              {allWarnings ? 'Show fewer' : `Show all ${report.warnings.length}`}
            </button>
          )}
        </div>
      )}
    </section>
  )
}

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}
