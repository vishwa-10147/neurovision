import { AlertTriangle } from 'lucide-react'
import { useState, type ReactNode } from 'react'
import { ApiError } from '../../api/client'
import { useModelView } from '../../app/modelView'
import { useTensor } from '../../hooks/useTensor'
import { formatShape } from '../../lib/graph'
import { formatValue } from '../../lib/values'
import type { GraphEdge, GraphNode } from '../../types/graph'
import { InitializerRefSchema, type InitializerRef, type TensorValues } from '../../types/imports'

const PREVIEW_VALUES = 12
const MAX_ATTRIBUTE_TEXT = 160

/** Details of one operator in an imported graph. */
export function OperatorDetails({ node }: { node: GraphNode }) {
  const onnx = (node.metadata.onnx ?? {}) as Record<string, unknown>
  const initializers = InitializerRefSchema.array().catch([]).parse(onnx.initializers ?? [])
  const weights = new Set(initializers.map((i) => i.name))
  const attributes = Object.entries(node.attributes)
  const io = node.op_type === 'Input' || node.op_type === 'Output'
  const measured = useModelView().runValues?.[node.id] ?? []

  return (
    <section aria-label={`Operator ${node.name}`}>
      <p className="inspector-kind">{io ? 'Graph ' + node.op_type.toLowerCase() : 'Operator'}</p>
      <h3>{node.op_type}</h3>
      {node.unsupported && (
        <p className="inspector-warning" role="note">
          <AlertTriangle size={14} aria-hidden="true" /> Not a standard ONNX operator (domain{' '}
          {String(onnx.domain)}). It is kept so the graph stays complete, but NeuroVision cannot
          describe what it computes.
        </p>
      )}
      <dl className="facts">
        <Fact label="Name">
          <span className="mono">{node.name}</span>
        </Fact>
        {!io && typeof onnx.domain === 'string' && <Fact label="Domain">{onnx.domain}</Fact>}
        {node.input_tensors.length > 0 && (
          <Fact label="Inputs">
            <TensorList names={node.input_tensors} shapes={node.input_shapes} weights={weights} />
          </Fact>
        )}
        {node.output_tensors.length > 0 && (
          <Fact label="Outputs">
            <TensorList names={node.output_tensors} shapes={node.output_shapes} weights={weights} />
          </Fact>
        )}
        {attributes.length > 0 && (
          <Fact label="Attributes">
            <ul className="plain-list">
              {attributes.map(([key, value]) => (
                <li key={key}>
                  <code>{key}</code> = <span className="mono">{attributeText(value)}</span>
                </li>
              ))}
            </ul>
          </Fact>
        )}
        {node.parameter_summary && (
          <Fact label="Weight values">{node.parameter_summary.parameter_count.toLocaleString()}</Fact>
        )}
        {typeof onnx.doc === 'string' && onnx.doc && <Fact label="Documentation">{onnx.doc}</Fact>}
      </dl>
      {measured.length > 0 && (
        <div className="initializers" aria-label="Output in the last run">
          <h4>Output in the last run</h4>
          {measured.map((tensor) => (
            <MeasuredTensor key={tensor.name} tensor={tensor} />
          ))}
        </div>
      )}
      {initializers.length > 0 && (
        <div className="initializers">
          <h4>Weights and constants</h4>
          {initializers.map((init) => (
            <InitializerView key={init.tensor_id} init={init} />
          ))}
        </div>
      )}
    </section>
  )
}

/** Details of one edge in an imported graph: the tensor passed along it. */
export function TensorEdgeDetails({ edge, source, target }: { edge: GraphEdge; source: GraphNode; target: GraphNode }) {
  const index = edge.tensor_name ? source.output_tensors.indexOf(edge.tensor_name) : -1
  const shape = index >= 0 ? source.output_shapes[index] : undefined
  const { runValues } = useModelView()
  const measured = runValues?.[source.id]?.find((t) => t.name === edge.tensor_name)
  return (
    <section aria-label="Tensor">
      <p className="inspector-kind">Tensor</p>
      <h3 className="mono">{edge.tensor_name ?? edge.id}</h3>
      <dl className="facts">
        <Fact label="From">
          {source.op_type} <span className="muted mono">{source.name}</span>
        </Fact>
        <Fact label="To">
          {target.op_type} <span className="muted mono">{target.name}</span>
        </Fact>
        <Fact label="Shape">{shape === undefined ? 'unknown' : formatShape(shape)}</Fact>
        {!measured && (
          <Fact label="Values">
            {runValues ? 'Not captured in the last run.' : 'Not computed yet. Run the model to see this tensor.'}
          </Fact>
        )}
      </dl>
      {measured && (
        <div className="initializers" aria-label="Values in the last run">
          <h4>Values in the last run</h4>
          <MeasuredTensor tensor={measured} />
        </div>
      )}
    </section>
  )
}

/** A tensor measured in the last run: its real shape, summary, and first values. */
function MeasuredTensor({ tensor }: { tensor: TensorValues }) {
  const s = tensor.summary
  return (
    <div className="initializer">
      <p>
        <span className="mono">{tensor.name}</span>{' '}
        <span className="muted">
          {formatShape(tensor.shape)} {tensor.dtype} · {tensor.count.toLocaleString()} values
        </span>
      </p>
      {s ? (
        <p className="mono">
          min {formatValue(s.min, 4)}, max {formatValue(s.max, 4)}, mean {formatValue(s.mean, 4)}, std{' '}
          {formatValue(s.std, 4)}
        </p>
      ) : (
        <p className="muted">No finite values.</p>
      )}
      {tensor.non_finite_count > 0 && (
        <p className="inspector-warning">{tensor.non_finite_count.toLocaleString()} values are NaN or infinite.</p>
      )}
      <p className="mono">
        {tensor.values
          .slice(0, PREVIEW_VALUES)
          .map((v) => (v === null ? 'NaN/∞' : formatValue(v, 4)))
          .join(', ')}
        {tensor.count > PREVIEW_VALUES ? ', …' : ''}
      </p>
    </div>
  )
}

function TensorList({
  names,
  shapes,
  weights,
}: {
  names: string[]
  shapes: (number | string | null)[][] | (null | (number | string | null)[])[]
  weights: Set<string>
}) {
  return (
    <ul className="plain-list">
      {names.map((name, i) => (
        <li key={`${name}-${i}`}>
          <span className="mono">{name}</span>{' '}
          <span className="muted">
            {shapes.length > i ? formatShape(shapes[i] ?? null) : 'shape unknown'}
            {weights.has(name) ? ' · weights' : ''}
          </span>
        </li>
      ))}
    </ul>
  )
}

/** One initializer: shape always, values fetched on request (bounded). */
function InitializerView({ init }: { init: InitializerRef }) {
  const [open, setOpen] = useState(false)
  const { modelId, weightsVersion } = useModelView()
  const tensor = useTensor(open ? modelId : null, init.tensor_id, weightsVersion)
  const count = init.shape.reduce((a, b) => a * b, 1)

  return (
    <div className="initializer">
      <p>
        <span className="mono">{init.name}</span>{' '}
        <span className="muted">
          {formatShape(init.shape)} {init.dtype} · {count.toLocaleString()} values
        </span>
      </p>
      {init.external ? (
        <p className="muted">Stored in an external data file that was not uploaded; values unavailable.</p>
      ) : !open ? (
        <button type="button" className="link-button" onClick={() => setOpen(true)}>
          Show values
        </button>
      ) : tensor.state === 'loading' || tensor.state === 'idle' ? (
        <p className="muted">Loading…</p>
      ) : tensor.state === 'error' ? (
        <p className="field-error">
          {tensor.error instanceof ApiError || tensor.error instanceof Error ? tensor.error.message : 'Could not load values.'}
        </p>
      ) : (
        <div className="tensor-values">
          {tensor.data.summary ? (
            <p className="mono">
              min {formatValue(tensor.data.summary.min, 4)}, max {formatValue(tensor.data.summary.max, 4)}, mean{' '}
              {formatValue(tensor.data.summary.mean, 4)}, std {formatValue(tensor.data.summary.std, 4)}
            </p>
          ) : (
            <p className="muted">No finite values.</p>
          )}
          {tensor.data.non_finite_count > 0 && (
            <p className="inspector-warning">
              {tensor.data.non_finite_count.toLocaleString()} values are NaN or infinite (left out of the summary).
            </p>
          )}
          <p className="mono">
            {tensor.data.values
              .slice(0, PREVIEW_VALUES)
              .map((v) => (v === null ? 'NaN/∞' : formatValue(v, 4)))
              .join(', ')}
            {tensor.data.count > PREVIEW_VALUES ? ', …' : ''}
          </p>
          <p className="muted">
            First {Math.min(PREVIEW_VALUES, tensor.data.count)} of {tensor.data.count.toLocaleString()} values, row-major.
          </p>
        </div>
      )}
    </div>
  )
}

function attributeText(value: unknown): string {
  const text = typeof value === 'string' ? value : JSON.stringify(value)
  return text.length > MAX_ATTRIBUTE_TEXT ? `${text.slice(0, MAX_ATTRIBUTE_TEXT - 1)}…` : text
}

function Fact({ label, children }: { label: string; children: ReactNode }) {
  return (
    <>
      <dt>{label}</dt>
      <dd>{children}</dd>
    </>
  )
}
