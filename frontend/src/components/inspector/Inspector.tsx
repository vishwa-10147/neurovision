import { ChevronLeft, ChevronRight } from 'lucide-react'
import type { ReactNode } from 'react'
import { ApiError } from '../../api/client'
import { useTensor, type TensorState } from '../../hooks/useTensor'
import { findNode, formatShape, showsUnits, unitCount } from '../../lib/graph'
import { formatValue } from '../../lib/values'
import { useModelView } from '../../app/modelView'
import type { Selection } from '../../stores/modelStore'
import { ACTIVATION_LABELS, type Activation } from '../../types/architecture'
import type { GraphNode, ModelGraph } from '../../types/graph'
import type { TensorResponse } from '../../types/training'
import { EmptyState } from '../states/StateViews'
import { OperatorDetails, TensorEdgeDetails } from './OperatorDetails'

const RUN_FORWARD_HINT = 'Run a forward pass to see the value this model computes.'
const TRAIN_HINT = 'Train the model to see gradients.'

const fmt = (v: number) => formatValue(v, 4)

export function Inspector({ graph }: { graph: ModelGraph | null }) {
  const { selection } = useModelView()
  return (
    <aside className="inspector" aria-labelledby="inspector-title">
      <h2 id="inspector-title">Inspector</h2>
      {graph && selection ? (
        <SelectionDetails graph={graph} selection={selection} />
      ) : (
        <EmptyState title="Nothing selected">
          Click a layer, unit, or connection in the diagram. With the keyboard, Tab to a layer and press
          Enter.
        </EmptyState>
      )}
    </aside>
  )
}

function SelectionDetails({ graph, selection }: { graph: ModelGraph; selection: NonNullable<Selection> }) {
  if (!showsUnits(graph)) {
    // Imported operator graphs: operators and the tensors between them.
    if (selection.kind === 'connection') {
      const edge = graph.edges.find((e) => e.id === selection.edgeId)
      const source = edge && findNode(graph, edge.source)
      const target = edge && findNode(graph, edge.target)
      return edge && source && target ? <TensorEdgeDetails edge={edge} source={source} target={target} /> : null
    }
    const node = findNode(graph, selection.nodeId)
    return node ? <OperatorDetails node={node} /> : null
  }
  if (selection.kind === 'connection') {
    const edge = graph.edges.find((e) => e.id === selection.edgeId)
    const source = edge && findNode(graph, edge.source)
    const target = edge && findNode(graph, edge.target)
    if (!edge || !source || !target) return null
    return <ConnectionDetails source={source} target={target} tensor={edge.tensor_name} selection={selection} />
  }
  const node = findNode(graph, selection.nodeId)
  if (!node) return null
  if (selection.kind === 'unit') return <UnitDetails graph={graph} node={node} index={selection.index} />
  return <LayerDetails node={node} />
}

/** Tensors of the selected model, refreshed whenever its weights change. */
function useModelTensor(tensorId: string | null): TensorState {
  const { modelId, weightsVersion: version } = useModelView()
  return useTensor(modelId, tensorId, version)
}

/** Measured values for one layer from the latest forward pass, if current. */
function useLayerValues(nodeId: string) {
  const { forward } = useModelView()
  const values = forward?.response.activations[nodeId]
  return values ? { pre: values.pre?.[0] ?? null, post: values.post[0] } : null
}

function LayerDetails({ node }: { node: GraphNode }) {
  const units = unitCount(node)
  const inFeatures = numberAttr(node, 'in_features')
  const outFeatures = numberAttr(node, 'out_features')
  const learned = node.op_type !== 'Input'
  const weights = useModelTensor(learned ? `${node.id}.weight` : null)
  const gradients = useModelTensor(learned ? `${node.id}.weight.grad` : null)
  const values = useLayerValues(node.id)

  return (
    <section aria-label={`Layer ${node.name}`}>
      <p className="inspector-kind">Layer</p>
      <h3>{node.name}</h3>
      <dl className="facts">
        <Fact label="Operator">{node.op_type}</Fact>
        <Fact label="Units">{units ?? 'Unknown'}</Fact>
        {learned && <Fact label="Activation">{activationLabel(node)}</Fact>}
        {node.input_shapes.length > 0 && <Fact label="Input shape">{node.input_shapes.map(formatShape).join(', ')}</Fact>}
        {node.output_shapes.length > 0 && <Fact label="Output shape">{node.output_shapes.map(formatShape).join(', ')}</Fact>}
        <Fact label="Parameters">
          {node.parameter_summary ? (
            <>
              {node.parameter_summary.parameter_count.toLocaleString()}
              {inFeatures !== null && outFeatures !== null && (
                <span className="muted">
                  {' '}
                  = {inFeatures} × {outFeatures} weights + {outFeatures} biases
                </span>
              )}
            </>
          ) : learned ? (
            'Unknown'
          ) : (
            'None (inputs are data, not learned)'
          )}
        </Fact>
        {learned && (
          <Fact label="Weights">
            <TensorView state={weights} render={summaryText} />
          </Fact>
        )}
        {learned && (
          <Fact label="Weight gradients">
            <TensorView state={gradients} render={summaryText} missing={TRAIN_HINT} />
          </Fact>
        )}
        <Fact label={learned ? 'Activations' : 'Values'}>
          {values ? <span className="mono">{values.post.map(fmt).join(', ')}</span> : RUN_FORWARD_HINT}
        </Fact>
      </dl>
    </section>
  )
}

function UnitDetails({ graph, node, index }: { graph: ModelGraph; node: GraphNode; index: number }) {
  const { select } = useModelView()
  const count = unitCount(node) ?? 0
  const inFeatures = numberAttr(node, 'in_features')
  const learned = node.op_type !== 'Input'
  const bias = useModelTensor(learned ? `${node.id}.bias` : null)
  const values = useLayerValues(node.id)
  const outgoing = graph.edges
    .filter((e) => e.source === node.id)
    .reduce((sum, e) => {
      const target = findNode(graph, e.target)
      return sum + ((target && unitCount(target)) ?? 0)
    }, 0)
  const step = (delta: number) => select({ kind: 'unit', nodeId: node.id, index: (index + delta + count) % count })

  return (
    <section aria-label={`Unit ${index + 1} of ${node.name}`}>
      <p className="inspector-kind">Unit</p>
      <div className="inspector-title-row">
        <h3>
          {node.name} · unit {index + 1} <span className="muted">of {count}</span>
        </h3>
        <div className="stepper">
          <button type="button" className="icon-button" aria-label="Previous unit" onClick={() => step(-1)}>
            <ChevronLeft size={16} aria-hidden="true" />
          </button>
          <button type="button" className="icon-button" aria-label="Next unit" onClick={() => step(1)}>
            <ChevronRight size={16} aria-hidden="true" />
          </button>
        </div>
      </div>
      <dl className="facts">
        {learned && <Fact label="Activation">{activationLabel(node)}</Fact>}
        {inFeatures !== null && <Fact label="Incoming connections">{inFeatures}</Fact>}
        <Fact label="Outgoing connections">{outgoing}</Fact>
        {inFeatures !== null && (
          <Fact label="Parameters">
            {inFeatures + 1} <span className="muted">= {inFeatures} weights + 1 bias</span>
          </Fact>
        )}
        {learned && (
          <Fact label="Bias">
            <TensorView state={bias} render={(t) => <span className="mono">{valueText(t.values[index])}</span>} />
          </Fact>
        )}
        {values && learned && values.pre && (
          <Fact label="Weighted sum (W·x + b)">
            <span className="mono">{fmt(values.pre[index])}</span>
          </Fact>
        )}
        <Fact label={learned ? 'Activation value' : 'Input value'}>
          {values ? <span className="mono">{fmt(values.post[index])}</span> : RUN_FORWARD_HINT}
        </Fact>
      </dl>
      <button type="button" className="link-button" onClick={() => select({ kind: 'layer', nodeId: node.id })}>
        Show the whole layer
      </button>
    </section>
  )
}

function ConnectionDetails({
  source,
  target,
  tensor,
  selection,
}: {
  source: GraphNode
  target: GraphNode
  tensor: string | null
  selection: Extract<Selection, { kind: 'connection' }>
}) {
  const sourceUnits = unitCount(source)
  const targetUnits = unitCount(target)
  const { sourceIndex, targetIndex } = selection
  const single = sourceIndex !== null && targetIndex !== null
  const weights = useModelTensor(`${target.id}.weight`)
  const gradients = useModelTensor(`${target.id}.weight.grad`)
  // A Linear weight is [target units][source units].
  const at = (t: TensorResponse) =>
    single && sourceUnits !== null ? t.values[targetIndex * sourceUnits + sourceIndex] : undefined

  return (
    <section aria-label="Connection">
      <p className="inspector-kind">{single ? 'Connection' : 'Layer connection'}</p>
      <h3>
        {source.name}
        {sourceIndex !== null && ` · unit ${sourceIndex + 1}`} → {target.name}
        {targetIndex !== null && ` · unit ${targetIndex + 1}`}
      </h3>
      <dl className="facts">
        {tensor && <Fact label="Tensor">{tensor}</Fact>}
        {!single && sourceUnits !== null && targetUnits !== null && (
          <Fact label="Connections">
            {(sourceUnits * targetUnits).toLocaleString()}{' '}
            <span className="muted">
              = {sourceUnits} × {targetUnits}
            </span>
          </Fact>
        )}
        <Fact label={single ? 'Weight' : 'Weights'}>
          <TensorView
            state={weights}
            render={(t) => (single ? <span className="mono">{valueText(at(t))}</span> : summaryText(t))}
          />
        </Fact>
        <Fact label="Gradient">
          <TensorView
            state={gradients}
            missing={TRAIN_HINT}
            render={(t) => (single ? <span className="mono">{valueText(at(t))}</span> : summaryText(t))}
          />
        </Fact>
      </dl>
    </section>
  )
}

/** Loading, missing (404), error, and ready states for one tensor value. */
function TensorView({
  state,
  render,
  missing,
}: {
  state: TensorState
  render: (tensor: TensorResponse) => ReactNode
  missing?: string
}) {
  if (state.state === 'idle') return null
  if (state.state === 'loading') return <span className="muted">Loading…</span>
  if (state.state === 'error') {
    if (missing && state.error instanceof ApiError && state.error.status === 404) return <>{missing}</>
    return <span className="field-error">{state.error instanceof Error ? state.error.message : 'Could not load values.'}</span>
  }
  return <>{render(state.data)}</>
}

function summaryText(t: TensorResponse) {
  const s = t.summary
  if (!s) return <span className="muted">No finite values.</span>
  return (
    <span className="mono">
      min {fmt(s.min)}, max {fmt(s.max)}, mean {fmt(s.mean)}
      {t.non_finite_count > 0 && ` (${t.non_finite_count} NaN/∞ excluded)`}
    </span>
  )
}

/** A single tensor value; null marks NaN or infinity, which JSON cannot carry. */
const valueText = (v: number | null | undefined) => (v === null || v === undefined ? 'NaN/∞' : fmt(v))

function Fact({ label, children }: { label: string; children: ReactNode }) {
  return (
    <>
      <dt>{label}</dt>
      <dd>{children}</dd>
    </>
  )
}

function numberAttr(node: GraphNode, key: string): number | null {
  const value = node.attributes[key]
  return typeof value === 'number' ? value : null
}

function activationLabel(node: GraphNode): string {
  const value = node.attributes.activation
  if (value === null || value === undefined) return 'None (raw outputs)'
  return typeof value === 'string' && value in ACTIVATION_LABELS ? ACTIVATION_LABELS[value as Activation] : String(value)
}
