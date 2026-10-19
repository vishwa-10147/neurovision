import { ChevronLeft, ChevronRight } from 'lucide-react'
import type { ReactNode } from 'react'
import { findNode, formatShape, unitCount } from '../../lib/graph'
import { useModelStore, type Selection } from '../../stores/modelStore'
import { ACTIVATION_LABELS, type Activation } from '../../types/architecture'
import type { GraphNode, ModelGraph } from '../../types/graph'
import { EmptyState } from '../states/StateViews'

const NOT_YET_MEASURED = 'Not available yet. Real values arrive with PyTorch models in Milestone 4.'

export function Inspector({ graph }: { graph: ModelGraph | null }) {
  const selection = useModelStore((s) => s.selection)
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

function LayerDetails({ node }: { node: GraphNode }) {
  const units = unitCount(node)
  const inFeatures = numberAttr(node, 'in_features')
  const outFeatures = numberAttr(node, 'out_features')
  return (
    <section aria-label={`Layer ${node.name}`}>
      <p className="inspector-kind">Layer</p>
      <h3>{node.name}</h3>
      <dl className="facts">
        <Fact label="Operator">{node.op_type}</Fact>
        <Fact label="Units">{units ?? 'Unknown'}</Fact>
        {node.op_type !== 'Input' && <Fact label="Activation">{activationLabel(node)}</Fact>}
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
          ) : node.op_type === 'Input' ? (
            'None (inputs are data, not learned)'
          ) : (
            'Unknown'
          )}
        </Fact>
      </dl>
    </section>
  )
}

function UnitDetails({ graph, node, index }: { graph: ModelGraph; node: GraphNode; index: number }) {
  const select = useModelStore((s) => s.select)
  const count = unitCount(node) ?? 0
  const inFeatures = numberAttr(node, 'in_features')
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
        {node.op_type !== 'Input' && <Fact label="Activation">{activationLabel(node)}</Fact>}
        {inFeatures !== null && <Fact label="Incoming connections">{inFeatures}</Fact>}
        <Fact label="Outgoing connections">{outgoing}</Fact>
        {inFeatures !== null && (
          <Fact label="Parameters">
            {inFeatures + 1} <span className="muted">= {inFeatures} weights + 1 bias</span>
          </Fact>
        )}
        <Fact label="Activation value">{NOT_YET_MEASURED}</Fact>
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
  const single = selection.sourceIndex !== null && selection.targetIndex !== null
  return (
    <section aria-label="Connection">
      <p className="inspector-kind">{single ? 'Connection' : 'Layer connection'}</p>
      <h3>
        {source.name}
        {selection.sourceIndex !== null && ` · unit ${selection.sourceIndex + 1}`} → {target.name}
        {selection.targetIndex !== null && ` · unit ${selection.targetIndex + 1}`}
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
        <Fact label={single ? 'Weight' : 'Weights'}>{NOT_YET_MEASURED}</Fact>
      </dl>
    </section>
  )
}

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
