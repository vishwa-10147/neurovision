import { Maximize, Minus, Plus, RotateCcw } from 'lucide-react'
import { useMemo, useRef, type KeyboardEvent, type MouseEvent } from 'react'
import { showsUnits } from '../../lib/graph'
import { activationFill } from '../../lib/values'
import { useModelView } from '../../app/modelView'
import type { Selection } from '../../stores/modelStore'
import type { GraphEdge, ModelGraph } from '../../types/graph'
import { computeLayout, UNIT_RADIUS, type GraphLayout, type LayerLayout } from './layout'
import { usePanZoom } from './usePanZoom'

/** Above this many unit-to-unit connections, edges are drawn as aggregated bands. */
export const MAX_INDIVIDUAL_CONNECTIONS = 1500

interface Network2DProps {
  graph: ModelGraph
  /** Shown over the diagram, e.g. while a newer graph is loading. */
  overlay?: string | null
  /** Measured post-activation values per layer from a forward pass, if any. */
  activations?: Record<string, number[]> | null
  /** Imported models: one measured magnitude per operator from the last run. */
  blockValues?: Record<string, number> | null
  /** What blockValues measure: a short name for tooltips and the legend text. */
  blockLabel?: BlockLabel
}

export interface BlockLabel {
  name: string
  legend: string
  /** Where the values come from, shown under the legend. */
  source?: string
}

const RUN_LABEL: BlockLabel = {
  name: 'output RMS',
  legend: 'Operator color: size (RMS) of its output in the last run, scaled across operators',
}

export function Network2D({ graph, overlay, activations, blockValues, blockLabel = RUN_LABEL }: Network2DProps) {
  const blockMax = blockValues ? Math.max(0, ...Object.values(blockValues)) : 0
  const layout = useMemo(() => computeLayout(graph), [graph])
  const { selection, select } = useModelView()
  const svgRef = useRef<SVGSVGElement>(null)
  const { transform, zoomIn, zoomOut, fit, reset, wasDragged, svgHandlers } = usePanZoom(
    svgRef,
    layout.bounds,
  )
  const aggregated = layout.connectionCount > MAX_INDIVIDUAL_CONNECTIONS

  const chooseIf = (next: Selection) => (event: MouseEvent) => {
    event.stopPropagation()
    if (!wasDragged()) select(next)
  }

  return (
    <div className="network2d">
      <div className="toolbar" role="toolbar" aria-label="Diagram view controls">
        <button type="button" className="icon-button" onClick={zoomIn} aria-label="Zoom in" title="Zoom in (+)">
          <Plus size={16} aria-hidden="true" />
        </button>
        <button type="button" className="icon-button" onClick={zoomOut} aria-label="Zoom out" title="Zoom out (−)">
          <Minus size={16} aria-hidden="true" />
        </button>
        <button type="button" className="icon-button" onClick={fit} aria-label="Fit to view" title="Fit to view (F)">
          <Maximize size={16} aria-hidden="true" />
        </button>
        <button type="button" className="icon-button" onClick={reset} aria-label="Reset view" title="Reset view to 100% (0)">
          <RotateCcw size={16} aria-hidden="true" />
        </button>
        <span className="toolbar-zoom" aria-live="polite">
          {Math.round(transform.k * 100)}%
        </span>
      </div>

      <svg
        ref={svgRef}
        className="network2d-canvas"
        tabIndex={0}
        aria-label="2D network diagram. Drag or use arrow keys to pan, scroll or plus and minus to zoom, F to fit. Tab to a layer and press Enter to select it."
        onClick={() => {
          if (!wasDragged()) select(null)
        }}
        {...svgHandlers}
      >
        <g transform={`translate(${transform.x} ${transform.y}) scale(${transform.k})`}>
          {graph.edges.map((edge) =>
            aggregated ? (
              <EdgeBand
                key={edge.id}
                edge={edge}
                layout={layout}
                selection={selection}
                onClick={chooseIf({ kind: 'connection', edgeId: edge.id, sourceIndex: null, targetIndex: null })}
              />
            ) : (
              <EdgeLines key={edge.id} edge={edge} layout={layout} selection={selection} chooseIf={chooseIf} />
            ),
          )}
          {layout.layers.map((layer) => (
            <Layer
              key={layer.nodeId}
              layer={layer}
              selection={selection}
              chooseIf={chooseIf}
              onSelectLayer={select}
              values={activations?.[layer.nodeId] ?? null}
              blockValue={blockValues?.[layer.nodeId] ?? null}
              blockMax={blockMax}
              blockName={blockLabel.name}
            />
          ))}
        </g>
      </svg>

      {overlay && <div className="network2d-overlay">{overlay}</div>}

      {showsUnits(graph) ? (
        <Legend aggregated={aggregated} connectionCount={layout.connectionCount} showsActivations={Boolean(activations)} />
      ) : (
        <OperatorLegend
          graph={graph}
          measured={blockValues && Object.keys(blockValues).length ? blockLabel.legend : null}
          source={blockLabel.source}
        />
      )}
    </div>
  )
}

type ChooseIf = (next: Selection) => (event: MouseEvent) => void

function Layer({
  layer,
  selection,
  chooseIf,
  onSelectLayer,
  values,
  blockValue = null,
  blockMax = 0,
  blockName = RUN_LABEL.name,
}: {
  layer: LayerLayout
  selection: Selection
  chooseIf: ChooseIf
  onSelectLayer: (s: Selection) => void
  values: number[] | null
  blockValue?: number | null
  blockMax?: number
  blockName?: string
}) {
  const shaded = !layer.units && blockValue !== null
  const maxAbs = values ? Math.max(0, ...values.map(Math.abs)) : 0
  const layerSelected = selection?.kind === 'layer' && selection.nodeId === layer.nodeId
  const selectedUnit = selection?.kind === 'unit' && selection.nodeId === layer.nodeId ? selection.index : null
  const unitsLabel = layer.units
    ? `${layer.units.length} units`
    : layer.label !== layer.name
      ? `${layer.label} operator${layer.unsupported ? ', unsupported' : ''}`
      : 'unit count unknown'

  const onKeyDown = (event: KeyboardEvent) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      event.stopPropagation()
      onSelectLayer({ kind: 'layer', nodeId: layer.nodeId })
    }
  }

  return (
    <g
      className={`layer layer-${layer.role}${layerSelected ? ' is-selected' : ''}${layer.units ? '' : ' is-block'}${layer.unsupported ? ' is-unsupported' : ''}`}
      data-node-id={layer.nodeId}
      role="button"
      tabIndex={0}
      aria-label={`${layer.name}, ${unitsLabel}`}
      aria-pressed={layerSelected}
      onKeyDown={onKeyDown}
    >
      <rect
        className="layer-box"
        x={layer.x - UNIT_RADIUS - 12}
        y={layer.top}
        width={(UNIT_RADIUS + 12) * 2}
        height={layer.bottom - layer.top}
        rx={14}
        data-value={shaded ? blockValue : undefined}
        style={shaded ? { fill: activationFill(blockValue, blockMax) } : undefined}
        onClick={chooseIf({ kind: 'layer', nodeId: layer.nodeId })}
      >
        {shaded && <title>{`${layer.label}: ${blockName} ${blockValue.toPrecision(3)}`}</title>}
      </rect>
      <text className="layer-label" x={layer.x} y={layer.top - 10} textAnchor="middle">
        {layer.label}
      </text>
      {layer.units?.map((unit) => {
        const value = values?.[unit.index]
        const measured = value !== undefined
        return (
          <circle
            key={unit.index}
            className={`unit${selectedUnit === unit.index ? ' is-selected' : ''}${measured ? ' has-value' : ''}`}
            data-unit={`${layer.nodeId}#${unit.index}`}
            data-value={measured ? value : undefined}
            style={measured ? { fill: activationFill(value, maxAbs) } : undefined}
            cx={layer.x}
            cy={unit.y}
            r={UNIT_RADIUS}
            onClick={chooseIf({ kind: 'unit', nodeId: layer.nodeId, index: unit.index })}
          >
            <title>
              {`${layer.name}, unit ${unit.index + 1}${measured ? `: ${value.toFixed(3)}` : ''}`}
            </title>
          </circle>
        )
      })}
    </g>
  )
}

function EdgeLines({
  edge,
  layout,
  selection,
  chooseIf,
}: {
  edge: GraphEdge
  layout: GraphLayout
  selection: Selection
  chooseIf: ChooseIf
}) {
  const source = layout.byId.get(edge.source)
  const target = layout.byId.get(edge.target)
  if (!source?.units || !target?.units) {
    const whole: Selection = { kind: 'connection', edgeId: edge.id, sourceIndex: null, targetIndex: null }
    return <EdgeLink edge={edge} layout={layout} selection={selection} onClick={chooseIf(whole)} />
  }

  const focus = connectionFocus(edge, selection)
  const lines = []
  for (const s of source.units) {
    for (const t of target.units) {
      const state = focus(s.index, t.index)
      lines.push(
        <line
          key={`${s.index}-${t.index}`}
          className={`connection${state ? ` is-${state}` : ''}`}
          data-connection={`${edge.id}:${s.index}:${t.index}`}
          x1={source.x}
          y1={s.y}
          x2={target.x}
          y2={t.y}
          onClick={chooseIf({ kind: 'connection', edgeId: edge.id, sourceIndex: s.index, targetIndex: t.index })}
        />,
      )
    }
  }
  return <g className={focus.any ? 'edge has-focus' : 'edge'}>{lines}</g>
}

/** Which connections of `edge` the current selection highlights. */
function connectionFocus(edge: GraphEdge, selection: Selection) {
  let test: (s: number, t: number) => 'selected' | 'related' | null = () => null
  let any = false
  if (selection?.kind === 'unit') {
    if (selection.nodeId === edge.source) test = (s) => (s === selection.index ? 'related' : null)
    if (selection.nodeId === edge.target) test = (_, t) => (t === selection.index ? 'related' : null)
    any = selection.nodeId === edge.source || selection.nodeId === edge.target
  } else if (selection?.kind === 'layer' && selection.nodeId === edge.target) {
    test = () => 'related'
    any = true
  } else if (selection?.kind === 'connection' && selection.edgeId === edge.id) {
    const { sourceIndex, targetIndex } = selection
    test = (s, t) =>
      (sourceIndex === null || s === sourceIndex) && (targetIndex === null || t === targetIndex)
        ? 'selected'
        : null
    any = true
  }
  return Object.assign(test, { any })
}

/**
 * A curved link between two blocks. Wrapped operator graphs run right to
 * left on every other row, and turn from one row to the next in place.
 */
function linkPath(source: LayerLayout, target: LayerLayout): string {
  const half = UNIT_RADIUS + 12
  const sy = (source.top + source.bottom) / 2
  const ty = (target.top + target.bottom) / 2
  if (Math.abs(target.x - source.x) < half && target.top > source.bottom) {
    // A row turn: straight down from one block to the next.
    const y1 = source.bottom
    const y2 = target.top - 24 // Leave the target's label visible.
    const bend = (y2 - y1) / 2
    return `M ${source.x} ${y1} C ${source.x} ${y1 + bend}, ${target.x} ${y2 - bend}, ${target.x} ${y2}`
  }
  const direction = target.x >= source.x ? 1 : -1
  const x1 = source.x + direction * half
  const x2 = target.x - direction * half
  const bend = direction * Math.max(40, Math.abs(x2 - x1) / 2)
  return `M ${x1} ${sy} C ${x1 + bend} ${sy}, ${x2 - bend} ${ty}, ${x2} ${ty}`
}

/** One curved link between two blocks (operators or layers without units). */
function EdgeLink({
  edge,
  layout,
  selection,
  onClick,
}: {
  edge: GraphEdge
  layout: GraphLayout
  selection: Selection
  onClick: (event: MouseEvent) => void
}) {
  const source = layout.byId.get(edge.source)
  const target = layout.byId.get(edge.target)
  if (!source || !target) return null
  const d = linkPath(source, target)
  const selected = selection?.kind === 'connection' && selection.edgeId === edge.id
  const related =
    (selection?.kind === 'layer' || selection?.kind === 'unit') &&
    (selection.nodeId === edge.target || selection.nodeId === edge.source)
  return (
    <g className={`edge-link${selected ? ' is-selected' : related ? ' is-related' : ''}`} data-edge-id={edge.id} onClick={onClick}>
      {/* A wide invisible stroke makes thin links easy to click. */}
      <path className="edge-link-hit" d={d} />
      <path className="edge-link-line" d={d} />
      <title>{edge.tensor_name ?? edge.id}</title>
    </g>
  )
}

function EdgeBand({
  edge,
  layout,
  selection,
  onClick,
}: {
  edge: GraphEdge
  layout: GraphLayout
  selection: Selection
  onClick: (event: MouseEvent) => void
}) {
  const source = layout.byId.get(edge.source)
  const target = layout.byId.get(edge.target)
  if (!source || !target) return null
  const inset = 10
  const points = [
    [source.x, source.top + inset],
    [target.x, target.top + inset],
    [target.x, target.bottom - inset],
    [source.x, source.bottom - inset],
  ]
    .map(([x, y]) => `${x},${y}`)
    .join(' ')
  const selected = selection?.kind === 'connection' && selection.edgeId === edge.id
  const related =
    (selection?.kind === 'layer' || selection?.kind === 'unit') &&
    (selection.nodeId === edge.target || selection.nodeId === edge.source)
  return (
    <polygon
      className={`edge-band${selected ? ' is-selected' : related ? ' is-related' : ''}`}
      data-edge-id={edge.id}
      points={points}
      onClick={onClick}
    />
  )
}

function OperatorLegend({
  graph,
  measured,
  source = 'Values come from running the model in ONNX Runtime; select an operator to see them.',
}: {
  graph: ModelGraph
  measured: string | null
  source?: string
}) {
  const unsupported = graph.nodes.filter((n) => n.unsupported).length
  const runnable = graph.model.capabilities.inference
  return (
    <div className="legend" aria-label="Diagram legend">
      {measured && (
        <span className="legend-item">
          <span className="activation-scale activation-scale-positive" aria-hidden="true" />
          {measured}
        </span>
      )}
      <span className="legend-item">
        <span className="swatch-block legend-input-bg" aria-hidden="true" />
        Graph input
      </span>
      <span className="legend-item">
        <span className="swatch-block legend-hidden-bg" aria-hidden="true" />
        Operator
      </span>
      <span className="legend-item">
        <span className="swatch-block legend-output-bg" aria-hidden="true" />
        Graph output
      </span>
      {unsupported > 0 && (
        <span className="legend-item">
          <span className="swatch-block swatch-unsupported" aria-hidden="true" />
          Unsupported operator ({unsupported})
        </span>
      )}
      <p className="legend-note">
        Each block is one operator, labeled with its type; each link is a tensor passed between
        operators.{' '}
        {measured
          ? source
          : runnable
            ? 'Run the model to color operators by the values they actually produce.'
            : 'This model cannot be run here (see the import report), so values are not computed.'}
      </p>
    </div>
  )
}

function Legend({
  aggregated,
  connectionCount,
  showsActivations,
}: {
  aggregated: boolean
  connectionCount: number
  showsActivations: boolean
}) {
  return (
    <div className="legend" aria-label="Diagram legend">
      {showsActivations ? (
        <span className="legend-item">
          <span className="activation-scale" aria-hidden="true" />
          Unit color: value from the last forward pass, scaled within each layer (orange negative, blue
          positive)
        </span>
      ) : (
        <>
          <span className="legend-item">
            <svg width="12" height="12" aria-hidden="true"><circle cx="6" cy="6" r="5" className="unit legend-input" /></svg>
            Input unit
          </span>
          <span className="legend-item">
            <svg width="12" height="12" aria-hidden="true"><circle cx="6" cy="6" r="5" className="unit legend-hidden" /></svg>
            Hidden unit
          </span>
          <span className="legend-item">
            <svg width="12" height="12" aria-hidden="true"><circle cx="6" cy="6" r="5" className="unit legend-output" /></svg>
            Output unit
          </span>
        </>
      )}
      <span className="legend-item">
        <svg width="18" height="12" aria-hidden="true"><line x1="0" y1="6" x2="18" y2="6" className="connection is-related" /></svg>
        Connections of the selection
      </span>
      <p className="legend-note">
        {aggregated
          ? `${connectionCount.toLocaleString()} connections are too many to draw one by one, so each layer pair is shown as a single band (limit ${MAX_INDIVIDUAL_CONNECTIONS.toLocaleString()}).`
          : `Lines show which units connect (${connectionCount.toLocaleString()} total).`}{' '}
        Line color and thickness do not encode weights; select a connection to see its weight in the
        inspector.
      </p>
    </div>
  )
}
