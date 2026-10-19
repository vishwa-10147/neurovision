/**
 * 2D layout derived from the normalized graph. Layout lives here, never in the
 * graph: the same graph can be laid out differently by each renderer.
 */
import { showsUnits, unitCount } from '../../lib/graph'
import type { ModelGraph } from '../../types/graph'

export const LAYER_GAP = 190
/** Operator graphs are long chains of narrow blocks; pack them tighter. */
export const OPERATOR_GAP = 110
export const UNIT_GAP = 34
export const UNIT_RADIUS = 11
/** Vertical size of a layer whose unit count is unknown. */
const OPAQUE_LAYER_HEIGHT = 80
const COLUMN_STACK_GAP = 60
const LAYER_PADDING = 18

export type LayerRole = 'input' | 'hidden' | 'output'

export interface UnitPosition {
  index: number
  y: number
}

export interface LayerLayout {
  nodeId: string
  name: string
  /** Short text drawn on the diagram: the layer name, or the operator type. */
  label: string
  role: LayerRole
  /** Imported operator outside the ONNX standard, shown for inspection only. */
  unsupported: boolean
  x: number
  top: number
  bottom: number
  /** null when the unit count is unknown; the layer is drawn as one block. */
  units: UnitPosition[] | null
}

export interface Bounds {
  minX: number
  minY: number
  maxX: number
  maxY: number
}

export interface GraphLayout {
  layers: LayerLayout[]
  byId: Map<string, LayerLayout>
  bounds: Bounds
  /** Total unit-to-unit connections across all edges with known sizes. */
  connectionCount: number
}

/** Longest-path depth per node; nodes in cycles fall back to depth 0. */
export function columnsOf(graph: ModelGraph): Map<string, number> {
  const incoming = new Map<string, number>(graph.nodes.map((n) => [n.id, 0]))
  const outgoing = new Map<string, string[]>(graph.nodes.map((n) => [n.id, []]))
  for (const edge of graph.edges) {
    incoming.set(edge.target, (incoming.get(edge.target) ?? 0) + 1)
    outgoing.get(edge.source)?.push(edge.target)
  }
  const depth = new Map<string, number>(graph.nodes.map((n) => [n.id, 0]))
  const queue = graph.nodes.filter((n) => incoming.get(n.id) === 0).map((n) => n.id)
  while (queue.length > 0) {
    const id = queue.shift() as string
    for (const next of outgoing.get(id) ?? []) {
      depth.set(next, Math.max(depth.get(next) ?? 0, (depth.get(id) ?? 0) + 1))
      const remaining = (incoming.get(next) ?? 1) - 1
      incoming.set(next, remaining)
      if (remaining === 0) queue.push(next)
    }
  }
  return depth
}

export function computeLayout(graph: ModelGraph): GraphLayout {
  const units = showsUnits(graph)
  const columns = columnsOf(graph)
  const hasOutgoing = new Set(graph.edges.map((e) => e.source))

  const byColumn = new Map<number, string[]>()
  for (const node of graph.nodes) {
    const column = columns.get(node.id) ?? 0
    byColumn.set(column, [...(byColumn.get(column) ?? []), node.id])
  }

  const layers: LayerLayout[] = []
  const byId = new Map<string, LayerLayout>()
  const bounds: Bounds = { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity }

  for (const node of graph.nodes) {
    const column = columns.get(node.id) ?? 0
    const siblings = byColumn.get(column) ?? [node.id]
    const count = units ? unitCount(node) : null
    const height = count === null ? OPAQUE_LAYER_HEIGHT : (count - 1) * UNIT_GAP
    // Stack nodes that share a column; a dense chain has one node per column.
    const offset = (siblings.indexOf(node.id) - (siblings.length - 1) / 2) * (height + COLUMN_STACK_GAP)
    const x = column * (units ? LAYER_GAP : OPERATOR_GAP)
    const layer: LayerLayout = {
      nodeId: node.id,
      name: node.name,
      label: units ? node.name : node.op_type,
      unsupported: node.unsupported,
      role: node.op_type === 'Input' ? 'input' : hasOutgoing.has(node.id) ? 'hidden' : 'output',
      x,
      top: offset - height / 2 - LAYER_PADDING,
      bottom: offset + height / 2 + LAYER_PADDING,
      units:
        count === null
          ? null
          : Array.from({ length: count }, (_, index) => ({
              index,
              y: offset + (index - (count - 1) / 2) * UNIT_GAP,
            })),
    }
    layers.push(layer)
    byId.set(node.id, layer)
    bounds.minX = Math.min(bounds.minX, x - UNIT_RADIUS - LAYER_PADDING)
    bounds.maxX = Math.max(bounds.maxX, x + UNIT_RADIUS + LAYER_PADDING)
    // Leave room for the layer label above the block.
    bounds.minY = Math.min(bounds.minY, layer.top - 28)
    bounds.maxY = Math.max(bounds.maxY, layer.bottom)
  }

  let connectionCount = 0
  for (const edge of graph.edges) {
    const source = byId.get(edge.source)?.units?.length
    const target = byId.get(edge.target)?.units?.length
    if (source && target) connectionCount += source * target
  }

  if (layers.length === 0) bounds.minX = bounds.minY = bounds.maxX = bounds.maxY = 0
  return { layers, byId, bounds, connectionCount }
}
