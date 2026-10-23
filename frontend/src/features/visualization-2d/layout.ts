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
/** Space between wrapped rows of a long operator graph. */
const ROW_GAP = 90

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
  // Operators not fed by a graph input (constants, shape arithmetic on
  // weights) would all land in column 0, often far from where they are used;
  // put each one just before its first consumer instead.
  const inputs = graph.nodes.filter((n) => n.op_type === 'Input').map((n) => n.id)
  if (inputs.length > 0) {
    const fed = new Set(inputs)
    const stack = [...inputs]
    while (stack.length > 0) {
      for (const next of outgoing.get(stack.pop() as string) ?? []) {
        if (!fed.has(next)) {
          fed.add(next)
          stack.push(next)
        }
      }
    }
    const loose = graph.nodes.filter((n) => !fed.has(n.id) && (outgoing.get(n.id)?.length ?? 0) > 0)
    // Chains of loose operators settle from their far end, one step per pass.
    for (let pass = 0, changed = true; changed && pass < graph.nodes.length; pass++) {
      changed = false
      for (const node of loose) {
        const consumers = (outgoing.get(node.id) ?? []).map((id) => depth.get(id) ?? 0)
        const wanted = Math.max(0, Math.min(...consumers) - 1)
        if (wanted > (depth.get(node.id) ?? 0)) {
          depth.set(node.id, wanted)
          changed = true
        }
      }
    }
  }
  return depth
}

/** Operator chains longer than this many columns are wrapped into rows. */
export const WRAP_COLUMNS = 32

export interface Placement {
  /** Position along its row (0 = row start, left to right). */
  slot: number
  row: number
}

/**
 * Deep operator graphs (a ResNet-50 is ~120 columns, a ViT-B/16 ~1,000) as
 * one row are unreadable when fitted to a screen. Wrap the columns into a
 * "snake": left to right, then right to left on the next row, so that
 * consecutive operators always stay next to each other. `aspect` is the
 * width / height of one column slot relative to one row; the row length is
 * chosen so the whole graph comes out roughly 16:9.
 */
export function wrapColumns(columnCount: number, aspect: number): (column: number) => Placement {
  if (columnCount <= WRAP_COLUMNS) return (column) => ({ slot: column, row: 0 })
  const perRow = Math.max(WRAP_COLUMNS / 2, Math.ceil(Math.sqrt((columnCount * 16) / (9 * aspect))))
  return (column) => {
    const row = Math.floor(column / perRow)
    const along = column % perRow
    return { row, slot: row % 2 === 0 ? along : perRow - 1 - along }
  }
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

  // Only operator graphs (imports) wrap; small dense networks keep one row.
  const tallest = Math.max(1, ...[...byColumn.values()].map((ids) => ids.length))
  const rowHeight = tallest * (OPAQUE_LAYER_HEIGHT + COLUMN_STACK_GAP) + ROW_GAP
  const place = units
    ? (column: number): Placement => ({ slot: column, row: 0 })
    : wrapColumns(byColumn.size, OPERATOR_GAP / rowHeight)

  const layers: LayerLayout[] = []
  const byId = new Map<string, LayerLayout>()
  const bounds: Bounds = { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity }

  for (const node of graph.nodes) {
    const column = columns.get(node.id) ?? 0
    const siblings = byColumn.get(column) ?? [node.id]
    const count = units ? unitCount(node) : null
    const height = count === null ? OPAQUE_LAYER_HEIGHT : (count - 1) * UNIT_GAP
    // Stack nodes that share a column; a dense chain has one node per column.
    const { slot, row } = place(column)
    const offset =
      row * rowHeight + (siblings.indexOf(node.id) - (siblings.length - 1) / 2) * (height + COLUMN_STACK_GAP)
    const x = slot * (units ? LAYER_GAP : OPERATOR_GAP)
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
