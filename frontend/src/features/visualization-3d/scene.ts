/**
 * Renderer-independent plan for the 3D view, derived from the normalized graph.
 * Pure functions only, so layout, detail limits, and highlighting are testable
 * without WebGL. The React Three Fiber components just draw this plan.
 */
import { showsUnits, unitCount } from '../../lib/graph'
import type { Selection } from '../../stores/modelStore'
import type { ModelGraph } from '../../types/graph'
import { columnsOf, wrapColumns, type LayerRole, type Placement } from '../visualization-2d/layout'

export type Vec3 = [number, number, number]

export const LAYER_SPACING = 5
export const UNIT_SPACING = 1
export const UNIT_RADIUS = 0.28
/** Height above a layer where its label is drawn. */
export const LABEL_CLEARANCE = 0.9
/** Above this many units, layers are drawn as solid blocks. */
export const MAX_3D_UNITS = 2048
/** Above this many unit-to-unit connections, one link per layer pair is drawn. */
export const MAX_3D_CONNECTIONS = 8000
/** Above this many nodes, the 3D view defers to the 2D view. */
export const MAX_3D_NODES = 1500
/** Above this many nodes, only the selected node is labeled. */
export const MAX_3D_LABELS = 60

export interface Unit3D {
  nodeId: string
  index: number
  role: LayerRole
  position: Vec3
}

export interface Layer3D {
  nodeId: string
  name: string
  /** Text drawn in the scene: the layer name, or the operator type. */
  label: string
  unsupported: boolean
  role: LayerRole
  center: Vec3
  /** Extent of the layer's unit grid (or block). */
  size: Vec3
  /** Instance ids of this layer's units in `Scene3D.units`; null when drawn as a block. */
  unitIds: number[] | null
}

export interface Connection3D {
  edgeId: string
  source: number
  target: number
  sourceIndex: number
  targetIndex: number
}

export interface LayerLink3D {
  edgeId: string
  source: string
  target: string
  from: Vec3
  to: Vec3
}

export interface Scene3D {
  layers: Layer3D[]
  layerById: Map<string, Layer3D>
  units: Unit3D[]
  /** Individual connections, or null when aggregated into `layerLinks`. */
  connections: Connection3D[] | null
  layerLinks: LayerLink3D[]
  detail: {
    unitCount: number
    connectionCount: number
    unitsAsBlocks: boolean
    aggregatedConnections: boolean
  }
  /** Axis-aligned box around all layers (including labels' clearance), plus its sphere. */
  bounds: { center: Vec3; radius: number; min: Vec3; max: Vec3 }
}

/** Arrange `count` units in a near-square grid in the y-z plane. */
function gridPositions(count: number, x: number): Vec3[] {
  const cols = Math.ceil(Math.sqrt(count))
  const rows = Math.ceil(count / cols)
  return Array.from({ length: count }, (_, i) => {
    const row = Math.floor(i / cols)
    const col = i % cols
    return [x, ((rows - 1) / 2 - row) * UNIT_SPACING, (col - (cols - 1) / 2) * UNIT_SPACING]
  })
}

export function planScene(graph: ModelGraph): Scene3D {
  const columns = columnsOf(graph)
  const hasOutgoing = new Set(graph.edges.map((e) => e.source))
  // Operator graphs (imports) draw one block per node, never neuron grids.
  const units = showsUnits(graph)
  const counts = new Map(graph.nodes.map((n) => [n.id, units ? unitCount(n) : null]))
  const totalUnits = [...counts.values()].reduce<number>((sum, c) => sum + (c ?? 0), 0)
  const unitsAsBlocks = totalUnits > MAX_3D_UNITS

  // Nodes sharing a column are stacked along z.
  const byColumn = new Map<number, string[]>()
  for (const node of graph.nodes) {
    const column = columns.get(node.id) ?? 0
    byColumn.set(column, [...(byColumn.get(column) ?? []), node.id])
  }

  // Long operator chains wrap into rows laid out in depth (see wrapColumns).
  const tallest = Math.max(1, ...[...byColumn.values()].map((ids) => ids.length))
  const rowDepth = (tallest + 2) * LAYER_SPACING
  const place = units
    ? (column: number): Placement => ({ slot: column, row: 0 })
    : wrapColumns(byColumn.size, LAYER_SPACING / rowDepth)

  const layers: Layer3D[] = []
  const layerById = new Map<string, Layer3D>()
  const unitList: Unit3D[] = []

  for (const node of graph.nodes) {
    const column = columns.get(node.id) ?? 0
    const siblings = byColumn.get(column) ?? [node.id]
    const { slot, row } = place(column)
    const x = slot * LAYER_SPACING
    const zOffset = row * rowDepth + (siblings.indexOf(node.id) - (siblings.length - 1) / 2) * LAYER_SPACING
    const role: LayerRole =
      node.op_type === 'Input' ? 'input' : hasOutgoing.has(node.id) ? 'hidden' : 'output'
    const count = counts.get(node.id) ?? null

    let unitIds: number[] | null = null
    let size: Vec3 = [UNIT_SPACING, 2 * UNIT_SPACING, 2 * UNIT_SPACING]
    if (count !== null && !unitsAsBlocks) {
      const positions = gridPositions(count, x).map(([px, py, pz]): Vec3 => [px, py, pz + zOffset])
      unitIds = positions.map((position, index) => {
        unitList.push({ nodeId: node.id, index, role, position })
        return unitList.length - 1
      })
      const cols = Math.ceil(Math.sqrt(count))
      const rows = Math.ceil(count / cols)
      size = [UNIT_SPACING, rows * UNIT_SPACING, cols * UNIT_SPACING]
    } else if (count !== null) {
      // Block size grows with the unit count so relative widths stay visible.
      const side = Math.max(1, Math.sqrt(count)) * UNIT_SPACING * 0.5
      size = [UNIT_SPACING, side, side]
    }

    const layer: Layer3D = {
      nodeId: node.id,
      name: node.name,
      label: units ? node.name : node.op_type,
      unsupported: node.unsupported,
      role,
      center: [x, 0, zOffset],
      size,
      unitIds,
    }
    layers.push(layer)
    layerById.set(node.id, layer)
  }

  let connectionCount = 0
  for (const edge of graph.edges) {
    const s = counts.get(edge.source)
    const t = counts.get(edge.target)
    if (s && t) connectionCount += s * t
  }
  const aggregatedConnections = unitsAsBlocks || connectionCount > MAX_3D_CONNECTIONS

  const layerLinks: LayerLink3D[] = []
  let connections: Connection3D[] | null = aggregatedConnections ? null : []
  for (const edge of graph.edges) {
    const source = layerById.get(edge.source)
    const target = layerById.get(edge.target)
    if (!source || !target) continue
    layerLinks.push({
      edgeId: edge.id,
      source: edge.source,
      target: edge.target,
      from: source.center,
      to: target.center,
    })
    if (!connections) continue
    if (!source.unitIds || !target.unitIds) {
      // Unknown sizes: fall back to layer links for the whole view.
      connections = null
      continue
    }
    source.unitIds.forEach((sourceId, sourceIndex) => {
      target.unitIds?.forEach((targetId, targetIndex) => {
        connections?.push({ edgeId: edge.id, source: sourceId, target: targetId, sourceIndex, targetIndex })
      })
    })
  }

  return {
    layers,
    layerById,
    units: unitList,
    connections,
    layerLinks,
    detail: {
      unitCount: totalUnits,
      connectionCount,
      unitsAsBlocks,
      aggregatedConnections: connections === null,
    },
    bounds: boundsOf(layers),
  }
}

function boundsOf(layers: Layer3D[]): Scene3D['bounds'] {
  if (layers.length === 0) return { center: [0, 0, 0], radius: 1, min: [-1, -1, -1], max: [1, 1, 1] }
  const min: Vec3 = [Infinity, Infinity, Infinity]
  const max: Vec3 = [-Infinity, -Infinity, -Infinity]
  for (const layer of layers) {
    for (let axis = 0; axis < 3; axis++) {
      min[axis] = Math.min(min[axis], layer.center[axis] - layer.size[axis] / 2)
      max[axis] = Math.max(max[axis], layer.center[axis] + layer.size[axis] / 2)
    }
    // Room for the layer label drawn above the layer.
    max[1] = Math.max(max[1], layer.center[1] + layer.size[1] / 2 + LABEL_CLEARANCE)
  }
  const center: Vec3 = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2]
  const radius = Math.max(1, Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) / 2)
  return { center, radius, min, max }
}

export type Emphasis = 'selected' | 'related'

export interface Highlight3D {
  units: Map<number, Emphasis>
  /** Indexes into `scene.connections`. */
  connections: Map<number, Emphasis>
  /** Edge ids whose layer link is emphasized (aggregated mode). */
  links: Map<string, Emphasis>
  /** Layers drawn with a selection frame. */
  layers: Set<string>
  /** True when something is selected, so everything else is dimmed. */
  active: boolean
}

/** Which instances the selection emphasizes; mirrors the 2D view's rules. */
export function highlightFor(scene: Scene3D, selection: Selection): Highlight3D {
  const h: Highlight3D = { units: new Map(), connections: new Map(), links: new Map(), layers: new Set(), active: false }
  if (!selection) return h
  h.active = true
  const unitId = (nodeId: string, index: number | null) =>
    index === null ? undefined : scene.layerById.get(nodeId)?.unitIds?.[index]

  if (selection.kind === 'layer') {
    h.layers.add(selection.nodeId)
    for (const id of scene.layerById.get(selection.nodeId)?.unitIds ?? []) h.units.set(id, 'selected')
    scene.connections?.forEach((c, i) => {
      if (scene.units[c.target].nodeId === selection.nodeId) h.connections.set(i, 'related')
    })
    for (const link of scene.layerLinks) {
      if (link.target === selection.nodeId) h.links.set(link.edgeId, 'related')
    }
    return h
  }

  if (selection.kind === 'unit') {
    const id = unitId(selection.nodeId, selection.index)
    if (id === undefined) {
      h.layers.add(selection.nodeId)
      return h
    }
    h.units.set(id, 'selected')
    scene.connections?.forEach((c, i) => {
      if (c.source === id || c.target === id) {
        h.connections.set(i, 'related')
        h.units.set(c.source === id ? c.target : c.source, 'related')
      }
    })
    for (const link of scene.layerLinks) {
      if (link.source === selection.nodeId || link.target === selection.nodeId) {
        h.links.set(link.edgeId, 'related')
      }
    }
    return h
  }

  // Connection: a single unit pair, or a whole layer-to-layer edge.
  h.links.set(selection.edgeId, 'selected')
  scene.connections?.forEach((c, i) => {
    if (c.edgeId !== selection.edgeId) return
    if (selection.sourceIndex !== null && c.sourceIndex !== selection.sourceIndex) return
    if (selection.targetIndex !== null && c.targetIndex !== selection.targetIndex) return
    h.connections.set(i, 'selected')
    h.units.set(c.source, 'related')
    h.units.set(c.target, 'related')
  })
  return h
}

/**
 * Camera position and orbit target that frame the whole scene in a viewport
 * with the given vertical field of view and aspect ratio (width / height).
 */
export function defaultCamera(
  scene: Scene3D,
  fovDegrees: number,
  aspect = 1,
): { position: Vec3; target: Vec3 } {
  const { center, radius } = scene.bounds
  // Before the canvas is measured its size may be 0; fall back to square.
  if (!Number.isFinite(aspect) || aspect <= 0) aspect = 1
  // Oblique view from the front-right and above, so each layer's unit grid
  // and the depth between layers both read clearly.
  const dir = normalize([0.6, 0.4, 0.7])
  const tanV = Math.tan((fovDegrees * Math.PI) / 360) * FIT_MARGIN
  const tanH = tanV * aspect
  const corners = boxCorners(scene.bounds)

  // Camera basis looking from `dir` toward the center (world up = +y).
  const forward = scale(dir, -1)
  const right = normalize(cross(forward, [0, 1, 0]))
  const up = cross(right, forward)
  const fits = (distance: number) =>
    corners.every((corner) => {
      const rel = sub(corner, add(center, scale(dir, distance)))
      const depth = dot(rel, forward)
      return (
        depth > 0 &&
        Math.abs(dot(rel, right)) <= depth * tanH &&
        Math.abs(dot(rel, up)) <= depth * tanV
      )
    })

  // Smallest distance at which every corner of the scene box is in view.
  let low = radius * 0.1
  let high = radius * 50
  for (let i = 0; i < 40; i++) {
    const mid = (low + high) / 2
    if (fits(mid)) high = mid
    else low = mid
  }
  return { target: center, position: add(center, scale(dir, high)) }
}

/** Keep a little space between the scene and the viewport edges. */
const FIT_MARGIN = 0.9

const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
const scale = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s]
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
]
const normalize = (a: Vec3): Vec3 => scale(a, 1 / Math.hypot(...a))

export function boxCorners({ min, max }: Scene3D['bounds']): Vec3[] {
  const corners: Vec3[] = []
  for (const x of [min[0], max[0]])
    for (const y of [min[1], max[1]]) for (const z of [min[2], max[2]]) corners.push([x, y, z])
  return corners
}
