import type { GraphNode, ModelGraph, Shape } from '../types/graph'

/** "[batch × 8]"; unknown dimensions show as "?" and unknown rank as "unknown". */
export function formatShape(shape: Shape): string {
  if (shape === null) return 'unknown'
  return `[${shape.map((d) => (d === null ? '?' : String(d))).join(' × ')}]`
}

/**
 * Number of units a layer produces, read from the last dimension of its first
 * output shape. Returns null when the shape is unknown or symbolic — the
 * renderer then shows the layer without individual units.
 */
export function unitCount(node: GraphNode): number | null {
  const shape = node.output_shapes[0]
  const last = shape?.[shape.length - 1]
  return typeof last === 'number' ? last : null
}

/**
 * Whether the graph's nodes are dense layers of individual units (neurons).
 * Imported graphs are operator graphs: a tensor's last dimension (e.g. image
 * width) is not a count of neurons, so they are drawn as operator blocks.
 */
export function showsUnits(graph: ModelGraph): boolean {
  return graph.model.source_format === 'educational'
}

export function findNode(graph: ModelGraph, id: string): GraphNode | undefined {
  return graph.nodes.find((node) => node.id === id)
}

export function totalParameters(graph: ModelGraph): number | null {
  let total = 0
  for (const node of graph.nodes) {
    if (node.op_type === 'Input') continue
    if (!node.parameter_summary) return null
    total += node.parameter_summary.parameter_count
  }
  return total
}
