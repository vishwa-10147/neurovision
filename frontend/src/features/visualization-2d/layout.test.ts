import { describe, expect, it } from 'vitest'
import onnxGraph from '../../../../shared/schemas/examples/onnx-import.graph.json'
import { denseGraph } from '../../test/denseGraph'
import { EXAMPLE_CONFIG } from '../../types/architecture'
import { ModelGraphSchema } from '../../types/graph'
import { columnsOf, computeLayout, LAYER_GAP, UNIT_GAP, WRAP_COLUMNS, wrapColumns } from './layout'
import { fitTransform, MAX_ZOOM, MIN_ZOOM, zoomAround } from './viewport'

describe('computeLayout', () => {
  const graph = denseGraph(EXAMPLE_CONFIG) // 2 -> 8 -> 6 -> 2

  it('derives columns and unit counts from the semantic graph', () => {
    const layout = computeLayout(graph)
    expect(layout.layers.map((l) => [l.nodeId, l.role, l.x, l.units?.length])).toEqual([
      ['input', 'input', 0, 2],
      ['hidden_1', 'hidden', LAYER_GAP, 8],
      ['hidden_2', 'hidden', LAYER_GAP * 2, 6],
      ['output', 'output', LAYER_GAP * 3, 2],
    ])
    expect(layout.connectionCount).toBe(2 * 8 + 8 * 6 + 6 * 2)
  })

  it('centers units vertically with a fixed gap', () => {
    const ys = computeLayout(graph).byId.get('hidden_2')?.units?.map((u) => u.y) ?? []
    expect(ys[0]).toBeCloseTo(-ys[ys.length - 1])
    expect(ys[1] - ys[0]).toBe(UNIT_GAP)
  })

  it('does not write layout data into the graph', () => {
    const before = JSON.stringify(graph)
    computeLayout(graph)
    expect(JSON.stringify(graph)).toBe(before)
  })

  it('draws imported operators as blocks labeled by type, even with numeric shapes', () => {
    // Same nodes, but marked as an imported operator graph: [batch, 8] is a
    // tensor shape there, not a count of neurons.
    const imported = { ...graph, model: { ...graph.model, source_format: 'onnx' as const } }
    const layout = computeLayout(imported)
    expect(layout.layers.every((l) => l.units === null)).toBe(true)
    expect(layout.layers.map((l) => l.label)).toEqual(['Input', 'Linear', 'Linear', 'Linear'])
    expect(layout.connectionCount).toBe(0)
  })

  it('draws layers with unknown sizes as blocks instead of inventing units', () => {
    const layout = computeLayout(ModelGraphSchema.parse(onnxGraph))
    expect(layout.byId.get('conv_0')?.units).toBeNull()
    expect(layout.byId.get('custom_1')?.units).toBeNull()
    expect(layout.connectionCount).toBe(0)
  })
})

describe('viewport math', () => {
  const bounds = { minX: 0, minY: -100, maxX: 400, maxY: 100 }

  it('fits and centers content in the viewport', () => {
    const t = fitTransform(bounds, 880, 480)
    // Center of the bounds maps to the center of the viewport.
    expect(200 * t.k + t.x).toBeCloseTo(440)
    expect(t.y).toBeCloseTo(240) // content y = 0
    expect(400 * t.k).toBeLessThanOrEqual(880 - 80)
  })

  it('zooms around a fixed screen point and clamps the scale', () => {
    const t = { x: 10, y: 20, k: 1 }
    const z = zoomAround(t, 2, 100, 100)
    // The content point under (100, 100) stays under it.
    expect((100 - z.x) / z.k).toBeCloseTo((100 - t.x) / t.k)
    expect((100 - z.y) / z.k).toBeCloseTo((100 - t.y) / t.k)
    expect(zoomAround(t, 1000, 0, 0).k).toBe(MAX_ZOOM)
    expect(zoomAround(t, 0.0001, 0, 0).k).toBe(MIN_ZOOM)
  })
})

describe('wrapping deep operator graphs', () => {
  /** A chain of `n` Relu operators, like a deep imported network. */
  function chain(n: number) {
    const base = ModelGraphSchema.parse(onnxGraph)
    const template = base.nodes.find((node) => node.op_type !== 'Input') ?? base.nodes[0]
    const nodes = Array.from({ length: n }, (_, i) => ({
      ...template,
      id: `op_${i}`,
      name: `/op_${i}`,
      op_type: 'Relu',
      unsupported: false,
    }))
    const edges = nodes.slice(1).map((node, i) => ({
      id: `e_${i}`,
      source: `op_${i}`,
      target: node.id,
      tensor_name: `t_${i}`,
    }))
    return { ...base, nodes, edges }
  }

  it('keeps short chains on one row', () => {
    const place = wrapColumns(WRAP_COLUMNS, 1)
    expect(place(WRAP_COLUMNS - 1)).toEqual({ slot: WRAP_COLUMNS - 1, row: 0 })
  })

  it('snakes long chains so neighbours stay adjacent', () => {
    const place = wrapColumns(120, 0.5)
    const placed = Array.from({ length: 120 }, (_, column) => place(column))
    const perRow = placed.filter((p) => p.row === 0).length
    expect(perRow).toBeGreaterThan(1)
    for (let i = 1; i < placed.length; i++) {
      const [a, b] = [placed[i - 1], placed[i]]
      // Either the next slot along the row, or straight down at a row turn.
      if (a.row === b.row) expect(Math.abs(a.slot - b.slot)).toBe(1)
      else expect([b.row - a.row, b.slot - a.slot]).toEqual([1, 0])
    }
  })

  it('puts constants just before the operator that uses them, not in column 0', () => {
    const graph = chain(6)
    graph.nodes[0] = { ...graph.nodes[0], op_type: 'Input' }
    const constant = { ...graph.nodes[1], id: 'const', name: '/const', op_type: 'Constant' }
    const graphWithConst = {
      ...graph,
      nodes: [...graph.nodes, constant],
      edges: [...graph.edges, { id: 'e_const', source: 'const', target: 'op_5', tensor_name: 'k' }],
    }
    const columns = columnsOf(graphWithConst)
    expect(columns.get('op_5')).toBe(5)
    expect(columns.get('const')).toBe(4)
  })

  it('lays a 100-operator chain out roughly 16:9 instead of one long row', () => {
    const { bounds } = computeLayout(chain(100))
    const ratio = (bounds.maxX - bounds.minX) / (bounds.maxY - bounds.minY)
    expect(ratio).toBeGreaterThan(0.8)
    expect(ratio).toBeLessThan(4)
  })
})
