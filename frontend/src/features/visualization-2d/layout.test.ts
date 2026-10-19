import { describe, expect, it } from 'vitest'
import onnxGraph from '../../../../shared/schemas/examples/onnx-import.graph.json'
import { denseGraph } from '../../test/denseGraph'
import { EXAMPLE_CONFIG } from '../../types/architecture'
import { ModelGraphSchema } from '../../types/graph'
import { computeLayout, LAYER_GAP, UNIT_GAP } from './layout'
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
