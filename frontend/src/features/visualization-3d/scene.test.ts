import * as THREE from 'three'
import { describe, expect, it } from 'vitest'
import onnxGraph from '../../../../shared/schemas/examples/onnx-import.graph.json'
import { denseGraph } from '../../test/denseGraph'
import { EXAMPLE_CONFIG, type DenseNetworkConfig } from '../../types/architecture'
import { ModelGraphSchema } from '../../types/graph'
import {
  boxCorners,
  defaultCamera,
  highlightFor,
  LAYER_SPACING,
  MAX_3D_CONNECTIONS,
  MAX_3D_UNITS,
  planScene,
} from './scene'

const example = denseGraph(EXAMPLE_CONFIG) // 2 -> 8 -> 6 -> 2

/** Bypasses editor limits on purpose, to exercise renderer safeguards. */
const wide = (hidden: number[], input = 4): DenseNetworkConfig => ({
  ...EXAMPLE_CONFIG,
  input_units: input,
  hidden_layers: hidden.map((units) => ({ units, activation: 'relu' })),
})

describe('planScene', () => {
  const scene = planScene(example)

  it('places one unit instance per graph unit, one layer per column', () => {
    expect(scene.units).toHaveLength(2 + 8 + 6 + 2)
    expect(scene.layers.map((l) => [l.nodeId, l.role, l.center[0]])).toEqual([
      ['input', 'input', 0],
      ['hidden_1', 'hidden', LAYER_SPACING],
      ['hidden_2', 'hidden', LAYER_SPACING * 2],
      ['output', 'output', LAYER_SPACING * 3],
    ])
    const keys = new Set(scene.units.map((u) => u.position.join(',')))
    expect(keys.size).toBe(scene.units.length)
  })

  it('builds every unit-to-unit connection for small graphs', () => {
    expect(scene.connections).toHaveLength(2 * 8 + 8 * 6 + 6 * 2)
    expect(scene.detail).toMatchObject({ connectionCount: 76, unitsAsBlocks: false, aggregatedConnections: false })
    expect(scene.layerLinks.map((l) => [l.source, l.target])).toEqual([
      ['input', 'hidden_1'],
      ['hidden_1', 'hidden_2'],
      ['hidden_2', 'output'],
    ])
  })

  it('handles the largest network the editor allows at full detail', () => {
    const largest = planScene(denseGraph(wide([32, 32, 32, 32, 32, 32], 16)))
    expect(largest.detail.connectionCount).toBe(16 * 32 + 5 * 32 * 32 + 32 * 2)
    expect(largest.detail.connectionCount).toBeLessThanOrEqual(MAX_3D_CONNECTIONS)
    expect(largest.detail.aggregatedConnections).toBe(false)
  })

  it('aggregates connections above the connection limit', () => {
    const dense = planScene(denseGraph(wide([100, 100])))
    expect(dense.detail.connectionCount).toBeGreaterThan(MAX_3D_CONNECTIONS)
    expect(dense.connections).toBeNull()
    expect(dense.detail.aggregatedConnections).toBe(true)
    expect(dense.units).toHaveLength(4 + 100 + 100 + 2)
  })

  it('draws layers as blocks above the unit limit', () => {
    const huge = planScene(denseGraph(wide([1500, 1500])))
    expect(huge.detail.unitCount).toBeGreaterThan(MAX_3D_UNITS)
    expect(huge.detail.unitsAsBlocks).toBe(true)
    expect(huge.units).toHaveLength(0)
    expect(huge.layers.every((l) => l.unitIds === null)).toBe(true)
  })

  it('never invents units for layers with unknown sizes', () => {
    const imported = planScene(ModelGraphSchema.parse(onnxGraph))
    expect(imported.units).toHaveLength(0)
    expect(imported.connections).toBeNull()
    expect(imported.layerLinks).toHaveLength(1)
  })

  it('draws imported operator graphs as labeled blocks and links', () => {
    const imported = planScene({ ...example, model: { ...example.model, source_format: 'onnx' } })
    expect(imported.units).toHaveLength(0)
    expect(imported.connections).toBeNull()
    expect(imported.layers.map((l) => l.label)).toEqual(['Input', 'Linear', 'Linear', 'Linear'])
    expect(imported.layerLinks).toHaveLength(3)
  })

  it('does not modify the graph', () => {
    const before = JSON.stringify(example)
    planScene(example)
    expect(JSON.stringify(example)).toBe(before)
  })

  it('falls back to a square viewport before the canvas is measured', () => {
    expect(defaultCamera(scene, 45, 0)).toEqual(defaultCamera(scene, 45, 1))
    expect(defaultCamera(scene, 45, Number.NaN)).toEqual(defaultCamera(scene, 45, 1))
  })

  it.each([0.6, 1, 2.2])('frames every corner of the scene at aspect %s', (aspect) => {
    const view = defaultCamera(scene, 45, aspect)
    expect(view.target).toEqual(scene.bounds.center)

    const project = (distanceScale: number) => {
      const camera = new THREE.PerspectiveCamera(45, aspect, 0.01, 1000)
      const offset = new THREE.Vector3(...view.position).sub(new THREE.Vector3(...view.target))
      camera.position.copy(new THREE.Vector3(...view.target).add(offset.multiplyScalar(distanceScale)))
      camera.lookAt(...view.target)
      camera.updateMatrixWorld()
      return boxCorners(scene.bounds).map((c) => new THREE.Vector3(...c).project(camera))
    }
    const inside = (p: THREE.Vector3) => Math.abs(p.x) <= 1 && Math.abs(p.y) <= 1
    expect(project(1).every(inside)).toBe(true)
    // The fit is tight: moving in by 10% pushes some corner out of view.
    expect(project(0.9).every(inside)).toBe(false)
  })
})

describe('highlightFor', () => {
  const scene = planScene(example)
  const unitId = (nodeId: string, index: number) => scene.layerById.get(nodeId)?.unitIds?.[index] as number

  it('is inactive without a selection', () => {
    const h = highlightFor(scene, null)
    expect(h.active).toBe(false)
    expect(h.units.size + h.connections.size).toBe(0)
  })

  it('emphasizes a unit, its connections, and their other ends', () => {
    const h = highlightFor(scene, { kind: 'unit', nodeId: 'hidden_1', index: 2 })
    expect(h.units.get(unitId('hidden_1', 2))).toBe('selected')
    expect(h.connections.size).toBe(2 + 6)
    // 2 input units + 6 hidden_2 units are related.
    expect([...h.units.values()].filter((e) => e === 'related')).toHaveLength(8)
    expect([...h.links.keys()]).toEqual(['e_input_hidden_1', 'e_hidden_1_hidden_2'])
  })

  it('emphasizes a layer and its incoming connections', () => {
    const h = highlightFor(scene, { kind: 'layer', nodeId: 'hidden_2' })
    expect(h.layers).toEqual(new Set(['hidden_2']))
    expect(h.units.size).toBe(6)
    expect(h.connections.size).toBe(8 * 6)
    expect([...h.links.keys()]).toEqual(['e_hidden_1_hidden_2'])
  })

  it('emphasizes one connection and both of its units', () => {
    const h = highlightFor(scene, {
      kind: 'connection',
      edgeId: 'e_hidden_1_hidden_2',
      sourceIndex: 0,
      targetIndex: 5,
    })
    expect(h.connections.size).toBe(1)
    const [index] = h.connections.keys()
    expect(scene.connections?.[index]).toMatchObject({ sourceIndex: 0, targetIndex: 5 })
    expect(h.units.get(unitId('hidden_1', 0))).toBe('related')
    expect(h.units.get(unitId('hidden_2', 5))).toBe('related')
  })

  it('emphasizes a whole layer-to-layer edge', () => {
    const h = highlightFor(scene, { kind: 'connection', edgeId: 'e_input_hidden_1', sourceIndex: null, targetIndex: null })
    expect(h.connections.size).toBe(16)
    expect(h.links.get('e_input_hidden_1')).toBe('selected')
  })
})
