import { describe, expect, it } from 'vitest'
import { StepHealthSchema } from '../../types/events'
import { linkedGraph } from '../../test/linkedGraph'
import { indexModules, liveColors, modulePath, nodesOf, statsForNode } from './graphMatching'

describe('modulePath', () => {
  it('reads the module from PyTorch-exported node names', () => {
    expect(modulePath('/fc1/Gemm')).toBe('fc1')
    expect(modulePath('/net/net.0/Gemm')).toBe('net.0')
    expect(modulePath('/layer1/layer1.0/relu_2/Relu')).toBe('layer1.0.relu_2')
    expect(modulePath('/encoder/layers/encoder_layer_0/self_attention/Softmax')).toBe(
      'encoder.layers.encoder_layer_0.self_attention',
    )
    expect(modulePath('/Flatten')).toBeNull()
    expect(modulePath('node_3')).toBeNull()
  })
})

const health = StepHealthSchema.parse({
  grad_norm: 1.2,
  parameters: { fc1: { grad_norm: 0.5, parameters: 96 }, block: { grad_norm: 2.0, parameters: 528 } },
  activations: {
    act1: { kind: 'relu', mean: 3, std: 4, dead_fraction: 0.25 },
    'block.1': { kind: 'relu', mean: 0, std: 1, dead_fraction: 0 },
    missing: { kind: 'tanh', mean: 0, std: 1 },
  },
})

describe('liveColors', () => {
  const index = indexModules(linkedGraph())

  it('puts an activation on the last operator of its module', () => {
    expect(nodesOf(index, 'block.1')).toEqual(['n3', 'n4'])
    const colors = liveColors(index, 'activations', health, {})
    expect(colors.values).toEqual({ n1: 5, n4: 1 })
    expect(colors.matched).toEqual(['act1', 'block.1'])
    expect(colors.unmatched).toEqual(['missing'])
  })

  it('prefers values sent by a probe over the health summary', () => {
    const colors = liveColors(index, 'activations', health, { act1: [3, 4, 0, 0] })
    expect(colors.values.n1).toBeCloseTo(2.5)
  })

  it('spreads a grouped gradient norm over the operators that own weights', () => {
    expect(liveColors(index, 'gradients', health, {}).values).toEqual({ n0: 0.5, n2: 2 })
  })

  it('shows dead units for ReLU layers', () => {
    expect(liveColors(index, 'dead', health, {}).values).toEqual({ n1: 0.25, n4: 0 })
  })

  it('finds the statistics behind a selected operator', () => {
    const graph = linkedGraph()
    const gemm = statsForNode(graph, 'n2', health, {})
    expect(gemm.module).toBe('block')
    expect(gemm.parameters?.grad_norm).toBe(2)
    expect(statsForNode(graph, 'n4', health, {}).activation?.std).toBe(1)
    expect(statsForNode(graph, 'n5', health, {}).module).toBeNull()
  })
})
