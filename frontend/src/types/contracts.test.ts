import { describe, expect, it } from 'vitest'
import exampleConfig from '../../../shared/schemas/examples/dense-network.config.json'
import denseGraph from '../../../shared/schemas/examples/dense-network.graph.json'
import invalidCases from '../../../shared/schemas/examples/invalid-cases.json'
import onnxGraph from '../../../shared/schemas/examples/onnx-import.graph.json'
import stepEvent from '../../../shared/schemas/examples/training-step.event.json'
import { DenseNetworkConfigSchema } from './architecture'
import { TrainingEventSchema } from './events'
import { ModelGraphSchema } from './graph'

// The same fixtures are validated by backend/tests/test_schemas.py, which keeps
// the Zod mirrors and the Pydantic source of truth in agreement.

describe('shared contract fixtures', () => {
  it.each([
    ['dense-network', denseGraph],
    ['onnx-import', onnxGraph],
  ])('accepts the %s example graph', (_, graph) => {
    expect(ModelGraphSchema.safeParse(graph).success).toBe(true)
  })

  it('keeps unknown dimensions as null instead of inventing them', () => {
    const graph = ModelGraphSchema.parse(onnxGraph)
    expect(graph.nodes[1].unsupported).toBe(true)
    expect(graph.nodes[1].output_shapes).toEqual([['N', null, null, null]])
  })

  it('accepts the example training event', () => {
    const event = TrainingEventSchema.parse(stepEvent)
    expect(event.event_type).toBe('training.step')
  })

  it.each(invalidCases.map((c) => [c.description, c] as const))('rejects: %s', (_, testCase) => {
    const schema =
      testCase.kind === 'graph'
        ? ModelGraphSchema
        : testCase.kind === 'architecture'
          ? DenseNetworkConfigSchema
          : TrainingEventSchema
    expect(schema.safeParse(testCase.document).success).toBe(false)
  })

  it('accepts the example dense-network config', () => {
    expect(DenseNetworkConfigSchema.safeParse(exampleConfig).success).toBe(true)
  })

  it('explains integrity errors with the offending id', () => {
    const dangling = invalidCases.find((c) => c.description.startsWith('edge'))
    const result = ModelGraphSchema.safeParse(dangling?.document)
    expect(result.success).toBe(false)
    expect(result.error?.issues[0].message).toBe("edge 'e' references unknown node 'missing'")
    expect(result.error?.issues[0].path).toEqual(['edges', 0, 'target'])
  })

  it('rejects non-finite metric values', () => {
    const event = { ...stepEvent, metrics: { loss: Number.POSITIVE_INFINITY } }
    expect(TrainingEventSchema.safeParse(event).success).toBe(false)
  })
})
