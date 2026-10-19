/**
 * Test double for POST /api/v1/models/educational: builds the same graph shape
 * as backend/app/services/educational_graph.py (which has its own tests).
 */
import type { DenseNetworkConfig } from '../types/architecture'
import { ModelGraphSchema, type ModelGraph } from '../types/graph'

export function denseGraph(config: DenseNetworkConfig, modelId = 'edu_test'): ModelGraph {
  const layers = [
    ...config.hidden_layers.map((l) => [l.units, l.activation] as const),
    [config.output_units, config.output_activation] as const,
  ]
  const nodes: unknown[] = [
    { id: 'input', name: 'Input', op_type: 'Input', output_tensors: ['x'], output_shapes: [['batch', config.input_units]] },
  ]
  const edges: unknown[] = []
  let prev = { id: 'input', tensor: 'x', units: config.input_units }
  layers.forEach(([units, activation], i) => {
    const isOutput = i === layers.length - 1
    const id = isOutput ? 'output' : `hidden_${i + 1}`
    const tensor = isOutput ? 'logits' : `h${i + 1}`
    nodes.push({
      id,
      name: isOutput ? 'Output Layer' : `Hidden Layer ${i + 1}`,
      op_type: 'Linear',
      input_tensors: [prev.tensor],
      output_tensors: [tensor],
      input_shapes: [['batch', prev.units]],
      output_shapes: [['batch', units]],
      attributes: { in_features: prev.units, out_features: units, activation },
      parameter_summary: { parameter_count: prev.units * units + units, trainable: true },
    })
    edges.push({ id: `e_${prev.id}_${id}`, source: prev.id, target: id, tensor_name: prev.tensor })
    prev = { id, tensor, units }
  })
  return ModelGraphSchema.parse({
    schema_version: '1.0',
    model: {
      id: modelId,
      name: config.name,
      source_format: 'educational',
      capabilities: { graph: true, inference: false, training: false, activations: false, gradients: false, attention: false },
    },
    nodes,
    edges,
  })
}
