import { ModelGraphSchema } from '../types/graph'

/** fc1 (Gemm, weights) → act1 (Relu) → block.0 (Gemm, weights) → block.1 (Relu, called twice) → out. */
export function linkedGraph() {
  const node = (id: string, name: string, op_type: string, weights = 0) => ({
    id,
    name,
    op_type,
    input_tensors: [],
    output_tensors: [`${id}_out`],
    input_shapes: [],
    output_shapes: [],
    parameter_summary: weights ? { parameter_count: weights } : null,
  })
  return ModelGraphSchema.parse({
    schema_version: '1.0',
    model: {
      id: 'onnx_live',
      name: 'spiral',
      source_format: 'onnx',
      capabilities: { graph: true, inference: true, activations: true, training: false, gradients: false, attention: false },
    },
    nodes: [
      node('input_0', 'input', 'Input'),
      node('n0', '/fc1/Gemm', 'Gemm', 96),
      node('n1', '/act1/Relu', 'Relu'),
      node('n2', '/block/block.0/Gemm', 'Gemm', 528),
      node('n3', '/block/block.1/Relu', 'Relu'),
      node('n4', '/block/block.1_1/Relu', 'Relu'),
      node('n5', '/Flatten', 'Flatten'),
      node('output_0', 'output', 'Output'),
    ],
    edges: [],
    warnings: [],
  })
}
