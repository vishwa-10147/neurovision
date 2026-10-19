/**
 * Zod mirror of shared/schemas/model-graph.schema.json (schema version 1.0).
 * The backend Pydantic models are the source of truth; both sides are tested
 * against the fixtures in shared/schemas/examples.
 */
import { z } from 'zod'

const Identifier = z.string().min(1).max(256)

/** Known size, symbolic size such as "batch", or null when unknown. */
export const DimensionSchema = z.union([z.number().int().nonnegative(), z.string().min(1), z.null()])

/** A tensor shape, or null when the rank itself is unknown. */
export const ShapeSchema = z.array(DimensionSchema).nullable()

const JsonValueSchema = z.json()
const NamespacedMetadata = z.record(z.string(), z.record(z.string(), JsonValueSchema))

export const CapabilitiesSchema = z.strictObject({
  graph: z.boolean(),
  inference: z.boolean(),
  training: z.boolean(),
  activations: z.boolean(),
  gradients: z.boolean(),
  attention: z.boolean(),
})

export const ModelInfoSchema = z.strictObject({
  id: Identifier,
  name: z.string(),
  source_format: z.enum(['educational', 'onnx']),
  capabilities: CapabilitiesSchema,
  metadata: NamespacedMetadata.default({}),
})

export const GraphNodeSchema = z.strictObject({
  id: Identifier,
  name: z.string(),
  op_type: z.string().min(1),
  input_tensors: z.array(z.string()).default([]),
  output_tensors: z.array(z.string()).default([]),
  input_shapes: z.array(ShapeSchema).default([]),
  output_shapes: z.array(ShapeSchema).default([]),
  attributes: z.record(z.string(), JsonValueSchema).default({}),
  parameter_summary: z
    .strictObject({
      parameter_count: z.number().int().nonnegative(),
      trainable: z.boolean().nullable().default(null),
    })
    .nullable()
    .default(null),
  unsupported: z.boolean().default(false),
  metadata: NamespacedMetadata.default({}),
})

export const GraphEdgeSchema = z.strictObject({
  id: Identifier,
  source: Identifier,
  target: Identifier,
  tensor_name: z.string().nullable().default(null),
})

export const ModelGraphSchema = z
  .strictObject({
    schema_version: z.literal('1.0'),
    model: ModelInfoSchema,
    nodes: z.array(GraphNodeSchema),
    edges: z.array(GraphEdgeSchema),
    warnings: z.array(z.string()).default([]),
  })
  .superRefine((graph, ctx) => {
    const nodeIds = new Set<string>()
    graph.nodes.forEach((node, i) => {
      if (nodeIds.has(node.id)) {
        ctx.addIssue({ code: 'custom', path: ['nodes', i, 'id'], message: `duplicate node id '${node.id}'` })
      }
      nodeIds.add(node.id)
      for (const [shapes, tensors, key] of [
        [node.input_shapes, node.input_tensors, 'input_shapes'],
        [node.output_shapes, node.output_tensors, 'output_shapes'],
      ] as const) {
        if (shapes.length > 0 && shapes.length !== tensors.length) {
          ctx.addIssue({
            code: 'custom',
            path: ['nodes', i, key],
            message: `node '${node.id}' has ${shapes.length} ${key.replace('_', ' ')} for ${tensors.length} tensors`,
          })
        }
      }
    })
    const edgeIds = new Set<string>()
    graph.edges.forEach((edge, i) => {
      if (edgeIds.has(edge.id)) {
        ctx.addIssue({ code: 'custom', path: ['edges', i, 'id'], message: `duplicate edge id '${edge.id}'` })
      }
      edgeIds.add(edge.id)
      for (const end of ['source', 'target'] as const) {
        if (!nodeIds.has(edge[end])) {
          ctx.addIssue({
            code: 'custom',
            path: ['edges', i, end],
            message: `edge '${edge.id}' references unknown node '${edge[end]}'`,
          })
        }
      }
    })
  })

export type Dimension = z.infer<typeof DimensionSchema>
export type Shape = z.infer<typeof ShapeSchema>
export type Capabilities = z.infer<typeof CapabilitiesSchema>
export type GraphNode = z.infer<typeof GraphNodeSchema>
export type GraphEdge = z.infer<typeof GraphEdgeSchema>
export type ModelGraph = z.infer<typeof ModelGraphSchema>
