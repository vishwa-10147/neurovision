/** Zod mirror of shared/schemas/import.schema.json. */
import { z } from 'zod'
import { ModelGraphSchema } from './graph'

/** Mirrors the backend default (NEUROVISION_MAX_UPLOAD_BYTES); the server enforces it. */
export const MAX_UPLOAD_BYTES = 64 * 1024 * 1024

export const InputSpecSchema = z.object({
  name: z.string(),
  node_id: z.string(),
  dtype: z.string(),
  /** Declared shape: numbers are fixed, strings are named (any size), null is unknown. */
  shape: z.array(z.union([z.number().int(), z.string(), z.null()])).nullable(),
})
export type InputSpec = z.infer<typeof InputSpecSchema>

export const ImportReportSchema = z.object({
  format: z.literal('onnx'),
  file_name: z.string(),
  file_bytes: z.number().int(),
  operator_count: z.number().int(),
  edge_count: z.number().int(),
  initializer_count: z.number().int(),
  parameter_count: z.number().int(),
  external_tensor_count: z.number().int(),
  unsupported_operators: z.array(z.object({ op_type: z.string(), domain: z.string(), count: z.number().int() })),
  opsets: z.array(z.object({ domain: z.string(), version: z.number().int() })),
  warnings: z.array(z.string()),
  inputs: z.array(InputSpecSchema).default([]),
  /** Why the model cannot run here; empty when it is expected to run. */
  inference_blockers: z.array(z.string()).default([]),
  /** Softmax nodes over square [..., S, S] tensors: likely attention weights. */
  attention: z
    .array(
      z.object({
        node_id: z.string(),
        name: z.string(),
        /** The Softmax output tensor, as named in run results. */
        tensor: z.string(),
        shape: z.array(z.union([z.number().int(), z.string(), z.null()])).nullable(),
        layout: z.array(z.string()),
      }),
    )
    .default([]),
})
export type AttentionCandidate = ImportReport['attention'][number]
export type ImportReport = z.infer<typeof ImportReportSchema>

/** Mirrors backend/app/schemas/inference.py (shared/schemas/onnx-run.schema.json). */
export const RUN_LIMITS = { maxElements: 1_000_000, maxDimension: 100_000, maxRank: 8, maxTypedValues: 4096 } as const
export const FILLS = ['random', 'zeros', 'ones', 'range'] as const
export type Fill = (typeof FILLS)[number]

export interface RunInput {
  shape: number[]
  values?: number[]
  fill?: Fill
  seed?: number
}

export const TensorValuesSchema = z.object({
  name: z.string(),
  node_id: z.string().nullable(),
  shape: z.array(z.number().int()),
  dtype: z.string(),
  count: z.number().int(),
  summary: z.object({ min: z.number(), max: z.number(), mean: z.number(), std: z.number() }).nullable(),
  non_finite_count: z.number().int(),
  values: z.array(z.number().nullable()),
  truncated: z.boolean(),
})
export type TensorValues = z.infer<typeof TensorValuesSchema>

export const RunResponseSchema = z.object({
  model_id: z.string(),
  /** Identifies this run's kept tensors for map views. */
  run_id: z.string(),
  /** False when the tensors were too large to keep (no map views). */
  retained: z.boolean(),
  inputs: z
    .array(z.object({ name: z.string(), node_id: z.string(), shape: z.array(z.number().int()), dtype: z.string() }))
    .default([]),
  runtime: z.string(),
  elapsed_ms: z.number(),
  outputs: z.array(TensorValuesSchema),
  intermediates: z.array(TensorValuesSchema),
  notes: z.array(z.string()),
})
export type RunResponse = z.infer<typeof RunResponseSchema>

export const ImportResponseSchema = z.object({
  model_id: z.string(),
  graph: ModelGraphSchema,
  report: ImportReportSchema,
})
export type ImportResponse = z.infer<typeof ImportResponseSchema>

export const SampleModelSchema = z.object({ name: z.string(), title: z.string(), description: z.string() })
export type SampleModel = z.infer<typeof SampleModelSchema>

/** Initializer references stored on imported operator nodes (metadata.onnx.initializers). */
export const InitializerRefSchema = z.object({
  tensor_id: z.string(),
  name: z.string(),
  shape: z.array(z.number().int()),
  dtype: z.string(),
  external: z.boolean(),
})
export type InitializerRef = z.infer<typeof InitializerRefSchema>

/** Client-side checks before uploading; the server repeats them. */
export function checkModelFile(file: File, limit: number = MAX_UPLOAD_BYTES): string | null {
  if (!file.name.toLowerCase().endsWith('.onnx')) return 'Choose an .onnx file. Other formats are not supported yet.'
  if (file.size === 0) return 'The file is empty.'
  if (file.size > limit) {
    return `The file is ${(file.size / 1024 / 1024).toFixed(1)} MB; the limit is ${Math.round(limit / 1024 / 1024)} MB.`
  }
  return null
}
