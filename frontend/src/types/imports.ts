/** Zod mirror of shared/schemas/import.schema.json. */
import { z } from 'zod'
import { ModelGraphSchema } from './graph'

/** Mirrors the backend default (NEUROVISION_MAX_UPLOAD_BYTES); the server enforces it. */
export const MAX_UPLOAD_BYTES = 64 * 1024 * 1024

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
})
export type ImportReport = z.infer<typeof ImportReportSchema>

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
export function checkModelFile(file: File): string | null {
  if (!file.name.toLowerCase().endsWith('.onnx')) return 'Choose an .onnx file. Other formats are not supported yet.'
  if (file.size === 0) return 'The file is empty.'
  if (file.size > MAX_UPLOAD_BYTES) {
    return `The file is ${(file.size / 1024 / 1024).toFixed(1)} MB; the limit is ${MAX_UPLOAD_BYTES / 1024 / 1024} MB.`
  }
  return null
}
