/** Zod mirror of shared/schemas/tensor-view.schema.json (2D tensor views). */
import { z } from 'zod'

const Cells = z.array(z.array(z.number().nullable()))

const ViewBase = {
  model_id: z.string(),
  run_id: z.string(),
  node_id: z.string(),
  tensor: z.string(),
  shape: z.array(z.number().int()),
  fixed: z.array(z.number().int()),
  /** Finite min/max over the viewed values; null when there are none. */
  min: z.number().nullable(),
  max: z.number().nullable(),
  non_finite_count: z.number().int(),
}

/** One 2D map: the last two dimensions at the `fixed` leading indices. */
export const TensorMapSchema = z.object({
  ...ViewBase,
  rows: z.number().int(),
  cols: z.number().int(),
  /** Block size used to shrink the map to ≤ 65,536 cells (1 = exact). */
  stride: z.number().int(),
  values: Cells,
})
export type TensorMap = z.infer<typeof TensorMapSchema>

/** One thumbnail per index along `axis` (channels, attention heads). */
export const TensorGridSchema = z.object({
  ...ViewBase,
  axis: z.number().int(),
  items_total: z.number().int(),
  rows: z.number().int(),
  cols: z.number().int(),
  items: z.array(
    z.object({
      index: z.number().int(),
      stride: z.number().int(),
      values: Cells,
      min: z.number().nullable(),
      max: z.number().nullable(),
      non_finite_count: z.number().int(),
    }),
  ),
})
export type TensorGrid = z.infer<typeof TensorGridSchema>

export interface ViewQuery {
  runId: string
  nodeId: string
  output: number
  fixed: number[]
}

/** What the tensor viewer is showing. */
export type ViewerTarget =
  | {
      kind: 'tensor'
      nodeId: string
      output: number
      /** Shape in the last run, e.g. [1, 8, 28, 28]. */
      shape: number[]
      /** Role of each dimension, e.g. ["batch", "heads", "query", "key"]. */
      layout: string[]
      attention: boolean
      label: string
    }
  | {
      kind: 'filters'
      nodeId: string
      tensorId: string
      shape: number[]
      label: string
    }
