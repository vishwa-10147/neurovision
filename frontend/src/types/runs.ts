/**
 * Zod mirror of RunInfo/RunList in shared/schemas/external-run.schema.json.
 * External runs are training scripts outside NeuroVision that stream their own
 * metrics and activations; their events use the training event format.
 */
import { z } from 'zod'

export const RUN_STATUSES = ['running', 'completed', 'failed'] as const
export type RunStatus = (typeof RUN_STATUSES)[number]

export const RunInfoSchema = z.object({
  run_id: z.string(),
  name: z.string(),
  model_id: z.string().nullable(),
  status: z.enum(RUN_STATUSES),
  total_epochs: z.number().int(),
  epoch: z.number().int(),
  step: z.number().int(),
  steps_logged: z.number().int(),
  /** Unix seconds. */
  created_at: z.number(),
  last_event_at: z.number(),
})
export type RunInfo = z.infer<typeof RunInfoSchema>

export const RunListSchema = z.object({ runs: z.array(RunInfoSchema) })

/** One logged step as charted: any metrics the script chose to send. */
export interface RunPoint {
  epoch: number
  step: number
  metrics: Record<string, number>
}
