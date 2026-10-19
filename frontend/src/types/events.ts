/**
 * Zod mirror of shared/schemas/training-event.schema.json (schema version 1.0).
 * Limits match backend/app/schemas/events.py.
 */
import { z } from 'zod'

export const MAX_METRICS = 16
export const MAX_ACTIVATION_LAYERS = 8
export const MAX_VALUES_PER_LAYER = 256

const Identifier = z.string().min(1).max(256)
const Count = z.number().int().nonnegative()
// z.number() rejects NaN and ±Infinity.
const Metrics = z
  .record(z.string(), z.number())
  .refine((m) => Object.keys(m).length <= MAX_METRICS, `at most ${MAX_METRICS} metrics`)

const base = {
  schema_version: z.literal('1.0'),
  job_id: Identifier,
}

export const TrainingStartedSchema = z.strictObject({
  ...base,
  event_type: z.literal('training.started'),
  model_id: Identifier,
  total_epochs: z.number().int().min(1),
})

export const TrainingStepSchema = z.strictObject({
  ...base,
  event_type: z.literal('training.step'),
  epoch: Count,
  step: Count,
  metrics: Metrics,
  selected_activations: z
    .record(Identifier, z.array(z.number()).max(MAX_VALUES_PER_LAYER))
    .refine(
      (a) => Object.keys(a).length <= MAX_ACTIVATION_LAYERS,
      `at most ${MAX_ACTIVATION_LAYERS} activation layers`,
    )
    .default({}),
})

export const TrainingCompletedSchema = z.strictObject({
  ...base,
  event_type: z.literal('training.completed'),
  epoch: Count,
  step: Count,
  metrics: Metrics,
})

export const TrainingCancelledSchema = z.strictObject({
  ...base,
  event_type: z.literal('training.cancelled'),
  step: Count,
})

export const TrainingFailedSchema = z.strictObject({
  ...base,
  event_type: z.literal('training.failed'),
  message: z.string().max(500),
})

export const TrainingEventSchema = z.discriminatedUnion('event_type', [
  TrainingStartedSchema,
  TrainingStepSchema,
  TrainingCompletedSchema,
  TrainingCancelledSchema,
  TrainingFailedSchema,
])

export type TrainingEvent = z.infer<typeof TrainingEventSchema>
export type TrainingStep = z.infer<typeof TrainingStepSchema>
