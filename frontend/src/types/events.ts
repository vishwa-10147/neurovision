/**
 * Zod mirror of shared/schemas/training-event.schema.json (schema version 1.0).
 * Limits match backend/app/schemas/events.py.
 */
import { z } from 'zod'

export const MAX_METRICS = 16
export const MAX_ACTIVATION_LAYERS = 8
export const MAX_VALUES_PER_LAYER = 256
export const MAX_HEALTH_LAYERS = 32

const Identifier = z.string().min(1).max(256)
const Count = z.number().int().nonnegative()
const NonNegative = z.number().nonnegative()
const Fraction = z.number().min(0).max(1)
const Message = z.string().max(300)
// z.number() rejects NaN and ±Infinity.
const Metrics = z
  .record(z.string(), z.number())
  .refine((m) => Object.keys(m).length <= MAX_METRICS, `at most ${MAX_METRICS} metrics`)
const atMost = (n: number) => (r: Record<string, unknown>) => Object.keys(r).length <= n

/** Training-health statistics measured by the script (ADR 0008). */
export const ParameterHealthSchema = z.strictObject({
  grad_norm: NonNegative.nullable().default(null),
  weight_norm: NonNegative.nullable().default(null),
  /** ‖Δw‖ / ‖w‖ since the previous measurement. */
  update_ratio: NonNegative.nullable().default(null),
  parameters: Count.default(0),
})

export const ActivationHealthSchema = z.strictObject({
  kind: z.enum(['relu', 'tanh', 'sigmoid', 'other']),
  mean: z.number(),
  std: NonNegative,
  dead_fraction: Fraction.nullable().default(null),
  saturated_fraction: Fraction.nullable().default(null),
})

export const StepHealthSchema = z.strictObject({
  grad_norm: NonNegative.nullable().default(null),
  learning_rate: NonNegative.nullable().default(null),
  nonfinite_grads: Count.default(0),
  nonfinite_metrics: z.array(Identifier).max(MAX_METRICS).default([]),
  parameters: z
    .record(Identifier, ParameterHealthSchema)
    .refine(atMost(MAX_HEALTH_LAYERS), `at most ${MAX_HEALTH_LAYERS} parameter groups`)
    .default({}),
  activations: z
    .record(Identifier, ActivationHealthSchema)
    .refine(atMost(MAX_HEALTH_LAYERS), `at most ${MAX_HEALTH_LAYERS} activation layers`)
    .default({}),
})

export const COMMAND_KINDS = ['pause', 'resume', 'stop', 'checkpoint', 'set_lr'] as const
export type CommandKind = (typeof COMMAND_KINDS)[number]

export const RunCommandSchema = z.strictObject({
  command_id: z.number().int().min(1),
  kind: z.enum(COMMAND_KINDS),
  value: z.number().positive().nullable().default(null),
  status: z.enum(['pending', 'applied', 'rejected', 'expired']),
  message: Message.nullable().default(null),
  created_at: z.number(),
  resolved_at: z.number().nullable().default(null),
})

export const ALERT_SEVERITIES = ['info', 'warning', 'critical'] as const

export const RunAlertSchema = z.strictObject({
  alert_id: z.number().int().min(1),
  kind: z.string().min(1),
  severity: z.enum(ALERT_SEVERITIES),
  step: Count,
  layer: Identifier.nullable().default(null),
  message: Message,
  advice: Message,
})

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
  /** External runs only: statistics the script measured. */
  health: StepHealthSchema.nullable().default(null),
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

/** A command to an external run was issued, or its status changed. */
export const RunCommandEventSchema = z.strictObject({
  ...base,
  event_type: z.literal('run.command'),
  command: RunCommandSchema,
  paused: z.boolean(),
  learning_rate: NonNegative.nullable().default(null),
})

/** The server's health rules flagged a problem in an external run. */
export const RunAlertEventSchema = z.strictObject({
  ...base,
  event_type: z.literal('run.alert'),
  alert: RunAlertSchema,
})

export const TrainingEventSchema = z.discriminatedUnion('event_type', [
  TrainingStartedSchema,
  TrainingStepSchema,
  TrainingCompletedSchema,
  TrainingCancelledSchema,
  TrainingFailedSchema,
  RunCommandEventSchema,
  RunAlertEventSchema,
])

export type TrainingEvent = z.infer<typeof TrainingEventSchema>
export type TrainingStep = z.infer<typeof TrainingStepSchema>
export type StepHealth = z.infer<typeof StepHealthSchema>
export type RunCommand = z.infer<typeof RunCommandSchema>
export type RunAlert = z.infer<typeof RunAlertSchema>
