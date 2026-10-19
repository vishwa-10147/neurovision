/**
 * Zod mirrors of shared/schemas/training-config.schema.json, forward.schema.json,
 * and tensor.schema.json. Limits match backend/app/schemas/training.py.
 */
import { z } from 'zod'

export const TRAINING_LIMITS = {
  samples: [50, 2000],
  noise: [0, 1],
  validationSplit: [0.1, 0.5],
  epochs: [1, 500],
  batchSize: [1, 512],
  learningRate: [0.00001, 1],
  epochDelayMs: [0, 500],
  forwardRows: 64,
  inputMagnitude: 1e6,
} as const

export const DATASETS = ['moons', 'circles', 'spiral'] as const
export type DatasetName = (typeof DATASETS)[number]
/** Datasets with a fixed class count; spiral has one arm per output unit. */
export const FIXED_CLASSES: Partial<Record<DatasetName, number>> = { moons: 2, circles: 2 }

const whole = (min: number, max: number, what: string) =>
  z
    .number({ error: 'Enter a whole number.' })
    .int({ error: 'Enter a whole number.' })
    .min(min, { error: `${what} must be at least ${min}.` })
    .max(max, { error: `${what} can be at most ${max}.` })

export const TrainingConfigSchema = z.strictObject({
  dataset: z.enum(DATASETS),
  samples: whole(...TRAINING_LIMITS.samples, 'Samples'),
  noise: z
    .number({ error: 'Enter a number.' })
    .min(0, { error: 'Noise cannot be negative.' })
    .max(1, { error: 'Noise can be at most 1.' }),
  validation_split: z
    .number({ error: 'Enter a number.' })
    .min(0.1, { error: 'Hold out at least 10% for validation.' })
    .max(0.5, { error: 'Hold out at most 50% for validation.' }),
  epochs: whole(...TRAINING_LIMITS.epochs, 'Epochs'),
  batch_size: whole(...TRAINING_LIMITS.batchSize, 'Batch size'),
  learning_rate: z
    .number({ error: 'Enter a number.' })
    .gt(0, { error: 'The learning rate must be greater than 0.' })
    .max(1, { error: 'The learning rate can be at most 1.' }),
  optimizer: z.enum(['sgd', 'adam']),
  seed: whole(0, 2 ** 31 - 1, 'The seed'),
  capture_layers: z.array(z.string()).max(8, { error: 'Capture at most 8 layers.' }),
  epoch_delay_ms: whole(...TRAINING_LIMITS.epochDelayMs, 'The delay'),
})
export type TrainingConfig = z.infer<typeof TrainingConfigSchema>

export const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
  dataset: 'moons',
  samples: 400,
  noise: 0.1,
  validation_split: 0.2,
  epochs: 100,
  batch_size: 32,
  learning_rate: 0.03,
  optimizer: 'adam',
  seed: 0,
  capture_layers: [],
  // A short pause per epoch so the charts can be watched filling in.
  epoch_delay_ms: 20,
}

export const JOB_STATUSES = ['running', 'completed', 'cancelled', 'failed'] as const
export type JobStatus = (typeof JOB_STATUSES)[number]

export const TrainingJobInfoSchema = z.object({
  job_id: z.string(),
  model_id: z.string(),
  status: z.enum(JOB_STATUSES),
  epoch: z.number().int(),
  total_epochs: z.number().int(),
  config: TrainingConfigSchema,
})
export type TrainingJobInfo = z.infer<typeof TrainingJobInfoSchema>

export const EpochMetricsSchema = z.object({
  epoch: z.number().int(),
  loss: z.number(),
  accuracy: z.number(),
  val_loss: z.number(),
  val_accuracy: z.number(),
})
export type EpochMetrics = z.infer<typeof EpochMetricsSchema>

export const MetricsResponseSchema = z.object({
  model_id: z.string(),
  job: TrainingJobInfoSchema.nullable(),
  history: z.array(EpochMetricsSchema),
})

const Matrix = z.array(z.array(z.number()))

export const ForwardResponseSchema = z.object({
  model_id: z.string(),
  outputs: Matrix,
  logits: Matrix,
  predictions: z.array(z.number().int()),
  activations: z.record(z.string(), z.object({ pre: Matrix.nullable(), post: Matrix })),
})
export type ForwardResponse = z.infer<typeof ForwardResponseSchema>

export const TensorResponseSchema = z.object({
  tensor_id: z.string(),
  shape: z.array(z.number().int()),
  dtype: z.string(),
  count: z.number().int(),
  /** Over finite values only; null when there are none. */
  summary: z.object({ min: z.number(), max: z.number(), mean: z.number(), std: z.number() }).nullable(),
  /** NaN or infinite values: excluded from the summary and null in `values`. */
  non_finite_count: z.number().int().default(0),
  offset: z.number().int(),
  values: z.array(z.number().nullable()),
  truncated: z.boolean(),
})
export type TensorResponse = z.infer<typeof TensorResponseSchema>
