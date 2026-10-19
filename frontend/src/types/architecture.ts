/**
 * Zod mirror of shared/schemas/dense-network-config.schema.json.
 * Limits match backend/app/schemas/architecture.py; messages are user-facing.
 */
import { z } from 'zod'
import exampleConfig from '../../../shared/schemas/examples/dense-network.config.json'

export const LIMITS = {
  inputUnits: 16,
  hiddenUnits: 32,
  outputUnits: 10,
  hiddenLayers: 6,
} as const

export const ACTIVATIONS = ['relu', 'sigmoid', 'tanh'] as const
export const ACTIVATION_LABELS: Record<Activation, string> = {
  relu: 'ReLU',
  sigmoid: 'Sigmoid',
  tanh: 'Tanh',
}

const units = (max: number, what: string) =>
  z
    .number({ error: 'Enter a whole number.' })
    .int({ error: 'Enter a whole number.' })
    .min(1, { error: `${what} needs at least 1 unit.` })
    .max(max, { error: `${what} can have at most ${max} units.` })

const ActivationSchema = z.enum(ACTIVATIONS, {
  error: 'Choose ReLU, Sigmoid, or Tanh.',
})

export const HiddenLayerConfigSchema = z.strictObject({
  units: units(LIMITS.hiddenUnits, 'A hidden layer'),
  activation: ActivationSchema,
})

export const DenseNetworkConfigSchema = z.strictObject({
  name: z
    .string()
    .trim()
    .min(1, { error: 'Give the network a name.' })
    .max(80, { error: 'Keep the name to 80 characters or fewer.' }),
  input_units: units(LIMITS.inputUnits, 'The input layer'),
  hidden_layers: z
    .array(HiddenLayerConfigSchema)
    .max(LIMITS.hiddenLayers, { error: `Use at most ${LIMITS.hiddenLayers} hidden layers.` }),
  output_units: units(LIMITS.outputUnits, 'The output layer'),
  output_activation: ActivationSchema.nullable().default(null),
})

export type Activation = (typeof ACTIVATIONS)[number]
export type HiddenLayerConfig = z.infer<typeof HiddenLayerConfigSchema>
export type DenseNetworkConfig = z.infer<typeof DenseNetworkConfigSchema>

export const EXAMPLE_CONFIG: DenseNetworkConfig = DenseNetworkConfigSchema.parse(exampleConfig)

/** Field path such as "hidden_layers.0.units" → first error message. */
export type ConfigErrors = Record<string, string>

export function validateConfig(
  config: DenseNetworkConfig,
): { ok: true; value: DenseNetworkConfig } | { ok: false; errors: ConfigErrors } {
  const result = DenseNetworkConfigSchema.safeParse(config)
  if (result.success) return { ok: true, value: result.data }
  const errors: ConfigErrors = {}
  for (const issue of result.error.issues) {
    const key = issue.path.join('.')
    errors[key] ??= issue.message
  }
  return { ok: false, errors }
}
