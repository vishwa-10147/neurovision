import { describe, expect, it } from 'vitest'
import { EXAMPLE_CONFIG, LIMITS, validateConfig, type DenseNetworkConfig } from './architecture'

const withChanges = (changes: Partial<DenseNetworkConfig>) => ({ ...EXAMPLE_CONFIG, ...changes })

describe('validateConfig', () => {
  it('accepts the example', () => {
    expect(validateConfig(EXAMPLE_CONFIG).ok).toBe(true)
  })

  it.each([
    [{ input_units: 0 }, 'input_units', 'The input layer needs at least 1 unit.'],
    [{ input_units: LIMITS.inputUnits + 1 }, 'input_units', 'The input layer can have at most 16 units.'],
    [{ input_units: Number.NaN }, 'input_units', 'Enter a whole number.'],
    [{ output_units: 2.5 }, 'output_units', 'Enter a whole number.'],
    [
      { hidden_layers: [{ units: 40, activation: 'relu' as const }] },
      'hidden_layers.0.units',
      'A hidden layer can have at most 32 units.',
    ],
    [
      { hidden_layers: Array(7).fill({ units: 2, activation: 'relu' }) },
      'hidden_layers',
      'Use at most 6 hidden layers.',
    ],
    [{ name: '   ' }, 'name', 'Give the network a name.'],
  ])('rejects %j with a clear message', (changes, field, message) => {
    const result = validateConfig(withChanges(changes as Partial<DenseNetworkConfig>))
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.errors[field]).toBe(message)
  })
})
