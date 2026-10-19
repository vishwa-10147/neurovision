import { describe, expect, it } from 'vitest'
import { activationFill, formatValue } from './values'

describe('formatValue', () => {
  it.each([
    [0, 3, '0.000'],
    [0.12345, 3, '0.123'],
    [-1.5, 4, '-1.5000'],
    [12345, 3, '1.23e+4'],
    // Tiny non-zero values must not round to zero.
    [0.00004, 4, '4.00e-5'],
    [-0.0002, 3, '-2.00e-4'],
    [Number.NaN, 3, 'NaN'],
  ])('formats %s with %s digits as %s', (value, digits, expected) => {
    expect(formatValue(value, digits)).toBe(expected)
  })
})

describe('activationFill', () => {
  it('scales strength by the largest magnitude and picks hue by sign', () => {
    expect(activationFill(2, 4)).toBe('color-mix(in srgb, var(--act-pos) 50%, var(--act-zero))')
    expect(activationFill(-4, 4)).toBe('color-mix(in srgb, var(--act-neg) 100%, var(--act-zero))')
    expect(activationFill(0, 0)).toBe('color-mix(in srgb, var(--act-pos) 0%, var(--act-zero))')
  })
})
