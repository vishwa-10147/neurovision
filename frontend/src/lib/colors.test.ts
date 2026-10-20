import { describe, expect, it } from 'vitest'
import { parseColor, scaleColor, type Palette } from './colors'

const palette: Palette = { negative: [255, 0, 0], zero: [0, 0, 0], positive: [0, 0, 255], missing: [0, 255, 0] }

describe('color scales', () => {
  it('parses hex theme colors and falls back otherwise', () => {
    expect(parseColor(' #4f9dff ', [1, 2, 3])).toEqual([79, 157, 255])
    expect(parseColor('rgb(1, 2, 3)', [1, 2, 3])).toEqual([1, 2, 3])
  })

  it('diverges from zero by sign and clamps at the limit', () => {
    const scale = { kind: 'diverging', limit: 2 } as const
    expect(scaleColor(0, scale, palette)).toEqual([0, 0, 0])
    expect(scaleColor(1, scale, palette)).toEqual([0, 0, 128])
    expect(scaleColor(-2, scale, palette)).toEqual([255, 0, 0])
    expect(scaleColor(-9, scale, palette)).toEqual([255, 0, 0])
    // An all-zero tensor has limit 0: everything stays neutral.
    expect(scaleColor(3, { kind: 'diverging', limit: 0 }, palette)).toEqual([0, 0, 0])
  })

  it('maps a sequential range and marks NaN/∞ as missing', () => {
    const scale = { kind: 'sequential', min: 0, max: 1 } as const
    expect(scaleColor(0.5, scale, palette)).toEqual([0, 0, 128])
    expect(scaleColor(1.5, scale, palette)).toEqual([0, 0, 255])
    expect(scaleColor(null, scale, palette)).toEqual([0, 255, 0])
    expect(scaleColor(Number.POSITIVE_INFINITY, scale, palette)).toEqual([0, 255, 0])
  })
})
