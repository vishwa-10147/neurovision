import { describe, expect, it } from 'vitest'
import { scatterBounds, toRows } from './visuals'

describe('visual helpers', () => {
  it('reshapes row-major values', () => {
    expect(toRows([1, 2, 3, 4, 5, 6], 2, 3)).toEqual([
      [1, 2, 3],
      [4, 5, 6],
    ])
  })
  it('keeps a single point drawable', () => {
    expect(scatterBounds([2], [3])).toMatchObject({ minX: 2, spanX: 1, spanY: 1 })
  })
})
