import { describe, expect, it } from 'vitest'
import { describeShape, layoutFor, singular, tensorTarget } from './viewerTargets'

const attention = { node_id: 'n', name: 'softmax', tensor: 'w', shape: ['batch', 4, 'seq', 'seq'], layout: ['batch', 'heads', 'query', 'key'] }

describe('viewer targets', () => {
  it('assumes NCHW for 4D tensors unless the tensor is attention', () => {
    expect(layoutFor([1, 8, 28, 28])).toEqual(['batch', 'channels', 'height', 'width'])
    expect(layoutFor([1, 4, 7, 7], attention)).toEqual(['batch', 'heads', 'query', 'key'])
    // A layout of the wrong rank is ignored rather than mislabelling dimensions.
    expect(layoutFor([4, 7, 7], attention)).toEqual(['batch', 'rows', 'columns'])
    expect(layoutFor([2, 3, 4, 5, 6])).toEqual(['dim 0', 'dim 1', 'dim 2', 'rows', 'columns'])
  })

  it('names one item along an axis', () => {
    expect(singular('channels')).toBe('channel')
    expect(singular('heads')).toBe('head')
    expect(singular('dim 0')).toBe('dim 0')
  })

  it('describes shapes with dimension roles', () => {
    expect(describeShape([1, 8, 28, 28], layoutFor([1, 8, 28, 28]))).toBe('[1 batch × 8 channels × 28 height × 28 width]')
    expect(tensorTarget('n', 0, [1, 4, 7, 7], 'w', attention)).toMatchObject({ kind: 'tensor', attention: true, layout: attention.layout })
  })
})
