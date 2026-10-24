import { describe, expect, it } from 'vitest'
import { layoutTree, majority } from '../../components/tree/treeLayout'
import { ALGORITHM_INFO } from './classical'

const node = (id: number, depth: number, left: number | null, right: number | null, counts = [1, 1]) => ({
  id,
  depth,
  feature: left === null ? null : 0,
  threshold: left === null ? null : 0.5,
  left,
  right,
  samples: counts.reduce((a, b) => a + b, 0),
  counts,
  impurity: 0.5,
})

describe('tree layout', () => {
  it('spaces leaves evenly and centers parents over their children', () => {
    const layout = layoutTree([
      node(0, 0, 1, 2),
      node(1, 1, null, null),
      node(2, 1, 3, 4),
      node(3, 2, null, null),
      node(4, 2, null, null),
    ])
    const x = Object.fromEntries(layout.nodes.map((n) => [n.id, n.x]))
    expect([x[1], x[3], x[4]]).toEqual([0, 1, 2])
    expect(x[2]).toBe(1.5)
    expect(x[0]).toBe(0.75)
    expect(layout.width).toBe(3)
    expect(layout.depth).toBe(3)
  })

  it('treats a node whose children were cut off as a leaf', () => {
    expect(layoutTree([node(0, 0, 7, 8)]).width).toBe(1)
  })
})

describe('majority', () => {
  it('finds the majority class and its share', () => {
    expect(majority([2, 6, 2])).toEqual({ label: 1, share: 0.6 })
  })
})

describe('algorithm info', () => {
  it('offers digits only where the result can be shown', () => {
    expect(ALGORITHM_INFO.decision_tree.datasets).not.toContain('digits')
    expect(ALGORITHM_INFO.pca.datasets).toContain('digits')
  })
})
