import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { TreeEnsembleView } from './TreeEnsembleView'

const stump = (feature: number, threshold: number) => [
  { id: 0, depth: 0, feature, threshold, mode: 'BRANCH_LEQ', left: 1, right: 2, values: [] },
  { id: 1, depth: 1, feature: null, threshold: null, mode: 'LEAF', left: null, right: null, values: [0.9, 0.1] },
  { id: 2, depth: 1, feature: null, threshold: null, mode: 'LEAF', left: null, right: null, values: [0.2, 0.8] },
]

describe('TreeEnsembleView', () => {
  it('draws the trees of an ONNX tree ensemble, one at a time', async () => {
    render(
      <TreeEnsembleView
        metadata={{ kind: 'classifier', trees: 5, outputs: 2, nodes_total: 15, shown: [stump(3, 0.8), stump(0, 5.5)] }}
      />,
    )
    expect(screen.getByText(/5 trees, 15 nodes in total; the first 2 are drawn/)).toBeInTheDocument()
    expect(screen.getByRole('img', { name: 'Tree 1 with 3 nodes' })).toBeInTheDocument()
    expect(screen.getByText('f3 ≤ 0.80')).toBeInTheDocument()
    expect(screen.getByText('class 1')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Tree 2' }))
    expect(screen.getByText('f0 ≤ 5.50')).toBeInTheDocument()
  })

  it('ignores metadata it does not understand', () => {
    const { container } = render(<TreeEnsembleView metadata={{ trees: 'many' }} />)
    expect(container).toBeEmptyDOMElement()
  })
})
