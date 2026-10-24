import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { fakeBackend, type FakeBackend } from '../../test/fakeBackend'
import { ClassicalLab } from './ClassicalLab'

let backend: FakeBackend
const cells = 40 * 40
const frame = (label: string, accuracy: number) => ({
  label,
  grid_class: Array(cells).fill(0),
  grid_confidence: Array(cells).fill(0.9),
  train_accuracy: accuracy,
  test_accuracy: accuracy - 0.05,
})
const leaf = (id: number, counts: number[]) => ({
  id,
  depth: 1,
  feature: null,
  threshold: null,
  left: null,
  right: null,
  samples: 1,
  counts,
  impurity: 0,
})

beforeEach(() => {
  backend = fakeBackend()
  backend.use((method, path) =>
    method === 'POST' && path === '/classical/fit'
      ? Response.json({
          algorithm: 'decision_tree',
          dataset: 'moons',
          classes: 2,
          features: 2,
          train: { x: [0, 1], y: [0, 1], labels: [0, 1] },
          test: { x: [0.5], y: [0.5], labels: [1] },
          bounds: [-1, 2, -1, 2],
          grid: 40,
          frames: [frame('depth 1', 0.8), frame('depth 2', 0.95)],
          fit_ms: 12.3,
          tree: [
            { id: 0, depth: 0, feature: 1, threshold: 0.25, left: 1, right: 2, samples: 2, counts: [1, 1], impurity: 0.5 },
            leaf(1, [1, 0]),
            leaf(2, [0, 1]),
          ],
          feature_importance: [0.3, 0.7],
          notes: [],
        })
      : undefined,
  )
})

afterEach(() => vi.unstubAllGlobals())

describe('Classical ML Lab', () => {
  it('fits, animates the frames and draws the tree', async () => {
    render(<ClassicalLab />)
    await userEvent.click(screen.getByRole('button', { name: 'Fit' }))
    await screen.findByText('Decision tree on Two moons (2D)')
    expect(backend.calls.find((c) => c.path === '/classical/fit')?.body).toMatchObject({
      algorithm: 'decision_tree',
      dataset: 'moons',
      max_depth: 5,
    })
    const slider = screen.getByRole('slider', { name: 'Step through tree depth' })
    await waitFor(() => expect(slider).toHaveValue('1'))
    expect(screen.getByLabelText('Scores')).toHaveTextContent('Train accuracy95.0%')
    const tree = screen.getByRole('region', { name: 'Decision tree' })
    expect(within(tree).getByText('y ≤ 0.25')).toBeInTheDocument()
    expect(within(tree).getByText('class 1')).toBeInTheDocument()
    expect(screen.getByRole('region', { name: 'Feature importance' })).toHaveTextContent('y70.0%')
  })

  it('shows only the parameters and datasets the algorithm uses', async () => {
    render(<ClassicalLab />)
    await userEvent.selectOptions(screen.getByRole('combobox', { name: 'Algorithm' }), 'kmeans')
    expect(screen.getByRole('spinbutton', { name: 'Clusters k' })).toBeInTheDocument()
    expect(screen.queryByRole('spinbutton', { name: 'Max depth' })).not.toBeInTheDocument()
    const datasets = within(screen.getByRole('combobox', { name: 'Dataset' })).getAllByRole('option')
    expect(datasets.map((o) => o.textContent)).not.toContain('Handwritten digits (64 pixels, 10 classes)')
  })
})
