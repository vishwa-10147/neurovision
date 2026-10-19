import { act, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { clearTensorCache } from '../../hooks/useTensor'
import { useModelStore } from '../../stores/modelStore'
import { fakeBackend, tensorResponse, type FakeBackend } from '../../test/fakeBackend'
import { ArchitectureStudio } from './ArchitectureStudio'

const initialModel = useModelStore.getState()
let backend: FakeBackend

// Values a forward pass of the example 2 -> 8 -> 6 -> 2 network might return.
const H1 = [0.9, 0, 0.3, 0, 1.2, 0, 0.05, 0.4]
const forwardResponse = {
  model_id: 'edu_1',
  outputs: [[0.7, -0.2]],
  logits: [[0.7, -0.2]],
  predictions: [0],
  activations: {
    input: { pre: null, post: [[0.5, -1]] },
    hidden_1: { pre: [[0.9, -0.4, 0.3, -1, 1.2, -0.1, 0.05, 0.4]], post: [H1] },
    hidden_2: { pre: [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], post: [[0.1, 0.2, 0.29, 0.38, 0.46, 0.54]] },
    output: { pre: [[0.7, -0.2]], post: [[0.7, -0.2]] },
  },
}
// hidden_2.weight is [6 target units][8 source units]; weight[j][i] = j * 10 + i.
const H2_WEIGHTS = Array.from({ length: 48 }, (_, k) => Math.floor(k / 8) * 10 + (k % 8))

const inspector = () => within(screen.getByRole('complementary', { name: 'Inspector' }))

beforeEach(() => {
  useModelStore.setState(initialModel, true)
  clearTensorCache()
  backend = fakeBackend()
  backend.use((method, path) =>
    method === 'POST' && path.endsWith('/forward') ? Response.json(forwardResponse) : undefined,
  )
  backend.use((_, path) =>
    path.endsWith('/tensors/hidden_2.weight') ? tensorResponse('hidden_2.weight', [6, 8], H2_WEIGHTS) : undefined,
  )
  backend.use((_, path) =>
    path.endsWith('/tensors/hidden_1.bias') ? tensorResponse('hidden_1.bias', [8], [0, 0.11, 0.22, 0, 0, 0, 0, 0]) : undefined,
  )
})

afterEach(() => vi.unstubAllGlobals())

async function renderStudio() {
  const view = render(<ArchitectureStudio />)
  await screen.findByRole('button', { name: 'Hidden Layer 1, 8 units' })
  return view
}

describe('Forward pass in the studio', () => {
  it('runs the inputs through the model and shades units by measured value', async () => {
    const { container } = await renderStudio()
    const x1 = screen.getByLabelText('Input 1')
    await userEvent.clear(x1)
    await userEvent.type(x1, '2')
    const x2 = screen.getByLabelText('Input 2')
    await userEvent.clear(x2)
    await userEvent.type(x2, '-1')
    await userEvent.click(screen.getByRole('button', { name: 'Run' }))

    expect(await screen.findByText(/Predicted class/)).toHaveTextContent('Predicted class 0 for (2.000, -1.000)')
    const forward = backend.calls.find((c) => c.path.endsWith('/forward'))
    expect(forward?.body).toEqual({ inputs: [[2, -1]] })

    const unit = container.querySelector('[data-unit="hidden_1#4"]')
    expect(unit).toHaveAttribute('data-value', '1.2')
    expect(screen.getByText(/value from the last forward pass, scaled within each layer/)).toBeInTheDocument()
  })

  it('shows the measured values and real bias for a selected unit', async () => {
    const { container } = await renderStudio()
    await userEvent.click(screen.getByRole('button', { name: 'Run' }))
    await screen.findByText(/Predicted class/)
    await userEvent.click(container.querySelector('[data-unit="hidden_1#2"]') as Element)

    const panel = screen.getByRole('complementary', { name: 'Inspector' })
    expect(panel).toHaveTextContent('Weighted sum (W·x + b)0.3000')
    expect(panel).toHaveTextContent('Activation value0.3000')
    expect(await inspector().findByText('0.2200')).toBeInTheDocument() // bias of unit 3
  })

  it('shows a real connection weight and explains missing gradients', async () => {
    const { container } = await renderStudio()
    await userEvent.click(container.querySelector('[data-connection="e_hidden_1_hidden_2:3:5"]') as Element)
    // weight[target 5][source 3] = 53
    expect(await inspector().findByText('53.0000')).toBeInTheDocument()
    expect(await inspector().findByText('Train the model to see gradients.')).toBeInTheDocument()
  })

  it('hides forward values once the weights change', async () => {
    const { container } = await renderStudio()
    await userEvent.click(screen.getByRole('button', { name: 'Run' }))
    await screen.findByText(/Predicted class/)
    act(() => useModelStore.getState().markWeightsChanged(useModelStore.getState().modelId as string))

    expect(await screen.findByText(/weights changed since the last run/)).toBeInTheDocument()
    expect(container.querySelector('[data-unit="hidden_1#4"]')).not.toHaveAttribute('data-value')
  })

  it('rejects inputs that are not numbers', async () => {
    await renderStudio()
    await userEvent.clear(screen.getByLabelText('Input 1'))
    expect(screen.getByText(/Each input must be a number/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Run' })).toBeDisabled()
  })

  it('keeps the existing model when the studio is opened again', async () => {
    const first = await renderStudio()
    first.unmount()
    await renderStudio()
    await new Promise((r) => setTimeout(r, 400))
    expect(backend.calls.filter((c) => c.path === '/models/educational')).toHaveLength(1)
  })
})
