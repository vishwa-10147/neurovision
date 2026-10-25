import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useModelStore } from '../../stores/modelStore'
import { denseGraph } from '../../test/denseGraph'
import type { DenseNetworkConfig } from '../../types/architecture'
import { ArchitectureStudio } from './ArchitectureStudio'

const initialStore = useModelStore.getState()

/** Fake backend for POST /models/educational; records submitted configs. */
function mockBackend() {
  const submitted: DenseNetworkConfig[] = []
  const fetchMock = vi.fn(async (_url: string, init?: RequestInit) => {
    if (_url.endsWith('/projects')) {
      if (init?.method === 'POST') {
        const body = JSON.parse(String(init.body)) as { config: DenseNetworkConfig }
        return Response.json({
          project_id: 'proj_test',
          name: body.config.name,
          description: '',
          schema_version: 1,
          created_at: '2026-01-01T00:00:00Z',
          updated_at: '2026-01-01T00:00:00Z',
          config: body.config,
        }, { status: 201 })
      }
      return Response.json({ projects: [] })
    }
    const config = JSON.parse(String(init?.body)) as DenseNetworkConfig
    submitted.push(config)
    return Response.json({ model_id: `edu_${submitted.length}`, graph: denseGraph(config, `edu_${submitted.length}`) }, { status: 201 })
  })
  vi.stubGlobal('fetch', fetchMock)
  return { submitted, fetchMock }
}

const lastOf = <T,>(items: T[]) => items[items.length - 1]
const inspector = () => within(screen.getByRole('complementary', { name: 'Inspector' }))

beforeEach(() => useModelStore.setState(initialStore, true))
afterEach(() => vi.unstubAllGlobals())

async function renderStudio() {
  const backend = mockBackend()
  const view = render(<ArchitectureStudio />)
  await screen.findByRole('button', { name: 'Hidden Layer 1, 8 units' })
  return { ...backend, ...view }
}

describe('Architecture Studio', () => {
  it('builds the example network on the backend and renders it in 2D', async () => {
    const { submitted, container } = await renderStudio()
    expect(submitted).toHaveLength(1)
    expect(submitted[0].hidden_layers).toEqual([
      { units: 8, activation: 'relu' },
      { units: 6, activation: 'tanh' },
    ])
    // 2 -> 8 -> 6 -> 2: 24 + 54 + 14 parameters; 16 + 48 + 12 connections.
    expect(screen.getByText(/92 trainable parameters/)).toBeInTheDocument()
    expect(container.querySelectorAll('circle[data-unit]')).toHaveLength(18)
    expect(container.querySelectorAll('line[data-connection]')).toHaveLength(76)
    expect(inspector().getByText('Nothing selected')).toBeInTheDocument()
  })

  it('saves the current architecture as a versioned project', async () => {
    await renderStudio()
    await userEvent.click(screen.getByRole('button', { name: 'Save' }))
    expect(await screen.findByRole('status')).toHaveTextContent('Project saved.')
  })

  it('updates the inspector when a unit is clicked', async () => {
    const { container } = await renderStudio()
    await userEvent.click(container.querySelector('[data-unit="hidden_1#2"]') as Element)
    expect(inspector().getByRole('heading', { name: /Hidden Layer 1 · unit 3 of 8/ })).toBeInTheDocument()
    expect(screen.getByRole('complementary', { name: 'Inspector' })).toHaveTextContent(
      'Parameters3 = 2 weights + 1 bias',
    )
    // 2 incoming from the input layer + 6 outgoing to Hidden Layer 2.
    expect(container.querySelectorAll('line[data-connection].is-related')).toHaveLength(2 + 6)

    await userEvent.click(inspector().getByRole('button', { name: 'Next unit' }))
    expect(inspector().getByRole('heading', { name: /unit 4 of 8/ })).toBeInTheDocument()
  })

  it('selects a single connection', async () => {
    const { container } = await renderStudio()
    await userEvent.click(container.querySelector('[data-connection="e_hidden_1_hidden_2:0:5"]') as Element)
    expect(
      inspector().getByRole('heading', { name: 'Hidden Layer 1 · unit 1 → Hidden Layer 2 · unit 6' }),
    ).toBeInTheDocument()
    expect(inspector().getByText(/Weight/)).toBeInTheDocument()
  })

  it('selects a layer with the keyboard and shows its parameter formula', async () => {
    await renderStudio()
    const layer = screen.getByRole('button', { name: 'Hidden Layer 2, 6 units' })
    layer.focus()
    await userEvent.keyboard('{Enter}')
    expect(layer).toHaveAttribute('aria-pressed', 'true')
    expect(inspector().getByText('= 8 × 6 weights + 6 biases')).toBeInTheDocument()
    expect(inspector().getByText('Tanh')).toBeInTheDocument()
  })

  it('clears the selection when the background is clicked', async () => {
    const { container } = await renderStudio()
    await userEvent.click(container.querySelector('[data-unit="input#0"]') as Element)
    await userEvent.click(container.querySelector('svg.network2d-canvas') as Element)
    expect(inspector().getByText('Nothing selected')).toBeInTheDocument()
  })

  it('rejects invalid widths without contacting the backend', async () => {
    const { submitted } = await renderStudio()
    const units = screen.getByLabelText('Hidden layer 1 units')
    await userEvent.clear(units)
    await userEvent.type(units, '99')
    expect(await screen.findByText('A hidden layer can have at most 32 units.')).toBeInTheDocument()
    expect(units).toHaveAttribute('aria-invalid', 'true')
    expect(screen.getByText(/diagram shows the last valid network/)).toBeInTheDocument()
    await new Promise((r) => setTimeout(r, 400))
    expect(submitted.every((c) => c.hidden_layers[0].units <= 32)).toBe(true)
  })

  it('adds and removes hidden layers', async () => {
    const { submitted } = await renderStudio()
    await userEvent.click(screen.getByRole('button', { name: 'Add hidden layer' }))
    await screen.findByRole('button', { name: 'Hidden Layer 3, 4 units' })
    expect(lastOf(submitted).hidden_layers).toHaveLength(3)

    await userEvent.click(screen.getByRole('button', { name: 'Remove hidden layer 1' }))
    await waitFor(() => expect(lastOf(submitted).hidden_layers).toEqual([
      { units: 6, activation: 'tanh' },
      { units: 4, activation: 'relu' },
    ]))
  })

  it('switches to aggregated bands for dense networks', async () => {
    const { container } = await renderStudio()
    useModelStore.getState().updateConfig((c) => ({
      ...c,
      input_units: 16,
      hidden_layers: Array.from({ length: 6 }, () => ({ units: 32, activation: 'relu' as const })),
    }))
    await screen.findByRole('button', { name: 'Hidden Layer 6, 32 units' })
    expect(container.querySelectorAll('line[data-connection]')).toHaveLength(0)
    expect(container.querySelectorAll('polygon[data-edge-id]')).toHaveLength(7)
    expect(screen.getByText(/too many to draw one by one/)).toBeInTheDocument()

    fireEvent.click(container.querySelector('[data-edge-id="e_hidden_1_hidden_2"]') as Element)
    expect(inspector().getByText('1,024')).toBeInTheDocument()
  })

  it('keeps the selection when switching between 2D and 3D', async () => {
    const { container } = await renderStudio()
    await userEvent.click(container.querySelector('[data-unit="hidden_2#4"]') as Element)
    const heading = /Hidden Layer 2 · unit 5 of 6/

    // jsdom has no WebGL, so the 3D view explains why and offers a way back.
    await userEvent.click(screen.getByRole('button', { name: '3D' }))
    expect(screen.getByRole('button', { name: '3D' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('alert')).toHaveTextContent('The 3D view needs WebGL')
    expect(useModelStore.getState().selection).toEqual({ kind: 'unit', nodeId: 'hidden_2', index: 4 })
    expect(inspector().getByRole('heading', { name: heading })).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: 'Switch to 2D view' }))
    expect(container.querySelector('[data-unit="hidden_2#4"]')).toHaveClass('is-selected')
    expect(inspector().getByRole('heading', { name: heading })).toBeInTheDocument()
  })

  it('shows backend validation errors with field details', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        Response.json(
          {
            error: {
              code: 'validation_error',
              message: 'The request did not pass validation.',
              details: [{ field: 'body.hidden_layers.0.units', message: 'Input should be less than or equal to 32' }],
            },
          },
          { status: 422 },
        ),
      ),
    )
    render(<ArchitectureStudio />)
    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent('The request did not pass validation.')
    expect(alert).toHaveTextContent('body.hidden_layers.0.units')
    expect(within(alert).getByRole('button', { name: /Try again/ })).toBeInTheDocument()
  })
})
