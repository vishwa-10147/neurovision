import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useModelStore } from '../../stores/modelStore'
import { useTrainingStore } from '../../stores/trainingStore'
import { FakeSocket, fakeBackend, stepEvent, type FakeBackend } from '../../test/fakeBackend'
import { DEFAULT_TRAINING_CONFIG } from '../../types/training'
import { TrainingLab } from './TrainingLab'

const initialModel = useModelStore.getState()
const initialTraining = useTrainingStore.getState()

let backend: FakeBackend

function jobInfo(modelId: string, status = 'running', total = 3) {
  return { job_id: 'job_1', model_id: modelId, status, epoch: 0, total_epochs: total, config: DEFAULT_TRAINING_CONFIG }
}

beforeEach(() => {
  useModelStore.setState(initialModel, true)
  useTrainingStore.setState(initialTraining, true)
  FakeSocket.install()
  backend = fakeBackend()
  backend.use((method, path) =>
    method === 'POST' && path.endsWith('/training') ? Response.json(jobInfo(path.split('/')[2]), { status: 202 }) : undefined,
  )
  backend.use((method, path) =>
    method === 'POST' && path.endsWith('/cancel') ? Response.json(jobInfo('edu_1', 'running'), { status: 202 }) : undefined,
  )
})

afterEach(() => vi.unstubAllGlobals())

async function renderLab() {
  const onOpenStudio = vi.fn()
  render(<TrainingLab onOpenStudio={onOpenStudio} />)
  await screen.findByRole('button', { name: 'Start training' })
  return { onOpenStudio }
}

describe('Training Lab', () => {
  it('builds the model from the architecture and shows an empty state', async () => {
    await renderLab()
    expect(backend.calls.some((c) => c.method === 'POST' && c.path === '/models/educational')).toBe(true)
    expect(screen.getByText('No training yet')).toBeInTheDocument()
    expect(screen.getByText(/Two-moons classifier · 4 layers · 92 parameters/)).toBeInTheDocument()
  })

  it('starts a job and streams real metrics and activations', async () => {
    await renderLab()
    await userEvent.click(screen.getByRole('button', { name: 'Start training' }))

    const start = backend.calls.find((c) => c.path.endsWith('/training'))
    expect(start?.body).toMatchObject({
      dataset: 'moons',
      epochs: 100,
      capture_layers: ['hidden_1', 'hidden_2', 'output'],
    })

    await waitFor(() => expect(FakeSocket.instances).toHaveLength(1))
    const socket = FakeSocket.latest()
    expect(socket.url).toMatch(/^ws:\/\/.+\/api\/v1\/training\/job_1\/events$/)
    act(() => {
      socket.emit({ schema_version: '1.0', event_type: 'training.started', job_id: 'job_1', model_id: 'edu_1', total_epochs: 3 })
      socket.emit(stepEvent('job_1', 1))
      socket.emit(stepEvent('job_1', 2, { selected_activations: { hidden_1: [0.5, -0.25, 0, 1, 0, 0, 0, 0] } }))
    })

    expect(await screen.findByText('Epoch 2 of 3')).toBeInTheDocument()
    const tiles = screen.getByLabelText('Latest metrics')
    expect(within(tiles).getByText('0.5000')).toBeInTheDocument() // training loss 1/2
    expect(screen.getByRole('progressbar', { name: 'Training progress' })).toHaveAttribute('aria-valuenow', '2')
    expect(screen.getByRole('list', { name: 'Hidden Layer 1 activations' })).toBeInTheDocument()
    expect(screen.getByRole('listitem', { name: 'Unit 2: -0.250' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Cancel training' })).toBeInTheDocument()

    act(() => {
      socket.emit(stepEvent('job_1', 3))
      socket.emit({ schema_version: '1.0', event_type: 'training.completed', job_id: 'job_1', epoch: 3, step: 30, metrics: {} })
    })
    expect(await screen.findByText('Completed')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Start training' })).toBeInTheDocument()
  })

  it('cancels a running job', async () => {
    await renderLab()
    await userEvent.click(screen.getByRole('button', { name: 'Start training' }))
    await userEvent.click(await screen.findByRole('button', { name: 'Cancel training' }))
    expect(backend.calls.some((c) => c.method === 'POST' && c.path === '/training/job_1/cancel')).toBe(true)
    act(() => {
      FakeSocket.latest().emit({ schema_version: '1.0', event_type: 'training.cancelled', job_id: 'job_1', step: 4 })
    })
    expect(await screen.findByText('Cancelled')).toBeInTheDocument()
  })

  it('reconnects after a dropped connection and does not double-count epochs', async () => {
    await renderLab()
    await userEvent.click(screen.getByRole('button', { name: 'Start training' }))
    await waitFor(() => expect(FakeSocket.instances).toHaveLength(1))
    act(() => FakeSocket.latest().emit(stepEvent('job_1', 1)))
    act(() => FakeSocket.latest().drop())

    await waitFor(() => expect(FakeSocket.instances).toHaveLength(2), { timeout: 2000 })
    act(() => {
      // The server replays history on the new connection.
      FakeSocket.latest().emit(stepEvent('job_1', 1))
      FakeSocket.latest().emit(stepEvent('job_1', 2))
    })
    await waitFor(() => expect(useTrainingStore.getState().history.map((m) => m.epoch)).toEqual([1, 2]))
  })

  it('shows a divergence failure from the server', async () => {
    await renderLab()
    await userEvent.click(screen.getByRole('button', { name: 'Start training' }))
    await waitFor(() => expect(FakeSocket.instances).toHaveLength(1))
    act(() => {
      FakeSocket.latest().emit({
        schema_version: '1.0',
        event_type: 'training.failed',
        job_id: 'job_1',
        message: 'The loss became nan in epoch 3. Lower the learning rate and try again.',
      })
    })
    expect(await screen.findByText(/Lower the learning rate/)).toBeInTheDocument()
    expect(screen.getByText('Failed')).toBeInTheDocument()
  })

  it('validates settings before starting', async () => {
    await renderLab()
    const rate = screen.getByLabelText('Learning rate')
    await userEvent.clear(rate)
    await userEvent.type(rate, '0')
    expect(await screen.findByText('The learning rate must be greater than 0.')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Start training' })).toBeDisabled()
  })

  it('explains datasets that do not fit the output layer', async () => {
    useModelStore.getState().updateConfig((c) => ({ ...c, output_units: 3 }))
    await renderLab()
    expect(
      screen.getByText(/This dataset has 2 classes, but the network has 3 outputs/),
    ).toBeInTheDocument()
    expect(screen.getByRole('option', { name: /Two moons.*needs 1 or 2 outputs/ })).toBeDisabled()
    await userEvent.selectOptions(screen.getByLabelText('Dataset'), 'spiral')
    expect(screen.getByRole('button', { name: 'Start training' })).toBeEnabled()
  })

  it('shows server errors when starting fails', async () => {
    backend.use((method, path) =>
      method === 'POST' && path.endsWith('/training')
        ? Response.json(
            { error: { code: 'too_many_requests', message: 'The server is already running 2 training jobs.', details: [] } },
            { status: 429 },
          )
        : undefined,
    )
    await renderLab()
    await userEvent.click(screen.getByRole('button', { name: 'Start training' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('already running 2 training jobs')
  })
})
