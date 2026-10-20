import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useLiveRunStore } from '../../stores/liveRunStore'
import { FakeSocket, fakeBackend, stepEvent, type FakeBackend } from '../../test/fakeBackend'
import { LiveRuns } from './LiveRuns'

const initial = useLiveRunStore.getState()
let backend: FakeBackend

function runInfo(runId: string, name: string, status = 'running', epoch = 0) {
  return {
    run_id: runId,
    name,
    model_id: null,
    status,
    total_epochs: 4,
    epoch,
    step: epoch,
    steps_logged: epoch,
    created_at: 1_700_000_000,
    last_event_at: 1_700_000_000,
  }
}

const started = (runId: string) => ({
  schema_version: '1.0',
  event_type: 'training.started',
  job_id: runId,
  model_id: runId,
  total_epochs: 4,
})

beforeEach(() => {
  useLiveRunStore.setState(initial, true)
  FakeSocket.install()
  backend = fakeBackend()
})

afterEach(() => vi.unstubAllGlobals())

function serveRuns(runs: object[]) {
  backend.use((method, path) => (method === 'GET' && path === '/runs' ? Response.json({ runs }) : undefined))
}

async function watchSpiral() {
  serveRuns([runInfo('run_a', 'Spiral MLP')])
  render(<LiveRuns />)
  await userEvent.click(await screen.findByRole('button', { name: /Spiral MLP/ }))
  await waitFor(() => expect(FakeSocket.instances).toHaveLength(1))
}

describe('Live runs', () => {
  it('explains how to start a run when there are none', async () => {
    serveRuns([])
    render(<LiveRuns />)
    expect(await screen.findByText('No runs yet')).toBeInTheDocument()
    expect(screen.getByText('Stream a training script')).toBeInTheDocument()
    expect(screen.getByText(/import neurovision_client as nv/)).toBeInTheDocument()
    expect(FakeSocket.instances).toHaveLength(0)
  })

  it('attaches to an external run and shows its streamed metrics and activations', async () => {
    serveRuns([runInfo('run_b', 'Second run', 'completed', 4), runInfo('run_a', 'Spiral MLP')])
    render(<LiveRuns />)
    const list = await screen.findByRole('list', { name: 'External runs' })
    expect(within(list).getAllByRole('button').map((b) => b.textContent)).toEqual([
      expect.stringContaining('Second run'),
      expect.stringContaining('Spiral MLP'),
    ])

    await userEvent.click(within(list).getByRole('button', { name: /Spiral MLP/ }))
    expect(within(list).getByRole('button', { name: /Spiral MLP/ })).toHaveAttribute('aria-pressed', 'true')
    await waitFor(() => expect(FakeSocket.instances).toHaveLength(1))
    const socket = FakeSocket.latest()
    // External runs share the training event stream.
    expect(socket.url).toMatch(/^ws:\/\/.+\/api\/v1\/training\/run_a\/events$/)

    act(() => {
      socket.emit(started('run_a'))
      socket.emit({ ...stepEvent('run_a', 1), metrics: { loss: 0.9, 'train.top1': 0.4 } })
      socket.emit({
        ...stepEvent('run_a', 2),
        metrics: { loss: 0.5, 'train.top1': 0.7 },
        selected_activations: { 'encoder.relu1': [0.5, -0.25, 1] },
      })
    })

    expect(await screen.findByText('Epoch 2 of 4 · step 20')).toBeInTheDocument()
    const tiles = screen.getByLabelText('Latest metrics')
    expect(within(tiles).getByText('train.top1')).toBeInTheDocument()
    expect(within(tiles).getByText('0.5000')).toBeInTheDocument()
    expect(screen.getByRole('progressbar', { name: 'Run progress' })).toHaveAttribute('aria-valuenow', '2')
    expect(screen.getByRole('list', { name: 'encoder.relu1 activations' })).toBeInTheDocument()
    expect(screen.getByRole('listitem', { name: 'Unit 2: -0.250' })).toBeInTheDocument()
    expect(await screen.findByRole('figure', { name: 'loss by step' })).toBeInTheDocument()

    act(() => {
      socket.emit({ schema_version: '1.0', event_type: 'training.completed', job_id: 'run_a', epoch: 2, step: 20, metrics: {} })
    })
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Completed'))
  })

  it('does not double-count steps replayed after a reconnect', async () => {
    await watchSpiral()
    act(() => FakeSocket.latest().emit(stepEvent('run_a', 1)))
    act(() => FakeSocket.latest().drop())
    await waitFor(() => expect(FakeSocket.instances).toHaveLength(2), { timeout: 2000 })
    act(() => {
      FakeSocket.latest().emit(stepEvent('run_a', 1))
      FakeSocket.latest().emit(stepEvent('run_a', 2))
    })
    await waitFor(() => expect(useLiveRunStore.getState().points.map((p) => p.step)).toEqual([10, 20]))
  })

  it('shows why a run failed', async () => {
    await watchSpiral()
    act(() => {
      FakeSocket.latest().emit({
        schema_version: '1.0',
        event_type: 'training.failed',
        job_id: 'run_a',
        message: 'No events arrived for 5 minutes, so this run was closed.',
      })
    })
    expect(await screen.findByRole('alert')).toHaveTextContent('No events arrived for 5 minutes')
    expect(screen.getByRole('status')).toHaveTextContent('Failed')
  })

  it('ignores malformed and oversized events', async () => {
    await watchSpiral()
    act(() => {
      FakeSocket.latest().emit({
        ...stepEvent('run_a', 1),
        selected_activations: { big: Array.from({ length: 257 }, () => 0) },
      })
      FakeSocket.latest().emit({ ...stepEvent('run_a', 2), extra: 'field' })
    })
    await new Promise((resolve) => setTimeout(resolve, 80))
    expect(useLiveRunStore.getState().points).toEqual([])
  })

  it('recovers from backend errors and keeps refreshing the list', async () => {
    let fail = true
    backend.use((method, path) => {
      if (method !== 'GET' || path !== '/runs') return undefined
      if (fail) {
        return Response.json({ error: { code: 'internal_error', message: 'Server down.', details: [] } }, { status: 500 })
      }
      return Response.json({ runs: [runInfo('run_a', 'Spiral MLP')] })
    })
    render(<LiveRuns />)
    expect(await screen.findByRole('alert')).toHaveTextContent('Server down.')
    fail = false
    await userEvent.click(screen.getByRole('button', { name: 'Try again' }))
    expect(await screen.findByRole('button', { name: /Spiral MLP/ })).toBeInTheDocument()
    const before = backend.calls.filter((c) => c.path === '/runs').length
    await waitFor(() => expect(backend.calls.filter((c) => c.path === '/runs').length).toBeGreaterThan(before), {
      timeout: 4000,
    })
  })
})
