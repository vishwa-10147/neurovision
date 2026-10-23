import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useLiveRunStore } from '../../stores/liveRunStore'
import { FakeSocket, fakeBackend, stepEvent, type FakeBackend } from '../../test/fakeBackend'
import { LiveRuns } from './LiveRuns'

const initial = useLiveRunStore.getState()
let backend: FakeBackend
let nextCommand = 1

const ALL = ['pause', 'resume', 'stop', 'checkpoint', 'set_lr']

function runInfo(controls: string[] = ALL) {
  return {
    run_id: 'run_a',
    name: 'ResNet',
    model_id: null,
    status: 'running',
    total_epochs: 10,
    epoch: 1,
    step: 1,
    steps_logged: 1,
    created_at: 1_700_000_000,
    last_event_at: 1_700_000_000,
    controls,
    paused: false,
    learning_rate: 0.01,
    pending_commands: 0,
    alerts: 0,
  }
}

function command(kind: string, status: string, extra: object = {}) {
  return {
    command_id: 1,
    kind,
    value: null,
    status,
    message: null,
    created_at: 1_700_000_001,
    resolved_at: null,
    ...extra,
  }
}

const commandEvent = (cmd: object, paused = false, learning_rate: number | null = 0.01) => ({
  schema_version: '1.0',
  event_type: 'run.command',
  job_id: 'run_a',
  command: cmd,
  paused,
  learning_rate,
})

beforeEach(() => {
  useLiveRunStore.setState(initial, true)
  FakeSocket.install()
  backend = fakeBackend()
  nextCommand = 1
})

afterEach(() => vi.unstubAllGlobals())

async function watch(controls: string[] = ALL) {
  backend.use((method, path, body) => {
    if (method === 'GET' && path === '/runs') return Response.json({ runs: [runInfo(controls)] })
    if (method === 'POST' && path === '/runs/run_a/commands') {
      const { kind, value = null } = body as { kind: string; value?: number }
      return Response.json(command(kind, 'pending', { command_id: nextCommand++, value }), { status: 202 })
    }
    return undefined
  })
  render(<LiveRuns />)
  await userEvent.click(await screen.findByRole('button', { name: /ResNet/ }))
  await waitFor(() => expect(FakeSocket.instances).toHaveLength(1))
}

describe('Run controls', () => {
  it('explains how to enable controls when the script did not opt in', async () => {
    await watch([])
    expect(screen.getByText(/cannot be controlled from here/)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /Pause/ })).not.toBeInTheDocument()
  })

  it('shows only the commands the script accepts', async () => {
    await watch(['pause', 'resume', 'stop'])
    expect(screen.getByRole('button', { name: /Pause/ })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /Save checkpoint/ })).not.toBeInTheDocument()
    expect(screen.queryByLabelText('Learning rate')).not.toBeInTheDocument()
  })

  it('sends a pause and shows what the script reported', async () => {
    await watch()
    await userEvent.click(screen.getByRole('button', { name: /Pause/ }))
    expect(backend.calls.find((c) => c.path === '/runs/run_a/commands')?.body).toEqual({ kind: 'pause' })
    const list = await screen.findByRole('list', { name: 'Recent commands' })
    expect(within(list).getByText('Waiting for the script…')).toBeInTheDocument()

    act(() => {
      FakeSocket.latest().emit(commandEvent(command('pause', 'applied', { message: 'Paused after step 1.' }), true))
    })
    expect(await within(list).findByText('Applied')).toBeInTheDocument()
    expect(within(list).getByText('Paused after step 1.')).toBeInTheDocument()
    // Paused: the badge says so and Resume replaces Pause.
    expect(screen.getByRole('status')).toHaveTextContent('Paused')
    expect(screen.getByRole('button', { name: /Resume/ })).toBeInTheDocument()
  })

  it('asks for confirmation before stopping', async () => {
    await watch()
    await userEvent.click(screen.getByRole('button', { name: /^Stop$/ }))
    expect(backend.calls.some((c) => c.path === '/runs/run_a/commands')).toBe(false)
    await userEvent.click(screen.getByRole('button', { name: /Keep running/ }))
    await userEvent.click(screen.getByRole('button', { name: /^Stop$/ }))
    await userEvent.click(screen.getByRole('button', { name: /Confirm stop/ }))
    expect(backend.calls.find((c) => c.path === '/runs/run_a/commands')?.body).toEqual({ kind: 'stop' })
  })

  it('validates and sends a new learning rate', async () => {
    await watch()
    const input = screen.getByLabelText('Learning rate')
    const set = screen.getByRole('button', { name: 'Set' })
    await userEvent.type(input, '-1')
    expect(set).toBeDisabled()
    await userEvent.clear(input)
    await userEvent.type(input, '0.001')
    await userEvent.click(set)
    expect(backend.calls.find((c) => c.path === '/runs/run_a/commands')?.body).toEqual({ kind: 'set_lr', value: 0.001 })
    act(() => {
      FakeSocket.latest().emit(
        commandEvent(command('set_lr', 'applied', { value: 0.001, message: 'Learning rate set to 0.001.' }), false, 0.001),
      )
    })
    expect(await screen.findByText(/Now 0\.001000/)).toBeInTheDocument()
  })

  it('reports a rejected command with the script’s reason', async () => {
    await watch()
    act(() => {
      FakeSocket.latest().emit(commandEvent(command('checkpoint', 'rejected', { message: 'on_checkpoint raised OSError.' })))
    })
    const list = await screen.findByRole('list', { name: 'Recent commands' })
    expect(within(list).getByText('Rejected')).toBeInTheDocument()
    expect(within(list).getByText('on_checkpoint raised OSError.')).toBeInTheDocument()
  })

  it('shows a stopped run as stopped and disables the controls', async () => {
    await watch()
    act(() => {
      FakeSocket.latest().emit({ schema_version: '1.0', event_type: 'training.cancelled', job_id: 'run_a', step: 5 })
    })
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Stopped'))
    expect(screen.getByRole('button', { name: /Pause/ })).toBeDisabled()
    expect(screen.getByText(/asked to from NeuroVision/)).toBeInTheDocument()
  })
})

describe('Run health', () => {
  const health = {
    grad_norm: 0.84,
    learning_rate: 0.01,
    nonfinite_grads: 0,
    nonfinite_metrics: [],
    parameters: {
      fc1: { grad_norm: 0.61, weight_norm: 4.2, update_ratio: 0.2, parameters: 96 },
      fc2: { grad_norm: 0.58, weight_norm: 5.9, update_ratio: 0.001, parameters: 1056 },
    },
    activations: {
      act1: { kind: 'relu', mean: 0.4, std: 0.6, dead_fraction: 0.75, saturated_fraction: null },
    },
  }

  it('shows the latest layer statistics and flags bad values', async () => {
    await watch()
    act(() => {
      FakeSocket.latest().emit(stepEvent('run_a', 1, { health }))
      FakeSocket.latest().emit(stepEvent('run_a', 2, { health: { ...health, grad_norm: 1.2 } }))
    })
    const panel = await screen.findByRole('region', { name: 'Layer health' })
    expect(within(panel).getByText(/step 20 · total gradient norm 1.200/)).toBeInTheDocument()
    const fc1 = within(panel).getByRole('row', { name: /fc1/ })
    expect(within(fc1).getByText('2.0e-1')).toHaveClass('value-bad')
    expect(within(within(panel).getByRole('row', { name: /fc2/ })).getByText('1.0e-3')).toHaveClass('value-ok')
    expect(within(within(panel).getByRole('row', { name: /act1/ })).getByText('75%')).toHaveClass('value-bad')
    // Gradient norm and learning rate trend charts.
    expect(await screen.findByRole('figure', { name: 'gradient norm by step' })).toBeInTheDocument()
    expect(screen.getByRole('figure', { name: 'learning rate by step' })).toBeInTheDocument()
  })

  it('lists alerts newest first with their advice, once each', async () => {
    await watch()
    expect(screen.getByText(/No problems detected so far/)).toBeInTheDocument()
    const alert = (id: number, kind: string, severity: string, message: string, layer: string | null = null) => ({
      schema_version: '1.0',
      event_type: 'run.alert',
      job_id: 'run_a',
      alert: { alert_id: id, kind, severity, step: id * 10, layer, message, advice: `Advice ${id}` },
    })
    act(() => {
      FakeSocket.latest().emit(alert(1, 'dead_units', 'warning', '75% of units in act1 are dead.', 'act1'))
      FakeSocket.latest().emit(alert(2, 'non_finite_loss', 'critical', 'loss became NaN or infinite.'))
      FakeSocket.latest().emit(alert(2, 'non_finite_loss', 'critical', 'loss became NaN or infinite.'))
    })
    const panel = await screen.findByRole('region', { name: 'Health alerts' })
    const items = await within(panel).findAllByRole('listitem')
    expect(items).toHaveLength(2)
    expect(items[0]).toHaveTextContent('Critical: loss became NaN or infinite.')
    expect(items[0]).toHaveTextContent('Advice 2')
    expect(items[1]).toHaveTextContent('Step 10 · act1')
  })
})
