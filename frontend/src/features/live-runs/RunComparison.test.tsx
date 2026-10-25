import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { cloneElement, type ReactElement } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useLiveRunStore } from '../../stores/liveRunStore'
import { FakeSocket, fakeBackend, stepEvent, type FakeBackend } from '../../test/fakeBackend'
import { LiveRuns } from './LiveRuns'

// jsdom has no layout, so give the charts a fixed size and let Recharts draw
// their lines and legends.
vi.mock('recharts', async (importOriginal) => {
  const actual = await importOriginal<typeof import('recharts')>()
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: ReactElement<{ width?: number; height?: number }> }) =>
      cloneElement(children, { width: 400, height: 220 }),
  }
})

const initial = useLiveRunStore.getState()
let backend: FakeBackend

function runInfo(runId: string, name: string, status = 'completed') {
  return {
    run_id: runId,
    name,
    model_id: null,
    status,
    total_epochs: 3,
    epoch: 3,
    step: 30,
    steps_logged: 3,
    created_at: 1_700_000_000,
    last_event_at: 1_700_000_000,
  }
}

const completed = (runId: string) => ({
  schema_version: '1.0',
  event_type: 'training.completed',
  job_id: runId,
  epoch: 3,
  step: 30,
  metrics: {},
})

beforeEach(() => {
  useLiveRunStore.setState(initial, true)
  FakeSocket.install()
  backend = fakeBackend()
  backend.use((method, path) =>
    method === 'GET' && path === '/runs'
      ? Response.json({
          runs: [runInfo('run_c', 'Third'), runInfo('run_b', 'spiral'), runInfo('run_a', 'spiral')],
        })
      : undefined,
  )
})

afterEach(() => vi.unstubAllGlobals())

describe('Run comparison', () => {
  it('overlays the metric curves of the selected runs by step', async () => {
    render(<LiveRuns />)
    const list = await screen.findByRole('list', { name: 'External runs' })
    const boxes = within(list).getAllByRole('checkbox', { name: 'Compare spiral' })

    await userEvent.click(boxes[0])
    expect(await screen.findByText(/Select at least one more run/)).toBeInTheDocument()
    expect(FakeSocket.instances).toHaveLength(0)

    await userEvent.click(boxes[1])
    const panel = await screen.findByRole('region', { name: 'Comparing 2 runs' })
    await waitFor(() => expect(FakeSocket.instances).toHaveLength(2))
    const sockets = Object.fromEntries(FakeSocket.instances.map((s) => [s.url.split('/').at(-2), s]))
    expect(Object.keys(sockets).sort()).toEqual(['run_a', 'run_b'])

    act(() => {
      for (const [runId, scale] of [
        ['run_b', 1],
        ['run_a', 2],
      ] as const) {
        const socket = sockets[runId]
        for (const epoch of [1, 2, 3]) {
          socket.emit(stepEvent(runId, epoch, { metrics: { loss: scale / epoch, val_loss: (scale + 0.5) / epoch } }))
        }
        socket.emit(completed(runId))
      }
    })

    // Same-named runs are told apart by their IDs in the key above the charts.
    const key = within(panel).getByRole('list', { name: 'Compared runs' })
    await waitFor(() =>
      expect(within(key).getAllByRole('listitem').map((item) => item.textContent)).toEqual([
        'spiral (run_b)3 steps · CSV',
        'spiral (run_a)3 steps · CSV',
      ]),
    )
    const chart = await within(panel).findByRole('figure', { name: 'loss by step' })
    // One color per run; validation curves are dashed.
    await waitFor(() => expect(chart.querySelectorAll('path.recharts-line-curve')).toHaveLength(4))
    const curves = [...chart.querySelectorAll('path.recharts-line-curve')]
    expect(curves.map((c) => c.getAttribute('stroke'))).toEqual([
      'var(--chart-train)',
      'var(--chart-train)',
      'var(--chart-val)',
      'var(--chart-val)',
    ])
    expect(curves.map((c) => c.getAttribute('stroke-dasharray'))).toEqual([null, '5 4', null, '5 4'])

    const csv = within(panel).getByRole('link', { name: 'Download metrics of spiral (run_a) (CSV)' })
    expect(csv).toHaveAttribute('href', expect.stringMatching(/\/api\/v1\/runs\/run_a\/metrics\.csv$/))

    // Deselecting a run ends the comparison and closes its stream.
    await userEvent.click(boxes[0])
    expect(screen.queryByRole('region', { name: /Comparing/ })).not.toBeInTheDocument()
    expect(FakeSocket.instances.every((s) => s.closed)).toBe(true)
  })
})
