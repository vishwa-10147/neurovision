import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { fakeBackend, type FakeBackend } from '../../test/fakeBackend'
import { NeuralLab } from './NeuralLab'

let backend: FakeBackend
const job = {
  job_id: 'lab_1234',
  template: 'rnn',
  run_name: 'Recurrent network forecasting a series · 1234',
  status: 'running',
  exit_code: null,
  started_at: 1,
  output: [],
}

beforeEach(() => {
  backend = fakeBackend()
  backend.use((method, path) => {
    if (method === 'GET' && path === '/lab/templates') {
      return Response.json({
        enabled: true,
        python: 'python',
        templates: [
          {
            id: 'rnn',
            label: 'Recurrent network forecasting a series',
            family: 'Recurrent',
            description: 'An LSTM, GRU or plain RNN.',
            shows: ['hidden state at every time step'],
            params: [
              { name: 'cell', label: 'Cell', kind: 'choice', default: 'lstm', choices: ['lstm', 'gru', 'rnn'] },
              { name: 'hidden', label: 'Hidden units', kind: 'int', default: 32, min: 2, max: 256 },
            ],
          },
        ],
      })
    }
    if (method === 'GET' && path === '/lab/runs') return Response.json({ jobs: [] })
    if (method === 'POST' && path === '/lab/runs') return Response.json(job, { status: 201 })
    return undefined
  })
})

afterEach(() => vi.unstubAllGlobals())

describe('Neural Lab', () => {
  it('launches a configured model and opens it in Live runs', async () => {
    const onWatch = vi.fn()
    render(<NeuralLab onWatch={onWatch} />)
    await userEvent.click(await screen.findByRole('button', { name: /Recurrent network/ }))
    await userEvent.selectOptions(screen.getByRole('combobox', { name: 'Cell' }), 'gru')
    const hidden = screen.getByRole('spinbutton', { name: 'Hidden units' })
    await userEvent.clear(hidden)
    await userEvent.type(hidden, '48')
    await userEvent.click(screen.getByRole('button', { name: 'Train live' }))
    expect(backend.calls.find((c) => c.method === 'POST' && c.path === '/lab/runs')?.body).toEqual({
      template: 'rnn',
      params: { cell: 'gru', hidden: 48 },
    })
    const runs = screen.getByRole('region', { name: 'Lab runs' })
    expect(await within(runs).findByText(job.run_name)).toBeInTheDocument()
    await userEvent.click(within(runs).getByRole('button', { name: /Watch in Live runs/ }))
    expect(onWatch).toHaveBeenCalledWith(job.run_name)
  })

  it('explains when the lab is turned off', async () => {
    backend.use((method, path) =>
      method === 'GET' && path === '/lab/templates' ? Response.json({ enabled: false, python: null, templates: [] }) : undefined,
    )
    render(<NeuralLab onWatch={() => {}} />)
    expect(await screen.findByText('The Neural Lab is turned off on this server')).toBeInTheDocument()
  })
})
