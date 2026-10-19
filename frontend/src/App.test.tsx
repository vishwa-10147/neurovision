import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from './App'

function mockFetch(impl: () => Promise<Response>) {
  vi.stubGlobal('fetch', vi.fn(impl))
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('App shell', () => {
  it('shows the backend as online when health check succeeds', async () => {
    mockFetch(async () =>
      Response.json({
        status: 'ok',
        service: 'neurovision-backend',
        version: '0.1.0',
        environment: 'development',
      }),
    )
    render(<App />)
    expect(await screen.findByText('Backend online')).toBeInTheDocument()
    expect(screen.getByText('neurovision-backend')).toBeInTheDocument()
  })

  it('shows an actionable error when the backend is unreachable', async () => {
    mockFetch(async () => {
      throw new TypeError('Failed to fetch')
    })
    render(<App />)
    expect(await screen.findByText('Backend offline')).toBeInTheDocument()
    expect(screen.getByRole('alert')).toHaveTextContent(/Cannot reach the backend/)
  })

  it('rejects a malformed health response', async () => {
    mockFetch(async () => Response.json({ status: 'broken' }))
    render(<App />)
    await waitFor(() =>
      expect(screen.getByRole('alert')).toHaveTextContent(/unexpected response shape/),
    )
  })

  it('reports a timeout when the backend never answers', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    try {
      mockFetch(
        () =>
          new Promise<Response>((_, reject) => {
            const signal = vi.mocked(fetch).mock.calls[0][1]?.signal
            signal?.addEventListener('abort', () => reject(signal.reason))
          }),
      )
      render(<App />)
      await vi.advanceTimersByTimeAsync(5_100)
      expect(await screen.findByRole('alert')).toHaveTextContent(/did not respond/)
    } finally {
      vi.useRealTimers()
    }
  })

  it('marks unbuilt sections as planned', async () => {
    mockFetch(async () => Response.json({}))
    render(<App />)
    await userEvent.click(screen.getByRole('button', { name: /Model Explorer/ }))
    expect(screen.getByRole('heading', { name: 'Model Explorer' })).toBeInTheDocument()
    expect(screen.getByText(/Not built yet/)).toBeInTheDocument()
  })
})
