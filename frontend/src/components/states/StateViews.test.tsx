import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ApiError } from '../../api/client'
import { ErrorBoundary } from './ErrorBoundary'
import { EmptyState, ErrorState, LoadingState } from './StateViews'

describe('state views', () => {
  it('announces loading', () => {
    render(<LoadingState label="Loading graph…" />)
    expect(screen.getByRole('status')).toHaveTextContent('Loading graph…')
  })

  it('renders an empty state with guidance', () => {
    render(<EmptyState title="No models yet">Import an ONNX file to begin.</EmptyState>)
    expect(screen.getByText('No models yet')).toBeInTheDocument()
    expect(screen.getByText('Import an ONNX file to begin.')).toBeInTheDocument()
  })

  it('shows API error details, hint, code, and retries', async () => {
    const onRetry = vi.fn()
    const error = new ApiError('The request did not pass validation.', {
      kind: 'http',
      status: 422,
      code: 'validation_error',
      details: [{ field: 'body.width', message: 'Input should be less than or equal to 64' }],
      hint: 'Correct the highlighted fields and try again.',
    })
    render(<ErrorState error={error} onRetry={onRetry} />)

    const alert = screen.getByRole('alert')
    expect(alert).toHaveTextContent('The request did not pass validation.')
    expect(alert).toHaveTextContent('body.width Input should be less than or equal to 64')
    expect(alert).toHaveTextContent('Correct the highlighted fields')
    expect(alert).toHaveTextContent('Error code: validation_error')

    await userEvent.click(screen.getByRole('button', { name: /Try again/ }))
    expect(onRetry).toHaveBeenCalledOnce()
  })

  it('contains a crashing child in an error boundary', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {})
    let crash = true
    function Fragile() {
      if (crash) throw new Error('boom')
      return <p>recovered</p>
    }
    render(
      <ErrorBoundary fallback={(e, reset) => <button onClick={reset}>{(e as Error).message}</button>}>
        <Fragile />
      </ErrorBoundary>,
    )
    crash = false
    await userEvent.click(screen.getByRole('button', { name: 'boom' }))
    expect(screen.getByText('recovered')).toBeInTheDocument()
    vi.restoreAllMocks()
  })

  it('handles non-API errors without crashing', () => {
    render(<ErrorState error="weird" />)
    expect(screen.getByRole('alert')).toHaveTextContent('Something went wrong')
    expect(screen.queryByRole('button')).not.toBeInTheDocument()
  })
})
