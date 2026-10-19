import { afterEach, describe, expect, it, vi } from 'vitest'
import { z } from 'zod'
import { ApiError, request } from './client'

const Schema = z.object({ ok: z.boolean() })

function respondWith(response: Response) {
  vi.stubGlobal('fetch', vi.fn(async () => response))
}

async function captureError(promise: Promise<unknown>): Promise<ApiError> {
  const error = await promise.catch((e: unknown) => e)
  expect(error).toBeInstanceOf(ApiError)
  return error as ApiError
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('request', () => {
  it('returns validated data', async () => {
    respondWith(Response.json({ ok: true }))
    await expect(request('/thing', Schema)).resolves.toEqual({ ok: true })
  })

  it('sends JSON bodies for POST', async () => {
    respondWith(Response.json({ ok: true }))
    await request('/thing', Schema, { method: 'POST', body: { width: 4 } })
    const [url, init] = vi.mocked(fetch).mock.calls[0]
    expect(url).toMatch(/\/api\/v1\/thing$/)
    expect(init?.method).toBe('POST')
    expect(init?.body).toBe('{"width":4}')
  })

  it('turns the backend error envelope into an actionable ApiError', async () => {
    respondWith(
      Response.json(
        {
          error: {
            code: 'validation_error',
            message: 'The request did not pass validation.',
            details: [{ field: 'body.width', message: 'Input should be less than or equal to 64' }],
          },
        },
        { status: 422 },
      ),
    )
    const error = await captureError(request('/thing', Schema))
    expect(error.kind).toBe('http')
    expect(error.status).toBe(422)
    expect(error.code).toBe('validation_error')
    expect(error.message).toBe('The request did not pass validation.')
    expect(error.details).toEqual([
      { field: 'body.width', message: 'Input should be less than or equal to 64' },
    ])
    expect(error.hint).toMatch(/Correct the highlighted fields/)
  })

  it('falls back to the status code when the error body is not an envelope', async () => {
    respondWith(new Response('<html>Bad gateway</html>', { status: 502 }))
    const error = await captureError(request('/thing', Schema))
    expect(error.message).toBe('Backend responded with HTTP 502.')
    expect(error.code).toBeNull()
    expect(error.hint).toMatch(/server problem/)
  })

  it('reports which fields of a successful response were malformed', async () => {
    respondWith(Response.json({ ok: 'yes' }))
    const error = await captureError(request('/thing', Schema))
    expect(error.kind).toBe('invalid_response')
    expect(error.details[0].field).toBe('ok')
  })

  it('reports network failures with a way to fix them', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new TypeError('Failed to fetch')
      }),
    )
    const error = await captureError(request('/thing', Schema))
    expect(error.kind).toBe('network')
    expect(error.hint).toMatch(/uvicorn/)
  })
})
