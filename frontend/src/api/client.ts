import { z } from 'zod'

export const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

export const REQUEST_TIMEOUT_MS = 5_000

/** Mirror of shared/schemas/error.schema.json. */
export const ErrorResponseSchema = z.object({
  error: z.object({
    code: z.string(),
    message: z.string(),
    details: z
      .array(z.object({ field: z.string().nullable().default(null), message: z.string() }))
      .default([]),
  }),
})

export type ApiErrorDetail = { field: string | null; message: string }

export type ApiErrorKind = 'network' | 'timeout' | 'http' | 'invalid_response'

export class ApiError extends Error {
  readonly kind: ApiErrorKind
  readonly status: number | null
  /** Backend error code such as "validation_error", when provided. */
  readonly code: string | null
  readonly details: ApiErrorDetail[]
  /** What the user can do about it. */
  readonly hint: string | null

  constructor(
    message: string,
    options: {
      kind: ApiErrorKind
      status?: number | null
      code?: string | null
      details?: ApiErrorDetail[]
      hint?: string | null
    },
  ) {
    super(message)
    this.name = 'ApiError'
    this.kind = options.kind
    this.status = options.status ?? null
    this.code = options.code ?? null
    this.details = options.details ?? []
    this.hint = options.hint ?? null
  }
}

const START_BACKEND_HINT =
  'Start the backend from backend/ with: uvicorn app.main:app --reload --port 8000 ' +
  '(or set VITE_API_BASE_URL if it runs elsewhere).'

interface RequestOptions {
  method?: 'GET' | 'POST'
  body?: unknown
  signal?: AbortSignal
}

/** Call an /api/v1 endpoint and validate the JSON response against `schema`. */
export async function request<T>(
  path: string,
  schema: z.ZodType<T>,
  { method = 'GET', body, signal }: RequestOptions = {},
): Promise<T> {
  const controller = new AbortController()
  let timedOut = false
  const timer = setTimeout(() => {
    timedOut = true
    controller.abort()
  }, REQUEST_TIMEOUT_MS)
  const onAbort = () => controller.abort()
  signal?.addEventListener('abort', onAbort)

  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}/api/v1${path}`, {
      method,
      signal: controller.signal,
      headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    })
  } catch (error) {
    if (signal?.aborted) throw error
    if (timedOut) {
      throw new ApiError(`Backend at ${API_BASE_URL} did not respond within 5 seconds.`, {
        kind: 'timeout',
        hint: 'The backend may be busy or stuck. Check its terminal output, then retry.',
      })
    }
    throw new ApiError(`Cannot reach the backend at ${API_BASE_URL}.`, {
      kind: 'network',
      hint: START_BACKEND_HINT,
    })
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', onAbort)
  }

  const payload: unknown = await response.json().catch(() => undefined)

  if (!response.ok) {
    const envelope = ErrorResponseSchema.safeParse(payload)
    if (envelope.success) {
      const { code, message, details } = envelope.data.error
      throw new ApiError(message, {
        kind: 'http',
        status: response.status,
        code,
        details,
        hint: hintForStatus(response.status),
      })
    }
    throw new ApiError(`Backend responded with HTTP ${response.status}.`, {
      kind: 'http',
      status: response.status,
      hint: hintForStatus(response.status),
    })
  }

  const parsed = schema.safeParse(payload)
  if (!parsed.success) {
    throw new ApiError('Backend returned an unexpected response shape.', {
      kind: 'invalid_response',
      status: response.status,
      details: parsed.error.issues.map((issue) => ({
        field: issue.path.length ? issue.path.join('.') : null,
        message: issue.message,
      })),
      hint: 'The frontend and backend versions may not match. Update both and reload.',
    })
  }
  return parsed.data
}

export function getJson<T>(path: string, schema: z.ZodType<T>, signal?: AbortSignal): Promise<T> {
  return request(path, schema, { signal })
}

function hintForStatus(status: number): string | null {
  if (status === 422 || status === 400) return 'Correct the highlighted fields and try again.'
  if (status === 404) return 'The requested item may have been removed. Reload and try again.'
  if (status === 413) return 'The upload or request is larger than the server allows.'
  if (status >= 500) return 'This is a server problem. Check the backend logs, then retry.'
  return null
}
