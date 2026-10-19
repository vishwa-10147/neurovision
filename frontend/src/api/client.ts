import { z } from 'zod'

export const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

export class ApiError extends Error {
  readonly status: number | null

  constructor(message: string, status: number | null = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

export const REQUEST_TIMEOUT_MS = 5_000

/** GET a JSON resource under /api/v1 and validate it against `schema`. */
export async function getJson<T>(
  path: string,
  schema: z.ZodType<T>,
  signal?: AbortSignal,
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
    response = await fetch(`${API_BASE_URL}/api/v1${path}`, { signal: controller.signal })
  } catch (error) {
    if (signal?.aborted) throw error
    if (timedOut) {
      throw new ApiError(`Backend at ${API_BASE_URL} did not respond within 5 seconds.`)
    }
    throw new ApiError(`Cannot reach the backend at ${API_BASE_URL}.`)
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', onAbort)
  }
  if (!response.ok) {
    throw new ApiError(`Backend responded with HTTP ${response.status}.`, response.status)
  }
  const parsed = schema.safeParse(await response.json().catch(() => null))
  if (!parsed.success) {
    throw new ApiError('Backend returned an unexpected response shape.', response.status)
  }
  return parsed.data
}
