/**
 * Test doubles for the backend: a fetch router and a controllable WebSocket.
 * Real backend behavior is covered by backend/tests; these let UI tests drive
 * specific responses and event sequences.
 */
import { vi } from 'vitest'
import type { DenseNetworkConfig } from '../types/architecture'
import { denseGraph } from './denseGraph'

export type Route = (method: string, path: string, body: unknown) => Response | undefined

export interface FakeBackend {
  calls: { method: string; path: string; body: unknown }[]
  /** Routes checked before the defaults; return undefined to fall through. */
  use: (route: Route) => void
}

const notFound = (message: string) => Response.json({ error: { code: 'not_found', message, details: [] } }, { status: 404 })

/** Default routes: model creation, metrics (none yet), gradients (not trained). */
export function fakeBackend(): FakeBackend {
  const routes: Route[] = []
  const calls: FakeBackend['calls'] = []
  let models = 0
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init?: RequestInit) => {
      const path = new URL(url).pathname.replace('/api/v1', '')
      const method = init?.method ?? 'GET'
      const body = init?.body ? JSON.parse(String(init.body)) : undefined
      calls.push({ method, path, body })
      for (const route of routes) {
        const response = route(method, path, body)
        if (response) return response
      }
      if (method === 'POST' && path === '/models/educational') {
        models += 1
        const id = `edu_${models}`
        return Response.json({ model_id: id, graph: denseGraph(body as DenseNetworkConfig, id) }, { status: 201 })
      }
      if (path.endsWith('/metrics')) {
        return Response.json({ model_id: path.split('/')[2], job: null, history: [] })
      }
      if (path.endsWith('.grad')) return notFound('No gradients yet. Train the model first.')
      return notFound(`No fake route for ${method} ${path}`)
    }),
  )
  return { calls, use: (route) => routes.unshift(route) }
}

/** Minimal tensor response for `values` laid out as `shape`. */
export function tensorResponse(tensorId: string, shape: number[], values: number[]) {
  const mean = values.reduce((a, b) => a + b, 0) / values.length
  return Response.json({
    tensor_id: tensorId,
    shape,
    dtype: 'float32',
    count: values.length,
    summary: { min: Math.min(...values), max: Math.max(...values), mean, std: 0 },
    offset: 0,
    values,
    truncated: false,
  })
}

export class FakeSocket {
  static instances: FakeSocket[] = []
  onopen: (() => void) | null = null
  onmessage: ((message: { data: string }) => void) | null = null
  onclose: (() => void) | null = null
  closed = false
  readonly url: string

  constructor(url: string) {
    this.url = url
    FakeSocket.instances.push(this)
    queueMicrotask(() => this.onopen?.())
  }

  /** Deliver an event as the server would. */
  emit(event: object) {
    this.onmessage?.({ data: JSON.stringify(event) })
  }

  /** The server or network closed the connection. */
  drop() {
    this.closed = true
    this.onclose?.()
  }

  close() {
    if (this.closed) return
    this.closed = true
    this.onclose?.()
  }

  send() {}

  static latest(): FakeSocket {
    const socket = FakeSocket.instances[FakeSocket.instances.length - 1]
    if (!socket) throw new Error('No WebSocket was opened')
    return socket
  }

  static install() {
    FakeSocket.instances = []
    vi.stubGlobal('WebSocket', FakeSocket)
  }
}

export const stepEvent = (jobId: string, epoch: number, extra: Partial<Record<string, unknown>> = {}) => ({
  schema_version: '1.0',
  event_type: 'training.step',
  job_id: jobId,
  epoch,
  step: epoch * 10,
  metrics: { loss: 1 / epoch, accuracy: 0.5 + epoch / 100, val_loss: 1.2 / epoch, val_accuracy: 0.5 + epoch / 110 },
  selected_activations: {},
  ...extra,
})
