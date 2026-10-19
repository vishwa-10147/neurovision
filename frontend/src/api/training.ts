import {
  ForwardResponseSchema,
  MetricsResponseSchema,
  TensorResponseSchema,
  TrainingJobInfoSchema,
  type ForwardResponse,
  type TensorResponse,
  type TrainingConfig,
  type TrainingJobInfo,
} from '../types/training'
import { API_BASE_URL, request } from './client'

const model = (modelId: string) => `/models/${encodeURIComponent(modelId)}`

export function runForward(modelId: string, inputs: number[][], signal?: AbortSignal): Promise<ForwardResponse> {
  return request(`${model(modelId)}/forward`, ForwardResponseSchema, {
    method: 'POST',
    body: { inputs },
    signal,
  })
}

export function getTensor(modelId: string, tensorId: string, signal?: AbortSignal): Promise<TensorResponse> {
  return request(`${model(modelId)}/tensors/${encodeURIComponent(tensorId)}`, TensorResponseSchema, { signal })
}

export function startTraining(modelId: string, config: TrainingConfig): Promise<TrainingJobInfo> {
  return request(`${model(modelId)}/training`, TrainingJobInfoSchema, { method: 'POST', body: config })
}

export function cancelTraining(jobId: string): Promise<TrainingJobInfo> {
  return request(`/training/${encodeURIComponent(jobId)}/cancel`, TrainingJobInfoSchema, { method: 'POST' })
}

export function getMetrics(modelId: string, signal?: AbortSignal) {
  return request(`${model(modelId)}/metrics`, MetricsResponseSchema, { signal })
}

/** ws:// or wss:// URL for a job's event stream, derived from the API base URL. */
export function trainingEventsUrl(jobId: string): string {
  return `${API_BASE_URL.replace(/^http/, 'ws')}/api/v1/training/${encodeURIComponent(jobId)}/events`
}
