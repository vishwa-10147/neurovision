import { z } from 'zod'
import {
  ImportResponseSchema,
  RunResponseSchema,
  SampleModelSchema,
  type ImportResponse,
  type RunInput,
  type RunResponse,
  type SampleModel,
} from '../types/imports'
import { TensorGridSchema, TensorMapSchema, type TensorGrid, type TensorMap, type ViewQuery } from '../types/maps'
import { request } from './client'

// Uploading and parsing a large model can take a while.
const IMPORT_TIMEOUT_MS = 120_000

/** Upload an ONNX file for inspection. It is parsed on the server, never executed. */
export function uploadOnnx(file: File, signal?: AbortSignal): Promise<ImportResponse> {
  return request(`/import/onnx?filename=${encodeURIComponent(file.name)}`, ImportResponseSchema, {
    method: 'POST',
    rawBody: file,
    signal,
    timeoutMs: IMPORT_TIMEOUT_MS,
  })
}

export function listSamples(signal?: AbortSignal): Promise<SampleModel[]> {
  return request('/import/onnx/samples', z.array(SampleModelSchema), { signal })
}

/** Run an imported model on the server, inside its isolated inference worker. */
export function runImported(
  modelId: string,
  inputs: Record<string, RunInput>,
  signal?: AbortSignal,
): Promise<RunResponse> {
  return request(`/models/${encodeURIComponent(modelId)}/run`, RunResponseSchema, {
    method: 'POST',
    body: { inputs },
    signal,
    // The first run may also start the worker and load the model.
    timeoutMs: 60_000,
  })
}

function viewPath(modelId: string, kind: 'maps' | 'grids', query: ViewQuery) {
  const params = new URLSearchParams({ run_id: query.runId, node_id: query.nodeId, output: String(query.output) })
  for (const index of query.fixed) params.append('fixed', String(index))
  return `/models/${encodeURIComponent(modelId)}/${kind}?${params}`
}

/** One 2D map of a tensor from a recent run (bounded, maybe block-averaged). */
export function getTensorMap(modelId: string, query: ViewQuery, signal?: AbortSignal): Promise<TensorMap> {
  return request(viewPath(modelId, 'maps', query), TensorMapSchema, { signal, timeoutMs: 30_000 })
}

/** Thumbnails along the third-to-last dimension (channels or heads). */
export function getTensorGrid(modelId: string, query: ViewQuery, signal?: AbortSignal): Promise<TensorGrid> {
  return request(viewPath(modelId, 'grids', query), TensorGridSchema, { signal, timeoutMs: 30_000 })
}

export function importSample(name: string): Promise<ImportResponse> {
  return request(`/import/onnx/samples/${encodeURIComponent(name)}`, ImportResponseSchema, { method: 'POST' })
}
