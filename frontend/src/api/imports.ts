import { z } from 'zod'
import { ImportResponseSchema, SampleModelSchema, type ImportResponse, type SampleModel } from '../types/imports'
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

export function importSample(name: string): Promise<ImportResponse> {
  return request(`/import/onnx/samples/${encodeURIComponent(name)}`, ImportResponseSchema, { method: 'POST' })
}
