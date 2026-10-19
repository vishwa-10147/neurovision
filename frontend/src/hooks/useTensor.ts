import { useEffect, useState } from 'react'
import { getTensor } from '../api/training'
import type { TensorResponse } from '../types/training'

export type TensorState =
  | { state: 'idle' }
  | { state: 'loading' }
  | { state: 'ready'; data: TensorResponse }
  | { state: 'error'; error: unknown }

const MAX_CACHED = 200
// Keyed by model, tensor, and weights version, so values refresh after training.
const cache = new Map<string, Promise<TensorResponse>>()

function load(modelId: string, tensorId: string, key: string): Promise<TensorResponse> {
  let promise = cache.get(key)
  if (!promise) {
    promise = getTensor(modelId, tensorId)
    cache.set(key, promise)
    promise.catch(() => cache.delete(key))
    if (cache.size > MAX_CACHED) cache.delete(cache.keys().next().value as string)
  }
  return promise
}

/** Fetch one tensor (weights, biases, or gradients) of the current model. */
export function useTensor(modelId: string | null, tensorId: string | null, weightsVersion: number): TensorState {
  const key = modelId && tensorId ? `${modelId}|${tensorId}|${weightsVersion}` : null
  const [result, setResult] = useState<{ key: string; data?: TensorResponse; error?: unknown } | null>(null)

  useEffect(() => {
    if (!key || !modelId || !tensorId) return
    let active = true
    load(modelId, tensorId, key).then(
      (data) => active && setResult({ key, data }),
      (error: unknown) => active && setResult({ key, error }),
    )
    return () => {
      active = false
    }
  }, [key, modelId, tensorId])

  if (!key) return { state: 'idle' }
  if (result?.key !== key) return { state: 'loading' }
  if (result.data) return { state: 'ready', data: result.data }
  return { state: 'error', error: result.error }
}

/** For tests: forget cached tensors. */
export function clearTensorCache() {
  cache.clear()
}
