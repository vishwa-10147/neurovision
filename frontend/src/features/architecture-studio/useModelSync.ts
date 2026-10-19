import { useCallback, useEffect, useMemo, useState } from 'react'
import { createEducationalModel } from '../../api/models'
import { useModelStore } from '../../stores/modelStore'
import { validateConfig, type DenseNetworkConfig } from '../../types/architecture'

/** Wait for typing to settle before asking the backend for a new graph. */
export const SYNC_DEBOUNCE_MS = 250

/**
 * Validates the draft configuration and, when valid, builds its graph on the
 * backend. Invalid drafts are never submitted; the last valid graph stays shown.
 */
export function useModelSync() {
  const config = useModelStore((s) => s.config)
  const { setLoading, setModel, setError } = useModelStore.getState()
  const validation = useMemo(() => validateConfig(config), [config])
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    if (!validation.ok) return
    // The current model already matches this config (e.g. returning to the
    // studio): keep it, so trained weights are not replaced by a fresh network.
    const { modelConfig, status } = useModelStore.getState()
    if (sameConfig(modelConfig, validation.value) && status.state === 'ready') return
    const controller = new AbortController()
    const timer = setTimeout(async () => {
      setLoading()
      try {
        const created = await createEducationalModel(validation.value, controller.signal)
        setModel(created.model_id, created.graph, validation.value)
      } catch (error) {
        if (!controller.signal.aborted) setError(error)
      }
    }, SYNC_DEBOUNCE_MS)
    return () => {
      clearTimeout(timer)
      controller.abort()
    }
  }, [validation, attempt, setLoading, setModel, setError])

  const retry = useCallback(() => setAttempt((n) => n + 1), [])
  return { validation, retry }
}

const sameConfig = (a: DenseNetworkConfig | null, b: DenseNetworkConfig) =>
  a !== null && JSON.stringify(a) === JSON.stringify(b)
