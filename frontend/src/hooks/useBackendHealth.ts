import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchHealth, type HealthResponse } from '../api/health'

export type BackendHealth =
  | { state: 'checking' }
  | { state: 'online'; data: HealthResponse; checkedAt: Date }
  | { state: 'offline'; message: string; checkedAt: Date }

const POLL_INTERVAL_MS = 15_000

export function useBackendHealth(): { health: BackendHealth; recheck: () => void } {
  const [health, setHealth] = useState<BackendHealth>({ state: 'checking' })
  const controllerRef = useRef<AbortController | null>(null)

  const check = useCallback(async () => {
    controllerRef.current?.abort()
    const controller = new AbortController()
    controllerRef.current = controller
    try {
      const data = await fetchHealth(controller.signal)
      setHealth({ state: 'online', data, checkedAt: new Date() })
    } catch (error) {
      if (controller.signal.aborted) return
      const message = error instanceof Error ? error.message : 'Unknown error.'
      setHealth({ state: 'offline', message, checkedAt: new Date() })
    }
  }, [])

  useEffect(() => {
    // check() only sets state after awaiting the network request.
    // oxlint-disable-next-line react/set-state-in-effect
    void check()
    const timer = window.setInterval(() => void check(), POLL_INTERVAL_MS)
    return () => {
      window.clearInterval(timer)
      controllerRef.current?.abort()
    }
  }, [check])

  const recheck = useCallback(() => {
    setHealth({ state: 'checking' })
    void check()
  }, [check])

  return { health, recheck }
}
