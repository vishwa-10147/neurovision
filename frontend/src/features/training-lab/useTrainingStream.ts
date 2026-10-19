import { useCallback, useEffect, useRef, useState } from 'react'
import { trainingEventsUrl } from '../../api/training'
import { useTrainingStore } from '../../stores/trainingStore'
import { TrainingEventSchema, type TrainingEvent } from '../../types/events'

const TERMINAL = new Set(['training.completed', 'training.cancelled', 'training.failed'])
/** Batch UI updates: at most one store update per this interval. */
const FLUSH_MS = 50
const MAX_RECONNECTS = 3

/**
 * Follows a training job's WebSocket. The server replays past events on every
 * connection, so reconnecting after a drop loses nothing.
 */
export function useTrainingStream(jobId: string | null, running: boolean) {
  const applyEvents = useTrainingStore((s) => s.applyEvents)
  const setStream = useTrainingStore((s) => s.setStream)
  const [attempt, setAttempt] = useState(0)
  const failures = useRef(0)

  useEffect(() => {
    if (!jobId || !running) return
    let closedByUs = false
    let finished = false
    let buffer: TrainingEvent[] = []
    let flushTimer: ReturnType<typeof setTimeout> | null = null
    let retryTimer: ReturnType<typeof setTimeout> | null = null
    let socket: WebSocket

    const flush = () => {
      flushTimer = null
      if (buffer.length === 0) return
      const events = buffer
      buffer = []
      applyEvents(events)
    }

    const connect = () => {
      setStream('connecting')
      socket = new WebSocket(trainingEventsUrl(jobId))
      socket.onopen = () => {
        failures.current = 0
        setStream('open')
      }
      socket.onmessage = (message) => {
        let data: unknown
        try {
          data = JSON.parse(String(message.data))
        } catch {
          return
        }
        const parsed = TrainingEventSchema.safeParse(data)
        if (!parsed.success) return
        buffer.push(parsed.data)
        if (TERMINAL.has(parsed.data.event_type)) {
          finished = true
          flush()
        } else if (flushTimer === null) {
          flushTimer = setTimeout(flush, FLUSH_MS)
        }
      }
      socket.onclose = () => {
        flush()
        if (closedByUs || finished) {
          setStream('closed')
          return
        }
        failures.current += 1
        if (failures.current > MAX_RECONNECTS) {
          setStream('error')
          return
        }
        setStream('connecting')
        retryTimer = setTimeout(connect, 500 * failures.current)
      }
    }

    failures.current = 0
    connect()
    return () => {
      closedByUs = true
      if (flushTimer) clearTimeout(flushTimer)
      if (retryTimer) clearTimeout(retryTimer)
      socket.close()
    }
  }, [jobId, running, attempt, applyEvents, setStream])

  /** Reconnect after the stream gave up. */
  const reconnect = useCallback(() => setAttempt((n) => n + 1), [])
  return { reconnect }
}
