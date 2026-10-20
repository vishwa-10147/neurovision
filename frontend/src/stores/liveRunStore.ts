/**
 * Live runs state: the external run being watched and the steps streamed from
 * it. Every value comes from the user's training script via the backend's
 * event stream; nothing is simulated.
 */
import { create } from 'zustand'
import type { TrainingEvent } from '../types/events'
import type { RunInfo, RunPoint, RunStatus } from '../types/runs'
import type { StreamState } from './trainingStore'

export interface WatchedRun {
  runId: string
  name: string
  totalEpochs: number
  status: RunStatus
  /** Present when the run failed. */
  failure: string | null
}

interface LiveRunState {
  run: WatchedRun | null
  points: RunPoint[]
  /** Latest measured activations per layer, as sent by the script. */
  activations: Record<string, number[]>
  stream: StreamState

  watch: (info: RunInfo) => void
  stopWatching: () => void
  applyEvents: (events: TrainingEvent[]) => void
  setStream: (stream: StreamState) => void
}

export const useLiveRunStore = create<LiveRunState>()((set, get) => ({
  run: null,
  points: [],
  activations: {},
  stream: 'idle',

  watch: (info) => {
    if (get().run?.runId === info.run_id) return
    set({
      run: {
        runId: info.run_id,
        name: info.name,
        totalEpochs: info.total_epochs,
        // The stream replays from the start and brings the status up to date.
        status: 'running',
        failure: null,
      },
      points: [],
      activations: {},
      stream: 'idle',
    })
  },
  stopWatching: () => set({ run: null, points: [], activations: {}, stream: 'idle' }),
  applyEvents: (events) => {
    const { run, points, activations } = get()
    if (!run) return
    let nextRun = run
    let nextPoints = points
    let nextActivations = activations
    for (const event of events) {
      if (event.job_id !== run.runId) continue
      switch (event.event_type) {
        case 'training.started':
          nextRun = { ...nextRun, totalEpochs: event.total_epochs }
          break
        case 'training.step': {
          // Steps strictly increase; replays after a reconnect repeat old ones.
          const last = nextPoints[nextPoints.length - 1]
          if (last && event.step <= last.step) break
          if (nextPoints === points) nextPoints = [...points]
          nextPoints.push({ epoch: event.epoch, step: event.step, metrics: event.metrics })
          if (Object.keys(event.selected_activations).length > 0) nextActivations = event.selected_activations
          break
        }
        case 'training.completed':
          nextRun = { ...nextRun, status: 'completed' }
          break
        case 'training.cancelled':
          nextRun = { ...nextRun, status: 'failed', failure: 'The run was cancelled.' }
          break
        case 'training.failed':
          nextRun = { ...nextRun, status: 'failed', failure: event.message }
          break
      }
    }
    set({ run: nextRun, points: nextPoints, activations: nextActivations })
  },
  setStream: (stream) => set({ stream }),
}))
